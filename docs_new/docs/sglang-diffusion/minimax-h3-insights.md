# MiniMax-H3 洞察文档

> 基于 MiniMax 官方模型卡与 SGLang 上游 PR [#33275](https://github.com/sgl-project/sglang/pull/33275)（已合并）及后续 CI/并行相关 PR 整理。  
> 目标读者：需要快速理解 **模型能力、架构取舍、SGLang 原生支持路径** 的工程师与研究者。

---

## 1. 模型概览

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 是一个面向通用多模态生成的 **音视频联合生成系统**：单次请求同时产出视频与同步立体声音轨。

| 维度 | 规格 |
| --- | --- |
| 输出时长 | 4–15 秒（含端点） |
| 帧率 | 24 FPS |
| 默认短边 | 768 px；宽高比覆盖 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16 等 |
| 音频 | 32 kHz stereo AAC |
| 对话语言 | 稳定支持 11 种（含中/英/日/韩等） |
| Hub ID | Hugging Face: `MiniMaxAI/MiniMax-H3`；ModelScope: `MiniMax/MiniMax-H3` |
| License | MiniMax-H3 Community License（生产/商用前需审阅） |

### 1.1 完整系统 vs 开源权重

完整 H3 产品链路由三部分组成：

```text
用户多模态输入
    │
    ▼
H3-Context-IR  ── 指令理解 / 跨模态关联 / 结构化中间表示（未开源，提供 API）
    │
    ▼
H3-Base        ── 本地可部署的联合音视频生成主干（本次开源主体）
    │
    ▼
H3-Regenerate-2K ── 以 in-context 方式把 768p 结果再生到 2K（尚未开源）
```

SGLang Diffusion 当前原生支持的是 **H3-Base** 的两个任务分区权重：

| Checkpoint / `--model-variant` | 公开 task | 条件输入 |
| --- | --- | --- |
| `FL2VA` / `fl2va` | `t2va`、`fl2va` | 纯文本；或首帧 / 末帧 / 首末帧 |
| `Ref2VA` / `ref2va` | `ref2va`（含 V2V） | 图像 / 视频 / 音频参考（可组合） |

要点：

- **V2V 不是独立 task**，而是 `ref2va` + video reference。
- `--model-path` 指向仓库根 ID，由 SGLang 按 `--model-variant` 映射到 `FL2VA` / `Ref2VA` 子目录；不要手动指到子目录。
- 发布权重是 **CFG-distilled**：只有一条 positive denoise branch，因此 **不支持 CFG parallel**。

---

## 2. 架构结构

### 2.1 端到端数据流

```text
Text / Image / Video / Audio conditions
        │
        ├─ H3-Encoder (Qwen3-VL-32B, hidden_states[50]) ──► text tokens
        ├─ H3-VisualVAE (f16t4d24) ──► video / keyframe / ref latents
        └─ H3-AudioVAE (32ch, 40 Hz/ch) ──► audio latents
        │
        ▼
 Packed multimodal sequence
   [ text | imgvid_cond | audio | video_target | pad ]
        │
        ▼
 H3-Omni-Transformer (≈33B dense DiT, MM-RoPE, modality AdaLN)
        │
        ├─ video latents ──► VisualVAE decode ──► H.264 @ 24fps
        └─ audio latents ──► AudioVAE decode ──► AAC stereo @ 32kHz
```

### 2.2 组件规格（与 SGLang 配置对齐）

#### H3-Encoder（文本/视觉语义）

- 基于 **Qwen3-VL-32B** 全量预训练权重。
- 向 DiT 提供 **第 50 层** hidden states（`text_dim = 5120`）。
- SGLang 侧类：`MiniMaxH3Qwen3VLEncoder`；强制 `output_hidden_states=False`、`use_cache=False`，只跑到所需层。
- 仓库自带 tokenizer/special tokens（如 `<d>`、`<Picture n>`、`<Video n>`、`<Audio n>`），必须使用 H3 发布配置。

#### H3-VisualVAE

| 项 | 值 |
| --- | --- |
| 标记 | `f16t4d24` |
| 空间压缩 | 16× |
| 时间压缩 | 4× |
| latent channels | 24 |
| patchify | `(1, 2, 2)` → Transformer 侧等效空间 32×、时间仍 4× |
| 解码策略 | overlapping tiled decode（质量契约） |
| 精度策略 | 权重 fp32 resident；decode 默认 fp16 autocast |

编码器训练后额外训练了 **ViT-based decoder**，以降低解码成本并提升重建质量。

#### H3-AudioVAE

| 项 | 值 |
| --- | --- |
| sample rate | 32 kHz |
| latent channels | 32 |
| 左右声道 | 共享编解码器、独立处理后再合并 |
| 时间率 | 每声道约 40 Hz latent tokens |
| 精度 | fp32 |

#### H3-Omni-Transformer（DiT）

| 项 | 值 |
| --- | --- |
| 参数量 | ≈33B dense；其中约 13B 位于 AdaLN 相关分支 |
| layers | 50 |
| hidden size | 5376 |
| attention heads | 56 × head_dim 128 |
| FFN hidden | 14336 |
| token refiner | 2 层（无 AdaLN / 无 RoPE 的 pre-norm block） |
| video latent dim | 24 |
| audio latent dim | 32 |
| AdaLN modality 数 | 3（video / audio / text-or-cond 语义标签） |
| packed alignment | 64 |
| 注意力 | 开源首发为 **full attention**；训练后期有 native sparse attention，稀疏推理实现后续单独发布 |

**Block 内结构（单流、模态无关主干）：**

```text
x ── RMSNorm ── modality AdaLN(scale/shift) ── Attention(+QK-Norm, MM-RoPE) ── gate residual
  ── RMSNorm ── modality AdaLN(scale/shift) ── SwiGLU MLP ── gate residual
```

- Attention / FFN **不含模态专用结构**；模态差异集中在 **输入投影、输出头、AdaLN 分支**。
- 每层 AdaLN：3 个 modality × 6 个 H-wide 向量（MSA/MLP 各一组 shift/scale/gate）。
- Final layer：单 modality AdaLN（shift/scale）后分别投影到 video patch logits 与 audio latents。
- RoPE：3D MM-RoPE，坐标轴 `(t, h, w)`；对 128 维 head 中的 **96 维**做旋转（`rotary_percent=0.75`）。

### 2.3 Packed Sequence 布局

SGLang 按发布契约物化 packed sequence：

```text
[ text L | imgvid_cond C | audio A(=t*2ch) | video_target V | pad P ]
```

关键细节：

- `token_tags` / `block_token_tags` / `update_mask` / `update_audio_mask` 标记哪些行可更新。
- `combined_indices = inverse_indices * modality_num + token_tags`，供 indexed AdaLN 调制。
- `img_position_ids` 使用 fp64 网格构造，再进入 RoPE；音频块钉在 w-grid 两端。
- 序列长度对齐到 64，以便 Ulysses 切分与 attention backend 路径稳定。

FL2VA keyframe 签名仅允许：`[0]`、`[-1]`、`[0, -1]`。

---

## 3. 创新点

### 3.1 原生联合音视频生成（T2VA / FL2VA / Ref2VA）

不是“视频模型 + 后期配乐”，而是 **同一 Transformer 联合预测 video/audio latents**，从训练与推理上保证时序与语义同步。输出契约固定为：H.264 24fps + AAC stereo 32kHz 单文件 MP4。

### 3.2 任务泛化的双分区权重

- `FL2VA`：文本与端点帧动画（含纯 T2VA）。
- `Ref2VA`：图像/视频/音频参考与 V2V。
- 共享 Encoder/VAE 组件语义，但 Omni-Transformer 分区特化；服务端用 partition admission 做任务准入，避免错分区静默跑错任务。

### 3.3 模态无关主干 + 模态专用 AdaLN

主干 Attention/FFN 共享，模态差异下沉到 AdaLN 与 I/O。带来：

- 更好的跨模态泛化与扩展；
- AdaLN 输出可预计算/缓存时，推理部署可降低有效加载压力（官方说明约 13B AdaLN 参数可在纯推理场景优化）；
- 训练/推理增量成本相对可控。

### 3.4 MM-RoPE + Packed Multimodal Sequence

把 text / condition / audio / video 排进同一 varlen packed 序列，用 3D RoPE 表达时空关系，使首末帧、参考视频、立体声音频能在同一注意力上下文中交互。

### 3.5 分离优化的 Visual / Audio VAE

- VisualVAE：因果时间结构 + latent-space 优化，兼顾重建与可学性；ViT decoder 降解码成本。
- AudioVAE：受 VA-VAE 启发优化 latent；左右声道独立编解码再合成，原生支持 stereo I/O。

### 3.6 CFG 蒸馏单分支

发布 checkpoint 只有一条 denoise branch，服务侧天然拒绝 CFG parallel，简化并行拓扑与数值一致性路径。

### 3.7 In-context 2K 再生（产品侧，暂未开源）

2K 不是独立超分模块，而是把 768p 结果 + 原始 context 再喂回 H3 做 in-context regenerate，以保留小字与细节；当前仅 API 可验证。

---

## 4. SGLang 如何支持

### 4.1 支持形态：原生 Pipeline，而非 Diffusers 包装

上游以 **fully native joint video/audio pipeline** 接入 `sglang.multimodal_gen`：

| 层级 | 关键路径 |
| --- | --- |
| Registry | `python/sglang/multimodal_gen/registry.py` 注册 `MiniMaxAI/MiniMax-H3` / `MiniMax/MiniMax-H3` |
| Pipeline | `runtime/pipelines/minimax_h3_pipeline.py` → `MiniMaxH3Pipeline` |
| Config | `configs/pipeline_configs/minimax_h3.py`、`configs/models/**/minimax_h3*.py` |
| Sampling / API lower | `configs/sample/minimax_h3.py` + video request hooks |
| DiT | `runtime/models/dits/minimax_h3.py` |
| Encoder | `runtime/models/encoders/minimax_h3_qwen3vl.py` |
| VAE | `runtime/models/vaes/minimax_h3*.py` |
| Stages | `runtime/pipelines_core/stages/model_specific_stages/minimax_h3/` |
| Scheduler | `scheduling_minimax_h3_euler_ancestral.py`（Euler ancestral η=0） |
| Cookbook | `docs_new/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` |

Pipeline stages（严格顺序）：

1. `InputValidation`
2. `PartitionAdmission`（校验 task ↔ FL2VA/Ref2VA）
3. `TextEncoding`（Qwen3-VL）
4. `VisualEncoding`（关键帧 / 参考视觉）
5. `AudioEncoding`（参考音频）
6. `LatentPreparation`（canvas、packed layout、初始噪声）
7. `TimestepPreparation`（video/audio 独立 sigma / flow shift）
8. `Denoising`（DiT loop）
9. `Decoding`（video + audio → MP4）

服务形态：**仅 monolithic**（`supports_disaggregation=False`）。对外暴露异步 OpenAI-compatible `/v1/videos`。

### 4.2 请求契约

核心字段：

```json
{
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": "...",
  "task": "t2va | fl2va | ref2va",
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0
  },
  "num_inference_steps": 50,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 1101,
  "quality": "lossless",
  "num_outputs_per_prompt": 1
}
```

- `target` 决定对齐后的画布与帧数；时长必须在 \[4, 15\]。
- `flow_shift` / `audio_flow_shift` **独立控制** 视频与音频扩散日程。
- `num_outputs_per_prompt` ∈ \[1, 10\]；seed 展开为 `seed + output_index`。
- Ref2VA prompt 中的 `<Picture n>` / `<Video n>` / `<Audio n>` 必须与 `conditions` 的模态顺序一致。
- 通用顶层 `video_path` / `video_url` **不会**自动降到 H3 reference conditions，必须走 `conditions[].uri`。

### 4.3 精度与数值契约

| 组件 | 精度策略 |
| --- | --- |
| DiT 主体 | bf16 |
| video/audio patch proj、time embedder、final heads、`rope.inv_freq` | **强制 fp32** |
| Text encoder | bf16 |
| Video VAE | fp32 resident；decode fp16 autocast |
| Audio VAE | fp32 |

`--performance-mode speed` **故意保持 DiT eager**：当前 `torch.compile` 会改变数值输出，不能用于一致性 GT。SageAttention 被拒绝（packed varlen 路径无法保输出）。

---

## 5. SGLang 支持细节

### 5.1 并行与放置

| 能力 | H3 行为 |
| --- | --- |
| Ulysses SP | 主推；packed multi-segment attention 的默认扩展方式 |
| Tensor Parallel | 可与 Ulysses 组合；要求 `heads % tp == 0` 且 `tp-local heads % ulysses == 0`，且 `64 % ulysses == 0` |
| Ring SP | **主线实现拒绝**（packed multi-segment 不兼容）；社区后续有跨节点 Ulysses×Ring 探索 PR |
| FSDP inference | 仅 shard DiT；保留 mixed BF16/FP32 all-gather |
| CFG parallel | 拒绝（蒸馏单分支） |
| Encoder parallel | `auto` 可在单机高带宽拓扑上 fold Qwen encoder 到空闲 Ulysses ranks；吞吐场景可用 DP（要求 TP1/DiT DP1） |
| Layerwise offload | 验证于 2×RTX 5090：20 DiT blocks resident；encoder/VAE decoder 一层 prefetch；video VAE encoder 与小体积 audio VAE 保持 resident |
| Breakable CUDA Graph | 可选；需匹配 warmup resolution 与 `bcg-text-buckets` |
| Online FP8 | B200/B300 验证路径；自动忽略必须保持 fp32 的 patch/time/final 层 |

验证过的推荐拓扑示例：

| 硬件 | 推荐 | 备注 |
| --- | --- | --- |
| 4×H200 | Ulysses4 resident | quality profile 严格校验拓扑 |
| 4×H100 | TP2 + Ulysses2 | 测得最快无损；纯 Ulysses4 难驻留 |
| 8×B200/B300 | Ulysses8 resident | 可叠加 online FP8 |
| 2×RTX 5090 | TP2 + layerwise | 32GB 级容量路径 |
| MI300X/MI355X | Ulysses + AITER packed attn | AMD 验证矩阵覆盖 1/2/4/8 卡 |

### 5.2 Kernel / 运行时加速点

原生 DiT 路径接入了多项 SGLang diffusion kernel：

- **fused QK-Norm + RoPE**（`fused_inplace_qknorm_rope`）
- **indexed AdaLN scale/shift/gate**（Triton `indexed_scale_shift_bf16_` / `indexed_gate_bf16_`）
- **SwiGLU silu-and-mul**（带 activation rounding）
- **Ulysses packed QKV all-to-all**（`usp_relayout`）
- Attention backends：FA / AITER / Torch SDPA（按平台选择）
- TP>1 时可 batch 多 block 的 AdaLN 投影以摊销 all-gather（Cache-DiT / layerwise / compile 时关闭）

### 5.3 Denoise 与调度

- Scheduler：`MiniMaxH3EulerAncestralEta0`（η=0 的 Euler 步进）。
- Video / audio 使用 **独立 sigma 日程**；`model_index` 可携带 `sigma_shift_scales`。
- Denoise loop 维护 packed rows、update masks、local embedding layout（Ulysses 分片时）。
- `quality` 请求字段挂载审计过的 Cache-DiT profile（仅严格 4×H200 T2VA 工作负载 fail-closed）：

| quality | warmup | RDT | max consecutive cache | 相对 lossless 延迟（4×H200 测量） |
| --- | ---: | ---: | ---: | ---: |
| lossless | — | — | — | 75.10 s（1.00×） |
| high | 4 | 0.04 | 1 | 53.70 s（1.40×） |
| medium | 4 | 0.12 | 3 | 30.23 s（2.48×） |
| low | 4 | 0.24 | 3 | 25.81 s（2.91×） |

Cache-DiT **不可**与 FSDP / DiT layerwise 同时开；BCG 优先时 Cache-DiT 保持关闭。

### 5.4 VAE 质量契约

- 默认 `parallel_decode_mode=tiled`：跨 GPU 分发 **完整 tile**，不改 tile 内计算。
- 明确拒绝 `spatial` / `spatial_shard` / `patch`（验证发现与发布质量不一致）。
- Video/Audio VAE 的 `latents_mean/std` 在 config 加载时做严格契约校验。

### 5.5 API / Serving 细节

- 异步任务：`POST /v1/videos` → poll status → `GET .../content`（多输出用 `?variant=`）。
- `output_quality` 仅控制封装压缩，与采样 `quality` 分离。
- 多输出时文本条件复用；在 32GB layerwise 配置上 denoise/decode 顺序执行以控峰值显存。
- Docker cookbook 会在镜像内安装 diffusion extra；FL2VA/V2V/Ref2VA 需挂载 host media 目录。

### 5.6 社区 PR 现状（截至 2026-08-03）

| PR | 状态 | 作用 |
| --- | --- | --- |
| [#33275](https://github.com/sgl-project/sglang/pull/33275) | Merged | 原生模型支持主体 |
| [#33282](https://github.com/sgl-project/sglang/pull/33282) | Merged | skills / benchmark preset / 文档 |
| [#33345](https://github.com/sgl-project/sglang/pull/33345) | Merged | cookbook 表格渲染修复 |
| [#33281](https://github.com/sgl-project/sglang/pull/33281) | Open | 2×H100 TP2 一致性 CI（4-GPU runner 不可用时的替代） |
| [#33327](https://github.com/sgl-project/sglang/pull/33327) | Open | 跨节点 Ulysses × Ring SP |
| [#33353](https://github.com/sgl-project/sglang/pull/33353) | Open | 对无 ring 支持路径 fail-closed |

---

## 6. 工程含义与使用建议

1. **先选对分区**：T2VA/FL2VA 用 `fl2va`；任何参考/V2V 用 `ref2va`。
2. **一致性与压测分开**：GT / 对齐实验用 eager BF16/FP32 + lossless；延迟实验再开 Cache-DiT / FP8 / BCG。
3. **H100 优先 TP2+Ulysses2**；内存更紧时 TP4 或 FSDP，而不是假设 FSDP 更快。
4. **不要开 CFG parallel / SageAttention / spatial VAE decode**——都会被显式拒绝或破坏契约。
5. **Context-IR 与 2K regenerate 仍在产品侧**：本地 SGLang 路径对应 768p H3-Base；要复现官方 2K 端到端质量需结合官方 API / 自建 prompt 预处理。

---

## 7. 参考资料

- 模型卡：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- SGLang cookbook：`docs_new/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`
- 主支持 PR：<https://github.com/sgl-project/sglang/pull/33275>
- 核心实现入口：
  - `python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/`
