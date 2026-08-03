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

`token_tags` 语义（AdaLN modality index，padding 为 `-1`，进 block 前 `clamp(min=0)`）：

| tag | 含义 | 典型行 |
| ---: | --- | --- |
| 0 | VIDEO | keyframe / ref 视觉行、video target 行 |
| 1 | TEXT | Qwen 文本行（FL2VA 中视觉 prompt span 可由 presentation 覆盖） |
| 2 | AUDIO | audio ref / audio target 行（channel-major） |
| -1 | PADDING | 对齐到 64 的填充行 |

### 2.4 模块功能关联：谁产出什么、谁消费什么

H3 的原生 pipeline 不是“encoder 直接喂 DiT”，而是 **先把多模态材料编码成行向量契约，再在 denoise stage 里组装 packed 输入**。各 stage 的职责与数据键如下。

```text
请求 (task/conditions/target/seed)
        │
        ▼
PartitionAdmission ── 校验 task ↔ FL2VA/Ref2VA 分区
        │
        ├─ TextEncoding ──► extra[minimax_h3_text_embeddings]
        │                    {positive: {hidden_states[L,5120], text_len, text_token_tags}}
        │
        ├─ VisualEncoding ──► keyframe_cond_rows / reference_image_rows / reference_video_rows
        │                      (patch 后的视觉 latent 行, 宽 96 = 24×1×2×2)
        │
        ├─ AudioEncoding ──► reference_audio_rows
        │                     (channel-major audio latent 行, 宽 32)
        │
        ├─ LatentPreparation ──► initial_video_rows [V,96], initial_audio_rows [A,32]
        │                         (独立 RNG：同 seed 分别 seed 视频/音频 generator)
        │
        ├─ TimestepPreparation ──► sigmas.video / sigmas.audio
        │                           (flow_shift / audio_flow_shift 各自 time-shift)
        │
        ▼
Denoising ── 组装 packed layout + DiT forward + Euler η=0 更新 target 行
        │
        ▼
Decoding ── unpatchify video / unpack audio → VisualVAE + AudioVAE → MP4
```

功能关联可以概括成四条边：

1. **语义边（Text → DiT）**  
   Qwen3-VL 产出 `hidden_states[L,5120]`。它们不直接进 50 层 DiT block，而是先经 `condition_proj`（5120→5376）+ `token_refiner`（2 层）变成 **refined text rows**，再 scatter 到 packed 序列的 text 槽位。`text_token_tags` 覆盖 packed 的 text 区间，用于 AdaLN modality 选择。

2. **条件边（VAE → packed cond/ref rows）**  
   Visual/Audio VAE 只编码 **条件材料**（首末帧、参考图/视频/音频），不编码将要生成的 target。这些行在 denoise 中被 pin 住：每步重写到 buffer，但不做 Euler 更新。Target 行才吃初始噪声并逐步更新。

3. **几何边（canvas/shape → packed indices）**  
   Pre-queue 的 `resolved_v2` shape 决定 `latent_t/h/w`、`audio_t`。`packed_sequence` 据此生成 `text_pos / img_pos / audio_pos / update_mask / img_position_ids / cu_seqlens`。没有这份几何，`_embed` 不知道把哪段 latent 写到哪一行。

4. **时间边（sigmas → unique_timesteps / AdaLN）**  
   视频与音频各有 sigma 日程。每步把不同行映射到不同 timestep：text/padding/video-target 跟 `t_video`；audio-target 跟 `t_audio`；视觉条件行钉在 `max(t_video, imgvid_cond_noise_aug)`（默认锚点约 `0.999`）；音频参考行钉在 `max(t_audio, audio_ref_cond_noise_aug)`（默认 `1.0`）。`inverse_indices` + `token_tags` 合成 `combined_indices`，供 indexed AdaLN 查表。

### 2.5 DiT 输入如何组装：从行向量到 `_embed`

DiT forward 的入口不是 `[B,C,T,H,W]` 张量，而是 **packed keyword contract**。服务热路径（`MiniMaxH3DenoiseBranch.forward_kwargs`）每步组装：

| kwarg | 来源 | 作用 |
| --- | --- | --- |
| `x` | `[1,S,96]` 持久 buffer | 全序列视频行槽；cond/ref 只写一次，之后只刷新 target |
| `audio_x` | `[1,S,32]` 持久 buffer | 全序列音频行槽；同上 |
| `prompt_embeds` | TextEncoding → 可选预计算 refine | 文本条件；若已 refine 则附带 `refined_prompt_embeds_length` |
| `img_position_ids` | packed 几何网格 | MM-RoPE 的 `(t,h,w)` |
| `rope_cache` | 请求级预计算 | 当前 Ulysses rank 的局部 cos/sin |
| `unique_timesteps` / `inverse_indices` | 逐步 host 去重 | 行 → timestep 槽 |
| `block_token_tags` / `block_combined_indices` | 局部 shard | AdaLN modality 索引 |
| `*_pos_info` | packed 位置 | text/img/audio/infer-output 行号 |
| `local_embedding_layout` | Ulysses 局部布局 | 避免每步 `nonzero` 扫描 |
| `packed_seq_params` / `refiner_packed_seq_params` | cu_seqlens | varlen attention / refiner 窗口 |
| `update_mask` | packed | 选出可更新的视频 target 行（服务路径常 `skip_mask_out_condition=True`，由 loop 自己只更新 target slice） |

#### 组装步骤 A：模态行准备（DiT 之外）

```text
视频噪声 [1,24,T,H,W]
  └─ patchify(1,2,2) ──► video_rows [T*(H/2)*(W/2), 96]

音频噪声 [A,32]  （A = audio_t * 2ch，左右声道行交替/channel-major）

条件视觉 latent ──► cond_rows [C,96]   （FL2VA keyframe 或 Ref2VA 参考块）
条件音频 latent ──► audio_ref_rows [R,32]
```

有条件时，target 噪声被 scatter 进“全 image/audio 行”缓冲：`update_mask==True` 的位置放噪声，前面 cond/ref 槽留给锚点行。

#### 组装步骤 B：`_embed`（DiT 内，每步或 BCG break 点）

`MiniMaxH3DiTModel._embed` 在 **当前 Ulysses rank 的行分片** `[row_start, row_stop)` 上构造：

```text
decoder_input [S_local, 5376] bf16
t_emb         [M, time_embed_dim] fp32   ← TimeEmbedder(unique_timesteps)
```

写入规则：

```text
1) Text
   prompt_embeds[L,5120]
     → (若未预计算) condition_proj → token_refiner(2 blocks)
     → text_embed[L,5376]
     → 按 text_pos ∩ local shard 写入（serving 用 index_copy 连续前缀）

2) Video
   x 中本 rank 拥有的 img_global_ids 行
     → video_patch_proj (fp32 Linear: 96 → 5376, gather_output)
     → cast bf16 写入 img_row_ids

3) Audio
   audio_x 中本 rank 拥有的 audio_global_ids 行
     → audio_patch_proj (fp32 Linear: 32 → 5376)
     → cast bf16 写入 audio_row_ids

4) Padding / 未覆盖行
   trusted layout 下先 empty 再对 live 后缀 zero；direct caller 用 zeros + index_add
```

要点：

- **patch/time/final 投影保持 fp32**，只在写入 bf16 序列时 cast。
- Text refine 是请求静态的：denoise stage 会调用 `refine_prompt_embeds` 一次，后续步跳过 refiner。
- RoPE cache 同样请求静态：按 Ulysses 局部行预计算 `cos/sin`。
- Ulysses 只在 Attention 内做 sequence↔heads all-to-all；`_embed`、MLP、FinalLayer 都是 **row-local**。

#### 组装步骤 C：block stack 与输出选择

```text
adaln_input = SiLU(t_emb).bf16

combined_indices[i] = inverse_indices[i] * 3 + token_tags[i].clamp(min=0)
                      └─ 选哪组 timestep 调制      └─ 选哪 modality 分支

for block in blocks[50]:
    AdaLN → Norm → Attn(+QK-Norm, RoPE) → gate
    AdaLN → Norm → MLP → gate

FinalLayer:
    单 modality AdaLN(shift/scale) → fp32
    → video_out / audio_out 对所有 local 行做 GEMM
    → (Ulysses all_gather 行) → index_select 出
         img_pos_for_infer_output（仅 video target）
         audio_pos（音频行；ref 行随后由 mask/slice 丢弃更新）
    → (TP all_gather 输出列)
```

服务路径里 DiT 返回的 video velocity 已是 **target 行**；audio velocity 再按 `audio_target_slice` 切片后做 Euler 更新。条件行每步仍出现在 packed 注意力上下文中（提供条件信息），但状态被 pin 在 noise-aug 锚点，不跟随 target 更新。

### 2.6 一步 denoise 的张量协作（功能时序）

```text
step k:
  1. 把 video_rows / audio_rows 的 target 子集 index_copy 进 x/audio_x buffer
  2. 取出预计算的 (unique_timesteps, inverse_indices, block_combined)
  3. DiT.forward → (v_video_target, v_audio_all)
  4. Euler η=0 只更新 target slice:
       denoised = state + sigma_t * velocity
       state ← (1 - σ_{k+1}/σ_k) * denoised + (σ_{k+1}/σ_k) * state
  5. cond/ref 行保持步骤 0 钉入的锚点值
```

因此“编码组装”与“生成更新”的边界是：

| 行类型 | 谁写入初值 | 每步是否进 Attention | 是否 Euler 更新 |
| --- | --- | --- | --- |
| text | TextEncoder → refine → `_embed` | 是 | 否（无 latent 状态） |
| imgvid cond / visual ref | VisualVAE (+ optional noise aug) | 是 | 否（pin） |
| audio ref | AudioVAE (+ optional noise aug) | 是 | 否（pin） |
| video target | LatentPreparation 噪声 | 是 | 是 |
| audio target | LatentPreparation 噪声 | 是 | 是 |
| padding | 零填充 | 是（占位对齐） | 否 |

### 2.7 VisualVAE / AudioVAE 编码配方

条件材料在进 DiT 之前，必须变成 **与 target 行同宽的 packed rows**。视觉与音频走不同的数值契约。

#### VisualVAE：三类材料，两种几何

| 材料链 (`material_chain`) | 任务 | 几何策略 | 编码 API |
| --- | --- | --- | --- |
| `image.target_canvas` | FL2VA | 对齐到 **target canvas**（与输出画布同尺寸） | `encode_images` |
| `image.reference_preserve` | Ref2VA | **自身几何**：短边 2048、可 upscale、LANCZOS、对齐到 32；**不继承** target canvas | `encode_images`（同 keyframe recipe） |
| `video.reference_preserve` / `video_audio.reference_preserve` | Ref2VA | 按 plan 解析的参考视频尺寸/帧数解码后编码 | `encode_videos` |

共同视觉 tokenizer 配方（`minimax_h3_encode_keyframe_cond_rows` / `encode_reference_video_rows`）：

```text
PIL / RGB frames
  → scoped fp32 VAE weights
  → fork_rng + seed=42  （VAE 后验是 SAMPLED，seed 是契约的一部分，与 request seed 无关）
  → encode_images / encode_videos (use_fp16_latent=True)
  → z.cpu().float()
  → (z - latents_mean) / latents_std     # loader 注入的 24-ch stats
  → patchify [1,2,2]                     # → rows [n, 96] fp32
```

要点：

- FL2VA 的同一批 prepared keyframe 图像 **同时喂给 Qwen TextEncoding 与 VisualVAE**（`minimax_h3_prepared_keyframes` 按请求缓存）。
- Ref2VA 参考图用独立 `reference_image_short_edge_v1` 策略；latent 网格是 `H/16 × W/16`，与 target 的 `latent_h/w` 可以不同。
- 参考视频：ffmpeg 一次解码 + transform + truncate（支持 `start_time_seconds`）；音画 seek 对齐。编码后丢弃 RGB frames，避免双份驻留。
- 多 keyframe / 多参考在同一 `minimax_h3_scoped_encode_fp32` 作用域内完成，避免反复切 VAE dtype。
- Parallel tiling 时各 replica 编码完整 tile 再 gather，再做 seeded posterior sample。

#### AudioVAE：均值编码 + 确定性后端

```text
uri → localize
  → probe facts (sample_rate / has_audio / duration)
  → load waveform
       · pure audio: 保留源采样率，归一化到 stereo
       · video/video_audio: 抽 44.1 kHz stereo PCM
  → resample → 32 kHz
  → _AudioVAEDeterminismContext
       (关 TF32 / 关 cuDNN / SDP=math，保证 encode 可复现)
  → AudioVAE posterior MEAN（不采样）
  → normalize by audio latents_mean/std
  → channel-major rows [T*2, 32] fp32
```

特殊情况：`video.reference_preserve` 且源无音轨时，仍保留视觉 block 的请求顺序位置，但音频条件写成 `ref_audio_t=0`、空 rows，避免打乱 Ref2VA block 顺序。`video_audio` 则 fail-closed：承诺有音频却缺失会报错。

#### Condition noise aug（可选）

在 denoise 组装前，可对 clean cond/ref rows 做 RF 混合：

```text
noised = noise_aug * clean + (1 - noise_aug) * noise
```

- `noise_aug=1.0`：保持干净锚点（默认视觉/音频锚点 timestep 接近 1）。
- 视觉噪声按每个 condition 的 `(latent_t,h,w)` 独立 generator（同 request seed）绘制，再 patchify。
- 该值同时作用到 **行数值** 与 DiT 行 timestep（`max(t_video, noise_aug)` / `max(t_audio, noise_aug)`）。

### 2.8 Text presentation：Qwen 看到什么

TextEncoding 不直接把用户 prompt 丢进 Qwen；它先构造与条件对齐的 **presentation token 流**，并同步生成 `text_token_tags`（与 ids 等长，防止 AdaLN tag 漂移）。

| 任务 | Presentation 形态 |
| --- | --- |
| `t2va` | 原文 prompt（无 vision block）；tags 全为 TEXT(1) |
| `fl2va` | 对每个 keyframe：`<Picture i>: ` + `<\|vision_start\|>` + N×`<\|image_pad\|>` + `<\|vision_end\|>`，再接原文 prompt；vision block tags=VIDEO(0) |
| `ref2va` | 按 `plan.materials` 顺序：`<Picture i>` / `<Audio j>` / `<Video k>` 标签；图像带 vision block；音频只有标签（**音频内容不进 Qwen**）；视频为时间戳文本 + 多个 VIDEO pad block；最后接原文 prompt |

Ref2VA 视频时间戳规则：

- Qwen3-VL temporal merge=2；奇数帧会重复末帧。
- 每个 merged block 发 `\<{t:.1f} seconds\>`（banker’s rounding）+ vision block。
- 纯 `video` 条件仅当 probe `has_audio=true` 时才额外贡献 `<Audio j>` 标签；`video_audio` 始终贡献 Audio 标签。

Qwen 前向：

```text
presentation ids (+ pixel_values / image_grid_thw / video tensors)
  → MiniMaxH3Qwen3VLEncoder.encode_ids
  → hidden_states[L, 5120]   # 固定取第 50 层
  → extra[minimax_h3_text_embeddings].positive
       {hidden_states, text_len, text_token_tags}
```

随后 denoise 把 `text_token_tags` **覆盖** packed 序列 text 区间的默认 tag，使 FL2VA/Ref2VA 里 vision span 在 AdaLN 上走 VIDEO modality，而不是 TEXT。

### 2.9 Ref2VA block 排序与 packed 布局

Ref2VA 的关键约束是：**请求/plan 中的 material 顺序 = presentation 标签顺序 = packed ref blocks 顺序 = cond/ref rows 拼接顺序**。

#### 排序源：`plan.materials`

`_ref2va_ordered_blocks_and_rows` 按 materials 线性扫描，产出：

```text
blocks:  [{kind, latent_*/ref_audio_t}, ...]
cond_rows:       cat(visual rows in that order)     # [C_vis, 96]
audio_ref_rows:  cat(audio rows in that order)      # [C_aud, 32]
```

| material_chain | block.kind | 视觉行 | 音频行 |
| --- | --- | --- | --- |
| `image.reference_preserve` | `image` | 有（自身 HxW） | 无 |
| `audio` | `audio` | 无 | 有（`ref_audio_t`） |
| `video.reference_preserve` | `video` | 有（`latent_t,h,w`） | 有（可为 0） |
| `video_audio.reference_preserve` | `video_audio` | 有 | 有（必须 >0） |

#### Packed 物理布局

```text
[ text L
| ref_block_0 ... ref_block_n     ← 按 materials 顺序展开
| audio_target A(=audio_t*2)
| video_target V
| pad ]
```

每个 ref block 的内部展开：

| kind | 在序列中的展开 | 时间原点 `t_cursor` 推进 |
| --- | --- | --- |
| `image` | 仅 visual rows | `+1` |
| `audio` | 仅 audio rows（L/R channel-major） | `+ref_audio_t` |
| `video` / `video_audio` | **先 audio rows，再 video rows**（共享同一 temporal origin） | `+max(ref_audio_t, video_t_span)` |

RoPE 网格细节：

- 参考图：用自身 `sqrt(h*w)` 建 h/w 网格，`t` 钉在当前 `t_cursor`。
- 参考视频：音频钉在该参考自己的 w-grid 两端；视频帧用 `_video_t_grid(origin=t_cursor)`。
- Target 音视频的时间原点接在所有 ref blocks 之后的 `t_cursor`。
- `img_pos = [所有 ref visual 行号] + [target video 行号]`  
  `audio_pos = [所有 ref audio 行号] + [target audio 行号]`  
  `update_mask` / `audio_update_mask` 前缀 False（pin），后缀 True（可更新）。

与 FL2VA 布局对比：

| | FL2VA / T2VA | Ref2VA |
| --- | --- | --- |
| 条件区 | 固定 `imgvid_cond` 槽（0/1/2 帧） | 变长 ref blocks 交织音画 |
| 条件几何 | 与 target canvas 同网格 | 每块可用自身网格 |
| 音频条件 | 通常无（除非额外 ref audio 路径） | 可多段，且可与 video 绑定 |
| 首末帧锚点 | RoPE `t` 钉到首帧或末帧语义位置 | 无 keyframe 语义；参考是 style/identity |

官方上限（模型卡，服务侧按 profile 校验）：图像 ≤9、视频 ≤3（各 2–15s，总长 ≤15s）、音频 ≤3（须伴随图像/视频，不能单独输入）、全类型文件总数 ≤12。

### 2.10 Decode：从 target 行回到 MP4

Denoise 结束后只发布 **target 切片**：

```text
video_rows[target] [V,96]
  → unpatchify [1,2,2] → latent [1,24,T,H,W]
  → reverse normalize (×std + mean)
  → VisualVAE tiled decode (fp16 autocast)
  → crop 回 target canvas（VAE tile pad 在右/下）
  → H.264 @ 24fps

audio_rows[target] [A,32]
  → unpack channel-major → [2,32,T]
  → reverse normalize
  → AudioVAE decode → waveform [2,1,L]
  → permute 为输出 [1,2,L]
  → AAC stereo @ 32kHz
```

Decode 与 encode 的不对称点：

- 视觉 encode 用 **sampled posterior + seed 42**；decode 走确定性 tiled 路径。
- 音频 encode 强制确定性后端；decode 不复用该 determinism 作用域。
- 拒绝 `spatial` / `spatial_shard` VAE parallel decode，以遵守发布质量契约。

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

Stage 之间不共享 Diffusers 式 `prompt_embeds + latents` 单一接口，而是通过 `batch.extra` 的 H3 专用键传递契约对象；Denoising 是唯一把各模态行 **scatter 进 packed buffer 并调用 DiT** 的汇合点。详见 §2.4–§2.10。

TextEncoding 额外职责：

- 用 H3 `processor` + 仓库 tokenizer 构造 presentation（含 `<Picture n>` / `<Video n>` / `<Audio n>` 材料标签与特殊 token）；详见 §2.8。
- 只取 Qwen3-VL 第 50 层 hidden states；多输出请求可对相同 fingerprint 去重，encoder DP 时按 presentation 分发整请求而非 stack batch。
- 产出的 `text_token_tags` 在 denoise 组装时覆盖 packed `token_tags` 的 text 区间（允许 FL2VA 视觉 span 覆盖默认 TEXT tag）。

Visual/Audio Encoding 与 Ref2VA 排序契约见 §2.7、§2.9；decode 回包装见 §2.10。

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
