# SGLang 特性与开关速查手册

> 面向 DeepSeek-V4 / V4-Pro 及相关 MoE Serving 场景整理。
> 覆盖：特性解释、应用场景、主开关、限制条件、互斥矩阵。
> 信息以当前 `sgl-project/sglang` 代码与 cookbook 为准，个别实验开关可能随版本变化。

---

## 0. 怎么用这份文档

先定目标，再开开关：

| 目标 | 先看章节 |
|---|---|
| 单机/多机怎么切卡 | [A. 并行](#a-并行-parallelism) |
| MoE 吞吐/通信 | [B. MoE / EP / Overlap](#b-moe-/-ep-/-overlap) |
| DSV4 混合注意力 | [C. Attention / DSV4 Hybrid](#c-attention-/-dsv4-hybrid) |
| 前缀复用 / 长上下文 | [D. Cache](#d-cache) |
| 降延迟 | [E. Speculative Decoding](#e-speculative-decoding) |
| 显存/精度 | [F. 量化 / Kernel](#f-量化-/-kernel) |
| Prefill/Decode 分离 | [G. PD Disaggregation](#g-pd-disaggregation) |
| Graph / 调度 | [H. CUDA Graph / Scheduler](#h-cuda-graph-/-scheduler) |
| 哪些不能一起开 | [互斥矩阵](#开关互斥矩阵) |
| DSV4 Pro 推荐组合 | [推荐配方](#dsv4-pro-推荐配方) |

**记忆用的 8 个主旋钮：**

1. `--tp/--dp/--ep` + `--enable-dp-attention`
2. `--moe-a2a-backend`（`deepep` / `megamoe`）
3. `--enable-two-batch-overlap`（TBO）
4. `--enable-single-batch-overlap`（SBO）
5. `--enable-hierarchical-cache`（HiCache）
6. `--enable-hisparse`（HiSparse）
7. `--speculative-algorithm`（MTP/EAGLE/DSpark）
8. `--disaggregation-mode`（PD）

---

## 特性分层总览

```text
请求进来
 ├─ 并行切分：TP / DP / EP / PP / CP
 ├─ 注意力路径：普通 / DSA / DSV4 hybrid(SWA+C4+C128)
 ├─ MoE 路径：DeepEP / MegaMoE / Waterfill + SBO/TBO
 ├─ 缓存路径：Radix / HiCache / HiSparse / UnifiedRadix(ShadowRadix)
 ├─ 投机解码：EAGLE/MTP / DSpark / DFlash
 └─ 执行加速：CUDA Graph / Overlap Scheduler / 量化 kernel
```

---

## A. 并行 (Parallelism)

### A1. Tensor Parallelism (TP)

| 项 | 内容 |
|---|---|
| **开关** | `--tp-size` / `--tp` |
| **解释** | 把权重/张量切到多卡 |
| **场景** | 单卡装不下；需要提高单副本吞吐 |
| **限制** | 跨节点 TP 可能通信受限；很多 EP backend 要求 `ep == tp` |

### A2. Data Parallelism (DP)

| 项 | 内容 |
|---|---|
| **开关** | `--dp-size` / `--dp` |
| **解释** | 多副本并行接请求 |
| **场景** | 内存够、要冲高并发 |
| **限制** | 多节点 DP 通常要配合 DP-Attention；副本间不共享 KV |

### A3. DP-Attention

| 项 | 内容 |
|---|---|
| **开关** | `--enable-dp-attention` |
| **解释** | Attention 走 DP，FFN/MoE 仍可 TP/EP |
| **场景** | DeepSeek 类大 MoE 常见高并发配方 |
| **限制** | 一般要求 `dp` 与 `tp` 对齐；部分 speculative 兼容有限；TBO non-EP 路径依赖它 |

### A4. Expert Parallelism (EP)

| 项 | 内容 |
|---|---|
| **开关** | `--ep-size` / `--ep` + `--moe-a2a-backend` |
| **解释** | Expert 分散到多卡，token all-to-all |
| **场景** | MoE 大模型标配 |
| **限制** | DeepEP/Mooncake/NIXL/MoRI/FlashInfer/MegaMoE 等常要求 `ep == tp` |

### A5. Pipeline Parallelism (PP)

| 项 | 内容 |
|---|---|
| **开关** | `--pp-size` |
| **解释** | 按层切流水线 |
| **场景** | 超大模型跨机切层 |
| **限制** | 常与 overlap schedule、speculative decoding 冲突；DSV4 TBO 仅支持 `pp==1` |

### A6. Prefill Context Parallel (CP)

| 项 | 内容 |
|---|---|
| **开关** | `--enable-prefill-cp`，可选 `--cp-strategy zigzag|interleave`，`--attn-cp-size` |
| **解释** | 长序列 prefill 按 context 切分 |
| **场景** | 超长上下文 TTFT 太高 |
| **限制** | PD 时只适合 prefill 实例；DSV4 TBO 当前禁用 CP；DSA cache layer-split 有额外约束 |

### A7. DWDP

| 项 | 内容 |
|---|---|
| **开关** | `--dwdp-size` |
| **解释** | MoE prefill 用权重预取替代 token A2A |
| **场景** | MoE prefill 通信瓶颈 |
| **限制** | 必须 `dwdp == tp`；仅 `disaggregation-mode null|prefill`；`pp==1`；无 speculative；无 TBO |

---

## B. MoE / EP / Overlap

### B1. DeepEP

| 项 | 内容 |
|---|---|
| **开关** | `--moe-a2a-backend deepep`，`--deepep-mode auto|normal|low_latency` |
| **相关 env** | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` |
| **解释** | 成熟 EP all-to-all 后端 |
| **场景** | 通用 EP；低延迟 decode 常用 `auto/low_latency` |
| **限制** | `normal` 可能禁 CUDA Graph；`max-running-requests * MTP_draft_tokens` 不要超过 dispatch token cap |

### B2. MegaMoE

| 项 | 内容 |
|---|---|
| **开关** | `--moe-a2a-backend megamoe` 或 `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` |
| **相关 env** | `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`<br/>`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS`<br/>`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND`<br/>`SGLANG_OPT_FIX_MEGA_MOE_MEMORY` |
| **解释** | 把 dispatch + L1 + SwiGLU + L2 + combine 融成 DeepGEMM mega-kernel |
| **场景** | **Blackwell 高吞吐**（B200/B300/GB200/GB300） |
| **限制** | 仅 Blackwell；cookbook 主要挂 high-throughput；RTX PRO 6000 不支持；开 MegaMoE 时不要手设 `--moe-runner-backend`；完整 DeepEP-SBO 不可用，只剩 shared 轻量重叠 |

#### MegaMoE 精度变体

| 模式 | env | 说明 |
|---|---|---|
| W4A8 | 默认 MegaMoE | FP4 weights + FP8 acts |
| W4A4 | `USE_FP4_ACTS=1` + `USE_MXF4_KIND=1` | FP4 acts，prefill 通常更快 |

### B3. MoE Runner Backend

| 项 | 内容 |
|---|---|
| **开关** | `--moe-runner-backend auto|deep_gemm|flashinfer_mxfp4|cutlass|...` |
| **解释** | 选 MoE grouped GEMM / runner |
| **场景** | 细调 FP8/FP4 MoE |
| **限制** | 一般用 `auto`；与 MegaMoE 同时手设容易踩坑 |

### B4. EPLB

| 项 | 内容 |
|---|---|
| **开关** | `--enable-eplb`，`--eplb-algorithm`，`--ep-num-redundant-experts` |
| **解释** | 专家放置重平衡 |
| **场景** | EP 负载倾斜 |
| **限制** | 需要足够流量统计；DSV4 cookbook 标实验性；与某些 PCG/recorder 路径冲突 |

### B5. Waterfill

| 项 | 内容 |
|---|---|
| **开关** | `--enable-waterfill` |
| **解释** | 把 fused shared expert 当额外 routed slot，填到最闲 EP rank |
| **场景** | DeepEP/MegaMoE shared expert 不均衡 |
| **限制** | 仅 DeepEP/MegaMoE；会隐式启用 shared-expert fusion；会削弱经典 SBO；DeepEP 生产 decode 建议 `auto/low_latency` 保 CUDA Graph |

### B6. TBO（Two Batch Overlap，vLLM 侧也称 DBO）

| 项 | 内容 |
|---|---|
| **开关** | `--enable-two-batch-overlap` |
| **相关** | `--tbo-token-distribution-threshold`（默认 0.48） |
| **解释** | 一个 batch 切成两个 ubatch，通信与计算互盖 |
| **场景** | Prefill EP/DP 通信空档大 |
| **限制（DSV4 重点）** | 目前 **prefill-only**；decode/target-verify 未实现；`pp==1`；不与 prefill CP 同开；DSpark aux capture 会跳过 TBO；non-EP 路径需要 DP-Attention；与 DWDP 不兼容；与 DSA index-topk sharing 不兼容；MegaMoE 不一定能拿到 DeepEP 式收益；attention TP>1 的 non-EP TBO 仍有已知问题 |

### B7. SBO（Single Batch Overlap）

| 项 | 内容 |
|---|---|
| **开关** | `--enable-single-batch-overlap` |
| **相关 env** | `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` |
| **解释** | 单 batch 内重叠 shared experts / down GEMM 与 dispatch/combine |
| **场景** | 小中 batch decode |
| **限制** | **SM90/Hopper 当前直接报错禁用**；完整路径依赖 DeepEP + 独立 shared experts；`is_nextn`/MTP draft 不参与；Waterfill/shared fusion 会掏空 shared 重叠点；MegaMoE 仅剩 shared 轻量重叠；Blackwell/Hopper 策略分叉（combine SM、signal 布局不同） |

---

## C. Attention / DSV4 Hybrid

### C1. Attention Backend

| 项 | 内容 |
|---|---|
| **开关** | `--attention-backend`，可选 `--decode-attention-backend` / `--prefill-attention-backend` |
| **解释** | 选 FlashInfer / FA / Triton / AITER 等 |
| **场景** | 硬件/模型特定优化 |
| **限制** | 与 page size、KV dtype、spec topk、sliding window、多模态相关 |

### C2. DSA Backend（V3.2 / GLM DSA）

| 项 | 内容 |
|---|---|
| **开关** | `--dsa-prefill-backend` / `--dsa-decode-backend` / `--dsa-topk-backend` |
| **解释** | DSA 稀疏注意力后端 |
| **场景** | DeepSeek-V3.2 / GLM DSA |
| **限制** | **不等于 DSV4**；DSV4 走自己的 dsv4 backend |

### C3. DSV4 Hybrid Attention 概念

| 概念 | 解释 | 备注 |
|---|---|---|
| **SWA** | 最近 128 token 滑窗 | 短命 |
| **C4** | 4:1 压缩 + top-k 稀疏 | HiSparse 主要对象 |
| **C128** | 128:1 压缩 dense | 不太适合 HiSparse |
| **ShadowRadix / UnifiedRadix** | 逻辑 FULL 坐标 + 多池 shadow 映射 | DSV4 前缀缓存底座 |
| **mHC** | Manifold-Constrained Hyper-Connections | TBO 下需禁跨层融合并 op 化 |

### C4. DSV4 相关开关

| 开关 / env | 解释 | 场景 | 限制 |
|---|---|---|---|
| `--enable-deepseek-v4-fp4-indexer` | FP4 C4 indexer | 长上下文 decode 实验 | 实验性；偏 SM100 |
| `SGLANG_DSV4_COMPRESS_STATE_DTYPE` | 压缩状态 dtype | 挤显存时可试 `bf16` | 只影响 compress state，不影响主权重 |
| `SGLANG_OPT_USE_ONLINE_COMPRESS` | 在线压缩 | DSV4 压缩路径 | 与 MTP online compress 相关 |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | attention 多流 | 小 batch decode 准备阶段 | 与 MoE SBO 抢 stream/SM，别混为一谈 |
| `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` | FlashMLA sparse prefill | 长上下文 prefill | 平台依赖 |
| `SGLANG_DSV4_FP4_EXPERTS` | 是否按 FP4 expert 解释权重 | 原 FP4 vs 转 FP8 checkpoint | 转 FP8 权重时应关掉 |

---

## D. Cache

### D1. Radix Cache

| 项 | 内容 |
|---|---|
| **开关** | 默认开启；关闭用 `--disable-radix-cache` |
| **解释** | 前缀 KV 复用 |
| **场景** | 多轮、共享 system prompt |
| **限制** | HiSparse 当前强制要求关闭；某些 scoring/MIS 路径会强制关闭 |

### D2. HiCache（Hierarchical Cache）

| 项 | 内容 |
|---|---|
| **开关** | `--enable-hierarchical-cache` |
| **关键参数** | `--hicache-ratio`（DSV4 常用）<br/>`--hicache-size`<br/>`--hicache-write-policy write_through|write_back|write_through_selective`<br/>`--hicache-io-backend`<br/>`--hicache-mem-layout`<br/>`--hicache-storage-backend file|mooncake|hf3fs|nixl|...` |
| **解释** | GPU → CPU → Storage 分层前缀缓存 |
| **场景** | 多轮/agent/共享前缀很多，GPU 装不下 |
| **限制** | 建立在 radix/UnifiedRadix 上；DSV4 常要求 `--hicache-ratio`；RTX PRO 6000 cookbook 不支持；与 HiSparse 当前互斥；L3 storage 推荐 `page_first_direct + direct + wait_complete` |

### D3. HiSparse

| 项 | 内容 |
|---|---|
| **开关** | `--enable-hisparse` |
| **配置** | `--hisparse-config='{"top_k":2048,"device_buffer_size":6144,"host_to_device_ratio":10}'` |
| **解释** | decode 时 GPU 只留 top-k 热 KV，完整稀疏 KV 放 CPU |
| **场景** | **PD decode + 超长上下文高并发** |
| **限制** | 仅 DSA / DeepSeek-V4；要求 `--disable-radix-cache`；文档定位 decode 实例；与 decode radix cache 不兼容；ROCm unified-KV 路径暂不支持；DSA 路径对 KV dtype/backend 有约束 |

### D4. UnifiedRadix / ShadowRadix

| 项 | 内容 |
|---|---|
| **开关** | `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`（多数 hybrid/DSV4 场景隐式启用） |
| **解释** | 统一树管理 Full/SWA/Mamba；DSV4 用逻辑 FULL + shadow 映射多池 |
| **场景** | DSV4/hybrid 前缀缓存正确性底座 |
| **限制** | 不是“分层存储”本身；分层靠 HiCache；HiSparse 另走热集路径 |

### 三者对照

| | ShadowRadix / UnifiedRadix | HiCache | HiSparse |
|---|---|---|---|
| 本质 | 索引/语义 | 跨请求分层存储 | 单请求 decode 热集 |
| 主要对象 | SWA/C4/C128 一致性 | compressed KV + 逻辑锚点 | C4 KV |
| 典型收益 | 让 V4 前缀缓存成立 | 多轮容量/复用 | 长上下文 decode 并发 |
| 与 radix | 就是 radix 的 V4 形态 | 依赖 radix | 当前要关 radix |

---

## E. Speculative Decoding

### E1. EAGLE / MTP / NEXTN

| 项 | 内容 |
|---|---|
| **开关** | `--speculative-algorithm EAGLE|EAGLE3|NEXTN`<br/>`--speculative-num-steps`<br/>`--speculative-eagle-topk`<br/>`--speculative-num-draft-tokens`<br/>`--speculative-draft-model-path`（如需） |
| **解释** | 草稿多 token，目标模型校验 |
| **场景** | 低延迟 / 平衡配方 |
| **限制** | 高吞吐饱和时常关；吃显存；接受率低会负优化；与部分 PP / DFlash / NGRAM 约束冲突 |

### E2. DSpark

| 项 | 内容 |
|---|---|
| **开关** | `--speculative-algorithm DSPARK` + dspark block/table 参数 |
| **解释** | confidence/block 调度投机 |
| **场景** | 有 DSpark draft / SPS/STS 表 |
| **限制** | 无 SPS 表可能退化；aux capture 与 TBO/CP 有互斥 |

### E3. DFlash / NGRAM

| 特性 | 开关 | 限制摘要 |
|---|---|---|
| DFlash | `--speculative-algorithm DFLASH` | 无 DP-Attn；`pp==1`；常禁 overlap schedule / mixed chunk |
| NGRAM | `--speculative-algorithm NGRAM` | CUDA-only；无 DP-Attn；常禁 overlap schedule / mixed chunk |

### E4. Adaptive / Decoupled Spec

| 特性 | 开关 | 场景 | 限制 |
|---|---|---|---|
| Adaptive | `--speculative-adaptive` | 接受率波动大 | 更适合 topk=1 |
| Decoupled | `--decoupled-spec-role verifier|drafter` | draft/verify 分资源池 | 运维复杂，端点/rank 要对齐 |

---

## F. 量化 / Kernel

| 特性 | 开关 | 解释 | 场景 | 限制 |
|---|---|---|---|---|
| Weight quant | `--quantization` | 权重量化加载 | 预量化模型 | 已预量化模型通常别再乱加 |
| KV quant | `--kv-cache-dtype fp8_e4m3/...` | KV 降精度 | 长上下文显存紧 | 缺 scale 可能精度掉；FP4 实验性 |
| FP8/FP4 GEMM | `--fp8-gemm-backend` / `--fp4-gemm-backend` | dense GEMM 后端 | 细调 | 硬件相关；优先 `auto` |
| ModelOpt | `--modelopt-quant` / `--quantize-and-serve` | NVIDIA 量化工作流 | FP8/FP4 校准导出 | 启动慢，依赖 ModelOpt |

---

## G. PD Disaggregation

| 特性 | 开关 | 解释 | 场景 | 限制 |
|---|---|---|---|---|
| PD mode | `--disaggregation-mode prefill\|decode` | Prefill/Decode 分实例 | Prefill 算力型、Decode 显存型 | 需要 router/网关与传输后端 |
| Transfer backend | `--disaggregation-transfer-backend mooncake\|nixl\|...` | KV 传输 | 多机 PD | 依赖 IB/NVLink；统一内存路径暂不兼容 PD |
| Decode radix | `--disaggregation-decode-enable-radix-cache` | decode 侧重用前缀 | 多轮 PD | **不兼容 HiSparse / speculative / fake transfer** |
| Decode offload | `--disaggregation-decode-enable-offload-kvcache` | decode 侧 offload | 显存紧 | 与具体 transfer/layout 相关 |
| PDMux | `--enable-pdmux` | 同机 PD 复用实验 | 实验 | 不兼容常规 disagg；常禁 overlap schedule |

---

## H. CUDA Graph / Scheduler

| 特性 | 开关 | 解释 | 场景 | 限制 |
|---|---|---|---|---|
| CUDA Graph | `--cuda-graph-max-bs*` / `--cuda-graph-config` / `--cuda-graph-backend-*` | 固化 kernel launch | decode 几乎必开 | prefill full graph 仍偏实验；PD 会自动禁另一侧 |
| Breakable / Piecewise | `--cuda-graph-backend-prefill breakable\|tc_piecewise` | 动态长度/可打断 | DP-Attn、动态 prefill | 与 speculative / PP / 某些 MoE A2A / LoRA / PD 等可能冲突 |
| Overlap schedule | 默认开；`--disable-overlap-schedule` | CPU 调度与 GPU 执行重叠 | 一般保持开 | PP、部分 speculative、PDMux、部分平台会禁 |
| Mixed chunk | `--enable-mixed-chunk` | prefill/decode 混合 | 混合负载 | 与某些 speculative 冲突 |
| Prefill delayer | `--enable-prefill-delayer` | 延迟碎片 prefill | DP-Attn 高并发 | 增加 TTFT |
| Continuous decode | `--num-continuous-decode-steps` | 连续多步 decode | 降调度开销 | 过大影响调度公平性 |

---

## I. DSV4 Serving 语义

| 特性 | 开关 | 解释 | 场景 | 限制 |
|---|---|---|---|---|
| Reasoning parser | `--reasoning-parser deepseek-v4` | 拆 thinking / final | DSV4 推理模式 | 与 chat template kwargs 配合 |
| Tool parser | `--tool-call-parser deepseekv4` | DSML tool call | 工具调用 | 语法/严格模式依赖 grammar backend |
| Default thinking | `SGLANG_DEFAULT_THINKING=1` / `--default-chat-template-kwargs '{"thinking":true}'` | 默认开思考 | chat serving | 影响延迟与输出格式 |
| Reasoning effort | `SGLANG_DSV4_REASONING_EFFORT=max` | high/max 思考强度 | 难任务 | max 需要更大上下文（如 384K） |

---

## 开关互斥矩阵

图例：

- ✅ 可一起开
- ⚠️ 有条件可开 / 收益或路径会降级
- ❌ 不兼容 / 当前实现互斥
- ◯ 无直接关系（通常可并存，但看场景）

### 矩阵 1：Cache / PD / Spec

|  | Radix | HiCache | HiSparse | PD Prefill | PD Decode | Spec(MTP/EAGLE) |
|---|---|---|---|---|---|---|
| **Radix** | - | ✅ | ❌（HiSparse 强制关） | ✅ | ⚠️（decode radix 另开） | ✅ |
| **HiCache** | ✅ | - | ❌ | ✅ | ⚠️ | ✅ |
| **HiSparse** | ❌ | ❌ | - | ⚠️（通常开在 decode） | ✅（主场景） | ❌/⚠️（decode radix/spec 冲突面） |
| **PD Prefill** | ✅ | ✅ | ⚠️ | - | ✅（成对） | ✅ |
| **PD Decode** | ⚠️ | ⚠️ | ✅ | ✅ | - | ⚠️ |
| **Spec** | ✅ | ✅ | ❌/⚠️ | ✅ | ⚠️ | - |

补充：

- `--disaggregation-decode-enable-radix-cache` ❌ `--enable-hisparse`
- `--disaggregation-decode-enable-radix-cache` ❌ speculative decoding
- `--disaggregation-decode-enable-radix-cache` ❌ `--disaggregation-transfer-backend fake`

### 矩阵 2：MoE / Overlap

|  | DeepEP | MegaMoE | Waterfill | SBO | TBO | DP-Attn | Prefill CP |
|---|---|---|---|---|---|---|---|
| **DeepEP** | - | ❌（二选一 backend） | ✅ | ✅（完整 SBO） | ✅（prefill） | ✅ | ✅ |
| **MegaMoE** | ❌ | - | ✅ | ⚠️（仅 shared 轻量） | ⚠️（收益/路径受限） | ✅ | ✅ |
| **Waterfill** | ✅ | ✅ | - | ⚠️（掏空独立 shared） | ◯ | ◯ | ◯ |
| **SBO** | ✅ | ⚠️ | ⚠️ | - | ◯（概念不同） | ◯ | ◯ |
| **TBO** | ✅ | ⚠️ | ◯ | ◯ | - | ✅（non-EP 必需） | ❌（DSV4） |
| **DP-Attn** | ✅ | ✅ | ◯ | ◯ | ✅ | - | ◯ |
| **Prefill CP** | ✅ | ✅ | ◯ | ◯ | ❌（DSV4） | ◯ | - |

补充：

- Waterfill ❌ `moe-a2a-backend` 不是 `deepep/megamoe`（会强制改到 deepep）
- SBO ❌ SM90/Hopper（启动报错）
- TBO ❌ DWDP
- TBO ❌ DSA index-topk sharing
- TBO ❌ DSV4 decode / PP>1 / DSpark aux capture

### 矩阵 3：并行 / Graph / 其它

| 组合 | 关系 | 说明 |
|---|---|---|
| PP + speculative | ❌/强冲突 | ServerArgs 多处限制 |
| PP + overlap schedule | ❌/强冲突 | 常被禁用 |
| PP + DSV4 TBO | ❌ | TBO 仅 `pp==1` |
| Unified memory + PD | ❌ | 暂不兼容 |
| Unified memory + speculative | ❌ | 暂不兼容 |
| PDMux + 常规 PD disagg | ❌ | 互斥 |
| PDMux + overlap schedule | ❌/常禁 | 实验路径 |
| DFlash/NGRAM + DP-Attn | ❌ | 文档限制 |
| Prefill CP + PD decode 实例 | ❌ | CP 只给 prefill |
| MegaMoE + 手设 moe-runner-backend | ⚠️ | cookbook 建议别手设 |
| Shared experts fusion + SBO | ⚠️/趋向 ❌ | fusion 后无独立 shared 可叠 |

---

## 每个开关的限制速查（浓缩版）

| 开关 | 关键限制 |
|---|---|
| `--tp-size` | 跨节点通信可能成为瓶颈；常与 `ep` 绑定 |
| `--dp-size` | 多节点通常要 DP-Attn |
| `--enable-dp-attention` | `dp/tp` 对齐；部分 speculative 受限 |
| `--ep-size` | 多数 a2a backend 要求 `ep==tp` |
| `--pp-size` | 与 overlap/spec/TBO 冲突面大 |
| `--enable-prefill-cp` | PD decode 不能开；DSV4 TBO 不能开 |
| `--moe-a2a-backend deepep` | `normal` 影响 CUDA Graph；注意 dispatch token cap |
| `--moe-a2a-backend megamoe` | 仅 Blackwell；高吞吐配方；完整 SBO 不可用 |
| `--enable-waterfill` | 仅 DeepEP/MegaMoE；隐式 shared fusion；削弱 SBO |
| `--enable-eplb` | 需流量统计；实验/运维成本高 |
| `--enable-two-batch-overlap` | DSV4 仅 prefill；无 CP/PP>1；non-EP 要 DP-Attn |
| `--enable-single-batch-overlap` | Hopper 禁；要 DeepEP+独立 shared；MTP draft 不参与 |
| `--disable-radix-cache` | 关闭前缀复用；HiSparse 必需 |
| `--enable-hierarchical-cache` | 依赖 radix；与 HiSparse 互斥；DSV4 常用 ratio |
| `--enable-hisparse` | 仅 DSA/V4；关 radix；偏 PD decode；与 decode radix/spec 冲突 |
| `--speculative-algorithm` | 吃显存；高吞吐未必赚；部分算法禁 DP-Attn/overlap |
| `--disaggregation-mode` | 需传输后端与路由；角色专属开关不同 |
| `--disaggregation-decode-enable-radix-cache` | 禁 HiSparse / speculative / fake transfer |
| `--enable-pdmux` | 实验；禁常规 PD / overlap |
| `--cuda-graph-*` | PD 自动禁另一侧；与某些 backend/特性冲突 |
| `--enable-deepseek-v4-fp4-indexer` | 实验性 |
| `--reasoning-parser deepseek-v4` | 需正确 chat template / thinking 配置 |
| `--tool-call-parser deepseekv4` | 需工具协议与 grammar 支持 |

---

## DSV4 Pro 推荐配方

### 1) 单机高吞吐（Blackwell）

```bash
# 核心
--tp 8 --dp 8 --enable-dp-attention
--moe-a2a-backend megamoe
--cuda-graph-max-bs <按显存调>

# MegaMoE env
export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
# 可选 W4A4
# export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
# export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1

# 可选
--enable-hierarchical-cache --hicache-ratio 2
# MTP：饱和时通常关或很浅
```

**别同时指望：** 完整 SBO、decode TBO。

### 2) 低延迟

```bash
--moe-a2a-backend deepep --deepep-mode auto
--speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
# Blackwell 且独立 shared 时可试
--enable-single-batch-overlap
```

**别开：** Hopper SBO；过深 MTP；无必要 MegaMoE。

### 3) 超长上下文 PD

```bash
# Prefill
--disaggregation-mode prefill
--enable-hierarchical-cache --hicache-ratio 2
# 可选 --enable-prefill-cp

# Decode
--disaggregation-mode decode
--enable-hisparse --disable-radix-cache
--hisparse-config='{"top_k":2048,"device_buffer_size":6144,"host_to_device_ratio":10}'
```

**别开：** decode 上的 HiCache/Radix 与 HiSparse 混用；decode radix + speculative。

### 4) Prefill 通信重叠

```bash
--enable-two-batch-overlap
# EP 路径：deepep/mori...
# 或 non-EP：--enable-dp-attention --moe-a2a-backend none
```

**确认未开：** prefill CP、PP>1、DWDP；不要对 decode 抱 TBO 预期。

---

## 常见误区

1. **HiCache ≠ HiSparse**
   HiCache=跨请求分层前缀；HiSparse=单请求 decode 热 KV。

2. **ShadowRadix ≠ HiCache**
   ShadowRadix/UnifiedRadix 是索引语义；HiCache 是存储分层。

3. **SBO ≠ TBO/DBO**
   SBO=单 batch 内重叠；TBO=两 ubatch 互重叠。

4. **MegaMoE ≠ 开了就能叠完整 SBO/TBO**
   MegaMoE 是 fused MoE 路径，很多经典插桩点没了。

5. **Waterfill ≠ EPLB**
   Waterfill 处理 shared expert 填平；EPLB 做专家放置重平衡。

6. **DSA flags ≠ DSV4 flags**
   V3.2 DSA backend 开关不能直接当成 V4 开关。

---

## 相关上游入口

- Roadmap: [sgl-project/sglang#23602](https://github.com/sgl-project/sglang/issues/23602)
- Day0 Blog: [DeepSeek-V4 on Day 0](https://lmsys.org/blog/2026-04-25-deepseek-v4/)
- Cookbook: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- HiCache: `docs_new/docs/advanced_features/hicache.mdx`
- HiSparse: `docs_new/docs/advanced_features/hisparse_guide.mdx`
- Expert Parallelism: `docs_new/docs/advanced_features/expert_parallelism.mdx`
- Speculative Decoding: `docs_new/docs/advanced_features/speculative_decoding.mdx`
- PD Disaggregation: `docs_new/docs/advanced_features/pd_disaggregation.mdx`

---

## 变更说明

本文是运维/性能向速查，不是完整 CLI 手册。若某个开关在新版本行为变化，以：

1. `python/sglang/srt/server_args.py`
2. `python/sglang/srt/arg_groups/*`
3. 对应 advanced_features 文档

为准。
