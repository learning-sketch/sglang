# DeepSeek-V4-Pro 性能数据（H200 及更低规格硬件）

> 来源优先级：  
> 1) cookbook `deepseek-v4-benchmarks.jsx`（Verified）  
> 2) 相关 docs PR（如 `#31363`, `#33109`）  
> 3) feature PR / issue（仅作补充，口径可能不一致）  
>
> 默认 workload：`random`, **ISL=8192**, **OSL=1024**  
> 指标：`tokens_per_sec_per_gpu` = 总吞吐/GPU（input+output 折算）  
> 延迟：cookbook 现口径多为 **P50**（`#31363` 后）

---

## 1. 结论先看

| 硬件 | Pro 官方测速 | 说明 |
|---|---|---|
| **8×H200** | ✅ 有（FP4，单机） | 但 **容量受限**：并发上来后吞吐几乎打平 |
| **H200 FP8 Pro** | ❌ cookbook 无数字 | 单元格是 `multi-2` 占位，未测 |
| **H100 Pro** | ❌ cookbook 无数字 | 单元格是 `multi-2` 占位，未测 |
| **H20 / L40 / RTX** | ❌ 无 Pro Verified 矩阵 | RTX PRO 6000 cookbook 明确不支持 Pro |
| **同硬件 Flash** | ✅ 数据完整 | 可作“同平台可跑性/上限参考”，**不能直接当 Pro 性能** |

一句话：

> **H200 上 Pro 有官方数字，但单机 8 卡会被 KV/显存卡住；更低规格（H100 及以下）基本没有公开 Pro Verified 吞吐矩阵，通常要多机或改跑 Flash。**

---

## 2. H200 · Pro · FP4（有数，重点）

来源：`deepseek-v4-benchmarks.jsx`，`sglang 0.5.15.post1`，**single-node**。

| Strategy | Concurrency | P50 TTFT (ms) | P50 TPOT (ms) | tok/s/GPU | 备注 |
|---|---:|---:|---:|---:|---|
| low-latency | 1 | 634 | 5.65 | **170** | |
| low-latency | 16 | 1727 | 23.12 | **559** | |
| balanced | 64 | 41506 | 26.14 | **589** | 已接近容量天花板 |
| balanced | 256 | 209586 | 28.23 | **591** | TTFT 爆炸=排队，不是算力线性扩展 |
| high-throughput | 1024 | 889185 | 66.39 | **594** | |
| high-throughput | 4096 | 1833386 | 65.86 | **601** | |

### 官方注释（很关键）

> 8×H200 跑 1.6T Pro 时 **capacity-bound**：KV 大概只能撑约 **~15 concurrent requests**。  
> 所以从 conc 64 一直到 ht/4096，`tok/s/GPU` 基本钉在 **~535–601**；更高并发只是排队，TTFT 到几十秒甚至分钟级。

### 解读

1. **单机 H200 不适合拿 Pro 冲高并发吞吐**  
2. low-latency 的 TPOT 还行（约 5.7–23 ms），但绝对吞吐远低于 B200/B300 Pro  
3. balanced/ht 的“高吞吐数字”几乎不涨，别被 concurrency 参数误导  
4. 若要更高并发，官方单元格倾向 **H200 Pro FP8 → multi-2**（但目前还没填测速）

对应 model path（H200 FP8 场景）：

- FP4 Instruct：`deepseek-ai/DeepSeek-V4-Pro`
- FP8 重打包：`sgl-project/DeepSeek-V4-Pro-FP8`（cookbook 注明 Hopper 对 FP4-mixed Instruct 不友好时用）

---

## 3. H200 · Flash（同平台对照，非 Pro）

同文件、同 workload，方便你判断“这台机器本身能跑到什么量级”。

### H200 Flash FP8

| Strategy | Conc | TTFT | TPOT | tok/s/GPU |
|---|---:|---:|---:|---:|
| low-latency | 1 | 183 | 3.26 | 632 |
| low-latency | 16 | 655 | 10.11 | 2752 |
| balanced | 64 | 880 | 40.63 | 3156 |
| balanced | 256 | 46563 | 89.82 | 3226 |
| high-throughput | 1024 | 217694 | 146.95 | 3975 |
| high-throughput | 4096 | 576540 | 148.29 | 3920 |

### H200 Flash FP4

| Strategy | Conc | TTFT | TPOT | tok/s/GPU |
|---|---:|---:|---:|---:|
| low-latency | 1 | 242 | 3.37 | 603 |
| low-latency | 16 | 498 | 10.19 | 2636 |
| balanced | 64 | 864 | 34.12 | 3072 |
| balanced | 256 | 3222 | 116.3 | 3768 |
| high-throughput | 1024 | 193812 | 126.31 | 4503 |
| high-throughput | 4096 | 499528 | 125.07 | 4546 |

### 补充：Flash Official + DSpark（`#33109`，H200×4，v0.5.16）

这是 **Flash Official**，不是 Pro；但 H200 上最新一波 verified：

| Strategy | Conc | P50 TTFT | P50 TPOT | tok/s/GPU |
|---|---:|---:|---:|---:|
| Low-latency / Marlin + DSpark | 1 | 308 | 1.72 | 606 |
| Low-latency / Marlin + DSpark | 16 | 662 | 8.39 | 2538 |
| Balanced / MXFP4 + DSpark | 64 | 1618 | 38.05 | 2994 |
| Balanced / MXFP4 + DSpark | 256 | 1932 | 104.53 | 4872 |
| High-throughput / Marlin target-only | 1024 | 195108 | 123.81 | 4573 |
| High-throughput / Marlin target-only | 4096 | 505509 | 123.97 | 4542 |

---

## 4. H100 及更低：Pro 几乎无官方吞吐矩阵

### H100 · Pro

cookbook 仅有占位：

```text
h100 / pro / fp4 / {low-latency,balanced,high-throughput} / nodes=multi-2
```

**没有 `speed[]` 数字。**

含义：

- 官方默认认为 H100 上 Pro 更偏 **多机（2 nodes）**
- 单机 8×H100 的 Pro Verified 吞吐目前空缺

### H100 · Flash FP4（有数，可作低配平台参考）

| Strategy | Conc | TTFT | TPOT | tok/s/GPU |
|---|---:|---:|---:|---:|
| low-latency | 1 | 205 | 3.19 | 319 |
| low-latency | 16 | 469 | 8.48 | 1539 |
| balanced | 64 | 726 | 23.11 | 2306 |
| balanced | 256 | 35793 | 48.46 | 2416 |
| high-throughput | 1024 | 209393 | 65.31 | 2252 |
| high-throughput | 4096 | 476764 | 66.0 | 2248 |

### 更低规格

| 硬件 | Pro 公开 Verified 吞吐 | 备注 |
|---|---|---|
| H20 | 未见 cookbook Pro 矩阵 | 社区 issue 多为 Flash/H20 讨论 |
| RTX PRO 6000 | 不支持 Pro | cookbook：V4-Pro 放不进 8×96GB |
| L40 / 消费卡 | 无 | 通常只讨论 Flash/实验路径 |
| MI300X/MI355X Pro | cookbook 有 recipe 占位 | **benchmarks.jsx 里 speed 为空** |

---

## 5. 和更高端卡的大致量级对比（帮助定预期）

同 workload（8192/1024），**Pro FP4 low-latency conc=1**：

| 硬件 | tok/s/GPU | TPOT |
|---|---:|---:|
| H200 | 170 | 5.65 ms |
| B200 | 243 | 4.25 ms |
| B300 | 243 | 4.2 ms |
| GB300 | 441 | 4.49 ms |

**Pro FP4 high-throughput 饱和区 tok/s/GPU：**

| 硬件 | 约值 | 说明 |
|---|---:|---|
| H200 | ~600 | 被 KV 容量钉死 |
| B200 | ~4200+ | 能继续随并发抬升 |
| B300 | ~4200–4400 | 同上 |
| GB300 | ~2800 | 另有配方/拓扑差异，勿直接横比绝对第一 |

---

## 6. 从 PR 还能再挖到什么（H200/低配）

| PR | 内容 | 对你是否有用 |
|---|---|---|
| [`#31363`](https://github.com/sgl-project/sglang/pull/31363) | 0.5.15 重测，含 H200 Pro FP4 | ✅ 主数据来源之一 |
| [`#33109`](https://github.com/sgl-project/sglang/pull/33109) | H200/B200 **Flash Official** verified | ⚠️ 仅 Flash，非 Pro |
| [`#29016`](https://github.com/sgl-project/sglang/pull/29016) | SM90 MegaMoE ON/OFF +22% | ⚠️ 多为 Flash/Hopper MoE 路径，不是完整 Pro 矩阵 |
| [`#23986`](https://github.com/sgl-project/sglang/pull/23986) | Revert 单机 H200 Pro low-latency recipe | 说明 H200 Pro 单机 recipe 曾不稳定 |
| [`#23896`](https://github.com/sgl-project/sglang/issues/23896) | H20 Flash balanced vs low-latency | Flash/H20，不是 Pro |

---

## 7. 实操建议（如果你主要只有 H200/更低卡）

### 想跑 Pro
1. **优先 8×H200 + FP4 low-latency / 浅并发**（官方有数）  
2. 需要更高并发：走 **多机（cookbook 的 multi-2 FP8/FP4 占位）**，但目前缺公开测速，需要自测  
3. 别指望单机 H200 上 ht/4096 还能线性涨吞吐——已经撞容量墙

### 想要可用吞吐/延迟
- 同平台优先看 **Flash** verified（H200/H100 数据完整）  
- 或自建 Pro 多机基线

### 复现命令骨架

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --max-concurrency <N> \
  --num-prompts <N_or_larger> \
  --flush-cache
```

注意：`#31363` 强调必须 **cache-cold（`--flush-cache`）**，否则 radix hit 会虚高。

---

## 8. 数据缺口清单（方便你继续追 PR/自测）

- [ ] H200 Pro FP8 multi-2 三策略测速  
- [ ] H100 Pro FP4 multi-2 三策略测速  
- [ ] H20 Pro（若社区有人跑）  
- [ ] MI355X Pro（recipe 有、benchmark 空）  
- [ ] 统一口径：output tok/s vs total tok/s/GPU、P50 vs Mean

---

## 来源文件

- `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`
- `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`
- PR `#31363`, `#33109`
