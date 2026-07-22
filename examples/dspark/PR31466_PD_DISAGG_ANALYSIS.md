# PR #31466 归纳总结:DSpark 支持 Prefill/Decode 分离(PD Disaggregation)

> 上游 PR:[sgl-project/sglang#31466](https://github.com/sgl-project/sglang/pull/31466) · "[Spec] DSpark support prefill/decode disaggregation"
> 作者 zhangxiaolei123456 · 目标分支 `main` · 状态 OPEN(截至 2026-07-22,存在冲突未合并)
> 体量:26 个文件,+4046/−153,104 个 commit
> 背景:roadmap [#30344](https://github.com/sgl-project/sglang/issues/30344)([Tracking] DSpark in SGLang);前身 [#30513](https://github.com/sgl-project/sglang/pull/30513)("DSpark support PD and DeepEP",已关闭——作者按评审意见把 DeepEP 部分剥离,本 PR 只做 PD
> 本文行号引用均为该 PR diff 中的新增代码位置。

---

## 目录

1. [要解决的问题](#1-要解决的问题)
2. [设计思路](#2-设计思路)
3. [架构变化总览](#3-架构变化总览)
4. [实现分析(逐层展开)](#4-实现分析逐层展开)
5. [兼容性与能力边界](#5-兼容性与能力边界)
6. [测试覆盖](#6-测试覆盖)
7. [评审动态与工程观察](#7-评审动态与工程观察)

---

## 1. 要解决的问题

回顾 DSpark 的核心机制(见 [DSPARK_DEEP_DIVE.md](./DSPARK_DEEP_DIVE.md)):draft 模型自己不算 prompt,它的全部上下文来自 **prefill 时把目标模型中间层 hidden 投影注入 draft KV cache**。这在单机(非分离)模式下没问题——prefill 和 decode 在同一进程,`_forward_prefill` 拿到 aux hidden 就地注入。

但在 **PD 分离**部署下,Prefill 和 Decode 是两个独立集群:

- Prefill 节点做完 prompt 前向,通过 Mooncake/NIXL 把 **KV cache** 传给 Decode 节点;
- 现有 PD 协议只传 KV(+ 少量请求级状态如 SWA ring / C128 state),**目标模型的 aux hidden 不在协议里**;
- 于是 Decode 侧的 draft KV 是空的、`spec_info` 也没有正确初始化——decode 一开始投机,draft 就在无上下文状态下瞎猜,等于 DSpark 在 PD 模式下不可用。

本 PR 的目标:**把"DSpark 引导所需的目标 hidden"变成 PD 传输协议的一等公民**,从 Prefill 捕获、传输到 Decode、在第一次 draft 步之前注入 decode 侧 draft KV;同时保证非 PD 路径行为完全不变。

一个使问题显著变难的现实约束:**长 prompt**。hidden 的体量是 `prompt_len × (层数 × hidden_size)`,如果一次性物化再传输,Prefill/Decode 两侧都要吃下一个巨大张量。所以 PR 演进出了**流式(streaming)传输**——这也是 104 个 commit 里迭代最多的部分(从 one-shot → packetized → streaming + ACK 流控)。

---

## 2. 设计思路

六条主线,均可从 PR 描述与代码相互印证:

**① 复用 PD 状态传输协议,而不是另起炉灶。** 现有协议已有 `StateType`(MAMBA / SWA_RING / C128_STATE 等请求级状态)的扩展点,本 PR 新增 `StateType.PD_HIDDEN`(`base/conn.py`),让 hidden 走与 KV/state 相同的 bootstrap-metadata-transfer 通路。传输元数据通过 `send_metadata(..., spec_metadata=dict)` 新参数携带,老后端不实现即忽略。

**② Decode 主导、Prefill 执行的"拉"式协商。** 需要什么 hidden 由 **Decode 在预分配阶段声明**(target layer ids、hidden 起止偏移、目的缓冲、radix 策略、PP 切片),打包成 `spec_metadata` JSON 随 bootstrap 发给 Prefill;Prefill 校验后按计划捕获与发送。这样"要不要传、传哪段、传到哪"的决策集中在一端,Prefill 保持无状态执行。

**③ PP 感知的切片。** 目标层可能分布在 Prefill 的不同 PP rank 上,只有**拥有 DSpark target layers 的那个 PP rank** 才捕获和发送 hidden;中间 rank 的捕获结果通过 PP proxy tensors(`pd_aux_hidden_states_*`)向后级传递。Decode 侧按 `get_pp_indices` 预先算好每个 PP rank 的 `(layer_ids, slice_start, slice_len)` 切片布局。

**④ 流式传输 + 完成语义分离。** 长 prompt 下 hidden 分 chunk 流水:Prefill 逐 prefill-chunk 捕获、写入源窗口、RDMA 发送、`PD_HIDDEN_CHUNK_READY` 通知;Decode 逐 chunk 注入 draft KV、CUDA event 同步后回 `PD_HIDDEN_CHUNK_ACK`;ACK 只控制**流式窗口复用**,`hidden_done` 与 `kv_done` 是两个独立状态位,请求成功仍走原 KV/metadata 通路——hidden 传输失败/成功不与 KV 完成混淆。

**⑤ 注册缓冲池走 GDR,避免 CPU 中转。** 两侧各建一个 `PDHiddenRowPool`(GPU 上的 `(rows, hidden_width)` 行池,注册给 RDMA 引擎),源/目的都是行寻址的注册显存,传输保持 GPU-Direct 路径;行池带区间合并的分配器,并支持按 chunk 的动态目的地址(`PDHiddenTransferPlan` / `dynamic_dst.row_chunks`)。

**⑥ PP 共识下的"无副作用探测 + 事务性提交"。** PP 模式下所有 rank 必须对"哪些请求进入流水线"达成一致,而 hidden 行池是有限资源——如果某 rank 在共识前就分配了行,另一 rank 因资源不足拒绝同一请求,就会 rank 间发散。PR 把准入拆成 `_probe_bootstrap_ready`(只验证信用额度,不占资源)→ 跨 rank 共识 → `finalize_bootstrap`(真正分配),并把 PP 共识的 rid 集合运算从无序 `set` 交并改成**保序**的 intersection/union(顺序不一致同样会导致 rank 发散)。

---

## 3. 架构变化总览

### 3.1 分层地图

```text
┌─ 协议层 ──────────────────────────────────────────────────────┐
│ base/conn.py         +StateType.PD_HIDDEN,sender 新钩子        │
│ common/conn.py       KVManager 流式能力探测/请求完成语义默认实现  │
│ fake|nixl|mori/conn  send_metadata 签名对齐(不实现流式)        │
├─ 状态/资源层 ─────────────────────────────────────────────────┤
│ common/utils.py      +PDHiddenChunk / PDHiddenRequestState      │
│                      TransferKVChunk 增 12 个 pd_hidden_* 字段  │
│ disaggregation/utils +PDHiddenRowPool(区间分配行池)            │
│                      +PDHiddenTransferPlan / MetadataBuffers 挂池│
│ hidden_events.py     ★新 461 行:PDHiddenEventManager           │
│                      (ACK 等待/park/唤醒/quiesce/完成队列)      │
│ hidden_state.py      ★新:per-Req 的 PDHiddenReqState(弱引用表)│
├─ 传输层 ──────────────────────────────────────────────────────┤
│ mooncake/conn.py     +819:READY/ACK ZMQ 包、_send_pd_hidden_    │
│                      packet、transfer_worker 的 hidden 先行流程  │
├─ 调度层 ──────────────────────────────────────────────────────┤
│ decode.py            +499:prealloc 期声明 spec_metadata、       │
│                      流式 chunk 排水/注入/ACK、双完成位提交       │
│ prefill.py           +883:bootstrap 两阶段、逐 chunk 捕获写池、  │
│                      _pd_hidden_payload、owner-direct 发送       │
│ scheduler_pp_mixin   +100:PP 共识保序化、pd_aux 走 proxy、       │
│                      owner-direct 判定                          │
│ scheduler_disaggregation_init.py ★新:init_disaggregation 迁移  │
├─ 模型层 ──────────────────────────────────────────────────────┤
│ deepseek_v4.py       捕获层来源改为 forward_batch 按批注入;      │
│                      PP proxy 携带/接力 pd_aux_hidden_states_*   │
│ forward_batch_info   +pd_hidden_capture_layer_ids(按批捕获开关) │
├─ 投机解码层 ──────────────────────────────────────────────────┤
│ dspark_disaggregation.py ★新 282 行:元数据配置解析 +            │
│                      resolve_hidden_bootstrap_plan(共享校验器)  │
│ dspark_worker_v2.py  +inject_pd_hidden_chunk(流式注入入口)      │
│ dspark_kv_inject.py  inject_target_hidden 返回 CUDA event        │
│ dflash_info_v2.py    DFlashDraftInputV2 增 prefill_tail_* 4 字段 │
│ spec_info.py         build_disagg_draft_input:DSpark 分支        │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 端到端数据流

```mermaid
sequenceDiagram
    participant D as Decode 调度器
    participant P as Prefill 调度器
    participant M as 目标模型(Prefill, PP)
    participant W as Mooncake worker(P 侧)
    participant DW as DSparkWorker(D 侧)

    D->>D: pop_preallocated:算 PP 切片布局,行池 alloc 接收窗口
    D->>P: send_metadata(kv indices + spec_metadata{PD_HIDDEN 计划})
    P->>P: probe(无副作用信用检查) → PP 共识 → finalize(行池 alloc 源窗口)
    loop 每个 prefill chunk
        M->>M: forward(pd_hidden_capture_layer_ids 按批捕获)
        M-->>P: aux hidden(PP 中间 rank 经 proxy 接力)
        P->>P: _write_pd_hidden_rows_for_batch:切本 rank slice 写入源行池
        P->>W: send_kv_chunk(附 pd_hidden_chunk_meta + source_event)
        W->>D: RDMA 行数据 → PD_HIDDEN_CHUNK_READY(ZMQ)
        D->>DW: 顺序校验(accept_chunk) → read_view → inject_pd_hidden_chunk
        DW-->>D: CUDA event(注入完成的异步凭据)
        D->>W: PD_HIDDEN_CHUNK_ACK(event.sync 后)
        W->>W: 唤醒 park 的下一 chunk;释放源窗口行
    end
    W->>W: last chunk ACK 齐 → mark_pd_hidden_request_done(早释放)
    D->>D: kv_done ∧ hidden_done → _commit_transfer_to_req
    D->>DW: build_disagg_draft_input(prefill 输出 token 作首个锚点)
    Note over DW: 首次 draft 步:draft KV 已含目标语义,正常出块
```

---

## 4. 实现分析(逐层展开)

### 4.1 协议扩展(base/common/conn.py)

- `StateType.PD_HIDDEN = "pd_hidden"`:注释明确其语义——"Target aux hidden rows used to bootstrap decode-side draft KV"。
- sender 基类新增两个**默认 no-op** 钩子:`set_source_event(event)`(记录源数据就绪的 CUDA event,worker 发送前 `synchronize`,解决"调度线程写池、传输线程读池"的跨线程 GPU 可见性)与 `set_pd_hidden_chunk_meta(hidden_start, row_len, is_last, release_indices)`(把当前 chunk 的窗口坐标挂到 sender 上)。
- `send_metadata(..., spec_metadata: Optional[dict])`:decode→prefill 的一次性协商载荷;Mooncake 侧以 JSON 附加在 ZMQ 消息第 10 段(`TransferInfo.from_zmq`),`req_to_pd_hidden_meta[room]` 缓存。
- `CommonKVManager` 提供整套**能力边界默认值**:`supports_pd_hidden_streaming() -> False`、`mark/pop_pd_hidden_request_done` no-op——NIXL/MORI 未实现流式时自然落在安全路径,不会误入半实现的释放语义(PR 描述明确了这一 capability boundary)。

### 4.2 状态与资源层

**`PDHiddenRequestState`(common/utils.py)** ——decode 侧每请求的 hidden 传输状态机,与 KV 状态严格分离:

- 三种构造:`disabled()`(非 DSpark 请求,hidden_done 恒真)/ `full(start, end)`(一次性传输)/ `streaming_state(start, end)`;
- `accept_chunk(chunk)` 返回 `accepted / future / stale` 三态:`hidden_start > next_start` 为 future、`<` 为 stale(乱序即 fail-fast 抛错),最后一个 chunk 必须精确终止在 `end`;`defer_hidden_done=True` 允许把 hidden_done 推迟到 ACK 完成后再置位;
- `request_done() = kv_done ∧ hidden_done`——这就是"完成语义分离"的落点。

**`PDHiddenRowPool`(disaggregation/utils.py)** ——注册给 RDMA 引擎的 GPU 行池:

- `(size, hidden_width)` 的单块 buffer,`get_state_buf_infos()` 暴露 `(data_ptr, nbytes, item_len)` 供 `setup_state_kv_args` 注册为 PD_HIDDEN 状态组件;
- 分配器用**有序空闲区间表**:优先找能放下 n 行的连续区间(连续行可走单次大 RDMA/`copy_`),碎片时回退为跨区间取最低行;`free` 做去重、容忍已释放行、并区间合并——这些行为都有单测锁定;
- `write` 对连续 index 走切片 `copy_`(窄于池宽时先 `zero_` 填充),`read_view` 对连续 index 返回**零拷贝视图**(decode 注入路径用它避免一次克隆)。

**`PDHiddenEventManager`(hidden_events.py,新 461 行)** ——被刻意设计成 backend 无关的并发状态中枢(注释:"can be reused by Mooncake, NIXL, MORI, or future backends"),Mooncake 的 manager 全部方法都是对它的一行转发。管的东西:

- **ACK 等待/park**:`park_chunk_for_ack` 把发完 READY 的 chunk 挂起(带 300s 定时器,超时置 Failed 并重投队列);`handle_chunk_ack` 计数达到 expected(= decode 侧接收者数)后唤醒;
- **per-room 串行**:`inflight_chunks[room]` + `park_chunk_behind_room` / `wake_next_room_waiter`——同一请求的流式 chunk 在 ACK 流控下必须保序,后一个 chunk 排队等前一个完成;
- **decode 侧完成流水**:`submit_chunk_ack` 起后台线程等注入 event 的 `synchronize()`,完成后进 `ack_completions` 队列并用 inproc ZMQ PUSH 自唤醒 decode 线程(`drain_ack_completions` 在那边发真正的 ACK 包);
- **quiesce/生命周期**:`begin/end_transfer` + `wait_transfers_quiesced`、`wait_ack_completions`(释放接收行前必须等在途 ACK 排干,否则行被复用时 RDMA 还在写)。

### 4.3 Decode 侧(decode.py,+499)

**预分配期(`pop_preallocated`)** 对 DSpark 请求追加一整段协商构造:

1. 计算 `pd_hidden_start = total_prefix_len`、`pd_hidden_len = origin_input_len − total_prefix_len`——**hidden 窗口与 KV 传输窗口对齐**(decode 本地已有 radix 前缀的部分,KV 不传,hidden 也不传);
2. 从 `model_runner.spec_aux_config` 拿 target layer ids,按 `get_pp_indices` 为每个 Prefill PP rank 算 `(layer_ids, slice_start, slice_len)`;当前实现要求**恰好一个非空 slice 且覆盖全宽**(target layers 跨 PP rank 的一般布局显式报错,留作 future work);
3. 行池 `alloc(min(hidden_len, pool.size))` 拿接收窗口;拿不到就跳过本请求并限频告警(30s)——**行池成为新的准入资源**;
4. 组 `spec_metadata`(streaming 标志、窗口行数、radix 策略、pp_slices、dst_indices、hidden_size、target_layer_ids)随 `send_metadata` 发出;`PDHiddenRequestState.streaming_state(...)` 建立状态机。

**传输期(`DecodeTransferQueue.pop_transferred`)** 每轮轮询做三件事:

- `_consume_pd_hidden_acked_chunks`:收割已 ACK 完成的 chunk,最后一个 chunk 置 `hidden_done`;
- `_drain_pd_hidden_ready_chunks`:取 READY 队列,按 `hidden_start` 排序后逐个 `accept_chunk` 校验(future/stale 一律 RuntimeError → 记失败),`read_view` 零拷贝读行,调 `draft_worker.inject_pd_hidden_chunk` 注入,拿到 CUDA event 后 `submit_pd_hidden_chunk_ack`;
- 提交门槛改为**双完成位**:`poll == Success` 时先 `mark_kv_done()`,`request_done()`(kv ∧ hidden)才 `_commit_transfer_to_req`;
- 所有失败/中止/释放路径(预分配失败、room 校验失败、abort、release_memory_occupation)都统一走 `_release_pd_hidden_rows`——先 `wait_pd_hidden_ack_completions` 排干在途 ACK,再按 PP 去重释放行池。

**注入入口(`DSparkWorkerV2.inject_pd_hidden_chunk`)** 是全 PR 最"轻"也最关键的一段:按 `hidden_start + row_len` 构造 positions、查 `req_to_token` 页表得 cache_loc、调既有的 `TargetHiddenKvInjector.inject_target_hidden`——**完全复用非 PD 路径的注入 kernel**(投影 + norm + RoPE + 写池),只是数据源从本进程前向变成了远端传来的行。`inject_target_hidden` 因此改为返回 CUDA event(`kv_inject.py`),给 ACK 流控提供异步完成凭据。

**draft 输入自举(`spec_info.py build_disagg_draft_input`)**:用 Prefill 传来的**输出 token 作为第一个 decode 锚点**构造 `make_next_draft_input(bonus_tokens=last_tokens, new_seq_lens=batch.seq_lens)`;overlap 模式下还要向 future_map publish/stash,保持与非 PD overlap 相同的中继协议。

### 4.4 Prefill 侧(prefill.py,+883)

**bootstrap 两阶段(核心的 PP 正确性设计)**:

- `_probe_bootstrap_ready(req, metadata_credits, hidden_row_credits)`:**纯探测**——校验 metadata(经共享的 `resolve_hidden_bootstrap_plan`)、计算本请求要消耗的 metadata buffer 数与 hidden 行数,与本地信用比较;不分配任何东西。`get_ready_bootstrapped_rids_for_pp` 用它产出**有序**的候选 rid 列表供 PP 共识;
- `finalize_bootstrap → _finalize_pd_hidden_bootstrap`:共识后才真正 `pool.alloc`(流式模式源行 lazy 分配,此处不占),把 plan 写进 per-Req 的 `PDHiddenReqState`(`capture_layer_ids / src_indices / dst_indices / written 位图`);失败则回滚刚分配的 metadata buffer 并 abort。

**逐 chunk 捕获与写池(`_write_pd_hidden_rows_for_batch`)**,挂在 `process_batch_result_disagg_prefill` 上:

- 从 batch result 提取 hidden(本 rank 前向产物,或 PP proxy 里的 `pd_aux_hidden_states_*` 拼接);要求捕获却拿不到 hidden 时对相关请求统一 fail-fast;
- 对每个请求算 `[chunk_start, chunk_end) ∩ [hidden_start, hidden_start+hidden_len)` 的交集——**radix 前缀裁剪的同时保持绝对 token 偏移**;PP 下再按本 rank slice 的 `(slice_start, slice_len)` 切列;
- 流式模式源行**按 chunk 分配**;写前若发现上一个 chunk 还没发出去(`current_start` 不同)则先 flush(`send_kv_chunk(end_idx=旧chunk末尾)`),防止 `dspark_hidden_current_*` 槽被覆盖——这是 PR 描述里点名的一个 hazard 修复;
- 非流式模式维护 `written` 位图,发送前 `_pd_hidden_payload` 检查 `all(written)`,缺行直接 RuntimeError(fail-fast on incomplete rows)。

**发送整合(`send_kv_chunk` 改造)**:有待发 hidden chunk 时,即使对齐后的 KV page 数为 0 也要发(hidden-only chunk);发送前录 `source_event`、挂 `pd_hidden_chunk_meta`;流式模式发出后立即把 `src_indices` 置 None(所有权移交传输层,ACK 后由 worker 释放)。

**优化的让路**:`_requires_pd_hidden_transfer` 为真的请求**禁用 optimistic prefill**(元数据未到就先跑前向的优化)——hidden 捕获必须在 bootstrap 计划确定后才能进行,否则第一个 chunk 的 hidden 已经错过。

### 4.5 Mooncake 传输层(mooncake/conn.py,+819)

`transfer_worker` 的 chunk 处理顺序被改造成 **hidden 先行**:

1. chunk 带 `source_event` 先 `synchronize`(等调度线程的池写入对传输线程可见);
2. 若 chunk 携带 PD_HIDDEN 状态且未发送:检查 per-room inflight(不是当前 chunk 则 park 到 room 队列)→ 对每个非 dummy 接收者 `_send_pd_hidden_packet`(直连注册缓冲的行寻址 RDMA;`dynamic_dst.row_chunks` 支持按 chunk 的目的指针,`packet_idx` 支持一个 chunk 拆多包)→ 发 `PD_HIDDEN_CHUNK_READY` → `park_pd_hidden_chunk_for_ack` 挂起等 ACK;
3. ACK 齐(或超时失败)后:`finish_streaming_chunk` 释放该 chunk 的源行、清 inflight、唤醒 room 队列里的下一个 chunk;最后一个 chunk 标记 `mark_pd_hidden_request_done`——**Prefill 源资源在 KV 请求完成之前就归还**(PR 描述:"Release Prefill hidden rows as soon as the hidden transfer finishes...instead of waiting for the full KV request success");
4. 之后才进入原有的 KV 页发送流程。decode 线程侧新增两个 ZMQ 消息分支:`PD_HIDDEN_CHUNK_READY`(入 ready 队列供调度线程排水)与 `PD_HIDDEN_CHUNK_ACK`(prefill 侧计数唤醒)。

失败路径全部汇入 `_mark_session_failed_and_sync` + `wake_ack_waiters`:任何 room 失败都会把 park 住的 chunk 全部放行(避免线程泄漏在 Condition 上),并同步状态给 decode 端点。

### 4.6 PP 支持(scheduler_pp_mixin.py + deepseek_v4.py)

- **捕获开关按批下发**:新的 `ForwardBatch.pd_hidden_capture_layer_ids` 从 batch 里任一请求的 `PDHiddenReqState.capture_layer_ids` 推导(`forward_batch_info.py`),并强制 `CaptureHiddenMode.FULL`。DeepSeek-V4 的 forward 优先读它、回退到全局 `self.dspark_layers_to_capture`——把"是否捕获"从**进程级静态配置**变成**请求级动态属性**(只有 PD 请求在场时才付捕获代价);
- **PP 接力**:非末级 rank 把捕获的 aux hidden 以 `pd_aux_hidden_states_{idx}` 放进 `PPProxyTensors` 传给下一级;末级 rank 若 `logits_output.hidden_states` 为空则用累计的 aux hidden 拼接填充。同时加了 `hidden_tokens != position_tokens` 的 fail-fast(捕捉 PP FIFO 错配);`_pp_prep_batch_result` 按 microbatch 从 proxy 提取 aux 键,避免按到达顺序错配请求(commit "Match PP outputs by microbatch id");
- **owner-direct 短路**:流式模式下,拥有 target layers 的 rank 可以在 `run_batch` 之后**直接发送 hidden-only chunk**(`_pp_maybe_send_dspark_owner_direct_hidden` → `send_dspark_owner_direct_hidden_for_batch`),并从 proxy 里剥掉 aux 键——不必让 hidden 随流水线走到最后一级再发,降低传输延迟与 PP IPC 体积;
- **共识保序**:`_pp_pd_get_bootstrapped_ids` / `_pp_pd_get_prefill_transferred_ids` 从 `set` 交并改为 `_pp_ordered_intersection/_pp_ordered_union`——集合运算丢失顺序会让不同 rank 以不同顺序 finalize,资源分配次序发散。

### 4.7 spec 层的数据结构扩展

`DFlashDraftInputV2` 新增四个字段:`prefill_tail_hidden_states / prefill_tail_valid_mask / prefill_tail_start_positions / prefill_tail_hidden_projected`,并补全 `filter_batch`(按 `valid_mask` 长度展开的 row_mask 过滤变长行)与 `merge_batch`(两侧缺省时用零行/零掩码对齐再 cat)——这是"decode 侧把收到的 hidden 尾部挂在跨步状态上、随批次分裂/合并保持一致"的载体。`make_draft_input_v2` / `make_next_draft_input` 相应加了透传参数。

### 4.8 配置与元数据解析(dspark_disaggregation.py,新 282 行)

- `resolve_disagg_metadata_config`:决定本进程要不要建 hidden 行池及其形状——`hidden_width = len(target_layer_ids) × hidden_size`;池行数默认取 `max(max_prefill_buffer_tokens, max_prefill_tokens)`,可用 `SGLANG_PD_HIDDEN_POOL_TOKENS` / `SGLANG_PD_HIDDEN_RECV_POOL_TOKENS`(decode 侧)覆盖;PP 下不拥有 target layers 的 Prefill rank 池行数为 0;**backend 白名单**:非 Mooncake/Fake 直接 `NotImplementedError`;
- `resolve_hidden_bootstrap_plan`:probe 与 finalize 共用的校验器,集中了全部协商不变量——`hidden_start == decode_prefix_len`(hidden 窗口必须与 KV 窗口对齐)、**prefill/decode radix 策略必须一致**(否则 Prefill 跳过的缓存前缀在 Decode 侧没有 hidden 来源)、PP slice 完整性、`slice_len == len(local_layers) × hidden_size`、prefill 配置层与 decode 元数据层一致、dst_indices 长度合法、池容量足够。每条失败都返回人类可读的错误串,由调用方决定 abort 或重试。

---

## 5. 兼容性与能力边界

PR 描述的承诺与代码的对应:

| 承诺 | 代码落点 |
|---|---|
| 非分离 DSpark 行为不变 | 所有新逻辑都 gate 在 `spec_algorithm.is_dspark() ∧ PD 模式 ∧ StateType.PD_HIDDEN in state_types`;模型层回退到全局捕获配置 |
| 既有 KV / C128 / 状态传输通路不动 | PD_HIDDEN 是新增的 state 组件,原组件的 payload 构造不变 |
| 只有 Decode 请求 PD_HIDDEN 元数据时才激活 | Prefill 由 `req_to_pd_hidden_meta`(decode 发来的)驱动 |
| 默认显存占用不变 | prewarm 默认关(`SGLANG_DSPARK_PD_HIDDEN_RECV_PREWARM_*`);无 target layers 的进程池行数为 0 |

**当前边界(作者在描述中如实列出)**:

1. **仅 Mooncake 支持流式**;NIXL/MORI 保持非流式默认能力(实际上 decode 路径当前要求 `supports_pd_hidden_streaming() ∧ inject_pd_hidden_chunk`,不满足直接 abort——即目前实际可用的只有 Mooncake/Fake);
2. **Prefill 与 Decode 的 radix cache 策略必须一致**;在 DeepSeek-V4 Decode radix 支持(#31097)落地前,Prefill radix 应保持关闭——否则 Prefill 会跳过缓存命中的 prompt 段,而那段的 hidden Decode 拿不到;
3. **PP 布局限制**:decode 侧固定行池要求 target layers 集中于单一 PP rank(单一非空 slice 覆盖全宽);跨 rank 切分、TP hidden 分片、CP token 范围重组都是 future work;
4. **DeepEP 支持被剥离**:前身 #30513 同时做 PD+DeepEP,评审后作者决定"changes too complicated",DeepEP 另开 PR。

## 6. 测试覆盖

`test/registered/unit/disaggregation/test_pd_hidden_state.py`(169 行)锁定两块纯逻辑:

- **`PDHiddenRequestState` 状态机**:disabled/full/streaming 三态的完成语义、`accept_chunk` 的 accepted/future/stale 判定、last chunk 偏移校验、越界拒绝、`defer_hidden_done` 与 ACK 的配合、`reset` 归位;
- **`PDHiddenRowPool` 分配器**:连续优先、释放合并、碎片回退取最低行、重复/已释放行的容忍。

分布式传输路径(Mooncake 线程、ZMQ、RDMA)没有单测——依赖端到端验证,这也是这类 PR 评审的常见难点。

## 7. 评审动态与工程观察

**评审现状**(截至写作时):OPEN、与 main 有冲突(`mergeable: false`);104 个 commit。maintainer hnyls2002 的意见很有代表性:*"Seems like a lot of duplicated code hunks. Could you please get a roadmap for your PR? This should have a clear breakdown for each hunk of changes."*——即体量与耦合面(disaggregation × PP × spec × 传输后端四个维度的笛卡尔积)使得单 PR 评审困难,后续大概率需要拆分。gemini bot 的自动评审指出过若干早期 bug(后续 commit 已多轮修复)。社区讨论:DeepEP 拆出、GLM-5.2 支持意向、与 #31513(dsv4-flash-dspark PD 修复)的关系澄清。

**从 commit 历史可见的演进轨迹**(值得学习的迭代过程):one-shot 整块传输 → packetized 分包 → streaming + ACK 流控;CPU 中转缓冲 → GPU 注册缓冲(GDR);请求数节流 → 行池信用制;隐藏在 KV 完成里的释放 → hidden/KV 完成语义分离。每一步都是被长 prompt 的内存/延迟现实推着走的。

**个人分析:三个最值得借鉴的设计点**

1. **完成语义分离**(`hidden_done ⊥ kv_done`):把"数据到齐"与"资源可释放"拆成独立事件,允许源端资源在整个请求成功之前归还——这是所有流式传输系统的通用范式,PR 用一个 12 行的小状态机把它表达得很干净,且有单测锁定;
2. **probe/commit 两阶段准入**:分布式共识(PP rank 一致性)与有限资源(行池)组合时,"先探测后提交"是避免 rank 发散的标准解;顺带修的 set→ordered 集合运算,是一类非常隐蔽的分布式 bug;
3. **注入路径的最大复用**:decode 侧新增的只有"查页表 + 调既有注入器"十几行,投影/norm/RoPE/掩蔽全部复用非 PD 代码——协议层动得多、计算层几乎不动,这让正确性论证可以聚焦在传输语义上。

**个人分析:潜在风险点**

- `hidden_events.py` 的并发模型重(4 把锁/条件变量 + 定时器线程 + 每 chunk 一个 ACK 等待线程 + inproc ZMQ 自唤醒),长尾场景(超时与 ACK 竞态、abort 与 park 竞态)的正确性主要靠 fail-fast 兜底,缺乏针对性的并发测试;
- 每个流式 chunk 一次 ZMQ READY + 一次 ACK + per-room 串行,小 chunk 高频场景的控制面开销可能显著,`enqueue_time` 字段暗示作者已在观测排队延迟;
- decode 准入把行池耗尽变成静默的 prealloc 阻塞(仅 30s 限频告警),池配小了会表现为"decode 吞吐莫名下降",可观测性还可以加强。

---

## 附:与本目录其他文档的关系

- 非 PD 模式的 DSpark 全链路(本 PR 的"被引导对象"):[DSPARK_DEEP_DIVE.md](./DSPARK_DEEP_DIVE.md);
- 交互式分步动画:[dspark_interactive_walkthrough.html](./dspark_interactive_walkthrough.html);
- 本 PR 补上的正是 DEEP_DIVE 第 3 章 "Prefill:注入 draft KV" 那一步在跨进程部署下的等价物:`inject_target_hidden` 的数据源从"本进程目标前向的 aux hidden"换成"经 PD_HIDDEN 协议从 Prefill 集群流式传来的行"。
