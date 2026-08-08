# DeepSeek-V4 Communication & Context Parallelism PR 核对清单

> 来源：[#33636](https://github.com/sgl-project/sglang/issues/33636) 分类 *Communication & context parallelism*  
> 状态抓取时间（UTC）：2026-08-08 08:08  
> 仓库：`sgl-project/sglang`  
> 用途：在另一个代码树/分支上逐项核对「做了没有」；**已合入 upstream main 的单独标注**。

## 0. 总表（先扫一眼）

| # | 状态 | 分类 | 一句话 | +/- | Author |
|---|---|---|---|---|---|
| [#30700](https://github.com/sgl-project/sglang/pull/30700) | ❌ OPEN | 通信原语 | FlashInfer MNNVL 纯 allreduce（多机 TP / GB200） | +476/-6 | @wenscarl |
| [#33236](https://github.com/sgl-project/sglang/pull/33236) | ❌ OPEN | Prefill CP 存储 | 去掉 prefill CP 每层 KV/compressor 完整物化，改 direct KV + semantic compress | +1533/-44 | @foraxe |
| [#28639](https://github.com/sgl-project/sglang/pull/28639) | ❌ OPEN | 通信重叠 | symmetric-memory 上 AG⊗GEMM / MoE⊗RS overlap | +1367/-37 | @Zqy11 |
| [#32059](https://github.com/sgl-project/sglang/pull/32059) | ❌ OPEN | Prefill CP 存储 | VMM Shared KV：每 CP rank 持有物理页分片 | +7748/-121 | @taoyuanyuan |
| [#33382](https://github.com/sgl-project/sglang/pull/33382) | ❌ OPEN | Prefill CP LayerSplit | CP Cache LayerSplit 公共 infra（1/N） | +513/-210 | @jellysnack |
| [#29187](https://github.com/sgl-project/sglang/pull/29187) | ❌ OPEN | Prefill CP LayerSplit | DSV4 CP KV LayerSplit 完整实现 | +3596/-415 | @jellysnack |
| [#33532](https://github.com/sgl-project/sglang/pull/33532) | ✅ MERGED | Prefill CP 策略 | DSV4 接入 CP-v2 / interleave strategy | +161/-59 | @hzh0425 |
| [#30416](https://github.com/sgl-project/sglang/pull/30416) | ❌ OPEN | Decode CP | DeepSeek V4 Decode Context Parallel (DCP) | +10346/-325 | @shiyu7 |
| [#33250](https://github.com/sgl-project/sglang/pull/33250) | ❌ OPEN | 正确性 | 修 non-EP TBO + attn TP>1（依赖 #31700） | +157/-34 | @mikekg |
| [#33217](https://github.com/sgl-project/sglang/pull/33217) | ❌ OPEN | 正确性 | 对非法 non-EP TBO+attnTP>1 做 policy guard | +62/-0 | @Oxygen56 |
| [#31700](https://github.com/sgl-project/sglang/pull/31700) | ❌ OPEN | 正确性 | dp_gather_partial → dp_gather_replicate（attn-TP 副本） | +14/-4 | @mikekg |
| [#30885](https://github.com/sgl-project/sglang/pull/30885) | ❌ OPEN | 产品形态 | DSV4 支持 PDMux（同进程 P/D mux） | +795/-44 | @shipiyouniao |

### 合入情况小结

- **已合入 upstream main：仅 [#33532](https://github.com/sgl-project/sglang/pull/33532)**（2026-08-07）。详见下方「已合入专节」。
- 其余 11 个 PR 截至抓取时均为 **OPEN**（#30416 仍带 DRAFT 色彩）。
- 正确性链路建议顺序：`#31700 → (#33217 guard / #33250 fix)`。
- Prefill CP 存储栈建议顺序：`#33532(已合) → #33236 → #32059 → #33382 → #29187`，`#28639` 可并行。

### ✅ 已合入专节（upstream main）

| PR | 合入日 | 核对关键词 |
|---|---|---|
| [#33532](https://github.com/sgl-project/sglang/pull/33532) CP V2 for dsv4 | 2026-08-07 | `SGLANG_ENABLE_CP_V2`, `cp_materialize_global_token_order`, `cp_round_robin_input_ids_v2`, `--cp-strategy interleave` |

若目标树 merge-base 晚于该合入，应已自带；若基于更旧 fork，按专节 / 下文 #33532 完整 diff 回放。

### 异地核对用法

对每个 PR，在目标分支上检查：

1. **开关/环境变量**是否存在（见各节「开关」）。
2. **关键符号/文件**是否存在（见各节「关键文件 / 符号」）。
3. **关键 diff 片段**是否已有等价改动（见各节「关键 diff」；大 PR 只贴代表性 hunk）。
4. 在本节末尾勾选框标注本地状态。

---

## PR #30700 — [NVIDIA] Add flashinfer MNNVL backend for allreduce only

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/30700
- **分类**：通信原语
- **Author**：@wenscarl  ·  diffstat `+476/-6`
- **一句话**：FlashInfer MNNVL 纯 allreduce（多机 TP / GB200）

### 做什么

FlashInfer MNNVL 增加 pure allreduce（不融合 RMSNorm），供多机 TP；custom-AR 在 GB200/GB300 多节点不适用。

### 开关 / 启用方式

无单独 CLI 大开关；通过 FlashInfer allreduce-only workspace 挂到 `GroupCoordinator.all_reduce`：

- 符号：`flashinfer_allreduce` custom op、`GroupCoordinator._fi_workspace_hint`、`_can_use_flashinfer_allreduce`
- 文件：`flashinfer_comm_fusion.py` / `parallel_state.py` 注册
- 单测：`test_flashinfer_comm_fusion.py`、`test_layer_communicator_fusion_gate.py`

### 关键文件 / 符号（异地核对点）

FlashInfer MNNVL pure AR；`parallel_state` bootstrap；layer communicator fusion gate

- `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/layers/communicator.py`
- tests under `test/registered/unit/layers/`

### 关键 diff

#### parallel_state.py

```diff
@@ -185,6 +185,22 @@ def outplace_all_reduce(
     return group._all_reduce_out_place(tensor, outplace_all_reduce_method)
 
 
+@register_custom_op(out_shape="tensor")
+def flashinfer_allreduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
+    """FlashInfer kAllReduce over ``group_name``.
+
+    Registered as a custom op so it stays opaque under Dynamo and can run inside
+    piecewise CUDA graphs. Applicability is decided by
+    ``GroupCoordinator._can_use_flashinfer_allreduce`` before the call -- this op
+    has no fallback of its own.
+    """
+    assert group_name in _groups, f"Group {group_name} is not found."
+    group = _groups[group_name]()
+    if group is None:
+        raise ValueError(f"Group {group_name} is destroyed.")
+    return group._flashinfer_allreduce(tensor)
+
+
 @register_custom_op(mutates_args=["output"])
 def reg_all_gather_into_tensor(
     output: torch.Tensor, input: torch.Tensor, group_name: str
@@ -291,6 +307,10 @@ def __init__(
         self.local_rank = local_rank
         self.device_group = None
         self.cpu_group = None
+        # Which FlashInfer fusion workspace this group owns, or None when the
+        # group is not eligible for the allreduce-only kAllReduce path. Stamped
+        # by _tag_groups_for_flashinfer_allreduce_only() after group init.
+        self._fi_workspace_hint: Optional[str] = None
         self.local_size = get_int_env_var("LOCAL_SIZE", 0)
 
         if is_cuda_alike():
@@ -666,6 +686,9 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
             return self.npu_communicator.all_reduce(input_)
 
         if torch.compiler.is_compiling():
+            if self._can_use_flashinfer_allreduce(input_):
+                return flashinfer_allreduce(input_, group_name=self.unique_name)
+
             # Byte-size thresholds in method selection (e.g. `_pick_algo` or
             # `should_mscclpp_allreduce`) would guard on the symbolic token dim
             # and recompile per shape; defer the selection to runtime inside
@@ -717,6 +740,9 @@ def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
                 self.pynccl_comm.all_reduce(input_)
                 return input_
 
+        if self._can_use_flashinfer_allreduce(input_):
+            return flashinfer_allreduce(input_, group_name=self.unique_name)
+
         outplace_all_reduce_method = self._resolve_outplace_all_reduce_method(
             input_=input_,
             should_use_pymscclpp_allreduce=should_use_pymscclpp_allreduce,
@@ -913,6 +939,29 @@ def _resolve_outplace_all_reduce_method(
             return "pynccl"
         return None
 
+    def _can_use_flashinfer_allreduce(self, input_: torch.Tensor) -> bool:
+        if self._fi_workspace_hint is None:
+            return False
+        from sglang.srt.layers.flashinfer_comm_fusion import (
+            can_use_flashinfer_allreduce,
+        )
+
+        return can_use_flashinfer_allreduce(
+            input_,
+            use_attn_tp_group=(self._fi_workspace_hint == "attn_tp"),
+            expected_world_size=self.world_size,
+            expected_group_key=(self.device_group, self.cpu_group),
+        )
+
+    def _flashinfer_allreduce(self, input_: torch.Tensor) -> torch.Tensor:
+        from sglang.srt.layers.flashinfer_comm_fusion import (
+            flashinfer_allreduce as _flashinfer_allreduce_impl,
+        )
+
+        return _flashinfer_allreduce_impl(
+            input_, use_attn_tp_group=(self._fi_workspace_hint == "attn_tp")
+        )
+
     def _all_reduce_out_place(
         self, input_: torch.Tensor, outplace_all_reduce_method: str
     ) -> torch.Tensor:
@@ -1999,6 +2048,7 @@ def graph_capture(stream=None):
 # Read once at import: whether CustomAllReduceV2 is opted in on a multi-node
 # (MNNVL) group. Used on the all_reduce hot path (see GroupCoordinator).
 _CA_V2_MULTINODE = envs.SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE.get()
+_ENABLE_FLASHINFER_ALLREDUCE_ONLY = False
 
 
 def set_custom_all_reduce(enable: bool):
@@ -2016,6 +2066,41 @@ def set_torch_symm_mem_all_reduce(enable: bool):
     _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = enable
 
 
+def set_flashinfer_allreduce_only(enable: bool):
+    global _ENABLE_FLASHINFER_ALLREDUCE_ONLY
+    _ENABLE_FLASHINFER_ALLREDUCE_ONLY = enable
+

# ... truncated 34 lines; see upstream PR ...
```

#### flashinfer_comm_fusion.py

```diff
@@ -860,6 +860,105 @@ def flashinfer_allreduce_residual_rmsnorm(
     return norm_out, residual_out
 
 
+def can_use_flashinfer_allreduce(
+    input_: torch.Tensor,
+    *,
+    use_attn_tp_group: bool,
+    expected_world_size: int,
+    expected_group_key: Tuple[Optional[ProcessGroup], Optional[ProcessGroup]],
+) -> bool:
+    """Whether ``flashinfer_allreduce`` can service this all-reduce.
+
+    Split out from the kernel call so the decision happens in plain Python,
+    outside the custom op: the op is opaque to Dynamo and has to return a
+    tensor, so it cannot carry a data-dependent fallback of its own.
+
+    ``expected_world_size`` / ``expected_group_key`` describe the calling group;
+    the workspace is only usable when it was rendezvoused on exactly those peers.
+
+    Every check here is rank-invariant by construction, and must stay that way:
+    a rank that quietly falls back to NCCL while its peers enter the kernel
+    mismatches and hangs. The unavailable flag and workspace initialization are
+    cross-rank synced at init time (``_sync_allreduce_unavailable_across_tp``);
+    the rest are pure functions of the group identity and of tensor metadata,
+    which is identical on every rank of the group.
+    """
+    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
+        return False
+
+    if input_.ndim != 2 or not input_.is_contiguous():
+        return False
+
+    workspace_manager = _get_workspace_manager(use_attn_tp_group)
+    if not workspace_manager.initialized or workspace_manager.workspace is None:
+        return False
+
+    # The two workspaces are keyed by attention-TP vs MoE, but the MoE one
+    # rendezvouses on either the EP or the MoE-TP group depending on topology.
+    # Under hybrid EP+TP those groups have equal world size but pair different
+    # ranks, so a mismatch here reduces across the wrong peers and silently
+    # produces garbage rather than failing. Require an exact match.
+    if (
+        workspace_manager.world_size != expected_world_size
+        or workspace_manager.group != expected_group_key
+    ):
+        return False
+
+    # Size checks stay last: they read the token dim, which is symbolic under
+    # Dynamo, so statically-off configs must short-circuit before reaching them
+    # (same ordering rule as apply_flashinfer_allreduce_fusion).
+    token_num, hidden_dim = input_.shape
+    if torch.compiler.is_compiling():
+        # Don't call into the flashinfer workspace object while tracing. The
+        # workspace was allocated for (max_token_num, hidden_dim, dtype) and
+        # vetted by is_buffer_size_sufficient() at init; the requirement is
+        # monotone in token_num/hidden_dim, so staying within the allocation
+        # (including dtype) is a conservative stand-in here.
+        return (
+            workspace_manager.max_token_num is not None
+            and workspace_manager.hidden_dim is not None
+            and workspace_manager.dtype is not None
+            and token_num <= workspace_manager.max_token_num
+            and hidden_dim <= workspace_manager.hidden_dim
+            and workspace_manager.dtype == input_.dtype
+        )
+
+    return workspace_manager.is_buffer_size_sufficient(
+        token_num=token_num,
+        hidden_dim=hidden_dim,
+        dtype=input_.dtype,
+    )
+
+
+def flashinfer_allreduce(
+    input_: torch.Tensor,
+    *,
+    use_attn_tp_group: bool,
+) -> torch.Tensor:
+    """Allreduce-only FlashInfer kAllReduce.
+
+    Assumes ``can_use_flashinfer_allreduce`` returned True for this call; there
+    is no fallback here. Kernel errors are deliberately not caught -- swallowing
+    one would put this rank on NCCL while its peers stay in the kernel, which
+    mismatch-hangs instead of failing.
+    """
+    workspace_manager = _get_workspace_manager(use_attn_tp_group)
+
+    output = torch.empty_like(input_)
+    kwargs = dict(
+        input=input_,
+        workspace=workspace_manager.workspace,
+        pattern=_flashinfer_comm.AllReduceFusionPattern.kAllReduce,
+        launch_with_pdl=True,
+        fp32_acc=False,
+        output=output,
+    )
+    if _flashinfer_allreduce_supports_trigger_completion:
+        kwargs["trigger_completion_at_end"] = False
+    _flashinfer_comm.allreduce_fusion(**kwargs)

# ... truncated 6 lines; see upstream PR ...
```

#### communicator.py

```diff
@@ -816,6 +816,18 @@ def should_fuse_mlp_allreduce_with_next_layer(
         if is_enable_moe_cp_allgather():
             return False
 
+        # Fusing makes the next layer's residual+LN absorb the post-experts
+        # all-reduce, and that fused kernel reduces over a single group. Under
+        # hybrid EP+TP the post-experts reduction spans two disjoint groups
+        # (moe_expert_parallel_all_reduce over _MOE_EP, then
+        # moe_tensor_model_parallel_all_reduce over _MOE_TP), and
+        # should_skip_post_experts_all_reduce() skips *both* once fusion is
+        # published -- so the fused reduce would cover only half the peers and
+        # silently return under-reduced activations.
+        parallel = get_parallel()
+        if parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1:
+            return False
+
         if (
             is_dp_attention_enabled()
             and self._speculative_algo is not None
```

### 本地核对

- [ ] 目标树已包含 #30700 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #33236 — [Perf][DSV4] Remove prefill CP KV and compressor materialization

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/33236
- **分类**：Prefill CP 存储
- **Author**：@foraxe  ·  diffstat `+1533/-44`
- **一句话**：去掉 prefill CP 每层 KV/compressor 完整物化，改 direct KV + semantic compress

### 做什么

Prefill CP 旧路径每层 gather 物化完整 KV 再 compress，费显存/通信。改为 direct CP KV store + semantic compress（cp_compress），避免完整 materialization。

### 开关 / 启用方式

```text
# 见 PR environ.py patch（名称以抓取时为准）
# 异地优先扫这些路径是否存在：
kernels/.../direct_cp_kv_store.cuh|.py
kernels/ops/attention/dsv4/cp_compress.py
compressor_v2.py 中的 direct/semantic 分支
```

### 关键文件 / 符号（异地核对点）

`direct_cp_kv_store`, `cp_compress`, compressor_v2 直写路径, memory_pool CP 视图

- `kernels/.../direct_cp_kv_store.cuh/.py`
- `kernels/ops/attention/dsv4/cp_compress.py`
- `compressor_v2.py`, `deepseek_v4_memory_pool.py`, `deepseek_v4.py`

### 关键 diff

#### environ.py

```diff
@@ -1117,6 +1117,13 @@ class Envs:
     SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
     SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
     SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
+    # Let each DSV4 CP rank publish its local BF16 KV rows directly into the
+    # replicated packed FlashMLA cache through a multicast mapping. This removes
+    # the BF16 KV all-gather and rerange materialization.
+    SGLANG_OPT_DSV4_CP_DIRECT_KV_STORE = EnvBool(False)
+    # CP4 prefill: publish per-window compressor states through symmetric
+    # memory instead of all-gathering token-level FP32 projections.
+    SGLANG_OPT_USE_CP_COMPRESS = EnvBool(False)
     SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
     SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
     # Deprecated: DSV4 compressor V2 is always used.
```

#### direct_cp_kv_store.py

```diff
@@ -0,0 +1,60 @@
+"""Producer-direct CP stores into replicated packed DSV4 KV caches."""
+
+from __future__ import annotations
+
+from typing import Any
+
+import torch
+import triton
+import triton.language as tl
+
+from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
+from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
+    _blockwise_barrier,
+)
+
+
+@triton.jit
+def _direct_store_barrier(signal_ptrs, RANK: tl.constexpr, WORLD: tl.constexpr):
+    _blockwise_barrier(signal_ptrs, RANK, WORLD, sem="acq_rel")
+
+
+@cache_once
+def _jit_direct_store(dtype: torch.dtype, index_dtype: torch.dtype, page_size: int):
+    args = make_cpp_args(dtype, index_dtype, page_size)
+    return load_jit(
+        "direct_cp_kv_store",
+        *args,
+        cuda_files=["deepseek_v4/direct_cp_kv_store.cuh"],
+        cuda_wrappers=[("run", f"DirectCPKVStoreKernel<{args}>::run")],
+    )
+
+
+def direct_cp_kv_store(
+    *,
+    cache: torch.Tensor,
+    handle: Any,
+    cache_multicast: int,
+    local_kv: torch.Tensor,
+    local_indices: torch.Tensor,
+    rank: int,
+    world_size: int,
+    page_size: int,
+) -> None:
+    if local_kv.shape != (local_indices.numel(), 512):
+        raise ValueError(
+            f"local KV shape {tuple(local_kv.shape)} does not match "
+            f"indices {local_indices.numel()} x 512"
+        )
+    _jit_direct_store(local_kv.dtype, local_indices.dtype, page_size).run(
+        local_kv,
+        cache,
+        int(cache_multicast),
+        local_indices,
+    )
+    _direct_store_barrier[(1,)](
+        handle.signal_pad_ptrs_dev,
+        RANK=rank,
+        WORLD=world_size,
+        num_warps=1,
+    )
```

#### compressor_v2.py（节选）

```diff
@@ -1,17 +1,28 @@
 from __future__ import annotations
 
+import logging
 from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union, cast
 
 import torch
-
 from sglang.kernels.jit.utils import is_hip_runtime
 from sglang.kernels.ops.attention.dsv4 import (
     CompressorDecodePlan,
     CompressorPrefillPlan,
     compress_forward,
     compress_norm_rope_store,
 )
+from sglang.kernels.ops.attention.dsv4.cp_compress import (
+    CPCompressorState,
+    cp_compress_aligned,
+    create_cp_compressor_state,
+)
+from sglang.srt.distributed.parallel_state import get_attn_cp_group
 from sglang.srt.environ import envs
+from sglang.srt.layers.attention.dsa.utils import (
+    dsa_use_prefill_cp,
+    is_dsa_prefill_cp_round_robin_split,
+)
+from sglang.srt.runtime_context import get_parallel, get_server_args
 
 if TYPE_CHECKING:
     from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata
@@ -26,6 +37,7 @@
 FusedCompressMetadata: TypeAlias = CompressMetadata
 
 _is_hip = is_hip_runtime()
+logger = logging.getLogger(__name__)
 
 
 def _use_online_compress(compress_ratio: int) -> bool:
@@ -130,6 +142,104 @@ class CompressorBackendMixin:
     def __init__(self):
         super().__init__()
         self.forward_metadata: DSV4Metadata
+        # Reused sequentially by all model layers; carries remain layer-local.
+        self._cp_compressor_states: dict[tuple[int, int], CPCompressorState] = {}
+        self._reported_cp_compressor_keys: set[tuple[int, int]] = set()
+
+    def initialize_cp_compressor_states(
+        self, device: torch.device, max_tokens: int
+    ) -> None:
+        """Collectively create stable peer objects before any model forward."""
+        if not envs.SGLANG_OPT_USE_CP_COMPRESS.get():
+            return
+        cp_group = get_attn_cp_group()
+        if cp_group.world_size != 4 or max_tokens <= 0:
+            raise ValueError("CP compressor requires CP4 and bounded chunked prefill")
+        for ratio, head_dim in ((4, 128), (4, 512), (128, 512)):
+            self._cp_compressor_states[(ratio, head_dim)] = create_cp_compressor_state(
+                cp_group.device_group,
+                cp_group.rank_in_group,
+                ratio,
+                head_dim,
+                max_tokens,
+                device,
+            )
+
+    def _cp_compress_context(
+        self,
+        forward_batch: ForwardBatch,
+        compressor: Compressor,
+        local_tokens: int,
+    ) -> Optional[tuple[int, int]]:
+        if (
+            not envs.SGLANG_OPT_USE_CP_COMPRESS.get()
+            or _is_hip
+            or not dsa_use_prefill_cp(forward_batch)
+            or not is_dsa_prefill_cp_round_robin_split()
+            or get_parallel().attn_cp_size != 4
+            or torch.cuda.is_current_stream_capturing()
+            or not forward_batch.forward_mode.is_extend_without_speculative()
+        ):
+            return None
+        extend_lens = forward_batch.extend_seq_lens_cpu
+        seq_lens = forward_batch.seq_lens_cpu
+        if extend_lens is None or seq_lens is None or len(extend_lens) != 1:
+            return None
+        global_tokens = int(extend_lens[0])
+        total_tokens = int(seq_lens[0].item())
+        prefix_tokens = total_tokens - global_tokens
+        max_tokens = get_server_args().chunked_prefill_size
+        if (
+            max_tokens is None
+            or max_tokens <= 0
+            or global_tokens != local_tokens * 4
+            or not 0 < global_tokens <= max_tokens
+            or global_tokens % compressor.ratio
+            or prefix_tokens < 0
+            or prefix_tokens % compressor.ratio
+        ):
+            return None
+        plan = self._get_paged_compress_metadata(compressor.ratio)
+        if (
+            plan.is_decode
+            or plan.compress_ratio != compressor.ratio
+            or plan.plan_c.dtype != torch.uint8
+            or not plan.plan_c.is_contiguous()
+            or plan.plan_c.shape != (global_tokens // compressor.ratio, 16)
+        ):
+            return None
+        if compressor.ratio == 4:
+            state_buffer = compressor.get_state_pool(self).kv_score_buffer.kv_score
+            if (
+                plan.plan_w.dtype != torch.uint8
+                or not plan.plan_w.is_contiguous()
+                or plan.plan_w.ndim != 2
+                or plan.plan_w.shape[1] != 8
+                or state_buffer.dtype != torch.float32
+                or not state_buffer.is_contiguous()
+                or state_buffer.ndim != 2
+                or state_buffer.shape[1] != 4 * compressor.head_dim
+            ):

# ... truncated 83 lines; see upstream PR ...
```

#### deepseek_v4.py（节选）

```diff
@@ -1062,15 +1062,23 @@ def _forward_prepare(
             kv = None
 
             if not unified and use_cp:
-                # DSA CP: keep bf16 kv around for the cross-rank all-gather, then
-                # write to the FlashMLA cache after gather.
+                # DSA CP: either publish each rank's local rows directly into all
+                # packed FlashMLA caches, or materialize the legacy full BF16 KV.
                 kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
-                kv = cp_all_gather_rerange_output(
-                    kv.contiguous(),
-                    self.cp_size,
-                    forward_batch,
-                    torch.cuda.current_stream(),
-                )
+                if attn_backend.can_store_cache_cp_direct():
+                    attn_backend.store_cache_cp_direct(
+                        layer_id=self.layer_id,
+                        local_swa_k=kv,
+                        forward_batch=forward_batch,
+                    )
+                    kv = None
+                else:
+                    kv = cp_all_gather_rerange_output(
+                        kv.contiguous(),
+                        self.cp_size,
+                        forward_batch,
+                        torch.cuda.current_stream(),
+                    )
         elif _is_npu:
             q_lora = self.q_norm(q_lora)
             q, _ = self.wq_b(q_lora)
@@ -1123,20 +1131,28 @@ def _forward_prepare(
                         torch.cuda.current_stream(),
                     )
             elif use_cp:
-                # NSA CP: keep bf16 kv around for the cross-rank all-gather, then
-                # write to the FlashMLA cache after gather.
+                # NSA CP: publish rank-local rows directly into the replicated
+                # packed cache when enabled; otherwise retain the BF16 gather.
                 kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
-                kv = cp_all_gather_rerange_output(
-                    kv.contiguous(),
-                    self.cp_size,
-                    forward_batch,
-                    torch.cuda.current_stream(),
-                )
-                attn_backend.store_cache(
-                    layer_id=self.layer_id,
-                    swa_k=kv,
-                    forward_batch=forward_batch,
-                )
+                if attn_backend.can_store_cache_cp_direct():
+                    attn_backend.store_cache_cp_direct(
+                        layer_id=self.layer_id,
+                        local_swa_k=kv,
+                        forward_batch=forward_batch,
+                    )
+                    kv = None
+                else:
+                    kv = cp_all_gather_rerange_output(
+                        kv.contiguous(),
+                        self.cp_size,
+                        forward_batch,
+                        torch.cuda.current_stream(),
+                    )
+                    attn_backend.store_cache(
+                        layer_id=self.layer_id,
+                        swa_k=kv,
+                        forward_batch=forward_batch,
+                    )
             else:
                 self._compute_kv_to_cache(
                     x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
```

#### memory_pool.py（节选）

```diff
@@ -21,7 +21,12 @@
 from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
 from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
 from sglang.srt.mem_cache.memory_pool import KVCache
-from sglang.srt.runtime_context import get_exec, get_server_args, get_spec
+from sglang.srt.runtime_context import (
+    get_exec,
+    get_parallel,
+    get_server_args,
+    get_spec,
+)
 from sglang.srt.utils import ceil_div, is_hip
 
 logger = logging.getLogger(__name__)
@@ -61,6 +66,7 @@ def __init__(
         enable_memory_saver: bool,
         start_layer: Optional[int] = None,
         end_layer: Optional[int] = None,
+        direct_cp_store: bool = False,
     ):
         super().__init__(
             size,
@@ -79,10 +85,70 @@ def __init__(
         self.quantize_block_size = 64
         self.rope_storage_dtype = torch.bfloat16
         self.k_with_scale_buffer_dtype = torch.int8
+        self.direct_cp_store = bool(direct_cp_store)
+        self.direct_cp_handle = None
+        self._direct_cp_root = None
         self._create_buffers()
 
     def _create_buffers(self):
         with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
+            if self.direct_cp_store:
+                if self.custom_mem_pool is not None:
+                    raise RuntimeError(
+                        "DSV4 direct CP KV store is incompatible with a custom KV "
+                        "memory pool"
+                    )
+                import torch.distributed._symmetric_memory as symm_mem
+
+                parallel = get_parallel()
+                group = parallel.attn_cp_group.device_group
+                world_size = parallel.attn_cp_size
+                if world_size != 4:
+                    raise RuntimeError(
+                        "DSV4 direct CP KV store currently requires attn CP size 4, "
+                        f"got {world_size}"
+                    )
+                num_pages = (self.size + self.page_size + 1) // self.page_size
+                bytes_per_token = self.get_bytes_per_token()
+                if bytes_per_token != 584 or self.store_dtype != torch.uint8:
+                    raise RuntimeError(
+                        "DSV4 direct CP KV store requires the 584-byte packed "
+                        "FlashMLA uint8 cache layout"
+                    )
+                self.kv_cache_total_dim = bytes_per_token
+                self.bytes_per_page_padded = (
+                    ceil_div(self.page_size * bytes_per_token, 576) * 576
+                )
+                symm_mem.set_signal_pad_size(
+                    max(symm_mem.get_signal_pad_size(), world_size * 4)
+                )
+                with torch.inference_mode(False), torch.no_grad():
+                    root = symm_mem.empty(
+                        (self.layer_num, num_pages, self.bytes_per_page_padded),
+                        dtype=self.store_dtype,
+                        device=self.device,
+                    )
+                root.zero_()
+                handle = symm_mem.rendezvous(root, group=group)
+                if handle.multicast_ptr == 0:
+                    raise RuntimeError(
+                        "DSV4 direct CP KV store requires an NVLink multicast mapping"
+                    )
+                if handle.rank != parallel.attn_cp_rank:
+                    raise RuntimeError(
+                        f"symmetric-memory rank {handle.rank} != attn CP rank "
+                        f"{parallel.attn_cp_rank}"
+                    )
+                self._direct_cp_root = root
+                self.direct_cp_handle = handle
+                self.kv_buffer = list(root.unbind(0))
+                logger.info(
+                    "Using direct CP publication into the packed DSV4 SWA cache "
+                    "(%d layers, %.2f GiB)",
+                    self.layer_num,
+                    root.nbytes / (1024**3),
+                )
+                return
             with (
                 torch.cuda.use_mem_pool(self.custom_mem_pool)
                 if self.custom_mem_pool
@@ -570,6 +636,31 @@ def __init__(
         )
 
         self._unified_kv = is_unified_kv_triton()
+        direct_cp_store = envs.SGLANG_OPT_DSV4_CP_DIRECT_KV_STORE.get()
+        if direct_cp_store:
+            server_args = get_server_args()

# ... truncated 90 lines; see upstream PR ...
```

### 本地核对

- [ ] 目标树已包含 #33236 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #28639 — [Perf][DSV4] add ag_gemm and moe_rs overlap kernels for dsv4 prefill

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/28639
- **分类**：通信重叠
- **Author**：@Zqy11  ·  diffstat `+1367/-37`
- **一句话**：symmetric-memory 上 AG⊗GEMM / MoE⊗RS overlap

### 做什么

Prefill CP 下 AllGather↔GEMM、MoE↔ReduceScatter 用 torch symmetric memory fused kernel 重叠，降 CP 扩展通信开销。

### 开关 / 启用方式

```text
SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL=1   # PR 新增，默认 False
```

### 关键文件 / 符号（异地核对点）

`allgather_gemm_op_symm_mem`, `moe_reduce_rs_symm_mem`, `_cp_fused_symm_mem_enabled()`

- `distributed/device_communicators/symm_mem_kernels/allgather_gemm_symm_mem.py`
- `.../moe_reduce_rs_symm_mem.py`
- `torch_symm_mem.py`, `deepseek_v4.py`, `deepseek_v2.py`

### 关键 diff

#### environ.py

```diff
@@ -845,6 +845,8 @@ class Envs:
     SGLANG_OPT_USE_JIT_EP_ACTIVATION = EnvBool(True)
     SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
     SGLANG_OPT_SWIGLU_CLAMP_FUSION = EnvBool(True)
+    # Fused AG-GEMM and MoE reduce-scatter via torch symmetric memory.
+    SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL = EnvBool(False)
 
     # Cache / overlap
     SGLANG_OPT_USE_FUSED_STORE_CACHE = EnvBool(True)
```

#### symm_mem_kernels/__init__.py

```diff
@@ -0,0 +1,25 @@
+# SPDX-License-Identifier: Apache-2.0
+
+from sglang.srt.distributed.device_communicators.symm_mem_kernels.allgather_gemm_symm_mem import (
+    AllGatherGemmContextSymmMem,
+    allgather_gemm_op_symm_mem,
+    create_allgather_gemm_context_symm_mem,
+    maybe_fused_ag_shared_experts,
+)
+from sglang.srt.distributed.device_communicators.symm_mem_kernels.moe_reduce_rs_symm_mem import (
+    MoEReduceRSSymmMemContext,
+    create_moe_rs_symm_mem_context,
+    maybe_fused_shared_add_rs,
+    moe_reduce_rs_symm_mem,
+)
+
+__all__ = [
+    "AllGatherGemmContextSymmMem",
+    "MoEReduceRSSymmMemContext",
+    "allgather_gemm_op_symm_mem",
+    "create_allgather_gemm_context_symm_mem",
+    "create_moe_rs_symm_mem_context",
+    "maybe_fused_ag_shared_experts",
+    "maybe_fused_shared_add_rs",
+    "moe_reduce_rs_symm_mem",
+]
```

#### torch_symm_mem.py（节选）

```diff
@@ -11,6 +11,7 @@
 from sglang.srt.distributed.device_communicators.all_reduce_utils import (
     TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
 )
+from sglang.srt.server_args import get_global_server_args
 from sglang.srt.utils import is_cuda, is_hip
 
 try:
@@ -58,9 +59,20 @@ def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
             device: Target CUDA device (index, 'cuda:X', or torch.device).
         """
 
+        # disabled: entire communicator unusable.
+        # allreduce_disabled: only the allreduce fast path is off; buffers
+        # may still serve fused-kernel contexts (RS/AG).
         self.disabled = True
+        self.allreduce_disabled = True
+        self.use_cp = False  # set True during CP-mode forward
         self.buffer = None
         self.max_size = 0
+        # Lazy fused-kernel contexts (cached on first use).
+        self._moe_rs_ctx = None
+        self._moe_rs_key = None
+        self._ag_gemm_ctx = None
+        self._ag_gemm_key = None
+        self._ag_gemm_stream: Optional[torch.cuda.Stream] = None
 
         if not torch_symm_mem_available:
             return
@@ -99,15 +111,32 @@ def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
             dtype=self.dtype,
         )
         handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
+        # Enable communicator; allreduce gated separately by multicast.
+        self.disabled = False
         if handle.multicast_ptr == 0:
             logger.warning(
                 "TorchSymmMemCommunicator: torch symmetric memory "
-                "multicast operations are not supported."
+                "multicast operations are not supported; symm-mem all-reduce "
+                "fast path disabled (fused-kernel contexts may still work)."
             )
-            self.buffer = None
-            self.disabled = True
+            self.allreduce_disabled = True
             return
-        self.disabled = False
+        self.allreduce_disabled = False
+
+    def set_use_cp(self, value: bool) -> None:
+        """Set the CP mode flag for fused kernels."""
+        self.use_cp = value
+
+    @staticmethod
+    def get_active_comm() -> "Optional[TorchSymmMemCommunicator]":
+        """Return the TP group's communicator if enabled and use_cp is active, else None."""
+        from sglang.srt.distributed import get_tp_group
+
+        # TODO(zxdu): maybe use cp group here?
+        comm = get_tp_group().torch_symm_mem_comm
+        if comm is None or comm.disabled or not comm.use_cp:
+            return None
+        return comm
 
     def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
         """
@@ -122,7 +151,7 @@ def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
         Returns:
             True if the symmetric-memory path can handle this tensor.
         """
-        if self.disabled:
+        if self.disabled or self.allreduce_disabled:
             return False
         if inp.device != self.device:
             return False
@@ -169,3 +198,106 @@ def all_reduce(
             )
         out.copy_(self.buffer[: inp.numel()].view(out.shape))
         return out
+
+    def _get_max_forward_tokens(self) -> int:
+        """Return max tokens per forward (chunked_prefill_size or max_prefill_tokens)."""
+        server_args = get_global_server_args()
+        cps = server_args.chunked_prefill_size
+        if cps is not None and cps > 0:
+            return cps
+        return server_args.max_prefill_tokens
+
+    def get_or_create_moe_rs_ctx(
+        self,
+        N: int,
+        num_experts: int,
+        topk: int,
+        dtype: torch.dtype,
+        n_chunks_max: int = 8,
+    ):
+        """Lazy-init / cache the MoE reduce-scatter symm-mem context."""
+        if self.disabled:
+            return None
+        key = (N, num_experts, topk, dtype, n_chunks_max, self.world_size)
+        if self._moe_rs_key == key and self._moe_rs_ctx is not None:
+            return self._moe_rs_ctx
+        if self._moe_rs_ctx is not None:
+            try:
+                self._moe_rs_ctx.finalize()
+            except Exception as e:  # pragma: no cover - defensive
+                logger.warning(
+                    "TorchSymmMemCommunicator: failed to finalize stale MoE RS "
+                    "context: %s",
+                    e,
+                )
+            self._moe_rs_ctx = None
+            self._moe_rs_key = None
+
+        from sglang.srt.distributed.device_communicators.symm_mem_kernels import (
+            create_moe_rs_symm_mem_context,
+        )
+
+        max_M = self._get_max_forward_tokens()
+

# ... truncated 62 lines; see upstream PR ...
```

#### deepseek_v4.py 接入

```diff
@@ -208,6 +208,9 @@ def _freqs_cis_to_cos_sin(
     _FREQS_CIS_TO_COS_SIN[key] = (cos, sin)
     return cos, sin
 
+def _cp_fused_symm_mem_enabled() -> bool:
+    """True when CP AG/RS should be handled by torch_symm_mem fused kernels."""
+    return envs.SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL.get() and not get_is_capture_mode()
 
 if TYPE_CHECKING:
     from sglang.srt.layers.attention.deepseek_v4_backend import (
@@ -1611,7 +1614,8 @@ def forward(
         )
         if _use_cp:
             if get_moe_a2a_backend().is_none():
-                hidden_states = dsa_cp_gather_hidden_states(hidden_states)
+                if not _cp_fused_symm_mem_enabled() or not self.mlp.experts.moe_runner_config.inplace:
+                    hidden_states = dsa_cp_gather_hidden_states(hidden_states)
             else:
                 assert get_moe_a2a_backend().is_deepep(), (
                     "CP requires DeepEP (moe_a2a_backend == deepep). "
@@ -1643,7 +1647,8 @@ def forward(
             skip_shared_experts=_do_shared_local,
         )
         if _use_cp and get_moe_a2a_backend().is_none():
-            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
+            if not _cp_fused_symm_mem_enabled():
+                hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
         elif _use_tp_moe_gather:
             hidden_states, global_hidden_states = (
                 get_local_dp_buffer(get_tp_group()),
@@ -1810,6 +1815,9 @@ def forward(
             input_ids_global = input_ids
 
         if dsa_use_prefill_cp(forward_batch):
+            _comm = get_tp_group().torch_symm_mem_comm
+            if _comm is not None and _cp_fused_symm_mem_enabled():
+                _comm.set_use_cp(True)
             if self.pp_group.is_first_rank:
                 hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
             positions = cp_split_and_rebuild_position(forward_batch, positions)
@@ -1848,6 +1856,9 @@ def forward(
                 hidden_states, prev_residual, prev_post, prev_comb
             )
 
+        _comm = get_tp_group().torch_symm_mem_comm
+        if _comm is not None:
+            _comm.set_use_cp(False)
         # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
         if self.pp_group.is_last_rank and dsa_use_prefill_cp(forward_batch):
             hidden_states = cp_all_gather_rerange_output(
```

### 本地核对

- [ ] 目标树已包含 #28639 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #32059 — [Feat][DeepSeek V4] Shared KV Cache for Prefill CP

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/32059
- **分类**：Prefill CP 存储
- **Author**：@taoyuanyuan  ·  diffstat `+7748/-121`
- **一句话**：VMM Shared KV：每 CP rank 持有物理页分片

### 做什么

Prefill CP 通过 CUDA VMM 做 Shared KV：每个 CP rank 拥有物理页分片，映射成共享虚拟视图，避免每 rank 复制完整 KV。Flash 已验，Pro 待验。

### 开关 / 启用方式

```bash
--enable-dsa-shared-kv-cache
# hook: validate_deepseek_v4_shared_target / validate_deepseek_v4_shared_release
```

新包：`python/sglang/srt/mem_cache/shared_kv/`（`vmm.py`, `family.py`, `layout.py`, ...）  
另有：`deepseek_v4_shared.py`, `shared_cache_access.py`, PD `shared_kv_staging.py`。

### 关键文件 / 符号（异地核对点）

`shared_kv.vmm`, `DeepSeekV4Shared*`, `shared_cache_access`, PD `shared_kv_staging`

见 PR 文件列表（~46 files）；核心 `mem_cache/shared_kv/vmm.py`, `deepseek_v4_shared.py`

### 关键 diff

#### server_args.py

```diff
@@ -1056,6 +1056,11 @@ class ServerArgs:
         "Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community.",
         NS("parallel"),
     ] = False
+    enable_dsa_shared_kv_cache: A[
+        bool,
+        "Share DeepSeek V4 GPU KV/indexer cache pages across context-parallel ranks.",
+        NS("parallel"),
+    ] = False
     enable_dsa_prefill_context_parallel: A[bool, Arg(no_cli=True), NS("parallel")] = (
         False
     )
@@ -4688,6 +4693,12 @@ def _handle_model_specific_adjustments(self):
         hf_config = self.get_model_config().hf_config
         model_arch = hf_config.architectures[0]
 
+        if self.enable_dsa_shared_kv_cache:
+            from sglang.srt.arg_groups.deepseek_v4_hook import (
+                validate_deepseek_v4_shared_target,
+            )
+
+            validate_deepseek_v4_shared_target(self, hf_config, model_arch)
         if self.enable_dsa_cache_layer_split and not is_deepseek_dsa(hf_config):
             raise ValueError(
                 "--enable-dsa-cache-layer-split is only supported for DSA "
@@ -4719,9 +4730,12 @@ def _handle_model_specific_adjustments(self):
         ]:
             from sglang.srt.arg_groups.deepseek_v4_hook import (
                 apply_deepseek_v4_defaults,
+                validate_deepseek_v4_shared_release,
             )
 
             apply_deepseek_v4_defaults(self, model_arch)
+            if self.enable_dsa_shared_kv_cache:
+                validate_deepseek_v4_shared_release(self, hf_config)
 
         if model_arch in [
             "DeepseekV3ForCausalLM",
```

#### deepseek_v4_hook.py（节选）

```diff
@@ -4,13 +4,147 @@
 from typing import TYPE_CHECKING
 
 from sglang.srt.environ import envs
+from sglang.srt.runtime_context import get_parallel
 
 if TYPE_CHECKING:
+    from sglang.srt.model_executor.model_runner import ModelRunner
     from sglang.srt.server_args import ServerArgs
 
 logger = logging.getLogger(__name__)
 
 
+def is_dsv4_cache_shared_enabled(model_runner: ModelRunner) -> bool:
+    """Whether DeepSeek V4 persistent cache pages are shared across CP ranks."""
+    from sglang.srt.configs.model_config import is_deepseek_v4
+
+    return (
+        not model_runner.is_draft_worker
+        and model_runner.server_args.enable_dsa_shared_kv_cache
+        and is_deepseek_v4(model_runner.model_config.hf_config)
+    )
+
+
+def get_dsv4_shared_info(model_runner: ModelRunner) -> tuple[int | None, int]:
+    if not is_dsv4_cache_shared_enabled(model_runner):
+        return None, 1
+    shared_size = get_parallel().attn_cp_size
+    if shared_size <= 1:
+        return None, 1
+    return get_parallel().attn_cp_rank, shared_size
+
+
+def validate_deepseek_v4_shared_target(
+    server_args: ServerArgs, hf_config, model_arch: str
+) -> None:
+    """Reject model targets outside the DSV4 Shared release."""
+
+    from sglang.srt.configs.model_config import is_deepseek_v4
+    from sglang.srt.utils import is_cuda
+
+    if server_args.enable_dsa_cache_layer_split:
+        raise ValueError(
+            "--enable-dsa-shared-kv-cache and "
+            "--enable-dsa-cache-layer-split cannot be enabled together."
+        )
+    if not is_cuda():
+        raise ValueError("--enable-dsa-shared-kv-cache requires NVIDIA CUDA.")
+    if not is_deepseek_v4(hf_config):
+        raise ValueError(
+            "--enable-dsa-shared-kv-cache is currently supported only "
+            "for DeepSeek V4."
+        )
+    if model_arch != "DeepseekV4ForCausalLM":
+        raise ValueError(
+            "DeepSeek V4 Shared KV release supports only the canonical "
+            "DeepseekV4ForCausalLM target architecture, not NextN or DSpark."
+        )
+
+
+def validate_deepseek_v4_shared_release(server_args: ServerArgs, hf_config) -> None:
+    """Reject configurations outside the validated Flash Prefill L1 release."""
+
+    ratios = list(getattr(hf_config, "compress_ratios", ()))
+    layer_count = int(getattr(hf_config, "num_hidden_layers", len(ratios)))
+    expected_ratios = [0, 0] + [4, 128] * 20 + [4, 0]
+    if layer_count != 43 or ratios != expected_ratios:
+        profile = (
+            layer_count,
+            len(ratios),
+            sum(ratio == 0 for ratio in ratios),
+            sum(ratio == 4 for ratio in ratios),
+            sum(ratio == 128 for ratio in ratios),
+        )
+        raise ValueError(
+            "DeepSeek V4 Shared KV release currently supports only the full "
+            "43-layer Flash profile (44 config entries; C1/C4/C128 = 3/21/20); "
+            f"got layers/ratios/C1/C4/C128 = {profile}."
+        )
+    if not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
+        raise ValueError("DeepSeek V4 Shared KV release requires Compressor V2.")
+    if envs.SGLANG_OPT_USE_OLD_COMPRESSOR.get():
+        raise ValueError(
+            "DeepSeek V4 Shared KV release does not support the old compressor."
+        )
+    if server_args.disaggregation_mode not in ("null", "prefill"):
+        raise ValueError(
+            "DeepSeek V4 --enable-dsa-shared-kv-cache is supported on the "
+            "Prefill worker only; Decode must use the ordinary DSV4 cache."
+        )
+    if (
+        server_args.disaggregation_mode == "prefill"
+        and server_args.disaggregation_transfer_backend != "mooncake"
+    ):
+        raise ValueError(
+            "DeepSeek V4 Shared KV Prefill PD currently requires the Mooncake "
+            "transfer backend."
+        )
+    # DSV4 resolves attn_cp_size later in the adjustment pass. Validate the
+    # canonical inputs that deterministically produce CP8 instead.

# ... truncated 83 lines; see upstream PR ...
```

#### shared_kv/__init__.py

```diff
@@ -0,0 +1,26 @@
+from sglang.srt.mem_cache.shared_kv.family import (
+    OwnerShardedFamily,
+    OwnerShardedFamilySpec,
+    SharedFamilyAccounting,
+)
+from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
+from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher
+from sglang.srt.mem_cache.shared_kv.transfer import OwnerShardedTransferBuffer
+from sglang.srt.mem_cache.shared_kv.vmm import (
+    RankMajorSharedSlab,
+    RankMajorSharedTensor,
+    create_rank_major_shared_slab,
+    create_rank_major_shared_tensor,
+)
+
+__all__ = [
+    "OwnerShardedFamily",
+    "OwnerShardedFamilySpec",
+    "OwnerShardedLayout",
+    "RankMajorSharedSlab",
+    "RankMajorSharedTensor",
+    "SharedFamilyAccounting",
+    "SharedWritePublisher",
+    "create_rank_major_shared_slab",
+    "create_rank_major_shared_tensor",
+]
```

#### shared_cache_access.py

```diff
@@ -0,0 +1,141 @@
+"""DeepSeek V4 view of the model-neutral owner-sharded cache layout."""
+
+from dataclasses import dataclass
+from typing import Any, Optional
+
+import torch
+
+from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
+
+
+class DSV4SharedCacheAccess:
+    """The sole attention-side entry point for DSV4 Shared cache behavior."""
+
+    def __init__(self, pool: Any) -> None:
+        self._pool = pool
+
+    def publish_writes(self) -> None:
+        self._pool.synchronize_shared_writes()
+
+    def plan_flashmla_kv_read(
+        self, pages: torch.Tensor, *, single_request: bool = False
+    ) -> tuple[dict[str | int, torch.Tensor], torch.Tensor]:
+        return self._pool.prepare_compressed_pages_for_read(
+            pages, single_request=single_request
+        )
+
+    def stage_sparse_pages(
+        self, layer_id: int, physical_pages: torch.Tensor
+    ) -> torch.Tensor:
+        return self._pool.stage_compressed_pages_with_indexer_plan(
+            layer_id, physical_pages
+        )
+
+    def prepare_indexer_pages(
+        self, pages: torch.Tensor
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        return self._pool.prepare_indexer_pages_for_read(pages)
+
+    def stage_indexer_pages(
+        self, layer_id: int, physical_pages: torch.Tensor
+    ) -> torch.Tensor:
+        return self._pool.stage_indexer_pages_with_plan(layer_id, physical_pages)
+
+    def prepare_swa_pages(
+        self, slots: torch.Tensor, *, single_request: bool = False
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        return self._pool.prepare_swa_slots_for_read(
+            slots, single_request=single_request
+        )
+
+    def stage_swa_pages(
+        self, layer_id: int, physical_pages: torch.Tensor
+    ) -> torch.Tensor:
+        return self._pool.stage_swa_slots_with_plan(layer_id, physical_pages)
+
+    def prepare_extra_pages(
+        self,
+        layer_id: int,
+        slots: torch.Tensor,
+        *,
+        single_request: bool = False,
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        if single_request:
+            return self._pool.prepare_extra_slots_for_read(
+                layer_id, slots, single_request=True
+            )
+        return self._pool.prepare_extra_slots_for_read(layer_id, slots)
+
+    def stage_extra_pages(
+        self, layer_id: int, physical_pages: torch.Tensor
+    ) -> torch.Tensor:
+        return self._pool.stage_extra_slots_with_plan(layer_id, physical_pages)
+
+    def translate_slots(
+        self, family: str, slots: torch.Tensor, *, layer_id: int
+    ) -> torch.Tensor:
+        if family == "swa":
+            return self._pool.translate_swa_slots_for_read(slots)
+        if family == "extra":
+            return self._pool.translate_extra_slots_for_read(layer_id, slots)
+        raise ValueError(f"unknown DSV4 Shared slot family: {family}")
+
+    def shared_dequant_params(self, family: str, *, layer_id: int) -> tuple[int, int]:
+        if family == "swa":
+            return self._pool.get_swa_shared_dequant_params(layer_id)
+        if family == "extra":
+            return self._pool.get_extra_shared_dequant_params(layer_id)
+        raise ValueError(f"unknown DSV4 Shared dequant family: {family}")
+
+    def kv_owner_write_target(
+        self, layer_id: int, *, is_indexer: bool
+    ) -> tuple[torch.Tensor, int, int]:
+        return self._pool.get_compressor_write_info(layer_id, is_indexer=is_indexer)
+
+    @staticmethod
+    def compressor_state_layout(state_pool: Any) -> tuple[int, int, int]:
+        return state_pool.get_shared_state_layout()
+
+

# ... truncated 42 lines; see upstream PR ...
```

#### deepseek_v4_shared.py（大幅，节选）

```diff
@@ -0,0 +1,1282 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+# ==============================================================================
+
+"""Page-sharded persistent cache storage for DeepSeek V4 prefill CP."""
+
+from __future__ import annotations
+
+import logging
+from dataclasses import dataclass
+
+import torch
+import torch.distributed as dist
+import triton
+import triton.language as tl
+from torch.distributed import ProcessGroup
+
+from sglang.jit_kernel.dsv4 import (
+    fused_k_norm_rope_flashmla,
+    fused_store_cache_shared,
+)
+from sglang.srt.layers.attention.dsv4.shared_cache_access import (
+    DSV4SharedCacheAccess,
+    DSV4SharedPageLayout,
+)
+from sglang.srt.mem_cache.deepseek_v4_compress_state import (
+    CompressStatePool,
+    KVAndScore,
+    get_compress_state_layout,
+)
+from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
+    ONLINE_C128,
+    DeepSeekV4IndexerPool,
+    DeepSeekV4SingleKVPool,
+    DeepSeekV4TokenToKVPool,
+)
+from sglang.srt.mem_cache.shared_kv.family import (
+    OwnerShardedFamily,
+    OwnerShardedFamilySpec,
+)
+from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
+from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher
+from sglang.srt.mem_cache.shared_kv.transfer import OwnerShardedTransferBuffer
+from sglang.srt.utils.common import ceil_div
+
+logger = logging.getLogger(__name__)
+
+
+def build_dsv4_shared_page_layout(
+    *, logical_size: int, page_size: int, cp_size: int
+) -> DSV4SharedPageLayout:
+    """Build the owner-page layout for one DeepSeek V4 cache family."""
+    if logical_size < 0:
+        raise ValueError(f"logical_size must be non-negative, got {logical_size}")
+    if page_size <= 0:
+        raise ValueError(f"page_size must be positive, got {page_size}")
+    if cp_size <= 1:
+        raise ValueError(f"shared cache requires cp_size > 1, got {cp_size}")
+
+    # Preserve a dummy slot/page and one spare owner page, matching the paged
+    # allocators' valid logical range.
+    logical_pages = ceil_div(logical_size + 1, page_size)
+    requested_pages = ceil_div(logical_pages, cp_size) + 1
+    return DSV4SharedPageLayout(
+        OwnerShardedLayout(
+            cp_size=cp_size,
+            ownership_granule=page_size,
+            logical_rows=requested_pages * cp_size * page_size,
+        )
+    )
+
+
+@triton.jit
+def _translate_shared_slots_kernel(
+    logical_slots,
+    physical_slots,
+    numel,
+    PAGE_SIZE: tl.constexpr,
+    CP_SIZE: tl.constexpr,
+    PAGES_PER_RANK: tl.constexpr,
+    PADDING_VALUE: tl.constexpr,
+    BLOCK_SIZE: tl.constexpr,
+):
+    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+    mask = offsets < numel
+    logical_slot = tl.load(logical_slots + offsets, mask=mask)
+    valid = logical_slot != PADDING_VALUE
+    safe_slot = tl.where(valid, logical_slot, 0)
+    logical_page = safe_slot // PAGE_SIZE

# ... truncated 103 lines; see upstream PR ...
```

### 本地核对

- [ ] 目标树已包含 #32059 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #33382 — feat: 1/N DeepSeek V4 CP Cache LayerSplit: Common Infrastructure

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/33382
- **分类**：Prefill CP LayerSplit
- **Author**：@jellysnack  ·  diffstat `+513/-210`
- **一句话**：CP Cache LayerSplit 公共 infra（1/N）

### 做什么

#29187 拆分第 1 弹：抽出 CP cache layer-split 公共目录/所有权/ broadcast / pool_base，供 DSA/DSV4 复用。

### 开关 / 启用方式

```bash
--enable-cp-cache-layer-split
# 兼容别名：--enable-dsa-cache-layer-split
# （本 PR 把 enable_dsa_cache_layer_split 重命名/别名到 enable_cp_cache_layer_split）
```

### 关键文件 / 符号（异地核对点）

`mem_cache/cp_cache_layer_split/{utils,pool_base,broadcast,dsa}.py`

`cp_cache_layer_split/*`, `server_args.py`, `pool_configurator.py`, unit tests

### 关键 diff

#### __init__.py

```diff
@@ -0,0 +1,11 @@
+"""CP Cache LayerSplit public helpers."""
+
+from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
+    CpCacheLayerSplitPoolBase,
+    is_cp_cache_layer_split_pool,
+)
+
+__all__ = [
+    "CpCacheLayerSplitPoolBase",
+    "is_cp_cache_layer_split_pool",
+]
```

#### utils.py

```diff
@@ -0,0 +1,78 @@
+"""Common configuration and ownership helpers for CP Cache LayerSplit."""
+
+from __future__ import annotations
+
+from typing import TYPE_CHECKING, Optional
+
+from sglang.srt.runtime_context import get_parallel
+
+if TYPE_CHECKING:
+    from sglang.srt.model_executor.model_runner import ModelRunner
+
+
+def should_use_cp_cache_layer_split_pool(model_runner: ModelRunner) -> bool:
+    """Whether this model runner should construct a Cache LayerSplit pool."""
+    return (
+        not model_runner.is_draft_worker
+        and model_runner.server_args.enable_cp_cache_layer_split
+    )
+
+
+def get_cp_cache_layer_shard_info(
+    model_runner: ModelRunner,
+) -> tuple[Optional[int], int]:
+    """Return ``(cp_rank, cp_size)`` for a Cache LayerSplit pool.
+
+    ``(None, 1)`` means that this model runner should use its regular pool.
+    """
+    if not should_use_cp_cache_layer_split_pool(model_runner):
+        return None, 1
+
+    parallel = get_parallel()
+    if parallel.attn_cp_size <= 1:
+        return None, 1
+    return parallel.attn_cp_rank, parallel.attn_cp_size
+
+
+def get_layer_shard_range(
+    rank: int, shard_size: int, total_layers: int
+) -> tuple[int, int]:
+    """Contiguous ``[start, end)`` local-layer range owned by ``rank``.
+
+    Layers are split as evenly as possible; the first ``total_layers %
+    shard_size`` ranks own one extra layer.
+    """
+    base = total_layers // shard_size
+    rem = total_layers % shard_size
+    start = rank * base + min(rank, rem)
+    end = start + base + (1 if rank < rem else 0)
+    return start, end
+
+
+def get_global_layer_shard_range(
+    rank: int,
+    shard_size: int,
+    start_layer: int,
+    layer_num: int,
+) -> tuple[int, int]:
+    """Global layer range owned by ``rank`` within one pipeline stage."""
+    if shard_size <= 0 or not 0 <= rank < shard_size:
+        raise ValueError(f"Invalid rank={rank} for shard_size={shard_size}")
+    if start_layer < 0 or layer_num <= 0:
+        raise ValueError(
+            f"Invalid stage start_layer={start_layer}, layer_num={layer_num}"
+        )
+    local_start, local_end = get_layer_shard_range(rank, shard_size, layer_num)
+    return start_layer + local_start, start_layer + local_end
+
+
+def get_layer_owner(local_layer_idx: int, shard_size: int, total_layers: int) -> int:
+    """CP rank that owns ``local_layer_idx`` under the contiguous split."""
+    for rank in range(shard_size):
+        start, end = get_layer_shard_range(rank, shard_size, total_layers)
+        if start <= local_layer_idx < end:
+            return rank
+    raise ValueError(
+        f"Invalid local_layer_idx={local_layer_idx} for "
+        f"shard_size={shard_size}, total_layers={total_layers}"
+    )
```

#### pool_base.py

```diff
@@ -0,0 +1,100 @@
+"""Common ownership contract for CP Cache LayerSplit pools."""
+
+from __future__ import annotations
+
+import logging
+
+from sglang.srt.mem_cache.cp_cache_layer_split.utils import (
+    get_global_layer_shard_range,
+    get_layer_owner,
+    get_layer_shard_range,
+)
+
+logger = logging.getLogger(__name__)
+
+
+class CpCacheLayerSplitPoolBase:
+    """Stage-local layer ownership shared by all Cache LayerSplit pools."""
+
+    def _init_cp_cache_layer_split(
+        self,
+        *,
+        cp_rank: int,
+        cp_size: int,
+        layer_shard_start_layer: int,
+        layer_shard_layer_num: int,
+    ) -> None:
+        """Initialize the stage-local layer ownership state."""
+        if cp_size <= 1:
+            raise ValueError(f"Cache LayerSplit requires cp_size > 1, got {cp_size}")
+        if not 0 <= cp_rank < cp_size:
+            raise ValueError(f"Invalid cp_rank={cp_rank} for cp_size={cp_size}")
+        if layer_shard_start_layer < 0 or layer_shard_layer_num <= 0:
+            raise ValueError(
+                "Invalid Cache LayerSplit stage: "
+                f"start_layer={layer_shard_start_layer}, "
+                f"layer_num={layer_shard_layer_num}"
+            )
+
+        self.cp_rank = cp_rank
+        self.cp_size = cp_size
+        self._layer_shard_start_layer = layer_shard_start_layer
+        self._layer_shard_layer_num = layer_shard_layer_num
+
+    def _local_layer_idx(self, layer_id: int) -> int:
+        local_layer_idx = layer_id - self._layer_shard_start_layer
+        if not 0 <= local_layer_idx < self._layer_shard_layer_num:
+            raise ValueError(
+                f"Layer {layer_id} is outside Cache LayerSplit stage "
+                f"[{self._layer_shard_start_layer}, "
+                f"{self._layer_shard_start_layer + self._layer_shard_layer_num})"
+            )
+        return local_layer_idx
+
+    def _owned_local_layer_range(self) -> tuple[int, int]:
+        return get_layer_shard_range(
+            self.cp_rank, self.cp_size, self._layer_shard_layer_num
+        )
+
+    def _owned_global_layer_range(self) -> tuple[int, int]:
+        return get_global_layer_shard_range(
+            self.cp_rank,
+            self.cp_size,
+            self._layer_shard_start_layer,
+            self._layer_shard_layer_num,
+        )
+
+    def _is_layer_owned(self, layer_id: int) -> bool:
+        owned_start, owned_end = self._owned_global_layer_range()
+        return owned_start <= layer_id < owned_end
+
+    def _get_layer_owner_rank(self, layer_id: int) -> int:
+        return get_layer_owner(
+            self._local_layer_idx(layer_id),
+            self.cp_size,
+            self._layer_shard_layer_num,
+        )
+
+    def _build_owned_layer_local_index_map(self) -> dict[int, int]:
+        owned_start, owned_end = self._owned_global_layer_range()
+        return {
+            layer_id: layer_id - owned_start
+            for layer_id in range(owned_start, owned_end)
+        }
+
+    def _log_layer_shard_plan(self) -> None:
+        owned_start, owned_end = self._owned_global_layer_range()
+        logger.info(
+            "Cache LayerSplit shard: stage=[%s,%s), cp_rank=%s/%s, owned=[%s,%s)",
+            self._layer_shard_start_layer,

# ... truncated 11 lines; see upstream PR ...
```

#### server_args.py（节选）

```diff
@@ -1090,10 +1090,13 @@ class ServerArgs:
         ),
         NS("parallel"),
     ] = None
-    # Split DSA GPU KV/indexer cache layers across CP ranks.
-    enable_dsa_cache_layer_split: A[
+    # Split GPU cache layers across CP ranks.
+    enable_cp_cache_layer_split: A[
         bool,
-        "Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community.",
+        Arg(
+            help="Split GPU cache layers across context-parallel ranks to reduce per-rank cache memory. Currently supported for DSA models with the mooncake transfer backend.",
+            aliases=["--enable-dsa-cache-layer-split"],
+        ),
         NS("parallel"),
     ] = False
     enable_dsa_prefill_context_parallel: A[bool, Arg(no_cli=True), NS("parallel")] = (
@@ -5039,9 +5042,9 @@ def _handle_model_specific_adjustments(self):
         hf_config = self.get_model_config().hf_config
         model_arch = hf_config.architectures[0]
 
-        if self.enable_dsa_cache_layer_split and not is_deepseek_dsa(hf_config):
+        if self.enable_cp_cache_layer_split and not is_deepseek_dsa(hf_config):
             raise ValueError(
-                "--enable-dsa-cache-layer-split is only supported for DSA "
+                "--enable-cp-cache-layer-split is only supported for DSA "
                 "(DeepSeek Sparse Attention) models."
             )
 
@@ -5158,26 +5161,26 @@ def _handle_model_specific_adjustments(self):
                         self.disaggregation_mode != "decode"
                     ), "CP is only supported for prefill when PD disaggregation, please remove --enable-prefill-cp."
                 if (
-                    self.enable_dsa_cache_layer_split
+                    self.enable_cp_cache_layer_split
                     and self.disaggregation_mode != "prefill"
                 ):
                     if self.disaggregation_mode == "decode":
                         raise ValueError(
-                            "--enable-dsa-cache-layer-split is not supported on "
+                            "--enable-cp-cache-layer-split is not supported on "
                             "decode workers. This flag is a prefill-CP "
                             "optimization; decode receives full cache shards "
                             "through PD transfer."
                         )
                     raise ValueError(
-                        "--enable-dsa-cache-layer-split is only supported on PD "
+                        "--enable-cp-cache-layer-split is only supported on PD "
                         "prefill workers. Non-PD workers also run decode and "
                         "require ordinary local decode cache semantics."
                     )
-                if self.enable_dsa_cache_layer_split and (
+                if self.enable_cp_cache_layer_split and (
                     not self.enable_prefill_cp or self.cp_strategy != "interleave"
                 ):
                     raise ValueError(
-                        "--enable-dsa-cache-layer-split requires "
+                        "--enable-cp-cache-layer-split requires "
                         "--enable-prefill-cp and --cp-strategy interleave "
                         "(or legacy --enable-nsa-prefill-context-parallel with "
                         "--nsa-prefill-cp-mode round-robin-split)."
@@ -5186,19 +5189,19 @@ def _handle_model_specific_adjustments(self):
                 # transfer path. mori/nixl support is a temporary limitation
                 # and will be added later by the community.
                 if (
-                    self.enable_dsa_cache_layer_split
+                    self.enable_cp_cache_layer_split
                     and self.disaggregation_transfer_backend != "mooncake"
                 ):
                     raise ValueError(
-                        "--enable-dsa-cache-layer-split currently only supports "
+                        "--enable-cp-cache-layer-split currently only supports "
                         "the mooncake transfer backend (mooncake / mooncake_tcp). "
                         f"Got --disaggregation-transfer-backend "
                         f"{self.disaggregation_transfer_backend!r}. mori/nixl "
                         "support will be added later by the community."
                     )
-                if self.enable_dsa_cache_layer_split and self.pp_size > 1:
+                if self.enable_cp_cache_layer_split and self.pp_size > 1:
                     raise ValueError(
-                        "--enable-dsa-cache-layer-split is not supported with "
+                        "--enable-cp-cache-layer-split is not supported with "
                         "pipeline parallelism (pp_size > 1) yet. It requires "
                         "prefill context parallelism, and CP + PP has not been "
                         "validated for this feature."
```

### 本地核对

- [ ] 目标树已包含 #33382 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #29187 — feat: Add DeepSeek V4 CP KV LayerSplit

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/29187
- **分类**：Prefill CP LayerSplit
- **Author**：@jellysnack  ·  diffstat `+3596/-415`
- **一句话**：DSV4 CP KV LayerSplit 完整实现

### 做什么

DSV4 完整 LayerSplit：按层把 KV/indexer/compress 状态分给不同 CP rank，配合 interleave CP；含 PD/HiCache/pool layout。依赖公共 infra（#33382）与 compressor v2 等。

### 开关 / 启用方式

```bash
--enable-cp-cache-layer-split          # 或旧别名 --enable-dsa-cache-layer-split
--enable-prefill-cp --cp-strategy interleave
SGLANG_OPT_USE_COMPRESSOR_V2=1
# server_args 还会校验 FlashMLA unified_kv / HiCache backend 等（见 validation patch）
```

### 关键文件 / 符号（异地核对点）

`build_cp_cache_layer_split_deepseek_v4_pool_layout`, `deepseek_v4_pool.py`, `deepseek_v4_helpers.py`, model 侧 `use_layer_split_prefill`

`cp_cache_layer_split/deepseek_v4_*.py`, `deepseek_v4.py`, disaggregation/*, hicache, server_args

### 关键 diff

#### layout.py

```diff
@@ -0,0 +1,93 @@
+"""DeepSeek V4 pool layouts for CP Cache LayerSplit."""
+
+from __future__ import annotations
+
+from dataclasses import dataclass
+
+from sglang.srt.mem_cache.cp_cache_layer_split.utils import (
+    get_global_layer_shard_range,
+)
+
+
+@dataclass(frozen=True)
+class CpCacheLayerSplitDeepSeekV4PoolLayout:
+    """Per-rank layer-buffer counts for each V4 KV sub-pool."""
+
+    swa_layer_num: int
+    c4_layer_num: int
+    c128_layer_num: int
+    c4_indexer_layer_num: int
+    c4_state_layer_num: int
+    c128_state_layer_num: int
+    c4_indexer_state_layer_num: int
+
+
+def build_cp_cache_layer_split_deepseek_v4_pool_layout(
+    cp_rank: int,
+    cp_size: int,
+    start_layer: int,
+    end_layer_exclusive: int,
+    compression_ratios: list[int],
+) -> CpCacheLayerSplitDeepSeekV4PoolLayout:
+    """Buffer counts for one attention-CP rank."""
+    if not 0 <= start_layer < end_layer_exclusive <= len(compression_ratios):
+        raise ValueError(
+            "Invalid DSV4 Cache LayerSplit stage: "
+            f"start_layer={start_layer}, end_layer={end_layer_exclusive}, "
+            f"compression_ratios={len(compression_ratios)}"
+        )
+    owned_start, owned_end = get_global_layer_shard_range(
+        cp_rank,
+        cp_size,
+        start_layer,
+        end_layer_exclusive - start_layer,
+    )
+
+    def _count(compress_ratio: int) -> int:
+        return sum(
+            1
+            for layer_id in range(owned_start, owned_end)
+            if compression_ratios[layer_id] == compress_ratio
+        )
+
+    c4_layer_num = _count(4)
+    c128_layer_num = _count(128)
+    return CpCacheLayerSplitDeepSeekV4PoolLayout(
+        swa_layer_num=owned_end - owned_start,
+        c4_layer_num=c4_layer_num,
+        c128_layer_num=c128_layer_num,
+        c4_indexer_layer_num=c4_layer_num,
+        c4_state_layer_num=c4_layer_num,
+        c128_state_layer_num=c128_layer_num,
+        c4_indexer_state_layer_num=c4_layer_num,
+    )
+
+
+def build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout(
+    cp_size: int,
+    start_layer: int,
+    end_layer_exclusive: int,
+    compression_ratios: list[int],
+) -> CpCacheLayerSplitDeepSeekV4PoolLayout:
+    """Max per-pool layer counts across all CP ranks."""
+    layouts = [
+        build_cp_cache_layer_split_deepseek_v4_pool_layout(
+            cp_rank,
+            cp_size,
+            start_layer,
+            end_layer_exclusive,
+            compression_ratios,
+        )
+        for cp_rank in range(cp_size)
+    ]
+    return CpCacheLayerSplitDeepSeekV4PoolLayout(
+        swa_layer_num=max(layout.swa_layer_num for layout in layouts),
+        c4_layer_num=max(layout.c4_layer_num for layout in layouts),
+        c128_layer_num=max(layout.c128_layer_num for layout in layouts),
+        c4_indexer_layer_num=max(layout.c4_indexer_layer_num for layout in layouts),
+        c4_state_layer_num=max(layout.c4_state_layer_num for layout in layouts),
+        c128_state_layer_num=max(layout.c128_state_layer_num for layout in layouts),
+        c4_indexer_state_layer_num=max(
+            layout.c4_indexer_state_layer_num for layout in layouts
+        ),
+    )
```

#### __init__.py exports

```diff
@@ -0,0 +1,19 @@
+"""CP Cache LayerSplit public helpers."""
+
+from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_layout import (
+    CpCacheLayerSplitDeepSeekV4PoolLayout,
+    build_cp_cache_layer_split_deepseek_v4_pool_layout,
+    build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout,
+)
+from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
+    CpCacheLayerSplitPoolBase,
+    is_cp_cache_layer_split_pool,
+)
+
+__all__ = [
+    "CpCacheLayerSplitDeepSeekV4PoolLayout",
+    "CpCacheLayerSplitPoolBase",
+    "build_cp_cache_layer_split_deepseek_v4_pool_layout",
+    "build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout",
+    "is_cp_cache_layer_split_pool",
+]
```

#### server_args validation（节选）

```diff
@@ -100,6 +100,7 @@
 
 # Define constants
 DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES = ()
+CP_CACHE_LAYER_SPLIT_HICACHE_STORAGE_BACKENDS = ("file", "mooncake")
 
 SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}
 if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
@@ -1051,10 +1052,13 @@ class ServerArgs:
         ),
         NS("parallel"),
     ] = None
-    # Split DSA GPU KV/indexer cache layers across CP ranks.
-    enable_dsa_cache_layer_split: A[
+    # Split model cache layers across CP ranks.
+    enable_cp_cache_layer_split: A[
         bool,
-        "Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community.",
+        Arg(
+            help="Split selected GPU cache layers across context-parallel ranks to reduce per-rank cache memory. Model-specific support is required.",
+            aliases=["--enable-dsa-cache-layer-split"],
+        ),
         NS("parallel"),
     ] = False
     enable_dsa_prefill_context_parallel: A[bool, Arg(no_cli=True), NS("parallel")] = (
@@ -3391,6 +3395,12 @@ def __post_init__(self):
         # Apply model-specific adjustments.
         self._handle_model_specific_adjustments()
 
+        # Re-apply after model-specific defaults resolve attention_backend so
+        # canonical CP mirrors to the right legacy runtime aliases before
+        # LayerSplit adjusts CUDA graph settings.
+        self._handle_legacy_cp_arguments()
+        self._handle_cp_cache_layer_split()
+
         # Set kernel backends.
         self._handle_sampling_backend()
         # Must run before _handle_attention_backend_compatibility so the
@@ -3452,6 +3462,7 @@ def __post_init__(self):
         from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
 
         handle_speculative_decoding(self)
+        self._validate_cp_cache_layer_split_speculative_decoding()
 
         # Needs the draft-token count derived just above.
         self._validate_gdn_replayssm_spec_ring()
@@ -4747,12 +4758,6 @@ def _handle_model_specific_adjustments(self):
         hf_config = self.get_model_config().hf_config
         model_arch = hf_config.architectures[0]
 
-        if self.enable_dsa_cache_layer_split and not is_deepseek_dsa(hf_config):
-            raise ValueError(
-                "--enable-dsa-cache-layer-split is only supported for DSA "
-                "(DeepSeek Sparse Attention) models."
-            )
-
         if self.enable_cp_decode_attn_tp:
             from sglang.srt.layers.cp.cp_decode_attn_tp import (
                 CP_DECODE_ATTN_TP_SUPPORTED_ARCHS,
@@ -4866,52 +4871,6 @@ def _handle_model_specific_adjustments(self):
                     assert (
                         self.disaggregation_mode != "decode"
                     ), "CP is only supported for prefill when PD disaggregation, please remove --enable-prefill-cp."
-                if (
-                    self.enable_dsa_cache_layer_split
-                    and self.disaggregation_mode != "prefill"
-                ):
-                    if self.disaggregation_mode == "decode":
-                        raise ValueError(
-                            "--enable-dsa-cache-layer-split is not supported on "
-                            "decode workers. This flag is a prefill-CP "
-                            "optimization; decode receives full cache shards "
-                            "through PD transfer."
-                        )
-                    raise ValueError(
-                        "--enable-dsa-cache-layer-split is only supported on PD "
-                        "prefill workers. Non-PD workers also run decode and "
-                        "require ordinary local decode cache semantics."
-                    )
-                if self.enable_dsa_cache_layer_split and (
-                    not self.enable_prefill_cp or self.cp_strategy != "interleave"
-                ):
-                    raise ValueError(
-                        "--enable-dsa-cache-layer-split requires "
-                        "--enable-prefill-cp and --cp-strategy interleave "
-                        "(or legacy --enable-nsa-prefill-context-parallel with "
-                        "--nsa-prefill-cp-mode round-robin-split)."
-                    )
-                # Layer split relies on the mooncake all-CP-rank KV/indexer
-                # transfer path. mori/nixl support is a temporary limitation
-                # and will be added later by the community.
-                if (
-                    self.enable_dsa_cache_layer_split
-                    and self.disaggregation_transfer_backend != "mooncake"
-                ):
-                    raise ValueError(
-                        "--enable-dsa-cache-layer-split currently only supports "
-                        "the mooncake transfer backend (mooncake / mooncake_tcp). "
-                        f"Got --disaggregation-transfer-backend "
-                        f"{self.disaggregation_transfer_backend!r}. mori/nixl "
-                        "support will be added later by the community."
-                    )
-                if self.enable_dsa_cache_layer_split and self.pp_size > 1:
-                    raise ValueError(
-                        "--enable-dsa-cache-layer-split is not supported with "
-                        "pipeline parallelism (pp_size > 1) yet. It requires "
-                        "prefill context parallelism, and CP + PP has not been "
-                        "validated for this feature."
-                    )
 
             else:
                 # DeepSeek V3/R1/V3.1
@@ -6045,6 +6004,129 @@ def _handle_context_parallelism(self):
 
         init_cp_strategy(self)
 
+    def _validate_cp_cache_layer_split_model(self) -> bool:
+        """Validate the model and return whether DSV4-specific guards apply."""
+        has_concrete_model_config = (
+            self.model_path.lower() not in ("none", "dummy")

# ... truncated 83 lines; see upstream PR ...
```

#### deepseek_v4.py 接入（节选）

```diff
@@ -144,6 +144,11 @@
         prepare_context_parallel_metadata,
     )
 
+from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_helpers import (
+    is_cp_cache_layer_split_deepseek_v4_pool,
+    maybe_prefetch_cp_kv_swa,
+    maybe_wait_cp_kv_swa_prefetch,
+)
 from sglang.srt.utils import (
     LazyValue,
     add_prefix,
@@ -323,6 +328,38 @@ def _freqs_cis_to_cos_sin(
     from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 
 
+def _can_dsa_cp_split_for_deepseek_v4(
+    input_ids_len: int,
+    cp_size: int,
+    use_dsa: bool,
+    forward_batch: ForwardBatch,
+) -> bool:
+    if can_dsa_cp_split(input_ids_len, cp_size, use_dsa, forward_batch):
+        return True
+    if (
+        not use_dsa
+        or not is_dsa_prefill_cp_round_robin_split()
+        or cp_size <= 1
+        or not forward_batch.forward_mode.is_context_parallel_extend()
+        or input_ids_len == 0
+        or input_ids_len % cp_size != 0
+    ):
+        return False
+    extend_seq_lens = forward_batch.extend_seq_lens_cpu
+    if extend_seq_lens is None:
+        return False
+    real_extend_tokens = sum(int(x) for x in extend_seq_lens)
+    if real_extend_tokens == 0:
+        return False
+    token_to_kv_pool = get_token_to_kv_pool()
+    # LayerSplit KV is CP-sharded, so tiny padded prefill batches still need
+    # the CP path to broadcast/remap non-owned layers before attention reads.
+    can_force_tiny_cp = is_cp_cache_layer_split_deepseek_v4_pool(token_to_kv_pool)
+    if not can_force_tiny_cp:
+        return False
+    return True
+
+
 @register_custom_op(mutates_args=["output"])
 @register_split_op()
 def deepseek_v4_attention_with_output(
@@ -722,13 +759,17 @@ def _compute_kv_to_cache(
         Replaces the bf16-kv-intermediate path. Used everywhere except the DSA
         prefill-CP case (which needs bf16 kv for the cross-rank all-gather).
         """
+        token_to_kv_pool = get_token_to_kv_pool()
+        if TYPE_CHECKING:
+            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
+        if is_cp_cache_layer_split_deepseek_v4_pool(
+            token_to_kv_pool
+        ) and token_to_kv_pool.should_skip_swa_write(self.layer_id):
+            return
         if qkv_a is not None:
             kv = qkv_a[..., self.q_lora_rank :]
         else:
             kv, _ = self.wkv(x)
-        token_to_kv_pool = get_token_to_kv_pool()
-        if TYPE_CHECKING:
-            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
         token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
             layer_id=self.layer_id,
             swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
@@ -964,8 +1005,12 @@ def _forward_prepare(
 
         unified = is_unified_kv_triton()
         is_decode = forward_batch.forward_mode.is_decode_or_idle()
+        token_to_kv_pool = get_token_to_kv_pool()
+        use_layer_split_prefill = (
+            is_cp_cache_layer_split_deepseek_v4_pool(token_to_kv_pool) and use_cp
+        )
         do_fused_store = (unified and is_decode) or (
-            not unified and self.use_fused_qk_norm_rope
+            not unified and self.use_fused_qk_norm_rope and not use_layer_split_prefill
         )
 
         if do_fused_store:
@@ -986,7 +1031,6 @@ def _forward_prepare(
                 else self.wkv(x_linear)[0]
             )
 
-            token_to_kv_pool = get_token_to_kv_pool()
             if unified:
                 swa_cache = token_to_kv_pool.get_unified_kv(self.layer_id)
                 # swa_loc is layer-independent; computed once per forward by the
@@ -1103,23 +1147,42 @@ def _forward_prepare(
                 )
                 kv = None
 
+        maybe_prefetch_cp_kv_swa(get_token_to_kv_pool(), self.layer_id, forward_batch)
+
         del qkv_a
 
+        token_to_kv_pool = get_token_to_kv_pool()
+        reorder_c4_extra = (
+            use_cp
+            and self.indexer is not None
+            and self.compressor is not None
+            and is_cp_cache_layer_split_deepseek_v4_pool(token_to_kv_pool)
+            and token_to_kv_pool.should_prefetch_extra_from_page_table(self.layer_id)
+        )
+
+        if reorder_c4_extra:
+            # LayerSplit runs C4 compression before the indexer so the extra-KV
+            # broadcast can overlap with indexer work on this layer.
+            attn_backend.forward_core_compressor(
+                x,
+                forward_batch,
+                self.layer_id,
+                self.compressor,
+            )

# ... truncated 83 lines; see upstream PR ...
```

### 本地核对

- [ ] 目标树已包含 #29187 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #33532 — [CP]: Support CP V2 Strategy for dsv4

- **Upstream 状态**：✅ **已合入** (2026-08-07)
- **URL**：https://github.com/sgl-project/sglang/pull/33532
- **分类**：Prefill CP 策略
- **Author**：@hzh0425  ·  diffstat `+161/-59`
- **一句话**：DSV4 接入 CP-v2 / interleave strategy

### 做什么

把 DSV4 Prefill CP 从旧 DSA CP-v1（`--enable-dsa-prefill-context-parallel --dsa-prefill-cp-mode round-robin-split`）接到统一 CP-v2（`SGLANG_ENABLE_CP_V2=1 --enable-prefill-cp --cp-strategy interleave`），并修 metadata pad vs cache-write 长度一致性。

### 开关 / 启用方式

```bash
SGLANG_ENABLE_CP_V2=1
--enable-prefill-cp
--cp-strategy interleave
# 内部仍会设置 dsa_prefill_cp_mode=round-robin-split（切分语义对齐）
```

### 关键文件 / 符号（异地核对点）

- `is_cp_v2_active` / `cp_materialize_global_token_order` / `cp_round_robin_input_ids_v2`
- `DSV4AttnMetadata.init_compression_metadata(num_tokens=...)`
- `apply_cp_reindex(num_tokens=...)`
- 测试：`test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` 改用 interleave + CP_V2

- `python/sglang/srt/layers/cp/utils.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `python/sglang/srt/models/deepseek_v4.py`
- `python/sglang/srt/models/deepseek_v4_nextn.py`
- `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- `python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py`
- `python/sglang/srt/model_executor/runner/eager_runner.py`

### 关键 diff

#### hook：强制/对齐 CP 配置

```diff
diff --git a/python/sglang/srt/arg_groups/deepseek_v4_hook.py b/python/sglang/srt/arg_groups/deepseek_v4_hook.py
index 8908b9bb2672..da21b91d92f8 100644
--- a/python/sglang/srt/arg_groups/deepseek_v4_hook.py
+++ b/python/sglang/srt/arg_groups/deepseek_v4_hook.py
@@ -167,6 +167,7 @@ def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
         )
 
     server_args.enable_dsa_prefill_context_parallel = True
+    server_args.enable_prefill_context_parallel = False
     server_args.dsa_prefill_cp_mode = "round-robin-split"
     server_args.enable_dp_attention = True
     server_args.moe_dense_tp_size = 1
```

#### cp/utils.py：v2 materialize / round-robin ids

```diff
diff --git a/python/sglang/srt/layers/cp/utils.py b/python/sglang/srt/layers/cp/utils.py
index e5cc4c6fa4b7..b397c4fed19e 100644
--- a/python/sglang/srt/layers/cp/utils.py
+++ b/python/sglang/srt/layers/cp/utils.py
@@ -34,6 +34,7 @@
     ZigzagContextParallelMetadata,
     ZigzagCPStrategy,
 )
+from sglang.srt.layers.moe.utils import get_moe_a2a_backend
 from sglang.srt.runtime_context import get_parallel
 
 if TYPE_CHECKING:
@@ -219,6 +220,17 @@ def cp_shard_position_ids(complete_position_ids: Any, forward_batch):
     return strategy.shard_position_ids(complete_position_ids, forward_batch)
 
 
+def cp_round_robin_input_ids_v2(input_ids: Any, forward_batch):
+    assert is_cp_v2_active(forward_batch)
+    if not get_moe_a2a_backend().is_none():
+        return cp_shard_hidden_states(input_ids, forward_batch)
+
+    physical_tokens = sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
+    padded_input_ids = input_ids.new_zeros(physical_tokens)
+    padded_input_ids[: input_ids.shape[0]] = input_ids
+    return padded_input_ids.view(-1, get_parallel().attn_cp_size).T.flatten()
+
+
 def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None):
     """Gather CP-v2 hidden states at the model boundary when this batch is active."""
     assert is_cp_v2_active(forward_batch)
@@ -226,18 +238,40 @@ def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None)
     assert strategy is not None
 
     if isinstance(x, tuple):
-        hidden_states, *rest = x
-        hidden_states = strategy.gather_hidden_states(
-            hidden_states, forward_batch, stream
+        gathered = tuple(
+            (
+                strategy.gather_hidden_states(item, forward_batch, stream)
+                if item is not None
+                else None
+            )
+            for item in x
         )
         # MiMo's text-only body returns (hidden_states, None); logits expects a tensor.
-        if len(rest) == 1 and rest[0] is None:
-            return hidden_states
-        return (hidden_states, *rest)
+        if len(gathered) == 2 and gathered[1] is None:
+            return gathered[0]
+        return gathered
 
     return strategy.gather_hidden_states(x, forward_batch, stream)
 
 
+def cp_materialize_global_token_order(
+    x: Any, forward_batch, stream: Optional[Any] = None
+):
+    """Materialize a CP tensor in the global logical token order."""
+    if is_cp_v2_active(forward_batch):
+        strategy = get_cp_strategy()
+        assert strategy is not None
+        return strategy.gather_kv_cache(x, forward_batch, stream)
+
+    # TODO(hzh0425): Keep the legacy gather temporarily for CP-v1 compatibility. Remove it
+    # with the follow-up CP-v1 cleanup.
+    from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
+
+    return cp_all_gather_rerange_output(
+        x, get_parallel().attn_cp_size, forward_batch, stream
+    )
+
+
 @contextmanager
 def cp_shard_model_inputs(
     complete_hidden_states: Any,
@@ -293,6 +327,8 @@ def _to_int_list(values) -> Optional[list[int]]:
     "get_cp_strategy",
     "is_cp_v2_active",
     "cp_gather_after_forward",
+    "cp_materialize_global_token_order",
+    "cp_round_robin_input_ids_v2",
     "cp_shard_hidden_states",
     "cp_shard_model_inputs",
     "cp_shard_position_ids",
```

#### backend：pad vs num_tokens / apply_cp_reindex

```diff
diff --git a/python/sglang/srt/layers/attention/deepseek_v4_backend.py b/python/sglang/srt/layers/attention/deepseek_v4_backend.py
index 402d3baa0355..0b7cf9195996 100644
--- a/python/sglang/srt/layers/attention/deepseek_v4_backend.py
+++ b/python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -58,6 +58,7 @@
     SparsePrefillWorkspace,
 )
 from sglang.srt.layers.attention.verify_mask import VerifyMask, maybe_create_verify_mask
+from sglang.srt.layers.cp.utils import is_cp_v2_active
 from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
 from sglang.srt.runtime_context import get_parallel, get_spec
@@ -277,11 +278,16 @@ def refresh_for_breakable_cuda_graph_replay_(self, other: DSV4AttnMetadata) -> N
         for field_name in reference_assign_fields:
             setattr(self, field_name, getattr(other, field_name))
 
-    def init_compression_metadata(self):
+    def init_compression_metadata(self, num_tokens: Optional[int] = None) -> None:
         assert self.page_table.dim() == 2
+        # CP-v2 pads causal metadata for per-rank partitioning, while cache-write
+        # locations remain one-per-logical-token. num_tokens tracks that unpadded
+        # length; legacy paths use the metadata length.
+        if num_tokens is None:
+            num_tokens = self.seq_lens_casual.shape[0]
         assert (
-            self.raw_out_loc.shape == self.seq_lens_casual.shape
-        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"
+            self.raw_out_loc.shape[0] == num_tokens
+        ), f"{self.raw_out_loc.shape=}, {num_tokens=}"
 
         (
             self.c4_out_loc,
@@ -305,6 +311,8 @@ def init_compression_metadata(self):
         self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
         self.swa_page_indices = _pad_last_dim(self.swa_page_indices)
 
+    # Cache-write locations stay in global logical order and are intentionally
+    # excluded from CP reindexing.
     _CP_REINDEX_FIELDS = [
         "seq_lens_casual",
         "positions_casual",
@@ -323,7 +331,7 @@ def init_compression_metadata(self):
         "c128_out_loc",
     ]
 
-    def apply_cp_reindex(self) -> None:
+    def apply_cp_reindex(self, num_tokens: Optional[int] = None) -> None:
         cp_rank = get_parallel().attn_cp_rank
         cp_size = get_parallel().attn_cp_size
         idx = slice(cp_rank, None, cp_size)
@@ -333,6 +341,8 @@ def apply_cp_reindex(self) -> None:
             "CP round-robin requires padding to ensure divisibility."
         )
         expected_local_len = pre_global_len // cp_size
+        if num_tokens is None:
+            num_tokens = pre_global_len
         for field_name in self._CP_REINDEX_FIELDS:
             val = getattr(self, field_name, None)
             assert isinstance(
@@ -350,9 +360,9 @@ def apply_cp_reindex(self) -> None:
             val = getattr(self, field_name, None)
             if val is None:
                 continue
-            assert val.shape[0] == pre_global_len, (
+            assert val.shape[0] == num_tokens, (
                 f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
-                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
+                f"!= num_tokens={num_tokens} (must remain global for compressor write path)"
             )
 
     def init_flashmla_related(self, is_prefill: bool = False):
@@ -721,13 +731,21 @@ def init_forward_metadata_prefill(
         use_prefill_cuda_graph: bool = False,
         online_c128_state_slot_offset: int = 0,
         dspark_block_size: Optional[int] = None,
+        forward_batch: Optional[ForwardBatch] = None,
     ) -> DSV4Metadata:
+        padded_num_tokens = out_cache_loc.shape[0]
+        cp_v2_active = forward_batch is not None and is_cp_v2_active(forward_batch)
+        if cp_v2_active:
+            cp_metadata = forward_batch.attn_cp_metadata
+            assert cp_metadata is not None
+            padded_num_tokens = sum(cp_metadata.per_rank_actual_token)
+
         seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
             num_tokens=num_tokens,
             seq_lens=seq_lens_cpu,
             extend_seq_lens=extend_seq_lens_cpu,
             req_pool_indices=req_pool_indices,
-            padded_num_tokens=out_cache_loc.shape[0],
+            padded_num_tokens=padded_num_tokens,
             seq_lens_tensor=seq_lens,
             extend_seq_lens_tensor=extend_seq_lens,
             extend_start_loc=extend_start_loc,
@@ -741,7 +759,11 @@ def init_forward_metadata_prefill(
             need_compress=need_compress,
             is_prefill=True,
             dspark_block_size=dspark_block_size,
+            num_tokens=num_tokens if cp_v2_active else None,
         )
+        if cp_v2_active:
+            core_attn_metadata.apply_cp_reindex(num_tokens=num_tokens)
+            core_attn_metadata.init_flashmla_related(is_prefill=True)
         indexer_metadata = (
             self.init_forward_metadata_indexer(
                 core_attn_metadata,
@@ -1458,6 +1480,7 @@ def _build_forward_metadata(
                 extend_start_loc=forward_batch.extend_start_loc,
                 need_compress=True,
                 use_prefill_cuda_graph=use_prefill_cuda_graph,
+                forward_batch=forward_batch,
             )
         else:
             raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")
@@ -1945,6 +1968,7 @@ def make_core_attn_metadata(
         need_compress: bool = True,
         is_prefill: bool = False,
         dspark_block_size: Optional[int] = None,
+        num_tokens: Optional[int] = None,
     ) -> DSV4AttnMetadata:
         assert self.swa_page_size == SWA_WINDOW
 
@@ -2001,7 +2025,7 @@ def make_core_attn_metadata(
         )
 
         if need_compress:
-            core_attn_metadata.init_compression_metadata()
+            core_attn_metadata.init_compression_metadata(num_tokens)
             core_attn_metadata.init_flashmla_related(is_prefill=is_prefill)
         else:
             core_attn_metadata.c4_sparse_topk_lengths = None
```

#### deepseek_v4.py：v1/v2 分支

```diff
diff --git a/python/sglang/srt/models/deepseek_v4.py b/python/sglang/srt/models/deepseek_v4.py
index 65dd55549924..1f89c0839762 100644
--- a/python/sglang/srt/models/deepseek_v4.py
+++ b/python/sglang/srt/models/deepseek_v4.py
@@ -59,6 +59,11 @@
     dsa_cp_reduce_scatter_hidden_states,
 )
 from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx
+from sglang.srt.layers.cp.utils import (
+    cp_materialize_global_token_order,
+    cp_round_robin_input_ids_v2,
+    is_cp_v2_active,
+)
 from sglang.srt.layers.dp_attention import (
     _tbo_event,
     attn_tp_all_gather,
@@ -1073,9 +1078,8 @@ def _forward_prepare(
                 # DSA CP: keep bf16 kv around for the cross-rank all-gather, then
                 # write to the FlashMLA cache after gather.
                 kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
-                kv = cp_all_gather_rerange_output(
+                kv = cp_materialize_global_token_order(
                     kv.contiguous(),
-                    self.cp_size,
                     forward_batch,
                     torch.cuda.current_stream(),
                 )
@@ -1124,9 +1128,8 @@ def _forward_prepare(
                     # unified_kv + DSA CP: the 2-source prefill path needs the
                     # FULL current-chunk KV (extend source + ring write), so
                     # all-gather the per-rank bf16 KV across the CP group.
-                    kv = cp_all_gather_rerange_output(
+                    kv = cp_materialize_global_token_order(
                         kv.contiguous(),
-                        self.cp_size,
                         forward_batch,
                         torch.cuda.current_stream(),
                     )
@@ -1134,9 +1137,8 @@ def _forward_prepare(
                 # NSA CP: keep bf16 kv around for the cross-rank all-gather, then
                 # write to the FlashMLA cache after gather.
                 kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
-                kv = cp_all_gather_rerange_output(
+                kv = cp_materialize_global_token_order(
                     kv.contiguous(),
-                    self.cp_size,
                     forward_batch,
                     torch.cuda.current_stream(),
                 )
@@ -2352,8 +2354,13 @@ def forward(
         input_embeds: Optional[torch.Tensor],
         pp_proxy_tensors: Optional[PPProxyTensors] = None,
     ) -> Union[torch.Tensor, PPProxyTensors]:
+        cp_v2_active = is_cp_v2_active(forward_batch)
+        use_prefill_cp = dsa_use_prefill_cp(forward_batch)
         if self.pp_group.is_first_rank:
-            hidden_states = self.embed_tokens(input_ids)
+            if input_embeds is None:
+                hidden_states = self.embed_tokens(input_ids)
+            else:
+                hidden_states = input_embeds
             hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
         else:
             assert pp_proxy_tensors is not None
@@ -2377,11 +2384,16 @@ def forward(
         else:
             input_ids_global = input_ids
 
-        if dsa_use_prefill_cp(forward_batch):
-            if self.pp_group.is_first_rank:
-                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
-            positions = cp_split_and_rebuild_position(forward_batch, positions)
-            input_ids = cp_round_robin_input_ids(input_ids)
+        if use_prefill_cp:
+            if cp_v2_active:
+                input_ids = cp_round_robin_input_ids_v2(input_ids, forward_batch)
+            else:
+                if self.pp_group.is_first_rank:
+                    hidden_states = cp_split_and_rebuild_data(
+                        forward_batch, hidden_states
+                    )
+                positions = cp_split_and_rebuild_position(forward_batch, positions)
+                input_ids = cp_round_robin_input_ids(input_ids)
             input_ids_global = input_ids
 
         # Reset Compressor's per-step freqs_cis cache from any previous step.
@@ -2389,7 +2401,7 @@ def forward(
             if hasattr(forward_batch, _attr):
                 delattr(forward_batch, _attr)
         capture_dspark = self.dspark_layers_to_capture is not None
-        if capture_dspark and dsa_use_prefill_cp(forward_batch):
+        if capture_dspark and use_prefill_cp:
             raise NotImplementedError(
                 "DSpark aux hidden-state capture is not supported together with "
                 "DeepSeek-V4 prefill context parallelism (attn_cp_size > 1). Disable one "
@@ -2444,7 +2456,7 @@ def forward(
                 )
 
         # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
-        if self.pp_group.is_last_rank and dsa_use_prefill_cp(forward_batch):
+        if self.pp_group.is_last_rank and use_prefill_cp and not cp_v2_active:
             hidden_states = cp_all_gather_rerange_output(
                 hidden_states,
                 self.cp_size,
```

#### CI 测试切换到 interleave + CP_V2

```diff
diff --git a/test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py b/test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
index cfac77f383fd..74b7ced6152a 100644
--- a/test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
+++ b/test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py
@@ -1,7 +1,7 @@
-"""B200 extra CI: DeepSeek-V4-Flash FP4 with attn-CP (DSA prefill CP).
+"""B200 extra CI: DeepSeek-V4-Flash FP4 with attn-CP.
 
 Balanced recipe (TP=4, DeepEP, EAGLE) plus --attn-cp-size=4 with the
-DSA prefill-CP round-robin-split mode. Split out of
+DSA prefill-CP interleave strategy. Split out of
 models_e2e/test_deepseek_v4_flash_fp4_b200.py so the `cp` group covers
 all context-parallel tests.
 
@@ -29,6 +29,7 @@
 DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
 
 _DEEPEP_ENV = {
+    "SGLANG_ENABLE_CP_V2": "1",
     "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
     # The draft-extend graph pool costs ~4.5 GB here (DeepEP MoE workspace is
     # captured at full dispatch capacity), which starves the eager prefill
@@ -38,6 +39,7 @@
 }
 
 _MEGAMOE_ENV = {
+    "SGLANG_ENABLE_CP_V2": "1",
     "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
     "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": "1",
     "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": "1",
@@ -81,9 +83,9 @@ def setUpClass(cls):
                 "1",
                 "--speculative-num-draft-tokens",
                 "2",
-                "--enable-dsa-prefill-context-parallel",
-                "--dsa-prefill-cp-mode",
-                "round-robin-split",
+                "--enable-prefill-cp",
+                "--cp-strategy",
+                "interleave",
                 "--deepep-config",
                 DEEPEP_CONFIG,
                 "--mem-fraction-static",
@@ -132,9 +134,9 @@ def setUpClass(cls):
                 "1",
                 "--speculative-num-draft-tokens",
                 "2",
-                "--enable-dsa-prefill-context-parallel",
-                "--dsa-prefill-cp-mode",
-                "round-robin-split",
+                "--enable-prefill-cp",
+                "--cp-strategy",
+                "interleave",
                 "--deepep-config",
                 DEEPEP_CONFIG,
             ],
@@ -181,12 +183,13 @@ def setUpClass(cls):
                 "1",
                 "--speculative-num-draft-tokens",
                 "2",
-                "--enable-dsa-prefill-context-parallel",
-                "--dsa-prefill-cp-mode",
-                "round-robin-split",
+                "--enable-prefill-cp",
+                "--cp-strategy",
+                "interleave",
                 "--moe-runner-backend",  # for fp4 checkpoint
                 "flashinfer_mxfp4",
             ],
+            env={"SGLANG_ENABLE_CP_V2": "1"},
         )
 
     @classmethod
```

### 本地核对

- [ ] 目标树已包含 #33532 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）
- [x] upstream main **已合入**（若目标树基于合入后的 main，此项应已满足）

---

## PR #30416 — [DRAFT] add DCP support for DeepSeek V4

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/30416
- **分类**：Decode CP
- **Author**：@shiyu7  ·  diffstat `+10346/-325`
- **一句话**：DeepSeek V4 Decode Context Parallel (DCP)

### 做什么

Decode Context Parallel：按 `token_id % dcp_size` 切 KV；C4 indexer shard；attention LSE merge（AG/RS 或 A2A）；PD mooncake + PCP–DCP affinity；MTP/graph。与 Prefill CP 互补。

### 开关 / 启用方式

```text
SGLANG_DSV4_ENABLE_DCP=1
--dcp-size <N>          # 通常要求 dcp_size == attn_tp_size
SGLANG_DSV4_DCP_SHARD_C4_INDEXER=1
SGLANG_DSV4_DCP_C4_PACKED_TOPK=...
SGLANG_DSV4_DCP_AG_RS=...
SGLANG_DSV4_DCP_A2A_LSE=...
SGLANG_DSV4_DCP_A2A_LSE_VERIFY=...
```

### 关键文件 / 符号（异地核对点）

`_try_forward_dcp_sharded_c4_indexer`, `merge_dcp_topk_candidates_512`, dcp group, mooncake DCP transfer

超大 PR（70+ files）：backend/indexer/compressor_v2/disaggregation/mooncake、environ、docs

### 关键 diff

#### environ.py DCP flags

```diff
@@ -371,6 +371,13 @@ class Envs:
     # computed dynamically at runtime based on cpu_count; see disaggregation backends.
     SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
     SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
+    # Maximum number of KV indices carried by one Mooncake transfer work item.
+    # 0 preserves the existing behavior. This is useful for long cached-prefix
+    # hits, which otherwise become one very large synchronous transfer.
+    SGLANG_DISAGGREGATION_KV_TRANSFER_CHUNK_SIZE = EnvInt(0)
+    # Emit five-second aggregate diagnostics for Mooncake prefill transfer
+    # queues, high-level send stages, and synchronous engine calls.
+    SGLANG_DISAGGREGATION_TRANSFER_DEBUG = EnvBool(False)
     SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
     SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
     SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
@@ -994,6 +1001,28 @@ class Envs:
     # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
     SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
     SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
+    # Master gate for DeepSeek V4 Decode Context Parallel (DCP). When False
+    # (default), DSv4 hook rejects --dcp-size > 1 with a clear error so users
+    # cannot enable an unvalidated combination by accident. Flip to 1 once
+    # numerical-equivalence regression has been verified on the target cluster.
+    SGLANG_DSV4_ENABLE_DCP = EnvBool(False)
+    # Experimental P0 path for DeepSeek V4 DCP decode: each rank scores only
+    # the C4 indexer entries owned by its DCP shard, then gathers local top-k
+    # candidates and merges them into the global C4 sparse top-k.
+    SGLANG_DSV4_DCP_SHARD_C4_INDEXER = EnvBool(False)
+    # Temporary A/B gate for the one-collective packed candidate merge.
+    SGLANG_DSV4_DCP_C4_PACKED_TOPK = EnvBool(False)
+    # Experimental DSV4 DCP attention merge: gather LSEs, then reduce-scatter
+    # the corrected FP32 output along the head dimension.
+    SGLANG_DSV4_DCP_AG_RS = EnvBool(False)
+    # Experimental DSV4 DCP attention merge: exchange per-destination head
+    # chunks with all-to-all instead of gathering all LSEs and all-reducing all
+    # output heads.
+    SGLANG_DSV4_DCP_A2A_LSE = EnvBool(False)
+    # Debug-only validation for the A2A LSE merge. Run both the reference and
+    # A2A paths on the same real FlashMLA tensors and assert numerical parity.
+    # This mode requires decode CUDA graphs to be disabled.
+    SGLANG_DSV4_DCP_A2A_LSE_VERIFY = EnvBool(False)
     # Default reasoning_effort for dsv4 chat encoder when request doesn't set it.
     # Accepts "", "max", "high" (empty string means unset); other values filtered to None.
     SGLANG_DSV4_REASONING_EFFORT = EnvStr("")
```

#### deepseek_v4_hook.py

```diff
@@ -62,6 +62,51 @@ def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None
                 server_args.speculative_eagle_topk == 1
             ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
 
+    validate_deepseek_v4_dcp(server_args)
+
+
+def validate_deepseek_v4_dcp(server_args: ServerArgs) -> None:
+    """Validate DeepSeek V4 DCP (decode context parallel) compatibility."""
+    if server_args.dcp_size <= 1:
+        return
+
+    if not envs.SGLANG_DSV4_ENABLE_DCP.get():
+        raise ValueError(
+            "DeepSeekV4 DCP (--dcp-size > 1) is gated behind "
+            "SGLANG_DSV4_ENABLE_DCP=1. Set the env var explicitly after "
+            "verifying numerical equivalence vs. dcp_size=1 on your cluster."
+        )
+
+    # DSV4 DCP reduce-scatters attention output across the attention-TP head
+    # dimension, so the DCP group must match the attention-TP group.
+    attn_tp = server_args.tp_size // max(server_args.dp_size, 1)
+    if attn_tp != server_args.dcp_size:
+        raise ValueError(
+            f"DeepSeekV4 DCP currently requires dcp_size ({server_args.dcp_size}) "
+            f"== attn_tp ({attn_tp}); configure tp/dp_size/dcp_size accordingly."
+        )
+
+    # Online c128 + DCP: untested combination; warn but allow.
+    if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
+        logger.warning(
+            "DeepSeekV4 DCP + SGLANG_OPT_USE_ONLINE_COMPRESS combo is "
+            "experimental; numerical correctness has not been validated."
+        )
+
+    # HiSparse + DCP: HiSparse C4 device pool uses index translation that is
+    # not yet DCP-aware in the fused C++ kernels; only the Triton fallback
+    # write path supports DCP.
+    if server_args.enable_hisparse:
+        logger.warning(
+            "DeepSeekV4 DCP + enable_hisparse falls back to Triton write "
+            "path for KV writes; expect throughput regression vs. fused C++."
+        )
+
+    logger.info(
+        f"DeepSeekV4 DCP enabled: dcp_size={server_args.dcp_size}, "
+        f"attn_tp={attn_tp}"
+    )
+
 
 def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
     """Validate DeepSeek V4 context-parallel configuration."""
```

#### indexer.py DCP shard（节选）

```diff
@@ -9,6 +9,8 @@
 from sglang.jit_kernel.dsv4 import (
     fused_q_indexer_rope_hadamard_fp4_quant,
     fused_q_indexer_rope_hadamard_quant,
+    merge_dcp_topk_candidates_512,
+    topk_candidates_512,
     topk_transform_512,
     topk_transform_512_v2,
 )
@@ -42,9 +44,8 @@
     from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
     from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 
-
 FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
-
+FP8_MAX = torch.finfo(FP8_DTYPE).max
 
 IndexerQuery: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
 
@@ -558,6 +559,217 @@ def _forward_nonpaged_indexer(
             max_seqlen_k=plan.max_seqlen_k,
         )
 
+    def _try_forward_dcp_sharded_c4_indexer(
+        self,
+        *,
+        q: torch.Tensor,
+        q_indexer: torch.Tensor,
+        weights: torch.Tensor,
+        logits_fn: Any,
+        use_tilelang: bool,
+        use_aiter: bool,
+        c4_indexer: C4Indexer,
+        forward_batch: ForwardBatch,
+        token_to_kv_pool: DeepSeekV4TokenToKVPool,
+        indexer_metadata: PagedIndexerMetadata,
+        page_table: torch.Tensor,
+        c4_seq_lens: torch.Tensor,
+        local_page_table: Optional[torch.Tensor],
+        local_c4_seq_lens: Optional[torch.Tensor],
+        c4_sparse_page_indices: torch.Tensor,
+        raw_indices: Optional[torch.Tensor],
+    ) -> bool:
+        """Score interleaved logical C4-page shards and merge local top-k.
+
+        The P0 path keeps the indexer cache replicated, but each DCP rank sends
+        only logical pages ``rank, rank + world_size, ...`` to the existing
+        paged-MQA logits kernel. A C4 page contains 64 compressed items (256 raw
+        tokens), so shard boundaries preserve the compressor's 8-token window.
+        """
+
+        if not envs.SGLANG_DSV4_DCP_SHARD_C4_INDEXER.get():
+            return False
+        if c4_indexer.use_fp4_indexer:
+            return False
+        if is_hip():
+            return False
+        if self.hisparse_coordinator is not None:
+            return False
+        if not forward_batch.forward_mode.is_decode():
+            return False
+        if self.debug_use_external_c4_sparse_indices:
+            return False
+        if not isinstance(q_indexer, torch.Tensor):
+            return False
+        if q_indexer.dim() != 3 or q_indexer.shape[-1] != c4_indexer.head_dim:
+            return False
+        if local_page_table is None or local_c4_seq_lens is None:
+            return False
+
+        from sglang.srt.distributed.parallel_state import get_dcp_group_no_assert
+
+        dcp_group = get_dcp_group_no_assert()
+        if dcp_group is None or dcp_group.world_size <= 1:
+            return False
+        if (
+            indexer_metadata.dcp_world_size != dcp_group.world_size
+            or indexer_metadata.dcp_rank != dcp_group.rank_in_group
+        ):
+            return False
+
+        batch_size = q_indexer.shape[0]
+        if batch_size == 0:
+            c4_sparse_page_indices.fill_(-1)
+            if raw_indices is not None:
+                raw_indices.fill_(-1)
+            return True
+
+        c4_page_size = indexer_metadata.c4_page_size
+        topk = c4_sparse_page_indices.shape[1]
+        if topk == 0:
+            return True
+
+        device = q_indexer.device
+        world_size = dcp_group.world_size
+        rank = dcp_group.rank_in_group
+        max_global_seq_len = indexer_metadata.max_c4_seq_len
+        max_global_pages = (max_global_seq_len + c4_page_size - 1) // c4_page_size
+        max_local_pages = max(
+            0, (max_global_pages + world_size - 1 - rank) // world_size
+        )
+        local_page_table = local_page_table[:, :max_local_pages]
+        local_seq_lens = local_c4_seq_lens.view(-1).to(torch.int32).contiguous()
+
+        if local_page_table.shape[1] == 0:
+            # A short request can leave a high DCP rank with no logical page.
+            # Keep all ranks in the candidate collectives without invoking the
+            # paged-MQA kernel on an empty page table.
+            local_page_table = page_table[:, :1].contiguous()
+            local_logits = torch.full(
+                (batch_size, c4_page_size),
+                float("-inf"),
+                dtype=torch.float32,
+                device=device,
+            )
+        else:
+            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
+                layer_id=c4_indexer.layer_id,
+            )

# ... truncated 83 lines; see upstream PR ...
```

#### backend.py（大幅，节选）

```diff
@@ -33,6 +33,11 @@
     BuildPageTablePositions,
     ExpandPrefillCausally,
 )
+from sglang.srt.distributed.parallel_state import (
+    get_dcp_group,
+    get_dcp_rank,
+    get_dcp_world_size,
+)
 from sglang.srt.environ import envs
 from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
 from sglang.srt.layers.attention.dsv4.compressor_v2 import (
@@ -51,6 +56,11 @@
     SparsePrefillChunkCache,
     SparsePrefillWorkspace,
 )
+from sglang.kernels.ops.attention.utils import (
+    cp_lse_a2a_out_rs,
+    cp_lse_ag_out_reduce_scatter,
+    cp_lse_ag_out_rs,
+)
 from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
 from sglang.srt.runtime_context import get_parallel
@@ -89,6 +99,17 @@
 PAGE_INDEX_ALIGNED_SIZE = 64
 
 
+def _expand_dcp_local_kv_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
+    mask = mask.reshape(-1)
+    if mask.numel() == target_len:
+        return mask
+    if mask.numel() == 0:
+        return torch.zeros(target_len, device=mask.device, dtype=torch.bool)
+    if mask.numel() < target_len:
+        return F.pad(mask, (0, target_len - mask.numel()), value=False)
+    return mask[:target_len]
+
+
 def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
     # IDLE is a real per-DP-rank mode. Do not let a stale _original_forward_mode
     # from a reused/padded ForwardBatch turn an empty rank into TARGET_VERIFY.
@@ -179,6 +200,9 @@ class DSV4AttnMetadata:
     c128_out_loc: Optional[torch.Tensor] = None
     c128_page_indices: Optional[torch.Tensor] = None
     c128_topk_lengths_clamp1: Optional[torch.Tensor] = None
+    dcp_swa_has_local_kv: Optional[torch.Tensor] = None
+    dcp_c128_has_local_kv: Optional[torch.Tensor] = None
+    dcp_has_local_kv: Optional[torch.Tensor] = None
 
     c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
     c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
@@ -223,6 +247,9 @@ def copy_(self, other: DSV4AttnMetadata) -> None:
                 "c4_sparse_topk_lengths",
                 "c4_sparse_page_indices",
                 "c4_sparse_raw_indices",
+                "dcp_swa_has_local_kv",
+                "dcp_c128_has_local_kv",
+                "dcp_has_local_kv",
             ],
             assign_fields=[
                 # Recomputed by the recorded init_forward_metadata_in_graph op
@@ -248,6 +275,9 @@ def refresh_for_breakable_cuda_graph_replay_(self, other: DSV4AttnMetadata) -> N
             "c4_topk_lengths_raw",
             "c4_topk_lengths_clamp1",
             "c4_sparse_topk_lengths",
+            "dcp_swa_has_local_kv",
+            "dcp_c128_has_local_kv",
+            "dcp_has_local_kv",
         ]
         reference_assign_fields = [
             "page_table",
@@ -300,9 +330,75 @@ def init_compression_metadata(self):
             compute_page_indices=True,
         )
 
+        self.apply_dcp_local_kv_indices()
         self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
         self.swa_page_indices = _pad_last_dim(self.swa_page_indices)
 
+    @staticmethod
+    def _compact_dcp_local_indices(
+        indices: torch.Tensor,
+        dcp_world_size: int,
+        dcp_rank: int,
+    ) -> Tuple[torch.Tensor, torch.Tensor]:
+        valid_mask = indices >= 0
+        local_mask = valid_mask & ((indices % dcp_world_size) == dcp_rank)
+        physical_indices = torch.where(
+            local_mask, indices // dcp_world_size, torch.full_like(indices, -1)
+        )
+
+        col = torch.arange(indices.shape[-1], device=indices.device).view(
+            *([1] * (indices.dim() - 1)), -1
+        )
+        order_key = torch.where(local_mask, col, col + indices.shape[-1])
+        order = torch.argsort(order_key, dim=-1)
+        compacted = torch.gather(physical_indices, dim=-1, index=order)
+
+        local_lengths = local_mask.sum(dim=-1).to(torch.int32)

# ... truncated 103 lines; see upstream PR ...
```

### 本地核对

- [ ] 目标树已包含 #30416 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #33250 — [Bugfix] Fix DeepSeek-V4 non-EP TBO for attention TP > 1

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/33250
- **分类**：正确性
- **Author**：@mikekg  ·  diffstat `+157/-34`
- **一句话**：修 non-EP TBO + attn TP>1（依赖 #31700）

### 做什么

真正修 non-EP TBO + attention TP>1 的通信；依赖 #31700 的 gather 语义。含 `dp_attention.py` 与模型侧 TBO collectives 调整。

### 开关 / 启用方式

无新用户开关；修复路径。依赖：#31700。

### 关键文件 / 符号（异地核对点）

TBO collectives / `dp_attention.py` 改动；单测 `test_deepseek_v4_tbo_collectives.py`

`dp_attention.py`, `deepseek_v4.py`, `deepseek_v4_nextn.py`, tests

### 关键 diff

#### dp_attention.py 关键 hunk

```diff
diff --git a/python/sglang/srt/layers/dp_attention.py b/python/sglang/srt/layers/dp_attention.py
index c1ea7893f277..626e3de1a63d 100644
--- a/python/sglang/srt/layers/dp_attention.py
+++ b/python/sglang/srt/layers/dp_attention.py
@@ -932,15 +932,33 @@ def dp_reduce_scatterv_async(
     sizes: List[int],
     event_key=("combine", 0),
 ) -> torch.cuda.Event:
-    """Launch the variable-length reduce_scatterv (combine) on the shared DP TBO
-    comm stream; re-record + return a PERSISTENT event (keyed by `event_key`).
-    Matches the gatherv (SUM_LEN) path."""
+    """Launch the TBO variable-length combine on the shared comm stream.
+
+    ``sizes`` describes one shard per full-TP rank. For attention TP > 1,
+    reduce-scatter first returns that rank's shard, then an attention-TP
+    all-gather reconstructs the replicated DP-local output.
+    """
     comm = get_dp_tbo_comm_stream()
     compute = torch.cuda.current_stream()
     ev = _tbo_event(event_key)
+    attn_tp_size = get_attn_tensor_model_parallel_world_size()
+    if attn_tp_size == 1:
+        output_shard = output_local
+    else:
+        shard_rows = sizes[get_tp_group().rank_in_group]
+        assert output_local.shape[0] == shard_rows * attn_tp_size
+        output_shard = get_tbo_persistent_buffer(
+            ("combine_shard", event_key),
+            shard_rows,
+            output_local.shape[1],
+            output_local.dtype,
+            output_local.device,
+        )
     with torch.cuda.stream(comm):
         comm.wait_stream(compute)
-        get_tp_group().reduce_scatterv(global_tokens, output=output_local, sizes=sizes)
+        get_tp_group().reduce_scatterv(global_tokens, output=output_shard, sizes=sizes)
+        if attn_tp_size > 1:
+            get_attn_tp_group().all_gather_into_tensor(output_local, output_shard)
         ev.record(comm)
     return ev
```

#### deepseek_v4.py 关键 hunk

```diff
diff --git a/python/sglang/srt/models/deepseek_v4.py b/python/sglang/srt/models/deepseek_v4.py
index 65dd55549924..7e033d8d9a1c 100644
--- a/python/sglang/srt/models/deepseek_v4.py
+++ b/python/sglang/srt/models/deepseek_v4.py
@@ -63,7 +63,6 @@
     _tbo_event,
     attn_tp_all_gather,
     attn_tp_all_reduce,
-    dp_gather_partial,
     dp_gather_replicate,
     dp_reduce_scatter_tensor,
     dp_reduce_scatterv_async,
@@ -224,6 +223,30 @@ def _is_fused_mhc_post_pre_enabled() -> bool:
 # gather instead of on the gathered global buffer. Requires
 # SGLANG_SHARED_EXPERT_TP1=1 (replicated shared expert). Default OFF.
 _SHARED_EXPERT_LOCAL = get_bool_env_var("SGLANG_DP_SHARED_EXPERT_LOCAL")
+
+
+def _tbo_collective_sizes(
+    rank_sizes: List[int], attn_tp_size: int
+) -> Tuple[List[int], List[int]]:
+    if attn_tp_size < 1 or len(rank_sizes) % attn_tp_size != 0:
+        raise ValueError(f"Invalid TBO topology: {len(rank_sizes)=}, {attn_tp_size=}")
+
+    dp_sizes = []
+    tp_sizes = []
+    for start in range(0, len(rank_sizes), attn_tp_size):
+        replicas = rank_sizes[start : start + attn_tp_size]
+        if any(size != replicas[0] for size in replicas):
+            raise ValueError(f"TBO token counts differ within attention TP: {replicas}")
+        if replicas[0] % attn_tp_size != 0:
+            raise ValueError(
+                f"TBO token count {replicas[0]} is not divisible by "
+                f"attention TP size {attn_tp_size}"
+            )
+        dp_sizes.append(replicas[0])
+        tp_sizes.extend([replicas[0] // attn_tp_size] * attn_tp_size)
+    return dp_sizes, tp_sizes
+
+
 _is_gfx95_supported = is_gfx95_supported()
 _is_gfx942_supported = is_gfx942_supported()
 
@@ -1854,7 +1877,9 @@ def _run_moe_ffn_dp_sync(
             )
             if _do_shared_local and local_hidden_states.shape[0] > 0:
                 _shared_local = self.mlp._forward_shared_experts(local_hidden_states)
-            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
+            # self_attn has already reduced across attention TP, so these hidden
+            # states are replicated and must not be summed by a partial gather.
+            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
         _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
         if _use_tp_attn_a2a_scatter:
             s, r = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
@@ -2036,18 +2061,15 @@ def op_mhc_postprocess(self, state):
     # op_dispatch/op_combine. op_mhc_* and op_attn are reused (local hidden).
     # ------------------------------------------------------------------
     def op_gather_a(self, state):
-        # Launch the all_gatherv (local hidden -> global buffer) + the input_ids
-        # replicate-gather on the shared comm stream; record an event.
+        # Launch the all_gatherv (local hidden -> global buffer) on the shared
+        # comm stream; record an event.
         fb = state.forward_batch
         local = state.pop("hidden_states_mlp_input")  # LOCAL [M_local, hidden]
-        # Shared-expert-local: compute on LOCAL hidden before the gather; added
-        # back after the combine (same as the non-fused forward). Skipped in the
-        # global MoE via skip_shared_experts.
-        do_shared_local = (
-            _SHARED_EXPERT_LOCAL
-            and getattr(self.mlp, "shared_experts", None) is not None
-            and getattr(self.mlp, "_shared_expert_tp1", False)
-        )
+        # A TP1 shared expert is replicated, so compute it before the gather and
+        # add it after the reducing combine.
+        do_shared_local = getattr(
+            self.mlp, "shared_experts", None
+        ) is not None and getattr(self.mlp, "_shared_expert_tp1", False)
         state.do_shared_local = do_shared_local
         state.shared_local = (
             self.mlp._forward_shared_experts(local)
@@ -2063,14 +2085,22 @@ def op_gather_a(self, state):
         global_hidden = get_tbo_persistent_buffer(
             ("gh", sub), global_rows, local.shape[1], local.dtype, local.device
         )
+        attn_tp_size = get_parallel().attn_tp_size
+        local_shard = local.tensor_split(attn_tp_size)[
+            get_parallel().attn_tp_rank
+        ].contiguous()
+        tp_sizes = fb._tbo_tp_sizes
+        assert local_shard.shape[0] == tp_sizes[get_tp_group().rank_in_group]
         comm = get_dp_tbo_comm_stream()
         compute = torch.cuda.current_stream()
         with torch.cuda.stream(comm):
             comm.wait_stream(compute)
-            dp_gather_partial(global_hidden, local, fb)
+            get_tp_group().all_gatherv(
+                local_shard, sizes=tp_sizes, output=global_hidden
+            )
             state.gather_event = _tbo_event(("gather", sub))
             state.gather_event.record(comm)
-        state.gather_keepalive = local
+        state.gather_keepalive = local_shard
         state.global_hidden = global_hidden
 
     def op_gather_b(self, state):
@@ -2108,7 +2138,7 @@ def op_combine_a(self, state):
         state.combine_event = dp_reduce_scatterv_async(
             local_out,
             global_out,
-            get_dp_global_num_tokens(),
+            state.forward_batch._tbo_tp_sizes,
             event_key=("combine", state.tbo_subbatch_index),
         )
         state.local_out = local_out
@@ -2240,6 +2270,8 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:
         DP-attention preparer allows it (mori `normal` mode permits prefill
         TBO). We additionally restrict to: prefill (EXTEND), single PP, and the
         non-CP path, which is the only case the DSV4 op strategy implements.
+        The non-EP strategy uses variable-size DP collectives and therefore
+        requires CP1, multi-rank DP, and SUM_LEN padding.
         """
         from sglang.srt.layers.moe import is_tbo_enabled
 
@@ -2252,6 +2284,15 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:
             # should enter the prefill TBO strategy.
             and forward_batch.global_forward_mode.is_extend_without_speculative()
             and not dsa_use_prefill_cp(forward_batch)
+            and (
+                not get_moe_a2a_backend().is_none()
+                or (
+                    get_parallel().attn_cp_size == 1
+                    and get_parallel().attn_dp_size > 1
+                    and forward_batch.dp_padding_mode is not None
+                    and forward_batch.dp_padding_mode.is_sum_len()
+                )
+            )
             and self.pp_group.world_size == 1
         )
 
@@ -2294,6 +2335,7 @@ def _forward_layers_tbo(

# ... truncated 66 lines ...
```

### 本地核对

- [ ] 目标树已包含 #33250 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #33217 — Fix non-EP DeepSeek-V4 TBO for attention TP > 1

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/33217
- **分类**：正确性
- **Author**：@Oxygen56  ·  diffstat `+62/-0`
- **一句话**：对非法 non-EP TBO+attnTP>1 做 policy guard

### 做什么

在修好 #33250 之前/之外，先禁止非 EP 且 attn_tp_size>1 时走 TBO，避免静默错误。

### 开关 / 启用方式

无新开关；改 `_can_run_tbo` 策略。

### 关键文件 / 符号（异地核对点）

`DeepseekV4Model._can_run_tbo`；单测 `test_deepseek_v4_tbo_policy.py`

`deepseek_v4.py`, `test/.../test_deepseek_v4_tbo_policy.py`

### 关键 diff

#### 完整 diff

```diff
diff --git a/python/sglang/srt/models/deepseek_v4.py b/python/sglang/srt/models/deepseek_v4.py
index 52fb8c098d8b..4d6080869bb4 100644
--- a/python/sglang/srt/models/deepseek_v4.py
+++ b/python/sglang/srt/models/deepseek_v4.py
@@ -2245,6 +2245,12 @@ def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:
             and forward_batch.global_forward_mode.is_extend_without_speculative()
             and not dsa_use_prefill_cp(forward_batch)
             and self.pp_group.world_size == 1
+            # The non-EP path gathers variable-length metadata across the full
+            # TP group and is only valid when each DP shard has one attention
+            # rank. EP/Mori TBO uses a different communication strategy.
+            and (
+                not get_moe_a2a_backend().is_none() or get_parallel().attn_tp_size == 1
+            )
         )
 
     def _forward_layers_tbo(
diff --git a/test/registered/unit/models/test_deepseek_v4_tbo_policy.py b/test/registered/unit/models/test_deepseek_v4_tbo_policy.py
new file mode 100644
index 000000000000..f45457133c86
--- /dev/null
+++ b/test/registered/unit/models/test_deepseek_v4_tbo_policy.py
@@ -0,0 +1,56 @@
+"""Unit tests for the DeepSeek-V4 two-batch-overlap policy."""
+
+import unittest
+from types import SimpleNamespace
+from unittest.mock import patch
+
+import sglang.srt.models.deepseek_v4 as deepseek_v4
+from sglang.test.ci.ci_register import register_cpu_ci
+from sglang.test.test_utils import CustomTestCase
+
+register_cpu_ci(est_time=2, suite="base-a-test-cpu")
+
+
+class TestDeepseekV4TboPolicy(CustomTestCase):
+    def _can_run_tbo(self, *, non_ep: bool, attn_tp_size: int) -> bool:
+        model = SimpleNamespace(pp_group=SimpleNamespace(world_size=1))
+        forward_batch = SimpleNamespace(
+            can_run_tbo=True,
+            tbo_children=[object(), object()],
+            global_forward_mode=SimpleNamespace(
+                is_extend_without_speculative=lambda: True
+            ),
+        )
+        backend = SimpleNamespace(is_none=lambda: non_ep)
+
+        with (
+            patch(
+                "sglang.srt.layers.moe.is_tbo_enabled",
+                return_value=True,
+            ),
+            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=False),
+            patch.object(
+                deepseek_v4,
+                "get_moe_a2a_backend",
+                return_value=backend,
+            ),
+            patch.object(
+                deepseek_v4,
+                "get_parallel",
+                return_value=SimpleNamespace(attn_tp_size=attn_tp_size),
+            ),
+        ):
+            return deepseek_v4.DeepseekV4Model._can_run_tbo(model, forward_batch)
+
+    def test_non_ep_tbo_falls_back_with_multiple_attention_tp_ranks(self):
+        self.assertFalse(self._can_run_tbo(non_ep=True, attn_tp_size=4))
+
+    def test_non_ep_tbo_remains_enabled_with_one_attention_tp_rank(self):
+        self.assertTrue(self._can_run_tbo(non_ep=True, attn_tp_size=1))
+
+    def test_ep_tbo_is_not_restricted_by_attention_tp_size(self):
+        self.assertTrue(self._can_run_tbo(non_ep=False, attn_tp_size=4))
+
+
+if __name__ == "__main__":
+    unittest.main()
```

### 本地核对

- [ ] 目标树已包含 #33217 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #31700 — Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/31700
- **分类**：正确性
- **Author**：@mikekg  ·  diffstat `+14/-4`
- **一句话**：dp_gather_partial → dp_gather_replicate（attn-TP 副本）

### 做什么

DP-attention 下，attention-TP 组内 hidden/token ids 已是 replicate；误用 `dp_gather_partial` 会把副本当 partial 累加，导致 attn_tp>1 数值错误。改 `dp_gather_replicate`，并对 input_ids gather 做 clone（避免 in-place zero）。

### 开关 / 启用方式

无新开关；行为修正（Breaking for buggy path）。

### 关键文件 / 符号（异地核对点）

`dp_gather_replicate` 替换 `_run_moe_ffn_dp_sync` / forward 中的 `dp_gather_partial`；`deepseek_v4_nextn.py` 同步。

`deepseek_v4.py`, `deepseek_v4_nextn.py`

### 关键 diff

#### 完整 diff（很小）

```diff
diff --git a/python/sglang/srt/models/deepseek_v4.py b/python/sglang/srt/models/deepseek_v4.py
index 65dd55549924..8916881b6623 100644
--- a/python/sglang/srt/models/deepseek_v4.py
+++ b/python/sglang/srt/models/deepseek_v4.py
@@ -1854,7 +1854,9 @@ def _run_moe_ffn_dp_sync(
             )
             if _do_shared_local and local_hidden_states.shape[0] > 0:
                 _shared_local = self.mlp._forward_shared_experts(local_hidden_states)
-            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
+            # self_attn has already reduced across attention TP, so these hidden
+            # states are replicated and must not be summed by a partial gather.
+            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
         _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
         if _use_tp_attn_a2a_scatter:
             s, r = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
@@ -2372,7 +2374,10 @@ def forward(
             )
             # Token ids are replicated within an attention-TP group. Use replicate
             # gather here to avoid summing duplicated ids when attention_tp_size > 1.
-            dp_gather_replicate(input_ids_global, input_ids[:, None], forward_batch)
+            # Clone because the MAX_LEN gather may zero its local input in place.
+            dp_gather_replicate(
+                input_ids_global, input_ids[:, None].clone(), forward_batch
+            )
             input_ids_global = input_ids_global.squeeze(-1)
         else:
             input_ids_global = input_ids
diff --git a/python/sglang/srt/models/deepseek_v4_nextn.py b/python/sglang/srt/models/deepseek_v4_nextn.py
index f8d03952478c..bfadeff15a70 100644
--- a/python/sglang/srt/models/deepseek_v4_nextn.py
+++ b/python/sglang/srt/models/deepseek_v4_nextn.py
@@ -14,7 +14,7 @@
     is_dsa_prefill_cp_round_robin_split,
 )
 from sglang.srt.layers.dp_attention import (
-    dp_gather_partial,
+    dp_gather_replicate,
     get_global_dp_buffer_len,
     is_dp_attention_enabled,
 )
@@ -162,7 +162,12 @@ def forward(
                 dtype=input_ids.dtype,
                 device=input_ids.device,
             )
-            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
+            # Token IDs are replicated within an attention-TP group. Use replicate
+            # gather to avoid summing duplicated IDs when attention_tp_size > 1.
+            # Clone because the MAX_LEN gather may zero its local input in place.
+            dp_gather_replicate(
+                input_ids_global, input_ids[:, None].clone(), forward_batch
+            )
             input_ids_global = input_ids_global.squeeze(-1)
         else:
             input_ids_global = input_ids
```

### 本地核对

- [ ] 目标树已包含 #31700 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## PR #30885 — [Feature] Support DeepSeek V4 in PDMux

- **Upstream 状态**：❌ **未合入 (OPEN)**
- **URL**：https://github.com/sgl-project/sglang/pull/30885
- **分类**：产品形态
- **Author**：@shipiyouniao  ·  diffstat `+795/-44`
- **一句话**：DSV4 支持 PDMux（同进程 P/D mux）

### 做什么

让 DeepSeek-V4 跑在 PDMux（同进程 Prefill/Decode multiplexing）：scheduler/policy、multiplexing_mixin、decode cuda graph、模型 split-prefill、topk dispatch。

### 开关 / 启用方式

PDMux 既有开关（`--enable-pdmux` / multiplex 相关，以目标树为准）；本 PR 主要是 DSV4 适配而非新发明 PDMux。

### 关键文件 / 符号（异地核对点）

`multiplexing_mixin` DSV4 分支、`split_prefill`、scheduler PDMux 路径、decode cuda graph runner

- `multiplex/multiplexing_mixin.py`
- `managers/scheduler.py`, `schedule_policy.py`
- `models/deepseek_v4.py`
- `decode_cuda_graph_runner.py`
- unit tests: `test_pdmux_*`, `test_deepseek_v4_split_prefill.py`

### 关键 diff

#### multiplexing_mixin.py（节选）

```diff
@@ -30,14 +30,13 @@
 
 
 class SchedulerMultiplexMixin:
-
     def init_pdmux(self: Scheduler):
         # The current split prefill batch
         self.split_prefill_batch: Optional[ScheduleBatch] = None
 
         # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
         self.pdmux_config = load_pdmux_config(self.server_args.pdmux_config_path)
-        initialize_stream_groups(self.gpu_id, self.pdmux_config)
+        initialize_stream_groups(self.ps.gpu_id, self.pdmux_config)
         self.stream_groups = get_stream_groups()
         self.sm_counts = get_sm_counts()
         self.real_sm_group_num = len(self.stream_groups)
@@ -94,6 +93,79 @@ def update_split_prefill_batch(self: Scheduler, sm_count: int) -> bool:
             return True
         return False
 
+    def _get_split_forward_count(self: Scheduler) -> int:
+        remaining_layers = (
+            self.model_config.num_hidden_layers - self.split_prefill_batch.split_index
+        )
+
+        # Splitting only benefits decode work that can run between prefill
+        # intervals. Without decode work, finish prefill in one model call to
+        # avoid repeating the full scheduler/model-runner setup per layer.
+        if self.running_batch is None or self.running_batch.is_empty():
+            return remaining_layers
+
+        if self.split_prefill_batch.extend_num_tokens <= 0:
+            return remaining_layers
+
+        forward_count = max(
+            1,
+            self.pdmux_config.split_forward_token_budget
+            // self.split_prefill_batch.extend_num_tokens,
+        )
+        return min(forward_count, remaining_layers)
+
+    def _get_pdmux_prefill_token_limit(
+        self: Scheduler, max_prefill_tokens: int
+    ) -> Optional[int]:
+        hard_limit = self.pdmux_max_prefill_plan_tokens
+        if not (self.enable_pdmux and hard_limit is not None):
+            return None
+
+        # PrefillAdder accounts input tokens in page-aligned units. Align the
+        # backend's raw-token limit down so every accepted request can consume
+        # the admission budget instead of remaining in the waiting queue.
+        limit = min(max_prefill_tokens, hard_limit)
+        return limit - limit % self.page_size
+
+    def _get_prefill_admission_config(
+        self: Scheduler, max_prefill_tokens: int
+    ) -> tuple[int, bool]:
+        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
+            self, max_prefill_tokens
+        )
+        if effective_limit is None:
+            return max_prefill_tokens, False
+        return effective_limit, True
+
+    def _get_max_req_input_len(self: Scheduler, max_req_input_len: int) -> int:
+        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
+            self, self.max_prefill_tokens
+        )
+        if effective_limit is None:
+            return max_req_input_len
+        # Request validation rejects lengths >= max_req_input_len.
+        return min(max_req_input_len, effective_limit + 1)
+
+    def _merge_finished_prefill_batch(
+        self: Scheduler,
+        prefill_result,
+        prefill_stream,
+        decode_stream,
+    ) -> None:
+        self.process_batch_result(self.split_prefill_batch, prefill_result)
+        if self.running_batch and not self.running_batch.is_empty():
+            self.running_batch.merge_batch(self.split_prefill_batch)
+        else:
+            self.running_batch = self.split_prefill_batch
+
+        self.split_prefill_batch = None
+
+        # merge_batch enqueues tensor concatenations on the prefill stream.
+        # The next loop prepares decode before the stream-group synchronization,
+        # so publish an explicit dependency before decode indexes those tensors.
+        merge_done = prefill_stream.record_event()
+        decode_stream.wait_event(merge_done)
+
     @torch.inference_mode()
     def event_loop_pdmux(self: Scheduler):
         """A scheduler loop for pd multiplexing."""
@@ -159,15 +231,7 @@ def event_loop_pdmux(self: Scheduler):
                     and not wait_prefill_kernel_done
                 ):
                     prefill_done = True

# ... truncated 34 lines; see upstream PR ...
```

#### scheduler.py

```diff
@@ -879,6 +879,19 @@ def init_model_worker(self):
             _,
             _,
         ) = self.tp_worker.get_worker_info()
+        self.pdmux_max_prefill_plan_tokens = (
+            model_runner.attn_backend.max_prefill_plan_tokens
+            if self.enable_pdmux
+            else None
+        )
+        if self.pdmux_max_prefill_plan_tokens is not None:
+            logger.info(
+                "PDMux prefill planner hard limit: %s tokens",
+                self.pdmux_max_prefill_plan_tokens,
+            )
+        # Keep accepted requests within any hard PDMux backend limit. PDMux
+        # cannot fall back to chunked prefill for an oversized first request.
+        self.max_req_input_len = self._get_max_req_input_len(self.max_req_input_len)
         # DFlash auto-enables the legacy formula; other workloads opt in via
         # --min-free-slots-delay. Built independently of the prefill delayer.
         self.min_free_slots_delayer: Optional[MinFreeSlotsDelayer] = None
@@ -2907,19 +2920,28 @@ def _get_new_batch_prefill_raw(
                 chunked_prefill_size = dynamic_size
 
         # Prefill policy
+        # DeepSeek V4 compressor plans encode ragged token ids as uint16. PDMux
+        # cannot use chunked prefill, so admission must keep the complete batch
+        # within that hard planner limit instead of treating max_prefill_tokens
+        # as a soft budget for the first request.
+        max_prefill_tokens, enforce_max_prefill_tokens = (
+            self._get_prefill_admission_config(self.max_prefill_tokens)
+        )
+
         adder = PrefillAdder(
             self.page_size,
             self.tree_cache,
             self.token_to_kv_pool_allocator,
             running_batch,
             self.new_token_ratio_tracker.current,
-            self.max_prefill_tokens,
+            max_prefill_tokens,
             chunked_prefill_size,
             running_bs if self.is_mixed_chunk else 0,
             self.priority_scheduling_preemption_threshold,
             max_prefill_bs=self.max_prefill_bs,
             max_running_requests=self.max_running_requests,
             prefill_max_requests=self.server_args.prefill_max_requests,
+            enforce_max_prefill_tokens=enforce_max_prefill_tokens,
             prefill_delayer_single_pass=prefill_delayer_single_pass,
             dllm_config=self.dllm_config,
             waiting_queue_len=len(self.waiting_queue),
```

#### schedule_policy.py

```diff
@@ -453,6 +453,7 @@ def __init__(
         max_prefill_bs: int = 0,
         max_running_requests: Optional[int] = None,
         prefill_max_requests: Optional[int] = None,
+        enforce_max_prefill_tokens: bool = False,
         prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
         dllm_config: Optional[DllmConfig] = None,
         waiting_queue_len: int = 0,
@@ -541,6 +542,7 @@ def __init__(
         self.max_running_requests = max_running_requests
         self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()
         self.prefill_max_requests = prefill_max_requests
+        self.enforce_max_prefill_tokens = enforce_max_prefill_tokens
         self.prefill_delayer_single_pass = prefill_delayer_single_pass
         self.max_prefill_bs = max_prefill_bs
         # Snapshot of scheduler waiting_queue length at the start of this
@@ -859,11 +861,18 @@ def _lock_node(self, last_node: TreeNode):
             else:
                 self.tree_cache.dec_lock_ref(last_node)
 
+    def _prefill_token_budget_exceeded(self, input_tokens: int) -> bool:
+        if self.enforce_max_prefill_tokens:
+            return input_tokens > self.rem_input_tokens
+        return bool(self.can_run_list) and input_tokens >= self.rem_input_tokens
+
     def add_one_req_ignore_eos(self, req: Req):
         cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
             req.prefix_indices
         )
         paged_input = self.ceil_paged_tokens(cand_extend_input_len)
+        if self._prefill_token_budget_exceeded(paged_input):
+            return AddReqResult.OTHER
         # Shared Mamba pool: fold the new mamba state's shared-gap cost into the
         # budget gate so admission can't over-commit (0 for baseline / non-Mamba).
         paged_input += self._mamba_gap_budget_for_req(req)
@@ -1028,10 +1037,8 @@ def add_one_req(
             if swa_needed >= self.rem_swa_tokens:
                 return AddReqResult.NO_TOKEN
 
-        if (
-            self.rem_chunk_tokens is None
-            and len(self.can_run_list) != 0
-            and real_input_tokens >= self.rem_input_tokens
+        if self.rem_chunk_tokens is None and self._prefill_token_budget_exceeded(
+            real_input_tokens
         ):
             # If without chunked prefill:
             # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
@@ -1066,10 +1073,8 @@ def add_one_req(
                 len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
             )
 
-            if (
-                self.rem_chunk_tokens is None
-                and len(self.can_run_list) != 0
-                and input_tokens >= self.rem_input_tokens
+            if self.rem_chunk_tokens is None and self._prefill_token_budget_exceeded(
+                input_tokens
             ):
                 # If without chunked prefill:
                 # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
```

#### deepseek_v4.py（节选）

```diff
@@ -2306,6 +2306,109 @@ def forward(
 
         return hidden_states, pre_hc_head
 
+    def forward_split_prefill(
+        self,
+        input_ids: torch.Tensor,
+        positions: torch.Tensor,
+        forward_batch: ForwardBatch,
+        split_interval: Tuple[int, int],
+        input_embeds: Optional[torch.Tensor] = None,
+    ):
+        start, end = split_interval
+
+        if start == 0:
+            hidden_states = self.embed_tokens(input_ids)
+            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
+
+            if get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none():
+                input_ids_global = torch.empty(
+                    (get_global_dp_buffer_len(), 1),
+                    dtype=input_ids.dtype,
+                    device=input_ids.device,
+                )
+                dp_gather_replicate(input_ids_global, input_ids[:, None], forward_batch)
+                input_ids_global = input_ids_global.squeeze(-1)
+            else:
+                input_ids_global = input_ids
+
+            if dsa_use_prefill_cp(forward_batch):
+                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
+                positions = cp_split_and_rebuild_position(forward_batch, positions)
+                input_ids = cp_round_robin_input_ids(input_ids)
+                input_ids_global = input_ids
+
+            for attr in ("freqs_cis_c4", "freqs_cis_c128"):
+                if hasattr(forward_batch, attr):
+                    delattr(forward_batch, attr)
+
+            # Cross-layer mHC fusion defers hc_post until the next layer, so the
+            # pending tensors must survive scheduler yields between split calls.
+            forward_batch.hidden_states = hidden_states
+            forward_batch.model_specific_states = {
+                "positions": positions,
+                "input_ids": input_ids,
+                "input_ids_global": input_ids_global,
+                "prev_residual": None,
+                "prev_post": None,
+                "prev_comb": None,
+            }
+
+        states = forward_batch.model_specific_states
+        hidden_states = forward_batch.hidden_states
+        prev_residual = states["prev_residual"]
+        prev_post = states["prev_post"]
+        prev_comb = states["prev_comb"]
+        last_layer = None
+
+        for i in range(start, end):
+            layer = self.layers[i]
+            last_layer = layer
+            ctx = (
+                nullcontext()
+                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
+                else get_global_expert_distribution_recorder().with_current_layer(i)
+            )
+            with ctx:
+                hidden_states, prev_residual, prev_post, prev_comb = layer(
+                    positions=states["positions"],
+                    hidden_states=hidden_states,
+                    forward_batch=forward_batch,
+                    input_ids=states["input_ids"],
+                    input_ids_global=states["input_ids_global"],
+                    prev_residual=prev_residual,
+                    prev_post=prev_post,
+                    prev_comb=prev_comb,
+                )
+
+        forward_batch.hidden_states = hidden_states
+        states["prev_residual"] = prev_residual
+        states["prev_post"] = prev_post
+        states["prev_comb"] = prev_comb
+
+        if end != self.end_layer:
+            return None
+
+        if self.use_fused_mhc_post_pre and last_layer is not None:
+            hidden_states = last_layer.hc_post(
+                hidden_states, prev_residual, prev_post, prev_comb
+            )
+
+        if dsa_use_prefill_cp(forward_batch):
+            hidden_states = cp_all_gather_rerange_output(
+                hidden_states,
+                self.cp_size,
+                forward_batch,
+                torch.cuda.current_stream(),
+            )
+
+        pre_hc_head = hidden_states.flatten(1)
+        hidden_states = self.hc_head(
+            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
+        )
+        hidden_states = self.norm(hidden_states)
+        forward_batch.hidden_states = hidden_states
+        return hidden_states, pre_hc_head
+
 
 class DeepseekV4ForCausalLM(nn.Module):
     def __init__(
@@ -2421,14 +2524,7 @@ def determine_num_fused_shared_experts(self):
         self.num_fused_shared_experts = self.config.n_shared_experts
 
     @torch.no_grad()
-    def forward(
-        self,
-        input_ids: torch.Tensor,
-        positions: torch.Tensor,
-        forward_batch: ForwardBatch,
-        input_embeds: Optional[torch.Tensor] = None,

# ... truncated 65 lines; see upstream PR ...
```

#### parallel_state.py

```diff
@@ -1701,6 +1701,14 @@ def set_pdmux_status(enable_prefill_multiplexing: bool):
     _ENABLE_PDMUX_P_TP = enable_prefill_multiplexing
 
 
+def is_pdmux_prefill_enabled() -> bool:
+    return _ENABLE_PDMUX_P_TP
+
+
+def is_pdmux_enabled() -> bool:
+    return _PDMUX_PREFILL_TP_GROUP is not None
+
+
 def get_tp_group() -> GroupCoordinator:
     if _ENABLE_PDMUX_P_TP:
         assert (
```

### 本地核对

- [ ] 目标树已包含 #30885 等价改动
- [ ] 开关/env 可用且行为符合预期
- [ ] 与相邻 PR 无冲突（见依赖）

---

## 依赖关系（核对时注意顺序）

```text
通信底座
  #30700 FlashInfer MNNVL pure AR
  #28639 ag_gemm / moe_rs symm-mem overlap

Prefill CP
  #33532 CP V2 interleave for DSV4     ✅ MERGED
       ↓
  #33236 remove KV/compressor materialization
       ↓
  #32059 Shared KV via VMM
       ↓
  #33382 LayerSplit common infra
       ↓
  #29187 DSV4 LayerSplit full

Decode CP
  #30416 DCP (draft/open)

正确性
  #31700 dp_gather_replicate
       ↓
  #33217 TBO policy guard
  #33250 TBO fix (depends #31700)

产品
  #30885 DSV4 × PDMux
```

## 异地快速符号扫描命令（可选）

```bash
# 已合入 CP-v2
rg -n "cp_materialize_global_token_order|is_cp_v2_active|cp_round_robin_input_ids_v2" -g '*.py'

# 未合入：正确性
rg -n "dp_gather_replicate" python/sglang/srt/models/deepseek_v4.py
rg -n "attn_tp_size == 1" python/sglang/srt/models/deepseek_v4.py

# 未合入：Prefill CP 存储/重叠
rg -n "direct_cp_kv_store|cp_compress" -g '*.py' -g '*.cuh'
rg -n "SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL|allgather_gemm_op_symm_mem|moe_reduce_rs_symm_mem" -g '*.py'
rg -n "shared_kv|deepseek_v4_shared|enable_.*shared" -g '*.py' | head
rg -n "enable_cp_cache_layer_split|cp_cache_layer_split" -g '*.py' | head

# 未合入：DCP / PDMux
rg -n "SGLANG_DSV4_ENABLE_DCP|_try_forward_dcp_sharded_c4_indexer" -g '*.py'
rg -n "pdmux|multiplexing_mixin|split_prefill" python/sglang/srt/models/deepseek_v4.py python/sglang/srt/multiplex | head

# 未合入：MNNVL pure AR
rg -n "flashinfer.*mnnvl|FlashInferMnnvl|pure.?allreduce|all_reduce_raw" -g '*.py' | head
```

## 备注

- 大 PR（#32059/#29187/#30416/#33236/#28639）此处只贴**代表性 diff**；完整以 GitHub PR Files 为准。
- 作者/行数以抓取时刻 `gh pr view` 为准，force-push 后可能变化。
- 「本地已有不等价 upstream」时，请在核对框旁注明 commit/hash。
