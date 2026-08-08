"""Minimal DSpark PD handoff helpers (metadata-buffer PoC).

Inspired by closed PR #29705 and scoped for short validation:
- Prefill may run without ``--speculative-algo DSPARK`` and with ``pp_size > 1``.
- Decode runs with DSPARK and receives a short target-aux hidden tail via the
  existing PD metadata buffers (not the full #31466 RDMA ``DSPARK_HIDDEN`` path).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

logger = logging.getLogger(__name__)

# Fixed default so Prefill (no DSPARK) and Decode (DSPARK) agree on the
# metadata wire schema without requiring matching speculative_* args.
DEFAULT_DSPARK_PD_PREFILL_TAIL_TOKENS = 8


def _parse_layer_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def infer_dspark_pd_target_layer_ids(
    *,
    server_args: ServerArgs,
    hf_config: Any = None,
) -> List[int]:
    """Resolve target layer ids for Prefill-side aux capture / metadata sizing."""
    env_ids = _parse_layer_ids(envs.SGLANG_DSPARK_PD_TARGET_LAYER_IDS.get())
    if env_ids:
        return env_ids

    try:
        from sglang.srt.speculative.dspark_components.dspark_config import (
            parse_dspark_draft_config,
        )

        draft_path = (
            server_args.speculative_draft_model_path or server_args.model_path
        )
        if draft_path is None:
            return []
        if hf_config is None:
            from sglang.srt.utils.hf_transformers_utils import get_config
            import json

            hf_config = get_config(
                draft_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=(
                    server_args.speculative_draft_model_revision or server_args.revision
                ),
                model_override_args=json.loads(server_args.json_model_override_args),
                model_config_parser=server_args.model_config_parser,
            )
        cfg = parse_dspark_draft_config(draft_hf_config=hf_config)
        if cfg.target_layer_ids:
            return [int(x) for x in cfg.target_layer_ids]
    except Exception:
        logger.debug("Failed to infer DSpark PD target layer ids", exc_info=True)
    return []


def resolve_dspark_pd_prefill_tail_len(
    *,
    server_args: Optional[ServerArgs] = None,
) -> int:
    del server_args  # Intentionally unused: keep P/D schema independent of local spec args.
    env_val = envs.SGLANG_DSPARK_PD_PREFILL_TAIL_TOKENS.get()
    if env_val is not None and int(env_val) > 0:
        return int(env_val)
    return DEFAULT_DSPARK_PD_PREFILL_TAIL_TOKENS


def is_dspark_pd_prefill_capture_enabled(server_args: ServerArgs) -> bool:
    if server_args.disaggregation_mode != "prefill":
        return False
    if not envs.SGLANG_DSPARK_PD_ENABLE_PREFILL_CAPTURE.get():
        return False
    # Prefill intentionally does NOT require --speculative-algo DSPARK.
    return bool(
        infer_dspark_pd_target_layer_ids(server_args=server_args)
        or server_args.speculative_algorithm == "DSPARK"
    )


def resolve_dspark_pd_metadata_hidden_spec(
    *,
    server_args: ServerArgs,
    model_hidden_size: int,
    model_dtype: torch.dtype,
    hf_config: Any = None,
) -> Optional[Tuple[int, torch.dtype, int, List[int]]]:
    """Return (hidden_width, dtype, tail_len, layer_ids) when DSpark PD is active."""
    mode = server_args.disaggregation_mode
    if mode not in ("prefill", "decode"):
        return None

    layer_ids: List[int] = []
    if mode == "decode" and server_args.speculative_algorithm == "DSPARK":
        layer_ids = infer_dspark_pd_target_layer_ids(
            server_args=server_args, hf_config=hf_config
        )
    elif mode == "prefill" and is_dspark_pd_prefill_capture_enabled(server_args):
        layer_ids = infer_dspark_pd_target_layer_ids(
            server_args=server_args, hf_config=hf_config
        )

    if not layer_ids:
        return None

    hidden_width = len(layer_ids) * int(model_hidden_size)
    tail_len = resolve_dspark_pd_prefill_tail_len(server_args=server_args)
    return hidden_width, model_dtype, tail_len, layer_ids


def extract_dspark_pd_hidden_from_logits(
    *,
    hidden_states: torch.Tensor,
    token_start: int,
    token_end: int,
    tail_len: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Slice one request's aux-hidden rows into last-token + short tail tensors."""
    if hidden_states is None or token_end <= token_start:
        empty = torch.empty((0,), dtype=torch.float16)
        return empty, None, None

    req_hidden = hidden_states[token_start:token_end]
    last_row = req_hidden[-1].detach().to("cpu").contiguous()
    if tail_len <= 0:
        return last_row, None, None

    copy_len = min(int(tail_len), int(req_hidden.shape[0]))
    tail = req_hidden.new_zeros((tail_len, req_hidden.shape[-1]))
    mask = torch.zeros((tail_len,), dtype=torch.bool, device=req_hidden.device)
    if copy_len > 0:
        tail[-copy_len:].copy_(req_hidden[-copy_len:])
        mask[-copy_len:] = True
    return (
        last_row,
        tail.detach().to("cpu").contiguous(),
        mask.detach().to("cpu").contiguous(),
    )


def populate_req_dspark_pd_hidden(
    req: Req,
    *,
    hidden_states: torch.Tensor,
    token_start: int,
    token_end: int,
    tail_len: int,
) -> None:
    last_row, tail, mask = extract_dspark_pd_hidden_from_logits(
        hidden_states=hidden_states,
        token_start=token_start,
        token_end=token_end,
        tail_len=tail_len,
    )
    req.hidden_states_tensor = last_row
    req.prefill_tail_hidden_states_tensor = tail
    req.prefill_tail_valid_mask = mask
    # DSpark PD does not use EAGLE topk metadata.
    req.output_topk_p = torch.zeros((1,), dtype=torch.float32)
    req.output_topk_index = torch.zeros((1,), dtype=torch.int64)


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    del server_args, future_map
    spec_info = make_draft_input_v2(
        bonus_tokens=last_tokens_tensor.to(dtype=torch.int64),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
    )

    req_hidden = [req.hidden_states_tensor for req in batch.reqs]
    if req_hidden and all(h is not None and h.numel() > 0 for h in req_hidden):
        spec_info.hidden_states = torch.stack(req_hidden, dim=0).to(batch.device)
    else:
        spec_info.hidden_states = torch.empty(
            (0, 0), dtype=torch.float16, device=batch.device
        )

    req_tail = [
        getattr(req, "prefill_tail_hidden_states_tensor", None) for req in batch.reqs
    ]
    req_mask = [getattr(req, "prefill_tail_valid_mask", None) for req in batch.reqs]
    if (
        req_tail
        and all(t is not None for t in req_tail)
        and all(m is not None for m in req_mask)
    ):
        spec_info.prefill_tail_hidden_states = torch.stack(req_tail, dim=0).to(
            batch.device
        )
        spec_info.prefill_tail_valid_mask = torch.stack(req_mask, dim=0).to(batch.device)
        # Absolute start positions for the transferred tail window.
        starts = []
        for req, mask in zip(batch.reqs, req_mask):
            valid = int(mask.sum().item()) if mask is not None else 0
            seq_len = int(getattr(req, "kv_committed_len", len(req.origin_input_ids)))
            starts.append(max(0, seq_len - valid))
        spec_info.prefill_tail_start_positions = torch.tensor(
            starts, dtype=torch.int64, device=batch.device
        )
        spec_info.pd_hidden_pending_inject = True
    else:
        spec_info.prefill_tail_hidden_states = torch.empty(
            (0, 0, 0), dtype=torch.float16, device=batch.device
        )
        spec_info.prefill_tail_valid_mask = torch.empty(
            (0, 0), dtype=torch.bool, device=batch.device
        )
        spec_info.prefill_tail_start_positions = torch.empty(
            (0,), dtype=torch.int64, device=batch.device
        )
        spec_info.pd_hidden_pending_inject = False

    spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
    return spec_info


def maybe_inject_pd_prefill_hidden(
    *,
    draft_input: DFlashDraftInputV2,
    batch: ScheduleBatch,
    kv_injector: Any,
    req_to_token_pool: Any,
) -> None:
    """Inject transferred Prefill aux-hidden rows into decode-side draft KV."""
    if not getattr(draft_input, "pd_hidden_pending_inject", False):
        return
    tail = getattr(draft_input, "prefill_tail_hidden_states", None)
    mask = getattr(draft_input, "prefill_tail_valid_mask", None)
    starts = getattr(draft_input, "prefill_tail_start_positions", None)
    if tail is None or mask is None or starts is None or tail.numel() == 0:
        draft_input.pd_hidden_pending_inject = False
        return

    bs, tail_len, hidden_dim = tail.shape
    del hidden_dim
    device = batch.device
    req_pool_indices = batch.req_pool_indices
    req_to_token = req_to_token_pool.req_to_token

    flat_hidden: List[torch.Tensor] = []
    flat_pos: List[torch.Tensor] = []
    flat_loc: List[torch.Tensor] = []
    for i in range(bs):
        valid = mask[i]
        n = int(valid.sum().item())
        if n <= 0:
            continue
        rows = tail[i, -n:]
        start = int(starts[i].item())
        positions = torch.arange(start, start + n, device=device, dtype=torch.int64)
        # Gather cache locations from the already-transferred target KV mapping.
        locs = req_to_token[req_pool_indices[i], positions].to(torch.int64)
        flat_hidden.append(rows)
        flat_pos.append(positions)
        flat_loc.append(locs)

    if not flat_hidden:
        draft_input.pd_hidden_pending_inject = False
        return

    kv_injector.inject_target_hidden(
        target_hidden=torch.cat(flat_hidden, dim=0).to(device=device, non_blocking=True),
        cache_loc=torch.cat(flat_loc, dim=0),
        positions=torch.cat(flat_pos, dim=0),
    )
    draft_input.pd_hidden_pending_inject = False


def local_pp_owns_any_layer(
    *,
    layer_ids: Sequence[int],
    start_layer: int,
    end_layer: int,
) -> bool:
    return any(start_layer <= int(layer_id) < end_layer for layer_id in layer_ids)
