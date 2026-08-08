# DSpark PD Prefill-PP PoC (metadata buffer)

This is a **short validation** path, not production-ready DSpark PD.

## Goal

- Prefill: `--disaggregation-mode prefill --pp-size > 1`, **no** `--speculative-algo DSPARK`
- Decode: `--disaggregation-mode decode --speculative-algo DSPARK --pp-size 1`
- Transfer a short target aux-hidden tail via existing PD metadata buffers
- Decode injects that tail into draft KV before the first draft step

## Env knobs

| Env | Default | Meaning |
|---|---|---|
| `SGLANG_DSPARK_PD_ENABLE_PREFILL_CAPTURE` | `1` | Enable Prefill aux capture without local DSPARK |
| `SGLANG_DSPARK_PD_TARGET_LAYER_IDS` | empty | Comma-separated layer ids; if empty, infer from draft/model config |
| `SGLANG_DSPARK_PD_PREFILL_TAIL_TOKENS` | `0` (= auto `8`) | Tail window length. Prefill/Decode must match; default is fixed `8` so schemas agree without local speculative args. |

## Example launch shape

```bash
# Prefill (PP allowed; do NOT set --speculative-algo DSPARK)
SGLANG_DSPARK_PD_TARGET_LAYER_IDS="<layer ids>" \
python -m sglang.launch_server \
  --model-path <DeepSeek-V4-Flash-DSpark> \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --pp-size 2 \
  --disable-radix-cache \
  ...

# Decode (DSPARK on; PP=1)
python -m sglang.launch_server \
  --model-path <DeepSeek-V4-Flash-DSpark> \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --speculative-algo DSPARK \
  --pp-size 1 \
  ...
```

## Limits

- Metadata-buffer path only: short prompt / short tail. Not full-prompt RDMA hidden (#31466).
- Prefill/Decode radix-cache policies should stay off / matched.
- No Mooncake streaming hidden, no NIXL/MORI special path.
- Decode still cannot enable `pp_size > 1` with DSPARK (mainline hard check).
