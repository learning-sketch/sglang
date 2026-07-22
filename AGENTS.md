# AGENTS.md

## Cursor Cloud specific instructions

### Environment reality: no GPU
This cloud VM is **CPU-only (no NVIDIA GPU/CUDA)**. The main product — the Python
`sglang` serving engine (`python/pyproject.toml`, `sgl-kernel/`) — pins CUDA-only
wheels (`torch`, `flashinfer[cu13]`, `flash-attn-4`, `sglang-kernel`, `cuda-python`,
`nvidia-cutlass-dsl[cu13]`, …) and **cannot be installed or run here**. Do not attempt
a full `pip install -e python`; it will fail on CUDA wheels. GPU-only tests are the
`base-b*/base-c*/nightly*` CI suites (see `test/README.md`); only `base-a-test-cpu`
is CPU-eligible.

### What IS runnable here: the Rust `sgl-model-gateway`
`sgl-model-gateway/` (the model routing control/data plane, binaries
`sgl-model-gateway`/`smg`/`amg`) is a pure-Rust network service with no CUDA
dependency and is the runnable, testable core product on this VM. All commands are
documented in `sgl-model-gateway/README.md` and `sgl-model-gateway/Makefile`; the
canonical ones:
- Build (debug, fast dev): `cd sgl-model-gateway && cargo build`
- Test: `cargo test` (390 lib unit tests pass; use `--lib --bins` to skip criterion benches)
- Lint: `cargo clippy --all-targets --all-features -- -D warnings` (aka `make check`)
- Run (starts with no workers): `./target/debug/sgl-model-gateway --enable-igw --host 127.0.0.1 --port 30000`

`experimental/sgl-router/` is a second, slimmer CPU-only Rust router with the same
`cargo build/test/clippy` flow.

### Non-obvious gotchas (read before building Rust)
- **Default compiler must be GCC, not clang.** `/usr/bin/c++` originally pointed at
  clang 18, which fails to find libstdc++ `<cstdint>` when compiling the C++ dep
  `esaxx-rs` (pulled in via the tokenizer), aborting the build. The snapshot has been
  switched to GCC. If a build ever regresses with `fatal error: 'cstdint' file not
  found`, re-run: `sudo update-alternatives --set c++ /usr/bin/g++` and
  `sudo update-alternatives --set cc /usr/bin/gcc`.
- **System build deps** (`libssl-dev`, `pkg-config`, `protobuf-compiler`) are required
  by the gateway (openssl 0.10 dep + prost/tonic `protoc` codegen) and are preinstalled
  in the snapshot. Alternatively build with `--features vendored-openssl` to avoid
  system OpenSSL.
- **Rust toolchain**: crates pin `1.90` via `rust-toolchain.toml`; `rustup` auto-fetches
  it on first use inside the crate dir. A clean first build of the gateway takes
  ~5–8 min on 4 CPUs (large dep tree: wasmtime, kube, tonic, redis, …).

### Testing the gateway's core functionality without workers
The reasoning/tool parsers (the Rust logic powering gRPC OpenAI-compatible serving)
can be exercised with no GPU worker via the admin endpoints:
- `POST /parse/reasoning` and `POST /parse/function_call`.
- Parser names in the request body are the **registered names with underscores**
  (e.g. `deepseek_r1`, `qwen3`, `glm45`, `step3`, `minimax`), NOT the hyphenated
  model-pattern aliases (`deepseek-r1` is only a model auto-detect pattern and is
  rejected by these endpoints).
- `POST /parse/function_call` requires a `tools` array in addition to `text` and
  `tool_call_parser`.
