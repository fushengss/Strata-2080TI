# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read [AGENTS.md](AGENTS.md) first. It points to the docs for installing Strata for a user ([docs/AI_SETUP.md](docs/AI_SETUP.md)) and for the engine internals ([docs/DETAILS.md](docs/DETAILS.md), [docs/paper/Strata-Paper.pdf](docs/paper/Strata-Paper.pdf)).

## What this is

Strata runs the Qwen3.8-Flash-Next MoE model (24,576 experts, 10 per token, 48 layers) on one consumer GPU plus system RAM. It has three parts:

- **C++/CUDA/HIP engine** (`src/`, `include/strata/`, same subdirectory names in both). The main binary is `strata` (`src/program/generate.cpp`). The other `strata-*` executables are diagnostics: `strata-gguf`, `strata-dequant`, `strata-plan`, `strata-device`, `strata-load`.
- **Python server** (`serve/`) with OpenAI, Anthropic and Responses APIs, plus the web app in `serve/web/`.
- **Installer** `setup.py` (one ~5,400-line file). `START-HERE.bat`, `SETUP.bat` and `setup.sh` start it, and `UPDATE.bat` / `update.sh` update without starting the model.

`sycl/` is a separate CMake project for Intel Arc ([docs/INTEL.md](docs/INTEL.md)). It reuses `../src` unless it carries a migrated copy (`src/X.cu` becomes `sycl/src/X.dp.cpp`). When you change engine code, check whether `sycl/src` has a copy of that file.

## Architecture (big picture)

- **Engine and server boundary.** `serve/server.py`'s `StrataEngine` keeps one `strata --serve` process running and talks to it over stdin/stdout with a line protocol: `GEN <max_new> <ids>`, `GENI` (with image embeddings), and `BGEN`/`BSTOP <slot>` for batching (`"parallel"`). It is parsed in `generate.cpp`. When you change the protocol, change both sides. `engine_args(cfg)` in `server.py` turns the JSON model config (`strata-<model>.json`, written by setup) into engine flags.
- **`MockEngine`** in `serve/server.py` is a scripted engine. Every API path can be tested with it, without a GPU or model: `python -m serve.server --engine mock --port 8095`.
- **`serve/frontend.py`** works with text only. It renders the model's Jinja chat template exactly as Hugging Face does (checked against `serve/chat_golden.json`). It parses streamed output into reasoning (`<think>`), content and XML tool calls, and never emits a partial tag. `serve/responses.py` is the Responses API (Codex), `serve/mcp.py` provides MCP tools to the web app, and `serve/structured.py` handles JSON response formats.
- **Where the work runs.** The GPU holds the dense parts (attention and DeltaNet mixers, routers, shared experts, head, MTP draft layer), the KV cache and an adaptive **expert cache** that fills the remaining VRAM (`src/core/expert_cache.cpp`). RAM holds every expert, pinned, and the CPU computes cache misses in place at the same time as the GPU (`src/kernels/cpu`, ggml-cpu for the i-quants). The SSD holds the 28.8 GB n-gram (PLE) table, read a few rows per token (`src/ngram`).
- **Speculative decoding** (`src/spec`): the model's MTP layer drafts up to 3 tokens, and prompt lookup drafts up to 5 where measured acceptance says it pays. Verification keeps the output identical.
- **Prompt processing** (`src/prefill`): chunks of up to 8,192 tokens. The next layer's experts stream over PCIe while the current layer's attention runs.
- **ggml** comes from a llama.cpp checkout: `-DSTRATA_GGML_DIR=...`, otherwise the commit pinned in `third_party/ggml/VERSION.txt` is fetched. It is used for i-quant CPU experts and the optional MMQ prompt kernels.
- **Engine versioning.** `project(strata VERSION a.b.c.d)` in `CMakeLists.txt` becomes `STRATA_VERSION`. `MIN_ENGINE` in `setup.py` (a 4-tuple, all four numbers compared) decides when an installed ready-made engine gets replaced. Its comment is a changelog, one entry per release. Setup rebuilds a locally compiled engine when the hash of `ENGINE_SOURCES` (`CMakeLists.txt`, `cmake`, `src`, `include`, `third_party/ggml`) changes.

## Commands

Python tests use `unittest`, run from the repo root. None of them need a GPU, a model or downloads, but they do need the packages in `requirements.txt` (`pip install -r requirements.txt`, or use the `.venv` that setup creates). Python 3.10 or newer.

```bash
python -m unittest discover -s serve -p "test_*.py"           # all server tests
python -m unittest serve.test_server -v                       # one module
python -m unittest serve.test_server.SomeClass.test_name      # one test
python tools/test_setup_choices.py                            # setup's tests (also test_setup_amd.py, ...)
```

Engine builds use CMake and Ninja. Setup installs both into `.venv`. You need a GPU toolkit, so the engine does not build on macOS. `strata-gguf` and `strata-plan` build without one.

```bash
# NVIDIA (setup's own flags; arch 120 is the default, use your card's sm, e.g. 86 or "86;89")
cmake -G Ninja -S . -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build --target strata

# AMD (docs/AMD_HIP.md): gfx1100 / gfx1101 / gfx1200 / gfx1201, or a list
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

C++ tests and kernel parity checks (`*_test.cpp`, `src/kernels/*_parity.cpp`, `tests/{core,cuda,hip,...}`) are registered with `add_test` and run through `ctest --test-dir <build> [-R name]`. `STRATA_BUILD_TESTS` defaults to ON only when `tests/CMakeLists.txt` and `bench/micro/` exist. They are missing from this checkout, so pass `-DSTRATA_BUILD_TESTS=ON` (and `-DSTRATA_BUILD_CONVERSATION_TESTS=ON` for the conversation cache tests). Some targets only exist with certain options (`STRATA_NATIVE_EXPERTS`, CUDA vs HIP). Other build options are listed at the top of `CMakeLists.txt`: `STRATA_PORTABLE`, `STRATA_WERROR`, `STRATA_MMQ_KQUANTS`, `STRATA_HIP_GFX906`, `STRATA_EXPERIMENTAL_SM60`, ...

Run the server against the real engine: `python -m serve.server --engine strata --config strata-<model>.json --port 8080`.

## Conventions

- Docs and code comments are written in plain words. Every performance claim states the measured number and the hardware and engine version it was measured on. Do not write a claim without a measurement. Issue numbers (`#1234`) are cited inline.
- Module docstrings start with the file path, a one-line purpose and how to run it. Follow that pattern for new files.
- The server binds `127.0.0.1` by default. Never expose it more widely without `--api-key`.
- `README.md` has translations (`README.*.md`). When README content changes, those files need the same change.
- PRs use `.github/pull_request_template.md` (Title/Issue, Summary, What changed, Extra Notes).
