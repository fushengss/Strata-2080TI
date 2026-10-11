<h1 align="center">Strata-2080TI</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

## This fork: 8x RTX 2080 Ti 22 GB

Strata-2080TI is based on [Niko1221/Strata](https://github.com/Niko1221/Strata) and tuned for one server with
**8x NVIDIA RTX 2080 Ti 22 GB** (Turing, sm_75). Installing and using Strata: see the README of [Niko1221/Strata](https://github.com/Niko1221/Strata).

**Test machine:** 8x RTX 2080 Ti 22 GB on PCIe only (no NVLink, no P2P), the cards spread over two NUMA nodes,
377 GB RAM, CUDA 12.8. Model: [Unsloth Q8_0](docs/UNSLOTH_Q8.md), split by layers across all 8 cards, every expert
in VRAM (99.8% cache hits). Engine 0.1.40.3, measured 2026-10-11.

**What changed:**

- `--split-chunks N` (new engine flag): on a layer split a prompt is read in about N chunks of 2048-4096 tokens, so
  the chunk buffers no longer borrow most of each card's expert cache. Before, `--prefill auto:32768` borrowed 2,295
  of each card's 3,072 slots and the experts crossed PCIe again during every prompt.
- The stages' prompt threads no longer inherit the host thread's pinned core. 35 engine threads were sharing CPU 0
  and the GPUs waited on it for 14% of a prompt's GPU time (8.4 s -> 0.4 s).
- Tuned settings for Turing: `STRATA_BF16_TC=1` (BF16 tensor cores), `STRATA_PREFILL_ZC=1` (resident experts read in
  place, opt-in, not bit-identical), `"parallel": 4` with `--batch-groups 4`.

**Reads your prompt** (tokens/s):

| Prompt | Before, `--prefill auto:32768` | This fork | Speedup vs. `auto:32768` |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1,080** | 1.5x |
| 16K | 891-908 | **2,240** | 2.5x |
| 32K | 809-883 | **2,920** | 3.4x |
| 100K | — | **3,974** | — |

**Writes answers:** 57-60 tokens/s for one request (unchanged: decoding is GPU-bound, the 8 cards run their layers
one after another). **8 requests at once:** 136 tokens/s in total, against 50 tokens/s one at a time.

The "before" column is upstream with `--prefill auto:32768`, two requests each, from [docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds) (the step by step
numbers are there too). Tried on these cards with no gain: `--spec 6` (slower), `--mtp-q4 all`, `STRATA_HC_Q8`.

The settings, added to the config setup wrote (`strata-<model>.json`):

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
