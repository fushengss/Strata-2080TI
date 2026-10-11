# Unsloth Q8_0 (experimental)

**Experimental. Setup installs it only from files you already have; it does not download it.** This is Unsloth's
plain 8-bit file of the same model the other packs use: the `Q8_0/` folder of
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (the same repository on
ModelScope works too). Back to [the models](MODELS.md).

## What is in the file

- **Routed experts:** gate, up and down are Q8_0 in every layer. One layer's experts are 3 x 512 x 640 x 2560 values
  at 34 bytes per 32: 2.67 GB a layer, **128.3 GB** for all 48 (UD-Q4_K_XL: 77 GB, UD-Q6_K_XL: 109 GB).
- **Everything else:** as in [UD-Q4_K_XL](UNSLOTH_Q4.md) and [UD-Q6_K_XL](UNSLOTH_Q6.md): the attention and
  shared-expert projections, the token embedding and the output head Q8_0, and the small projections the engine reads
  as BF16 converted by the packer's `--compat-bf16`.
- **The PLE table** (`per_layer_token_embd.weight`): Q8_0, 170-byte rows, 54.4 GB, as in UD-Q6_K_XL.

Together about 188 GB. This is computed from UD-Q6_K_XL's measured file sizes (59.9 GB beside its experts) plus the
Q8_0 experts, not measured on this file. Setup does not pin the shard count or SHA-256 of this file. It reads the
count from the first shard's name (`...-00001-of-0000N.gguf`) and checks every shard against its own tensor
directory, as it does for any `--gguf-dir`.

## What the engine runs it with

Each piece was already in the engine for other models; this page adds no engine code.

| Part | Path | Used before by |
| --- | --- | --- |
| Experts on the GPU, decode | grouped expert kernels and MMVQ, type 8 (`STRATA_GU_FMTS`, `STRATA_D_FMTS` in `src/kernels/cuda/iq_kernels.cu`) | UD-Q6_K_XL's and UD-Q4_K_XL's layer 2 (Q8_0 gate/up), UD-IQ4_XS's Q8_0 downs |
| Experts, prompts | llama.cpp's MMQ for Q8_0 (`src/prefill/moe_mmq.cu`, built in every CUDA engine) | the draft layer's dense matrices |
| Experts on the CPU | ggml-cpu's Q8_0 dot products; the AVX2 rows of `src/kernels/cpu/kq_avx2.cpp` with `STRATA_KQ256=1` (opt-in) | UD-Q4_K_XL's Q8_0 downs |
| Dense projections | native MMVQ from the GGUF (`src/core/native_dense.cpp`) | UD-Q4_K_XL |
| PLE table | the Q8_0 row format (`src/kernels/ngram.cpp`) | UD-Q6_K_XL, Swift-1.5 Q4_K_L |

## Setup

Point `--gguf-dir` at the repository folder (setup then looks in its `Q8_0/` subfolder) or at the `Q8_0/` folder
itself:

```sh
./setup.sh --setup --family unsloth --model Q8_0 \
  --gguf-dir /data/models/modelscope/models/unsloth/Qwen3.8-Flash-Next-GGUF
```

What setup does for this model:

- It lists Q8_0 in the size menu only with `--gguf-dir`. `--model Q8_0` without `--gguf-dir` stops before anything
  is downloaded and says what to add.
- **One GPU:** the RAM budget is the RAM less 24 GB, at most all 119 GiB of experts. Setup recommends Q8_0 from 160 GB
  of RAM, where every expert fits. With less, the experts outside the budget are read from the SSD for every token.
- **Several GPUs** (`--gpus 0,1,2,3`, `--gpus all`): the layer split, without a RAM budget. All the experts are
  loaded into RAM from the GGUFs at start, as for UD-Q4_K_XL on several GPUs (#498). Setup recommends it where the
  RAM holds the files and 24 GB more (~212 GB). Each card keeps an expert cache for its own layers
  ([MULTI_GPU.md](MULTI_GPU.md)).
- It packs with `--compat-bf16` and never writes `experts.bin`: the experts are read from the GGUF files in place.
- The MTP draft layer comes from the original Qwen checkpoint as for every model (~5 GB of tensors;
  `--source modelscope` reads them from ModelScope). Unsloth's `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` is not used: the
  engine's draft layer is built by `tools/mtp_pack.py` and `tools/mtp_rt.py` from the checkpoint's own tensors.
- Images are asked, off by default. The image encoder is the original model's (`mmproj`), downloaded when you say
  yes; it is untested with this file.

A PC with 8 RTX 2080 Ti (22 GB each, compute capability 7.5) and 377 GB of RAM, for example:

```sh
./setup.sh --setup --family unsloth --model Q8_0 --source modelscope --gpus 0,1,2,3 \
  --gguf-dir /data/models/modelscope/models/unsloth/Qwen3.8-Flash-Next-GGUF
```

Cards 0-3 share one NUMA node there; `--gpus all` adds the other four (behind the CPU interconnect). Which one is
faster on such a PC has not been measured.

## The pack and the server (by hand)

The same steps as [UD-Q4_K_XL](UNSLOTH_Q4.md#the-pack-by-hand), with the Q8_0 names:

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-0000N.gguf \
  --out packs/unsloth-q8_0 --compat-bf16
```

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/unsloth-q8_0",
    "--native", "/path/to/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-0000N.gguf",
    "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "32768", "--kv", "int8"
  ],
  "gpu": [0, 1, 2, 3],
  "layer_split": "auto",
  "cwd": ".",
  "tokenizer": "packs/unsloth-q8_0/tokenizer",
  "model_name": "qwen3.8-flash-next-unsloth-q8_0",
  "host": "127.0.0.1",
  "port": 8080
}
```

`--ple-gguf` is not needed: the engine finds the table's shard from shard 1. On one GPU, add
`"--resident-budget-gib", "119"` (or the RAM less 24 GB, whichever is smaller) and drop `gpu`/`layer_split`.

## Status: what is verified and what is not

Verified (2026-10-07, macOS, no GPU): setup's tests for this model (`python -m unittest tools.test_setup_unsloth`):
the menu, the refusal without `--gguf-dir`, the repository folder and the `Q8_0/` folder as `--gguf-dir`, the pack
command, the config on one GPU (budget 119 GiB at 377 GB of RAM) and on eight (no budget).

Not verified: the engine has not run this file yet. Speed, VRAM use and output quality against llama.cpp are not
measured, and neither is the shard layout of the real file. Please report how it runs, with `strata-unsloth-q8_0.log`.
