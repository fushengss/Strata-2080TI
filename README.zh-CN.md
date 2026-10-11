<h1 align="center">Strata-2080TI</h1>

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

## 本分支：8× RTX 2080 Ti 22 GB

Strata-2080TI 基于 [Niko1221/Strata](https://github.com/Niko1221/Strata)，专门针对一台装有
**8 张 NVIDIA RTX 2080 Ti 22 GB**（Turing，sm_75）的服务器做了优化。Strata 的安装和使用方法见 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的 README。

**测试机器：** 8× RTX 2080 Ti 22 GB，仅走 PCIe（无 NVLink、无 P2P），显卡分布在两个 NUMA 节点上，377 GB 内存，
CUDA 12.8。模型：[Unsloth Q8_0](docs/UNSLOTH_Q8.md)，按层切分到全部 8 张卡上，所有专家都在显存里（缓存命中率 99.8%）。
引擎 0.1.40.3，测于 2026-10-11。

**改了什么：**

- `--split-chunks N`（新的引擎参数）：按层切分时，提示被分成约 N 块、每块 2048-4096 token 来读，块缓冲区不再借走每张卡
  大部分的专家缓存。之前 `--prefill auto:32768` 会借走每张卡 3,072 个槽位中的 2,295 个，每次读提示时专家都要重新走一遍 PCIe。
- 各阶段读提示的线程不再继承主线程绑定的 CPU 核。原来有 35 个引擎线程挤在 CPU 0 上，GPU 在读提示的 GPU 时间里有 14%
  在等它（8.4 秒 -> 0.4 秒）。
- 针对 Turing 调好的设置：`STRATA_BF16_TC=1`（BF16 张量核）、`STRATA_PREFILL_ZC=1`（常驻专家原地读取，需手动开启，
  结果不逐位相同）、`"parallel": 4` 配合 `--batch-groups 4`。

**读提示**（tokens/s）：

| 提示长度 | 优化前，`--prefill auto:32768` | 本分支 | 相对 `auto:32768` 的提升 |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1,080** | 1.5 倍 |
| 16K | 891-908 | **2,240** | 2.5 倍 |
| 32K | 809-883 | **2,920** | 3.4 倍 |
| 100K | — | **3,974** | — |

**写回答：** 单个请求 57-60 tokens/s（没有变化：解码受 GPU 限制，8 张卡依次跑各自的层）。**同时 8 个请求：**
总计 136 tokens/s，逐个处理时为 50 tokens/s。

"优化前"一列是上游版本用 `--prefill auto:32768`，每项测了两个请求，出自 [docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds)（逐项改动的数据也在那里）。
在这些卡上试过但没有提升的：`--spec 6`（更慢）、`--mtp-q4 all`、`STRATA_HC_Q8`。

设置方法，加到 setup 写出的配置文件（`strata-<model>.json`）里：

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
