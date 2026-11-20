# 局部 TP2/4 独立微基准

TP2/4 原型及两轮实测已完成。主表使用保留的 `.mtp2` 二进制；分片通信与分解 reference 检查全部通过。**T=1–3 时，当前分解原型比 production fused 慢，且没有全面通过 production bitwise 检查，不接入线上。** T=8 有局部收益，不能据此判断当前 solo decode 或整模型 TP 提速。

| 项目 | 固定值 |
|---|---|
| 硬件 | 10.100.3.3，RTX 2080 Ti（nvidia-smi 总显存 22001 MiB/卡），取同 NUMA 的 GPU 0,1 / 0,1,2,3 |
| 保留二进制 | `/srv/Strata-decode-head-20261107/engine/local_tp_bench.mtp2` |
| 二进制 SHA256 | `7a88dea05b45b41851f3bb7b50cef196a4a2b389c21c20cb8c6534a3940db6be` |
| runner SHA256 | `f1e5339be2a7121f25b752788fd566c7b54850712c665ca125402b973630c791` |
| 主测量目录 | `tp2-mtp2/`、`tp4-mtp2/` |
| 启动时间 UTC | 2026-11-07 10:45:10 / 10:45:14 |
| model | 当前 Q5_K_M 三片 GGUF，由 `--gguf` 第一片解析其它片 |
| layer / rows | layer 3、6；T=1、2、3、8 |
| 实际工作 | 固定十个 routed experts，所有 T 行共享这些 experts；不含 router/shared expert/HC/attention |
| 三臂 | production fused、单卡分解、行分片 TP |

## 实际执行命令

下面由 `tp2-mtp2/metadata.json` 与 `tp4-mtp2/metadata.json` 的 `command` 和 `environment` 还原，是 runner 实际启动的子程序及记录到的环境。runner 负责采集计时、日志和 GPU 状态。没有把未执行的环境变量加入命令。

```sh
env STRATA_SM75_PERF=1 \
  /srv/Strata-decode-head-20261107/engine/local_tp_bench.mtp2 \
  --gguf /models/orcarouter/Qwen3.8-Flash-Next-Uncensored-Q5/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf \
  --devices 0,1 --layers 3,6 --experts 5,42,79,116,153,190,227,264,301,338 \
  --warmup 10 --iterations 100 --parity-cases 4
```

```sh
env STRATA_SM75_PERF=1 \
  /srv/Strata-decode-head-20261107/engine/local_tp_bench.mtp2 \
  --gguf /models/orcarouter/Qwen3.8-Flash-Next-Uncensored-Q5/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf \
  --devices 0,1,2,3 --layers 3,6 --experts 5,42,79,116,153,190,227,264,301,338 \
  --warmup 10 --iterations 100 --parity-cases 4
```

环境记录仅有 `STRATA_SM75_PERF=1`。当前源码没有读取 `STRATA_NO_PCIE_GROUP`；本微基准 `Production::run` 直接调用 `native_expert_grouped`，其 `grid_groups` 默认为 0，与 `verify.cpp` 的 all-resident 分支相同。因此没有设置该变量不会改变本微基准的专家 dispatch。没有设置 `CUDA_VISIBLE_DEVICES`，metadata 记录了物理拓扑。

## 测量和完整性

原始数据：[TP2 summary](tp2-mtp2/summary.json)、[TP4 summary](tp4-mtp2/summary.json)，各目录还保存 `results.jsonl`、`metadata.json`、`stderr.log` 和 `gpu-samples.jsonl`。[verification.json](verification.json) 保存四次运行的完整性核验摘要。

两次主测量均为 exit 0，各有 **8 个性能 case、32 个 parity case、20 个 collective case**；layer/T/payload 组合完整，所有性能 case 三臂各 100 个样本，二进制 hash 与上表一致。所有 hidden Q8_1、TP-vs-decomposed、collective 检查通过，没有观测到非有限输出。production 差异 case 集合与第一轮完全一致。

每个数据点为 10 次 warmup 后 100 次、三臂交错执行；不是 100 次独立进程或整机 A/B。固定十个 expert 被全部 T 行复用，是高复用路由。程序只读取指定 expert，不上传完整模型。

`row_tp.wall_us_p50/p95` 包含 host 提交、切 device、portable mapped host 传输及所有 rank 同步，和 `production_fused.wall_us_*` 比较。`decomposed_single_gpu` 用于核对分解开销与分片 parity，不能只选它作为性能基线。计算阶段 CUDA event 排除了跨卡交换，但跨 rank 的 sum/max 都不是重建的 critical path。collective 的 event 区间包含复制 kernel 和等待，不是纯计算时间。

runner 的 `completed` 仅表示 exit 0，不代表 production parity 通过；非零子程序退出码会传递，77 为 GPU 不可用的 SKIP。启动失败、JSON 损坏、输出目录非空会报错；`nvidia-smi` 采样失败只被记录，不会自动使程序失败。以上统计经过独立读取 JSON 核验，而不是仅凭 completed 判断通过。

## 冻结二进制实测结果

倍率为 `production wall p50 / TP wall p50`，大于 1 才代表该局部计算更快。production bitwise 是该 layer/T 的四个输入 case 全部通过；它不等于全模型 exact 证明。T=4 未测，不能用 T=8 代替当前小 speculative window。

| TP | layer / down | T | production p50 / p95 μs | 单卡分解 p50 μs | TP p50 / p95 μs | 对 production 倍率 | production bitwise |
|---|---|---:|---:|---:|---:|---:|---|
| TP2 | 3 / Q8_0 | 1 | 101.19 / 102.50 | 188.42 | 180.44 / 191.98 | 0.561× | 否 |
| TP2 | 3 / Q8_0 | 2 | 134.83 / 136.39 | 224.13 | 235.23 / 245.45 | 0.573× | 否 |
| TP2 | 3 / Q8_0 | 3 | 180.19 / 188.76 | 252.62 | 260.06 / 273.37 | 0.693× | 否 |
| TP2 | 3 / Q8_0 | 8 | 447.39 / 448.79 | 468.24 | 367.63 / 385.90 | 1.217× | 否 |
| TP2 | 6 / Q5_1 | 1 | 90.03 / 90.92 | 176.78 | 178.20 / 187.78 | 0.505× | 是 |
| TP2 | 6 / Q5_1 | 2 | 118.78 / 120.09 | 217.78 | 222.70 / 225.92 | 0.533× | 否 |
| TP2 | 6 / Q5_1 | 3 | 159.64 / 162.35 | 256.52 | 249.08 / 252.85 | 0.641× | 否 |
| TP2 | 6 / Q5_1 | 8 | 400.58 / 401.75 | 512.59 | 380.34 / 383.65 | 1.053× | 是 |
| TP4 | 3 / Q8_0 | 1 | 102.46 / 103.56 | 190.31 | 224.55 / 233.23 | 0.456× | 否 |
| TP4 | 3 / Q8_0 | 2 | 136.27 / 137.70 | 225.68 | 260.29 / 271.30 | 0.524× | 否 |
| TP4 | 3 / Q8_0 | 3 | 182.01 / 183.40 | 254.42 | 284.37 / 291.61 | 0.640× | 否 |
| TP4 | 3 / Q8_0 | 8 | 441.93 / 451.32 | 463.97 | 372.85 / 383.80 | 1.185× | 否 |
| TP4 | 6 / Q5_1 | 1 | 90.81 / 91.92 | 178.69 | 224.00 / 232.70 | 0.405× | 是 |
| TP4 | 6 / Q5_1 | 2 | 119.87 / 121.11 | 219.35 | 262.51 / 270.16 | 0.457× | 否 |
| TP4 | 6 / Q5_1 | 3 | 161.39 / 162.22 | 258.36 | 285.48 / 293.01 | 0.565× | 否 |
| TP4 | 6 / Q5_1 | 8 | 397.08 / 401.12 | 509.30 | 394.85 / 399.10 | 1.006× | 是 |

**结论**：T=1–3 时，TP2 wall latency 为 production 的约 **1.44–1.98 倍**，TP4 为约 **1.56–2.47 倍**。T=8 layer3 的倍率为约 1.217×/1.185×，但未通过 production bitwise；layer6 为约 1.053×/1.006×。两轮结果方向一致，当前分解原型没有提供可用于 solo 小窗口的收益，保留 production fused 专家路径。不能把局部倍率外推为整模型 token/s。

## 数值差异及边界

- 所有主测量中，TP2/4 的 hidden Q8_1 与单卡分解 reference 逐字节一致，最终 F32 输出也 bitwise 一致；全部 collective raw-bit 检查通过。没有观测到行分片、拼接或传输引入数值变化。
- 对 production fused，layer3 的所有非零输入 case 都有差异，最大绝对误差约 **5.59e−9**，一般约三分之二的输出 float 不同；零输入相同。
- layer6 的 T1/T8 四个输入 case 均与 production 相同。T2 的 case 3、T3 的 case 1 有差异，最大绝对误差分别为 **1.97e−6**、**1.16e−6**。TP2/TP4 及两轮测量的失败 case 集合完全一致，但 layer3/T8 的不同元素数存在小幅变化，不能声称各次生产输出逐位相同。接近零的参考值会放大 `max_rel`，也不能凭绝对误差小就宣称模型 token/路由不变。
- **差异定位在实现族，具体中间值未逐阶段定位**：原型使用 `native_mmvq` 和 `native_swiglu_quantize_q8_1`；production 使用 `native_expert_grouped` 分组 GU/down。源码中前者 SwiGLU 使用 `__fdividef`（`native_mmvq.cu:183`），后者使用普通浮点除法（`iq_kernels.cu:2866`）；CMake 的 `--use_fast_math` 源文件列表包含前者，没有包含后者。dot kernel 的 lane/归约组织也不同。这些是可产生末位差及量化阈值翻转的实际差异，但本轮没有导出 production 的 GU/Q8/down 中间值，不能断言某项就是全部误差的根因。

因此“保留完整 dot、未改 GGUF 量化”只在分解 reference 内通过验证；本原型不是生产 exact 替换。当前原型每个 expert 单独发 GEMV，生产路径融合 groups；负收益只拒绝该实现，不能证明融合 TP 不可行。

## Host collective 实测

以下是 **T=1** host-inclusive allgather p50/p95，不是 NCCL、DMA copy benchmark 或仅 PCIe 线路延迟。每次包含各 rank 提交、mapped-host publish/acquire、event 等待及所有 rank 同步。payload 为每验证行大小；T 行时 bytes 乘 T，collective 次数没有随 T 再乘一遍。

| 每行 payload B | TP2 p50 / p95 μs | TP4 p50 / p95 μs |
|---:|---:|---:|
| 6912 | 26.76 / 27.85 | 56.72 / 61.59 |
| 7200 | 26.55 / 27.10 | 56.26 / 60.96 |
| 10240 | 26.89 / 27.88 | 56.28 / 58.68 |
| 24576 | 28.15 / 29.07 | 57.45 / 63.24 |
| 40960 | 30.12 / 30.56 | 58.94 / 63.86 |

7–10 KiB 小 payload 的 TP2 约 27 μs、TP4 约 56–57 μs，符合本实现受到提交/同步成本限制的现象。T=8 时 40 KiB/行（共 320 KiB）的 TP2/4 p50 为 **98.29/107.87 μs**。完整 20 个点在 summary；不能把一次 host 同步时间直接乘 96，用来预测 GPU 内部握手或融合 collective 的 TP8 速度。

## 时钟和适用范围

主测量完整 runner wall time 为 TP2 **3.52 s**、TP4 **4.20 s**；每秒采样得到 3/4 条运行中记录，无法精确对应每个微基准 case。记录包含启动阶段 P5/810 MHz 显存时钟、执行期间 P2/6800 MHz，以及进出任务时 P0/7000 MHz。P2 活跃 SM 时钟约 1845–1890 MHz，相关 GPU 温度约 43–54°C。未锁时钟，没有温度过高证据，不能声称所有 case 始终处于同一频率。

T1–3 大幅负收益在两轮重复，足以拒绝当前实现的线上集成。T8 的局部优势不代表实际 solo decode 收益：路由固定且高度复用、样本是短进程中的微基准，production bitwise 也未全面通过。本实验没有测 prefill、真实路由多样性、T4、48 层 TP、KV/GDN 状态桥或端到端 decode，因此没有给出这些场景的性能或等价性保证。

## 附录：第一轮对照

第一轮 [TP2](tp2/summary.json)、[TP4](tp4/summary.json) 分别于 2026-11-07 **10:34:41 / 10:34:56 UTC** 启动，执行时路径为 `local_tp_bench.mtp1`，SHA256 为 `4e0686b14de35d2f243dfa0928301d4e34f0b68e981d33a9d96f7631d2ada320`。该路径之后被 relink 覆盖，本报告主表改用另名保留的 `.mtp2`；第一轮 raw data 与 metadata 未改动。第一轮 runner wall 为 3.73/4.25 s，与主测量使用相同原型源、runner、参数和记录环境，但二进制链接产物不同。

第一轮同样各有 8/32/20 个 case，hidden Q8、TP-vs-decomposed、collective 均通过；production 失败 case 集合和最大误差与主测量一致。T1–3 TP2 为生产延迟的 1.44–2.01 倍，TP4 为 1.57–2.49 倍。

| TP | layer / down | T | production p50 / p95 μs | 单卡分解 p50 μs | TP p50 / p95 μs | 对 production 倍率 | production bitwise |
|---|---|---:|---:|---:|---:|---:|---|
| TP2 | 3 / Q8_0 | 1 | 101.91 / 103.41 | 188.21 | 191.92 / 194.28 | 0.531× | 否 |
| TP2 | 3 / Q8_0 | 2 | 134.84 / 136.01 | 223.65 | 241.93 / 246.69 | 0.557× | 否 |
| TP2 | 3 / Q8_0 | 3 | 179.44 / 180.43 | 251.99 | 257.75 / 266.83 | 0.696× | 否 |
| TP2 | 3 / Q8_0 | 8 | 445.58 / 448.08 | 467.38 | 369.82 / 374.77 | 1.205× | 否 |
| TP2 | 6 / Q5_1 | 1 | 90.67 / 91.94 | 178.09 | 181.82 / 190.17 | 0.499× | 是 |
| TP2 | 6 / Q5_1 | 2 | 119.56 / 120.46 | 219.03 | 225.28 / 228.22 | 0.531× | 否 |
| TP2 | 6 / Q5_1 | 3 | 160.35 / 161.17 | 257.53 | 252.94 / 259.40 | 0.634× | 否 |
| TP2 | 6 / Q5_1 | 8 | 401.34 / 403.34 | 514.81 | 382.54 / 384.62 | 1.049× | 是 |
| TP4 | 3 / Q8_0 | 1 | 102.45 / 103.47 | 188.96 | 225.65 / 232.94 | 0.454× | 否 |
| TP4 | 3 / Q8_0 | 2 | 135.52 / 136.73 | 224.01 | 262.32 / 266.46 | 0.517× | 否 |
| TP4 | 3 / Q8_0 | 3 | 179.74 / 180.67 | 251.93 | 281.45 / 290.46 | 0.639× | 否 |
| TP4 | 3 / Q8_0 | 8 | 438.24 / 446.17 | 460.27 | 371.00 / 378.36 | 1.181× | 否 |
| TP4 | 6 / Q5_1 | 1 | 90.79 / 91.79 | 177.74 | 225.79 / 231.97 | 0.402× | 是 |
| TP4 | 6 / Q5_1 | 2 | 119.49 / 120.54 | 218.88 | 263.12 / 270.25 | 0.454× | 否 |
| TP4 | 6 / Q5_1 | 3 | 160.32 / 161.15 | 257.31 | 287.13 / 292.82 | 0.558× | 否 |
| TP4 | 6 / Q5_1 | 8 | 392.93 / 396.82 | 503.83 | 391.12 / 395.32 | 1.005× | 是 |
