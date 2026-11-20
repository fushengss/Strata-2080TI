# 8× RTX 2080 Ti、Q5_K_M、100k 测试与优化（2026-11-07）

第一阶段按用户要求保持服务在线，测量 10.100.3.3 在其他用户持续请求下的性能。
随后用户明确进入维护窗口、授权停启与修改，第二阶段进行独占微基准及隔离端口 8091 的配置 A/B。
下面的现网数字与独占测试分开记录，不能互相当作配置收益。

**此前配置阶段部署：** 只把线上 `--batch-groups 4` 改为 `8`。均匀分层 `6,12,18,24,30,36,42`、prefill 配置保持原值。
2026-11-07 16:43（Asia/Taipei）复核服务为 0.1.57、8 个并发槽，HTTP 生成测试返回 `READY`。
本轮长上下文六并发合计速度 +约 7%，短请求八并发 -约 10.5%；单请求 decode 没有显著变化。
这是按本次 100k 长上下文目标选择的取舍，不是所有工作负载都更快。CPU governor 保持原来的 `schedutil`，V2K 没有开启。

用户随后授权直接修改引擎，decode 输出头分担方案的完整 ABBA、正确性检查及最终服务状态另见
[decode 输出头测试报告](decode-head/README.md)。完整 decode TP8 的只读可行性分析见
[TP8 布局、显存与通信分析](decode-head/tp8-feasibility.md)。下文保留此前配置实验的测量口径与历史结果。

## 机器与版本

- 8× RTX 2080 Ti 22 GiB，双路 Xeon E5-2650 v4，377 GiB RAM；驱动 580.178.04。
- GPU0–3 / GPU4–7 分属两个 NUMA 节点；`nvidia-smi topo -p2p r` 的卡间结果全部为 CNS/TNS，没有可用 P2P。
- `/srv/Strata-0157/engine/strata`，服务自报 0.1.57；实际二进制 SHA256 见 `metadata.json`。
- orca Qwen3.8-Flash-Next-Uncensored Q5_K_M；8 段，每段 6 层，专家全部驻留显存。
- `parallel=8`、`batch-groups=4`、`pipeline-windows=2`、`spec=3`；INT8 KV，`kv-resident=32768`，最大上下文 262144。
- 保留原 prefill 配置：`split-chunks=2`、chunk max 8192、resident sort、split MTP batch、MMQ KQ、BF16 TC 等。
- 运行进程已启用 `SM75_PERF`、`BATCH_SELF_COMMIT`、`CKPT_POOL`、`SLOT_COPY_DIRECT`、`BATCH_GDN_GROUPS`；没有设置 `EXPERT_V2K` 或 `KQ_SPLIT`。
- 本地当前 main 是 `25cd4ab4`，与线上不同。分析以 `upstream-0.1.57@9d71f701` 和远端源码为准；相关远端文件 hash 在 `source-hashes.json`。

## 测量口径

`benchmark_live.py` 是最初六次测量实际执行的脚本。代码、散文、开启思考三类，各两次，每次最多输出 768 token。
非思考请求使用 greedy，明确关闭 presence/frequency/repetition penalty；思考请求保留服务端模型卡采样。
使用与此前测试相同的合成重复文本，前缀每次不同；实际输入 101121–101174 token，API 与引擎都确认缓存为 0。
这不是自然文档或真实 agent 历史的代表性语料，不能把它的 prefill 速度外推到所有输入。

- **prefill**：引擎日志的实际新读 token 数 ÷ `read in ... ms`，与客户端首 token 时间分开。
- **TTFT**：请求发出到第一个非空 content/reasoning delta，包含排队、分词、prefill 和首次输出。
- **decode**：`(completion_tokens - 1) / (最后非空 delta 时间 - 首个非空 delta 时间)`。这是流式客户端估计，包含中途被其他请求阻塞的时间；delta 不保证逐 token 发送。
- batch admission 日志里 `1 generated in 18 ms` 只是进入 batch 前的首次输出，**不是整个回答的 decode 速度**。
- 每 0.5 秒记录在途请求数与 GPU 指标。初始与结束累计请求计数也用于发现中途加入的请求。
- `benchmark.py` 在同一工作负载上增加 SSE 段时间和引擎日志观察时间，用于最后一次诊断；日志观察精度约 0.5 秒。
- `gpu.csv` 的时间戳是服务器 UTC；对用户报告的时间换算为 Asia/Taipei。

运行 `python3 analyze.py` 可从原始结果重建 `summary.json`。记录中不包含 API key、真实用户提示词或回答。

## 第一阶段：现网结果

时间为 2026-11-07 15:45–15:55（Asia/Taipei），每条输出 768 token，全部零前缀缓存。

| 工作负载 | 新输入 token | 引擎 prefill 秒 | prefill tok/s | 客户端 TTFT 秒 | 客户端有效 decode tok/s | 最大在途请求 |
|---|---:|---:|---:|---:|---:|---:|
| 代码 1 | 101145 | 15.026 | 6731.4 | 15.24 | 19.55 | 7 |
| 散文 1 | 101121 | 14.870 | 6800.4 | 15.08 | 23.70 | 6 |
| 思考 1 | 101174 | 15.524 | 6517.2 | 21.50 | 22.00 | 6 |
| 代码 2 | 101145 | 15.179 | 6663.4 | 21.04 | 18.68 | 6 |
| 散文 2 | 101121 | 15.667 | 6454.4 | 15.88 | 13.78 | 6 |
| 思考 2 | 101174 | 15.143 | 6681.3 | 15.38 | 13.78 | 6 |
| 代码，逐段计时 | 101145 | 14.684 | 6888.0 | 14.88 | 41.11 | 4 |

最后一次代码请求出现 **4.646 秒没有输出**（请求开始后 25.639–30.285 秒）；同一时间段另一请求读取
2429 个新 token 花了 4509 ms（日志在约 30.652 秒观察到）。与源码中 admission 暂停 decode 的行为一致。
该次输出阶段各卡 GPU busy 均值约 31%–57%，但含其他请求与 prefill；不是本请求的独占 SM 或带宽利用率。

## 第二阶段：独占 Q5 专家微基准

GPU0，T=4、40 个专家 entry，真实 GGUF layer3 为 Q5_K/Q8_0、layer6 为 Q5_K/Q5_1。
50 次迭代、两个交替轮次，单位 µs；两列数字分别是两种量化组合的进程结果中位数。

| 专家共享 | baseline | V2K | KQ_SPLIT |
|---|---:|---:|---:|
| 每专家 1 token（G40） | 327.8 / 288.8 | 342.6 / 318.7 | 385.2 / 335.7 |
| 每专家 2 token（G20） | 234.3 / 202.4 | 202.5 / 193.9 | 222.1 / 203.5 |
| 每专家 4 token（G10） | 219.2 / 196.7 | 142.7 / 145.6 | 170.0 / 169.0 |

48 个输出各 409600 字节，跨进程与原内核逐字节一致，非有限值为零。固定权重和一个激活种子，不代表完整质量测试。
V2K 在充分共享时减少 26%–35% 的专家计算时间，无共享时反而增加 4.5%–10.3%；必须用真实路由的整机结果决定。
复现文件与原始结果在 `kernels/`，实际 dump 字节留在远端 `/srv/bench100k-20261107-kernels/`。

隔离端到端结果由 `summarize_ab.py --root isolated --baseline baseline` 汇总到 `isolated/ab-comparison.md`。

## 第三阶段：隔离整机 A/B

相同模型、固定输入、greedy、无惩罚、不思考；每臂从新进程启动，prefill 参数保持不变。
冷请求各三条（代码 768 token 两条，散文 512 token 一条）、100k 对话追加 638 个实际新 token 后输出 256 token。
并发六条长请求复用 98304-token 前缀，各再读 2861 token、输出 1024 token；八条短请求各输出 1024 token。
每种配置一套完整测试，不能当作多次独立重复的置信区间。

| 配置 | 冷 prefill 三次（秒） | 100k 代码 decode 两次（客户端 tok/s） | 100k 散文 decode | 6 路长上下文合计 tok/s | 8 路短请求合计 tok/s |
|---|---|---|---:|---:|---:|
| 原配置 groups4 | 14.486 / 16.837 / 16.777 | 133.36 / 137.58 | 81.29 | 111.47 | 276.00 |
| groups4 + V2K | 14.493 / 16.735 / 16.718 | 134.91 / 138.88 | 81.51 | 110.92 | 268.15 |
| groups8 | 14.506 / 16.772 / 16.773 | 133.29 / 137.55 | 81.26 | 119.24 | 246.97 |
| groups8 + V2K | 14.468 / 16.726 / 16.735 | 134.72 / 138.54 | 81.46 | 117.14 | 241.01 |

所有候选的 18 条固定请求输入与原配置一致，输出文本 SHA256 也全部一致，没有请求错误或 CUDA/显存错误记录。
groups8 的长上下文并发合计快约 7%，短请求并发慢约 10.5%；单请求速度与 prefill 没有可见变化。
V2K 全局开启没有体现微基准中高共享形状的收益，且并发变慢，因此不采用该开关。
各卡仍保留全部 3072 个专家槽，以及各自 3.80 GiB prompt buffers，没有借用专家槽。
启动时较早的“8192 chunk lends 1022 cache slots”警告早于最终 split own-buffer 决策，不能用它断言实际 prefill 在借槽。

### 分层平衡试验：因 prefill 回退而拒绝

依据尾卡负载，另测 `[7,6,7,6,6,6,6,4]` 层数分配，边界 `7,13,20,26,32,38,44`，groups8。
全部专家仍驻留，四条单请求输出均与原配置一致，但冷 prefill 变为 **15.495 / 17.498 / 17.712 秒**，
中位数比原配置慢 **4.30%**，超过事先采用的 3% 回归门槛。因此没有继续该配置的并发测试，已撤回，没有部署。
这证明直接把尾卡的层移走会影响 prefill，不能当作满足本次约束的免费优化。

### CPU governor：ABBA 没有 decode 收益

在 8080 临时只允许 loopback 访问，等待在途请求归零后测试；原始与结束请求计数、0.5 秒采样均未发现竞争请求。
固定约 100k 前缀、相同请求输出 768 token，ABBA 的客户端 decode 为：

| 调频策略 | A1 schedutil | B1 performance | B2 performance | A2 schedutil |
|---|---:|---:|---:|---:|
| decode tok/s | 132.64 | 132.56 | 132.60 | 132.42 |
| decode 阶段 CPU0 采样平均 MHz | 2887 | 2900 | 2894 | 2891 |

四次请求和输出 hash 全部一致。`schedutil` 在 decode 时已经升到接近 2.9 GHz，没有可见收益。
新前缀 prefill 分别为 16.780 / 15.389 / 15.388 / 15.499 秒；返回 schedutil 后也维持约 15.5 秒，
不能把首次 A1 的差值全部归因于 governor。所有 48 个 policy 均恢复原值；没有修改频率上下限，临时 INPUT 规则已删除。

## 第四阶段：100k 单请求 Nsight profile

`profile/metadata.json` 对应单条 101168-token 输入、4096-token 输出，PW2、spec3、groups8 配置。
因为只有一个请求，它走 solo PW2，不是八组 batch。首 token 后采集 CUDA graph node trace，事件跨度约 10.124 秒。
追踪会增加开销，下面只用来分析结构，不当作未追踪的速度基准。

按内核名称归类的跨 GPU 累计时间（总计 17.511 秒，可跨卡/stream 重叠，**不是端到端关键路径占比**）：

| 分类 | 累计时间 | 累计时间占比 |
|---|---:|---:|
| Dense / head / shared / router | 5.894 s | 33.66% |
| 专家 gate/up/down/SwiGLU | 5.118 s | 29.23% |
| Hyper-connection | 2.750 s | 15.71% |
| Attention | 1.275 s | 7.28% |
| GDN | 0.697 s | 3.98% |
| QSA score/top-k | 0.448 s | 2.56% |
| KV/copy | 0.379 s | 2.16% |

GPU7 kernel 执行区间并集占采样窗 **47.1%**，其余卡 **13.9%–18.7%**。
独立的、未追踪的并发 GPU 采样中，groups8 的 C6 共同输出区间 GPU7 busy **93.8%**，其他卡 **52.5%–60.7%**；
C8 时 GPU7 为 **96.6%**。两个场景的测量口径不同，但都显示尾段负载集中。

### Q5 专家之外，Q6 输出头已经接近单卡读带宽

这份模型的专家是 Q5，输出/草稿 head 仍包含 Q6_K。SQLite 核对的 GPU7 draft subset head 为
106299×2560、T=1、Q6_K 每 256 个权重 210 字节；2553 次调用累计 1.104543 秒，平均 432.65 μs。
每次逻辑权重为 223227900 字节（212.9 MiB），因此 **逻辑权重字节数÷kernel 时间约 515.96 GB/s**。
按采样看到的 6800 MHz DDR、352 bit 计算，理论带宽约 598.4 GB/s，比值约 86.2%。

这是有效权重带宽推算，**不是 DRAM 流量计数器实测**。在每份权重从显存读一次的简化模型里，理想耗时为 373.04 μs，
该 kernel 剩余时间空间约 13.8%，不能据此期望单纯重写 Q6 解码就让整机大幅加速。
`STRATA_Q6_PACKED=1` 可作为小幅改进候选，本次没有测试或开启；它另需约 710 MiB 显存，且 main head 的部分 IL 路径绕过它。

更值得下一步实现的是 **decode 专用输出头设备分离**：保留现有层和 prefill placement，在较空闲卡算 main head 与采样，
跨卡只传小尺寸 hidden/Q8_1 输入和必要结果，避免每步传全词表 logits。先评估对应 API 的采样、penalty、回滚和 slot 状态。
当前 C6/C8 不跑 batch MTP，只移动 drafter 不能解决它们的 main-head 尾卡瓶颈。
单请求可再拆分 drafter 的 decode placement，但需要保留原 prefill 路径、处理 KV/缓存交接，交接耗时仍计入 TTFT。
现成 `STRATA_MTP_STAGE0=1` 会改变/关闭 split MTP batch prefill 的选择，不能直接作为无损 prefill 的替代。

## 为什么现网 decode 慢

0.1.57 的调度路径与观察一致：

1. `serve/server.py:2074` 的 `_take_control()` 保证一次只读一个 prompt。
2. `src/program/generate.cpp:15136` 收到新 BGEN 等控制命令后先 `pipe_drain()`，排空在途 decode。
3. `include/strata/program/batch_read.hpp:24` 的 piped 路径默认不在 prefill 块间插入 decode；`STRATA_PIPED_READ_PIECE` 默认 0。
4. 当前普通 batch 路径每 slot 每 window 发出一个 token。单请求的 `spec=3`、MTP、PW2 不意味着 batch 同时使用 MTP。
5. `STRATA_BATCH_SELF_COMMIT=1` 与 `STRATA_PIPED_MTP` 冲突。历史本机测试里单独调整 batch MTP 也没有证明有收益。
6. `direct_slot_copy()`（`generate.cpp:14084`）仍有跨设备同步和状态复制。已有 direct-copy/batchsync 优化减少了部分等待，但没有消除整个 admission 停顿。

专家全在显存消除了专家 RAM/磁盘缺失，不会消除层间依赖、GPU 间传递、KV streaming、admission 或采样等待。
8 卡层切分也不等于每个 token 的计算会在 8 卡同时展开。GPU busy 百分比不是 DRAM 带宽占峰值百分比。

现网额外做了 20 秒、49 Hz 的 CPU perf 采样：约 75.23% 的 CPU cycles 样本在 libcuda，10.08% 在 libcudart；调用栈包含
`batch_poll`、`cudaStreamQuery`、`cudaStreamSynchronize`。这是 CPU 活跃周期的采样，**不是 GPU 内核耗时分账，也不能据此断言 CPU 是唯一瓶颈**。
第一阶段没有重启注入 Nsight；上述 CPU 采样不能提供 CUDA kernel 时间、实际 DRAM GB/s 或精确 GPU 空闲原因。

## 在不牺牲 prefill 的约束下，下一步怎么优化

### 0. 优先测纯 `batch-groups=8`，不同时开启 batch MTP

当前配置是 8 stages、8 slots、4 groups，每 group 两个 slot 行。`generate.cpp:14675` 创建 4 个 group；
同一个 group 在途时不能再次发出（约 14977 行），上一 stage 的 compute/copy/commit 完成后才能推进，最后 stage 完成才清除 inflight。
因此，**普通 batch decode 最多四个 group window 在途、最多四个 stage 同时执行这些窗口**。这不是 prefill 或全进程所有 GPU 工作的绝对上限。
HTTP 在途数还包含正在读取与等待的请求，也不能直接当 active decode slots。

纯 `batch-groups=8` 会把每组两行变成每组一行；5–7 个 active decode slots 时可以提供更多独立窗口，减少流水线空泡。
大块 prefill 实现、chunk 设置和整段 admission 规则不变，值得作为最先测试的配置候选。
代价是减少同窗口的权重共享、增加 host launch/poll 与跨段交接；图和 drain 开销也会变，必须实测，不能保证一定提升。
以前只测四个请求的 groups8，或同时开 batch MTP 的 groups8，都不足以判断当前 5–7 并发下纯 groups8 的收益。
它不会解决新 prompt 造成的整段暂停。隔离 A/B 的结果与取舍见上表。

### 1. 优先独立验证 Q5 decode 专用内核

`src/kernels/cuda/iq_kernels.cu` 已有两个候选，线上默认未开：

- `STRATA_KQ_SPLIT=1`（约 898–913 行）：Q5_K/Q4_K/Q8_0 的权重解码结果跨同专家的 token 复用；Q5_1 因浮点收缩差异被排除。
- `STRATA_EXPERT_V2K=1`（约 3878、6306 行）：Q5_K gate/up + Q5_1/Q8_0 down；共享内存保存激活、多行处理、融合 SwiGLU 与 Q8_1，并复用权重解码。

现有 `SM75_PERF=1` 并未覆盖全部 Q5 专家路径，`MMQ_KQ=1` 主要作用于 prefill。不能把这两个开关已开等同于 Q5 decode 已调到最好。
这两个候选修改 grouped expert decode 的分发，不改变大块 prefill 的 MMQ 实现，符合本次约束的优先顺序。
对短追加的小块路径、显存和整机调度仍需回归；本次 V2K 的隔离整机收益很小且并发变慢，KQ_SPLIT 仅测微基准，没有作为整机候选。

应分别测，不能同时打开后归因：V2K 提前 return，会遮住后面的路径。
正确性也要独立验证：V2K 在旧内核切换之前分发，现有同进程 parity 可能两边都运行 V2K。
应跨进程保存相同输入的实际输出字节/hash，覆盖 Q5_K/Q8_0 与 Q5_K/Q5_1，并检查输出有限。
`native_expert_bench` 在 V2K 下的 gu/down 分项会误测全套，只能采用 all 时间。
后续整机 A/B 应交替重复，固定输入/采样，检查长 prefill、短追加、专家驻留数、scratch 余量、输出正确性与 batch 混合负载。

### 2. 减少 admission 的复制和全局同步

prefill 直接写目标 slot、按依赖用 CUDA event 替代可避免的全设备同步，有机会同时改善 prefill 结束交接和正在输出请求的停顿。
这是状态归属与调度的代码改造，需要独立验证；不是已有单一开关的确定收益。

把 prompt 拆小并强插 decode 时间片会延长新 prompt 的完成时间；直接并发 prefill/decode 会争抢 GPU 资源。
当前 `STRATA_SPLIT_MTP_BATCH=1` 还会阻止 `STRATA_PIPED_CONCURRENT`。这些方案不能宣称无代价地满足“不影响 prefill”，本次不改。

### 3. 根据 profile 再决定 QSA 和 KV 优化

100k 上下文不等于 100k KV 全在显存：当前 resident=32768。需要测各 stage 的 KV resolve/miss 开销，再决定是否扩大 resident。
8 个并发槽和 prefill scratch 都要计入预算，不能为了 KV 把专家挤回 RAM。
日志 KV 命中率是累计、且只涵盖该设备拥有的 QSA 层，不能把一条日志直接解释为当前请求的全机命中率。

QSA 的多 query 共享 key 已存在。0.1.57 也有长上下文多 CTA top-k 的可选路径，但 Turing 需验证编译条件、dispatch 和额外 launch 成本。
本机编译的 `STRATA_EXPERIMENTAL_SM60=OFF`，不能只设置环境变量就假定该长 top-k 实现生效。
本次 profile 中 QSA score/top-k 仅占累计 kernel 时间 2.56%，优先级低于输出头与尾段负载。

## 1Cat-vLLM 中可借鉴的部分

参考固定版本 `c98a8d66482333eb8d8b87be72e8856445ac99ec`，而不是把其广告速度直接套到本机：

- [小 M 权重复用](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/gguf_dense_weight_reuse.md)：按 token 数量选择专用路径，解码一次权重后服务多个 token，保留不同 shape 的 fallback。与上述 Q5 候选方向一致。
- [QSA 结构优化](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/flashnext_qsa_router_structure.md)：共享 key、缩小候选集合后精确选择 top-k；保留 FP32 cutoff、tie 和索引顺序。其准入范围不能直接外推至本机 100k。
- [完整推测轮次分账](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/flashnext_mtp4_structural_costs.md)：同时看 verify、draft、head、采样、等待、接受率与每轮实际输出 token，不能只看算子时间。

其约 260 tok/s 示例是 4× V100、27B NVFP4 + DFlash2；与本机 Flash-Next Q5、2080 Ti、无 P2P 的层切分不是同条件。
NVLink TP 通信优化、降量化位数、增加整份 FP16 权重、重做专家并行，都不应当成当前不损 prefill 的直接替代方案。

## 产物、部署和回滚

- `live-results.json`、`timing-results.json`、`summary.json`：现网原始结果、输出停顿和 GPU 采样汇总。
- `isolated/ab-comparison.md`：整机逐项结果、固定输入/输出 hash 对照；`isolated/raw/*.tar.gz` 保留完整原始 SSE、提示词、结果。
  可读 JSON 中省略重复的逐段数据；归档 SHA256 在 `isolated/raw/index.json`。
- `kernels/`：临时微基准源码、构建/运行脚本、ABBA 与输出正确性记录。
- `profile/`：Nsight 摘要。完整 `.nsys-rep` / SQLite 留在远端 `/srv/bench100k-20261107-candidates/profile-groups8/`。
- `cpu-governor-report.json`、`balanced-decision.json`：未采用方案的测量及恢复记录。
- `deployment.json`、`final-validation.json`：唯一线上配置变更、备份、最终检查。

线上配置：`/srv/Strata/strata-orca-q5_k_m.57.json`，只将 `--batch-groups` 后的 `4` 改为 `8`。
原配置备份：`/srv/Strata/strata-orca-q5_k_m.57.json.bak-groups4-20261107`。
需要回到原配置时，在服务器上执行（会重启服务并清空内存中的会话缓存）：

```sh
cp /srv/Strata/strata-orca-q5_k_m.57.json.bak-groups4-20261107 /srv/Strata/strata-orca-q5_k_m.57.json
/srv/Strata/restart57.sh
```

本地 main 的原有未提交源码/文档改动未纳入这次部署；线上引擎二进制未替换。
