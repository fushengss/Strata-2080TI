# 8×2080 Ti：保留现有 prefill 的 decode TP8 可行性

2026-11-07。只读分析；没有启动 TP8、运行 collective 基准或占用 GPU。现网/候选版本以 `Strata-decode-head` 的上游 0.1.57 为基础。本文不代替正在进行的 head 分片 ABBA 结果。

**可以实现，而且仓库已有同模型 TP 的代码基础；当前配置不能直接打开 TP8。** 难点是保留原 prefill 权重布局的显存、原生 Q5 混合张量兼容、长上下文状态交接和小消息同步延迟。没有 P2P 不等于 TP8 必然更慢；也不能把八张卡的闲置比例换算成八倍加速。

## 证据边界

- **已有实测/文件事实**：`hardware.csv`、`topology.txt` 显示 8×22528 MiB 2080 Ti，GPU0–3/4–7 分属两个 NUMA 节点；P2P 检查为 CNS/TNS，没有可用 P2P。已有 profile 的单请求 PW2 窗口中，GPU7 kernel union 为 47.12%，其余为 13.92–18.67%；这是带追踪的单请求窗口，不是六请求 cohort。
- **只读检查**：本次读取实际三片 GGUF 的 tensor header，统计得到专家 payload 为 **91,855,257,600 B = 85.546875 GiB**。48 层 gate/up 全是 Q5_K；down 24 层 Q8_0、24 层 Q5_1。没有读取/改写权重内容。
- **下文推算**：显存与通信量由实际 shape 和存储格式计算；未测 TP2/4/8 延迟、带宽、接受率和端到端收益。profile kernel 累计时间不是 critical-path 时间。

## 实际模型怎样分八份

几何来自 `include/strata/core/geometry.hpp:24–58`，与实际 GGUF 一致；不是普通 dense Qwen 的配置。

| 部分 | 实际形状 | TP8 可行分法与限制 |
|---|---|---|
| 主干 | 48 层，hidden 2560，HC 4 路残差共 10240，HC low-rank 320 | 初版复制 HC、norm、router，避免它们增加同步；这部分工作不会获得 8 倍加速。 |
| GDN，36 层 | 16 个 K head、48 个 V head，各 128；卷积 10240 通道、历史 3 步 | 每卡 2 K / 6 V head，状态随 head 切；输出 6144→2560 可以列切后求和，或输出行切后 allgather。 |
| QSA，12 层 | 24 Q / **2 KV** head，head_dim 256；Q+gate 投影 12288 | 每卡 3 Q head；按 head 切时每个 KV head 被 4 个 rank 使用，KV 至少组内四份。另一条是按 token/position 切 KV，再归并 attention；现有 split 引擎采用后一条。 |
| QSA indexer | 4 query head×128，pool 每 4 token 一行，top-k 2048 | 不能简单把 4 个 query head 分给 8 卡。可复制/集中计算选择，或切 pooled-key 区域后合并 top-k；需要额外通信及 exact top-k 验证。 |
| Routed MoE | 每层 512 专家，top-10，FF 640 | 每张卡切每个专家，或改成每卡 64 个完整专家。后者是 EP，top-10 路由负载不保证均衡，不能按 TP8 等量工作估计。 |
| 专家 GU / down | GU 输入 2560、输出各 640，Q5_K；down 输入 640、输出 2560，Q5_1/Q8_0 | 640/8=80 不是 down 的 32 值量化块边界。已有 `card_split.hpp:25–26` 自动分成 **96,96,96,96,64,64,64,64**，最大 FF 工作比理想均分多 20%；不是数学上不能 TP8。 |
| 主 head | 2560→248320，实际 Q6_K | 词表行切保留每行 dot，不需 logits 求和；greedy 可只汇合原始 score/id。当前两卡 head 分片正是局部 TP，不能外推为整模型 TP8 的收益。 |

普通列切 TP 会改变部分和的加法顺序，原模型 Q5 精度没变也不意味着 logits bitwise 一致。可考虑 **GU 输出行切→精确 Q8_1 hidden allgather→down 输出行切→最终输出 allgather**：每个完整 dot 的 K 维不切，专家 combine 保留原顺序。attention/GDN 的输出投影也能采用这一结构。它避免新增量化误差和部分和归约，但 kernel dispatch、归一化、量化块边界仍需逐行 parity 测试，不能预先保证完全一致。

## 显存：整模型双布局是当前主要约束

现在是 layer pipeline：每卡拥有连续 6 层及这些层的全部 512 专家，每卡 3072 个专家；这不是现成 TP/EP。远端 `profile-groups8/engine.log` 记录 GPU0/7 专家各 11.43 GiB，其余各 10.45 GiB；从 GGUF 精确计算分别为 11.42578125 / 10.44921875 GiB。

若 prefill 仍读原来的完整层权重，同时另建均分 TP8 decode 权重，最理想可以复用原 owner 上的 1/8；仍需新增 **85.546875×7/8 = 74.853516 GiB 专家权重**。这是均匀逻辑分片的估算，不含 padding、dense、head、状态、图和通信缓冲。96/64 的不等 FF 切分会改变逐卡分布，必须逐卡做预算；不会消除这几十 GiB 的重复。

同一旧启动日志在 ownbuf 前的 free GiB 为 `[7.72,7.03,8.81,8.27,8.81,8.27,8.81,5.97]`，随后每卡分配 3.80 GiB prefill 缓冲：

- 合计剩余约 **33.29 GiB**；后续 graph/head 等分配只会减少它。这是旧日志估算，不是本次实时显存查询。
- 即使把全部 prefill 缓冲释放，也只约 **63.69 GiB**，不足上述专家双布局，且下次 prefill 需要还原缓冲。
- 再移除八个 batch slot，按日志每卡 slot 0.15/0.21 GiB 的四卡各一组估算最多约 11.52 GiB；合计约 75.21 GiB，几乎被专家副本吃完，尚无 dense/KV/graph 余量，而且改变了并发服务能力。日志值有四舍五入，不能作为可分配保证。

因此 **当前服务配置不能靠简单常驻第二套 TP8 权重保持原 prefill**。这不是证明所有 TP8 方案都装不下：独立 TP8 引擎只保留一套分片权重可以重新预算；局部层 TP、权重逐块共享/调度也可研究。每次请求把约 75 GiB 权重在 PP/TP 布局之间搬运会引入明显的待测阶段成本，不能算“prefill 不受影响”。

74.85 GiB 不是所有 TP8 设计的不可绕过成本。只保留一套 canonical TP 分片，可以在 prefill 时 gather 临时完整层，或在 PP/TP 阶段之间重排权重；这些传输和分配必须计入 TTFT/首个 decode 间隔。也可以让 prefill 直接使用 TP 分片，但这会改变其执行路径，需要重新证明性能门槛。无 P2P 时，UVA 指针本身不能让 PP owner 直接读取其它七张卡的 VRAM；host-mapped 共享权重则把原来的显存读取换成 PCIe 读取，不能预设速度不变。[CUDA 多 GPU 内存访问规则](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/multi-gpu-systems.html)

## 100k 的 KV/GDN 状态交接

`src/core/session.cpp:38–40`：每 GDN 层状态 `(128×48×128 + 10240×3)×4 = 3,268,608 B`，36 层共 **112.21875 MiB/序列**。head 切分后的总量不增加，但 prefill 的层 owner 和 decode 的 head owner 不同，需要精确 scatter；恢复/append/切回 batch 时还需要 gather 或一致的双状态。

`include/strata/kernels/kv_q8.hpp:25–26`：本机 INT8 KV 含每 64 值 FP16 scale，**1056 B/token/QSA 层**；不是只算 1024 个 int8。12 层 100000 token 为 1,267,200,000 B；indexer pooled FP32 keys 另约 153,612,288 B。两者合计约 **1.323 GiB/序列有效历史**，不含 GDN、RoPE、page table 和 scratch。

现配置 `kv_resident=32768,max_context=262144`：12 层 KV VRAM payload 合计 **396 MiB**；完整历史有 mapped host backing，indexer pooled 容量约 **384 MiB**（有效 100k 部分约 146.5 MiB）。见 `src/core/layer.cpp:733–806`。因此“专家全驻留”并不等于“100k KV 全驻留”。

- Q-head TP8 若一 KV head 给四个 rank，KV 的 aggregate replica 变为四份；当前 396 MiB resident payload 增至 1584 MiB，额外 1188 MiB/序列。indexer 若每卡复制，容量也会膨胀；可另行设计共享/分区。
- token/position 分片可保持 KV 总 payload，避免四倍复制，但需要跨 rank attention softmax 归并和 QSA 选择合并；现有 sFNX 有这条路径，数值/状态布局与当前 engine 不同。
- 可以沿用 portable host KV 作为共同 backing，避免把全部历史复制一遍；各 rank 的 resident pages、pooled keys、GDN state、MTP snapshots/rollback、PLE history 和 conversation cache 仍需适配。
- 初次状态 scatter 的耗时必须计入 TTFT/首个 decode gap，append 切换耗时也需计入。仅保持 prefill kernel 不动不能证明用户看到的 prefill 性能不变。

## 通信量：字节并不大，轮次不能忽略

以下是指定设计的**理想 payload 下界/基线**，不是所有 TP 算法的绝对下界，也不是实测。先忽略 HC、indexer/attention merge、PLE、MTP、flag、padding 和失败重试，设一条主干验证行 `T=1`。**96/192 次是每个 verification window 的 collective 轮数；表中 bytes 是每验证行的量。** 支持批量 T 行时，消息乘 T，轮数不必乘 T。

| 设计 | 每层同步 | TP8 每 rank 发送量，48 层 |
|---|---|---|
| 常规列切、复制 HC/router | attention 输出及 MoE 输出各一次 2560×FP32 allreduce：共 96 次 | `96×2×7/8×10240` = **1.640625 MiB**；每 rank 接收同量，全机发送 13.125 MiB。 |
| 保留完整 dot 的输出行切方案 | 6912 B attention/GDN Q8_1、10240 B 输出、7920 B routed+shared Q8_1 hidden、10240 B combine 输出：每层 4 次 allgather，共 192 次 | `48×7/8×(6912+10240+7920+10240)` = **1,483,104 B ≈1.4144 MiB**。 |
| 当前 PP 主干边界 | `Verifier::handoff_floats` 为 `4×2560+2560+4` float，7 个 stage 边界 | 每边界 51,216 B；全机 7 次单向交接合计 **358,512 B/验证行**，另有控制/返回。 |

无 P2P 的 host staging 路径还会产生相应接收方向 PCIe 流量；常规 allreduce 全机发送+接收 payload 约 26.25 MiB/验证行，不包含跨 NUMA 内存读写。NCCL 官方说明 P2P 不可用时 SHM transport 使用 host memory；这台机器跨 CPU socket 的布置使 NUMA 成为需要实测的变量。[NCCL SHM 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-shm-disable)

`2×(p−1)/p` 的 allreduce、`(p−1)/p` 的 allgather 系数来自 NVIDIA 的 collective 通信模型。它给出 payload/bandwidth 关系，不给出这台机器的延迟。[NVIDIA nccl-tests 性能定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

如果一次有效小 collective 延迟假设为 5/10/20 μs，96 次就对应 0.48/0.96/1.92 ms，192 次加倍；**这些是敏感性例子，均非本机测量**。多行 PW2 验证窗口会一次搬 `T` 行，collective 轮次可以共享；换算到用户输出 token 还必须使用实际接受 token 数，包括被拒绝的草稿和 MTP 工作。不能直接把每验证行通信当成每输出 token 通信。

## 仓库里已经有什么，哪里不能直接用

**最接近的是 `src/split`，不是另一个模型的 TP。**

- `src/split/src/model.hpp:1–3,14–19,60–70`：同一 qwen4exp 几何，每层跨卡、GDN/FF/head 分片，HC/router/embedding 复制。底层列出 2/3/4/6/8；`card_split.hpp:11–26` 已实现 TP8 几何及 96/64 FF 分配。
- `src/split/src/engine.cu:159,277–314`：`FNX_HOSTRING=1` 用 mapped pinned host buffer 交换，不启用 NVIDIA peer access。多卡 one-hop 默认可用；FP16 交换与 F32 ring 是不同数值路径，不能当当前 engine 的 exact 替换。
- **集成限制**：`src/split/CMakeLists.txt:104–106` 只允许构建 2/3/4 卡。`strata_control/split_engine.py:40–76` 的 NVIDIA BUILT/TESTED 为 sm86、sm120，MAX_CARDS=4，TESTED_CONTEXT=131072；当前 sm75、8 卡、配置 max-context262144 都不满足自动选择。底层已有 `<80` 的 m8n8k4 fallback（`moe_mma.cuh:270,307–308`），所以 sm75 不是已证明的数学/指令硬障碍，但还未验证该完整引擎在本机编译、正确性和性能。
- **当前 GGUF 类型不兼容**：`model.cpp:484–485,820–821,867,1322` 分别要求 HC、shared 和 token embedding 的 Q8_0 路径；本机 token embedding Q5_K、HC down Q5_K/up Q5_1、shared GU Q5_K。`model.cpp:263–266` 的列切要求 32 值块，而本机 ssm_out/attn_output 是 Q5_K 的 256 值块。`gguf.cpp:118–122` 仅特定 IQ/Q2 专家触发 small-tensor 转换，本机 Q5 不触发。粗暴强制转 Q8 会改变数值和显存，不能作为无损开关。
- **CLI/执行语义**：独立程序接口是 `strata-split-N --serve --native ... --mtp ... --draft-vocab ... --max-context ... --spec ... --batch ...`；`tools/sfnx.cpp:94` 明确忽略传入的 `--layer-split`。它有自己的 `prefill.cu/session.cpp/batch.cpp`，整个替换会改变 prefill，当前 core 的 PP prefill→TP decode 没有现成状态桥。
- **有 MTP/window，但不是当前 PW2 调度**：`model.hpp:43` 的 MAXT=16；`engine.hpp:48–51,92–100` 支持多行 verify/commit 和 MTP draft。`Session::generate` 在 `session.cpp:471` 验证整个窗口，接受/commit 后在 `:595` 起生成下一轮草稿。`engine_common.cu:853,1462` 分别同步 window/draft 的 stream，未找到当前 core 的两个在途 PW2 窗口调度。不能把它当作只支持 T=1，也不能假定保持本机 PW2 的重叠收益；公平比较必须记录窗口行数、接受率、MTP 耗时及主干/draft 重叠。
- **别的引擎可借鉴**：`src/dsv41/ds4_model.hpp:59` 有 TP group，`ds4_kernels.cuh:286–321` 有 host-mapped handoff/ArGroup，适合复用通信、seq/flag 与 timeout 思路；模型执行图不能直接套到 Qwen 的 HC/GDN/QSA。`src/qwen35/q36_engine.cu` 的当前家族引擎搜索到的是 hybrid expert/MTP 路径，没有找到可直接复用的多 GPU TP collective 实现；不要把 hybrid CPU/GPU experts 称为 TP。

## 后续最小可审核步骤

1. 当前 head 分片 ABBA 完成后再安排独占实验。本次没有启动另一套 TP 引擎。
2. 优先复用 split/DS4 的 host collective，测同 NUMA TP2、TP4、跨 NUMA TP8 的 7–40 KiB payload、T=1/2/3/8，记录 p50/p95 latency、真实 rank 同步轮次和 CPU/NUMA 开销；不需要先装满整模型。
3. 对一个实际 Q5 GDN 层、一个 QSA 层和 MoE 建 decode-only 原型，明确选择列切求和或保留 dot 的行切；比较完整 logits、路由、状态及草稿接受率。先局部层/TP2–4，可以控制重复权重成本。
4. 只有在逐卡显存预算与状态桥明确后再做整模型 TP8。保留原 prefill gate：冷 100k TTFT、warm-cache、append、prefill token/s、首 decode gap、C6/C8 吞吐及输出一致性都需同二进制 A/B；state scatter/恢复时间不能从统计中扣除。

现有 profile 说明有值得研究的并行空间；它不能回答 TP8 会快多少。现成同模型 TP 代码显著降低了起点成本，但“保持当前 prefill、模型原量化与服务并发不变”的整模型 TP8，需要独立的布局和状态设计。

若优先追求下一项可独立验证的单请求优化，旧 trace 更支持先研究 **MTP draft head 两卡行分片**：subset head 2553 次累计 1104.543 ms，比主 head 的 458.899 ms 更大。把它恰好减半只是旧 10.124 秒追踪窗约 5% 的算术空间，绝非实测收益。这里不能照搬只返回 argmax：`mtp.cpp` 的 `row_top_prob` / `chain_prob` 依赖全词表概率，并控制草稿链长度。首版可汇回完整 subset logits，在原卡运行原概率归约，保留数值与 subset→token 映射；但每行 425196 字节的往返和同步可能吃掉收益，必须重新做 logits、草稿接受率和 100k A/B。局部层 TP2/4 则留作后续更大改动，需要更多状态交接和 collective 验证。
