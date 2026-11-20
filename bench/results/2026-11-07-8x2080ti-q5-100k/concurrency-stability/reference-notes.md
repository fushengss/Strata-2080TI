# 1Cat 并发 MTP 与混合负载参考核实

核实时间：2026-11-07。公开仓库 `git ls-remote HEAD` 与本地参考 checkout 均为 `c98a8d66482333eb8d8b87be72e8856445ac99ec`。下文代码和设计文档固定到这个 commit；PR/issue 页面状态也是本次核实时所见。仓库主干、发布 profile 和已发布 wheel 不应视为完全相同的版本。

并行方式说明：资料中的 TP 数来自作者配置；下文 PP1 表示公开配方没有启用 pipeline parallel、按默认值推断，不是新增的硬件观测。mixed latency controller 另有明确的 PP=1 代码准入；所有这些结果都不构成 PP8 验证。

这份笔记仅提供迁移思路。下列性能数字来自 1Cat 作者或 issue 作者，**不是本机 Strata 的测量，也不能证明本机收益**。本机是 8×2080Ti、SM75、无 P2P 的八段层流水线，原生 Q5 专家全部驻留，使用自己的 Q8_1 激活、int8 KV 和状态布局。与 V100 SM70、TP4、NVFP4/AWQ、FP16 激活和 E4M3 KV 的实现有实质差异。本次整理没有修改引擎、执行服务请求或使用 GPU。

## 1. 竞争时限制 prefill，需要同时保护新请求 TTFT

当前调度器先识别已有 resident decode，预留其 token capacity，再限制同一步的 aggregate prefill；没有竞争时保留正常大块 prefill。长 prompt 阈值也只在至少两个 eligible requests 时生效。`MixedPrefillBudget` 使用完成的 GPU timing，不等待未完成 event；当前代码已支持 `mixed_prefill_min_tokens`、固定开销加每 token 开销的拟合，并避免短尾块和小图样本污染 eager-step 成本估计。

**准入限制：**当前 mixed latency controller 明确要求 `pipeline_parallel_size == 1`，还要求 CUDA、chunked prefill、已有 resident decode 等条件。Strata PP8 只能借鉴预算分配思路，不能直接套用这个调度器。其 250ms 是软目标，预算下限或固定开销可能使实际 step 更长。

主干代码：

- [scheduler.py，resident reservation 和 PP=1 准入](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/vllm/v1/core/sched/scheduler.py#L603-L652)
- [MixedPrefillBudget，当前固定开销拟合与预算下限](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/vllm/v1/core/sched/mixed_prefill.py)
- [配置项及默认值](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/vllm/config/scheduler.py#L85-L102)

最初 #860 的测量使用 Qwen3.8-27B QUASAR NVFP4、DFlash2 q7、4×V100-SXM2 32GB（两个 NVLink pair）、TP4/PP1、CUDA 12.8、Torch 2.10、FP16 compute、E4M3 KV、262144 最大上下文、8192 batch token budget、16 slots、2048/8192 KV/state blocks、align prefix cache、thinking off。两个 resident decoder 加两个新 32K prompt；一次 cold 后取两次 warm 中位数：

| 在线混合负载指标 | 对照 | 自适应 250ms |
| --- | ---: | ---: |
| 新请求 prefill 期间 resident 合计输出 tokens/s | 6.88 | 30.34 |
| resident 最长 stream update gap，秒 | 2.873 | 0.310 |
| 第一条新请求 TTFT，秒 | 14.287 | 17.994 |
| 第二条新请求 TTFT，秒 | 20.631 | 35.491 |

resident gap 改善同时伴随新请求 TTFT 约 +26%/+72%。这是组合候选的在线测量，不能归因于单个 kernel，也不能称为 prefill 无损。来源：[混合延迟设计与完整测量条件](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/mixed_prefill_latency.md)、[已合并 #860](https://github.com/1CatAI/1Cat-vLLM/pull/860)。

更接近低互联场景的用户报告使用 4×V100 PCIe 32GB、无 NVLink、Flash-Next AWQ W4A16 g32、TP4/PP1 + QSA DCP2、MTP4、E4M3 KV、524288 context、四槽、1632-token Mamba state blocks，并有 64GiB CPU KV offload。三条 16K resident 加一条新 64K prompt，固定 1632-token mixed cap 的两次 TTFT 都约 26.9s；最初 250ms controller 得到 286.0/287.7s。该作者估计每 step 固定约 0.2s，再加约 0.32ms/token，小预算使长 prompt 被分为大量步骤。这是 issue 作者的测量，不是我们复现。来源：[仍 Open 的 #880](https://github.com/1CatAI/1Cat-vLLM/issues/880)。

**未合并内容：**#943 在核实时仍 Open，提出 floor 加 state-block rounding。其同机两次实验中，250ms + floor 1632 得到 TTFT 26.99/26.74s，与固定 1632 对照 26.78/26.68s 接近；mixed step p95 约 0.69s，已经超过 250ms 目标。当前主干虽然已有独立的 `min_tokens` 支持和改进拟合，不能据此宣称 #943 的整套 state-block rounding 已合并。来源：[#943 状态、条件和结果](https://github.com/1CatAI/1Cat-vLLM/pull/943)。

当前发布 profile 文档明确将 mixed latency target 设为 0。其 recipe 是四张 peer-connected V100-SXM2 16/32GB、TP4/PP1、27B NVFP4 + DFlash2、FP16 compute、E4M3 KV、4096 prefill budget、四槽；文档解释小块重复支付固定 step 成本，反而长期占用 mixed 路径。这不是建议所有场景关闭控制，而是说明 250ms 不能作为跨硬件默认答案。来源：[SM70 release profile](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/serving/sm70_release_profile.md)。

对 Strata 的判断：先保持纯 prefill 的原路径；若以后改变竞争态 admission/chunk budget，应同时检查 resident gap、注入请求 TTFT、prefill read time、吞吐和恢复。纯 prefill 不变不能替代混合负载 TTFT 防回退。合理最小块和 decode 预留量必须在本机测，不直接采用 250ms 或 1632 tokens。

## 2. 混合批次应保留 MTP 快路径和准确状态边界

1Cat 的混合批次工作有两点可迁移：

- 将 resident verification rows 按 request 分组，保留每行因果长度；使用 device 上权威 `seq_lens`，不能用异步推测下可能只是上界的 host copy。host row layout 每步构建一次，避免每层重复组织。
- GDN verifier 根据 `spec_token_indx` gather/scatter，仅操作对应推测请求的 token/state，不因旁边存在 prefill 就整体退回 generic recurrence。prefix replay 必须同时保存匹配位置的 recurrent state 和 draft sliding-window 数据。

其 attention 案例是 27B NVFP4 + DFlash2、E4M3 KV、page 1648、TP4/PP1、V100；operator timing 在单张卡测量。四条 resident 各八行 verification 加一个 2048-token chunk，32K/131072 context 下逐层 scalar attention 分别 6.01/22.70ms，grouped FP32 分别 1.10/4.08ms。**这是算子时间，不是模型加速倍数。**默认 8192 chunk 的整步仍由 prefill 主导，最大 gap 约 2.87s；文档明确该改动尚未把 decode-only steps 插入 prefill chunks 之间。来源：[mixed-batch decode rows](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/sm70_mixed_batch_decode_rows.md)。

对 Strata 的判断：先完成现有 `STRATA_BATCH_MTP`/`STRATA_PIPED_MTP` 对照，确认真实 batch 路径、接受率、采样、取消与状态恢复，再定位混合 admission 是否导致 dispatch 退化。这不构成重新实现已有 batch MTP 的理由，也不是再推荐已存在的 QSA 多 query 共享 key。

## 3. MTP 的矩阵 batch 不能代替请求并发验证

1Cat Flash-Next MTP4 GDN 文档中的主要 M5/M10 数据，来自**一个请求的 MTP verification 矩阵**。测量合同为 Flash-Next NVFP4、V100-SXM2 32GB、TP4/PP1、FP16 activations/KV、FP32 SSM、MTP4 greedy draft、32768 context、8192 token cap、prefix cache、FULL_AND_PIECEWISE graphs。不能把这些数字当作 C4/C8 公平性或长上下文稳定性证明。

其中 packed GDN input 为每 rank 的 36 层增加 725.625MiB 权重副本，原权重仍供 prefill/M1 使用。该候选的完整轮次中位数为 27.309572→26.495768ms（约 2.98%），但 KV token capacity 从 153910 降为 122631（约 -20.3%），因此仍 opt-in。后续更低轮次数字对应更多合并优化，不能归因于这一个 kernel。来源：[Flash-Next MTP4 batch qualification](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/sm70_flash_next_mtp4_batch_gdn.md)。

对 Strata 的判断：保留 shape fallback、按完整推测轮次统计、记录实际接受 token 和显存/槽位容量值得借鉴。SM70 FP16 Tensor Core 打包路径不能直接替换原生 Q5 dot，也不能在八槽和 prefill 缓冲预算之外额外保留权重而不重做显存验证。

## 4. 通信优化需要服从物理拓扑

1Cat 的 DeepSeek-V4 TP8 CUDA Graph collective 只针对八张 V100-SXM2、两个四卡 NVLink clique 加配对 NV2 links、每 rank 4096 个 FP16 元素（8KiB）、batch-one decode。实现依赖 P2P IPC、peer-visible buffer 和双槽 readiness 协议，准入时验证所需边。它不是无 P2P PCIe 的 host staging 实现。来源：[TP8 collective 的拓扑与同步条件](https://github.com/1CatAI/1Cat-vLLM/blob/c98a8d66482333eb8d8b87be72e8856445ac99ec/docs/design/sm70_deepseek_v4_tp8_graph_allreduce.md)。

#957 fused allreduce/norm 则在 4×V100-SXM2 16GB、full NV2、TP4/PP1、27B NVFP4 target/FP8 head、FP16 DFlash2 q7、256K max context 上测得单请求每 step 约省 0.065ms；作者明确不归因任何并发加速。它说明减少同步可能有价值，但既不能给本机收益预算，也不能把 kernel-only 节省直接加到端点吞吐。来源：[已合并 #957](https://github.com/1CatAI/1Cat-vLLM/pull/957)。

对 Strata 的判断：可以复用“分别测 input-ready skew、启动等待、transport tail、整轮 latency”的诊断方法；不能直接复用 NVLink peer-polling kernel。在无 P2P 的本机，先以已测 host collective 和真实生产专家为基准，保留本次局部 TP2/4 微基准的限制。

## 5. 本轮采用的验证口径

- 独立比较现有 batch/piped MTP 两个开关，不与 prefill guard、量化或 placement 改动混合归因。
- C2/C4 有空槽时的新请求 admission，与 C8 满槽后的排队分别报告。
- resident 的 reasoning/content/tool 都算进度；SSE update gap 不命名为逐 token TBT。MTP 可能一次 chunk 返回多个 token。
- 比较相同 occupancy 的 baseline/mixed/recovery，以及注入请求 TTFT、冷/暖 100k prefill、接受率和结果正确性。算子 parity、单请求 MTP matrix batch、端点并发是不同证据层级。

已有参考报告中的小 M 权重复用、QSA 共享 key、完整推测轮次分账不在此重复作为新候选；本机已否决的全局 V2K 开关也不会因为 1Cat 的其他量化路径有效而重新视为已验证方案。
