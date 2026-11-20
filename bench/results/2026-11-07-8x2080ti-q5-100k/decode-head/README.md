**8×2080 Ti / Q5 / 100k：decode head 词表分片最终基准**

2026-11-07，10.100.3.3。部署状态：**已部署并于 18:08（Asia/Taipei）恢复服务**。原始引擎默认关闭兼容性检查、最终线上进程与生成验证全部通过。

最终同二进制 ABBA 中，`STRATA_DECODE_HEAD_GPU=2,4` 的 case 配对 decode 中位收益为代码 **+1.852%**、散文 **+2.084%**。两类各 6/6 个固定 case 都更快；冷 prefill 读时间中位数分别变化 +0.380%、−0.226%，满足预先约定的 ≤2% 回退门槛。所有输出、路由、缓存和集成检查通过。[完整离线评估](abba-final/evaluation.json)、[紧凑数据表](abba-final/evaluation.txt)。

机器是 8×RTX 2080 Ti 22528 MiB，CUDA 13 / sm75；模型为 Qwen3.8-Flash-Next-Uncensored-Q5_K_M。各卡持有连续 6 层，全部专家常驻显存；保留原有 prefill、`--batch-groups 8`、8 个服务 slot、PW2、spec=3、INT8 KV 与 `kv_resident=32768`。实际冷提示长度为 101104–101130 token。实验只在 loopback:8091 启动独立服务，原服务由维护流程停止；运行器结束时 GPU 进程为空，原配置文件内容未变。[运行清单](abba-final/manifest.json)、[参数与启动状态](abba-final/02-on/metadata.json)。

新增路径只把 PW2 主 head 的词表行分到 GPU2 和 GPU4，保留每行 dot 的计算顺序，再汇合 score/id；它不改变专家量化。两卡模式只对 greedy 生效，`temperature>0`、logit bias、presence/frequency/repetition penalty 等请求回退原路径。默认不设置或设 `-1` 时关闭。单卡模式也支持无历史惩罚/无 bias 的采样，但最终测量与候选配置为两卡 greedy 路径。

三阶段结果如下；pilot 各只有一次 off/on、每类一个固定 case，不能代替最终 ABBA。

| 阶段 | 比较 | 100k cold/warm engine decode 收益 | 结果 |
|---|---|---:|---|
| GPU4 整个主 head 移动 | 同 candidate 二进制，-1 vs 4 | 0.000%–0.238% | 输出/路由通过；收益不足 |
| GPU2+4 词表分片 pilot | 同 sharded 二进制，-1 vs 2,4 | 1.944%–2.161% | 输出/路由通过；冷读时间 +0.480%/+0.390% |
| 最终 ABBA | 同 final 二进制，off/on/on/off | 代码 +1.852%，散文 +2.084% | 全部门槛及原引擎兼容性通过，已部署 |

两个 pilot 的原始证据为 [GPU4 archive](raw/pilot-gpu4.tar.gz)、[GPU2+4 archive](raw/pilot-shards24.tar.gz)，摘要 SHA 在 [pilot index](raw/index.json)。最终源码提交 `235e20211a243d48b8af838794596fa38c9dfd95`；四臂均使用 `strata.final`，SHA256 `372da958cd4b1ba25b2ab42a7e92480c4cb31973994ebc8d8355e5644766fc22`。每臂确认二进制、基准脚本哈希不变；除 head flag 和日志路径外配置相同，CPU governor 均保持 schedutil。[构建验证](../decode-head-build/final-build-verification.json)。

最终每臂先做 2 次短预热以吸收 graph capture，再做 6 个短正确性 case，随后代码/散文各 3 次 cold 和同提示 warm，另做 3 次代码 append。cold/warm 固定 greedy、固定 768 输出 token；append 使用所有臂相同的已存 assistant 文本和续接内容，固定 384 输出 token。冷请求实测 0 token 复用；warm/append 实测复用超过 90000 token，不能把缓存恢复当完整 prefill。

预先约定的主要收益定义：对每一个固定 case，先计算 `mean(两次 on engine tok/s) / mean(两次 off engine tok/s) - 1`，再对该类型的 6 个 cold+warm case 取中位数。代码/散文分别要求 ≥1%，并且多数 case 改善。另报告相邻 ABBA 配对 B1/A1、B2/A2 的变化；其代码/散文中位收益为 +1.862%/+2.057%，全部 24 个相邻 decode 配对为正，范围分别 +1.659%–1.996%、+1.990%–2.242%。两个 baseline 首末漂移的 case 中位数仅 −0.107%/−0.059%。[评估规则](evaluation-plan.json)、[离线 evaluator](evaluate_final.py)。

下表绝对速率为对应类别六次观测的中位数；收益为上面的固定 case 均值比中位数，因此不要求等于两列中位数直接相除。client 是首末非空 SSE 的 `(usage completion_tokens−1)/时间` 估计，与 engine 计时独立保留。

| 工作负载 | engine off→on tok/s | engine case 收益 | client off→on tok/s | client case 收益 |
|---|---:|---:|---:|---:|
| code cold | 140.35 → 143.05 | +1.802% | 140.67 → 143.37 | +1.785% |
| code warm | 140.80 → 143.60 | +1.903% | 141.09 → 143.86 | +1.861% |
| prose cold | 80.10 → 81.75 | +2.060% | 80.15 → 81.83 | +2.089% |
| prose warm | 80.35 → 82.05 | +2.109% | 80.38 → 82.10 | +2.101% |
| code append | 123.70 → 126.25 | +2.130% | 124.13 → 126.66 | +2.108% |

prefill/TTFT 单独审阅。冷 prefill 门槛使用每类型六次 on 与六次 off 的读时间中位数；下表同时保留固定 case 配对中位变化。TTFT 包括服务器与传输开销，不能用 decode 速率替代。

| 工作负载 | engine read off→on 秒 | read 相邻配对中位变化 | client TTFT off→on 秒 | TTFT 相邻配对中位变化 |
|---|---:|---:|---:|---:|
| code cold | 16.3330 → 16.3950 | +0.082% | 16.5192 → 16.5508 | +0.082% |
| code warm | 2.3480 → 2.3285 | -3.762% | 2.4435 → 2.4184 | -3.547% |
| prose cold | 16.6185 → 16.5810 | +0.252% | 16.8043 → 16.7627 | +0.252% |
| prose warm | 2.3595 → 2.3120 | -2.202% | 2.4540 → 2.4066 | -2.167% |
| code append | 1.1395 → 1.1570 | +0.312% | 1.2434 → 1.2555 | +0.066% |

冷 prefill engine 吞吐中位数：代码 6191.6→6168.35 tok/s，散文 6083.8→6097.6 tok/s。六个 cold case 各自两次 on/off 均值的读时间变化为代码 +0.082%、+0.421%、+0.085%；散文 +0.375%、−0.199%、+0.178%。冷 TTFT 的全部相邻配对最坏 +0.908%，未达到 2%。

warm/append 按每个 case 的两次均值比较，最大读时间增加为 append-code-1 的 **17.5 ms（+1.536%）**；另外两个 append 分别 −44.5 ms、+11 ms，warm 的最大均值增加仅 +0.5 ms。单次仍有波动：warm-code-3 的一个相邻 pair 增加 52 ms（+2.267%），TTFT 增加约 62.4 ms（+2.621%），另一个 pair 更快，该 case 的平均读时间反而减少 29.5 ms。因此 17.5 ms 是 case 均值比较的最大增加，不能解读为单请求最坏延迟上限。本轮没有看到持续的 warm/append 回退；[独立复核记录](ab-review.json)已接受该有限样本波动，默认关闭兼容性检查随后通过。

验证证据：

- 四臂共 92 个主测试请求：8 个预热、24 个短正确性请求、60 个性能请求。对应输入、输出文本、reasoning、输出 token 数均一致；cold/warm/append 的缓存要求和所有预期 offload/fallback 路由通过。短 case 含 greedy、seed17/29 采样、penalty、bias、thinking。thinking 由服务拆成两次 engine 调用，只用于 API 正确性，不把某一段当完整请求计时。
- 每臂 11 个完成的集成请求，加 1 个主动取消流。总计 44 个完成请求、4 个取消；8 组 concurrent2 的实际内容流重叠 4.298–4.363 秒。取消后的两条 marker 响应均等于控制请求，服务最后 idle。六个臂间组合共 66 个请求配对全部精确一致，包含 off/off 和 on/on 重复性比较。
- 每臂 3 个 logprobs 请求；六种臂间组合共 18 组 logprobs 比较、108 个返回 logprob 数值比较，最大绝对差均为 0。parallel8 的 API 只允许 max_tokens=1，因此这是 admission 路径兼容性检查，**没有覆盖 PW2 offload 的 API logprobs**。后者的数值依据来自下述底层完整 logits 测试。
- 最终 CUDA 构建的底层测试通过：128 个 head window、111247360 个 logits bitwise 一致；覆盖 T=1/2/3/8、IL/argmax 设置、greedy/固定种子采样、非有限值标记、两个 parity buffer 和 host/event 交接。另有 16 个两卡分片 window，完整 logits、gather score、greedy 合并选择和非有限值 guard 全部 bitwise/精确通过。CPU shard-argmax 合并边界测试通过。本轮未发现 CUDA、非有限值或 API 错误。

[集成六组比较索引](abba-final/integration-comparison-index.json)；[最终底层日志](abba-final/head-parity-final.log)；[各项 gate](abba-final/evaluation.json)。

资源成本：每张目标卡新增约 **497 MiB 的 Q6_K head 权重**。包含运行缓冲后，本轮 GPU CSV 的显存占用中位数实测 GPU2 为 18025→18533 MiB、GPU4 为 18029→18537 MiB，即各 +508 MiB；GPU7 为 20421→20455 MiB（+34 MiB），其余卡中位数相同。两张 on 臂与两张 off 臂分别一致。原 prefill 分配和 8 slot 没有缩减。[off GPU samples](abba-final/01-off/gpu.csv)、[on GPU samples](abba-final/02-on/gpu.csv)。

支持范围：CUDA 13/sm75 已完成构建、底层正确性、单请求 100k ABBA 与并发/取消/API 集成验证。HIP 没有进行完整编译或设备测试。Metal 只编译了 stub 与实际 shared verifier，做了 relocatable link 和 CPU contract/merge 测试；没有完整 Metal 引擎构建或设备执行。不能把 CUDA 结果当作 AMD/Metal 验证。[可移植性检查](../decode-head-build/metal-portability-verification.json)。

这里的 100k 是固定合成长提示与固定代码/散文生成，不包含生产用户内容；收益适用于此次 isolated solo greedy 测量，不能外推所有聊天任务、采样或多请求吞吐。全部专家驻留也不代表 100k KV 全驻留。进一步 TP8 的权重布局、量化块、状态交接与通信限制见 [TP8 可行性分析](tp8-feasibility.md)；该文的性能估算没有作为本报告实测。

完整原始证据已归档 322 个隔离测试文件、32043957 字节，包含合成请求 JSON/SSE、engine/client/server 日志及 GPU samples。逐文件检查了实际 API-key 字节，命中 0；未导出 credential 配置或生产服务日志。[full archive](raw/abba-final-full.tar.gz) SHA256：`c4e7fd5a4cf65d3130759f3194440ddf43e8bfeb561247bc9cf01300608fc187`。[逐文件来源/导出哈希收据](abba-final/archive-receipt.json)、[archive SHA receipt](raw/abba-final-full.tar.gz.sha256)。另有 [紧凑证据 archive](raw/abba-final.tar.gz)，SHA256 `168de3c42d961545bb59568c288937e3bef2d524cb0d03a6d323d4baf4abf254`。

离线复算（不接触远端服务）：

```sh
python3 /Users/fu/project/HAI/Strata/bench/results/2026-11-07-8x2080ti-q5-100k/decode-head/evaluate_final.py \
  --root /Users/fu/project/HAI/Strata/bench/results/2026-11-07-8x2080ti-q5-100k/decode-head/abba-final \
  --parity-log /Users/fu/project/HAI/Strata/bench/results/2026-11-07-8x2080ti-q5-100k/decode-head/abba-final/head-parity-final.log
```


原始引擎兼容性复核使用相同源码基线的未修改重建二进制 `strata.baseline`，SHA256 `12ceada2901a3f59d72f21aca4f29617d17a74efce8acff7ecce1e310262c135`，同一套冻结脚本、配置、模型和固定输入，另起一轮 off、repeats=1。11 个非预热请求与最终 ABBA 的两个 off 臂分别对照，输入、输出全文、reasoning、usage 和结束原因全部一致；启动参数只允许可执行文件路径不同，配置另允许测试日志路径不同。五个性能 case 的默认关闭 decode 差为 −0.080% 至 +0.060%，仅作兼容性参考，不把一次原始启动作为精确速度回归估计。[默认关闭复核结果](default-off-compatibility/default-off-compatibility.json)、[摘要](default-off-compatibility/default-off-compatibility.txt)、[原始运行归档](raw/compatibility-original.tar.gz)。该单臂运行器附带的 ABBA 摘要会提示不完整，正式结论使用上述 original-vs-final comparator。首次启动预检因上一轮端口 TIME_WAIT 拒绝，未启动进程/测量；端口释放后用完全相同的脚本重新运行。

实际部署路径为 `/srv/Strata-0157/engine/releases/decode-head-20261107/strata`，配套同目录 `libstrata_kq.so`，两者 SHA 与最终构建一致。线上配置只改 `/exe` 和 `/env/STRATA_DECODE_HEAD_GPU=2,4`；原分层、所有 prefill 选项、8 slots、groups8、API key、监听配置、CPU governor 以及其它选项均保留。原二进制没有覆盖。源码位于 `/Users/fu/project/HAI/Strata-decode-head` 的 `codex/decode-head-offload`，基于实际上线的 upstream 0.1.57；原工作区较旧 main 的既有改动没有改写。[部署决策](deployment-decision.json)、[安装与私有远端备份记录](deployment.json)、[可应用源码补丁](decode-head.patch)。

恢复后的只读进程检查确认：实际 executable SHA 正确，head 环境变量为 `2,4`，已测 K-quant 库实际映射进进程，全部 24576 个专家仍驻留，CPU 48 个 policy 均为 schedutil，服务报告 0.1.57 且 requested/serving 均为 8。线上 synthetic greedy 请求得到与 ABBA 完全相同的 192-token 回答，日志记录 79 个真实 offload window，无请求或 CUDA 错误；测试端口 8091 已无监听者，最后活动为 1 个已完成请求、0 在途。[线上验证](production-verification.json)、[恢复状态](maintenance-restored.json)、[验证请求归档](raw/production-smoke.tar.gz)。图像 reader 的配置路径保持原值，与原 package 中的二进制 SHA 相同；本轮没有重跑图片性能测试。[reader 文件核对](image-reader-verification.json)。

回滚在远端执行以下命令，恢复本次改动前的 groups8 配置与原引擎路径；私有备份只保存在远端、权限 0600，没有导出凭据：

```sh
cp -- /srv/Strata/strata-orca-q5_k_m.57.json.bak-decode-head-20261107-100607 /srv/Strata/strata-orca-q5_k_m.57.json
bash /srv/Strata/restart57.sh
```
