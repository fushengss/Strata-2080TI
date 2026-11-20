# MTP draft head：GPU3+5，保持 prefill 分层

**已完成并部署。** 2026-11-07 19:20（UTC+8）恢复服务，上线验证全部通过。greedy 单请求在固定 100k 场景下，
配对 decode 收益为 code +2.276%、prose +1.681%，未测到持续 prefill 退化。局部专家 TP2/4 原型未部署。

2026-11-07，10.100.3.3，8×RTX 2080 Ti 22 GiB，当前 Q5_K_M 模型。
本目录记录主 head 已在 GPU2+4 的基线上，将 MTP draft head 分为 GPU3+5 两片的实验。
原来的六层/GPU 分层、8 槽、groups=8、PW2、INT8 KV、prefill 缓冲和 CPU governor 保持不变。

候选 `strata.mtp2` SHA256 为
`113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f`。
配置变量仅 `STRATA_MTP_HEAD_GPU=-1` / `3,5`。
源代码在 `/Users/fu/project/HAI/Strata-decode-head` 的 `codex/mtp-draft-head` 分支。
提交 `b4c45a6cc8167ac61407f07c1171b6b4995eec10`，可审阅 [引擎补丁](engine-change.patch)。

## 已完成的低层验证

实际 106299×2560 Q6_K draft subset 分为 53120/53179 行。两卡返回完整 logits，GPU7 保留原概率、选词和强制 token 的计算。
只在支持的 greedy solo PW2 请求启用；采样、penalties、bias、batch 和 prefill 保留原路径。

- 144 个输出，共 15307056 个 logits 逐位一致，概率和词 ID 映射也一致。
- 16 条连续八步链、128 个输出通过，无逐步 CPU 同步，覆盖共享缓冲复用。
- 默认关闭和不支持布局的接口检查通过。
- 首个候选暴露奇数词表尾部的向量拷贝对齐限制；mtp2 改用任意长度的原始 int32 拷贝，保留失败与修复后的日志。

见 [构建和低层结果](../mtp-head-build/candidate2.json) 与 [完整 parity 日志](../mtp-head-build/mtp2-parity.log)。
CUDA 13 / SM75 构建通过；HIP/SYCL 未构建，Metal 仅做共享 C++ 语法检查。

## 首轮 A/B（仅用于决定是否复测）

每臂一次加载，输入约 101k tokens；cold/warm 每次输出 768 tokens，append 输出 384 tokens。
以下是匹配完整请求的引擎计时，不把 SSE 估算值混入同一表。

| 请求 | 关闭 decode tok/s | 开启 decode tok/s | 变化 | 关闭/开启 read ms |
|---|---:|---:|---:|---:|
| cold code | 141.3 | 144.9 | +2.55% | 14650 / 14634 |
| warm code | 142.1 | 145.7 | +2.53% | 见原始结果 |
| append code | 127.1 | 129.5 | +1.89% | 见原始结果 |
| cold prose | 85.1 | 86.7 | +1.88% | 16905 / 16887 |
| warm prose | 85.8 | 87.4 | +1.86% | 见原始结果 |

13 个对应请求（含 warmups）的输入、输出、reasoning、结束原因和 usage 相同。
两臂各 11 个 integration completed requests 完全一致，包含并发、取消恢复和单 token API logprobs 检查。
后者只覆盖 admission 路径，不能替代低层完整 logits 验证。

code 接受数/提议数相同；prose 接受数均为 373，但提议数减少了 3–4 个，说明调度也发生了变化。
因此这里报告的是整个 decode 路径的变化，不把收益全部归因于单个 head kernel。
两类 cold prefill 约 -0.11%；warm code 和 append 的 read 各增加 54 ms，当时据此安排正式复测；此现象未稳定复现。

[首轮对照](pilot35/comparison.txt)、[完整数据包](pilot35.tar.gz)、[归档哈希与凭证扫描](pilot35-archive.json)。

## 正式验证与部署

正式 ABBA 已完成，预设检查全部通过。每臂三轮，92 个主请求、44 个 integration completed requests 和四次取消恢复。
四臂对应输入、输出、reasoning、结束原因与 usage 完全一致；六个 integration 配对全部 exact。
108 个比较的 API logprob 数值来自单 token admission 路径，覆盖范围与低层完整 logits 检查不同。

| 100k 工作负载 | cold decode 关闭/开启 tok/s | warm decode 关闭/开启 tok/s | 配对 cold/warm 收益 | cold read 关闭/开启 s |
|---|---:|---:|---:|---:|
| Code | 143.10 / 146.50 | 143.65 / 146.95 | **+2.276%** | 16.405 / 16.352 |
| Prose | 81.75 / 83.20 | 82.10 / 83.65 | **+1.681%** | 16.6275 / 16.5505 |

吞吐和 read 时间为合并样本中位数；收益单独按每个固定 case 的两次 on 均值 / 两次 off 均值计算，再取六个 case 的中位数。
两类各六个 case 全部改善。首末基线 decode 漂移的中位数为 code -0.067%、prose 0%，范围分别为 -0.211%～+0.131%、-0.122%～0%。
冷 prefill 中位变化为 **-0.323% / -0.463%**；逐固定 cold case 均值变化为 -0.609%～+0.159%。

warm/append 没有持续退化：逐同名 warm case 的 read 均值变化为 -5.265%～+1.208%，append 为 -0.316%～+0.776%。
同名请求的复用 token 数在四臂一致。append-code-2 固定复用 101123、读取 1411 tokens，而 append-code-1 固定复用 101897、读取 639；
二者输入不同，不能跨编号把约 3.7 s 与 1.2 s 的差异归因于 MTP 分片。prose 的少量提议/接受计数随调度变化，原始数据保留了逐请求计数。

[正式评估](abba-final/evaluation.txt)、[完整 ABBA 数据包](abba-final.tar.gz)、[归档哈希与凭证扫描](abba-final-archive.json)。
预设方法见 [evaluation-plan.json](evaluation-plan.json)，八个执行/评估文件在运行前冻结，见 [harness-freeze.json](harness-freeze.json)。

正式 48 个 cold/warm decode 请求的粗 GPU 采样显示，GPU3/5 利用率均值从 17.22%/16.39% 升到 23.43%/22.79%，
GPU7 从 46.84% 降到 37.91%。这支持负载转移观察；约 545–547 ms 的采样不能推断瞬时峰值、SM occupancy 或满载程度。
见 [GPU 采样报告](abba-final/gpu-decode-summary.md)。

旧二进制默认关闭兼容检查通过：11 个固定请求对两个最终 off 臂均 exact，decode 变化范围 -0.235%～+0.176%。
此单次旧二进制启动用于兼容检查，不用于声称精确的默认关闭性能差异。
见 [兼容检查](default-off-compatibility/default-off-compatibility.txt)、[独立 warm/append 审核](warm-append-review.json)。

候选已安装至 `/srv/Strata-0157/engine/releases/mtp-head-20261107/strata`，并在同目录保留实测版本的 `libstrata_kq.so`。
配置仅改变 `/exe` 与 `/env/STRATA_MTP_HEAD_GPU`，主 head 继续为 `2,4`，新增 MTP 为 `3,5`。
仅 greedy solo PW2 获益；采样和并发仍走原路径。
见 [部署判定](deployment-decision.json) 与 [部署收据](deployment.json)。
服务已恢复，18 项上线检查全部通过：实测二进制和 KQ 库已加载、8 槽、全部专家驻留、原 prefill/主 head 设置保留、
48 个 CPU governor 仍为 schedutil。固定 192-token 请求与 ABBA 完全一致，实际主 head 分担 76 个窗口，MTP head 分担 431 步。
见 [上线验证](production-verification.json)、[服务恢复状态](maintenance-restored.json) 与 [上线合成请求归档](production-smoke.tar.gz)。

回滚至本轮之前已部署的主 head GPU2+4 版本，在 10.100.3.3 上执行：

```sh
cp -- /srv/Strata/strata-orca-q5_k_m.57.json.bak-mtp-head-20261107-111739 /srv/Strata/strata-orca-q5_k_m.57.json
bash /srv/Strata/restart57.sh
```

备份含原凭证，仅保存在远端，权限 0600；未复制到本目录。原引擎与库均未覆盖。
部署后配置 SHA256 为 `93492533633510b37d8ec25a825a2c2313eb03dd94dfecbf2e8f84b4b6772be0`，
回滚配置 SHA256 为 `4605e1ae5f9bc9dee6ecdee787f950464707e31abf06e9fbb61890897ce3e76a`。

## 局部 TP2/4

独立测试已经完成，见 [实测报告](../local-tp/README.md)。当前每专家分解 GEMV 原型在 T=1–3 比生产 fused 路径慢，
并未与生产内核全面逐位一致，因此不接入引擎。此结果只否定当前原型，不否定未来融合 TP 内核；未测完整 48 层 TP。
