# Piped decode 优化验证报告

状态：已完成 P1/P2 独立 A/B、P3 分段诊断及上线验收。生产已启用 P1 输出头分担；P2 关闭，P3 不部署；8 个槽位在线，八路采样 smoke 通过。

## 已验证结论

P1 将并发流水线的 target 输出头交给 GPU 2/4，保留原 prefill 分层、完整 logits 返回和原采样规则。两种测试顺序下，C8 完整请求吞吐分别提升 **4.97%**、**4.61%**。100k cold/warm prefill 的读入耗时和首输出延迟中位数均未上升。正确性、预算、取消及恢复检查通过。

收益目前主要见于八路短 prompt 持续 decode。四路混合新 prompt 的整场吞吐接近持平，长 prefill 造成的约 **10 秒无输出停顿仍存在**。四条 cold 100k 请求的整场吞吐没有提升。不能据此声称所有并发业务都加速，或稳定性问题已经解决。

## 配置与测试口径

- 机器：10.100.3.3，8 × RTX 2080 Ti，Q5_K_M，8 个服务槽位；全部专家驻留显存。
- 分层保持 `6,12,18,24,30,36,42`；target head 原位置配置 `2,4`，MTP draft head `3,5`。
- P1 A/B 使用同一个引擎，仅改变 `STRATA_PIPED_TARGET_HEAD=0/1`；每张卡保留独立 3.80 GiB prefill 缓冲。
- P1 引擎 SHA256：`8b27e040d26b51a67cfef4d7c7154fa8cc83f6d0a1cf1b1826f9eb0295bd30ab`。
- 共享库 SHA256：`fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661`。
- server SHA256：`40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9`。
- 原生产配置 SHA256：`f5376cc83215fc54b46d20094373f20a41b93daa8bf07262d2ac67dc13297220`。隔离 A/B 阶段原配置保持不变，manifest 中无剩余测试 GPU PID；最终生产配置的有限变更见部署部分。

API 吞吐使用最终 completion token 总数除以整场请求时间，包含 admission、TTFT 和结束阶段。Native 吞吐使用原生 piped busy block 的实际输出数除以该区间耗时，也包含区间内 admission 停顿。两者边界不同，不能混用。Verification rows 含提案，不能当作实际输出 token。

## 锁频与硬件采样

驱动接受了八张卡的 `-lgc 2000,2000` 请求；功率上限没有更改，采样时仍为每卡 250 W。**请求锁频不等于实际达到 2000 MHz**。

以下是第一轮 C8 pilot 的八流共同输出区间、两端各去掉 1 秒后的采样均值；off/on 各约 28/26 个样本，不是整个业务周期的利用率：

| 指标 | P1 off | P1 on |
|---|---:|---:|
| 八卡平均 GPU active | 64.46% | 69.89% |
| 八卡平均功率之和 | 1739.5 W | 1848.7 W |
| GPU 7 active | 96.89% | 80.46% |
| GPU 2 active | 57.54% | 74.88% |
| GPU 4 active | 54.86% | 74.27% |
| GPU 7 实际 SM 时钟均值 | 1671 MHz | 1852 MHz |

GPU 7 的采样均触发软件功率限制；P1 on 后 GPU 2/4 也在该区间每个样本触发功率限制。该证据支持输出头工作由 GPU 7 分担到 GPU 2/4，不能解释为所有 GPU 已满负载。NVML active 是采样活动比例，不是 SM occupancy、FLOPS 或显存带宽利用率。

## P1 性能与正确性

### C8：两种顺序复现收益

每组八条请求，各输出 1024 token；四条 sampled、四条 greedy。下表均按 off → on 便于比较，第二轮实际运行顺序是 on → off。

| 范围 | off | on | 变化 |
|---|---:|---:|---:|
| 第一轮 API tok/s | 285.903 | 300.119 | +4.97% |
| 第二轮 API tok/s | 271.876 | 284.404 | +4.61% |
| 第二轮 native 输出 tok/s | 297.411 | 311.990 | +4.90% |

两轮的绝对基线有波动，因此不跨轮直接相减归因。第二轮 native emitted rows 为 8180/8181、accepted 为 3499/3500，提案 offered 都是 4675；API 两组都为 8192 token 且输出精确一致。Native 计数边界不是 API 完整输出边界，不能泛称所有 MTP 计数相同。

### 100k 单请求：cold/warm 各三次

实际 prompt 为 101,139 token，输出 256 token。Cold 从零读入；warm 复用 101,132 token、读入 7 token，read_ms 包含恢复成本。

| 中位数 | off | on | 变化 |
|---|---:|---:|---:|
| cold read | 16.481 s | 16.453 s | -0.17% |
| cold TTFT | 16.668 s | 16.646 s | -0.14% |
| warm read | 2.263 s | 2.187 s | -3.36% |
| warm TTFT | 2.351 s | 2.289 s | -2.63% |
| cold solo decode | 122.8 tok/s | 122.4 tok/s | -0.33% |
| warm solo decode | 123.2 tok/s | 123.3 tok/s | +0.08% |

未观察到 prefill 中位数退化，solo decode 基本持平。每类只有三次，warm on 的单次 read 最大为 2.491 s，off 最大 2.266 s；这些数据不足以证明尾延迟 SLO。256 token 的 decode 也不是长时间单请求测速。

### C4 mixed：新 prompt 仍造成明显停顿

四个 resident 请求与两次注入，全部 sampled。标称 10k/60k 的 fixture 实际为 **8794/52081 token**，低于 ±10% 容差；两组实际请求完全匹配，但不能将它写成严格 10k/60k 测试。

| 指标 | off | on |
|---|---:|---:|
| 整场 API tok/s | 105.688 | 105.532 |
| native 输出 tok/s | 113.529 | 113.317 |
| 8794-token 注入 TTFT | 5.825 s | 5.839 s |
| 52081-token 注入 TTFT | 10.079 s | 10.043 s |
| mixed 最大无输出间隔 | 10.501 s | 10.380 s |
| mixed 字节进度 Jain 公平性 | 0.99955 | 0.99580 |
| mixed 至少一流超过 1 s 无输出的采样占比 | 73.81% | 71.43% |

C4 native 日志没有墙钟时间戳：通过套件执行顺序、前面的 C8 和后面的 budget/long 原生数字摘要逐项匹配，唯一定位中间的 C4 batch。没有用 head verification rows 代替实际输出。

恢复指标的“连续两个 2 秒分箱超过 baseline 字节率 90%”起点为 2 → 6 秒。Off 的 4–6 秒分箱为 1001 B/s，刚超过阈值 1000.404；on 对应分箱为 969.5 B/s，低于阈值 982.226。两组恢复阶段每个分箱最大 chunk gap 都约 25–27 ms，**不能把该阈值结果解释为额外停顿 4 秒**。分阶段 token 率按 usage 与文本字节分摊估算，不是真实逐 token 时间戳。

### 预算、取消与四路 cold 100k

完整反向验证独立比较 34 个匹配用例：33 个完整请求的请求/输出哈希、结束原因和逻辑 usage 精确一致；1 个主动取消请求比较有效、非空共同前缀。四条 reasoning-budget continuation 都只新增读取 19 token，取消后 recovery 输出与控制请求一致。Recovery 的缓存 metadata 可由 0 变为 29，而逻辑 token 计数保持一致，应分开记录。

四条 cold 100k 请求都输出 512 token，公共 SSE 输出区间分别约 7.361/7.420 秒。整场 API 吞吐为 26.203 → 25.862 tok/s（-1.30%），native 为 34.535 → 34.422（-0.33%）；最大 TTFT 为 70.416 → 71.312 秒。这一场景包含四个长 prompt 的 admission/prefill，不能用作纯 decode 吞吐，且本轮没有显示收益。

P1 门槛意见：正确性和单请求 100k prefill 中位数门槛通过，可作为重复测得约 5% C8 收益的部署候选，并作为 P2 独立 A/B 的固定基线。混合业务稳定性收益尚未证实，约 10 秒 admission 停顿仍是后续重点。

## P2：异步采样——正确性通过，生产不选用

实现使用每行持久 pinned/device history，并在采样事件完成前保持最后一级占用。本轮固定 P1=1，按 off → on 顺序，仅切换 `STRATA_PIPED_ASYNC_SAMPLE=0/1`；两组使用同一候选引擎 SHA256 `ff2cc241def6a4636fea6ce20d4b7ba5ca7554bf4025304e5ce903af4079e778`。引擎、实际映射的共享库、server、原配置与 14 个冻结测试文件的身份检查通过，原配置未变，测试结束无剩余 GPU 进程。

六套测试均完成，独立审计结果为 `COMPLETE_CORRECTNESS_IDENTITY_PASS`。比较共 **48 个匹配用例：46 个完整请求精确一致，2 个主动取消请求共同前缀一致**。完整请求按请求/输出哈希、结束原因和三项逻辑 usage 比较；保存全文的 C8、correctness、budget 还重算正文、reasoning 与 SSE 拼接结果。Prefill、mixed、long 仅保存通道哈希，审计重算组合哈希，不声称重新获取了原始文本。取消用例不要求截断长度或取消 usage 相同；两套取消后 recovery 均与各自控制请求输出和逻辑 usage 一致。这证明取消后的恢复正确，不证明复用了同一个物理 slot。

History fixture 的实际发送值为 `mixed-greedy: last_n=128, repetition_penalty=1.05`、`mixed-sampled: last_n=32, presence_penalty=0.7, frequency_penalty=0.1`、`mixed-penalties: last_n=257, presence_penalty=1.1, frequency_penalty=0.3, repetition_penalty=1.05`；两组分别完整输出 384、448、512 token，均超过对应历史窗口，另有 `logit_bias={'32':1.0}` 用例。On 的 C8 实际完成 **2395 个异步采样窗口、4788 个 sampled rows**，correctness/mixed/budget/long 也都有正计数；off 全部为零，prefill 两组均为零。所有计数已与原始引擎日志逐项匹配。这些是采样工作量，不能当作接受的输出 token，也没有逐个 history 请求的 GPU 拷贝计数。四条预算续接两组均只新增读取 19 token。

| 指标 | P2 off | P2 on | 变化 |
|---|---:|---:|---:|
| C8 完整请求 API tok/s | 284.346 | 283.843 | -0.177% |
| C8 native 输出 tok/s | 313.112 | 312.323 | -0.252% |
| C4 mixed 整场 API tok/s | 106.285 | 106.516 | +0.218% |
| C4 mixed native 输出 tok/s | 114.081 | 114.367 | +0.251% |
| mixed 最大无输出间隔 | 10.489 s | 10.449 s | -0.385% |
| 四路 cold 100k 整场 API tok/s | 26.245 | 25.678 | -2.161% |
| 四路 cold 100k native 输出 tok/s | 34.429 | 34.380 | -0.143% |
| cold 100k read 中位数 | 16.364 s | 16.555 s | +1.167% |
| cold 100k TTFT 中位数 | 16.550 s | 16.754 s | +1.232% |
| warm 100k read 中位数 | 2.199 s | 2.215 s | +0.728% |
| warm 100k TTFT 中位数 | 2.296 s | 2.321 s | +1.093% |

这是一次 off/on 配对。Cold/warm 各仅三次，read 样本区间分别为 off/on 的 **14.524–16.898 / 14.582–16.872 s**、**2.161–2.568 / 2.087–2.277 s**，明显重叠；不能仅据约 1% 的中位数差异认定 prefill 退化，也不能证明严格零损失。四路 100k 整场吞吐包含 admission/prefill 与结束阶段，不是纯 decode。Mixed 最大停顿仍约 10.45 秒，恢复与公平性没有显示足以支持启用 P2 的稳定收益；mixed 阶段 resident 字节率还由 108.04 降至 102.29 B/s，但这是窗口内进度代理，不是逐 token 测速。

**P2 不选入生产配置：本轮正确性通过，但 C8、混合接入和长上下文没有证明实际收益。** 生产选择保留 P1 引擎 `8b27e040d26b51a67cfef4d7c7154fa8cc83f6d0a1cf1b1826f9eb0295bd30ab`，保持 `STRATA_PIPED_TARGET_HEAD=1`、`STRATA_PIPED_ASYNC_SAMPLE=0`；最终安装与启动是否完成，以后文部署回执为准。不得把 P1 已测得的收益归给 P2。完整数字及来源绑定可由同目录 `p2-validation-analysis.py/.json/.txt` 离线复核。

## P3：MTP 分段观测——C8 诊断完成，未实现 batching

P3 是默认关闭的每槽位 CUDA event 观测，**没有实现 MTP body packing/batching，也没有已测得的性能收益**。源码提交为 `fb73be5a44fe4101fd1773f6fb4414b6344551d4`。本次固定 P1=1、P2=0、P3 观测=1，`01-off` 目录名仅代表 P2 关闭，不代表观测关闭。

首次引擎 `b987f212fe8c50b9c2e4e1cde0b7c53e5934f37f571ba6353dbe8087d45c1a41` 在 ready 前崩溃，**构建无效，排除出运行结果**：归档源码时间早于旧对象文件，增量构建重编了 `mtp.cpp`、`prefill.cpp`，却没有重编 `generate.cpp`；新增 `MtpDrafter` 成员使新旧对象布局不兼容。这次失败没有有效分段计时。证据保留于 `p3-build/`、`p3-diagnostic/` 和 `p3-startup-independent-review.json/.txt`。

保持同一提交及源码内容，更新构建输入时间后，三个相关编译单元均重编通过。新引擎 `strata.piped-mtp-observe2` 的 SHA 为 `f36c12833e900a77cdc966298863f15ded275696855405816b4b2377208683b2`，构建用时 82.52 s。第二次启动、8 槽位、专家全驻留、各卡 3.80 GiB prefill buffer 和观测事件检查通过；C8 的 8 条正常完成请求与 P2-off 基线的输出、usage、finish reason 完全一致。

**测量范围仅为短 prompt C8：每条实际 66/68 prompt tokens、1,024 output tokens，四条 greedy、四条 sampled，共 8,192 API output tokens。** 本轮没有 100k 或 mixed admission 的 P3 分段诊断，也没有同一二进制下观测开关 off/on 的开销 A/B；不能据此宣称观测无开销、长上下文表现相同或取消/工具调用路径通过。

单个完整 drain 报告包含 4,673 条完成的 draft chain，本配置每条只有一个 draft step。按所有槽位的 `sum_ms / count` 汇总：

| 每个 draft step 的 source-stream 区间 | 次数 | 加权平均 | 观测最小 / 最大 |
|---|---:|---:|---:|
| body | 4,673 | 0.620737 ms | 0.428032 / 0.689056 ms |
| head return | 4,673 | 0.319316 ms | 0.302496 / 0.364544 ms |
| tail | 4,673 | 0.030594 ms | 0.022528 / 0.066656 ms |

同一步的三个区间平均合计 **0.970647 ms**。Body 是这三个区间中最大的；head return 包含激活发布、远端投影、依赖等待和 logits 返回，body 不包含此前 residual staging 的等待。它们不是纯 kernel 活跃时间，不是整机 wall time 或 GPU occupancy；不同槽位区间可重叠，不能据相加总量估算吞吐或可获得的加速。

Host 记录了 4,681 次 verdict，4,680 个相邻 gap 平均 **5.603772 ms**、最大 **818.959927 ms**。该范围包含 admission、drain/refill 和调度，不能把最大 gap 归因给 MTP，或把平均值视为纯 decode 的到达间隔。Pump 入口的 submitted-but-unreaped 直方图为 **0 条：3,293,396 次；1 条：671,338 次；2–8 条：0 次**。这类链可能已在 GPU 完成但尚未回收，因此这些数值仅是 pump 样本计数，不能解释为耗时占比。Verdict 前 unsubmitted jobs 为 **0 条：8 次；1 条：4,673 次；2–8 条：0 次**，符合 8 groups × 1 slot 的当前结构；它们尚未执行 body，也不是 post-attention ready queue。

出现 **665,896 次 stage0 draft-not-ready group skip**，说明调度确实遇到 draft 未就绪。循环对同一条件可重复计数，不能说有同样数量的独立慢请求，也不能换算成累计停顿时长。当前采样没有显示多条未回收链的积压，但这不证明等待凑批会有收益。下一步若验证 packing，应先测量兼容任务真实 ready 的时间和重叠，再保留仅一条 ready 时立即执行的路径；不能仅凭 body 平均耗时较大就认定它位于整机关键路径。

诊断数值及来源绑定可由 `p3-diagnostic-analysis.py/.json/.txt` 离线复核，16 项审计检查通过；原始有效观测在 `p3-diagnostic-v2/`，八路正常完成的一致性检查在 `p3-compatibility/`。**P3 只作为诊断，不加入生产；生产选择仍为已验证的 P1 引擎，P2 关闭。**

## 最终部署与恢复——已完成

生产于 **2026-11-07T23:58:01+08:00** 通过新启动验收；`loaded=true`、`serving=8/requested=8`。实际进程为 PID 102086，加载的是固定的 P1 候选，而非 P2/P3 实验引擎。

- 路径：`/srv/Strata-0157/engine/releases/piped-decode-20261107/strata`。
- 引擎 SHA256：`8b27e040d26b51a67cfef4d7c7154fa8cc83f6d0a1cf1b1826f9eb0295bd30ab`。
- 生产配置 SHA256：`f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd`。
- 配置开关：`STRATA_PIPED_TARGET_HEAD=1`、`STRATA_PIPED_ASYNC_SAMPLE=0`；MTP 观测、原先被拒绝的 read guard 保持未启用。
- 安装只改变 `exe` 和以上两个 env；API key、其余配置、分层、server 及共享库逐项保持验证通过。运行中的 `/proc/exe`、库映射及全部配置 env 也已匹配。
- 新启动确认：8 张卡均有独立 3.80 GiB prefill buffer，全部专家驻留，MTP draft head 仍在 GPU3/5，target head 的并发路由为 GPU2/4。
- 回滚副本仅在主机 `/srv/bench100k-20261107-piped-target-head/rollback-piped-decode-5yijad5j`，目录 0700、配置 0600。没有下载私有配置。

上线后八条 sampled synthetic 请求全部正常完成，各 128 token，共 **1024 个实际 API completion tokens**；状态计数只增加 8 条，测试前后没有 in-flight 请求。日志证实 target head offload 已运行（662 windows / 1322 verification rows），MTP head 也有正计数，async sampler 无启用计数。这是功能/路由 smoke，不是性能 A/B；verification rows 不等于输出 token。上线前的取消/恢复门槛已由隔离测试验证，此次线上 smoke 不额外声称覆盖取消场景。

服务恢复后再次对全部 8 卡执行 `-lgc 2000,2000`，驱动逐卡接受，仍为 250 W 上限。`clock-final.json` 的即时频率快照不作为持续负载频率证明；负载中的实际频率和功率限制见前面的 C8 数据。

完整回执：`deployment-receipt.json`、`maintenance-restored.json`、`deployment-verified.json`、`deployment-smoke/deployment-smoke.json`、`clock-final.json`。生产当前已恢复，未留下测试服务或要求用户补做的上线步骤。

## 审计材料

同目录 `p1-validation-analysis.py/.json/.txt` 可离线重建全文所用 P1 完整比较；`p1-pilot-analysis.py/.json/.txt` 保存 pilot 性能、硬件采样与局限。`p2-p3-independent-review.txt` 为代码静态审查；`deploy_piped_decode.py`、`DEPLOY.md` 和 `deploy-offline-check.json` 为部署准备。
