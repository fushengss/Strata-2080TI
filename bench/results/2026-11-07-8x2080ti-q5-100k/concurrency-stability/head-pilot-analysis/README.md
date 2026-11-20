# 共享 piped MTP head 单轮对照

本报告仅使用 head-pilot 的一次 off/on 对照；两臂 batch/piped MTP 始终开启、预算 cap=true，仅 `STRATA_PIPED_MTP_HEAD` 从 0 改为 1。

完整性与身份检查：全部通过。

引擎 `c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6`；server `40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9`。实际进程 exe/映射库、冻结 harness、8 槽、全专家驻留及每卡 3.80 GiB prefill buffer 分别检查。

| 检查 | 成对数量 | 全部通过 |
|---|---:|---|
| 预算完整请求及取消共同前缀 | 10 | True |
| 100k 冷/暖 prefill | 6 | True |
| C8 完整请求 | 8 | True |
| C4 resident+injector | 6 | True |

预算项含 9 条完整请求和 1 条取消流；取消只要求共同前缀相同，不要求完整 hash 相同。prefill 另外检查实际输入长度、cold cache=0、warm 精确 cache/read；性能请求检查完整输出、分通道 hash、全部 usage。

| 指标 | off | on | on 相对变化 |
|---|---:|---:|---:|
| C8 整批 API tokens/s | 248.752 | 264.296 | +6.25% |
| C4 整场 API tokens/s | 106.426 | 106.153 | -0.26% |
| 100k cold read_ms 中位数 | 16012.000 | 16294.000 | +1.76% |
| 100k cold ttft_s 中位数 | 16.202 | 16.495 | +1.81% |
| 100k warm read_ms 中位数 | 2186.000 | 2056.000 | -5.95% |
| 100k warm ttft_s 中位数 | 2.281 | 2.152 | -5.65% |

冷 prefill read 中位数本轮变化 +1.76%；仅三次观测，不能据此保证 prefill 无回退，也不能断言微小变化来自 head 开关。
吞吐包含 admission/prefill、输出和尾部 drain，不是纯 decode kernel 速度。延迟负变化更快；吞吐正变化更快。

| C4 阶段 | off/on bytes/s | off/on p99 gap (s) | off/on max gap (s) | off/on Jain |
|---|---:|---:|---:|---:|
| baseline | 1060.0/1055.1 | 0.0246/0.0250 | 0.0249/0.0263 | 0.9970/0.9977 |
| mixed | 99.6/100.5 | 6.1405/6.1757 | 10.4329/10.4413 | 0.9825/0.9822 |
| recovery | 956.4/944.6 | 0.0249/0.0251 | 0.0265/0.0268 | 0.9981/0.9982 |

SSE 出流间隔包含 reasoning/content/tool；没有逐 token 时间，不能称 TBT。阶段 bytes/s 会受文本位置影响，完整 usage/hash 才证明整场工作量一致。

| 实际共享 head steps | off | on |
|---|---:|---:|
| budget | 0 | 2097 |
| prefill | 0 | 0 |
| c8 | 0 | 4672 |
| performance | 0 | 5219 |

新版 head=0 与旧 v4/v5 MTP-on 的兼容核对：

| 旧数据对应项 | 数量 | 输出/usage 或取消共同前缀通过 |
|---|---:|---|
| budget | 10 | True |
| prefill | 6 | True |
| c8 | 8 | True |
| c4 | 6 | True |

这些跨轮比较仅证明固定 synthetic 请求的输出与计数兼容；suite 顺序不同，不比较历史性能。

适用范围：

- One sequential pair; no confidence interval or stable throughput claim.
- C8 uses short prompts; C4 resident contexts and injector lengths are reported from actual usage. Neither establishes four-way 100k performance.
- Whole API TPS includes admission/prefill and final drain; chunk gaps are not per-token TBT.
- Per-phase byte throughput/fairness is descriptive because each phase may contain different content positions.
- Prefill medians have only three cold/three warm requests. Warm read time includes restore.
- Cancelled streams compare their shared prefixes; full cancelled text hashes need not match.

复现：`python3 analyze.py --input ../head-pilot --out .`。所有输入 SHA256、逐请求比较、阶段占用/停顿、缓存明细与实际 steps 保存在 `summary.json`。
