# 并发 read guard 离线审计

仅比较三个 guard 参数：off=0/0/0，on=16384/1/400；两臂共享 head、batch/piped MTP、self_commit=0 和预算 cap=true。

完整性检查：全部通过。尚无自动部署结论。

| 单独100k | off | on | 变化% |
|---|---:|---:|---:|
| cold read_ms 中位数 | 16423.000 | 16383.000 | -0.244 |
| cold ttft_s 中位数 | 16.608 | 16.570 | -0.231 |
| warm read_ms 中位数 | 2232.000 | 2133.000 | -4.435 |
| warm ttft_s 中位数 | 2.328 | 2.229 | -4.265 |

| C4阶段 | pooled p99 gap off/on s | pooled max gap off/on s | 最坏lane截断停顿 off/on s | >1s/>5s lane停顿次数 off→on |
|---|---:|---:|---:|---:|
| baseline | 0.025/0.025 | 0.026/0.025 | 0.026/0.025 | 0/0 → 0/0 |
| mixed | 5.887/6.219 | 10.246/6.470 | 10.246/6.470 | 12/8 → 24/16 |
| recovery | 0.025/0.025 | 0.038/0.038 | 0.026/0.027 | 0/0 → 0/0 |

| 新prefill请求 | TTFT off/on s | engine read off/on ms |
|---|---:|---:|
| injected-0-10000 | 5.852/5.850 | 5784.000/5784.000 |
| injected-1-60000 | 10.109/22.752 | 9972.000/22608.000 |

C4整场 API tokens/s：104.817 → 91.165（-13.024%）。包含 admission/prefill 与最终 drain。

精确配对：solo prefill 6/6，C4 6/6。

C4按实际prompt_tokens精确配对；名义10k/60k的tolerance资格另行保留，不能把相同实际输入说成名义长度合格。

完整 JSON 保留逐请求usage/reuse、两种gap口径、每lane截断边界及>1/>5秒计数、采样stall比例、实际piece/decode行数和全部输入SHA。

限制：

- One sequential pair and three cold/warm samples; no confidence interval or automatic performance pass.
- SSE chunk gaps are not per-token TBT. Report phase-clipped silence separately from fully observed gaps.
- Pooled gap quantiles weight faster streams more; worst-lane and clipped stalls are also included.
- Occupancy stall fractions count sampled client observations, not GPU occupancy or a continuous-time fraction.
- Engine read_ms includes interleaved decode and related waiting; it is not pure prefill kernel time.
- 400ms caps requested decode budget; one pump and drain can exceed it, so it is not a gap guarantee.
- Whole API TPS includes admission, prefills, reasoning/output and final drain.
- C4 actual prompt counts and nominal-tolerance qualifications are retained. Equal actual work does not imply nominal10k/60k qualification.
- Stored build receipts bind known source/binary hashes; offline analysis cannot independently reconstruct a remote build.
