# Guard 本轮审阅

建议不部署 `16384 / decode-only=1 / max-ms=400` 这个参数组合。最长暂停缩短，但本轮较大注入请求的延迟和整场吞吐代价过高，p99 也没有改善。

56 项身份、来源、资源和正确性检查全部通过。solo 100k 的 6 对 cold/warm 请求、C4 的 6 对完整请求，body hash、完整输出/分通道 hash、finish、prompt/completion/total 三计数及全部缓存 metadata 均一致；solo 的 engine reuse/read 计数也一致。相同 c930 可执行文件、40c5 server、源文件/测试脚本 SHA、实际进程 flags、8 槽、每卡原有 3.80 GiB prompt buffer 和全专家驻留均已核实。冻结分析源码未修改。

| 指标 | off | on | 变化 |
|---|---:|---:|---:|
| C4 整场完成 API token/s | 104.817 | 91.165 | -13.02% |
| mixed 最长 SSE 间隔 | 10.246 s | 6.470 s | -36.85% |
| mixed pooled p99 SSE 间隔 | 5.887 s | 6.219 s | +5.64% |
| 8776-token 注入 TTFT | 5.852 s | 5.850 s | -0.02% |
| 8776-token 注入 engine read | 5784 ms | 5784 ms | 0% |
| 52063-token 注入 TTFT | 10.109 s | 22.752 s | +125.08% |
| 52063-token 注入 engine read | 9972 ms | 22608 ms | +126.71% |

两臂整场都完成 8320 API tokens，计时 79.376 / 91.263 秒，包含初始 admission/prefill 和最终 drain。C4 resident 实际输入为 8776 tokens，两个注入请求为 8776 / 52063 tokens；两臂名义 10k/60k 的 ±10% 长度资格均为 false，报告保留该标记。配对按实际相同输入完成，不能称为合格的名义 10k/60k 测量。

mixed 阶段长度从 20.674 秒增至 33.238 秒。四路合计的阶段截断停顿次数，>1 秒为 12→24，>5 秒为 8→16；窗口长度不同，因此这些不是归一化事件率。采样点上“至少一路已停顿 >5 秒”的占比从 31.33% 降到 15.79%，>1 秒的占比从 73.49% 降到 70.68%。这说明最长连续阻塞被切开，同时较大 prefill 更久地占用了混合阶段。SSE 间隔不是逐 token TBT，采样占比也不是 GPU 利用率。

单独 101140-token cold prefill 的 read 中位数 16423→16383 ms（-0.24%），TTFT 16.608→16.570 秒（-0.23%），未观察到冷 prefill 回退。warm 每次精确复用 101133 tokens、读取 7 tokens，read 中位数 2232→2133 ms。各仅三次观测，不把细小变化认定为优化收益。

路线证据明确：off 无分段/lead-in；两臂 solo prefill 都没有分段。on 的 52063-token 读取记录了三段 16384 tokens，单段 6093 / 6176 / 6351 ms，段间实际 decode 108 / 102 / 114 rows，用时 420 / 421 / 423 ms；最后剩余 2911-token 段没有同样的边界日志。另有两次 lead-in，各 415 / 424 ms。共享 head 两臂实际 offload 5203 / 5166 steps。

400 ms 限制的是所请求的 decode 预算，pump/drain 可以超时；16k read 本身仍持续约 6 秒，因此不能把此参数当成输出间隔上限。单轮结果足以拒绝本参数组合的部署候选，不构成所有分段策略都无效的结论。

详细证据：[自动审计报告](README.md)、[全部数值与检查](summary.json)、[审阅记录](review.json)。本轮没有混入 head-reverse 数据。
