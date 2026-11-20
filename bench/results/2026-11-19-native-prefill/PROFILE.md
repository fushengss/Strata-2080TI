# Q5 专家算子与八卡 100k prefill 时间线

2026-11-19 在 `10.100.3.3` 测试。模型为 Orca Qwen3.8-Flash-Next-Uncensored Q5_K_M，8 张 RTX 2080 Ti，每张 22 GiB、250 W。沿用原来的 8 卡分层、专家全驻留、3072-token 分块、16 个并发槽及 MTP 设置。两组使用 `bd3b7ac80` 同一二进制，通过 `STRATA_PF_FUSED_SM75_KQ` 切换。[此前的冷 prefill A/B](README.md)

## 单卡专家计算

使用生产 MMQ 的按行数排序、16 专家分组 gather 和 scatter 量化，耗时为单层中位数。合成路由的均匀组和 32 个热点专家组均覆盖全部 512 个专家。

| 权重组合 | 路由 | MMQ | 原生 | 吞吐变化 |
| --- | --- | ---: | ---: | ---: |
| Q5_K / Q5_1 | 均匀 | 20.750 ms | 16.458 ms | +26.1% |
| Q5_K / Q8_0 | 均匀 | 21.756 ms | 17.860 ms | +21.8% |
| Q5_K / Q5_1 | 热点 | 20.767 ms | 16.731 ms | +24.1% |
| Q5_K / Q8_0 | 热点 | 21.625 ms | 18.285 ms | +18.3% |

两组路由的独立 FP32 采样对照和路由校验均通过。原始测试程序还测了预先排好权重的 MMQ 对照，两种组合分别是 **15.049、15.244 ms**。该对照省略权重 gather。原始程序用旧跨路径逐值阈值判断 Q8_0，虽两条路径相对 FP32 的 RMS 都是 1.118%，仍以错误码 1 结束。表中使用生产式测试，保留 FP32 比较和路由校验。[均匀路由](profile-raw/operator-resident-uniform.log) · [热点路由](profile-raw/operator-resident-skew.log) · [单层测试补丁](operator-bench.patch)

## 八卡冷 100k

| 指标 | MMQ | 原生 |
| --- | ---: | ---: |
| 无 trace 引擎 prefill | 11.826 s | 11.653 s |
| 无 trace API 首字 | 12.032 s | 11.862 s |
| 有 trace 引擎 prefill | 12.770 s | 12.631 s |
| 有 trace API 首字 | 13.046 s | 12.824 s |
| 关键路径同 stream 的 GPU kernel | 9.192 s | 9.418 s |
| 关键路径上 240 段 MoE 的墙钟时间 | 2.999 s | 3.156 s |
| 上述 MoE 区间内的同 stream GPU kernel | 2.927 s | 3.151 s |
| 上述 MoE 区间内在该卡上无可见 CUDA 工作的时间 | 72.05 ms | 4.15 ms |
| 关键路径中的跨卡激活传输 | 604.9 ms | 602.7 ms |
| 已知前驱完成后至下一段或下一块开始 | 11.77 ms | 11.55 ms |
| 段内在该卡上无可见 CUDA 工作的时间 | 1.992 s | 1.636 s |
| GPU7 中位分块时长 | 316.27 ms | 311.50 ms |
| GPU7 中位块间隔 | 0.045 ms | 0.053 ms |

七个跨卡边界均匹配 32 块 × 3072 行和一块 × 1689 行，共 99,993 行。两条观测关键路径都覆盖 GPU7 的 33 块和 GPU0–6 各一块。GPU7 的专家 down 全部为 Q8_0。提前完成的阶段有可见回压，在接收阶段仍处理上一块时，下载完成到下一卡上传之间的间隔不能全计为关键路径等待。

GPU7 的 MMQ 分组 gather 共 1021 次，摊到 33 块 × 6 层约每层每块 5.2 次。单层合成测试覆盖 512 个专家，MMQ 需要 32 组。这条 100k 提示的路由在尾段比合成用例稀疏。原生在每层每块发射四个 128 专家批次，未路由专家没有计算 tile。

240 段 MoE 从路由 kernel 完成到专家结果合并 kernel 启动，覆盖关键路径上 40 个阶段分块的 6 层，包含共享专家等共同工作。两组锚点全部匹配。原生的 MoE kernel 在这条路径上多花约 224 ms，MoE 区间墙钟多花约 156 ms。整个关键路径段内无可见工作区间缩短约 356 ms，抵消额外的 kernel 时间。跨卡传输变化约 2 ms，前驱完成后的单独间隔变化不足 1 ms。

这些时间来自每种路径一次 CUDA 追踪。分类按内核名称，MMQ 类包含尾卡草稿等少量同名计算。追踪有 Nsight 的通用事件完整性警告。八张卡的分层、跨卡 copy 序列和算子调用数符合代码预期。段内无可见 CUDA 工作的时间可以包含 CPU 路由处理、同步、提交和追踪开销，不能直接作为可回收时间。两组配对请求的输出摘要一致。[MMQ 时间线](profile-raw/mmq-trace.json) · [原生时间线](profile-raw/native-trace.json) · [MMQ 请求记录](profile-raw/mmq-request.json) · [原生请求记录](profile-raw/native-request.json)

## 测量方法

- 单层测试用 GPU0，3072 个 token、512 个专家、top-10。两个权重组合的 gate/up 为 Q5_K，down 分别是 Q5_1 和 Q8_0。使用合成输入并交替执行两种实现。完整计算路径和预先排好权重的 MMQ 对照分别计时。
- 八卡每组先跑一条不抓 trace 的冷 100k，再用另一条冷 100k 抓 Nsight Systems CUDA 时间线。相应请求在两组间按请求摘要配对，实际 prompt 都是 100,000 token，缓存命中为 0。Nsight 的 graph node tracing 会改变绝对时长，trace 用于分解路径，无 trace 的速度另列。
- 每个跨卡边界以同形状的 FP32 激活 D2H 和 H2D 为锚点。按每行 40,960 字节配对。第一个 kernel 之前和最后一个 kernel 之后的时间单列。按每段的同一 stream 和两个已知前驱（同块的上一段、同段的上一块）回溯一条观测关键路径。[运行脚本](profile_ab.py) · [时间线分析器](analyze_prefill_timeline.py)

完整 `.nsys-rep` 和 UTC SQLite 导出保存在测试机的 `/srv/bench-20261119-prefill-profile/`。

两组测试实例都只监听 `127.0.0.1:8080`。测试完成后恢复原线上监听、配置和二进制，并确认 16 槽可用。[执行记录](profile-raw/run.log) · [恢复记录](profile-raw/restore-receipt.json)
