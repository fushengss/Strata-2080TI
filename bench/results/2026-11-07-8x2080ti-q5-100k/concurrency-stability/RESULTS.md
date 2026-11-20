# 8×2080 Ti：并发 MTP、共享 head 与 100k 验证

2026-11-07，测试机 `10.100.3.3`，8×RTX 2080 Ti（每卡 22528 MiB），Q5 专家全部驻留显存，无 GPU P2P。目标已从单请求峰值转为并发吞吐及新 prefill 到来时的输出稳定性。**batch/piped MTP、共享head和预算修复已部署；服务已恢复、8槽在线，部署后检查通过。guard不部署。**

已测得现有 batch/piped MTP 在短 C4/C8 的整场吞吐收益，以及共享 piped MTP head 在短 C8 的两轮收益；两者都没有证明解决混合 prefill 的长停顿。共享 head 在 C4 没有收益，四路约 100k 的单次整场收益仅 0.64%。

所有下表吞吐均为完整 API completion tokens ÷ 整场时间，包含请求入场、prefill、生成与最终排空，**不是纯 decode tokens/s**。各组是独立对照，不能把 +9.83% 与约 +6.3% 相乘后称为测得的总收益。

| 独立对照 | 固定条件及实际工作量 | off → on 合计 tokens/s | 变化 | 证据 |
|---|---|---:|---:|---|
| v4：batch/piped MTP | 两臂 cap=true；C4 sampled/thinking，4×8793 输入，各输出2048；另注入8793/52080输入，各输出64，共8320输出 | 101.409 → 105.402 | +3.94% | [v4完整审计](budget-v4-analysis/README.md) |
| v5：batch/piped MTP | 两臂 cap=true；C8短prompt，输入66/68，4 sampled + 4 greedy，每路1024输出，共8192 | 244.572 → 268.619 | +9.83% | [C8精确比较](c8-v5-compare/c8-comparison.json) |
| 共享 head 首轮 AB | batch/piped MTP、cap均开启，仅 head=0→1；同一C8短prompt | 248.752 → 264.296 | +6.25% | [首轮审计](head-pilot-analysis/README.md) |
| 共享 head 反轮 BA | 同上，先on后off | 256.544 → 272.790 | +6.33% | [反轮复核](head-reverse-analysis/README.md) |
| 共享 head：C4混合 | 与v4相同实际尺寸、8320输出，仅head变化 | 106.426 → 106.153 | −0.26% | [首轮审计](head-pilot-analysis/README.md) |
| 共享 head：四路约100k | 输入101147/101148，cold cache=0；2 sampled + 2 greedy，每路512输出，共2048 | 26.2246 → 26.3936 | +0.644% | [长上下文审计](head-long-analysis/README.md) |

C4的名义10k/60k fixture实际为8793/52080，原始 `prompt_within_nominal_tolerance=false` 保留；只能比较相同实际输入，不能称为名义尺寸验证通过。上表两组MTP开关测试均固定cap=true，**没有单独测量cap修复的性能贡献**。head正反两轮的suite顺序及热状态不完全相同，因此不是严格连续ABBA，也没有长期收益或尾延迟置信区间。

以下三组单独100k prefill的检查使用相同greedy请求，每条101139输入、256输出；cold复用0，warm精确复用101132并读取7。完整输入/输出hash、usage和cache/reuse均配对一致。

对应本轮head-on配置，单请求cold prefill的两轮中位数为 **16.294 / 16.442 s**（engine read约 **6207 / 6151 tokens/s**），随后256-token输出的engine decode中位数为 **122.5 / 122.4 tokens/s**（2.089 / 2.091 s）。这是该固定合成提示、greedy、短输出、无其他并发下的测量；六次decode观测范围104.7–123.9 tokens/s，不是所有业务的性能保证，也不能与先前768-token code/prose实验直接比较。[首轮原始结果](head-pilot/02-on/prefill/results.json)、[反轮原始结果](head-reverse/01-on/prefill/results.json)。

| 对照 | cold read中位数 off→on | 变化 | warm read中位数 off→on | 变化 |
|---|---:|---:|---:|---:|
| v4，batch/piped MTP | 16.491 → 16.433 s | −0.35% | 2.225 → 2.181 s | −1.98% |
| 共享head首轮 | 16.012 → 16.294 s | +1.76% | 2.186 → 2.056 s | −5.95% |
| 共享head反轮 | 16.498 → 16.442 s | −0.34% | 2.206 → 2.179 s | −1.22% |

每轮仅三次cold、三次warm；共享head的cold变化没有跨轮同向复现，目前不足以判定稳定回退，也不能承诺prefill零损失。warm时间包含状态恢复，不能拿7 tokens除以它解释为计算吞吐。八槽、八卡全专家驻留、每卡3.80 GiB自有prefill缓冲均保持并经启动日志核验。

**为何原并发路径没有发挥MTP，以及还未解决的问题。** 原服务器的 `PARALLEL_STACK_ENV` 自动补入 `STRATA_BATCH_SELF_COMMIT=1`，与piped MTP所需的延迟提交路径冲突；只开启两个MTP旗标不够。本次有效对照为 `BATCH_MTP/PIPED_MTP/SELF_COMMIT` 从 `0/0/1` 改为 `1/1/0`，并通过实际进程环境及proposal日志确认生效。`/metrics` 的draft计数不覆盖全部batch路径，不能以它的0单独判断没有MTP。

先前cap关闭的v3试验暴露thinking预算边界：部分续写重新读取1554而非18 tokens，且固定输出发生差异，不能用那组结果支持发布。修复后的v4两臂均cap=true，续写为10328 reused +19 read，预算专项、取消后恢复和完整输出比较通过；取消流只比较共同前缀，不声称取消全文hash相同。

v4 mixed阶段最大SSE gap仍为10.111→10.527 s；共享head C4为10.433→10.441 s。它们提高部分场景吞吐，没有消除新prefill让resident一起停顿的问题。SSE chunk可携带多个token，其间隔不是逐token TBT；阶段bytes/s也不能冒充token速度。四路100k的整场耗时78.095→77.595 s，共同出流窗口字节率反而约−1.48%；两臂入场顺序不同，不能按lane编号归因TTFT变化。

**正确性与版本绑定。** 共享head低层测试在同机source GPU7、targets GPU3/5、无P2P下，覆盖真实Q6_K draft词表106299行、N=2560、53120+53179分片。8槽共享只读权重、独立workspace测试通过1416 cases、150519384个logits逐位比较，包含8条source stream、owner reset、slot rebind与部分链排队；另复跑144个单helper cases，检查原argmax、两种概率路径、dvocab映射及8步链复用。[构建与测试记录](head-build/build-result.json)、[原始parity日志](head-build/parity.log)。

端到端各完整请求同时核对输入hash、完整及分通道输出hash、finish和三项usage；head=0还与旧v4/v5固定请求检查兼容，跨轮只用于正确性，不比较历史性能。低层parity不能代替端到端状态、取消及采样检查。

| 组件 | SHA256 |
|---|---|
| v4/v5引擎 | `113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f` |
| 共享head候选引擎 | `c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6` |
| 修复预算边界的服务器 | `40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9` |
| 相邻 `libstrata_kq.so` | `fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661` |

主head固定GPU2/4，solo MTP head固定GPU3/5；新 `STRATA_PIPED_MTP_HEAD` 才是本次共享并发draft head开关。源码默认关闭；所有实验记录实际exe、映射库、配置差异、冻结harness和临时凭证扫描，避免把仅写入配置误当成实际执行。

1Cat参考采用已核实的[参考笔记及固定commit来源](reference-notes.md)。可借鉴的是仅在decode竞争时限制prefill、同时检查resident停顿与新请求TTFT，以及完整推测轮次和状态边界验证。其TP4/PP1、SM70、NVFP4/AWQ或NVLink实现不能直接套用到本机SM75、Q5、八段层流水线、无P2P；参考作者的收益不计入本机测量。

**guard实测：不部署16384/1/400。** 独立测试只改变 `STRATA_PIPED_READ_PIECE / STRATA_PIPED_READ_DECODE_ONLY / STRATA_PIPED_DECODE_MAX_MS`：`0/0/0 → 16384/1/400`，固定共享head、batch/piped MTP和cap开启。设计为有active decode才切片，单独prefill保持原路径；400ms是请求的decode预算，不是最大停顿保证。本轮C4实际resident及小injector输入8776、长injector输入52063，不能与上方不同fixture合并。

| guard独立对照 | off → on | 变化或解释 |
|---|---:|---|
| 整场8320-token吞吐 | 104.817 → 91.165 tokens/s | −13.024% |
| 52063-token新请求TTFT | 10.1087 → 22.7523 s | +125.08% |
| 同请求engine read | 9.972 → 22.608 s | +126.71%；包含交错decode及等待 |
| 8776-token新请求TTFT | 5.8516 → 5.8503 s | 基本不变；read同为5.784 s |
| mixed最大SSE gap | 10.246 → 6.470 s | 最长停顿改善 |
| mixed pooled p99 gap | 5.887 → 6.219 s | 尾部指标未同步改善 |
| 单独100k cold read中位数 | 16.423 → 16.383 s | −0.24% |
| 单独100k warm read中位数 | 2.232 → 2.133 s | −4.44% |

mixed窗口由20.674延长至33.238 s；四lane合计>1s/>5s截断停顿次数为12/8→24/16，不能忽略窗口变长直接比较频率。采样时刻“至少一条resident超过5s无输出”的比例确实从31.3%降至15.8%。但新长请求TTFT翻倍、整场吞吐下降，无法满足不牺牲prefill的目标，因此拒绝本组参数。这里不否定其他调度设计，只记录本候选的代价。

guard的56项审计全部通过：六条solo和六条C4请求的输入、完整/分通道输出、usage/cache精确一致；solo输入101140、cold复用0、warm只读7。on实际出现3条read-piece日志、324行交错decode，off及solo均未切片。审计脚本SHA为 `4991ade9bb7e084ff30629f80bddb283e7360b12ef63763ecfc9d73907f34076`；[完整guard报告](guard-analysis/README.md)保留原始日志匹配、每lane完整/截断停顿、实际长度资格及来源SHA。

[guard源码审查](guard-analysis/source-notes.md)指出：本机八段流水线每个piece调用完整 `Prefill::run`，随后排空整条prefill流水线，再执行并排空decode，下一piece重新填充流水线。这是值得profile的额外开销来源，**属于源码推断，尚未测出各项耗时或关键路径占比**。没有证据表明每段重新加载模型或重建整张CUDA图；已有模型、缓冲与图会复用。后续应先测各stage等待与排空，再设计保留在途prefill的调度，不能直接删同步或把源码推断当作已验证的优化收益。

**最终部署与验收。** 主控已完成安装及恢复，2026-11-07台北时间21:36:56（13:36:56 UTC）服务加载成功、8 requested/8 serving；21:37:55核验实际进程与资源，21:39:43再次观察在线路由，部署后检查通过。选择 `STRATA_BATCH_MTP=1`、`STRATA_PIPED_MTP=1`、`STRATA_BATCH_SELF_COMMIT=0`、`STRATA_PIPED_MTP_HEAD=1`，并启用 `reasoning_budget_engine_cap=true`。guard不部署，API凭证、原prefill环境参数及layer split保持原样。[恢复记录](maintenance-restored.json)、[实际进程及资源核验](deployment-verified.json)。

- 安装引擎：`/srv/Strata-0157/engine/releases/concurrent-mtp-20261107/strata`，SHA为上表 `c9305b08…`；相邻 `libstrata_kq.so` 为 `fb67bffc…`，服务器为 `40c5e9df…`。
- 安装后配置SHA256：`f5376cc83215fc54b46d20094373f20a41b93daa8bf07262d2ac67dc13297220`。
- [正式进程read策略核验](deployment-read-policy.json)：PID34054的 `STRATA_PIPED_READ_PIECE`、`STRATA_PIPED_READ_DECODE_ONLY`、`STRATA_PIPED_DECODE_MAX_MS`、`STRATA_PIPED_CONCURRENT` 全部未设置，确认guard及concurrent-read路径关闭。
- 原配置与服务器的私有回滚备份：`/srv/bench100k-20261107-concurrency-mtp/rollback-20261107-133356`，目录0700、文件0600。[部署及回滚receipt](deployment-receipt.json)记录原始/安装后SHA与路径；它的 `service_started=false` 描述安装步骤，后续恢复由上述独立记录确认。文档不保存API凭证。

部署后[四路API smoke](deployment-smoke.json)各完成128 tokens、`finish=length`，四条均无请求错误。这不是性能benchmark。初次smoke命令返回1，仅因附加要求“200ms内出现batch结束的head汇总”未满足；当时还有其他测试请求在途，原smoke产物保留不改，记录的head steps仍为0。

随后[独立在线路由观察](deployment-route-observed.json)确认从本次启动起的新增日志累计4419个实际head offload steps、3122/4429个MTP proposals被接受，`all_postflight_checks_passed=true`。这份计数包含恢复后的其他测试流量，**不能归给四条smoke请求**。核验同时确认八卡专家驻留、每卡3.80 GiB自有prefill缓冲及GPU3/5上的一套共享head分片权重及8套独立slot workspace。
