# 分段 prefill 的只读源码审查

日期：2026-11-07。检查工作树 `/Users/fu/project/HAI/Strata-decode-head`。本笔记只读源码；没有新增 GPU 实验、Nsight capture 或运行时 profile。行号对应审查时源码。

源码快照 HEAD：`70915eee8a0a193819a944223fb8a61b23b158df`。文件 SHA256：

- `src/program/generate.cpp`：`8fe6b0f2b53c1a2a756697ae92f1044e8c38281a7a0ff67622d38cd215969ddf`
- `src/prefill/prefill.cpp`：`9aa0af49ac59783eee42c9dc5650772680050108d681a9dfe3e0089bd49d31a5`
- `src/core/mtp.cpp`：`81fd16a3464aacfce1faf68194ca981bcac033b44b5cef3274d309a2f9054808`

主控本轮报告：guard 输出全部 exact，但 52063-token 新请求 TTFT 从 10.1087 秒增加到 22.7523 秒，整场吞吐下降 13.024%，因此不部署该 guard。这里解释代码允许发生的开销，不把它当作已测得的耗时分解。

## 首要线索：每个 piece 排空整个八级流水线

1. [generate.cpp:17063](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17063) 的 `read_part` 对整个 segment 计划一次 chunk；17084 调用 `plan_chunk`，17172 在每次 `run` 前通过 `force_chunk` 复用计划。guard 不会按每个 16K piece 重新选择小 chunk。
2. [generate.cpp:17663](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17663) 每个 piece 调用 `run_conc(q,r)`。但 [17197](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17197) 在 `piped=true` 或存在 layer-split stages 时直接退回 `run(q,r)`；[17179](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17179) 再次调用同步完成的 `sp.run`。函数名 `run_conc` 不代表本机这个分支让 prefill/decode 同时执行。
3. [prefill.cpp:7907](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/prefill/prefill.cpp#L7907) 的 `Prefill::run` 调用 `run_impl` 后，无条件调用 `drain_pipeline`。[7814](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/prefill/prefill.cpp#L7814) 等待当前 successor future，再递归等待后续所有 stage。
4. 一次正常长 `run_impl` 的重叠来自 [prefill.cpp:6225](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/prefill/prefill.cpp#L6225)：只等待直接后继的上一 chunk，再异步交给下一级；[6307](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/prefill/prefill.cpp#L6307) 明确让中间 stage 返回时后面的 GPU 继续处理旧 chunk。分成 piece 后，顶层 `run` 的全链 drain 会在每段边界结束这种重叠。
5. 完整 prefill piece 返回后，[generate.cpp:17602](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17602) 才运行 decode budget；17612 又 `pipe_drain()` 等全部 decode groups 完成，然后下一段 prefill 从空的流水线重新开始。实际 400ms budget 可以因最后在途窗口超过 400ms；本轮主控观察约 420ms。

**代码推断，未 profile：**如果实际 chunk 为 8192、八个 stage 近似等耗时，52K 约七个 chunk。整读的简化流水线时间约为 `(7+7)u=14u`；四个 piece 含 `2,2,2,1` chunks，则约为 `(2+7)+(2+7)+(2+7)+(1+7)=35u`，还未加 decode 插入时间。一个 16K piece 本身约 `9u`，可以占整读 `14u` 的较大部分。这与主控看到每个 16K piece 约 6.1 秒、整读约 10 秒的量级相容。

这个模型忽略 stage 不平衡、尾 chunk 较小、attention 上下文长度、PCIe 重叠等差异；需以实际 chunk 日志确认前提。不能据此声称已经测出 2.5 倍开销或某项精确占比。

## 哪些操作重复，哪些本来就逐 chunk 执行

| 操作 | 代码位置 | 能从代码确认的范围 |
|---|---|---|
| 完整 stage 链 drain 与随后 decode drain | `prefill.cpp:7907–7910,7814–7826`; `generate.cpp:17602–17619` | 每个 piece 都执行，增加流水线填充/排空次数；优先排查。 |
| stream 收尾同步 | `prefill.cpp:6329–6340`，`3276–3281` | 每个 `run_impl` 结束等待 compute/copy stream；KV/PLE 额外 stream 的等待取决于相应功能是否启用。较后 stage 原本每个 chunk 就单独进入 `run_impl`，不能把所有等待都算成 piece 新增。 |
| PLE read-ahead 生命周期 | `prefill.cpp:3337–3361,3516–3521,3559–3569` | `ple_next` future 与双缓冲索引是本次 run 的局部状态；只预读本次 `n` 范围内的下一个 chunk。piece 最后一个 chunk 不预取下个 piece 的 PLE，下一段重新启动首 chunk gather。持久 host 预取可作为明确的复用方向，耗时未测。 |
| MMQ identity 表、token/step records | `prefill.cpp:3326,3466,3577` | identity 表每个 run 初始化；token、QSA step records 按 chunk 上传。chunk 形状保持相同时，后两者的总数据量并不会因切 piece 凭空增加。 |
| stage 之间 residual 交接 | `prefill.cpp:6200–6223,3422–3448` | 每 chunk D2H 到已有 pinned buffer，再由下一卡 H2D。`N=2560, HC=4`（207行），8192-token chunk 每边界每方向约320MiB。相同 chunks 的总字节量不增加，但分段会失去跨 chunk 的持续重叠。 |
| 后继线程与双 handoff buffer | `prefill.cpp:6230–6250` | `next_run_`、`hand_buf_`、已有 pinned handoff buffer 属于持久对象；默认路径每 chunk `std::async`。不是每 piece 重建整个 Prefill。已有 `STRATA_HANDOFF_PIPE`/worker 选项（483–503行），不能当作尚未实现的新方案，也不能把线程开销直接当作6秒归因。 |

## 没有发现“每 piece 重新加载模型或重建整张图”的证据

- 原有 `Prefill::Impl`、模型权重及约3.80GiB/card prompt buffers 跨 `run` 存在。当前全专家驻留且各卡自有 prompt buffers；[generate.cpp:16930](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L16930) 在无借槽参与者时直接跳过 `lend`，[16904](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L16904) 的 refill 也只处理实际借出的 slots。没有正常 piece 边界重载全部专家这一条路径。
- 正常 piece 边界没有将整个 prompt 的 KV/GDN 状态复制到 slot 再拷回。`copy_to_slot` 出现在真实 `BYIELD` 的 [generate.cpp:17277](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17277)，不是每段无条件执行；periodic checkpoint 在 [12639](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L12639) 按 `prompt_cache_every` 和已到位置判断，不能直接算作每 piece 新开销。
- 主 prefill 的 `run_impl` 直接发射计算，未发现每个 piece 捕获整张 prefill CUDA graph。
- 当前 `STRATA_SPLIT_MTP_BATCH=1` 在最后 stage 的每-chunk callback 走 [generate.cpp:12547](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L12547) → `Prefill::draft_kv`。[prefill.cpp:2239](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/prefill/prefill.cpp#L2239) 从已有 `m.region` 划分 scratch，2265 等 drafter idle，2289/2294 上传 bounds/token，2314 做 residual D2D，2344 等本 chunk K/V 完成；这条函数没有每 piece 新 `cudaMalloc` 或图捕获。这些计算原来每个 chunk 也执行。
- 如果 draft K/V 退回 drafter 自身路径，已有图也会复用：[mtp.cpp:1437](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/core/mtp.cpp#L1437) 和1448在 `prefill_exec_[T]`/`prefill_dev_exec_[T]` 已存在时立即返回。初次新 shape 的准备和分配应另行测量，不能假定每段重建。

## 后续代码方向

1. **先记录排空等待，再决定优化目标。**在 `read_part::run`/`Prefill::run` 区分第一 stage 计算、等待 successor、全链 drain、decode drain；保留实际 chunk/stage 数。已有 `STRATA_PREFILL_TIMING` 的 `prefill.cpp:6366–6371` 能区分部分 GPU、host setup、chunk wait、draft callback，但现有每-stage累计数不能直接相加当作请求关键路径。这里只指出可观测位置，未开启计时或运行新实验。
2. **持久化每次 prompt 的准备状态。**考虑 `begin_prompt / enqueue_chunk / finish_prompt` 生命周期，复用原 chunk plan、token/step staging 和跨 piece 的 PLE future/双缓冲；无 active decode 时保持现有整个 run。单做准备复用不能消除全链 fill/drain 成本。
3. **真正避免排空，要改变调度边界。**探索在各 stage 的 chunk/event 边界安排该卡的 decode 工作，保留其他 stage 已排队的 prefill，而不是每16K都先等所有GPU空闲。需要明确每个 stage 的 scratch、handoff、session、MTP callback 与 stream ownership；`next_run_` 还在读 buffer 时不得复用。不能简单删除 `drain_pipeline`，否则现有 decode pump 和 prefill callback 的状态/缓冲安全条件会失效。
4. **限定最初适用条件。**本机的全专家驻留、自有 prefill buffers、独立 slot session 是可缩小审查范围的条件；借槽、KV streaming、取消/给位、checkpoint 和 draft 回调仍需分别验证。已有 `STRATA_PIPED_CONCURRENT` 在 [generate.cpp:17313](https://github.com/fushengss/Strata-2080TI/blob/3ebdd9994c5fa908671c5b198ad61da18984612d/src/program/generate.cpp#L17313) 明确排除当前 `STRATA_SPLIT_MTP_BATCH` 等组合，不应直接开开关代替兼容性工作。

本轮 guard 被实际 TTFT/吞吐结果拒绝。后续方向的目标是保留原 prefill 吞吐并减少长期无出流间隔；上述方案尚未实现、未验证，不能声称满足该目标。此次审查没有进行 CPU 频率归因。
