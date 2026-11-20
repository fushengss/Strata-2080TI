# 并发稳定性测试与合成客户端

2026-11-07 已在 10.100.3.3 的独立维护测试服务完成 GPU 对照，并部署已通过验证的并发 MTP、思考预算修复和共享输出头。正式服务已恢复 8 槽在线。见 [完整结果与部署记录](RESULTS.md)，以及 [预算修复与并发 MTP](budget-v4-analysis/README.md)、[输出头正反轮次](head-reverse-analysis/README.md)、[四路冷 100k](head-long-analysis/README.md)、[未部署的分段调度](guard-analysis/REVIEW.md)。这些报告中的测量结果不能由下文的计划尺寸推断。客户端默认 short profile 和 dry-run；实际发送请求必须同时给出 `--execute --exclusive-benchmark --port --config --out`。地址固定为 `127.0.0.1`，输出目录必须新建。`client.py` 本身不启停服务、不改配置、不清缓存、不读服务日志，也不执行模型返回的工具调用；独立 runner 管理它启动的测试进程。

## 本轮短模式

先测 C4，再单独测 C2、C8。short 默认 resident 上下文约 10k、每次最多 2048 输出 tokens；先建立全部 resident 的持续出流，稳定 2 秒后测 5 秒 baseline，再注入约 10k、60k 各一个新请求，计划相隔 8 秒，最后测 10 秒恢复。请求自然结束后补位，补位空档计入实际并发和公平性，不假设 resident 始终在 GPU 上运行。

C4/C2 是有空槽时的混合 admission；**C8 + 一个 injected 请求是八槽容量下的排队压力**，不能当成 C4 preemption 结果。全部 resident 连续出流才能进入 baseline，baseline 至少 95% 采样点达到目标 active stream 数才合格。另记录 1 秒/5 秒没有任何新输出的 stream 计数；reasoning 或 tool delta 会更新进度，不能误判成可见正文停顿。

以下为在指定测试服务已加载且空闲时运行的命令示例。实际执行命令及冻结源码保存在各轮 manifest/harness 中。greedy 和 sampled 使用独立目录；A/B 各臂使用相同 `--fixture-id`，并由外部 runner 保证每臂进程和配置一致。

```sh
python3 client.py --execute --exclusive-benchmark \
  --port 8092 --config /dev/shm/private-benchmark.json \
  --out /srv/concurrency-off-sampled --profile short \
  --sampling-profile sampled --levels 4 --fixture-id concurrency-env-ab-20261107
```

`--sampling-profile greedy` 使用 temperature 0；`sampled` 使用 temperature 0.8、top_p 0.95、top_k 20，固定 seed。两者全程分组，不合并 tail 或吞吐统计。默认启用 thinking，分别记录首个任意输出、首个 reasoning、首个可见正文、首个 tool delta。

short 场景上限 240 秒，首输出超时 45 秒，无任意新输出超时 30 秒，最后最多自然排空 30 秒。超时只关闭本客户端的连接，并保存删去文本的证据。C8 可能测得排队超时；这应报告为 TTFT 下界，不能当成已完成请求。short 是短资格测试，**其 resident 约 10k，不能称为 100k 并发结果**。`--resident-context-tokens 100000` 只改上下文，保留所选 profile 超时；完整计划 `--profile full --levels 4` 提供较长观察和六次新 prefill，但不是本轮默认执行项。

## 输出和口径

每档目录含逐请求 `.json`、无正文的 `.chunks.jsonl`、`occupancy.json`、`summary.json`；外层 `manifest.json` 保存解析后的计划、fixture ID 和脚本 hash。不保存 config 内容/API key、请求正文、生成正文、reasoning 文本、tool 参数或原始 HTTP 错误体。

- `ttft_s` 是首次非空 reasoning/content/tool delta 延迟；`first_visible_content_s`、`first_reasoning_s`、`first_tool_delta_s` 单列。逐请求及分组给出 p50/p95/p99/max；未开始出流的超时另给 censored lower bound，不混入完成请求的分位数。
- `output_inter_chunk_gaps` 测量任意输出 SSE chunk 间隔；每种 channel 也单独统计。**没有逐 token 时间，不能将其称为 token TBT。** chunk 可能含多个 token，或只含一个 token 的部分文本。
- 每个 chunk 保存时间边界、channel、字节数；请求结束后按最终 usage 与各 chunk 输出字节比例分摊 token 估计。缺 usage 时使用明确标记的 byte/4 启发式。两者都不是 token 边界；未完成请求比例会影响吞吐估计的可信度。JSONL 是实时原始计数，最终 token 估计位于请求 JSON。
- baseline/mixed/recovery 主吞吐为包括 reasoning/tool 文本在内的输出 bytes/s，可见正文 bytes/s 单列。每条 resident lane 用同一墙钟区间计算进度率和 Jain 指数；这是出流公平性代理，不是等量工作或精确 token 公平性。另给最终 API usage 覆盖的整场吞吐，缺 usage 时为下界。
- 新请求的 start→first-output 只是排队/admission/prefill/首 token 的合并区间；不能称为纯 GPU prefill。实际 prompt/cached tokens 由 API usage 核验，`verified_zero_cache` 为 false/unknown 时，不冒充已验证冷请求。
- 恢复定义为最终注入结束后，连续两个完整时间 bin 的 aggregate output-byte rate 达到 baseline 的 90%；short bin 为 2 秒。报告首次达到区间和确认时刻；这不是 GPU 硬件恢复时间。

`summary.error` 非空、真实请求失败、baseline occupancy 不合格均导致退出 1 并跳过后续档位；预期的 `bounded_drain_cancel` 单列，不作为真实请求错误，但它会缺失最终 usage。运行后若服务仍有请求在途，客户端拒绝继续下一档。baseline 合格也不代表 candidate 性能合格：应比较 resident gap 尾部/max stall、新请求 TTFT、公平性和恢复，吞吐仅为次要指标。short 每尺寸只有一次新请求，p95/p99 不具备稳定尾部估计能力。

## 离线验证

```sh
python3 client.py --self-test
python3 client.py --profile short --sampling-profile sampled --levels 4
```

第一条使用内存 fake HTTP transport，验证 SSE framing、reasoning/content/tool 计时、greedy/sampled 请求参数、usage 分摊和脱敏产物；没有网络。第二条只打印计划，不读私有 config、不联网。另已验证缺少 exclusive 标志时在读取 config 或创建连接之前拒绝执行。离线自测不代表 GPU 正确性或性能通过；实测结论以各轮证据和报告为准。
