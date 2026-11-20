# 共享 piped MTP head 正反轮次复核

仅离线分析已保存的 head-pilot 与 head-reverse；binary `c9305b08…`、server `40c5e9df…`。两臂 piped MTP 始终开启，只切换 `STRATA_PIPED_MTP_HEAD`。

身份、资源、完整输出、usage/cache 检查：全部通过。

| 每轮指标 | 首轮 AB off | 首轮 AB on | 变化 | 反轮 BA off | 反轮 BA on | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| cold read_ms 中位数 | 16012.000 | 16294.000 | +1.76% | 16498.000 | 16442.000 | -0.34% |
| cold ttft_s 中位数 | 16.202 | 16.495 | +1.81% | 16.693 | 16.640 | -0.32% |
| warm read_ms 中位数 | 2186.000 | 2056.000 | -5.95% | 2206.000 | 2179.000 | -1.22% |
| warm ttft_s 中位数 | 2.281 | 2.152 | -5.65% | 2.312 | 2.287 | -1.09% |
| C8 API tokens/s | 248.752 | 264.296 | +6.25% | 256.544 | 272.790 | +6.33% |

每轮各有三次 cold、三次 warm；延迟正变化更慢、吞吐正变化更快。C8 每臂完成 8192 API tokens，整批 TPS 包含 admission/prefill 与尾部 drain。

首轮顺序为 budget→prefill→C8→C4，反轮为 prefill→C8；两轮之间还有其他测试。因此以下合并值只是描述统计，不是热状态与 suite 顺序严格一致的连续 ABBA。

| 描述性合并指标 | off | on | 变化 |
|---|---:|---:|---:|
| cold read_ms，每臂6次的中位数 | 16255.000 | 16368.000 | +0.70% |
| cold ttft_s，每臂6次的中位数 | 16.447 | 16.567 | +0.73% |
| warm read_ms，每臂6次的中位数 | 2196.000 | 2172.000 | -1.09% |
| warm ttft_s，每臂6次的中位数 | 2.301 | 2.275 | -1.13% |
| C8 总tokens/总耗时 | 252.588 | 268.476 | +6.29% |

每个 fixture 的配对变化及其变化率中位数见 `summary.json`，不以合并值替代上方逐轮结果。

两轮各检查六条 prefill、八条 C8 的完整 request/output/reasoning/channel hashes、全部 usage/cache；另外比较相同 arm 在两轮之间的输出兼容。检查实际进程 exe/映射库、冻结源码、唯一 head flag 差异、全专家驻留、8槽与每卡3.80GiB prefill buffer；确认 prefill 的 shared head steps=0、C8 on 存在实际 steps。

C8 是短 prompt 场景，不能外推四路 100k；chunk arrivals 不能当作逐 token TBT。样本量小，顺序和热状态有差异，本报告不自动批准部署。

复现：`python3 analyze.py --pilot ../head-pilot --reverse ../head-reverse --out .`。所有读取文件的 SHA256 和每项检查均在 `summary.json`。
