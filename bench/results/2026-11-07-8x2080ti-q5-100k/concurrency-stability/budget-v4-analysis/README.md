# v4 budget cap 两臂独立分析

本报告只读取 `budget-v4-c4-sampled/01-off` 和 `02-on`，不合并 v3 或后续 C8。它是一次顺序 off/on 的短 C4 sampled 对照，不能作为稳定尾延迟或长期收益的统计证明。

两臂都使用 `reasoning_budget_engine_cap=true`，因此比较的是修复边界后的 batch/piped MTP 路径；本实验不隔离 cap 本身的性能贡献。全部结果均来自保存的 artifacts；本分析没有服务请求或 GPU 操作。

## 配置与来源核验

- Server SHA256：`40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9`。
- Engine SHA256：`113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f`。
- 动态库 SHA256：`fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661`。
- 配置去除声明变量、log/key 后的 hash 一致，全部 suite 返回 0；八卡专家全驻留、每卡 own prefill buffer 3.80GiB。
- 实际 route 差异完整为：`BATCH_MTP 0→1`、`PIPED_MTP 0→1`，以及该路径要求的 `BATCH_SELF_COMMIT 1→0`。主头 2,4 / MTP 头 3,5 和其他固定设置相同；这不是新共享 batch-head 候选的测试。

## 完整 C4 workload

两臂均成功完成 6 条请求、8320 个 API completion tokens，无缺失 usage。整场 82.044→78.936s，合计 101.409→105.402 tokens/s，吞吐 **+3.94%**。这包括 initial admission、mixed prefill、恢复和排空，不是纯 decode 速度。

6 对 request hashes、usage 和 reasoning/content/tool 各 channel hashes 均完全相同；4 条 resident 各 2048 输出、2 条 injected 各 64 输出。全部实际输入缓存为 0。resident 实际输入 8793 tokens，两个 injected 实际为 8793 / 52080，不能称为精确 10k/60k 或 100k 并发。

| 请求 | 输入/输出 tokens | 完整输出 hash 前12位（两臂相同） | 首任意输出 off/on 秒 | 完成 off/on 秒 |
| --- | ---: | --- | ---: | ---: |
| injected-0-10000 | 8793/64 | `a2fa39f910d7` | 38.025/38.390 | 39.338/39.892 |
| injected-1-60000 | 52080/64 | `19ba155751dc` | 50.356/50.701 | 52.780/53.505 |
| resident-0-0 | 8793/2048 | `56a499764082` | 25.190/6.552 | 81.700/78.401 |
| resident-1-0 | 8793/2048 | `ebc3a68f2411` | 19.355/12.779 | 81.708/78.813 |
| resident-2-0 | 8793/2048 | `c8d9639dd1c9` | 7.504/25.117 | 81.599/78.219 |
| resident-3-0 | 8793/2048 | `3a57a227a38a` | 13.499/18.958 | 81.690/78.311 |

时间均相对各自 scenario 起点；admission 顺序不同，不能按 lane 编号把首输出差直接归因于 MTP。完整 hash 和每路 usage 位于 `summary.json`。

| 整场时间分段 | off 秒 | on 秒 | on−off 秒 |
| --- | ---: | ---: | ---: |
| startup_to_baseline_s | 27.259 | 27.559 | +0.299 |
| baseline_s | 5.000 | 5.000 | -0.000 |
| mixed_s | 20.525 | 20.951 | +0.426 |
| recovery_s | 10.000 | 10.000 | -0.000 |
| drain_s | 19.259 | 15.425 | -3.834 |

主要节省出现在固定 recovery 窗结束后的 drain：19.259→15.425s。初始 admission +0.299s、mixed +0.426s；最后仅一个 resident 尚未完成的区间是 0.008→0.412s，不能解释整体收益。

两臂性能日志各有四条 10347-token thinking continuation，均为 **10328 reused +19 read**；对应 read_ms 顺序为 off `[1076,400,469,475]`、on `[1068,487,442,434]`。按长度匹配而非 request ID，不能将日志顺序冒充 lane 顺序。19 tokens 包含 native cap 尚未喂回的最后一个 token，与18-token wrap 的边界相符。

## 混合停顿与公平性

| 阶段 | resident bytes/s off/on | chunk-gap p99 off/on 秒 | 最大 gap off/on 秒 | Jain bytes公平性 off/on |
| --- | ---: | ---: | ---: | ---: |
| baseline | 957.16/1063.36 | 0.01839/0.02447 | 0.01980/0.02537 | 0.996824/0.997717 |
| mixed | 111.67/100.85 | 5.77806/6.12036 | 10.11075/10.52714 | 0.986954/0.987289 |
| recovery | 795.08/961.38 | 0.01848/0.02471 | 0.02083/0.03476 | 0.997111/0.997963 |

两臂 baseline occupancy 合格且 baseline/mixed/recovery 采样中均保持四个 resident streams。**MTP 没有解决新 prefill 导致的长停顿**：mixed 最长 gap 10.111→10.527s；mixed bytes/s 反而降低约9.7%。mixed 中至少一条 resident 超过1秒无新输出的采样比例为69.5%→72.3%，超过5秒为28.0%→31.3%；最多四条同时停顿。fairness数值接近，但“大家一起停”也会有高公平性。

实际8793/52080-token injected 请求的首任意输出TTFT分别为5.761/10.064→5.826/10.111s。recovery中没有观察到超过1秒的resident停顿；按bytes基线90%、连续两bin定义，on从第2秒开始满足且第6秒确认，off在10秒窗内未满足。此指标受输出阶段和文本密度影响，不能称为GPU恢复时间。

这里记录的是任意非空 reasoning/content/tool 的 SSE chunk 间隔，**不是逐token TBT**。MTP可能同一轮发出多个chunk；p99的sample定义也随出流组织变化。各阶段bytes比较仅为描述性指标，整场8320-token和完整输出相同才支持本次总吞吐比较。

## 100k prefill 对照

6 对 fixed greedy 请求的输入hash、完整输出hash、usage均相同，全部valid。每条101139 prompt tokens、256 completion tokens；cold复用0，warm复用101132并只读7。以下每类只有3次，不据此声称稳定尾分布。

| 请求 | read_ms off/on | TTFT off/on 秒 |
| --- | ---: | ---: |
| cold-1 | 14747/14684 | 14.931/14.862 |
| warm-1 | 2225/2200 | 2.323/2.306 |
| cold-2 | 16551/16662 | 16.750/16.860 |
| warm-2 | 2239/2052 | 2.344/2.143 |
| cold-3 | 16491/16433 | 16.685/16.633 |
| warm-3 | 2060/2181 | 2.156/2.287 |

cold read_ms中位数16491→16433（−0.35%），TTFT16.685→16.633s（−0.31%）；warm read_ms2225→2181（−1.98%），TTFT2.323→2.287s（−1.57%）。本组中位数未见prefill回退；单次cold-2、warm-3仍有上浮，不能把3次中位数外推为每次都不慢。warm read_ms含状态恢复，不代表7-token计算吞吐。

## 预算专项与取消

9 对完整专项请求的 request、reasoning、正文hash及usage相同，所有专项checks通过。两臂各4条budget wrap续写都只读19 tokens；取消后的恢复请求与控制相同。

主动取消流不能声称全文hash相同：off/on已接收正文为131/170 bytes，短者是长者的共同前缀，reasoning hash相同；两臂均在生成中关闭且无最终usage。它验证取消与后续恢复，不是等长输出parity。

## 复现与限制

`analyze.py` 只读本地原始产物，校验每个关键条件并生成带输入文件SHA256的 `summary.json`；不会请求HTTP、SSH、GPU或修改服务：

```sh
python3 /Users/fu/project/HAI/Strata/bench/results/2026-11-07-8x2080ti-q5-100k/concurrency-stability/budget-v4-analysis/analyze.py
```

本次结果支持继续评估“cap修复 + batch/piped MTP”的并发吞吐，尚不支持宣称解决mixed admission停顿。单对顺序测试仍受时钟、先后顺序和admission调度影响；未把后续C8或新共享batch-head候选结果混入，也未据此宣布部署。
