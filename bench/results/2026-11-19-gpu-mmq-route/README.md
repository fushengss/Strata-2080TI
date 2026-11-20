# GPU 路由分组接入常驻 MMQ

在 `10.100.3.3` 测试 Orca Qwen3.8-Flash-Next-Uncensored Q5_K_M。8 张 RTX 2080 Ti，每张 22 GiB、250 W，16 个并发槽、8 个 batch group、3072-token prefill 分块、专家全驻留。沿用部署中的 `2a409c93f` 和并发 MTP 配置。

GPU 组按 `(专家行数, 专家 ID)` 稳定排序，保持每个专家内部的 token/rank 顺序，直接写 MMQ 的 `slot/src` 设备表。主机只回读 512 个计数，合计 2 KiB，并构造原有 MMQ launch 参数。CPU 和 GPU 组运行同一个二进制，通过 `STRATA_MMQ_GPU_ROUTE` 切换。CUDA 单测覆盖 26 组路由，并验证非法专家 ID 会被拒绝。

启用范围是使用常驻排序的原生布局、全驻留且无需专家流式搬运的合格分块。小分块、非全驻留、启用 peer 或使用原始路由回调的路径继续使用 CPU 分组。候选 `f2b4bcbbb` 的日志确认八张卡均使用了 GPU 分组。[CUDA 分组](https://github.com/fushengss/Strata-2080TI/blob/f2b4bcbbb27d80512249da5151843973b1ef3013/src/prefill/moe_mmq.cu) · [接入点](https://github.com/fushengss/Strata-2080TI/blob/f2b4bcbbb27d80512249da5151843973b1ef3013/src/prefill/prefill.cpp) · [CUDA 单测输出](raw/route-parity.log)

## 测量结果

### 冷 100k，三轮

| 指标 | CPU 分组 | GPU 分组 |
| --- | ---: | ---: |
| 引擎 prefill 中位数 | 12.309 s | 12.242 s |
| API 首字中位数 | 12.469 s | 12.408 s |

三对提示逐轮分别省 25.9、67.4、74.7 ms，引擎吞吐中位增幅 **0.55%**。请求和贪婪输出摘要、token 用量和结束原因逐对一致。[CPU 原始记录](raw/cpu-prefill.json) · [GPU 原始记录](raw/gpu-prefill.json)

### 并发 decode 中插入冷 100k，每组两轮

| 指标 | CPU 分组 | GPU 分组 |
| --- | ---: | ---: |
| 8 路，重叠期合计 tok/s | 91.46 | 91.51 |
| 8 路，重叠期 SSE p95 | 232 ms | 234 ms |
| 8 路，重叠期最长 SSE 间隔 | 708 ms | 699 ms |
| 8 路，长请求首字 | 13.093 s | 13.016 s |
| 16 槽，排队前 16 路合计 tok/s | 530.63 | 535.39 |
| 16 槽，实际读取时余下 15 路合计 tok/s | 138.63 | 135.94 |
| 16 槽，重叠期 SSE p95 | 279 ms | 279 ms |
| 16 槽，重叠期最长 SSE 间隔 | 437 ms | 583 ms |
| 16 槽，长请求已观测到的排队时长 | 18.455 s | 18.515 s |
| 16 槽，长请求首字（含排队） | 32.220 s | 32.127 s |

表中的吞吐、p95、首字和排队取两轮中位数，最长 SSE 间隔取两轮最坏值。八路重叠吞吐两个配对分别为 89.96 → 91.27、92.97 → 91.76 tok/s。16 槽的重叠吞吐分别为 137.58 → 133.74、139.68 → 138.14 tok/s。四组测试都证明实际重叠，饱和测试还核对长请求复用了预设短请求释放的槽。52 对混合请求的提示摘要、输出摘要、用量和结束原因完全一致。[8 路 CPU](raw/cpu-mixed8.json.gz) · [8 路 GPU](raw/gpu-mixed8.json.gz) · [16 槽 CPU](raw/cpu-mixed16.json.gz) · [16 槽 GPU](raw/gpu-mixed16.json.gz)

冷 prefill 有小幅改善，8 路重叠吞吐持平。16 槽下读长提示时，余下 15 路两轮均减速，最坏输出间隔拉长。本次保留本地候选，生产实例恢复原监听、二进制和配置，API 状态确认 16 槽可用。组间整套测试按 CPU、GPU 顺序运行，每组混合场景两轮，时间值反映这轮 A/B 的波动范围。[运行记录](raw/run.log) · [分组执行记录](raw/route-coverage.log) · [恢复记录](raw/restore-receipt.json)

## 测量方法

- 冷 100k 每组三条贪婪请求。比较引擎 prefill、API 首字、输出摘要和 token 计数。
- 8 路混合负载先建立全体 decode，再插入一条无缓存 100k。每组两轮。吞吐用长请求处于读取时、八条既有请求仍保持 decode 的服务端输出计数，输出间隔使用 SSE 内容事件。
- 16 槽饱和测试先建立 16 路 decode，再发长请求并核对排队。预设较短的一路完成后，长请求复用该槽，余下 15 路继续 decode。分别报告 16 路稳态、真实重叠段和排队时长。每组两轮。
- A/B 运行前核对原服务和候选二进制。所有测试请求只在 `127.0.0.1:8080` 上发送。运行脚本需要将本文目录的两个 Python 文件和前几轮的 `run_ab.py`、`prefill_ab.py`、`bench_decode.py` 放在同一个私有远端实验目录。[测试脚本](run_gpu_mmq_ab.py) · [混合负载](bench_decode_insert.py)
