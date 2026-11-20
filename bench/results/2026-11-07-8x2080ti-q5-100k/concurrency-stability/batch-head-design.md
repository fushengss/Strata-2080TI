# Pipelined MTP 的 slot draft head 分担设计

状态：**候选，尚未实现或执行 GPU 测试**。先等待现有 pipelined MTP 的隔离 A/B 正确性和收益判定，再决定是否实施。本文件基于 `Strata-decode-head` 工作树的 `b4c45a6c`；并发 prefill guard 的独立改动不属于此设计。

当前机器是 8×RTX 2080 Ti、Q5 主模型，生产已有主 head `2,4` 与 solo MTP draft head `3,5`。本次只读检查没有访问 GPU、读取服务状态或操作服务。下文显存数字是源码尺寸计算，不是新测量，也不代表速度收益。

## 现状与不能直接复用的部分

现有 `MtpDrafter` slot clone 已共享主 drafter 的 dense、expert、Q6 subset head 权重和 subset→global 映射。每个 slot 仍有自己的 KV、残差、Q8、logits、stream 和 graph。`STRATA_MTP_HEAD_GPU=3,5` 目前只初始化主 drafter 的 `MtpHeadShards`，只在 solo PW2 作用域激活；slot 的 piped chain 仍在 GPU7 计算完整 draft head。

`MtpHeadShards` 并不是可直接由多个 slot 同时调用的共享对象。每个实例只有以下一组可变资源：

- 一份 host-mapped Q8 activation 和一份完整 subset logits。
- 每张目标卡一份 Q8、半份 logits、一个 stream 和一个捕获了这些地址的 graph executable。
- 八个 source producer events、每个目标卡八个 finished events。
- 一个 `last_source_stream_`，用于 `drain()` 等待最后一次 source 消费。

当前安全性依靠**同一 source stream**上的顺序：body 写 Q8 → producer event → 两目标 head → finished events → source waits → 完整 logits 回传 → 原 tail → 下一步 body。同一 chain 的多步可一次性排队；同一 step index 只能在上次 chain 排空后复用。

八个 slot 各有 source stream，并且都会使用 `step=0`。把它们指向同一个 helper 会覆盖 mapped activation/logits、重录同一 event，且 `last_source_stream_` 只保留最后一个消费者。只加 host mutex 不能解决：`enqueue()` 返回时 GPU 仍在消费这些数据。共享 target stream 也不能单独防止另一 source stream 提前覆盖 mapped activation。

## 最小实现边界

推荐把 helper 分成**共享只读分片权重**和**每个 drafter 私有 workspace**。主 drafter 和八个 slot 引用同一对 GPU3/5 分片；每个 slot 保留独立 mapped buffers、Q8/logits、streams、events 和 target graphs。图中只共享只读 weight pointer，所有写入地址仍私有。

权重所有者宜用引用计数表达生命周期。每个 workspace 在销毁时先等待自身 source 消费和 target streams，再释放其权重引用；最后一个引用才释放权重。不能仅让 clone 保存裸指针、仍由主 helper 无条件 `cudaFree(weights)`。现有主模型权重的 clone 生命周期规则继续保留。

具体入口：

1. `include/strata/core/mtp_head.hpp`、`src/core/mtp_head.cpp`：增加共享 weight bank 或经过完整验证的 `init_shared`，分离 weight 释放与 workspace 释放；保留 CUDA/HIP device guard、Metal unavailable stub、部分初始化失败的清理。
2. `MtpDrafter::bind(..., shared)`：目前复制 `dhead_`、类型、`dvocab_` 和词表长度，但不复制 `dhead_plain_`。应复制可信布局元数据，或在新 shared-prepare API 中从 owner 验证它。不能直接删除现有 `prepare_head_shards()` 的 `owns_draft_head_` 检查，让任意 borrowed head 绕过布局检查。
3. 增加类似 `prepare_shared_head_shards(const MtpDrafter& owner, err)` 的明确 API：检查 source device、head/subset/map 指针、类型、N、词表长度和目标设备均匹配 owner；只建立私有 workspace，不重新拷权重。
4. `generate.cpp` 的 slot `bind/prepare_chain` 目前约在11627行，早于主 helper 约11833行的初始化。最小调整是在主 helper 成功初始化后，为已绑定的 slot 建立 shared helper，再调用 `prepare_chain()` 补捕获 body/tail。已有普通图会命中缓存，无需删除或挪动原 slot 初始化流程。必须在任何请求/GPU chain 运行前完成全部分配和捕获。
5. 为 piped-slot head 增加独立 opt-in，默认关闭，例如 `STRATA_PIPED_MTP_HEAD=1`。初版依赖现有 `STRATA_MTP_HEAD_GPU` 的已初始化权重，并要求 `piped_mtp` 真正生效。仅在 slot `chain_launch` 使用分片；admission 的 `draft_first`、prefill 和普通 draft 图先保持原路径。
6. 每个 slot 的 chain 仍是 body → source event → 两目标 head → 完整 logits 回传 → 原 argmax / row_top_prob / subset map / mtp_select / force tail。保留已有 `chain_live_` 与 `chain_poll` 复用保护；取消、缓存保存、slot↔solo 切换先经过 `pmtp_settle`。统计每个 slot 的分片 step 数，避免开关配置被误当成路径生效。

只需改 helper、MTP 包装和 generate 初始化/激活钩子；当前 CMake 已包含 helper。不要在这一候选中改变主模型 verifier、分布式 softmax、接受规则或 prefill chunk。

**采样范围有一处区别：**现有 slot drafter 没有调用 `set_draft_sampling(req_sp)`，其 proposals 始终 greedy，即使 target 请求的温度为1或带 penalty/bias。分片若保持 slot 的完整 logits、argmax 和原概率 tail 精确一致，原则上可覆盖这些 sampled-target slot。应按 drafter 实际模式检查 `!coupled_active_`，不能照搬 solo 的“target 必须 greedy”条件而遗漏真实工作负载；这只是设计可行性，仍需采样端到端验证。未来若给 slot 加 coupled sampling，此候选必须回退或明确拒绝该模式。

## 静态资源估算

沿用当前 plain Q6_K subset：`N=2560`、`vocab=106299`；分片切点保持128行对齐，即53120/53179行。每行 Q6_K 为2100 bytes，每个 Q8_1 activation 为2880 bytes。

现有主 helper 已占有 GPU3 106.384 MiB 和 GPU5 106.502 MiB 的只读分片权重。下面仅是新增 slot workspace 的显式分配量：

| 新增资源 | 每 slot | 4 slots | 8 slots |
| --- | ---: | ---: | ---: |
| GPU3 Q8 + 半份 logits | 0.2054 MiB | 0.8215 MiB | 1.6431 MiB |
| GPU5 Q8 + 半份 logits | 0.2056 MiB | 0.8224 MiB | 1.6449 MiB |
| Host-mapped Q8 + 完整 logits | 0.4082 MiB | 1.6330 MiB | 3.2660 MiB |
| GPU7 新增显式数据 buffer | 0 | 0 | 0 |
| 额外 source body/tail graph executables，max_t=3 | 6 | 24 | 48 |
| 额外 target graph executables | 2 | 8 | 16 |
| 额外 target streams | 2 | 8 | 16 |
| helper events，沿用八步容量 | 24 | 96 | 192 |

GPU7 使用各 slot 本来就有的 `xq_`、`head_logits_` 和输出/概率 buffers。**表中没有计算 CUDA graph、event、stream、映射页表及驱动的实际显存开销**；尤其48张 source body/tail 图可能使实测增量明显超过上面的数据 buffer 量，必须以加载后的 free VRAM 和 graph 捕获后的变化为准。

若直接对每个 slot 调用当前 `init()`，每次都会重复上传两份分片。八个 slot 将额外浪费 GPU3约851.07 MiB、GPU5约852.02 MiB的权重，而不是上表约1.64 MiB的 scratch。共享权重是避免这一成本的关键，不能只共享 source GPU7 的原始 subset pointer。

这些数字是在已经启用 slot MTP 的基础上增加的。此前普通 piped MTP 的8个 clone、deferred-commit buffers、pinned KV 是另一笔成本，不能用本表替代。

## 正确性与收益的风险

- **跨 slot 覆盖：**必须有不同的 mapped地址、graph输出地址和event对象；不能仅靠不同的event index区分全部slot。
- **生命周期：**source 的 logits copy/tail 也消费 helper-owned mapped memory。只等待目标卡完成就释放会有悬空读取；权重共享也必须等所有 workspace 排空。
- **布局：**维持 plain Q6_K、精确 Q8_1 和原 subset row 顺序。Q4、repacked Q6、full-vocab 或不匹配的共享布局继续明确拒绝。106299与53179不是四的倍数，回传沿用任意N的32-bit word copy，不能回退到要求16-byte对齐/四元素倍数的 copy。
- **图内地址：**target graph 与 workspace 绑定，不能直接共享同一个 graph executable 给不同 slot。不要在多个在途 launch 之间修改同一 graph 的节点参数。
- **拥塞：**当前 GPU3/5 的低占用来自此前 solo trace；并发 piped decode 时其 stage 工作更密集，不能用 solo 空闲比例预测收益。八个高优先级 head streams 也可能拖慢这些卡的 trunk stage。跨设备 host 调度、mapped完整logits回传会增加固定开销。
- **收益范围：**此候选只移走 draft head，GPU7 的 MTP body、KV以及 batch 主 head仍在原卡。它不能单独消除新 prefill 造成的调度空档。

最低验证顺序是：先确认普通 piped MTP 在本机通过；随后低层测试两个以上 workspace 真正同时在途、相同 step index、不同Q8输入、打乱poll顺序，逐slot检查完整logits/probability/map bitwise；再验证单workspace销毁或取消不破坏其他workspace。之后做同一binary开关A/B，覆盖4路seed sampled、mixed penalties/bias、取消恢复、slot↔solo、长prompt加入和100k solo prefill。记录3/5/7卡的实测显存、每slot offloaded steps、接受率、逐路token gap及吞吐。只有正确性、容量与实际收益同时成立，才考虑启用。

参考实现：`include/strata/core/mtp_head.hpp`、`src/core/mtp_head.cpp`、`src/core/mtp.cpp` 的 `bind` / `prepare_head_shards` / `prepare_chain` / `chain_launch`，以及 `src/program/generate.cpp` 的 `slot_mtp` 初始化、`pmtp_verdict`、`pmtp_settle`。现有 `tests/core/mtp_head_test.cpp` 验证的是单helper多步事件复用，不能替代多slot同时在途的新测试。
