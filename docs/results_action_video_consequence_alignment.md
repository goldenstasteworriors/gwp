# Action–Video Consequence Alignment：实验记录

更新时间：2026-08-13

## 研究问题

目标不是让视频头生成得更好，而是检验视频预测误差能否作为动作头的 consequence critic：动作头提出未来动作，视频头预测该动作对应的未来，视频流匹配误差对动作求梯度，并只把这部分梯度回传给动作分支。核心比较如下：

- `off`：原始 GigaWorld Policy 联合视频/动作训练。
- `policy_only`：consequence 梯度只更新动作路径。
- `joint`：consequence 梯度同时更新动作和视频路径。
- `video_only`：动作条件 detach，只更新视频路径。
- `action_only`：不使用视频辅助损失。

## 评测协议

1. 开环动作误差：每个 LIBERO task 取相同的 episode，预测 48-step action chunk。
2. 同初态 consequence：专家动作和模型动作在两个相同初始化的 LIBERO 环境中执行，比较 1/6/12/24/36/48 step 的 EEF 位姿与机器人 mask。
3. 相机抖动补偿：在机器人区域之外用 ORB+RANSAC 估计 homography，将预测 mask 配准后计算 IoU/Dice；不允许用机器人本身估计相机运动。
4. 闭环控制：使用训练图像相同的双轴翻转约定、官方 10-step settling、每 8 步重新规划，最终报告 LIBERO success rate。

## 1k-step 早期消融（LIBERO-Spatial）

开环结果来自每个 task 两个 episode，共 20 个 episode。数值均为 MSE，越低越好。

| 方法 | 总 MSE | 平移 | 旋转 | 夹爪 |
|---|---:|---:|---:|---:|
| off | 0.386897 | 0.741032 | 0.017767 | 0.431877 |
| policy_only | **0.219720** | **0.436389** | **0.005948** | **0.211032** |
| joint | 0.258641 | 0.491121 | 0.009311 | 0.309191 |
| video_only | 0.285401 | 0.519761 | 0.012400 | 0.401321 |
| policy_only, weight=0.4 | 0.305727 | 0.555397 | 0.014338 | 0.430882 |
| policy_only, every step | 0.308820 | 0.562911 | 0.016241 | 0.424283 |
| action_only | 0.333669 | 0.619624 | 0.013693 | 0.435737 |

相对 `off`，默认 `policy_only` 的总 MSE 降低 43.2%，平移/旋转/夹爪 MSE 分别降低 41.1%/66.5%/51.1%。更大的对齐权重或每步计算 consequence 都更差，说明该信号更适合作为稀疏、较弱的 policy regularizer。

10 个任务各一个 episode 的同初态 consequence 指标：

> 2026-08-13 初态溯源审计：这里的历史几何数值使用了“同 task 内 episode rank → `.pruned_init` rank”的映射。`.pruned_init` 是另行采样的 benchmark 评测初态，并非 demonstration 初态；因此这些数值只保留为旧版诊断记录，不能作为与训练视频对齐的证据。开环动作 MSE 不受此问题影响。

| 方法 | 配准后 mask IoU ↑ | EEF 像素误差 ↓ | EEF 坐标轴端点误差 ↓ | SO(3) 误差/度 ↓ |
|---|---:|---:|---:|---:|
| off | 0.5194 | 45.51 | 45.09 | 11.60 |
| policy_only | 0.5082 | 39.86 | 39.75 | **7.94** |
| joint | **0.5257** | **37.14** | **37.32** | 9.42 |
| video_only | 0.5131 | 44.34 | 44.16 | 8.84 |

这里出现了有价值的分离：`policy_only` 和 `joint` 改善 EEF consequence，但 mask IoU 没有同步单调改善。因此 mask 只能作为几何诊断指标，不能替代控制成功率。

相机补偿指标的受控验证：给相同 rollout 人为加入最多 3 px、0.6° 抖动，原始 IoU 为 0.8182，背景配准后为 0.9777，EEF 误差为 0.20 px；加入动作噪声后，配准 IoU 仅为 0.7729、EEF 误差为 5.90 px。说明该指标能去除相机抖动，同时不会把真实动作误差一并消掉。

修正在线图像方向后，四个核心方法各 20 次闭环 rollout 均为 0% success。该结果说明 1k steps 不足以验证任务成功率；不能据此声称方法改善了控制。

## 6k-step 主实验（LIBERO-Spatial）

五个方法均从各自 step-1000 checkpoint 恢复 optimizer state 并训练至 step 6000。开环仍使用完全相同的 20 个 episode。

| 方法 | 总 MSE | 平移 | 旋转 | 夹爪 |
|---|---:|---:|---:|---:|
| off | **0.069097** | **0.129222** | 0.003119 | **0.086655** |
| policy_only | 0.073330 | 0.133820 | 0.002922 | 0.103085 |
| joint | 0.075305 | 0.133581 | **0.002778** | 0.118057 |
| video_only | 0.097592 | 0.186205 | 0.004642 | 0.110602 |
| action_only | 0.099046 | 0.181536 | 0.003860 | 0.137133 |

`policy_only` 的总 MSE 比 `off` 高 6.1%，因此 1k 时的 43.2% 优势没有保持。`action_only` 比 `off` 高 43.3%，说明原始视频辅助任务对动作学习有明显价值；但在其上额外施加 consequence gradient 没有带来稳定收益。

同初态几何 consequence 使用与 1k 相同的 10 个任务/episode：

> 本表同样受上述 `.pruned_init` 映射问题影响，不再用于支持方法结论；后续几何 cache 必须通过 raw HDF5 demo state 溯源门禁。

| 方法 | 配准后 mask IoU ↑ | EEF 像素误差 ↓ | EEF 坐标轴端点误差 ↓ | SO(3) 误差/度 ↓ | 48-step EEF 误差 ↓ |
|---|---:|---:|---:|---:|---:|
| off | 0.6354 | 19.82 | 19.74 | 8.56 | 38.46 |
| policy_only | 0.6315 | 20.57 | 20.69 | 8.19 | 39.07 |
| joint | **0.6491** | **19.02** | **19.26** | **7.21** | **36.37** |
| video_only | 0.6333 | 21.49 | 21.36 | 7.67 | 44.40 |

`joint` 在几何代理上最好，但其开环动作 MSE 不如 `off`；`policy_only` 仅改善旋转误差，其 EEF 和 mask 指标均未优于 `off`。这进一步表明“优化视觉 consequence 代理”不等于“得到更好的动作策略”。

四个核心方法各执行 20 个 LIBERO-Spatial 闭环 rollout，全部为 0/20 success。该训练预算下所有方法都尚未达到可用闭环水平，因此闭环结果不能区分方法，但足以否定“当前实验已证明控制收益”。

## 三套件复现实验

`off` 与 `policy_only` 从相同 GWP-0.5 初始化，在 LIBERO-Spatial+Goal+Object 的 1314 episodes、171996 states 上共同训练 6000 steps。每个 suite 的开环结果来自每个 task 两个相同 episode，共 20 个 episode：

| Suite | off 总 MSE | policy_only 总 MSE | 相对变化 |
|---|---:|---:|---:|
| LIBERO-Spatial | **0.096449** | 0.099050 | +2.7% |
| LIBERO-Goal | **0.089440** | 0.093570 | +4.6% |
| LIBERO-Object | 0.053820 | **0.053416** | -0.8% |

Object 上的微小改善来自夹爪项（0.0407 vs 0.0484），但平移和旋转 MSE 均更差；Spatial 和 Goal 的总误差均退化。因此没有跨 suite 的一致收益。

闭环在每套 suite 的 10 个任务上各跑 1 次；`off` 与 `policy_only` 均为 Spatial 0/10、Goal 0/10、Object 0/10，总计各 0/30。

## 结论与下一版方案

本轮结果不支持当前 `policy_only` 形式。最合理的解释是：视频 flow-matching loss 对动作的梯度主要反映高维像素生成的局部敏感性，并不等价于任务进度或可控性；在训练早期它像一个正则项，随后原始动作监督追上并超过它。与此同时，`action_only` 显著更差，说明应保留 GigaWorld Policy 原有的视频辅助训练。

下一版不应继续强化“让动作与整帧生成结果对齐”，而应把对齐目标改成低维、控制相关且时标一致的 consequence：

1. 从未来视频 latent 解码 EEF/机器人 occupancy/object-contact token，只对这些 token 与动作 rollout 的几何结果做一致性，不对整帧 flow loss 求动作梯度。
2. 将 48-step 动作聚合到视频帧对应的 12/24/36/48 step，显式解决动作与视频采样间隔不同的问题。
3. 使用 stop-gradient target encoder 或置信度门控，只在视频 consequence 预测可靠时更新动作；保留原始动作监督为主目标。
4. 首先在可达到非零 success 的更长训练/EMA 配方上复现实验，再用 success rate 选择方法，EEF/mask 只作为机制诊断。

只有当新方法在多个 suite 上改善闭环 success rate，并且优于 `joint`/`video_only`，才能支持“视频 consequence 改善动作”的核心因果叙事；只改善 MSE 或 EEF 误差时，结论应降级为 trajectory regularization。

## 反事实机器人 mask 方案（进行中）

前面的 `policy_only`、`joint`、`video_only` 都把预测动作条件下的视频输出与专家动作产生的真实未来视频比较，反事实目标不成立。新版不再使用这项 loss，而是把监督拆成两条语义独立的数据流：

1. Demo branch：真实动作条件下继续使用真实未来 RGB flow supervision，保留原始 GWP 的视频辅助任务。
2. Counterfactual branch：缓存策略预测的 48-step action chunk，从同一 LIBERO 初态执行该动作，在 12/24/36/48 step 渲染 robot mask；未来 visual latent 从纯噪声开始，只从 future visual tokens 预测这四张 mask，不读取专家未来 RGB，也不经过 RGB `proj_out`。

旧 v2 单 episode 链路曾得到四个未来 mask 的 robot pixels 为 4960、6522、7093、7648；这些数值只说明该错误初态内的 rollout 不是静态复制，不再视为正式数据验证。v3 必须重新生成相同 smoke sample 后才能报告新的动态性结果。

方向审计发现 v1 cache 只对 raw MuJoCo segmentation 做了竖直翻转，而 LeRobot LIBERO 训练视频来自 `LiberoEnv.render()`，其明确同时翻转高、宽两个轴；因此 v1 mask 与视频 token 存在左右镜像错位，不能用于训练。v2 修正了 H/W flip，但随后初态溯源审计发现 v2 仍把“同 task 内 LeRobot episode rank”当成 `.pruned_init` 编号。服务器 episode 0 的 v2 horizon-0 EEF 与数据集首帧相差约 6.46 mm；枚举 50 个 `.pruned_init` 也只能找到约 1 mm 的近邻，因为这些文件本来就是独立采样的评测初态，并非 demonstration state。因此 v1/v2 都被拒绝用于正式训练。

cache schema 现升级为 `libero-counterfactual-robot-mask-v3`：生成器必须读取原始 LIBERO `<task>_demo.hdf5`，对 raw action 做与 LeRobot 转换相同的 no-op 过滤，并把 raw gripper `{-1,+1}` 转成 LeRobot `{1,0}`，再用完整 action 序列唯一匹配 demo。审计还发现 LeRobot 的 `task_index` 顺序与 LIBERO suite 顺序不同；v3 从 `meta/episodes` 读取语言指令，再与 suite task language 做唯一精确匹配，同时记录 dataset task index 和 LIBERO task index，禁止按整数直接对应。

服务器 v3 smoke 中，dataset task 0 被正确映射到 LIBERO task 6；LeRobot episode 0 唯一匹配 raw `demo_40`，110-step action、10-step settling 后的 8D frame-0 state 最大误差和 L2 误差在 float32 下均为 0。v3 将 raw HDF5/demo/state 哈希及两侧 frame-0 state 写入 cache，生成时和 preflight 都以 `source_state_max_abs <= 1e-5` 为硬门禁。mask 仍采用与训练视频相同的 H/W flip。只有通过这三类门禁的 v3 cache 才能进入 stage-1。

该 v3 smoke 的 horizon 0/12/24/36/48 robot pixels 为 4211/5060/6618/7186/7763，相对初态的 dynamic pixels 为 2679/4335/5489/6886。`copy-initial` 的 full IoU 均值仍有 0.3999，但 dynamic IoU 和 new-occupancy IoU 都为 0、gripper-region recall 均值仅 0.1510；因此正式 mechanism gate 继续要求 action-dependent 的 dynamic/new-occupancy/gripper 指标，不能只看 full mask。

服务器的LIBERO配置不在默认 `~/.libero`，而在数据盘的 `datasets/libero_config/config.yaml`。cache生成和闭环评测脚本会在导入LIBERO前从dataset父目录非交互地发现并验证该配置；cache manifest记录配置路径与SHA256，preflight拒绝配置缺失或生成后发生变化的cache，避免长时间watcher最终因交互提示或仿真资产路径漂移失败。

旧 v2 cache 上的 `copy-initial` 数值也因源初态错误而作废；“必须同时报告 full/dynamic/new-occupancy/gripper”这一评价原则仍保留。评价实现位于 `world_action_model/alignment/geometry.py`，并显式报告 empty-union fraction，避免空动态区域悄悄抬高均值。

### 必须报告的 mask 指标与对照

训练 loss 下降不能作为方法有效的证据。held-out cache 至少需要报告：

1. `full robot IoU/Dice`：预测未来 robot mask 与同一预测动作 rollout 得到的 target 比较。
2. `dynamic mask IoU`：分别计算预测/target mask 相对初态 mask 的 XOR，再比较两者，防止模型只复制静态机器人区域。
3. `new occupancy IoU`：仅比较未来新出现的机器人像素 `M_future & ~M_initial`。
4. `gripper-region recall`：target gripper mask 被预测 robot mask 覆盖的比例，避免大面积机械臂和基座掩盖末端错误。
5. 分 horizon 报告 12/24/36/48 step，而不是只给平均值。

必要对照为：

- `copy-initial`：四个时刻都复制 horizon-0 robot mask。
- `per-horizon mean mask`：不看动作，只输出训练集每个 horizon 的平均 mask。
- `shuffle-action`：保持初态不变，打乱 batch 内动作；正确模型的 mask 指标应显著下降。
- `zero/noise-action`：从相同初态执行零动作或扰动动作，检验 head 是否真的读取 action consequence。
- `head-only`：冻结 backbone，只训练 mask head；若与全量更新相同，则不能声称 backbone 学到了额外未来信息。
- `GT-action mask`：用专家动作对应 mask 训练，区分“mask 辅助任务本身”与“适应策略动作分布”的收益。

控制结论仍以同预算、同初始化的 LIBERO success rate 为准。只有当 counterfactual mask 方法同时优于原始 GWP、`head-only` 和 `GT-action mask`，并且 held-out shuffle-action 检验成立时，才支持“让视频/共享 backbone 适应动作头的预测分布，从而改善控制”的叙事。

### 第一阶段固定协议

先只在 LIBERO-Spatial 做最小实验，不同时扩展 suite 或增加其他对齐头。使用 seed 20260813 按 task 分层选择 20 个 train episode 和 5 个 held-out episode，共 200/50，episode 交集为 0；split manifest 位于 `/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_splits/seed_20260813/libero_spatial_20train_5heldout/manifest.json`。每个 episode 只使用 frame 0 的双视图观测和 baseline 预测的第一个 48-step action chunk。

从同一个 30k baseline EMA 初始化，先各继续训练 200 steps：

1. `matched-data control`：读取完全相同的 200-sample sidecar，设置 mask weight 为 0，同时开启 `GWP_COUNTERFACTUAL_MASK_COMPUTE_CONTROL=1`；它仍采样相同 future noise 并执行相同 mask forward，用于同时对齐数据、RNG消耗和计算量，仅移除 mask 梯度。
2. `counterfactual-mask full`：数据、初始化、步数、batch、随机种子均相同，增加 mask BCE+Dice，允许 mask 梯度更新共享 backbone 和 mask head。
3. 暂不立即运行 `head-only`。只有 full 方法在 held-out 上的 correct-action 指标明显优于 shuffled-action，并且优于 copy-initial/mean-mask，才运行 head-only 来判断收益是否只来自新 head。

第一阶段使用的是30k baseline在固定初态预测后缓存的 action chunk，训练时不重新调用当前动作头，也不让 mask loss 对 action tensor 求梯度。因此它只检验“共享backbone能否学习 baseline policy 分布上的反事实动作后果”，不是在线动作—视频闭环对齐。若机制 gate 和控制评测均成立，下一次最小增量才从更新后的 full policy 重新导出 action/mask cache，再做一轮相同 matched/full 比较；若缓存刷新后收益保持，才能声称该方法能迭代适应当前策略分布。

服务器当前单个 FP32 5B transformer checkpoint 约 23 GB，且框架的 `checkpoint_save_optimizer=False` 不会删除唯一/最新 checkpoint 的 optimizer。为避免两个对照挤满磁盘，stage-1 明确设置 `GWP_WITH_EMA=0`、`GWP_CHECKPOINT_MODEL_DTYPE=bf16` 和 `GWP_STRIP_OPTIMIZER_AFTER_CHECKPOINT=1`：只将推理模型浮点张量以 BF16 保存，并在保存后移除本次运行的 optimizer/DeepSpeed resumability state。原始30k baseline checkpoint不受影响；stage-1 checkpoint不能用于精确恢复 optimizer，但可直接用于推理和 held-out 评测。

第一阶段机制 gate 为 held-out `correct - shuffled` 的 dynamic IoU、new-occupancy IoU 和 gripper recall 均为正，并且 correct-action 超过两个 action-independent baseline。未通过时停止增加训练预算，先检查 action conditioning 或更换监督表示；通过后才比较同预算动作误差和 LIBERO success rate。

正式训练前必须在含 LIBERO/HDF5 依赖的 `gigaworld-libero-eval` 环境运行 `scripts/preflight_counterfactual_stage1.py`。它逐个重新验证 raw HDF5 哈希、语言到 suite task 的对应、完整 demo action 匹配、`states[0]` 与 frame-0 state、预测与NPZ哈希、action clipping关系、horizon、二值mask、gripper包含关系和有限值，并验证split零重叠、policy/dataset一致、tracked Git状态干净以及checkpoint空间预算。只有生成 `status: passed` 的审计报告才允许启动两组训练。

原始 LIBERO-Spatial 的 10 个 HDF5 固定到 Hugging Face revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`；文件名、字节数和 SHA256 记录在 `configs/libero_spatial_raw_demos.json`，由 `scripts/verify_libero_raw_demos.py` 在 cache 生成前先做 size gate、再做完整 SHA256 gate。正式链路统一由 `scripts/run_counterfactual_stage1_pipeline.sh` 等待 30k EMA checkpoint 和 raw demo 门禁，依次执行 v3 单步 smoke、预测动作导出、200/50 cache、preflight、matched/full 训练、held-out 评价和 pair audit，不再使用结果目录中指向 v2 cache 的临时 watcher。

服务器 `/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_stage1_v3_pipeline_30k_ema/raw_demo_verification.json` 已对 10/10 文件通过精确字节数和 SHA256 检查，pipeline 随后进入 `waiting_checkpoint`。为防止 30k baseline 的模型、EMA 和 DeepSpeed resumability state 挤满结果盘，`scripts/compact_completed_checkpoint.py` 只在训练 PID 退出且 `transformer`/`transformer_ema` 各自的推理权重均超过下限后删除同一 checkpoint 的 optimizer/`pytorch_model` 状态，并保留 JSON 审计报告；它不会删除任一推理权重。

baseline 的四套件评测和 stage-1 都使用 GPU 2–5，因此不能在 checkpoint 刚出现时竞态启动。`scripts/evaluate_libero_standard4_parallel.py` 现在要求训练 PID 已退出且 checkpoint 同时存在 config 和非空权重文件；stage-1 pipeline 则继续等待 baseline 评测进程结束，并验证四个 suite 各 100、总计 400 rollout 的 `summary.json` 与 30k EMA checkpoint 一致，之后才允许执行 smoke 和 action/cache 导出。

训练日志中 LeRobot episode 82 的 AV1 解码会偶发失败并被 dataset 回退为全零；它不在 stage-1 的 200/50 split 内，但仅靠训练日志没有覆盖全部选中样本。`scripts/audit_lerobot_video_split.py` 因此在 action 导出前以同一 PyAV backend 逐个解码 250 个 frame-0 source 在 0/12/24/36/48 horizon 的 agent/wrist 双视图，拒绝缺帧、非有限值和全零/常量 fallback，并把逐 episode 统计写入 pipeline 审计报告。

两组训练统一通过 `scripts/launch_counterfactual_stage1_train.py` 启动。launcher 只接受当前 Git commit 生成且输入未变更的 passed preflight 报告，固定 GPU 2–5、200 steps、单卡 batch 1、梯度累积 4、seed 6666 和 mask weight 0.1，并为两组生成相同的 `comparison_id`。`matched` 与 `full` 的唯一训练差异分别为 `mask weight=0 + compute-control=1` 和 `mask weight=0.1 + compute-control=0`；若另一组 manifest 的公共设置不同则拒绝启动。

GigaTrain 在 `Trainer.__init__` 中先调用 `set_seed(seed)`，随后才在 `prepare()` 中构造模型，因此 baseline checkpoint 中不存在的新 mask head 应在两组中得到相同初始化。trainer 额外记录 `Counterfactual mask head initial SHA256`；正式结果审计要求 matched/full 日志中的该哈希完全相同，以直接排除随机 head 初始化差异。

两组训练与 held-out 评测完成后使用 `scripts/audit_counterfactual_stage1_pair.py` 做只读审计：它验证 preflight/launch 哈希、公共配置、仅允许的三个环境差异（输出目录、mask weight、compute-control）、初始 mask-head 哈希、step-200 compact checkpoint、50/200 held-out/train cache 和 taskwise derangement，并独立重算 mechanism gate。科学 gate 失败会被记录为 `mechanism_gate.status=failed`，但与产物损坏导致的 audit failure 明确区分。

mini MoT 的 mask-only backward 已确认这条机制在计算图上成立：mask head、visual patch embedding、action encoder、state encoder 和 joint-attention blocks 的梯度绝对值和均非零；未调用的 action decoder 与 RGB `proj_out` 梯度为 0。也就是说 mask supervision 会训练“如何把动作 token 作为未来视觉 consequence 的条件”以及共享交互层，但不会把 mask loss 直接当成 action decoder 的目标；动作输出是否改善仍必须由 matched-data control 和控制评测决定。

服务器CPU机制审计进一步验证了两项不变量：future visual token可以读取完整48-step action chunk，而state/action/ref token均不能读取future visual noise；双视图训练中右侧wrist的valid mask为0，其logit梯度严格为0，改变右半屏伪target也不改变loss。因而当前分支既有动作到未来视觉的信息通路，也不会把未知wrist区域错误监督成背景。

## 结果位置

- Spatial 6k：`/home/ubuntu/project/WM/gwp-action-video-results/libero_spatial_1000step/<method>/`
- 三套件 6k：`/home/ubuntu/project/WM/gwp-action-video-results/libero_sgo_6000/<method>/`
- 每个方法下的 `open_loop_*`、`consequence_pose_*`、`closed_loop_*` 分别保存动作、几何 overlay/summary 和闭环结果。
