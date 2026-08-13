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

第一阶段机制 gate 要求 held-out `correct - shuffled` 的 dynamic IoU、new-occupancy IoU 和 gripper recall 均为正，correct-action 超过 copy-initial 与 train-mean 两个 action-independent baseline，并且 full 的正确动作指标与动作敏感性都超过 matched compute control。报告另给出三类配对效应的 95% bootstrap 区间；未通过硬 gate 时停止增加训练预算，先检查 action conditioning 或更换监督表示，通过后才比较同预算动作误差和 LIBERO success rate。

正式训练前必须在含 LIBERO/HDF5 依赖的 `gigaworld-libero-eval` 环境运行 `scripts/preflight_counterfactual_stage1.py`。它逐个重新验证 raw HDF5 哈希、语言到 suite task 的对应、完整 demo action 匹配、`states[0]` 与 frame-0 state、预测与NPZ哈希、action clipping关系、horizon、二值mask、gripper包含关系和有限值，并验证split零重叠、policy/dataset一致、tracked Git状态干净以及checkpoint空间预算。只有生成 `status: passed` 的审计报告才允许启动两组训练。

原始 LIBERO-Spatial 的 10 个 HDF5 固定到 Hugging Face revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`；文件名、字节数和 SHA256 记录在 `configs/libero_spatial_raw_demos.json`，由 `scripts/verify_libero_raw_demos.py` 在 cache 生成前先做 size gate、再做完整 SHA256 gate。正式链路统一由 `scripts/run_counterfactual_stage1_pipeline.sh` 等待 30k EMA checkpoint 和 raw demo 门禁，依次执行 v3 单步 smoke、预测动作导出、200/50 cache、preflight、matched/full 训练、held-out 评价和 pair audit，不再使用结果目录中指向 v2 cache 的临时 watcher。

服务器 `/home/ubuntu/project/WM/gwp-action-video-results/counterfactual_mask_stage1_v3_pipeline_30k_ema/raw_demo_verification.json` 已对 10/10 文件通过精确字节数和 SHA256 检查，pipeline 随后进入 `waiting_checkpoint`。为防止 30k baseline 的模型、EMA 和 DeepSpeed resumability state 挤满结果盘，`scripts/compact_completed_checkpoint.py` 只在训练 PID 退出且 `transformer`/`transformer_ema` 各自的推理权重均超过下限后删除同一 checkpoint 的 optimizer/`pytorch_model` 状态，并保留 JSON 审计报告；它不会删除任一推理权重。

baseline 的四套件评测和 stage-1 都使用 GPU 2–5，因此不能在 checkpoint 刚出现时竞态启动。`scripts/evaluate_libero_standard4_parallel.py` 现在要求训练 PID 已退出且 checkpoint 同时存在 config 和非空权重文件；stage-1 pipeline 则继续等待 baseline 评测进程结束，并验证四个 suite 各 100、总计 400 rollout 的 `summary.json` 与 30k EMA checkpoint 一致，之后才允许执行 smoke 和 action/cache 导出。

训练日志中 LeRobot episode 82 的 AV1 解码会偶发失败并被 dataset 回退为全零；它不在 stage-1 的 200/50 split 内，但仅靠训练日志没有覆盖全部选中样本。`scripts/audit_lerobot_video_split.py` 因此在 action 导出前以同一 PyAV backend 逐个解码 250 个 frame-0 source 在 0/12/24/36/48 horizon 的 agent/wrist 双视图，拒绝缺帧、非有限值和全零/常量 fallback，并把逐 episode 统计写入 pipeline 审计报告。

服务器视频审计已覆盖 2500 张选中帧并以 `zero_or_constant_fallbacks=0` 通过。另一次 checkpoint 路径审计发现 Accelerate 的 DeepSpeed save hook 不会预填 `weights`；BF16 compact hook 现复用父实现的语义，在空列表时通过 accelerator gather 完整 state dict 后再 cast，避免 matched/full 到 step 200 才因错误假设而保存失败。

pipeline 重启时允许复用已通过的视频审计，但会重新验证 dataset 绝对路径、两个 split 的路径与 SHA256、250 episode、五个 horizon、双视图、PyAV backend 和零 fallback；任何字段漂移都会硬失败，不会用旧报告跳过覆盖检查。

两组训练统一通过 `scripts/launch_counterfactual_stage1_train.py` 启动。launcher 只接受当前 Git commit 生成且输入未变更的 passed preflight 报告，固定 GPU 2–5、200 steps、单卡 batch 1、梯度累积 4、seed 6666 和 mask weight 0.1，并为两组生成相同的 `comparison_id`。`matched` 与 `full` 的唯一训练差异分别为 `mask weight=0 + compute-control=1` 和 `mask weight=0.1 + compute-control=0`；若另一组 manifest 的公共设置不同则拒绝启动。

stage-1 继续使用 30k official baseline 的 `visual_loss_weight=1`、`action_loss_weight=5`、CAME8Bit `lr=6e-5` 和 constant scheduler；早期 launcher 曾误把 action weight 写成默认值 1，但在任何正式 stage-1 训练启动前已修正。因而 matched/full 相对 30k baseline 的共同变化只包含训练数据限定、200-step 预算、batch/梯度累积以及新增但零权重的 mask 计算图，不会额外改变动作/视频主损失的相对权重。

GigaTrain 在 `Trainer.__init__` 中先调用 `set_seed(seed)`，随后才在 `prepare()` 中构造模型，因此 baseline checkpoint 中不存在的新 mask head 应在两组中得到相同初始化。trainer 额外记录 `Counterfactual mask head initial SHA256`；正式结果审计要求 matched/full 日志中的该哈希完全相同，以直接排除随机 head 初始化差异。

两组训练与 held-out 评测完成后使用 `scripts/audit_counterfactual_stage1_pair.py` 做只读审计：它验证 preflight/launch 哈希、公共配置、仅允许的三个环境差异（输出目录、mask weight、compute-control）、初始 mask-head 哈希、step-200 compact checkpoint、50/200 held-out/train cache 和 taskwise derangement，并独立重算 mechanism gate。除 full 必须优于 shuffled/copy-initial/train-mean 外，gate 还要求 full 的正确动作指标和 `correct-shuffled` 动作敏感性都优于 matched compute control，避免把随机 mask head 或共享 backbone 原有的动作敏感性误报为学到的对齐；两组评测同时保留逐样本结果，并以固定 seed 做 10,000 次配对 bootstrap，报告动作敏感性、full 对 matched 的正确动作提升和敏感性提升的 95% 区间。科学 gate 失败会被记录为 `mechanism_gate.status=failed`，但与产物损坏导致的 audit failure 明确区分。

机制 gate 通过后的控制收益必须在相同 suite/task/init-state/rollout index 上配对评价。新版 `evaluate_libero_closed_loop.py` 在 summary 中固定记录 dataset、task ids、rollout 数、max/replan steps、视图数、环境与policy seed、checkpoint、动作采样步数和model dtype；`scripts/compare_libero_paired_rollouts.py` 拒绝 rollout key、任务描述或科学协议不同的输入，并报告 paired wins/losses/ties、exact two-sided McNemar p 值、逐任务差异以及固定 seed 的 10,000 次 task-stratified paired bootstrap 95% 区间。旧版 baseline summary 若缺少协议字段仍可按 rollout key 比较，但报告会明确标记 protocol 未完整验证，不能把它与 matched/full 的严格配对比较混为同等级证据。

Open-loop 动作误差仅作为控制变化的诊断，不作为策略优劣的主结论。新版 `evaluate_libero_open_loop.py` 固定保存预测与 expert action、对应 SHA256 和完整 policy 协议，除 MSE 外报告 MAE、translation/rotation/gripper 分量、gripper 符号一致率、动作越界与 clip 后误差，以及与 mask 对齐的 1/12/24/36/48-step prefix 指标；`scripts/compare_libero_open_loop.py` 只接受相同 episode、相同 expert target 和相同采样协议，并对每项 left-vs-right 诊断做 task-stratified paired bootstrap。这样闭环收益若来自 gripper、连续控制或单纯 clipping，可以被区分，但 expert action 多解问题仍由闭环评测兜底。

Mechanism gate 通过后，baseline/matched/full 分别由 `scripts/evaluate_libero_policy_bundle.py` 在独占 GPU 与端口上运行同一套 held-out open-loop 和 LIBERO-Spatial 10-task × 10-rollout closed-loop。Bundle 拒绝 tracked-dirty 工作树与不完整 checkpoint，固定单个 inference server 的 checkpoint/seed/dtype/sampling steps，记录 Git commit、checkpoint config/weight sizes、VAE config、norm stats、dataset info、episode split 和两个 summary 的 SHA256；三组完成后再由两个 paired comparator 读取，避免 open-loop 与 closed-loop 意外使用不同 policy 配置。

`scripts/run_counterfactual_stage1_postgate_control.sh` 固定到显式 `GWP_CF_TARGET_COMMIT`，等待 Stage-1 pipeline 正常退出后先验证 pair audit；机制 gate 失败时以 `stopped_mechanism_gate_failed` 正常停止，不同步代码也不占 GPU，只有 gate 通过才 fast-forward 到固定 commit 并在 GPU 2/3/4 并行运行三个 bundle。最终 `scripts/audit_counterfactual_control_evaluations.py` 验证 bundle、checkpoint、split 与 summary 哈希，并从原始 summaries 独立重算四份 paired comparison；仅当 full 相对 matched 的闭环成功率差为正且 paired wins 多于 losses，才允许继续 head-only 与 GT-action control，统计显著性另以 McNemar p 值和 task-stratified CI 单独报告，不用一次小样本正差值冒充显著结论。

mini MoT 的 mask-only backward 已确认这条机制在计算图上成立：mask head、visual patch embedding、action encoder、state encoder 和 joint-attention blocks 的梯度绝对值和均非零；未调用的 action decoder 与 RGB `proj_out` 梯度为 0。也就是说 mask supervision 会训练“如何把动作 token 作为未来视觉 consequence 的条件”以及共享交互层，但不会把 mask loss 直接当成 action decoder 的目标；动作输出是否改善仍必须由 matched-data control 和控制评测决定。

服务器CPU机制审计进一步验证了两项不变量：future visual token可以读取完整48-step action chunk，而state/action/ref token均不能读取future visual noise；双视图训练中右侧wrist的valid mask为0，其logit梯度严格为0，改变右半屏伪target也不改变loss。因而当前分支既有动作到未来视觉的信息通路，也不会把未知wrist区域错误监督成背景。

### 与 rendered-action / mask world model 前例的边界

相关工作核对后，第一阶段的结论必须限定为“predicted-action consequence representation”，不能直接写成“预测 RGB 与动作已经对齐”：

- [BridgeV2W](https://arxiv.org/abs/2602.03793) 将由 URDF、动作和相机参数渲染的 embodiment mask 作为 ControlNet-style **输入条件**，再用 logged action 对应的真实视频训练生成目标。它解决 coordinate action 到 pixel-space condition 的接口问题，但没有给偏离 logged action 的 policy prediction 构造 counterfactual RGB target。
- [Robot-Factored World Models via Robot Rendering](https://arxiv.org/abs/2607.22535) 先用机器人控制器/运动学把命令实现成 deployment-available nominal trajectory，再渲染 URDF mesh RGB 与 depth 作为视频模型输入；真实视频仍来自 logged trajectory。其消融说明二维重叠不能表达接触与遮挡顺序，加入 end-effector/scene depth 后更好。该工作同时强调不能把 logged future state 直接作为 condition，否则会泄漏需要预测的 interaction outcome。
- [Masked Visual Actions](https://arxiv.org/abs/2607.19343) 同样把部分可见的 robot/object trajectory 作为 pixel-space action interface；重点是 rendered action prompt，而不是在没有 counterfactual RGB 时监督 policy-predicted branch。
- [Mask World Model](https://arxiv.org/abs/2604.19683) 把真实 demonstration 的未来语义 mask 经同一个 video VAE 编码，并以 flow matching 预训练 mask dynamics；第二阶段不再使用 mask loss，而是让 action diffusion head cross-attend 多层 predictive feature bank，并仅以 action loss 联合微调。它证明 mask-centric prediction 对控制可能有用，但没有处理“predicted action 无真实未来 RGB/mask”的反事实监督问题。
- [Mask2Real-WM](https://arxiv.org/abs/2607.04546) 将 action-conditioned mask dynamics 与 mask-to-RGB rendering 拆成两阶段，是“先学可控几何、再恢复外观”的最近表示前例；其 mask target 来自仿真/真实轨迹，而非一个联合 WAM 自己预测动作所诱导的无 RGB 分支。

当前实现把 mask head 接在 `visual_for_out` hidden tokens 上，而不经过 RGB `proj_out`。因此 held-out correct-action 超过 shuffled/copy/mean 只能证明这些 future visual tokens 学到了 predicted action 的可渲染机器人后果；它不能证明 RGB `proj_out` 生成的机器人位置也与该动作一致。这一点与 BridgeV2W 的“rendered mask 是输入条件”以及 MWM 的“mask 本身是生成目标”都不同，也是当前方案最重要的证伪边界。

更严格地说，机器人世界模型的 action-conditioned dynamics 可以拆成两段：

```text
action command -> controller / kinematics -> robot trajectory
robot trajectory + scene -> contact / object response -> future observation
```

当前 counterfactual robot-mask branch 只为第一段提供 predicted-action 分布上的监督。它没有凭空恢复第二段缺失的 counterfactual object motion；“scene response”仍只能从 expert-action + real future RGB 的 demo branch 学习。当前假设因此应表述为：先消除 world model 在 off-expert action realization 上的分布偏移，使 demo branch 已学到的 interaction prior 能在 policy trajectory 上被调用。若 mask gate 通过但 RGB-path/控制均不改善，结果就说明只补第一段不足，不能声称模型已经学到预测动作的完整未来后果。

后续严格采用单变量分叉，不同时增加多个模块：

1. 若第一阶段 mechanism gate 失败，停止控制评测，先修复 action conditioning 或监督时标，不增加 RGB/depth/object target。
2. 若 mechanism gate 通过且 full 相对 matched 的控制收益为正，先运行既定 `head-only` 与 `GT-action mask`，判断收益来自共享 backbone 还是独立 head/普通 mask 辅助任务。
3. 若 mechanism gate 通过但控制不提升，下一项最小实验不先扩大成完整 BridgeV2W/ControlNet，也不直接叠加 depth。只改变 **mask loss 的挂载点**：先在真实 RGB latent 与真实 robot mask 上后训练一个 mask probe 并冻结；counterfactual branch 从 RGB `proj_out` 的 flow 还原 predicted-clean video latent，经冻结 probe 得到 mask，再与同一 predicted-action rollout mask 比较。其余 action cache、200/50 split、预算、matched compute control 和 gate 保持不变。这样 mask loss 只有在 RGB 生成路径包含正确机器人几何时才能下降，可直接区分“独立 auxiliary head 学会了 mask”与“视频输出真的服从预测动作”。
4. 只有第三步仍表现为“二维 mask 可学但接触相关控制不升”，才把 target 从 binary robot mask 单变量升级为 link-aware occupancy + robot/end-effector depth；依据 Robot-Factored 的结果，这一步针对的是 link 对应、遮挡顺序和假接触歧义，而不是再次更换整体架构。

第三步仍不让 consistency loss 直接优化 action 数值：同一个 action 同时决定 rendered target 和 video condition，自一致性本身没有任务最优方向；若直接回传到 action，策略可能只选择视频模型最容易解释的动作。动作是否改善继续由共享表示与原始 expert action loss决定，并以 paired LIBERO success rate 作唯一控制结论。

### 第二种挂载点：clean-video-latent frozen probe（实现中，尚无结果）

为避免把独立 auxiliary mask head 学会几何误写成“视频生成结果与动作一致”，新增了一个默认关闭的 `clean_latent_probe` 路径。反事实分支仍使用同一初态、同一 predicted-action rollout mask、同一纯噪声和 `action.detach()`；唯一变化是必须先经过视频模型的 RGB `proj_out` 得到 flow。在 `sigma=1` 下用 `predicted_clean = noise - predicted_flow` 恢复 one-step clean video latent，再经过预先在“真实视频 clean latent → 同步真实 robot mask”上拟合并冻结的 probe，最后计算相同 BCE+Dice。这样梯度路径变为 `mask loss → frozen probe → predicted clean latent → RGB proj_out → shared backbone`，而不是绕过 `proj_out`。

该路径目前只完成了代码挂载点与形状/梯度 smoke test，还不能进入正式比较：必须先生成与真实示范视频严格同步的 robot-mask 标签，单独训练 probe，并在 held-out demo 上证明 probe 的 dynamic/new-occupancy/gripper 指标足够高后冻结。随机或弱 probe 会给视频模型提供无意义梯度，因此不会为了尽快开跑而使用。另一个明确限制是当前从纯噪声做 one-step clean estimate；若 probe gate 通过而 action sensitivity 不通过，下一次只将其替换为少步可微 flow rollout，不同时修改 target 或 loss。

实现位置为 `world_action_model/alignment/latent_mask_probe.py`、`world_action_model/trainer/wa_casual_trainer_mot.py` 和 `configs/giga_world_policy_0_5_libero_finetune.py`。默认 `GWP_COUNTERFACTUAL_MASK_ALIGNMENT_MODE=hidden_head`，所以该实现不会改变正在等待的 Stage-1；只有显式设置 `clean_latent_probe` 并提供冻结 probe checkpoint 才会启用。

为训练这个 probe，新增了与 counterfactual cache 隔离的 demonstration-mask cache：它从已经审计过的 episode→raw-demo 唯一匹配出发，以 raw simulator 语义的 expert action（保留 `{-1,+1}` gripper，不使用 LeRobot 的 `{1,0}` 编码）重放同一初态，在 0/12/24/36/48 渲染 robot/gripper mask。cache 中的 replay RGB 被 manifest 硬标记为 `registration_audit_only_not_a_probe_or_video_target`，只用于证明 replay mask 与 LeRobot 中真实 demo 视频的时空位置一致，不能成为反事实 RGB target。

probe 使用一个固定的小型局部 decoder，而不是仅用逐 latent-cell 线性分类器：`Conv3d(48→128, kernel=1×3×3) → SiLU → Conv3d(128→4×8×8, kernel=1)`，再做 temporal/spatial pixel shuffle，正式参数量 88,448。这样 probe 有最小的局部上下文来恢复机器人边界，避免把“mask 非线性可解码但非线性 probe 太弱”误报成视频 latent 不含几何；它仍远小于 backbone，并由独立 200/50 held-out gate、冻结加载和 checkpoint approval 限制。mini MoT 复测中 `proj_out_grad=0.0888`、action-encoder grad=0.314，probe grad tensor 数为 0，detached action input 也无梯度。

注册门在看到正式结果前固定为：position error p95≤5 mm、rotation error p95≤2°、gripper max-abs p95≤0.003、机器人区域最佳平移 p95≤4 px、全图 RGB MAE p95≤10/255、最佳平移后的 robot-region RGB MAE p95≤15/255。实现于 `scripts/audit_libero_demonstration_mask_registration.py`。probe 训练器 `scripts/train_libero_latent_mask_probe.py` 拒绝任何未通过注册门的 train/held-out cache；训练后的 held-out probe 门固定为 full IoU≥0.65、dynamic/new-occupancy IoU≥0.40、gripper recall≥0.70，且这四项同时超过 copy-initial 和 train-mean。失败 probe 会保留作负结果，但 checkpoint 写入 `approved_for_alignment=false`，`LatentRobotMaskProbe.from_pretrained` 会硬拒绝加载。

服务器已排队一个单 episode 注册 smoke，固定 commit `6bfd735` 和 GPU 5，只等待 Stage-1 退出；post-gate 控制固定使用 GPU 2/3/4，二者可以安全并行。当前状态位于 `/home/ubuntu/project/WM/gwp-action-video-results/clean_latent_probe_registration_smoke_op88/status`。只有该 smoke 通过上述固定门，才生成完整 200/50 demonstration-mask cache；不会先跑大规模数据再根据结果放宽标准。

probe 的真实视频输入与正式 GWP 固定为同一双视图布局：`agent-view | wrist-view` 横向拼接成 256×512 后再过同一个 Wan VAE，而不是用单 agent-view latent 训练 probe、再外推到双视图 latent。监督 target/valid 同样是 256×512，但只有左侧 agent-view valid=1，右侧 wrist valid=0。真实 LeRobot 单样本审计已验证输入 `[5,3,256,512]`、未来 mask/valid `[4,256,512]`、左侧有效像素 262144、右侧有效像素严格为 0；该改动只消除 probe 与 GWP 的输入分布混杂，没有改变注册门、probe gate 或 counterfactual loss。

clean-latent 正式训练还增加一个训练前 action-conditioning gap gate，不能仅凭“GWP 在 expert action 下应该能生成视频”作为未经验证的前提。`scripts/evaluate_clean_latent_action_gap.py` 对每个 held-out 初态固定同一 future noise，分别评价 `expert correct/shuffled` 与 `policy correct/shuffled`，前者使用已注册的 demonstration mask，后者使用 predicted-action rollout mask；shuffle 均为 task 内无固定点循环置换。dynamic IoU、new-occupancy IoU、gripper recall 三项必须同时满足 `expert correct - expert shuffled > 0` 且该动作敏感性大于 policy-action 动作敏感性，才允许开始 clean-latent 对齐训练；报告同时保存逐样本效应和固定 seed 的 10,000 次 paired bootstrap CI。若 expert gate 失败，说明 one-step `sigma=1` clean estimate 尚不能表达 baseline 已有的视频能力，下一步只改少步 flow rollout，不训练当前 loss；若没有 expert-policy gap，则该方法的核心分布偏移叙事在当前表示上不成立。

action-gap 中 expert action 的语义也做了真实数据核对：episode 0 的 raw HDF5 action 经相同 no-op filter、gripper `{-1,+1}→{1,0}` 转换后，与 LeRobot parquet 的前 48 步逐元素完全一致（转换前与 q01/q99 归一化后 `max_abs` 均为 0）。因此 expert/policy 差异不会来自两套 action 编码。

`scripts/launch_counterfactual_stage1_train.py` 现将 `clean_latent_probe` 作为独立协议，而不是给 hidden-head launcher 临时塞一个 probe 路径。clean-latent 启动必须同时提供 passed probe report 和 passed action-gap report；launcher 会重新核验 checkpoint/VAE、200/50 episode 顺序、双视图布局、probe config/weights 与所有 cache/report SHA256，并锁定 probe 训练参数、注册阈值、held-out 最低指标、action-gap seed/threshold/10,000 次 bootstrap 以及 paired-noise/taskwise-derangement 语义。任一 report 为空、字段缺失、哈希漂移或 gate 失败都会拒绝训练。上述 provenance 进入 matched/full 共用的 `comparison_id`，并将已批准 probe 的绝对路径写入训练环境；默认 `hidden_head` 不接受 probe/gap 参数，避免两条挂载点误配。

旧 Stage-1 evaluator 曾在 eval mode 向 `_forward_inference` 传入仅训练路径支持的 `output_action/output_visual` 参数；mini MoT 实测会报 `unexpected keyword argument`。旧等待型 pipeline 在任何正式 action/cache/matched/full 产物生成前已终止并归档到 `counterfactual_mask_stage1_v3_pipeline_30k_ema_superseded_02a9b97_20260813T1007Z`，没有删除。新 pipeline 固定到已同时通过 hidden-head 与 clean-latent 前向 smoke 的 commit `6bfd735`，代码记录、preflight、训练、held-out evaluator 与 post-gate target 使用同一 commit；当前结果根仍为 `counterfactual_mask_stage1_v3_pipeline_30k_ema`。

30k 共同 baseline 在 2026-08-13 10:28 UTC 于 step 17,940 收到 SIGTERM；日志最后一个完整 step 的 loss 有限，但进程未正常完成。该次运行只在 step 30,000 保存，因此 `models/` 为空，自动 evaluator 明确报出“training process exited before a complete checkpoint appeared”，compactor 也以缺少目标 checkpoint 失败。Stage-1 和 probe watcher 仍分别停在 `waiting_checkpoint`/`waiting_dependencies`，没有把这次中断误当成可用 baseline，也尚未产生正式 matched/full 或 clean-latent 结果。下一次启动前必须先解决长任务生命周期和同盘仅余约 84 GiB、无法安全同时保留大型可恢复 checkpoint 的问题；不会从 step 17,940 的日志伪造恢复点。

## 结果位置

- Spatial 6k：`/home/ubuntu/project/WM/gwp-action-video-results/libero_spatial_1000step/<method>/`
- 三套件 6k：`/home/ubuntu/project/WM/gwp-action-video-results/libero_sgo_6000/<method>/`
- 每个方法下的 `open_loop_*`、`consequence_pose_*`、`closed_loop_*` 分别保存动作、几何 overlay/summary 和闭环结果。
