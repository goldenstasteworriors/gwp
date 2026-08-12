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

## 结果位置

- Spatial 6k：`/home/ubuntu/project/WM/gwp-action-video-results/libero_spatial_1000step/<method>/`
- 三套件 6k：`/home/ubuntu/project/WM/gwp-action-video-results/libero_sgo_6000/<method>/`
- 每个方法下的 `open_loop_*`、`consequence_pose_*`、`closed_loop_*` 分别保存动作、几何 overlay/summary 和闭环结果。
