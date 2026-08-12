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

## 正在进行的主实验

- LIBERO-Spatial：`off`、`policy_only`、`joint`、`video_only` 从 step 1000 保留 optimizer state 续训到 step 6000。
- LIBERO-Spatial+Goal+Object：`off` 与 `policy_only` 使用 1314 episodes、171996 states，从相同 GWP-0.5 初始化训练 6000 steps。
- 训练完成后重复开环、同初态几何 consequence 和闭环 success rate；跨套件结果分别报告，不把不同 suite 混成一个平均数。

## 当前判定标准

只有当 `policy_only` 在多个 suite 上改善闭环 success rate，并且优于 `joint`/`video_only`，才能支持“视频 consequence 梯度改善动作”的核心因果叙事。若只改善动作 MSE 或 EEF 误差而不改善成功率，则结论应降级为 representation/trajectory regularization；若 6k 与跨套件实验均无闭环收益，则应否定当前形式，而不是继续强调视频生成质量。
