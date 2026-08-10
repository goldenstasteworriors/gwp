# Track B：透明倒液的 2D 可见性审计 v0

本目录是只读数据审计与仿真设计产物，不修改或依赖 GigaWorld Policy 核心模型。它回答一个先决问题：现有 Ego 倒液视频是否足以监督 mesh-free 的 boundary/rim/liquid/stream/contact 场，而不是假定透明物体能被稳定重建。

## 抽样与可复现性

- 数据源：`/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers`。
- episode：0、29、58、30、14。episode 0 是唯一有人工 contact/separation 时刻的段；其余分别取 66 段 processed 长度分布约 25%、50%、75%、90% 分位的最近 episode。
- 每段把 processed action 时间轴等分为 20 个相位箱；每箱选择 `action_delta_eef` 鲁棒归一化 L2 运动幅值与 `hand_status` 跳变加权后的最大值。
- episode 0 已有的 contact/separation 提示只覆盖其各自所在箱，不传播到其他段，也不作为视觉真值。
- processed 帧到视频帧按 `camera_timestamp`（ns）与 depth sidecar 的 `color_system_timestamps_us`（us）最近邻同步。当前 100 帧记录的最大同步误差为 0 ms。
- 本样本是 phase-stratified feasibility audit，不是独立同分布随机样本，也没有为成功/失败 outcome 做平衡。

复现或一致性检查：

```bash
cd /home/ykj/project/WM/giga-world-policy
python track_b/visibility_audit_v0/build_visibility_audit.py
python track_b/visibility_audit_v0/build_visibility_audit.py --check-only
```

## 标注对象与 0–3 可见性编码

标注的是“能否从当前 RGB 帧局部化为 2D 场”，不是标注者根据动作常识推断其存在。

| 字段 | 操作定义 |
|---|---|
| `boundary` | 所有任务相关容器的外轮廓；至少足以形成容器占据/碰撞 proxy |
| `source_rim` | 当前承担倒出角色的容器口沿中心线或闭合椭圆 |
| `target_rim` | 当前承担接收角色的容器口沿中心线或闭合椭圆 |
| `liquid_surface` | 任一任务相关容器内部可见的液面边界；备注需说明 source 或 target |
| `stream` | source 与 target 之间的可见连续/断续液流区域，不用手势推断 |
| `contact` | 手/末端执行器与容器的实际接触界面；仅轮廓重叠不等于可见接触点 |

统一编码：

- `0`：当前帧不存在或不适用；若时序证据不足以确认存在，也记 0 并备注。
- `1`：根据相邻时序预计存在，但当前帧不能可靠局部化；包括透明、反光、运动模糊或完全遮挡。
- `2`：部分可见或有歧义，但能给出带不确定性的 2D proxy。
- `3`：清楚可见，可稳定局部化。

特别规则：`0` 与 `1` 不得合并。训练 mask 应让 `0` 表示“无实例”，让 `1` 表示“未知/不可观测”，否则会把不可见液流错误监督为零液流。

## 双人标注流程

`annotations_dual_template.csv` 已含 100 行与 A/B 独立列。执行正式标注时：

1. A、B 只看单帧及相邻 ±2 原视频帧，独立填写六个 0–3 值、confidence 和 notes；不得查看另一人的列。
2. 先报告每特征的加权 Cohen's kappa、完全一致率，以及“可局部化（2/3）/不可局部化（0/1）”二值一致率。
3. 任一特征相差 2 级以上、0/1 有分歧、或 1 与 2/3 有分歧时必须裁决，结果写入 `adjudicated_*`。
4. 通过门槛：rim/boundary 的二值 kappa ≥0.70；liquid/stream/contact ≥0.60；任何特征 0/1 分歧率 ≤10%。未通过则先修订定义并重标 20 帧校准集。

`prescreen_annotator_a.csv` 是单人 contact-sheet 预审。它只能判断任务是否值得正式双标，不能作为训练真值，也不能用于报告人际一致性。

## 单人预审结论

详见 `prescreen_summary.json`。100 帧中可局部化（2/3）的比例：boundary 70%，source rim 70%，target rim 100%，liquid surface 20%，stream 0%，contact 0%。liquid surface 的 20 个清晰帧全部来自有紫色液体的 episode 58；其余透明液面只被评为 1。26 个预计存在 stream 的帧和 49 个预计存在 contact 的帧都无法从静态 RGB 可靠定位。

因此现有数据可支持 rim/boundary teacher 的可见性验证，但不能直接支持 liquid/stream/contact 的像素级监督。后者至少需要短时序标注、带染色液体/偏振或背光的补采，以及力/触觉或受控仿真事件标签。

## 文件说明

- `manifest.csv`：100 帧来源、时间戳、同步误差和抽样理由。
- `frames/`：100 张带 audit ID 的 320×240 JPEG。
- `contact_sheets/`：每 episode 一张 4×5 审阅图。
- `annotations_dual_template.csv`：A/B 独立标注与裁决模板。
- `prescreen_labels.json`：人工预审原始记录。
- `prescreen_annotator_a.csv`、`prescreen_summary.json`：预审展开表与统计。
- `sim_scenario_spec.json`：与 simulator 无关的场景设计和 outcome 定义。

## 下一门槛

先完成第二名标注者和裁决；若 rim/boundary 通过一致性门槛，再在不训练 GWP 的情况下做一个冻结 2D teacher 的离线 precision/recall 与 temporal-jitter baseline。stream/contact 不应在当前 RGB 数据上伪造 dense ground truth；必须先有至少 10 段可见液流/同步接触传感的补充验证集，或仅把仿真字段作为 risk auxiliary target。

