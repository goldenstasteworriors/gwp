# 双人标注与裁决协议

## 独立轮

- A/B 先对同一 episode 独立填写 coverage，再填写 physical segments；不共享屏幕、ID 或中间 CSV。
- 独立轮 `annotation_role=independent_a|independent_b`、`review_status=submitted`、`label_origin=human_independent`。自动候选和 episode 0 timing hint 必须隐藏。
- 标注者可看原速、逐帧及 ±10 帧上下文。直接可见用 `rgb_direct`，依赖状态变化判断用 `rgb_temporal`；不得把 hand_status、prescreen、模型输出或常识当物理接触真值。
- 透明器材界面不可辨时，把类别留空或将边界记 unknown；禁止补写 stream、liquid 或接触位置标签。

## 匹配与一致性

先在同一 episode 内以 segment temporal IoU 最大化做一对一匹配；候选需 IoU≥0.5，且双方已知 side 不冲突。同一 A/B segment 最多匹配一次，避免一个长事件被多个短事件重复计数。

报告：

- coverage mode 的加权 Cohen's kappa；
- 双方 exhaustive 区间交集上的逐帧 contact Cohen's kappa 和正类/负类分开一致率；
- event-level precision、recall、F1，以及匹配 segment 的 temporal IoU；
- 双方均 observed 的 onset/release MAE（frame 与 ms）；
- 匹配且双方已知字段上的 side/site/role/mode kappa；
- left/right censored 与 unknown 的混淆矩阵。

不能只报告总体逐帧 accuracy：contact 稀疏时，大量共同负帧会掩盖事件分歧。

进入裁决的最小门槛：coverage weighted κ≥0.70、event F1≥0.80、匹配 segment median tIoU≥0.70、observed boundary median MAE≤2 processed frames（100 ms）、类别 κ≥0.60。未达门槛先修订定义并重做 3 段校准，不得用裁决覆盖系统性低一致性。

## 裁决轮

- 第三人同时看 A/B 原记录和原视频，不能由 A 或 B 单方面覆盖。
- 每个 consensus segment 创建新的全局 `segment_id`；`parent_segment_ids` 记录 A/B ID，以分号分隔。observed onset/release 创建新的 event ID；删失/unknown 边界留空。
- consensus coverage 同样创建新 `coverage_id`，`parent_coverage_ids` 记录来源。
- 裁决表必须为 `annotation_role=adjudicated`、`review_status=reviewed`、`label_origin=human_adjudicated`，builder 才接受。
- 无法裁决的区间降级为 `positive_only` 或 `unknown`，不能为了增加训练量升级为 exhaustive。

正式启动门槛：12 段最小集全部双标、指标过线、validator 通过，且人工抽查 NPZ provenance 能由任意 window query 回溯到唯一 global segment。否则 sidecar 仅可用于工程调试。
