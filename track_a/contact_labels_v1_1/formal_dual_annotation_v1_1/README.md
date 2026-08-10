# Ego contact v1.1 正式双标执行包

本目录是 12 段 minimum set 的**空白正式任务包**，不是 GT。它固定了同一份
episode coverage、逐 processed 帧的视频证据索引、A/B 独立身份槽和第三人裁决门槛。
任何 timing hint、prescreen、hand status、动作 saliency 或模型 proposal 都没有进入
A/B packet。

## 已审计的数据边界

- canonical：`913811dab0a7ee74a0493f19d7bab40225546d99`。
- source：`/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers`。
- processed HDF5 与原始 MP4 均为 episode 0–65，共 66 段；processed 总计 20,013 帧。
- minimum assignment：12 段、3,695 个 processed 20 Hz 帧（184.75 秒），coverage 均为完整 episode。
- `coordinator/source_episode_manifest.csv` 记录 66 段的绝对路径、字节数、HDF5/MP4
  SHA-256、视频尺寸/帧率和 merge sync 统计。只有入选 12 段的 depth timestamp sidecar
  计算 SHA-256，因为它们才参与时间轴证据映射。
- 每个 packet 的 `timeline_evidence_index.csv` 将每个 processed frame 映射到最近的原视频
  frame，仅提供 opaque `media_id`、隔离目录相对路径、相对帧/秒和视频 hash。绝对 source
  路径、原始采集 timestamp、processed HDF5 路径/hash 均只留在 coordinator 区域。最大允许
  同步误差为 50 ms。
- 未生成缩略图、contact sheet 或视频副本；不会把私有视觉内容带入 git。

## 隔离和分发（协调员执行）

仓库中的 A/B 目录只是协调员母版，不能把整个仓库权限给标注者。分发前：

1. 给 A、B 分配真实且互不相同的 `annotator_id` 和 `annotation_session_id`。
2. 分别替换 `annotator_a/identity.json` 与 `annotator_b/identity.json`，并同步替换各自
   `coverage.csv` 的 `annotator_id`，将 `identity_status` 改为 `assigned_by_coordinator`。不要改
   packet ID、任务表或证据表。
3. 把 `annotator_a/` 和 `annotator_b/` 复制到两个互不可见的访问控制位置。协调员把 12 个
   已核 hash 的 MP4 复制或只读挂载到各自 packet 的 `media/<task_id>.mp4`；禁止使用指向
   source 根目录的 symlink。每人只能访问自己的 packet 与这些隔离媒体，不能获得 source
   根目录、processed HDF5、`coordinator/`、`adjudication/`、另一 packet 或仓库权限。
4. 分发前和收件后都运行 checker。annotator 根目录只允许 receipt 中列出的 7 个文件；可选
   `media/` 也只允许精确的 12 个 MP4 名称，内容必须匹配 source hash。任何额外文件、目录、
   symlink、proposal、GT、prescreen 或另一方材料都会被拒绝。
5. 保存分发时各文件 SHA-256 和访问控制记录。机器检查能拒绝相同路径、硬链接、相同
   annotation hash 和相同身份/session，但**不能证明两个人现实中没有沟通**。

`assignment_receipt.json` 固定母版文件 allowlist、可选媒体 allowlist、coordinator audit、
tasks 和 timeline hash；assignment 阶段还固定空白 identity/coverage/segments。身份或标签开始
修改后，immutable receipt 仍必须通过，正式交付另保存 checker 输出作为 submission receipt。

## 标注动作

每位标注者独立完成以下流程，定义见 [LABEL_DICTIONARY.md](LABEL_DICTIONARY.md)：

1. 对每段原视频以原速、慢速、逐帧和 ±10 processed 帧上下文完整检查。
2. 先填一行 full-episode coverage。只有逐帧完整排查、可见性足以判断所有接触时，才可
   填 `exhaustive`；只确认正例但无法排除遗漏时填 `positive_only`；整体不可用填 `unknown`。
3. 每个真实物理接触区间填一行 `segments.csv`，区间使用 `[start_frame,end_frame)`。
   同一物理接触持续区间只有一个 `segment_id`，不能按滑窗拆分。
4. `review_status=submitted`、`label_origin=human_independent`、`annotation_role` 保持 packet
   固定值；`source_proposal_id` 与 `parent_segment_ids` 必须为空。
5. `review_timestamp_utc` 使用形如 `2026-08-10T12:34:56Z` 的 UTC 时间。

直接可见的接触填 `rgb_direct`；需要前后状态变化确认填 `rgb_temporal`。不要看或使用
`hand_status`、episode 0 timing、prescreen、模型输出。透明界面、遮挡或运动模糊不足以
支持类别/边界时必须留空或标 unknown，不允许按任务常识补标签。

查询任一 processed frame 对应的原视频帧和 ±10 帧上下文（只读、无图片输出）：

```bash
python scripts/contact_labels/lookup_contact_timeline_evidence_v1_1.py \
  --evidence-csv track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/annotator_a/timeline_evidence_index.csv \
  --episode 43 \
  --processed-frame 120 \
  --context 10
```

A 使用 `annotator_a` 路径，B 必须换成自己隔离副本里的 `annotator_b` 路径。输出中的
`media_relpath` 与 `video_time_s` 可直接用于隔离目录中的本地播放器定位；不要截取或提交
媒体副本到 git。

## 校验命令

以下命令需要按展示位置换行，每个反斜杠后立即换行。

母版检查（CPU，只检查空白与隔离，不认可任何 GT）：

```bash
cd /home/ykj/project/WM/gwp-agent-op17-formal-label-packet
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py \
  --phase assignment
```

A/B 都提交后，先检查身份、session、CSV 语义和文件独立性：

```bash
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py \
  --phase submission
```

计算机器可读一致性报告：

```bash
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py \
  --phase agreement \
  --write-report track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/agreement_report.json
```

只有报告 `all_gates_pass=true` 才能启动第三人裁决。替换 `adjudication/identity.json` 中的
真实第三人身份/session，将 `identity_status` 改为 `assigned_after_agreement_gates_pass`，完成
裁决表后运行：

```bash
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py \
  --phase adjudication \
  --agreement-report track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/agreement_report.json \
  --adjudication-root track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/adjudication
```

每次 checker 都会重算全部 66 段 processed HDF5/MP4 以及入选 12 段 timestamp sidecar 的
SHA-256，并从 source timestamp 数组重新生成 3695 行映射。不存在 size-only hash 复用或仅信任
CSV 中 sync 数值的快速通道。

## 质量门槛与 No-Go

沿用现有协议：coverage 线性加权 κ≥0.70、event F1≥0.80、匹配 segment median tIoU≥0.70、
双方 observed boundary median MAE≤2 processed frames（100 ms）、side/site/object role/contact
mode 各自至少 2 个可比匹配且 κ≥0.60。低于任一门槛，必须修订定义并独立重标 3 段校准集，
不能靠裁决“洗掉”系统性低一致性。

`--phase agreement` 在任何 gate 失败时写出 `status=failed` 并以非零状态退出；报告仍保留完整
指标用于诊断。第三方裁决 coverage 必须精确引用同 episode 的 A+B coverage。裁决 segment 的
parent 只能来自同 episode；只有单方 parent 时，`notes` 必须写
`unilateral_parent_reason=<原因>`，明确记录另一方遗漏、不可见或不接受该事件的裁决理由。

即使门槛通过，也只有第三人裁决表能进入现有 sidecar builder。没有两位真实标注者、真实
独立 session、完整逐帧复核或第三人裁决时，本目录仍是 **No-Go / 非 GT**。66 段缺失的
subject/session/object/liquid/batch 元数据也继续禁止 OOD 或 strategy claim。

## 工时估计

12 段原视频约 3 分钟，但正式逐帧、回看边界、填类别和自检通常需要每位标注者约 2–3
小时；第三人裁决约 1–1.5 小时；协调员分发、收件和报告约 0.5–1 小时。总计约 5.5–8.5
人时。若 3 段校准重标或透明/遮挡分歧多，需另加约 1.5–3 人时。

## 隐私、授权与留存

- 分发前，协调员必须记录数据控制者、获批用途和授权/同意或适用的伦理审批编号；没有明确
  允许人工标注和向指定标注员展示的授权时不得分发。
- 原视频可能包含可识别的人体、声音、环境或采集时间信息，应按受限人类参与者数据处理。
  Packet 已移除绝对路径和绝对 timestamp，但协调员仍须完成可识别信息复核与最小化。
- 使用逐人、最小权限、到期自动撤销的只读访问；记录授权人、标注员、文件 hash、开放/撤销
  时间和访问日志。禁止下载到个人云盘、转发、二次使用、截图或录屏。
- 隔离媒体只保留到该轮 submission 验收完成；最长保留期必须写入分发记录。到期后删除所有
  工作副本并记录销毁确认。CSV、receipt 和审计日志按项目获批期限留存。
- 发现越权访问、错误分发或可识别信息泄漏时立即撤销访问、停止标注、保全日志并通知数据
  控制者；完成事件评估和重新授权前不得恢复。
