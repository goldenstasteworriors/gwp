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
  frame，并同时绑定源文件路径和 hash。最大允许同步误差为 50 ms。
- 未生成缩略图、contact sheet 或视频副本；不会把私有视觉内容带入 git。

## 隔离和分发（协调员执行）

仓库中的 A/B 目录只是协调员母版，不能把整个仓库权限给标注者。分发前：

1. 给 A、B 分配真实且互不相同的 `annotator_id` 和 `annotation_session_id`。
2. 分别替换 `annotator_a/identity.json` 与 `annotator_b/identity.json`，并同步替换各自
   `coverage.csv` 的 `annotator_id`。不要改 packet ID、任务表或证据表。
3. 把 `annotator_a/` 和 `annotator_b/` 复制到两个互不可见的访问控制位置；每人只获得
   自己的目录，以及 `timeline_evidence_index.csv` 中 12 个 `video_path` 对应 MP4 的
   **逐文件只读 allowlist**。
   不要授予 source 根目录浏览权限或 processed HDF5 权限，因为同目录存在 timing 文件，
   HDF5 也含禁止作为真值依据的 hand status。不要发送 `coordinator/`、`adjudication/`、
   另一个 packet，或仓库中旧的 proposal/prescreen/timing 文件。
4. 保存分发时各文件 SHA-256 和访问控制记录。机器检查能拒绝相同路径、硬链接、相同
   annotation hash 和相同身份/session，但**不能证明两个人现实中没有沟通**。

`assignment_receipt.json` 只记录未分发母版的初始 hash。身份或标签开始修改后它不会自动
更新，正式交付应使用检查器输出作为新 receipt。

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
`video_time_s` 可直接用于本地播放器定位；不要截取或提交媒体副本到 git。

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
真实第三人身份/session，完成裁决表后运行：

```bash
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py \
  --phase adjudication \
  --agreement-report track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/agreement_report.json \
  --adjudication-root track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/adjudication
```

若需要重新核验所有大源文件 hash，在任一命令末尾添加 `--verify-source-hashes`；默认检查
存在性、字节数及 manifest 中 hash 形状，避免每次重复读取大文件。

## 质量门槛与 No-Go

沿用现有协议：coverage 线性加权 κ≥0.70、event F1≥0.80、匹配 segment median tIoU≥0.70、
双方 observed boundary median MAE≤2 processed frames（100 ms）、side/site/object role/contact
mode 各自至少 2 个可比匹配且 κ≥0.60。低于任一门槛，必须修订定义并独立重标 3 段校准集，
不能靠裁决“洗掉”系统性低一致性。

即使门槛通过，也只有第三人裁决表能进入现有 sidecar builder。没有两位真实标注者、真实
独立 session、完整逐帧复核或第三人裁决时，本目录仍是 **No-Go / 非 GT**。66 段缺失的
subject/session/object/liquid/batch 元数据也继续禁止 OOD 或 strategy claim。

## 工时估计

12 段原视频约 3 分钟，但正式逐帧、回看边界、填类别和自检通常需要每位标注者约 2–3
小时；第三人裁决约 1–1.5 小时；协调员分发、收件和报告约 0.5–1 小时。总计约 5.5–8.5
人时。若 3 段校准重标或透明/遮挡分歧多，需另加约 1.5–3 人时。
