# SM-FET D0 calibration metadata packet v1

- 日期：2026-08-12
- 构建基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 状态：`D0_CALIBRATION_ONLY / NON_BLIND / NON_GT / METADATA_ONLY`
- 数据集：Ego `7_4_pour_liquid_between_beakers`
- 任务数：3
- formal blind12 交集：0

本目录只冻结 D0 非盲校准所需的小型 metadata 与空白 worksheet。它不包含 MP4、HDF5、截图、
缩略图、timeline proposal、事件 proposal、旧标签、裁决或 GT，也不授权训练、模型实现或
正式 blind 标注。六张 annotation CSV 只有 header；任何 coverage、contact、event、node、edge、
pose 都必须由真实 A/B 在
各自隔离副本中从原始 RGB 独立填写。

## 1. 与 formal packet 的硬边界

本 packet 是独立 sibling：`track_a/sm_fet_d0_calibration_packet_v1/`。它不属于
`track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/`，也不能复制、移动或软链接进
formal annotator 根目录。

formal contact v1.1 的每位 annotator 根目录永远恰好七个文件：

1. `ANNOTATION_GUIDE.md`
2. `assignment_receipt.json`
3. `identity.json`
4. `tasks.csv`
5. `timeline_evidence_index.csv`
6. `coverage.csv`
7. `segments.csv`

本目录中的 D0 `events/node snapshots/edges/pose` worksheet、身份/session 模板和 media
allowlist 都不能成为第八个 formal 文件。D0 结果不回填 formal CSV，不进入 formal agreement，
也不升级为 GT。

## 2. 规范状态与本轮边界

本 packet 的字段名只复刻 standard-r1 annotation/pairing protocol 的人工 worksheet 接口，
用于暴露定义、missingness、reference integrity 与容量歧义。统一研究索引已裁定：

- standard-r1 handbook/protocol 是历史 baseline，不能原样充当当前 minimal-r1d 的 compiler、
  checker 或 action admission contract；
- minimal-r1d 只有书面 structural freeze，CPU conformance 仍 pending；
- 因而本 packet 不声称 annotation→wire round-trip 已闭合，也不声称 D0 已通过。

本轮统一使用原始 MP4 的 `raw_video_frame_30fps` 半开 coverage，避免向 annotator 暴露
processed HDF5、HDF5-derived timeline 或绝对 source 路径。若后续冻结协议改用另一时钟，必须
升 packet/schema version、重新生成空白 packet 与 receipt；不得静默换算已有行。

只允许以下证据：本人的 packet、hash 匹配的三段隔离原始 MP4、冻结且双方相同的指南/任务
ontology、原速/慢速/逐 raw-video frame 上下文。永久禁止 hand status、hand active、depth、
contact visualization、MANO/SMPL、tracker、模型/proposal、prescreen、旧 GT、另一方首轮结果。

## 3. Coordinator proposal 隔离

episode 选择理由和 selection bias 已移到
`coordinator/selection_proposal_audit.md`。该文件不在 recipient exact allowlist 中；A/B 实际
packet roots 也没有 README、`coordinator/` 或 selection audit。`task_manifest.csv` 只保留执行所需
的中性 episode/media/clock binding，不含 event、边界、双手、遮挡、透明或 outcome hint。

未来 D1 blind A/B 不得收到本 README、coordinator selection audit、D0 讨论纪要、calibration
CSV 或任一 proposal。D0 首轮 A/B 也必须先独立锁定全部三段，不能用选择理由预填 worksheet。

## 4. 文件与分发面

Git 中的 coordinator source bundle 是：

- `README.md`：通用 coordinator runbook，不在 recipient allowlist；
- `coordinator/selection_proposal_audit.md`：唯一 episode-specific 选择审计，coordinator-only；
- `coordinator/assignment_index.json`：从 coordinator 侧冻结两套 receipt/manifest/identity/session
  SHA-256，避免 recipient 自行改写 manifest 与 receipt 后仍被接受；
- `task_manifest.csv`：三段的中性任务绑定，无 selection stratum、event hint 或 semantic proposal；
- `media_allowlist.csv`：仅中性媒体 ID、文件名、bytes、SHA-256、帧数、时长、fps、尺寸和 codec；
- `schema_manifest.json`：冻结全部八张 CSV 的 exact ordered header、header SHA-256、schema version
  与 assignment/blank file SHA-256；
- `templates/coverage.csv`；
- `templates/segments.csv`；
- `templates/events_v1.csv`；
- `templates/event_node_snapshots_v1.csv`；
- `templates/event_edges_v1.csv`；
- `templates/pose_keys_2d_v1.csv`；
- `templates/identity.json`；
- `templates/session.json`；
- `assignments/annotator_a/` 与 `assignments/annotator_b/`：实际已赋值、彼此独立的 13-file
  blank-before-assignment roots，含 receipt、recipient manifest、exact allowlist、identity/session、
  两张 contact 表与四张 SM-FET 表。

D0 recipient packet root 的唯一合法文件集合由 `recipient_exact_allowlist.json` 冻结：两张中性
manifest、`coverage.csv`、`segments.csv`、四张 SM-FET worksheet、identity、session、assignment
receipt、recipient manifest 与 exact allowlist，共 13 文件。指南/ontology 必须是双方相同、已
hash 的独立只读输入。media root 只允许 `media_allowlist.csv` 指定的三个 MP4，且与 packet root、
state root 两两不嵌套；Git packet 自身不复制媒体。

不得把本 README、`coordinator/`、绝对 source 路径、整个仓库、source root、processed HDF5、formal
coordinator/adjudication、另一方 packet 或 proposal audit 分发给 A/B。D0 是 non-blind，含义是
两人首轮锁定后允许按流程共同讨论定义，不代表首轮 CSV 可以被 proposal 预填。

ACL 最低要求：逐人 owner-only 目录；媒体与 immutable metadata 只读；六张 annotation CSV 仅 owner 可写；
禁止 symlink/hard link；禁止云同步、截图、录屏和转发；记录授权人、recipient、开放/撤销时间、
各文件 hash 和销毁确认。没有数据控制者授权/伦理或用途依据时 fail closed，不分发媒体。

## 5. 执行与 protocol-freeze 顺序

1. **Coordinator preflight**：运行 stdlib checker `--phase source`；另由 coordinator 的获批媒体
   探测流程重算三个 MP4 SHA-256/帧数/时长/fps；验证与 blind12 零交集；验证六张 annotation CSV 只有 frozen exact header、source
   identity/session template 未赋值、recipient A/B 已赋值且不同、packet 内没有媒体/截图/HDF5。
2. **Assign**：给 A、B 分配真实且不同的 `annotator_id`、`annotation_session_id` 和 owner-only
   roots；保存 assignment receipt。A/B 得到同一 task/media bytes，不得看到 selection rationale。
3. **Independent first pass**：A、B 各自完整完成全部三段后锁定提交 hash；不能边讨论边填，
   不能看对方 CSV。所有 row 仍是 `human_independent_calibration`，不是 GT。
4. **Locked comparison**：只有两方三段全部锁定并撤销写权限后，协调员才计算结构性分歧并开放
   共同讨论。讨论只针对定义、反例、unknown/mask/censor/reference/overflow，不投票造真值。
5. **One revision maximum**：协调员最多修订一次 dictionary/task ontology；任何字段/topology/
   frame-clock 改动必须升 version。保留旧/新文档 hash 和 change log，不覆盖首轮 CSV。
6. **Protocol freeze**：冻结 protocol、dictionary、task ontology、schema/header、frame clock、checker
   build 和 round-trip receipt。校准 CSV 封存为 `NON_GT`，不得合并成 adjudicated rows。
7. **D0 closeout**：撤销并销毁 calibration media working copies，保留最小访问/销毁日志；D1
   开始前对 A/B roots 做 denylist 扫描，确认没有 D0 README、讨论纪要、proposal 或 calibration CSV。
8. **Future D1 only after gates**：只有独立 checker/round-trip 满足当前冻结结构的全部 D0 条件，
   才能另行准备 D1。现有 formal contact 7-file packet 边界保持不变。

## 6. Fail-closed checker

已实现 `scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py`。它只用 Python stdlib，
不生成或修改 annotation、proposal、GT、receipt 或媒体。任一未知、缺文件、多文件、hash 不符、
ACL 不足或引用不唯一都非零退出；没有 warning-only 或 `--force` 绕过。

从仓库根目录执行 source 与 A/B assignment 检查：

```bash
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase source \
  --source-root track_a/sm_fet_d0_calibration_packet_v1

python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase assignment \
  --packet-root track_a/sm_fet_d0_calibration_packet_v1/assignments/annotator_a \
  --peer-packet-root track_a/sm_fet_d0_calibration_packet_v1/assignments/annotator_b \
  --schema-manifest track_a/sm_fet_d0_calibration_packet_v1/schema_manifest.json
```

以上代码块按展示位置换行；每个反斜杠后立即换行。若 Git checkout 后目录/文件权限不再是
`0700/0600`，assignment 必须先失败；由 coordinator 在真实分发 root 恢复 ACL 后再重跑，不得
通过关闭检查继续。

### `source-preflight`

- exact allowlist：source、templates、coordinator 与两个 recipient root 各自使用写死集合；拒绝
  MP4/HDF5/图片、额外 CSV、
  symlink、hard link、socket/device、重复 inode、NaN/Inf、JSON duplicate key；
- canonical/base、packet version、三任务数、episode 唯一性与 blind12 disjoint 均精确匹配；
- media allowlist 三行一一对应 task，并冻结已审计的 bytes/SHA-256/frames/duration/fps/size/codec
  metadata；stdlib source phase 不解析视频容器，实际媒体 bytes/SHA 由 distribution phase 重算；
- 六张 annotation CSV 必须只有 exact UTF-8 header 和 LF，零 data row；source identity/session
  template 必须 unassigned；
- 扫描 header 与所有文本，拒绝绝对 source/HDF5 路径、proposal value、GT/adjudicated value、
  timing hint 和受禁证据引用。

### `assignment`

- A/B packet exact allowlist、所有 immutable file hash 与 assignment receipt 一致；
- A/B 的 annotator/session/packet root/state root 不同，task/media bytes 相同；路径不嵌套、无共享
  inode、权限为 owner-only，媒体只读；
- 首轮前六张 CSV 仍为空；身份、session、role、protocol/header/media/task hashes 完整绑定；
- recipient denylist 中不含 README、selection audit、D0 讨论、另一方内容、formal
  coordinator/adjudication、HDF5/depth/hand/contact proposal。

### `submission`

- 每行 schema/episode/annotator/session/role 与 receipt 一致；每个 episode 恰有一条完整 raw-frame
  coverage，范围与 task/media manifest 一致；
- 六张表的 primary ID 非空且唯一，所有 foreign key 同 episode 闭合；
- `label_origin` 只接受 `human_independent_calibration`，`source_proposal_id` 必须为空；拒绝
  proposal、model、adjudicated 或 GT row。

此外 checker 对 `events.coverage_id`、`segments.coverage_id`、segment onset/release event、
node/edge/pose event 与 `edge.contact_segment_id` 做同 episode 唯一引用闭合。assignment phase
要求所有 annotation 表为空；submission phase 才接受真实 human-independent calibration rows。

### `distribution`

在真实媒体分发前必须额外提供 packet/media/state roots。checker 要求三者两两不嵌套，root
mode 精确 `0700`、packet files `0600`、media files `0400`、state 初始为空，并逐字节重算三个
外部 MP4 的 bytes/SHA-256；任一 symlink、hardlink、共享 inode、额外媒体或 proposal 文件即失败。
Git 只保存 executable bit，不能保存 `0700/0600/0400` 的完整 ACL，所以每次 checkout/复制后
必须重新 chmod 并通过 `--phase assignment/distribution`；Git mode 永远不算 ACL 证据。

### `discussion-freeze`

- A/B 全任务提交 hash 与 write-lock receipt 齐全后才允许比较；只输出 disagreement/change-log，
  不输出 merged label 或 GT；
- dictionary/task ontology 最多一次 revision；freeze receipt 绑定所有规范/header/checker hashes；
- D0 media ACL 已撤销、working copy 已销毁，calibration output 标记 `NON_GT/EXCLUDED_FROM_D1`；
- D1 preflight 对每个未来 blind A/B root 执行 proposal/selection/calibration denylist，且正式
  contact packet 仍严格通过原有 7-file checker。

本 checker 的授权范围只到结构、hash、ACL、引用与 assignment 空白态；它不判断 event 语义、
不生成标签，也不实现 action wire。未来 D0 admission checker 仍必须另行实现 annotation→当前
冻结 wire→annotation 的唯一 round-trip、reachable capacity/codebook/public-state receipt；在
index 所列 minimal-r1d CPU conformance 未通过前，任何 worksheet 都不能宣称 D0 pass 或 action
admission。

## 7. 人时预算

本三段 D0 session 预计共 **3.25–4.0 人时**：coordinator preflight/分发/收件约 0.5；A、B
独立首轮各约 0.75–1.0；锁定后的共同讨论约 0.75–1.0；一次修订、freeze receipt 与 ACL 撤销约
0.5。若校准难例产生大量 field-level unknown，按上界执行；不得以赶 timebox 为由猜值。

## 8. 一句话状态

三段只用于在非盲讨论中暴露协议缺口；选择理由是 coordinator-only proposal，六张 CSV 保持
空白，不生成 GT，D1 必须在协议/实现/round-trip 全部冻结后用隔离且看不到 proposal 的 A/B
另行开始。
