# SM-FET D0 calibration metadata packet v1

- 日期：2026-08-12
- 构建基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 状态：`D0_CALIBRATION_ONLY / NON_BLIND / NON_GT / METADATA_ONLY`
- 数据集：Ego `7_4_pour_liquid_between_beakers`
- 任务数：3
- formal blind12 交集：0

本目录只冻结 D0 非盲校准所需的小型 metadata 与空白 worksheet。它不包含 MP4、HDF5、截图、
缩略图、timeline proposal、事件 proposal、旧标签、裁决或 GT，也不授权训练、模型实现或
正式 blind 标注。四张 CSV 只有 header；任何 event、node、edge、pose 都必须由真实 A/B 在
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

## 3. 三段选择（coordinator-only proposal audit）

formal blind12 固定为：

`0, 9, 12, 34, 36, 38, 43, 50, 51, 53, 54, 58`

本次冻结为：

| task | episode | coordinator-only 选择理由（proposal，不是标签） |
|---|---:|---|
| `sm-fet-d0-cal-00` | 46 | 单手、时序较短、彩色内容对比度较高，作为普通可见/较清楚边界候选；未见明显 regrasp 仅是采样审计印象，须由 A/B 自行判断 |
| `sm-fet-d0-cal-01` | 65 | 原始 RGB 采样中可见两只手同时进入操作区，一手操作、一手靠近/稳定透明容器，手—杯—rim 互遮挡，覆盖双手 bundle、hand roster 与 occluded/missing 字段 |
| `sm-fet-d0-cal-02` | 1 | 透明杯与低对比度/近无色液体使容器边缘、stream、meniscus、effect/outcome 的可见证据弱；与 46/65 的紫色高对比内容形成 visibility 差异 |

选择只依据 coordinator 对原始 RGB 的只读时间采样、MP4 技术 metadata、formal blind episode
清单和现有 proposal-level 规范审计；未读取或使用 `hand_status`、depth、contact visualization、
模型输出、旧 GT 或人工标签。选择理由本身是 proposal，不能预填进四张 CSV，也不能在 D1
blind 开始前作为 hint 分发。

### Selection bias 记录

- 这是 purposive calibration sampling，不是随机样本；三段被主动挑来放大清楚边界、双手
  遮挡/roster 与透明液体可见性差异。
- 三段都来自同一 human-pour 数据源，不能估计自然 prevalence、OOD、跨任务、跨人、跨 session、
  跨容器或跨液体泛化。
- 当前 66 段缺 authoritative subject/session/operator/object/liquid/batch provenance；episode ID
  或颜色差异不能替代这些 group 字段。
- coordinator 已看过 selection-level proposal cues，因此 coordinator 不能充当未来 D1 blind A/B；
  D1 A/B 不得收到本 README、选择审计、D0 讨论纪要或任一 proposal。
- calibration 不进入 blind agreement、validation/test、阈值调参或 GT；若协议重开，必须另选与
  本三段及 blind12 都不重叠的新 calibration。

## 4. 文件与分发面

Git 中的 coordinator source bundle 是：

- `README.md`：本 runbook 与 coordinator-only selection audit；不得发给未来 D1 A/B；
- `task_manifest.csv`：三段的中性任务绑定，无 selection stratum、event hint 或 semantic proposal；
- `media_allowlist.csv`：仅中性媒体 ID、文件名、bytes、SHA-256、帧数、时长、fps、尺寸和 codec；
- `templates/events_v1.csv`；
- `templates/event_node_snapshots_v1.csv`；
- `templates/event_edges_v1.csv`；
- `templates/pose_keys_2d_v1.csv`；
- `templates/identity.json`；
- `templates/session.json`。

D0 分发时协调员分别建立 A、B 的 packet root 与 media root。annotator packet root 只允许平铺
复制 `task_manifest.csv`、`media_allowlist.csv`、四张空白 CSV、已赋值的 `identity.json` 和
`session.json`；指南/ontology 必须是双方相同、已 hash 的独立只读输入。media root 只允许
`media_allowlist.csv` 指定的三个 MP4，且与 packet root、state root 两两不嵌套。

不得把本 README 的第 3 节、绝对 source 路径、整个仓库、source root、processed HDF5、formal
coordinator/adjudication、另一方 packet 或 proposal audit 分发给 A/B。D0 是 non-blind，含义是
两人首轮锁定后允许按流程共同讨论定义，不代表首轮 CSV 可以被 proposal 预填。

ACL 最低要求：逐人 owner-only 目录；媒体与 immutable metadata 只读；四张 CSV 仅 owner 可写；
禁止 symlink/hard link；禁止云同步、截图、录屏和转发；记录授权人、recipient、开放/撤销时间、
各文件 hash 和销毁确认。没有数据控制者授权/伦理或用途依据时 fail closed，不分发媒体。

## 5. 执行与 protocol-freeze 顺序

1. **Coordinator preflight**：重算三个 MP4 SHA-256/帧数/时长/fps；验证与 blind12 零交集；
   验证四张 CSV 只有 exact header、身份/session 未赋值、packet 内没有媒体/截图/HDF5。
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

## 6. Fail-closed checker 设计（本轮未实现）

未来 checker 应按 phase 运行，任一未知、缺文件、多文件、hash 不符或语义不唯一都非零退出；
禁止 warning-only 与 `--force` 绕过。

### `source-preflight`

- exact allowlist：本目录只能出现本文列出的 9 个小文件；拒绝 MP4/HDF5/图片、额外 CSV、
  symlink、hard link、socket/device、重复 inode、NaN/Inf、JSON duplicate key；
- canonical/base、packet version、三任务数、episode 唯一性与 blind12 disjoint 均精确匹配；
- media allowlist 三行一一对应 task，重算 bytes/SHA-256/decoded frames/duration/fps/width/height/
  codec；实际媒体只能位于 coordinator 指定的独立 media root；
- 四张 CSV 必须只有 exact UTF-8 header 和 LF，零 data row；identity/session 必须保持 unassigned；
- 扫描 header 与所有文本，拒绝绝对 source/HDF5 路径、proposal value、GT/adjudicated value、
  timing hint 和受禁证据引用。

### `assignment`

- A/B packet exact allowlist、所有 immutable file hash 与 assignment receipt 一致；
- A/B 的 annotator/session/packet root/state root 不同，task/media bytes 相同；路径不嵌套、无共享
  inode、权限为 owner-only，媒体只读；
- 首轮前四张 CSV 仍为空；身份、session、role、protocol/header/media/task hashes 完整绑定；
- recipient denylist 中不含 README 第 3 节、selection audit、D0 讨论、另一方内容、formal
  coordinator/adjudication、HDF5/depth/hand/contact proposal。

### `submission`

- 每行 episode/task/annotator/session/role 与 receipt 一致，raw frame 落在 `[0,video_frames)`；
- event ID/顺序/半开 support、boundary exact/interval/censor/unknown、review state、合法 transition、
  verify-only outcome 与 `Q_episode` overflow 均可机械验证；
- 每个实际 snapshot 对同一 event/role/frame 恰有四个 fixed-role rows；reference union、ABSENT/
  UNKNOWN/MISSING、observability/geometry 约束完整；
- 每 event edge order 只可 0/1；actor/primary/secondary/relation/contact-mode 引用闭合，第三条 edge
  必须显式 overflow，不能截断；
- pose 只作 auxiliary/probe；不可见 landmark 坐标必须空，可见 landmark 坐标须有限且在 `[0,1]`；
  `entry/turn/exit/anchor` 与 event/frame/hand side 引用一致；
- `label_origin` 只接受 human independent calibration；拒绝 proposal、model、adjudicated、GT 和
  另一 annotator parent ID；拒绝任何被预填、跨副本相同的非空 annotation row/hash。

### `discussion-freeze`

- A/B 全任务提交 hash 与 write-lock receipt 齐全后才允许比较；只输出 disagreement/change-log，
  不输出 merged label 或 GT；
- dictionary/task ontology 最多一次 revision；freeze receipt 绑定所有规范/header/checker hashes；
- D0 media ACL 已撤销、working copy 已销毁，calibration output 标记 `NON_GT/EXCLUDED_FROM_D1`；
- D1 preflight 对每个未来 blind A/B root 执行 proposal/selection/calibration denylist，且正式
  contact packet 仍严格通过原有 7-file checker。

checker 还必须实现 annotation→当前冻结 wire→annotation 的唯一 round-trip、reachable capacity/
codebook/public-state receipt；在 index 所列 minimal-r1d CPU conformance 未通过前，任何 worksheet
都不能宣称 D0 pass 或 action admission。

## 7. 人时预算

本三段 D0 session 预计共 **3.25–4.0 人时**：coordinator preflight/分发/收件约 0.5；A、B
独立首轮各约 0.75–1.0；锁定后的共同讨论约 0.75–1.0；一次修订、freeze receipt 与 ACL 撤销约
0.5。若 episode 65 的双手 roster/遮挡或 episode 1 的透明液体 evidence 产生大量 field-level
unknown，按上界执行；不得以赶 timebox 为由猜值。

## 8. 一句话状态

三段只用于在非盲讨论中暴露协议缺口；选择理由是 coordinator-only proposal，四张 CSV 保持
空白，不生成 GT，D1 必须在协议/实现/round-trip 全部冻结后用隔离且看不到 proposal 的 A/B
另行开始。
