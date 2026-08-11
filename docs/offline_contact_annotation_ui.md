# Contact v1.1 离线双标工作台

## 结论与边界

这个工具是 `formal_dual_annotation_v1_1` 的单人离线填写器，不是新协议，也不是
checker、裁决器或自动标注器。它只修改当前标注者 packet 中协议已有的
`coverage.csv` 和 `segments.csv`。协调员收回两个隔离 submission 后，仍须使用现有
`check_formal_contact_dual_annotation_v1_1.py --phase submission` 做正式验收。

隔离端不能自行运行正式 submission checker：现有 checker 会重算协调侧证据并检查 A/B
文件独立性，而这些材料按协议不能发给任一标注者。这是刻意的权限边界，
不是 UI 缺失；工具不会为了本地“通过”而复制或放宽这些材料。

应用由两个文件组成，均位于 packet 外：

- `tools/contact_annotation_ui/contact_annotation_ui.html`：单文件静态网页，无外部依赖，
  CSP 明确禁止网络连接；视频只通过本地 file handle 转成内存 `blob:` URL。
- `scripts/contact_labels/prepare_offline_contact_annotation_ui.py`：单 packet 的 fail-closed
  预检、状态初始化、CSV roundtrip、staged 输出语义检查和原子提交器。它只接受一个精确
  7 文件 packet、一个独立 12-MP4 media root、一个外部 state 目录和 UI 文件，不读取任何
  协调侧数据或其他标注目录。

## 协调员准备

先按原协议给 A/B 分配不同的真实 `annotator_id` 与 `annotation_session_id`，同步修改自己
`coverage.csv` 的 `annotator_id`，再分别复制到互不可见的位置。每个 packet 根目录必须
始终精确包含 7 个协议文件，不能有任何子目录。12 个 MP4 放在另一个独立 ACL media root
中，文件名与内容必须精确匹配 assignment receipt；packet、media root、state 三者不能
相同或互为父子，也不能使用 symlink 或 hard link。

预检采用 owner-only POSIX 权限作为机器可审计的最低基线：packet/state 根目录不可由
group/world 访问；5 个不可变文件和媒体只读；两份输出 CSV 仅 owner 可读写。若机构使用
ACL 或专用隔离组，应先在外层访问控制上完成等价隔离，再为实际标注账号制作下面这种
owner-only 工作副本。不要把 UI、日志、autosave 或 receipt 放进 packet。

下面命令中，每个反斜杠后都需要立即换行；请把四个示例路径替换成该标注员实际的绝对
路径。命令不会启动 server。

```bash
chmod 700 /absolute/path/to/annotator_packet
chmod 400 \
  /absolute/path/to/annotator_packet/ANNOTATION_GUIDE.md \
  /absolute/path/to/annotator_packet/assignment_receipt.json \
  /absolute/path/to/annotator_packet/identity.json \
  /absolute/path/to/annotator_packet/tasks.csv \
  /absolute/path/to/annotator_packet/timeline_evidence_index.csv
chmod 600 \
  /absolute/path/to/annotator_packet/coverage.csv \
  /absolute/path/to/annotator_packet/segments.csv
chmod 500 /absolute/path/to/annotator_media
chmod 400 /absolute/path/to/annotator_media/*.mp4
mkdir -m 700 /absolute/path/to/annotator_ui_state
```

先运行只读 dry-run。它检查精确 episode 顺序、完整 processed 时间轴、12 个视频的 SHA-256、
identity/session、immutable receipt、CSV header 与 parse/write/parse roundtrip、目录白名单、
symlink/hard-link 和 owner-only 权限。这里展示的命令需要按行换行：

```bash
cd /absolute/path/to/giga-world-policy
python scripts/contact_labels/prepare_offline_contact_annotation_ui.py \
  --packet-root /absolute/path/to/annotator_packet \
  --media-root /absolute/path/to/annotator_media \
  --state-dir /absolute/path/to/annotator_ui_state \
  --ui-file /absolute/path/to/giga-world-policy/tools/contact_annotation_ui/contact_annotation_ui.html \
  --dry-run
```

dry-run 通过后初始化 identity-locked session receipt。这里展示的命令需要按行换行：

```bash
python scripts/contact_labels/prepare_offline_contact_annotation_ui.py \
  --packet-root /absolute/path/to/annotator_packet \
  --media-root /absolute/path/to/annotator_media \
  --state-dir /absolute/path/to/annotator_ui_state \
  --ui-file /absolute/path/to/giga-world-policy/tools/contact_annotation_ui/contact_annotation_ui.html \
  --initialize
```

初始化 receipt 不记录任何绝对数据路径，只固定 packet ID、角色、标注员、session、
immutable 文件 hash、UI build/hash，以及 12 个 opaque task/media hash。若 state 目录已经绑定
别的身份或 session，初始化会拒绝覆盖。

## 标注员操作

用本机 Chromium、Chrome 或 Edge 直接打开 HTML 文件，不要使用 Web server，也不要使用
需要同步账号的持久浏览器 profile。页面依次选择外部 state、精确 7 文件 packet、独立
12-MP4 media root。浏览器端会再次校验三者互不相同、packet 根白名单、任务、身份、
immutable hash、完整时间轴和全部媒体 hash；校验完成前工作区不会出现。

界面支持：

- 原速、`0.5×`、`0.25×` 播放，processed 帧滑杆与逐帧/±10 帧移动；
- 以 `I` 记录第一接触帧、以 `O` 记录第一非接触帧，区间始终是
  `[start_frame,end_frame)`；
- `unknown`、`positive_only`、`exhaustive` coverage；
- observed/unknown/censored 边界，以及 side/site/object_role/contact_mode 的空值屏蔽；
- `Space` 播放、方向键逐帧、`Shift+方向键` ±10 帧、`N` 新区间、`I/O` 边界快捷键；
- 每次修改 debounce autosave 到外部 state 目录：`coverage.autosave.csv` 与
  `segments.autosave.csv` 保留精确协议 header 和当前 draft，配套 JSON 仅保存 UI 恢复状态；
  不在 packet 增加运行文件。

新 segment 只在标注员点击或按 `N` 后生成，分类、边界、置信度和证据都不预填。
observed event ID 由当前 packet 角色、episode、单调序号与 session tag 确定；未知或 censored
边界不会得到 event ID。类别不知道时保持空白，不会写字面量 unknown。UI 不提供 notes 输入，
从而避免把路径或额外材料写入 submission。

只有 12 个 episode 都满足协议局部约束时才可生成输出。浏览器只在 state 中生成
`segments.submission.csv`、`coverage.submission.csv` 和
`contact_annotation_ui_submission_receipt.json`，写后逐字节回读；不会直接写 packet。该
receipt 仅是文件交接记录，不声称正式 checker 已通过，也不能把独立标注升级为 GT。

随后运行原子提交。CLI 会重新验证 identity/session、staged CSV hash、12-episode 集、CSV
roundtrip 与所有协议局部语义；要求 state 与 packet 在同一文件系统，并用 `os.replace` 对
现有两个 CSV 分别做原子替换。替换前的两份文件只备份在外部 state；第二步失败会原子回滚
第一步。packet 不会出现命名临时文件，提交前后都保持精确 7 文件。以下命令中反斜杠后立即
换行：

```bash
cd /absolute/path/to/giga-world-policy
python scripts/contact_labels/prepare_offline_contact_annotation_ui.py \
  --packet-root /absolute/path/to/annotator_packet \
  --media-root /absolute/path/to/annotator_media \
  --state-dir /absolute/path/to/annotator_ui_state \
  --ui-file /absolute/path/to/giga-world-policy/tools/contact_annotation_ui/contact_annotation_ui.html \
  --commit-staged-output
```

## 收件与正式 checker

标注完成后先在隔离端检查 UI 产物。这里展示的命令需要按行换行：

```bash
cd /absolute/path/to/giga-world-policy
python scripts/contact_labels/prepare_offline_contact_annotation_ui.py \
  --packet-root /absolute/path/to/annotator_packet \
  --media-root /absolute/path/to/annotator_media \
  --state-dir /absolute/path/to/annotator_ui_state \
  --ui-file /absolute/path/to/giga-world-policy/tools/contact_annotation_ui/contact_annotation_ui.html \
  --validate-output
```

协调员分别收回 A/B 的 7 文件 packet，确认两者访问已撤销，再放回完整正式双标目录。以下
命令中反斜杠后立即换行：

```bash
cd /absolute/path/to/giga-world-policy
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py \
  --packet-root /absolute/path/to/formal_dual_annotation_v1_1 \
  --phase submission
```

只有该命令成功，才说明两个 submission 的文件、身份、session、CSV 语义和独立性机器门槛
通过。后续 agreement 与第三人 adjudication 完全沿用原协议。

## 隐私、ACL 与销毁

- 应用不使用 `fetch`、XHR、WebSocket、service worker、CDN 或遥测；CSP 的
  `connect-src 'none'` 会阻断网络连接。
- 浏览器只获得用户明确选择的 state、单人 packet 和独立 media root 三个目录 handle。
  packet 白名单外任何条目或 media root 中非 12-MP4 条目都会导致 fail-closed；媒体不会被
  复制到 git、state 或浏览器下载目录。
- state 只保存锁定 receipt、autosave CSV/恢复 JSON、staged CSV、submission receipt 和 precommit CSV
  备份；它不保存媒体、缩略图、截图或任何绝对数据路径。state、packet 与 media root
  两两不能相同、互为父子或相同 inode。
- 标注员完成并由协调员验收后，按原协议撤销 packet/media 访问并销毁媒体工作副本；state
  的留存期由数据控制者规定。关闭非持久浏览器会话并清理其站点权限记录。
- 发现额外文件、权限过宽、身份漂移、hash 不符或错误分发时立即停止，不要通过改 receipt、
  关闭检查或把额外内容移到隐藏目录来继续标注。
