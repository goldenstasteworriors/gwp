# SM-FET D0 calibration packet 独立审计

- 日期：2026-08-12
- 审计基线：canonical `e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 被审 commit：`3082f05d30c4a685a9303c95ae0d62fc251be2f2`
- 被审目录：`track_a/sm_fet_d0_calibration_packet_v1/`
- 审计方式：只读检查 Git tree、formal blind12、现有 protocol/handbook、Ego 原始 MP4 与本地文件系统状态；未修改被审 worktree，未读取 HDF5 内容，未生成截图或媒体副本
- 总结论：**No-Go（不得分发给 A/B、不得开始 D0 标注）**；仅可作为明确标记为 `NON_BLIND / NON_GT / METADATA_ONLY` 的 coordinator-side 草案留存

## 1. 结论摘要

被审 commit 的供应链边界是干净的：其唯一父提交正是指定 canonical；只新增一个 sibling 目录中的 9 个普通文本文件；未触碰 formal 目录；三段 calibration 与 blind12 零交集；三段原始 MP4 的 bytes、SHA-256、完整解码帧数、fps、分辨率和按帧数计算的时长均与 allowlist 一致。四张 worksheet 只有 UTF-8/LF header，identity/session 保持空值与 `unassigned`，没有媒体、HDF5、图片、截图、标签行、GT、绝对路径、symlink、hardlink 或额外文件。

但它尚不是可执行、可分发的 D0 A/B packet，存在两个 P0 gate：

1. protocol 与 handbook 都要求 A/B 先标 coverage/contact，再标 event/node/edge/pose；protocol 还明确要求未来 sibling packet 原样保留 `coverage.csv` 和 `segments.csv`。当前 README 的 A/B flat-copy allowlist 没有这两张表，然而 `events_v1.csv.coverage_id` 与 `event_edges_v1.csv.contact_segment_id` 又依赖它们，形成不可闭合引用。
2. 当前只有 coordinator source bundle，没有实际 A/B packet/media/state roots、assignment receipt、已赋值 identity/session、冻结 protocol/task/media hash 或可运行的 SM-FET checker。README 自己也把 checker 写为“未来实现”。因此 owner-only ACL、A/B 隔离、README/proposal denylist、identity/session lock 与 fail-closed 只能是文字要求，不能被本 commit 机械证明。

所以本审计不否定该 commit 作为“不可执行草案”的自我描述；No-Go 针对的是任何 assignment、分发、标注启动或 D0 pass 声明。

## 2. 审计依据与复现边界

### 2.1 规范输入

| 输入 | 固定版本 | SHA-256 / 状态 |
|---|---|---|
| canonical | `e0ebc6929ab25bd5793ef64939b4b831e15b12c2` | 被审 commit 的唯一父提交与其完全一致 |
| annotation/pairing protocol | source commit `288ef5dcf4e6299871dc50d33caa603e46345fcd` | `c4f1f811f796e7c8604a0d0510229677693d6d1cc501b78f59f01cd4793e4a92` |
| D0 handbook | commit `00817c58d85a322c67fafb8ee1ee52e6f89a4616` | `27e70a8b4ff69d4aae369dfb11e229e8daccda750befcdfc964182f4e70436c4` |
| formal contact v1.1 packet/checker | 被审 commit 中现有版本 | `--phase assignment` 返回 `status=valid` |
| Ego source MP4 | episode 46、65、1 | 逐字节 SHA-256 + OpenCV 4.11.0 全帧解码 |

protocol 是“设计稿/最小字段表”，handbook 明确声明自己“不是 CSV schema”，两者都没有冻结四张表的 exact ordered-header hash。这一点影响第 5 节的“exact header”判定。

### 2.2 工具限制

系统没有 `ffprobe`，审计没有安装软件或改变任何环境。视频核验使用已有 OpenCV 4.11.0：读取容器报告属性并将每帧完整解码到 EOF。codec 的独立观察值为 FourCC `FMP4`，与 allowlist 的通用 codec 名 `mpeg4` 语义一致；如果未来 gate 要求 ffprobe 的精确 `codec_name/time_base` receipt，仍须由正式 checker 再冻结。

## 3. P0 / P1 memo

### P0-1：A/B annotation 引用链不闭合

**事实：**

- protocol 第 4.1 节要求事件轮独立填写 coverage、contact segment、event、四 role node snapshots 与 edge；第 9 节要求新 sibling packet“原样保留 `coverage.csv` 和 `segments.csv`，新增”四张 SM-FET worksheet。
- handbook Pass 1 同样先要求 coverage/contact segment。
- 当前 README 第 4 节规定 A/B packet 只平铺复制 `task_manifest.csv`、`media_allowlist.csv`、四张 worksheet、`identity.json`、`session.json`，共 8 文件，没有 coverage/contact 表。
- worksheet 却含 `events_v1.csv.coverage_id` 与 `event_edges_v1.csv.contact_segment_id`；当前 bundle 没有合法来源表、schema/header 或 receipt 可闭合这些引用。
- README 又正确禁止把 D0 表混进 formal 7-file packet，因此不能通过借用或修改 formal packet 绕过缺口。

**影响：** 无法按现有 protocol/handbook 完成一个引用闭合的独立 D0 首轮，也无法 fail-closed 区分合法引用、空引用和伪造引用。启动标注会造成离线 side table、自由 ID 或误改 formal packet 三选一。

**Gate：** 在独立版本中冻结 D0 coverage/contact 表或一个同等严格、hash 绑定且不污染 formal packet 的引用机制，并由 checker 验证唯一引用闭合之前，No-Go。

### P0-2：没有可审计的 assignment、ACL 与 fail-closed 实现

**事实：**

- 只有一个 coordinator source bundle；未出现实际 A/B packet/media/state roots 或 assignment/access/destruction receipts。
- `identity.json` 和 `session.json` 正确保持空值、`unassigned`，但 `protocol_sha256`、`task_manifest_sha256`、`media_allowlist_sha256` 也尚未绑定。
- 没有本 packet 的 packager/checker；README 第 6 节明确把 checker 定义为未来工作。
- source checkout 当前目录权限为 `0775`、文件为 `0664`。这可作为 coordinator Git working copy，但不满足 handbook 对实际 annotator root 的 owner-only 要求。
- README 与选择理由和 recipient 文件同处一个 source 目录；预期隔离依赖协调员手工挑选 8 文件，而非工具生成与 denylist 验证。

**影响：** 当前无法证明 A/B 身份/session 唯一、路径不嵌套、媒体只读、无共享 inode、对方不可见、proposal 未误发、hash 锁定或撤权销毁。空身份本身是安全默认值，但没有执行器时只是静态占位，不等于 fail-closed assignment。

**Gate：** 生成两套实际隔离 roots 和 receipts，并用独立 checker 对 exact allowlist、权限/ACL、inode/link、identity/session、全部 immutable hash、proposal denylist 与空 worksheet 执行非零失败检查之前，No-Go。

### P1-1：四张 CSV 是 protocol minimum-field 的有序超集，不是可证明的 exact schema

按 protocol 第 3 节逐组拼接最小字段后比较：

| worksheet | protocol minimum | packet header | 缺字段 | 额外字段 | protocol 公共字段相对顺序 |
|---|---:|---:|---|---|---|
| `events_v1.csv` | 35 | 36 | 无 | `annotation_session_id` | 完全一致 |
| `event_node_snapshots_v1.csv` | 25 | 26 | 无 | `annotation_session_id` | 完全一致 |
| `event_edges_v1.csv` | 30 | 31 | 无 | `annotation_session_id` | 完全一致 |
| `pose_keys_2d_v1.csv` | 43 | 45 | 无 | `annotation_session_id,label_origin` | 完全一致 |

这些新增字段符合 A/B session/provenance 隔离意图，并非 action-visible 语义扩张；因此本审计不把它们判为字段语义错误。但现有 protocol 只给“最小字段表”，没有 exact header 字符串、版本号或 hash，handbook 又明确不是 schema。故“exact 符合 annotation protocol”目前不可证明，只能判为“无缺字段的有序超集”。在 D0 使用前必须把 exact header 与 schema version/hash 一起冻结。

### P1-2：proposal 目前没有进入预期 recipient payload，但隔离仍靠手工

- coordinator README 第 3 节包含 episode 46/65/1 的选择理由，包括单/双手、透明容器、遮挡、stream/meniscus、液体对比度与 regrasp 印象；这些内容被明确标记为 `coordinator-only proposal`，不是标签。
- 对预期下发的 8 文件逐文本扫描，没有发现选择理由、透明/遮挡描述、proposal、GT、旧标签、HDF5/depth、hand status、prescreen 或 timing hint；四张 worksheet 也没有 data row。
- 透明/遮挡只在 coordinator README 的 episode-specific selection audit 中出现；A/B 表只保留中性的 `observability` 等字段名。handbook 中的通用透明/遮挡判定规则不是 episode proposal。
- 但实际 A/B packet 尚不存在，且没有 packager/checker；因此只能证明“allowlisted source files 干净”，不能证明“已经发生的分发没有泄漏”。

建议将 coordinator selection audit 移到不可被 recipient packager遍历的独立 root，或至少让 packager 从写死的 8-file allowlist 生成 packet 并立即做内容 denylist/hash receipt。此项在真实分发前与 P0-2 一起升级为硬 gate。

## 4. Blind12 与 formal 隔离审计

### 4.1 episode 集合

formal coordinator `tasks.csv` 的固定顺序为：

```text
43, 50, 51, 58, 34, 38, 9, 54, 36, 53, 12, 0
```

排序集合为：

```text
0, 9, 12, 34, 36, 38, 43, 50, 51, 53, 54, 58
```

calibration 为 `46,65,1`，三者唯一，集合交集为空。formal source manifest 对 episode 46、65、1 的 `assignment_selected` 均为 `false`。结论：**PASS，三段确实不是 formal blind12。**

### 4.2 formal 7-file root

在被审 commit 上检查 `annotator_a/` 与 `annotator_b/`：

- 每个根目录恰有 7 个普通文件：`ANNOTATION_GUIDE.md`、`assignment_receipt.json`、`identity.json`、`tasks.csv`、`timeline_evidence_index.csv`、`coverage.csv`、`segments.csv`；
- 无 symlink、无 hardlink、无第八个文件；
- `python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py --phase assignment` 返回 `status=valid`、`tasks=12`、`timeline_rows=3695`、`annotation_files_independent=true`；
- `e0ebc692..3082f05d` 在 formal subtree 下没有任何 diff；新 D0 文件全部位于 sibling `track_a/sm_fet_d0_calibration_packet_v1/`。

结论：**PASS，commit 没有污染 formal 7-file 母版。** 仓库 checkout 中 formal roots 也是 `0775/0664`，但 formal README 已说明它们只是 coordinator 母版；这不能替代真实分发位置的 owner-only ACL 证明。

## 5. 媒体、manifest 与 frame clock

| task | episode | bytes | SHA-256 | decoded / reported frames | fps | duration | size | codec observation | 结论 |
|---|---:|---:|---|---:|---:|---:|---|---|---|
| `sm-fet-d0-cal-00` | 46 | 3,422,955 | `4e03396690a21bc81ce1a20904692cd5a477b312022a74fe13c62bff0902c692` | 324 / 324 | 30.0 | 10.800000 s | 640×480 | `FMP4` / MPEG-4 | PASS |
| `sm-fet-d0-cal-01` | 65 | 3,823,238 | `061f0b24aee2396bde201cb0fbd884c16cca14ed35236af2b45b25631488cf2a` | 384 / 384 | 30.0 | 12.800000 s | 640×480 | `FMP4` / MPEG-4 | PASS |
| `sm-fet-d0-cal-02` | 1 | 6,595,743 | `5582effa3a634b735a2441cc9087e3a52685192e80bf7d0c6225205885387fd0` | 599 / 599 | 30.0 | 19.966667 s | 640×480 | `FMP4` / MPEG-4 | PASS |

补充核对：

- SHA-256 直接对本地 Ego 原始 MP4 重算，不复用 CSV 值；
- 每段都解码到 EOF，decoded frame count 与容器报告 frame count 相同；
- `task_manifest.csv` 的 coverage 均为 `[0,video_frames)`，时钟为 `raw_video_frame_30fps`；
- episode 1 的时长为 `599/30=19.966666...`，allowlist 的 `19.966667` 是正确的六位小数舍入；
- allowlist 与现有 formal coordinator source manifest 对三段的 bytes/hash/frame/fps/size 也一致；
- 审计没有读取同名 HDF5、depth、contact visualization 或 sidecar 内容。

## 6. Packet tree、链接、权限与内容检查

### 6.1 Git 与文件系统

- 被审 commit 的唯一父提交为 canonical `e0ebc6929ab25bd5793ef64939b4b831e15b12c2`。
- commit 恰好新增 9 个 path：root 下 `README.md`、`media_allowlist.csv`、`task_manifest.csv`，以及 `templates/` 下四张 CSV 与两个 JSON。
- Git tree 中 9 个 blob 全为 mode `100644`；working tree 中均为普通文件、inode 各异、link count 全为 1。
- 无 symlink、hardlink、socket、device、FIFO、隐藏文件或额外文件。
- 无 MP4、HDF5/H5、NPZ、图片、缩略图或截图文件；所有文件均识别为 text/CSV/JSON。
- 9 个文件均无 BOM、NUL、CR/CRLF，且以单个 LF 结尾。

### 6.2 数据内容

- `task_manifest.csv`：1 header + 3 task rows；episode 唯一且与 media 一一对应。
- `media_allowlist.csv`：1 header + 3 metadata rows；没有绝对 source 路径。
- 四张 worksheet：各 1 header、0 data row、无重复列名。
- `identity.json`：`annotator_id=""`、`annotation_role=""`、`identity_status="unassigned"`。
- `session.json`：session/annotator/time/hash 均为空、`session_status="unassigned"`。
- 两个 JSON 解析成功，无 duplicate key；所有 packet 文本没有本地绝对路径或 `file://` 泄漏。
- 没有 human/model/adjudicated/GT annotation row。需要注意，README 本身确实含 coordinator proposal audit；因此“整个 source directory 完全无 proposal”是错误说法，正确边界是“预期 recipient 8-file payload 无 proposal”。

## 7. Gate matrix

| 检查项 | 结果 | 说明 |
|---|---|---|
| canonical parent | PASS | target 的唯一父提交精确为 `e0ebc692...` |
| episode 46/65/1 非 blind12 | PASS | 集合交集为空，formal manifest 均 `assignment_selected=false` |
| media bytes/SHA/frame/fps/duration/size | PASS | 三段逐字节与全帧解码复核一致 |
| codec | PASS with tool note | OpenCV FourCC `FMP4`，对应 allowlist `mpeg4`；无 ffprobe receipt |
| 无媒体/HDF5/截图/GT row | PASS | 仅 9 个文本文件；四表零数据行 |
| source bundle 无 proposal | FAIL by design | coordinator README 含选择 proposal；recipient allowlist 文件无 proposal |
| CSV protocol minimum fields | PASS | 无缺字段且公共字段顺序一致 |
| exact frozen header | NOT PROVABLE / P1 | protocol/handbook 未给 exact ordered-header hash；存在明确 provenance 扩展列 |
| identity/session 空值 | PASS | 均为空且 status=`unassigned` |
| identity/session fail-closed | FAIL / P0 | 无 assignment receipt、hash freeze 或可运行 SM-FET checker |
| A/B selection-rationale leakage | NOT OBSERVED, NOT CERTIFIABLE | 预期 8-file payload 干净，但真实 A/B roots 尚不存在、复制依赖手工 |
| episode-specific透明/遮挡描述 | PASS for source separation | 只在 coordinator README proposal；未进入 recipient files/rows |
| formal 7-file 隔离 | PASS | 两 root 各 7 文件，formal checker valid，commit 未修改 formal subtree |
| symlink/hardlink/extra file | PASS | packet 与 formal 母版均未发现 |
| absolute path leakage | PASS | 被审 9 文件内未发现 |
| owner-only ACL / root separation | FAIL / P0 for execution | 当前只是 `0775/0664` source checkout；没有实际 A/B packet/media/state roots |
| coverage/contact reference closure | FAIL / P0 | recipient allowlist 缺 `coverage.csv` 与 `segments.csv` |

## 8. 最终裁定

**No-Go：不得把 commit `3082f05d30c4a685a9303c95ae0d62fc251be2f2` 直接复制给 A/B，不得开始 D0 independent first pass，不得把它称为 fail-closed packet 或 D0 pass。**

允许的唯一状态是 coordinator-side、metadata-only、NON_GT 草案。重新申请 Go 时，至少要同时提交：引用闭合的 coverage/contact 契约、冻结的 exact headers/schema/hash、独立 packager/checker、两套真实 A/B roots 与 owner-only ACL/receipt、赋值且不同的 identity/session，以及能证明 README/selection proposal 未进入 recipient payload 的机器检查结果。所有这些都必须继续保持与 formal 7-file packet 完全隔离。
