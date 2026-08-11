# SM-FET D0 calibration packet fix1 独立复审

- 日期：2026-08-12
- 被审 commit：`9dba875af67e05d4e72a1011cb8a1ae63c22d09e`
- 上一版：`3082f05d30c4a685a9303c95ae0d62fc251be2f2`
- canonical：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 复审范围：上一轮 P0/P1、source/assignment/submission 状态、recipient/proposal/ACL/checker 与 formal 隔离
- 总结论：**No-Go（end-to-end packet engineering）**；**Go（source + blank assignment 结构准备）**；真人媒体分发与标注尚未发生，也未在本复审中模拟为已发生

## 1. 原问题复审

| 项目 | fix1 结果 | 证据 |
|---|---|---|
| `coverage.csv` / `segments.csv` 引用链 | **已关闭** | 两表已进入 templates 与 A/B 13-file roots；schema manifest 冻结 `event→coverage`、`segment→coverage/event`、`edge→segment` 等同 episode foreign key；self-test 正例通过、dangling segment 负例被拒绝 |
| exact CSV headers | **已关闭** | `schema_manifest.json` 冻结八张 CSV 的 exact ordered header、header SHA-256 与 blank/assignment SHA-256；checker 内置同一 header 常量和 schema-manifest hash；source phase 通过 |
| A/B roots / receipts / ACL | **assignment snapshot 已关闭，真实分发未发生** | 两套 root 各恰有 13 文件、root `0700`、文件 `0600`、不同 packet/annotator/session、无共享 inode；receipt、recipient manifest、coordinator index 的 hash 闭合；assignment phase 通过 |
| checker | **source/assignment/reference gate 已实现；submission gate 未关闭** | stdlib checker 的 source、assignment、self-test 均通过；见第 2 节 P0 |
| coordinator proposal 隔离 | **已关闭** | episode-specific 理由只在 `coordinator/selection_proposal_audit.md`；A/B roots 无 README/coordinator/selection audit，文本扫描无相应透明、遮挡、双手、颜色或 regrasp cue |
| recipient exact allowlist | **已关闭** | A/B 各自 `recipient_exact_allowlist.json` 精确列出 13 文件并 denylist README/coordinator/selection audit；额外文件、symlink、hardlink、绝对路径均被拒绝 |
| 空白/source template | **通过** | 六张 annotation CSV 均只有 header；source identity/session 全空且 `unassigned` |
| assignment 状态 | **通过** | A/B 六表仍空；identity 为 `assigned_by_coordinator`，session/receipt 为 `assigned_not_started`，A/B ID/session 不同 |
| submission 状态 | **失败 / P0** | checker 可接受 draft rows 与 `assigned_not_started` receipt/session 的矛盾组合 |
| formal 7-file packet | **未受影响** | fix1 对 formal subtree 零 diff；原 formal `--phase assignment` 仍返回 `status=valid`、12 tasks、3695 timeline rows、A/B annotation files independent |

## 2. P0：submission 状态机 fail-open

`phase_submission()` 允许 annotation CSV 变化，但仍调用要求以下 assignment-time 状态不变的检查：

- `session_status="assigned_not_started"`；
- `assignment_status="assigned_not_started"`（由 coordinator-frozen receipt hash 固定）；
- `annotation_started=false`；
- `opened_at_utc=""`、`closed_at_utc=""`；
- `session.json` 又被 recipient allowlist 列为 immutable。

同时，submission row 检查没有要求 `review_status=submitted`，也没有 submission receipt、提交 hash、关闭时间或 write-lock receipt。

独立 CPU 负例复现使用 A packet 的隔离副本，仅向 `coverage.csv` 写入三个 episode 的完整 coverage，另外五张 annotation 表保持空白：

1. 三行 `review_status=submitted` 时，checker 返回 `status=valid, phase=submission, rows=3, references_closed=true`，尽管 receipt/session 仍称从未开始；
2. 将三行改为 `review_status=draft` 后，checker仍返回相同的 valid 结果。

这不是单纯“尚未真人标注”的外部事实，而是 checker 会把内部矛盾状态判为有效 submission。修复前不得锁定提交、比较 A/B 或开放共同讨论。

**Gate：** submission 必须绑定独立 submission/lock receipt，要求所有实际 row 为获准终态，记录 opened/closed 或等价状态证明，并拒绝 `annotation_started=false` / `assigned_not_started` 与非空 submission 并存。若 assignment receipt 按设计永久不可变，应另有被 checker 强制要求的 current-session/submit receipt，不能靠口头解释。

## 3. P1：ACL 文档与 checker mode 不一致

README 要求“immutable metadata 只读，六张 annotation CSV 仅 owner 可写”，但 checker 对 packet root 下全部 13 文件统一要求 `0600`。实际 A/B roots 中 `task_manifest.csv`、`media_allowlist.csv`、identity/session、allowlist 与 receipts 也都是 owner-writable `0600`，并非只读。

hash gate 能在复查时发现篡改，但不能使“immutable metadata 只读”的 ACL 陈述成立。真实分发前应区分 immutable `0400` 与 mutable `0600`，或把文档改成准确的“owner-writable + hash fail-closed”并说明威胁模型。

另一个待真实分发关闭的 P1 是指南/ontology：receipt 已绑定 protocol 与 handbook hash，但 distribution CLI 不接收实际外部指南/ontology path，receipt 也没有 task-ontology hash。当前只能证明 packet 内 metadata 一致，不能证明 A/B 实际拿到同一份已冻结外部输入。

## 4. CPU 正负例

| 测试 | 结果 |
|---|---|
| `--phase source` | PASS：3 tasks、2 recipients、6 blank templates |
| A/B `--phase assignment` | PASS：13-file roots、blank files、identity/session/hash/ACL/inode 闭合 |
| built-in `--phase self-test` 正引用 | PASS |
| built-in dangling `contact_segment_id` | PASS（正确拒绝） |
| blank A root 直接跑 submission | PASS（正确拒绝：缺每 episode 唯一 full coverage） |
| A root 同时作为 packet 与 peer | PASS（正确拒绝：roots 相同/嵌套） |
| 三条 submitted coverage + assignment-not-started 状态 | **FAIL（错误接受）** |
| 三条 draft coverage + assignment-not-started 状态 | **FAIL（错误接受）** |
| formal contact v1.1 assignment checker | PASS |

测试仅使用 CPU/stdlib 与一个审计 worktree 内的临时 packet 副本；副本已在测试后删除，没有创建或遗留测试脚本，也未修改被审 worktree。

## 5. 分层裁定

- **Source engineering：Go。** coverage/contact、exact headers、schema/hash、coordinator proposal 分区、13-file recipient allowlist 和 formal sibling 边界均已补齐。
- **Blank assignment engineering：Go（仅当前 checkout 快照）。** 当前 A/B roots 通过 `0700/0600`、blank/hash/identity/session/inode 检查；Git 不保存完整 ACL，复制后必须重跑。
- **Submission engineering：No-Go。** P0 状态机可把 draft/non-started 组合判为 valid。
- **真实 distribution：尚未发生 / NOT EVALUATED。** 没有实际 A/B 用户隔离位置、外部 media/state roots、数据授权/access receipt、指南/ontology交付证明或 distribution pass；仓库里同属 `ykj` 的 roots 不能冒充真人隔离事实。
- **真人 annotation、A/B lock、比较、讨论、GT、D0 pass：均未发生，仍为 No-Go。**

最终：fix1 已实质关闭上一轮的引用、header 与 proposal/recipient 结构问题，但在 submission 状态机修正并补齐真实分发证据前，不能给出 end-to-end packet engineering Go。
