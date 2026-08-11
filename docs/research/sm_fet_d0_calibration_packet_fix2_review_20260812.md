# SM-FET D0 calibration packet fix2 最终独立复审

- 日期：2026-08-12
- 被审 commit：`c5e63590ec5535068db937ee46d0c064d25e6943`
- 上一版：`9dba875af67e05d4e72a1011cb8a1ae63c22d09e`
- canonical：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 复审范围：17-file recipient contract、handbook/ontology、file role/mode、submission lifecycle、content lock、no-event/FK、A/B/proposal/formal 隔离
- 总结论：**No-Go（source、assignment 与当前 artifact 的 end-to-end submission）**；fix2 的 **submission lifecycle/FK/content-lock 工程机制本身 Go**，但被 source ontology P0 继承阻断；真人分发、标注与提交均尚未发生

## 1. P0：冻结 task ontology 与同包 handbook/canonical protocol 冲突

`frozen_inputs/TASK_ONTOLOGY.json` 的 `entity_roles` 是：

1. `MANIPULATED_OR_SOURCE`
2. `RECEPTACLE_OR_DESTINATION`
3. `TOOL_OR_MEDIATOR`
4. `SUPPORT_OR_CONTEXT`

而与其一同冻结、逐字节交付的 `ANNOTATION_HANDBOOK.md` 明确要求固定顺序：

1. `manipulated_or_source`
2. `tool`
3. `receptacle_or_destination`
4. `fixture_or_support`

canonical annotation protocol 和 minimal-r1d schema/freeze contract 也固定为 `MANIPULATED_OR_SOURCE / TOOL / RECEPTACLE_OR_DESTINATION / FIXTURE_OR_SUPPORT`，ordinal 为 1/2/3/4。ontology 因此同时存在两类不兼容：第二、第三 role 顺序颠倒；`TOOL_OR_MEDIATOR`、`SUPPORT_OR_CONTEXT` 不是 canonical role token，且包内没有 alias/mapping 合同。

这是 source contract 内部矛盾，不是格式偏好。标注员若服从 handbook，ontology 不再是其合法词表；若服从 ontology，则 node role/token/ordinal 与 r1d contract 不兼容。checker 当前只核 `TASK_ONTOLOGY.json` 的冻结 SHA/byte equality，没有核 role 的 exact tokens/order，所以 `--phase source` 对此错误返回 valid。

**Gate：** 在任何真人分发前，冻结 ontology 必须与 handbook/canonical protocol 的 exact role vocabulary 和顺序一致，checker 还应对该语义合同做 fail-closed 校验，而不只验 hash。

## 2. fix2 已关闭的工程项

| 项目 | 结果 | 独立证据 |
|---|---|---|
| 17-file exact allowlist | PASS | A/B root 各恰好 17 文件；新增 handbook、ontology、recipient schema manifest、submission receipt；README/coordinator/selection audit 在 denylist，extra file 被 checker 拒绝 |
| handbook/ontology bytes | PASS（仅字节冻结） | source 与 A/B 三份逐字节相同；handbook SHA-256 `27e70a8b4ff69d4aae369dfb11e229e8daccda750befcdfc964182f4e70436c4`，ontology SHA-256 `546acd3d5a3cc3be529fc7c20b458065502a198126d62cf49d71b27ae1518f1a`；语义 P0 见第 1 节 |
| file roles / modes | PASS | assignment root `0700`；immutable 八文件 `0400`，六张 annotation CSV 与 identity/session/submission receipt `0600`；submission 要求全部 17 文件 `0400`。symlink、hardlink 与共享 inode 检查闭合。Git tree 只保存 `100644`，checkout/复制后仍必须恢复 ACL 并重跑 checker |
| lifecycle / terminal state | PASS | submission 强制 identity/session/receipt=`submitted`、`annotation_started=true`、opened/closed/submitted 时间闭合、`write_locked=true`、`resigned_by_role=coordinator`，并要求每个实际 row `review_status=submitted` |
| content lock / re-sign consistency | PASS | submission receipt 绑定 identity、session 与六表 SHA；content-lock 覆盖 receipt 的其余字段。修改 annotation/identity/session、不重签或只改 lock 均被拒绝 |
| no-event coverage | PASS | 每 episode 恰一条 full-episode `EXHAUSTIVE` coverage，允许该 episode 其余五表为零行；独立三 episode no-event terminal submission 返回 valid、`rows=3` |
| same-episode FK | PASS | event→coverage、segment→coverage/event、node/pose→event、edge→event/segment 均检查同 episode closure；built-in 与独立 dangling-reference 负例被拒绝 |
| A/B 隔离 | PASS（仓库快照） | A/B packet/annotator/session ID 不同、root 不同、无共享 inode；冻结中性文件相同；同 root 作为 peer 被拒绝 |
| proposal 隔离 | PASS | episode-specific selection/proposal 仅在 coordinator 区；recipient 17-file roots 不含 README、selection audit、截图、媒体、HDF5、proposal 或 GT，文本扫描没有 episode-specific cue/绝对路径泄漏 |
| formal 隔离 | PASS | fix1→fix2 对 formal subtree 零 diff；formal assignment checker 仍为 valid：12 tasks、3695 timeline rows、A/B annotation files independent |

## 3. lifecycle 攻击复现与 CPU 正负例

测试使用 stdlib/CPU，在审计 worktree 内创建隔离 packet 副本；完成后已删除全部副本，没有创建或遗留测试脚本，也未改动被审 worktree。

| 测试 | 结果 |
|---|---|
| `--phase source` | checker PASS：3 tasks、2 recipients、6 templates；但未发现第 1 节 ontology 语义冲突，因此总体 source verdict 仍 No-Go |
| A/B `--phase assignment` | PASS：17-file roots、blank annotations、identity/session/receipt、hash、ACL、inode 闭合 |
| `--phase self-test` | PASS：正引用/no-event 正例；assigned-not-started、annotation-started=false、draft row、POSITIVE_ONLY coverage、dangling reference 均正确拒绝 |
| blank assignment 直接 submission | PASS（正确拒绝） |
| 三 episode full-coverage、零 event/contact/node/edge/pose 的 terminal submission | PASS（正确接受，`rows=3`、`references_closed=true`） |
| 改 coverage、不更新 receipt/lock | PASS（正确拒绝：annotation hash mismatch） |
| 改 coverage，并重算 table hash/lock，但 row 为 `draft` | PASS（正确拒绝：review status） |
| 把 identity/session 改回 assigned-not-started，并重算相应 hash/lock | PASS（正确拒绝：terminal lifecycle） |
| submission immutable file 保持 `0600` | PASS（正确拒绝：exact mode） |
| lock 改为错误值 | PASS（正确拒绝：content-lock mismatch） |

fix1 的核心 lifecycle 攻击——非空/draft annotations 与 `assigned_not_started`、`annotation_started=false` 并存却被判 valid——在 fix2 已不能复现。

## 4. content-lock 边界

`content_lock_sha256` 是有效的一致性/完整性闭包，但不是密码学签名：持有 root 写权限者可以重算各 SHA 与 lock；`resigned_by_role="coordinator"` 也是 receipt 中的自述字符串。checker 能证明“这 17 个文件彼此一致并处于声明终态”，不能独立证明实际重签者身份或真实 coordinator 授权。真实分发流程仍需用访问控制与外部审计记录建立身份来源；本项记为边界/P1，不否定本轮 lifecycle 机制 Go。

## 5. 分层裁定

- **Source engineering：No-Go。** 17-file/source/hash 结构通过，但冻结 ontology 的 role tokens/order 与同包 handbook 及 canonical contract 冲突，且 checker fail-open。
- **Assignment engineering：No-Go（artifact overall）。** 17-file、ACL、blank state、A/B isolation 等 assignment 机制通过，但两个 recipient 都收到相同的无效 ontology，不能分发给真人。
- **Submission lifecycle/FK engineering：Go（机制隔离裁定）。** fix1 状态机攻击已关闭；terminal state、row status、content binding、write lock、no-event 与 FK 正负例均符合预期。
- **Submission engineering：No-Go（当前 artifact overall）。** submission 机制的 Go 不能洗掉 source contract P0；必须先修正并重新冻结/交付 ontology。
- **真实 distribution、真人 independent annotation、提交/lock、A/B 比较与共同讨论：均尚未发生 / NOT RUN。** 仓库中的 A/B fixture 不构成真人分发或真人标注证据。

最终：**fix2 不应进入真人分发。** 清除 ontology P0、加入 exact ontology 语义 gate，并在重新冻结后完整重跑 source→assignment→submission 正负例，才可重新申请 end-to-end engineering Go。
