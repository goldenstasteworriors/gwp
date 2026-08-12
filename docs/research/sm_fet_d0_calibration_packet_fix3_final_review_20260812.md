# SM-FET D0 calibration packet fix3 最终独立复审

- 日期：2026-08-12
- 被审 commit：`b082680a74b1f0720c8b5e1023feeb8b39fd6cb9`
- 上一版：`c5e63590ec5535068db937ee46d0c064d25e6943`
- canonical baseline：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 聚焦范围：ontology、source/assignment/submission overall、fix1/fix2 旧攻击
- 最终结论：**Engineering Go**；真人分发、真人独立标注与真人提交尚未发生，**D0 仍未通过**

## 1. fix2 ontology P0 已关闭

fix3 的冻结 ontology 与 canonical minimal-r1d human shared semantic contract、annotation protocol 和同包 handbook 一致：

| ordinal | node slot | focus code | canonical ID | handbook token |
|---:|---:|---:|---|---|
| 1 | 0 | 1 | `MANIPULATED_OR_SOURCE` | `manipulated_or_source` |
| 2 | 1 | 2 | `TOOL` | `tool` |
| 3 | 2 | 3 | `RECEPTACLE_OR_DESTINATION` | `receptacle_or_destination` |
| 4 | 3 | 4 | `FIXTURE_OR_SUPPORT` | `fixture_or_support` |

- `human_shared_semantic_schema_id` 精确为 `sm_fet_r1d_shared_semantic_key_v1`，task ontology ID 精确为 `biochem4.v1`，task family 为 `POUR`。
- 四条 `meaning_zh` 与 handbook 第 6.1 节逐字一致；旧 token `TOOL_OR_MEDIATOR`、`SUPPORT_OR_CONTEXT` 不再出现在 ontology role 定义中。
- schema manifest 将 events focus、node role/slot、edge primary/secondary role、segment object role 明确绑定到 canonical enum/slot mapping；checker 同时核 exact headers 与 submission row values。
- source、A、B 的 ontology 逐字节相同，均为 2321 bytes、SHA-256 `ccb600e7ccbe2762ab6903df0c206026306cd9a8b0fdeba12d6999a01ef7a372`；三份 schema manifest 也逐字节相同，均为 8712 bytes、SHA-256 `0f5d757edcb912b6c9e12f271e314c760e5e4c148c7a49b56131e5a22383dd31`。

checker 不再只匹配冻结 SHA：source 和每个 recipient root 都解析 ontology、handbook、schema manifest，并校验 exact key set、schema/ontology IDs、role enum/order、ordinal/slot/focus、ID/token/meaning、worksheet mapping 与 headers。

## 2. 独立语义正负例

| 测试 | 结果 |
|---|---|
| 当前 ontology + handbook + schema role contract | PASS |
| fix2 commit 中的完整旧 ontology | PASS（正确拒绝：exact ontology schema/key set 不符） |
| 分别突变 role 顺序、ID、meaning、ordinal、node slot、focus code、worksheet mapping | PASS（全部正确拒绝） |
| canonical worksheet role values | PASS |
| event ontology ID / focus role 使用错误值 | PASS（正确拒绝） |
| node slot 非法或 slot-role 交换 | PASS（正确拒绝） |
| segment object role 使用旧 token | PASS（正确拒绝） |
| edge primary/secondary role 使用旧 token | PASS（正确拒绝） |
| built-in `legacy_role_value`、`node_slot_role_mismatch`、legacy ontology self-tests | PASS（全部正确拒绝） |

## 3. source、assignment、submission 与旧攻击

| Gate | 结果 | 独立证据 |
|---|---|---|
| Source engineering | **Go** | `--phase source` valid：3 tasks、2 recipients、6 exact blank templates；ontology/handbook/schema 语义合同和冻结 bytes 闭合 |
| Assignment engineering | **Go** | A→B、B→A 均 valid；六表 blank，17-file/hash/identity/session/ACL/A-B isolation 继续闭合；两个 recipient 得到同一 canonical ontology/schema bytes |
| Submission engineering | **Go** | 独立构造三 episode full-episode `EXHAUSTIVE`、零 event/contact/node/edge/pose 的 terminal no-event submission，checker 返回 valid、`rows=3`、`references_closed=true` |
| fix1 assigned-not-started lifecycle 攻击 | PASS（正确拒绝） | 非空 annotations 与 `assigned_not_started`/`annotation_started=false` 状态即使重算 receipt hashes/content lock，仍被 submission gate 拒绝 |
| fix1 draft-row 攻击 | PASS（正确拒绝） | terminal receipt 下把 coverage row 改为 `draft` 并重算 annotation hash/content lock，仍被 exact submitted-row gate 拒绝 |
| fix2 lifecycle/FK/content-lock | **保持闭合** | built-in self-test 的 no-event/reference 正例通过；assigned-not-started、annotation-started=false、draft、POSITIVE_ONLY、dangling FK、legacy role 与 slot-role mismatch 负例全部拒绝 |

fix2→fix3 的 formal subtree 零 diff；本轮没有发现新的 P0/P1。测试只使用 CPU/stdlib 和审计 worktree 内的临时副本；临时脚本、副本及导入产生的 cache 均已删除，被审 worktree 与审计 worktree 在写 memo 前均洁净。

## 4. 最终裁定与事实边界

- **Source engineering：Go。** canonical ontology 的角色、ID、meaning、slot、focus 与 header/value gates 已完整冻结并 fail-closed。
- **Assignment engineering：Go。** A/B blank recipient packet 可进入真实分发准备流程。
- **Submission engineering：Go。** terminal lifecycle、no-event、FK、canonical worksheet values 与 content binding 的正负例均通过。
- **End-to-end packet engineering：Go。** 本裁定只表示仓库 artifact/checker 已具备工程准备度，不表示任何真人流程已经执行。
- **真实 distribution、真人 independent annotation、真人 submission/lock、A/B comparison/discussion：尚未发生 / NOT RUN。** 仓库 fixture 不能冒充真人证据。
- **D0：未通过。** D0 只能在真实 A/B 分发、独立标注、锁定提交和规定的后续 gate 完成后裁定；本轮 engineering Go 不得写成 D0 pass。

最终：fix3 可从 packet engineering 角度进入下一步受控真人分发，但截至本 memo，真人未分发、未标注、未提交，D0 未通过。
