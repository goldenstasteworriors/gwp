# SM-FET D0 三段可观测性独立复审

- 日期：2026-08-12
- 独立工作树：`codex/sm-fet-d0-observability-review-20260812`
- 基线：canonical `e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 被复审 memo：`sm_fet_d0_three_episode_observability_audit_20260812.md`，source commit `664dfc6c5baa10a709b2996178a7238f48247f20`
- 复审性质：只读、CPU-only、proposal review；不是 annotation、GT、adjudication、D0/D1 gate receipt 或训练结果
- 禁止边界：未读三段 HDF5/深度/hand 状态，未读 contact visualization/annotation，未改原 memo、packet、GT、代码或模型

## 0. 独立结论

原 memo **可以保留为单人 prescreen/proposal**，但不能被引用成 D0 结论。独立复审得到以下结论：

1. **媒体身份正确。** 原始 episode 43/50/58 MP4 的 SHA-256、字节数、编码、分辨率、恒定 30 fps、帧数与时长均复核一致；三个 hash 与 formal packet coordinator manifest 一致。
2. **粗视觉事实基本成立。** 三段都可见一只手拿起有紫色内容的 source，将其移到透明 destination 附近并倾斜，随后 source 变空、destination 出现紫色内容，再将 source 放回。43 为镜像布局，50 的关键杯口区遮挡最强，58 的首尾静止上下文最长；未见稳定、可定位的液流/contact interface。
3. **事件、side、focus、edge 与区间仍只是 proposal。** `RIGHT/LEFT`、六段候选事件、长 `RELEASE`、`destination` focus、两条 edge、`STICK`、`OPENING/RIM_EDGE`、`TRANSFER` 优先于 `ROTATE`，都需要 A/B 先独立回答，不能由本审计或原审计冻结。
4. **原 `3 V / 11 P / 4 U` 不是 D0 证据。** 它把“有容易的局部样例”“字段已可重复编码”“三段覆盖了该字段词表”混在同一 V/P/U 轴上。按 code-level 可复现性重读，我只保留 `center_cell` 为局部 V；把 `primary_node`、`secondary_node` 从 V 降为 P，得到 **1 V / 13 P / 4 U**。这仍然只是第二位审计者的 proposal，不是 agreement。
5. **`0/3 roster complete` 与 `0/3 oracle proven` 作为“本次审计没有证明”是合理的；作为“这些视频不可能完成”则不成立。** 另一只手未入画不能由单人非正式审计自动变成 `NOT_INVOLVED`；但未来 A/B 做完整逐帧 exhaustive review 后，仍可能一致判定无第二条 task-relevant hand edge。oracle 是否可编必须由正式 annotation 与 checker决定；有 UNKNOWN code 的字段不是天然 packet 拒绝理由。
6. **未发现语义标签泄漏的正证据，但不能从最终 commit 证明“从未看过”。** 被复审 commit 只新增一份 Markdown，没有媒体/截图/CSV；原视频没有 contact overlay，三段也没有同号 contact-visualization 文件。memo 未暴露 hand 状态或 contact annotation 值。不过它读取的 coordinator manifest/timeline 含 processed HDF5 路径/hash及 HDF5-derived 时间同步元数据，所以“完全没有使用 HDF5”应收窄为“没有直接读取 HDF5 数组、没有用其动作/手/contact状态作语义判断”。
7. **43/50/58 不能在现行协议中被称为 D0 calibration。** authoritative protocol 已把它们冻结在 12 段 blind set，并要求 calibration 从其余 54 段选且互斥。原 memo 的开头“保留 43/50/58 作为校准样本”与现行 assignment 冲突；最多称为 coordinator-only prescreen。若真要改作 calibration，必须先发布新 protocol/assignment version、替换 blind episode、重新冻结 hashes，并保证 A/B blind 从未见本 memo。
8. **D0 仍未通过。** A/B calibration 只能帮助冻结 dictionary/task ontology；正式 D0 是 checker/round-trip/grammar/capacity 的机械 gate。任何人对三段视频的 V/P/U 看法都不能替代 annotation→82-code→annotation round-trip 与 capacity receipt。

## 1. 复审输入与证据边界

### 1.1 文档输入

| 输入 | SHA-256 | 用途 |
|---|---|---|
| 被复审三段 audit memo | `b5b3d29bad30d2829a5897bf62734fedf2b3548b887e89d508a68c38b733e660` | 核对原主张、V/P/U、区间与 gate 解释 |
| D0 annotation handbook | `27e70a8b4ff69d4aae369dfb11e229e8daccda750befcdfc964182f4e70436c4` | 核对 A/B 隔离、证据 allowlist、roster/missingness、操作顺序 |
| annotation/pairing protocol | `c4f1f811f796e7c8604a0d0510229677693d6d1cc501b78f59f01cd4793e4a92` | authoritative 3+12、字段 gate、D0/D1 定义 |
| formal source manifest | `bcc3f46167f53d1a3e0caa2366f9d3b6173857db290376484073165b0e9b2ae2` | 只核媒体身份、processed coverage与 sync summary |
| coordinator timeline index | `ef163e1c7fa5cca702a803ba72b45b82c6e941a3762b1f08db4cd94a38ae7920` | 只核 processed frame 到原视频帧/秒映射 |

协议中与本复审直接相关的不变量是：

- calibration 与 12 blind 必须不重叠；固定 blind set 含 `43,50,51,58,34,38,9,54,36,53,12,0`；
- A/B 必须先分别独立做完 calibration，才能共同讨论；讨论结果不进 blind agreement、不升级 GT；
- `hand_status`、HDF5、contact/proposal、另一标注员输出均不得成为 A/B truth；
- D0 要求 checker 证明 UNKNOWN/ABSENT/MISSING/censor、grammar、focus/reference、overflow 与 82-code round-trip 唯一；
- 单类或稀疏字段必须 `NOT_EVALUABLE`，不能以 100% agreement 冒充通过。

### 1.2 视频查看方法

只读取以下原始 MP4：

```text
/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/episode_43.mp4
/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/episode_50.mp4
/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/episode_58.mp4
```

使用现有 `decoupled_vla_collection` 环境内的 `ffprobe/ffmpeg` 绝对路径做 CPU 解码与临时帧查看。临时 contact sheets/keyframes 只放在本工作树 `.d0_review_scratch/`，提交前全部删除；没有截图、媒体或衍生视觉资产进入 git。

仓库中的另一个 LeRobot 数据集还存在同号 `episode_000043/50/58.mp4`，但其 hash 分别为 `e15e.../5a2f.../05b4...`，不是 formal packet 的原始 MP4，也不能仅凭同号认作同一采集或再编码版本。本复审没有用这些异源媒体做语义判断。

## 2. 媒体与时间轴复核

### 2.1 Hash、帧率、帧数与时长

| episode | bytes | SHA-256 | ffprobe stream | format duration | manifest |
|---:|---:|---|---|---:|---|
| 43 | 4,081,679 | `36bee62146e5351a096e06bb489cdc941c70af92e2d58bf1be6eee0fb91b94e3` | MPEG-4, 640×480, yuv420p, `30/1`, 400 frames, 13.333333 s | 13.334 s | 一致 |
| 50 | 3,587,063 | `ace55b2d1a563141c4afc6765c5c528a31946d4554a6fed715efb280a898a618` | MPEG-4, 640×480, yuv420p, `30/1`, 338 frames, 11.266667 s | 11.267 s | 一致 |
| 58 | 4,387,613 | `92e9b10dfc7bb86b67021b44fdd026ea1f6fe4a3f06280c110580bacafa1180a` | MPEG-4, 640×480, yuv420p, `30/1`, 444 frames, 14.800000 s | 14.800 s | 一致 |

manifest 的 processed coverage `264/223/294` 帧与最大 sync error `20.562336/23.365848/25.731104 ms` 也与原 memo 一致，均小于 50 ms 上限。

### 2.2 原 memo 的 `v` 时间不是严格 raw-video 时间

原 memo 把多数括号内 `v` 时间直接写成 `processed_frame / 20`。timeline index 表明 processed 与 raw video 的相对起点/采样会逐渐产生约 1–3 个原视频帧的偏移；sync error 仍小，是因为 sync 统计使用 capture timestamps，而不是简单比较两个相对时间列。

| episode / proposal window | memo 写法 | timeline 对应 raw video time |
|---|---|---|
| 43 `MANIPULATE p72–136` | `v3.6–6.8s` | `3.633333–6.866667 s` |
| 43 `RELEASE p136–187` | `v6.8–9.35s` | `6.866667–9.433333 s` |
| 50 `MANIPULATE p68–120` | `v3.4–6.0s` | `3.433333–6.033333 s` |
| 50 `RELEASE p120–168` | `v6.0–8.4s` | `6.033333–8.466667 s` |
| 58 `MANIPULATE p88–136` | `v4.4–6.8s` | `4.400000–6.833333 s` |
| 58 `RELEASE p136–191` | `v6.8–9.55s` | `6.833333–9.600000 s` |

因此：

- `p[...]` 作为宽松 evidence window 的时序顺序基本正确，且都在 coverage 内；
- 括号中的 `v` 应称为 **processed-time alias**，不能声称是逐点查表后的 raw-video time；
- 这些区间有意重叠、也没有 start/end boundary support，不能变成 exact/interval annotation；
- 最大问题不是几十毫秒的显示误差，而是窗口本身由单一审计者按语义切出，尚无 A/B 独立复现。

### 2.3 逐段视觉复核

| episode | 独立可复核的强事实 | 仍不能冻结的内容 |
|---:|---|---|
| 43 | source 初始在画面右、destination 在左；一只手从右上进入并持杯；前后帧清楚显示 source 由紫色变空、destination 由透明变紫；倾倒时手/source 遮住部分 rims | anatomical `RIGHT` 虽有拇指拓扑支持，仍需 A/B；`PRECONTACT/ACQUIRE/HOLD` 分界、`p72` manipulate onset、`p136` 长 release onset、stream/contact cell不可定 |
| 50 | source 左、destination 右；可见手 side 候选为 `LEFT`；`p72–116` 对应的倾倒阶段确是三段最强遮挡；destination 中紫色内容在时序中出现 | 遮挡程度是相对判断；杯口不是每帧完全不可见；不能据此冻结 relation/cell/contact mode；release 起点仍是意图/grammar依赖判断 |
| 58 | source 左、destination 右；可见手 side 候选为 `LEFT`；前后 idle 最长；约 `p100–116` destination 中出现紫色内容，`p136` 后开始明显回正/移回 | 长静止不自动建立 `VERIFY_OUTCOME`；无 outcome criterion；`MANIPULATE→RELEASE` 与 focus/effect仍需 A/B |

三段都支持 `rgb_temporal` 的“内容转移结果”候选；它不等于逐帧 `FLOW`、流率、体积、spill fraction、3D `INSIDE/TOUCHING` 或 task success。这里同意原 memo 的保守边界。

## 3. 18 个 action-visible 字段的独立复核

这里区分三件事：

1. **raw distinguishability**：某些帧能看出候选值；
2. **code-level repeatability**：A/B 能按冻结定义重复编码；
3. **vocabulary coverage**：样本覆盖该字段的多个重要类别。

原 memo 的 V/P/U 主要描述第 1 项，却在统计段落中容易被误读为第 2/3 项。下表按第 2 项给出更严格的独立 proposal。

| action-visible 字段 | 原 memo | 独立复审 | 证据与限制 |
|---|---:|---:|---|
| node `observability` | P | P | source/destination 可见→局部遮挡→可见；未覆盖完整词表，`PARTIAL/OCCLUDED_TRACKABLE` 分界需 A/B |
| node `center_cell` | V | **V（仅局部）** | 未遮挡 source/destination 的粗 4×4 cell 可直接判；snapshot/frame 与跨 cell 支持仍未冻结 |
| node `scale_band` | U | U | 像素尺度可见，但 SMALL/MEDIUM/LARGE 阈值未冻结，不能 code-level 评估 |
| node `orientation_2d` | U | U | 无标记透明圆杯具旋转对称性；倾倒轴/杯身朝向定义也未冻结 |
| event `kind` | P | P | broad approach/grasp/transport-tilt/putdown 可见；六类切分与未见 regrasp/verify不能由一人冻结 |
| event `focus_node` | P | P | source/destination 可辨，但“失败敏感角色”与每 event 的唯一 focus依赖冻结 ontology |
| event `onset_code` | P | P | transition 可缩小到宽窗口；尚无 A/B boundary lo/hi，future bin/censor未覆盖 |
| event `duration_code` | P | P | 只能在 boundary 后机械派生；原窗口不是 boundary annotation，right-censor/GT96未覆盖 |
| event `object_effect` | P | P | before/after 支持 transfer候选，杯体旋转也直接可见；单 code 的 `ROTATE/TRANSFER` 优先级主观 |
| event `outcome` | U | U | 未建立可观察 verify event及冻结 success/partial/failure criterion；最终有紫色不等于 task success |
| edge `actor` | P | P | visible-hand side有拓扑候选；OBJECT_OBJECT edge与完整 per-side roster尚未独立复核。actor字段与 roster completeness应分开统计 |
| edge `primary_node` | V | **P（从 V 下调）** | source/destination raw role可辨，但选哪条 edge、谁是 primary依赖 task ontology/reference union，尚非 code-level V |
| edge `primary_part` | P | P | hand→source BODY容易讨论；target OPENING/RIM_EDGE在关键阶段部分遮挡，part选择与edge定义未冻结 |
| edge `primary_surface_cell` | U | U | hand-source patch不可见、rim关键局部受遮挡，part-local frame/canonicalization也未冻结 |
| edge `secondary_node` | V | **P（从 V 下调）** | source/destination可区分；是否需要独立 object-object edge、primary/secondary方向仍是审计者 proposal |
| edge `relation_before` | P | P | SEPARATE/NEAR/OVERLAP_2D有局部视觉支持；before anchor与 predicate未冻结，不能提升3D关系 |
| edge `relation_after` | P | P | 同上；关键 turn 的 rim alignment不稳定，需允许 reviewed unknown |
| edge `contact_mode` | P | P | hand与source持续共运动支持 STICK候选；物理接触边界/patch不可见，slide/roll/impact未覆盖 |

更严格的计数为：

```text
局部 V： 1 / 18  center_cell
P：     13 / 18  observability, kind, focus, onset, duration, effect,
                 actor, primary_node, primary_part, secondary_node,
                 relation_before, relation_after, contact_mode
U：      4 / 18  scale_band, orientation_2d, primary_surface_cell, outcome
```

这个 `1/13/4` 也不应进入 gate。正确 gate 输入只能是 A/B 独立 rows、review-state-aware comparable set 与 checker receipt，而不是再增加一个专家投票式 V/P/U 数字。

## 4. 泄漏与证据来源审计

### 4.1 可以确认的事实

- 被复审 commit 只新增 `docs/research/sm_fet_d0_three_episode_observability_audit_20260812.md`；没有提交媒体、截图、contact sheet、HDF5、CSV或代码。
- 三个原始 MP4 是无 overlay 的普通 RGB；source 根目录未发现 `episode_43/50/58_contact_visualization.mp4`。
- 原 memo 没有写出任何 `hand_status/hand_active` 值、contact annotation row、tracker/MANO keypoint 或 detector score；视觉结论也没有明显依赖这些量的特征。
- 原 memo 的 transfer结论能由 raw RGB before/after 独立复现，side候选能由可见手形讨论，不需要 HDF5/proxy 才能得到。

### 4.2 不能过度声称的事实

- Git tree 和成文 memo只能证明“没有把泄漏产物提交/显式引用”，不能证明审计过程中从未打开过某文件。没有进程审计日志、ACL receipt 或 immutable viewer receipt时，负面访问史不可证。
- 原 memo 明确读取 coordinator manifest/timeline；这两份 CSV 包含 processed HDF5 绝对路径/hash，timeline 的 frame mapping与 sync统计来自 HDF5 camera timestamp。它们是 **HDF5-derived technical metadata**，虽没有提供 hand/contact truth，却使“完全没有读取或使用 HDF5”措辞过强。
- 对 proposal-only coordinator audit，使用 timeline mapping本身不构成语义泄漏；但这些 coordinator CSV 绝不能分发给 A/B。A/B只能拿 opaque、隔离后的 timeline与原始媒体。
- 原 memo 本身包含 episode选择、候选 windows、side、event与edge proposal，因此它就是 protocol 所称 prescreen/proposal。未来 blind A/B 不得阅读本 memo，本次独立 review也不能发给 blind annotator。

结论：**没有发现 HDF5/contact visualization/hand-status 被用于语义判断的正证据；存在 HDF5-derived 时间元数据依赖与不可证明的访问史，必须在表述和分发上诚实区分。**

## 5. `0/3 roster` 与 `0/3 oracle` 的正确解释

### 5.1 Roster

三段 raw RGB 都只见一只操作手，另一只 anatomical hand没有进入画面。按 handbook，`out_of_frame/occluded` 不是 `NOT_INVOLVED`，单一 prescreen也不是 exhaustive human review。因此：

- “本 memo 未证明 3 段任一段的完整 hand roster”——**成立**；
- “另一只手全程画外，所以 3 段必然都不能被 A/B 判为 single-hand exhaustive”——**不成立/尚未验证**；
- 真正问题应交给 A、B各自逐帧回答：所有 task-relevant object/contact区域是否得到完整覆盖，是否存在第二条 hand edge的可疑时段，`NOT_INVOLVED` 是证据支持还是仅因未见。

完整 roster是 episode/event coverage结论，不应塞进单条 `edge.actor` 的 V/P/U。已知一条 edge 的 LEFT/RIGHT 可以 known，同时 roster coverage仍不完整；二者要分报。

### 5.2 Oracle packet

“0/3 完整 oracle packet得到证明”是正确的，因为本轮没有 sibling annotation rows、A/B agreement、adjudication、compiler或 round-trip checker。

但不能反推：

- `scale/orientation/cell/outcome` 不 known，所以 packet必然拒绝。前几项有显式 UNKNOWN code；非 verify event 的 outcome本来就是 N/A/0。
- single-person proposal中的 `0/3` 是模型/媒体容量结论。真正 packet admission取决于 required reference是否可解析、review state是否合法、actor/focus/primary等无 UNKNOWN code字段是否齐全、reference union/grammar/capacity是否通过 checker。
- 三段以后永远不能生成 oracle。未来独立 A/B与正式裁决可能生成只含审计过的 known/UNKNOWN字段的合法 pour-local packet，也可能因 roster/actor/reference而 fail closed；当前证据无法选择其中一种。

因此建议以后统一写成：

```text
当前提交中可验证的 admitted oracle packet：0/3（未执行 annotation/compiler/checker）。
这不是对三段潜在可编译性的判决。
```

## 6. 哪些结论只是审计者主观、不能推广

以下项目即使视觉上“很像”，也没有 A/B agreement或冻结 criterion，必须保持 proposal：

1. `43=RIGHT, 50/58=LEFT` 的 anatomical side；
2. 六类固定候选路径，以及“没有 regrasp/verify”的 exhaustive否定；
3. 每个 p evidence window的起止、boundary exact/interval与相邻事件归属；
4. 把回正、移回、放杯整段归为长 `RELEASE`，而不是对 grammar/ontology发出 No-Go；
5. manipulate 的 focus一定是 destination；
6. 每段恰好需要 hand→source与source↔destination两条edge，及 object-object edge的primary/secondary方向；
7. source part=`BODY`、mode=`STICK`，destination part=`OPENING/RIM_EDGE`；
8. 主 effect=`TRANSFER` 而杯体 `ROTATE` 只是实现动作；
9. 另一只手不能在 exhaustive review中被判 `NOT_INVOLVED`；
10. 43/50/58分别代表“镜像/困难/普通”并足以覆盖 calibration难度轴；
11. `3/11/4` 或本复审的 `1/13/4` 能代表 schema readiness；
12. 三段未出现的 regrasp、verify、bimanual、censor、overflow、tool/fixture positive、failure/partial对其它 episode也不可标。

能推广的只有较窄的反例边界：透明 top-down pour 中不能由2D overlap推出3D contact，不能由 wrist tilt推出 flow，不能由最终颜色变化推出 outcome，不能由未看见另一只手直接生成 absence negative。这些是协议规则，不是三段数据统计。

## 7. 真正 calibration 中 A/B 最少要回答的问题

先修正身份：在现行协议下，calibration必须从 **43/50/58之外且不属于固定12 blind的其余54段**预先冻结三段，覆盖标准成功、高事件复杂度、missingness三类。43/50/58只能留在 blind set；A/B不能看到本 memo或本 review。

A、B应先各自完成三段，再比较以下 **5 个结构化问题**。少于这五组会遗漏至少一个 action-visible core或 admission条件；不需要让他们对“18字段是否可见”投总体票。

### Q1. 事件与边界

```text
这段的 maximal event 序列是什么？
对每个 event给 confirmed-positive support、start/end exact|interval|censor|unknown；
明确回答 PRECONTACT、REGRASP、VERIFY_OUTCOME 是否存在，
以及 HOLD→MANIPULATE、MANIPULATE→RELEASE 的 boundary support。
```

它同时检验 kind、onset、duration、grammar、censor和“长 release”分歧。

### Q2. Role、focus 与 node

```text
冻结的四 role→entity binding是什么？每个 event唯一 focus role是什么？
在实际 entry/turn/exit/anchor 中，对四 role的 observability、center、scale、orientation
分别给 KNOWN|REVIEWED_UNKNOWN|NOT_REVIEWED，并说明 ABSENT是否只由reference union机械产生。
```

它检验 focus/reference fail-closed、node四字段与 role overflow；不允许用“对象很显眼”代替 focus。

### Q3. Edge bundle 与完整 hand roster

```text
每个 event的全部 task-relevant edge是什么（最多2条）？
逐条给 actor、primary role/part/cell、secondary role、contact mode；
分别说明 left/right hand roster coverage是否EXHAUSTIVE，未见的手是NOT_INVOLVED还是OUT_OF_FRAME/NOT_REVIEWED；
若需要第3条 edge或 actor/primary不可解，明确报 overflow/inadmissible。
```

它把 visible edge actor与 episode roster completeness分开，直接回答原 memo 的 `0/3`争议。

### Q4. Relation、effect 与 outcome criterion

```text
对有 secondary的edge，relation_before/after是什么、证据是rgb_direct还是rgb_temporal？
manipulate唯一主effect为何是ROTATE、TRANSFER或REVIEWED_UNKNOWN？
是否真的存在VERIFY_OUTCOME；若存在，预先冻结的success/partial/failure criterion是什么？
```

它专门处理 2D/3D、rotate+transfer并发与“最终紫色=成功”的错误捷径。

### Q5. Missingness 与 packet admission

```text
对每个未给known值的required field，它究竟是REVIEWED_UNKNOWN、NOT_REVIEWED/MISSING、ABSENT、censor还是invalid？
在不看proposal/HDF5/hand status/contact visualization的前提下，
该episode能否进入compiler；若不能，唯一首要拒绝原因是什么？
```

它迫使 A/B保留 field-level证据边界，并为 checker准备确定输入。只有 A/B独立输出、comparison report与最多一次 dictionary修订后，才能冻结 protocol hash。

### A/B 不能回答、必须由 D0 checker回答的问题

以下不是让 A/B“达成共识”即可通过：

- annotation到82个 canonical codes是否唯一；
- UNKNOWN/ABSENT/MISSING/censor是否机械可逆；
- grammar、verify-only outcome、focus/reference union与overflow是否被拒绝/报告；
- positive-zero、duration/onset bins与unknown-terminal是否正确；
- packet再解码回annotation是否 round-trip；
- exact reachable capacity是否与 `207.749746919520 bit` receipt一致。

## 8. Gate 判定与使用限制

| claim | 独立复审判定 | 允许的表述 |
|---|---|---|
| raw MP4 identity/metadata | **PASS** | 三段hash/帧率/帧数/时长已独立复核 |
| broad raw-video observability prescreen | **PASS with caveats** | 可作为单人proposal；raw `v`时间需按timeline修正 |
| 18字段 `3/11/4` | **不采纳为gate** | 最多引用为原审计者taxonomy；本复审更严格proposal为`1/13/4` |
| 无语义泄漏 | **未发现正证据，但不可证明完整访问史** | 无直接HDF5/action/contact/hand-state语义依赖；使用了HDF5-derived时间元数据 |
| 43/50/58作为calibration | **No-Go under current protocol** | 它们属于固定blind；改用途需新版本与新blind set |
| hand roster complete | **0/3得到证明** | 只表示本审计未证明；未来A/B exhaustive结论未知 |
| oracle packet admitted | **0/3得到证明** | 未执行annotation/compiler/checker；不是潜在可编译性判决 |
| D0 | **未通过** | 缺checker/round-trip/grammar/capacity receipt |
| D1 | **未执行** | 无合法disjoint calibration、12 blind A/B、agreement与裁决 |

最终结论：原 memo 最有价值之处是诚实暴露透明倾倒中的不可见性与 proposal/GT边界；最危险的误用是把单人窗口、事件语义、edge设计和 V/P/U计数抬成 D0 calibration结论。正确下一步不是再由第三位审计者决定“谁标得对”，而是先恢复 calibration/blind互斥，用上述五组问题获得 A/B独立 rows，再让 D0 checker对编码唯一性和 round-trip作机械判决。
