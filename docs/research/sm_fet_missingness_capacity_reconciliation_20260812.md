# SM-FET `HAND_UNRESOLVED` 与容量口径独立裁决

- 日期：2026-08-12
- 性质：只读 reconciliation / decision memo；不改代码、不训练、不使用 GPU
- 裁决对象：standard-r1 与新增 `HAND_UNRESOLVED` 的候选 r2
- 唯一推荐：**保留 r1；当前不把 `HAND_UNRESOLVED` 加入 action-visible actor 字典**

## 0. Executive decision

`reviewed but side-ambiguous confirmed hand contact` 是一个必须保真的**标注事实**，但在现有机器人执行契约下不是一个必须新增的**动作状态**。当前唯一推荐如下：

```text
r1 action actor:
  ABSENT | LEFT | RIGHT | OBJECT_OBJECT

annotation / planner supervision:
  edge existence = CONFIRMED_PRESENT
  actor review state = REVIEWED_UNKNOWN
  actor target mask = 0
  其它已知 edge 字段按字段监督

oracle/action compiler:
  required actor unresolved -> packet inadmissible

deployment control plane:
  HOLD/REOBSERVE；获得新证据或 allocator 唯一解后重编译
```

不得把该样本改写为 `ABSENT`，也不得把整件事伪装成已有的 event-kind `UNKNOWN`。standard-r1 没有 actor-level `UNKNOWN` code；因此正确的 r1 映射是 annotation-side `REVIEWED_UNKNOWN`、actor loss mask、保留其它字段监督，以及 oracle/compiler fail-closed，而不是借用错误层级的 wire code。

这里把两个方案明确命名：

- **r1**：`sm_fet_hard_grammar_v2_standard_r1`，四码 actor，exact capacity `207.74974691951957 bit`，最小索引宽度 208 bit；
- **r2**：其它 grammar 不变，仅把 `HAND_UNRESOLVED` 加成第五个 actor code，exact capacity `210.24289126401328 bit`，最小索引宽度 211 bit。

r2 是可检验的候选，不是当前规范。只有目标机器人上证明该码带来一种不同于“猜 side”及“hold/reobserve”的、确定且安全的可执行行为，并通过下面的容量匹配 benchmark，才值得另立 schema/grammar version。人类视频中 side 难判本身不满足这个条件。

## 1. 六份规范的冲突来源

### 1.1 已冻结的上位规范共同支持 r1 fail-closed

1. `gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_schema_architecture_20260812.md` 冻结 standard-r1 四码 actor；actor/primary reference 无法确定时没有合法 packet，invalid/overflow 进入 safe hold/reobserve。它给出的 exact reachable alphabet 是 `207.74974691951957 bit`。
2. `gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_final_architecture_decision_20260812.md` 明确决定“保持 r1 action wire 与固定全局 grammar，不为适配尚未产生的标签扩大 schema”，并把 D0 round-trip 与 `207.749746919520-bit` receipt 设为实现前 gate。
3. 修订后的 `gwp-agent-temporal-event-meta/docs/research/sm_fet_annotation_and_pairing_protocol_20260812.md` 明确写出 actor 没有 UNKNOWN code；actor/primary node 无法确定时 edge 不进入 standard packet。`MISSING/NOT_REVIEWED` 不变成 UNKNOWN，只有 EXHAUSTIVE negative 才变成 ABSENT。
4. `gwp-agent-sm-fet-focus-token/docs/research/sm_fet_dynamic_focus_token_spec_20260812.md` 选择复用现有 focus/node/edge 绑定，要求不增加 action-visible token，并把 `207.74974691951957 bit` 作为不变量。
5. `gwp-agent-sm-fet-transparent/docs/research/sm_fet_transparent_nonreconstruction_contract_20260812.md` 同样规定：focus、actor、primary node这类没有 UNKNOWN code的 required reference 无法确定时，packet `INVALID_REFERENCE`，部署 hold/reobserve，不猜值。

这些 memo 并没有把“未知”误当作“不存在”。它们选择的是：unknown/missing 可以存在于 annotation 与 factorized planner loss，但不是所有 unknown 都必须占一个 action code。

### 1.2 missing-hands memo 提出的是 schema 扩展 r2

`docs/research/sm_fet_missing_hands_contract_20260812.md` 正确识别了三件事：

- confirmed hand contact、confirmed absence 与 unreviewed missing 必须分开；
- tracker inactive、遮挡、出画和 last-value hold 都不能 author GT；
- side ambiguity 不能被强行标成 left/right。

但它进一步把 reviewed ambiguity 编为 action-visible `HAND_UNRESOLVED`，允许其进入 oracle packet。这一步不只是补 annotation 缺口，而是改变 actor alphabet、validator、enumerative codec、equal-bit control和 oracle 的完整性定义。因此它与上述 final architecture、annotation、dynamic-focus 和 non-reconstruction memo 不是同一 r1 规范下的兼容补丁，而是候选 **r2**。

本裁决接受 missing-hands memo 的数据语义，拒绝其当前 action-visible admission。

## 2. 为什么“确认有接触”不自动推出“动作必须看见 side ambiguity”

### 2.1 三个层次不能合并

```text
physical truth:
  至少一条 task-relevant hand contact 确认存在

epistemic state:
  anatomical side 经复核仍不可判

robot command:
  哪个目标末端执行何种动作
```

前两项是可信 annotation；第三项需要目标机器人上的唯一执行解释。`HAND_UNRESOLVED` 只回答前两项，没有回答第三项。

### 2.2 在机器人侧只有三种情况

1. **allocator 可由目标机器人 proprio、可达性与安全约束唯一选出 effector。** 此时应先得到唯一 robot actor，再编译 `LEFT` 或 `RIGHT`。人类 anatomical side 并不是动作所需信息，`HAND_UNRESOLVED` 不具独立 action value。
2. **多个 effector 都可行且行为等价。** r2 可能执行成功，但这只能说明 actor side 对该任务不重要，不能证明新增 ambiguity code必要；actor-ablated或 r1 mask route应取得同等结果。
3. **多个 effector 的后果不同，且 allocator 不能唯一选择。** `HAND_UNRESOLVED` 不是一条完整命令。让 WAM从数据相关性暗选 side会失去可审计性；唯一安全行为仍是 hold/reobserve/abstain。

因此，r2 当前没有定义一种 r1 control plane 无法表达的可操作行为。它可以是 uncertainty telemetry，但 telemetry 不应仅因“有限”就自动获得 action visibility。

### 2.3 不能借用已有 `UNKNOWN`

已有 UNKNOWN codes 是 field-specific：event kind、node observability、part/cell、relation、contact mode、effect/outcome等各有自己的语义。把 side ambiguity改成 `event_kind=UNKNOWN` 会同时丢掉已知 event kind，并错误宣称“不知道是什么事件”；把 edge删掉或写 `ABSENT` 则会制造 false negative。

所以 r1 的诚实选择不是“找一个现成 UNKNOWN 塞进去”，而是：

```text
reviewed actor ambiguity
-> annotation REVIEWED_UNKNOWN
-> actor target mask = 0
-> known edge existence/payload保留为 planner-side字段监督
-> hard action packet inadmissible
-> HOLD/REOBSERVE
```

如果未来实现中的 packed actor把 presence和side完全耦合，confirmed presence仍必须留在 sibling annotation/provenance及独立 factorized planner loss中；不能为了提高 packed compile coverage而造一个 action code。

## 3. 唯一标注与编译映射

| 证据状态 | annotation truth | planner/loss | r1 hard packet | 部署行为 |
|---|---|---|---|---|
| EXHAUSTIVE review确认没有该 edge | `NOT_INVOLVED` / confirmed negative | presence negative | tail `actor=ABSENT`，payload全 `+0` | 正常编译 |
| hand edge存在，side确认 left | `KNOWN: LEFT` | actor及已知payload监督 | `LEFT` | 正常编译 |
| hand edge存在，side确认 right | `KNOWN: RIGHT` | actor及已知payload监督 | `RIGHT` | 正常编译 |
| object-object edge确认 | `KNOWN: OBJECT_OBJECT` | actor及已知payload监督 | `OBJECT_OBJECT` | 正常编译 |
| hand edge确认存在，双标/裁决后side仍不可判 | `CONFIRMED_PRESENT + REVIEWED_UNKNOWN`，reason保留 | presence与其它known字段可监督；actor mask=0 | **不产生 oracle/action edge；packet inadmissible** | hold/reobserve；新证据后重编译 |
| 可能有hand edge但side或参与未复核 | `NOT_REVIEWED/MISSING` | 相应mask=0 | 不编 UNKNOWN/ABSENT | hold/reobserve或排除样本 |
| occluded/out-of-frame/tracker inactive，但其它证据已确认side/contact | observation reason独立保存 | 已确认字段照常监督 | 按确认的 LEFT/RIGHT 编译 | 正常编译；reason不进 action |
| 模型/VLM/MANO/时序hold给出side | `MODEL_PROPOSAL/PSEUDO` | 仅proposal baseline或auxiliary | 永不升级为oracle GT | 不据此解除 fail-closed |

补充规则：

- 一侧已知、另一侧可能参与但未审，不能编成 `left-only/right-only` oracle；
- `hand_roster_coverage` 非 EXHAUSTIVE时，空尾不产生 ABSENT negative；
- occlusion不是 censor；只有 observation coverage 截断真实 onset/offset才产生 left/right censor likelihood；
- 人类 left/right 不直接作为 target robot left/right 的跨域配对 cost；robot actor必须来自目标域证据或显式 allocator；
- safe hold/reobserve是 control-plane outcome，不新增为 action token，也不进入容量计数。

## 4. 207.7497 与 210.242891 的独立容量复核

### 4.1 两个数字共享的计数骨架

两者都保持以下 standard grammar 不变：四个 role nodes、action `active+next_1+next_2`、每 event 最多两条 canonical unique edges、固定 kind transition、UNKNOWN terminal、future onset非降序，以及 node reference union包含–排除。

给定 packet 中恰有 `n` 个 present nodes，edge除 actor外的公共因子为：

```text
n primary nodes * 9 primary parts * 17 primary cells * 6 contact modes
= 918*n
```

若有 `h` 种 hand-like actor，每种 hand actor允许无 secondary，或引用其它 `n-1` nodes中的一个并带 `9*9` 个 relation before/after；OBJECT_OBJECT要求非空 secondary。因此：

```text
E_n(h) = 918*n * [h*(1 + 81*(n-1)) + 81*(n-1)]
       = 918*n * [h + 81*(h+1)*(n-1)]

B_n(h) = 1 + E_n(h) + C(E_n(h), 2)
A_n(h) = 88*n*B_n(h)
```

其中 `B_n` 是 0/1/2 条不重复、canonical-sorted edge bundle数，`88*n` 来自 focus、duration与effect基数。kind path按 VERIFY 的四种 outcome加权：

```text
C_p, p=1..8 = [12, 39, 108, 289, 759, 2013, 5408, 14562]
T_action(p) = C(17+p-2, p-1)
```

对最多 K 个 action events：

```text
F_K(n) = sum_{p=1..K} T_action(p) * C_p * A_n^p
H_K(n) = sum_{j=0..n} (-1)^(n-j) * C(n,j) * F_K(j), F_K(0)=0

S_action = 1
         + sum_{n=1..4} C(4,n) * 1802^n * H_3(n)
```

`1802` 是一个 present node 的合法状态数；最前面的 `1` 是全局 NULL。

### 4.2 r1：207.74974691951957 bit

r1 有两种 hand actors，故 `h=2`：

```text
E_n(r1), n=1..4
= [1,836, 449,820, 1,343,952, 2,684,232]

S_action_r1
= 345864043392188289642033658185203030253953267411599196753571073

log2(S_action_r1)
= 207.74974691951957 bit

ceil(log2(S_action_r1))
= 208 bit
```

这正是 schema、final architecture、dynamic focus、transparent non-reconstruction共同引用的 standard-r1 validator reachable alphabet。它不是 82-byte存储宽度、经验熵或互信息。

### 4.3 r2：210.24289126401328 bit

r2 把 `HAND_UNRESOLVED` 当作第三种 hand-like actor，故 `h=3`：

```text
E_n(r2), n=1..4
= [2,754, 600,372, 1,792,854, 3,580,200]

S_action_r2
= 1947227281526202347439908176679818858538657207150616941292393985

log2(S_action_r2)
= 210.24289126401328 bit

ceil(log2(S_action_r2))
= 211 bit

delta(r2-r1)
= 2.4931443444937145 bit
```

独立复算也复现了 memo 中的 Q8 planner bounds：

```text
r1 Q8 = 478.9093895593358 bit，ceil = 479
r2 Q8 = 485.5578814616901 bit，ceil = 486
delta_Q8 = 6.648491902354294 bit
```

### 4.4 精确解释

r2 仍可装在相同的 82 个 `uint8` fields、13 tokens中，因为 storage layout没有增加字段；但其**可达语义 alphabet**已经扩大，最小 enumerative index由208变为211 bit。以下说法不能同时成立：

```text
actor增加 HAND_UNRESOLVED
且
exact action-visible capacity仍为207.7497 bit / fixed index仍为208 bit
```

如果采用 r2，必须新建 schema/grammar/capacity receipt，重做 rank/unrank与全 packet equal-bit permutation，并把 actor head/embedding、参数量、FLOPs和经验利用熵一并报告。拿 r2 semantic route 对 r1 的208-bit equal control不是容量公平比较。

## 5. Oracle sufficiency 裁决

### 5.1 r1 的 oracle定义更干净

oracle route的目的，是在目标机器人当前 native/sensor-confirmed state下检验有限 packet是否保留足够控制信息。一个没有唯一 effector assignment、且后续行为依赖assignment的 packet不是完整 actuator oracle。r1 把这类样本拒于 oracle action condition之外，符合“required target不完整则 fail-closed”。

这不会授权只报 complete cases。必须同时报告：

- natural missingness下的 oracle compile coverage与拒绝率；
- rejected样本按 phase、task、side、bimanual、group与observation reason的分层；
- sensor/mocap fully observed subset上的完整 resolved-side oracle；
- controlled hiding只改变可见输入，不改变原 fully observed oracle truth。

由此可以量化保守性与 selection shift，而不把不完整 packet洗成oracle。

### 5.2 r2 会改变问题，而不是免费提高oracle coverage

把 `HAND_UNRESOLVED` 样本纳入 `event_tube_v2_oracle` 会把“oracle”改成“ambiguity-aware condition”。若 allocator另有 native state并最终唯一选 side，真正完成执行语义的是 allocator的额外信息；若没有，action head只能猜或隐式学习选择。两者都不能直接证明 r2 packet自身 sufficient。

r2 不必然违反 oracle sufficiency，但必须另名 route并回答一个不同问题：

> 在明确暴露 actor ambiguity时，一个受约束的 allocator/controller能否比 r1 hold/reobserve更安全高效地恢复？

在该证据出现前，把 r2 称为 oracle、或用更高 compile coverage替代控制成功，是 oracle laundering。

## 6. 科学利弊

| 方案 | 科学收益 | 科学代价 |
|---|---|---|
| **r1（推荐）** | action语义最小；与四份上位规范一致；oracle target完整；208-bit equal control已定义；部署fail-closed可审计 | actor ambiguity不参与hard actor CE；oracle coverage更低；若只做complete-case可能产生selection bias |
| r2 | wire显式区分“确认有手但side未知”和unreviewed/absent；可能改善ambiguity calibration、减少false absence；对side-invariant动作或可恢复controller可能有用 | epistemic状态进入action；本身不选择robot effector；扩大到210.242891 bit；需新codec/equal control；可能泄漏annotator/dataset missingness并污染oracle口径 |

r1 的 coverage代价通过 field-level masks、confirmed-presence sidecar、risk-coverage报告和 controlled hiding解决；不能通过错误的 ABSENT 或 complete-case-only报告掩盖。r2 的潜在收益则必须来自机器人行为，不是来自“多编译了多少人类片段”。

## 7. 预注册 benchmark

### 7.1 D0：r1 contract reconciliation，任何训练前

只批准 annotation/checker 的 CPU-only验证设计，要求：

1. `ABSENT / REVIEWED_UNKNOWN / NOT_REVIEWED / LEFT / RIGHT` round-trip不互换；
2. reviewed side ambiguity保留 confirmed presence与其它known payload，但actor梯度为零；
3. reviewed side ambiguity和unreviewed possible hand都不能生成 r1 oracle packet；
4. 不允许把 actor ambiguity编成 event-kind UNKNOWN或edge ABSENT；
5. inference unresolved required actor稳定产生同一 fail-closed receipt与 hold/reobserve；
6. r1 exact integer、`207.74974691951957`和208-bit receipt逐字一致；
7. missingness reason、confidence、proposal logits/hidden改变时，合法 hard packet不变；
8. robot allocator若给出唯一、有receipt的target actor，重编译结果只能是 LEFT或RIGHT。

任一失败：停止 UI/model 扩展。不能临时引入 r2让样本通过validator。

### 7.2 D1：3 calibration + 12 blind human-pour，只验证可标注性

沿修订 protocol做 A/B独立标注与第三人裁决，并增加：participation macro-F1 `>=0.80`、side三类 annotation κ `>=0.70`、false-ABSENT为0、incomplete packet误admission为0、censor-vs-occlusion confusion `<5%`、critical reviewed unknown `<=40%`、unresolved/invalid/overflow分别 `<=5%`。支持不足的 ambiguous/bimanual类别必须标 `NOT_EVALUABLE`，不能以零错误过门。

D1只能说明 annotation-side ambiguity是否可重复，不支持 r2 action visibility。

### 7.3 B2：missingness gold set

任何 r2 promotion实验前至少需要：

```text
>=300 adjudicated events
>=50 confirmed bimanual events
>=30 production groups
sensor/mocap-confirmed fully observed subset
natural occlusion/out-of-frame/inactive strata
```

同时做 natural missingness与controlled hiding。controlled hiding的side/contact truth只能来自遮挡前的native/mocap truth，不来自模型补全。

### 7.4 R-action：只有在考虑r2 promotion时执行

在36 human + 36 target-robot同任务 pilot及native robot truth到位后，使用相同 split、backbone、数据、action horizon、execute-8/replan、observation budget与安全controller比较互斥 routes：

1. `resolved_side_oracle_r1`：native target robot side真值，控制上界；
2. `mask_hold_reobserve_r1`：本裁决主方案；
3. `hand_unresolved_r2`：明确的r2 schema，加预注册 allocator；
4. `equal_bit_r2`：在 r2 全packet rank上的可逆非语义置换，NULL固定；
5. `continuous_trajectory`：互斥信息上界。

必须报告：task/event success、correct-first-contact、unintended contact、arbitrary/wrong-side rate、hold/reobserve频率、time-to-resolution、watchdog恢复率、risk-coverage、actor calibration、single/bimanual/alternating strata，以及按cell/clip/rollout cluster bootstrap的95% CI。r2和r1的直接比较回答系统取舍；r2和`equal_bit_r2`回答新增语义是否超越额外容量。

### 7.5 上位 oracle/predicted gates不因本裁决放宽

- native robot oracle finite route相对oracle 22D-only：outcome-sensitive success `>=+8 pp` 或 event completion `>=+10 pp`；
- 相对full continuous trajectory下降不超过`3 pp`，safety不退化且cluster CI支持正优势；
- predicted strict route恢复至少`70%`的正oracle gain；
- human literal side、missingness reason、proposal/imputer hidden不得绕过hard packet进入action/cache。

## 8. r2 promotion 与停止条件

### 8.1 只有全部满足才允许从r1另立r2

1. gold set中 `HAND_UNRESOLVED` annotation κ `>=0.70`、F1 `>=0.80`，且不是主要吸收 NOT_REVIEWED；
2. 目标机器人域存在足量、自然发生且reobserve后仍持续的actor ambiguity，不只是人类视频laterality难判；
3. allocator看到该码后有预注册的确定安全行为，不能随机选side，也不能由不可审计hidden绕过；
4. r2相对r1 mask+hold在至少一个预注册核心策略指标上 point gain `>=5 pp`、cluster 95% CI下界`>0`，同时wrong-side/unintended-contact与safety不恶化；
5. r2相对其210.242891-bit equal control有同方向优势，且经验token利用率支持语义使用；
6. 新schema/grammar/codec/capacity receipt完整，所有208-bit主张改为211-bit最小索引宽度主张。

这些是 promotion gate，不是当前已满足事实。

### 8.2 任一发生则停止r2

- ambiguity标签 κ/F1低于`0.70/0.80`，或主要混入unreviewed missing；
- allocator只能随机/隐式选side，reobserve也不能降低风险；
- r1 mask+hold不劣于r2，或r2只减少abstain却增加wrong-side/unintended contact；
- r2 equal-bit与semantic route持平且利用率相当；
- r2收益只来自missingness reason、confidence、teacher hidden或其它action旁路；
- 用r1的208-bit equal control、receipt或论文口径报告r2。

发生以上任一项，回到r1 strict mask+abstain；不得继续增加hand count、visibility、confidence或latent code保护结果。

### 8.3 finite route的共同停止条件

- resolved-side oracle finite packet不优于22D-only，或距full trajectory超过3 pp；
- 只有continuous pose/trajectory、ROI/heatmap或重建action-visible才有效；
- predicted route恢复不到70% oracle gain；
- incomplete packet在bimanual/alternating子集触发unsafe action而非可校准abstain；
- action随human frame、missingness metadata或proposal/imputer hidden变化。

此时event/missingness数据仍可保留为annotation benchmark，但停止small finite action representation与non-reconstruction sufficiency主张。

## 9. 最终唯一建议

**采用 r1。** 将 reviewed-but-side-ambiguous confirmed contact保存为 annotation-side `CONFIRMED_PRESENT + REVIEWED_UNKNOWN`，对actor做mask、对其它已知字段做factorized planner监督，并拒绝其进入oracle/action packet；部署使用现有hold/reobserve，或在目标机器人allocator产生唯一、有receipt的side后重编译为LEFT/RIGHT。

当前不采用 r2，因为 `HAND_UNRESOLVED` 表达的是可信的不确定性，却尚未表达一个唯一可执行的机器人动作。它会把exact action capacity从`207.74974691951957`提高到`210.24289126401328 bit`，把最小索引宽度从208提高到211 bit，并迫使oracle与equal-capacity benchmark重新定义。只有机器人域的独立行为证据通过第8.1节全部promotion gates，才允许把它作为另一个正式版本，而不是静默修改standard-r1。
