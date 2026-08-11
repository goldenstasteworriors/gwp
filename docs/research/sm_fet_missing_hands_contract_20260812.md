# SM-FET Ego 双手非持续真值：唯一 missingness contract

- 日期：2026-08-12
- 审查性质：独立结构审查；只做文献、schema、loss 与实验契约，不实现代码、不训练、不使用 GPU/服务器
- canonical 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 工作树：`codex/sm-fet-missing-hands-contract-20260812`
- 总体 idea：`b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`
- 本轮 critic discussion：`2de8b06f-ab83-4197-8fc2-3c903374ce08`
- `idea-discussion`：已同步 Zotero 100 条记录，`memory_updated=true`

## 0. 唯一裁决

SM-FET 必须采用一个 **reason-coded、field-level、per-side、fail-closed** 的 missingness contract。其核心不是“给不可见手补姿态”，而是把五件互相正交的事分开：

```text
物理参与       involved / confirmed-not-involved
观测原因       visible / partial / occluded / out-of-frame / motion-blur
标注状态       known / reviewed-unknown / not-reviewed
采集有效性     tracker-active / inactive / unavailable
时间边界       exact / interval / left-censored / right-censored / unknown
```

唯一推荐的 action-visible 变化只有一项：

```text
edge.actor:
  ABSENT | LEFT | RIGHT | OBJECT_OBJECT | HAND_UNRESOLVED
```

`HAND_UNRESOLVED` 表示：**至少一条 task-relevant human-hand contact edge 已由人工/传感证据确认存在，但该 edge 的 anatomical side 无法可靠判为 left/right**。它是有限语义，不是填补出来的左右手，也不等于未复核。

其它 missingness 原因不进入 action packet：

- `OCCLUDED`、`OUT_OF_FRAME`、`MOTION_BLUR`、`TRACKER_INACTIVE`、`NOT_REVIEWED` 只作 loss mask、missingness auxiliary、分层指标与 provenance；
- confirmed `NOT_INVOLVED` 只在对应 side/event coverage 为 `EXHAUSTIVE` 时授权 edge-slot `ABSENT` negative；
- `LEFT/RIGHT_CENSORED` 仍按时间删失进入 boundary likelihood；当前 active event 的 left censor 编译成 `onset=ACTIVE`，right censor 编译成现有 `duration=RIGHT_CENSORED`；
- 不完整 hand roster、第二只手可能存在但未复核、两条 side assignment 无法形成 canonical packet时，不把已知一条 edge 编译成“单手完整 GT”，而是保留字段监督并拒绝 oracle action condition；
- 模型补全、VLM/MANO/HandOccNet 估计、时序插值、last-value hold、`hand_status` 与 `hand_active` 一律不是 GT。

这个裁决刻意不新增 event-level `hand_scope`、hand visibility token、confidence token 或 imputed pose token。完整性由 annotation coverage 与 compiler admission 控制；动作只看经过完整性检查的有限语义。这样解决“缺一只手被误当单手”的问题，同时保持 wire 仍为 82 bytes、13 tokens。

当前状态：

- **Go**：修订 D0 annotation/compiler 契约、做 CPU-only 容量与 round-trip 设计、执行 3+12 missingness 双标；
- **Conditional Go**：只有 per-side completeness、unknown/censor 与 bimanual gate 通过，才进入 36+36 同任务 pilot；
- **No-Go**：当前实现/训练 WAM、把缺手补成 GT、把 tracker inactive 当无接触、把现有 66 Ego 当持续双手真值，或引用 predicted hand mesh 作为 gold。

本轮项目内证据账本：

- `gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_schema_architecture_20260812.md`：standard-r1 字段、语法、容量与 wire 基线；
- `gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_final_architecture_decision_20260812.md`：最终 route、gate 与停止条件；
- `gwp-agent-temporal-event-meta/docs/research/sm_fet_annotation_and_pairing_protocol_20260812.md`：`UNKNOWN/ABSENT/MISSING/censor` 和 actor fail-closed 规则；
- `gwp-agent-op28-grip-evidence/docs/audits/ego_action_grip_evidence_20260811.md` 与 `evidence/ego_stage1/7_4_pour_liquid_between_beakers/ego_action_grip_evidence_v2.json`：66 段 Ego hand-active/grip 证据；
- `docs/research/idea_b_asymmetric_contact_topology_review_20260811.md`：Idea B 的 functional topology、allocator 与 hands-first 限制；
- `gwp-agent-final-idea-portfolio/docs/research/wam_biochemical_manipulation_idea_portfolio_20260812.md`：original issue 2、Portfolio D 与统一实验预算。

## 1. 为什么原问题是真问题，而不是数据清洗细节

### 1.1 canonical 代码和 op28 已经证明 tracker validity 极不均匀

op28 的 66 段 `7_4_pour_liquid_between_beakers` 审计给出：

| 量 | left | right |
|---|---:|---:|
| processed 20 Hz active frames | 7,087 / 20,013（35.41%） | 6,574 / 20,013（32.85%） |
| zero-current-supervision episodes | 30 / 66 | 29 / 66 |
| future H=48 valid grip targets | 335,074 | 308,116 |
| closed-proxy but inactive frames | 7,331 / 11,005 | 7,842 / 10,621 |

两手同时 active 只有 165 帧，占 0.824%；两手同时 inactive 有 6,517 帧，占 32.56%。这些数字只描述 tracker validity，不描述物理参与或接触。`world_action_model/ego_hand_active_sidecar.py:1-6,36-45` 已明确把 `hand_active` 与 proxy value、physical contact label 分离；`world_action_model/datasets/wam_lerobot_dataset.py:326-367` 还要求 receipt 中 `physical_contact_truth=false`。

因此：

1. 没有某手的 active proxy，不能推断该手未参与；
2. 有 active proxy，也不能推断该手触物；
3. 某 episode 一侧零监督，不能支持 left-only/right-only/bimanual truth；
4. 对 inactive proxy 做 zero fill 或 last hold，会把采集缺失机制写成动作标签。

### 1.2 现有 standard-r1 对 ambiguous side 是 fail-closed，但还缺一个诚实 action 类

当前 v1.1 `contact_transition_schema.py:65-89` 的 side 只有 `left/right/object_object`；`contact_transition_schema.py:629-711` 只有 `coverage_is_exhaustive=True` 时才监督空 slot presence，未知 categorical 由 field mask 去掉。这一原则是对的。

SM-FET v2 annotation memo 进一步规定：actor 没有 UNKNOWN code时，side/primary node 不可判的 edge 不得编 packet。问题在于部署 planner 遇到“确认 hand edge，但 laterality 不可判”时只能：

- 武断选 left/right；
- 丢掉真实 edge；或
- 整 packet abstain。

前两者错误，第三者对所有 side ambiguity 又过度保守。`HAND_UNRESOLVED` 是唯一必要的有限补丁：它表达已知存在的 hand edge，但不伪造 laterality。

### 1.3 occlusion 是 MNAR 候选，不能默认 ignorability

手更可能在 acquire/hold/manipulate 时被物体遮挡，出画也可能与大幅腕部动作、双手协作或佩戴者视线有关；缺失概率可能依赖尚未观测到的 pose/contact 本身。统计上这至少是 MNAR 候选，而不是可随意假设 MCAR/MAR。

- [Mohan & Pearl 的 missingness graph](https://arxiv.org/abs/1801.03583) 强调 missingness mechanism、recoverability 与 testability 必须显式建模；MNAR 下并非所有目标分布可由 observed cases 识别。
- [not-MIWAE](https://arxiv.org/abs/2006.12871) 展示了当 missing process 依赖 missing value 时，忽略 missing mechanism 会偏置 likelihood；但它的 generative imputation 不能被移植成 SM-FET 的 GT authoring。
- [nnPU](https://arxiv.org/abs/1703.00593) 和 [SAR-PU](https://arxiv.org/abs/1808.08755) 说明 unlabeled 不等于 negative，而且 PU 风险估计依赖 SCAR/SAR、class prior 或 propensity 等假设。Ego 遮挡显然没有预先证明这些假设。

结论不是“用更复杂模型自动补全”，而是：主监督只使用可审计 observed support；MNAR/PU 方法只作敏感性 baseline 和 missingness calibration，不生成 GT。

### 1.4 视觉论文能提供 proposal 或 fully observed audit set，不能把推断变真值

- [100 Days of Hands](https://arxiv.org/abs/2006.06669) 已覆盖 hand location、side、contact state 与 contacted object box；因此这些视觉分类本身不是新意。
- [HandOccNet](https://arxiv.org/abs/2203.14564) 明确针对严重 hand-object occlusion 预测完整 hand mesh；它证明“可预测”而非“预测就是 GT”。
- [HOT3D](https://arxiv.org/abs/2406.09598) 用外部 motion capture 给双手/对象 3D pose，并显式发布 valid-frame mask：1.5M multi-view frames 中 1.16M 全标且通过人工检查。它支持“GT 有 validity domain”的做法。
- [ARCTIC](https://arxiv.org/abs/2204.13662) 提供双手、articulated object 与动态 contact 的 3D ground truth；适合作为 fully observed/occlusion stress 的外部 sanity set，不自动成为本项目动作语义 GT。
- [2HandedAfforder](https://arxiv.org/abs/2503.09320) 已区分单/双手 affordance，并覆盖两手协作；本工作的残余不是二手分类，而是 finite WAM 中 absence/missingness/censor 的可审计编译。

## 2. 五轴数据模型：禁止再用一个 `hand_valid` 混装

推荐新增 episode-level sparse sibling 表 `event_hand_observations_v1`。它不替换 `events_v1`/`event_edges_v1`，而是为每个 event 的 wearer hand roster 提供 completeness receipt。

### 2.1 最小字段

| 轴 | 字段 | 值域/约束 | action-visible |
|---|---|---|---|
| identity | `event_id,hand_observation_id,hand_entity_local_id` | `hand_entity_local_id` 只在 episode/annotation 内稳定；不进 packet | 否 |
| participation | `participation_value` | `INVOLVED / NOT_INVOLVED` | 只机械决定 edge presence；不单独编码 |
| participation review | `participation_review_state` | `KNOWN / REVIEWED_UNKNOWN / NOT_REVIEWED` | UNKNOWN/MISSING 不直接编码 |
| side | `side_value` | `LEFT / RIGHT / AMBIGUOUS` | `AMBIGUOUS -> HAND_UNRESOLVED` |
| side review | `side_review_state` | `KNOWN / REVIEWED_UNKNOWN / NOT_REVIEWED` | reviewed ambiguity才可编；未审 mask |
| observation | `observation_state` | `VISIBLE / PARTIAL / SELF_OCCLUDED / OBJECT_OCCLUDED / OUT_OF_FRAME / MOTION_BLUR / UNKNOWN` | 否 |
| coverage | `hand_roster_coverage` | `EXHAUSTIVE / POSITIVE_ONLY / NOT_REVIEWED`，event-level冻结 | 否；只控制 ABSENT admission |
| tracker | `tracker_state` | `ACTIVE / INACTIVE / UNAVAILABLE / NOT_APPLICABLE` | 否；只控制 proxy loss |
| contact evidence | `edge_evidence_state` | `DIRECT / TEMPORAL / SENSOR_CONFIRMED / GEOMETRY_PROXY / MODEL_PROPOSAL` | 否 |
| confidence | `confidence_for_loss_only` | `[0,1]`，不可进入 planner/action | 否 |
| provenance | `source_frame_ids,annotator_id,adjudication_id,teacher_version,receipt_sha256` | proposal 必须留 teacher/version | 否 |
| boundary | `onset/offset lo,hi,status` | exact/interval/left/right censor/unknown | 只有编译后的 onset/duration/censor语义 |

每个 `event_hand_observations_v1` 行还必须关联零到多条 `event_edges_v1`。一个 physical hand 可在同 event 产生两条不同 contact edge；不要把 edge 数当手数。

### 2.2 三种“没有”必须分开

| 概念 | 含义 | 允许的监督 |
|---|---|---|
| `NOT_INVOLVED` | 该 anatomical hand 在 reviewed event 中确认没有 task-relevant 参与 | 只有 `EXHAUSTIVE` roster 才授权对应 negative/空 edge slot |
| edge `ABSENT` | finite packet 的该 edge slot 未使用 | compiler 产物；不声称手不存在、物体不存在或没有未审行为 |
| `NOT_REVIEWED/MISSING` | 没有证据或没有完成复核 | 所有相应 loss mask=0；不得编为 ABSENT/UNKNOWN |

`node.observability=ABSENT` 仍表示 node 不在 active/next2 reference union，不表示物理对象不存在。hand roster 的 `NOT_INVOLVED` 也不占 node slot。

### 2.3 occluded、out-of-frame 与 tracker inactive

三者都是观测/采集状态，不是物理标签：

- `OCCLUDED`：手预计仍在图像 frustum 内，但被自身或对象遮挡；
- `OUT_OF_FRAME`：手的 task-relevant geometry 不在该相机画面内；
- `TRACKER_INACTIVE`：特定 tracker/proxy 在该采样点无有效输出；可能可见，也可能遮挡/出画，除非另有 evidence 不做等价映射。

如果 temporal/multiview/sensor evidence 足以确认 participation、side 或 contact field，这些字段照常监督，同时保留 observation reason；否则分别写 `REVIEWED_UNKNOWN` 或 `NOT_REVIEWED`。不允许“一行遮挡，所以整 event 全 mask”，也不允许“hand detector 没框，所以 not involved”。

### 2.4 ambiguous side 的严格定义

`side_value=AMBIGUOUS` 只在以下条件同时满足时使用：

1. reviewer 确认至少一条 hand edge 存在；
2. primary role/node 可解析，至少能形成有用的 contact target；
3. 多帧回看后仍不能可靠分 left/right；
4. 该状态经过独立双标或裁决；
5. 不是简单的 `NOT_REVIEWED`、tracker inactive 或 detector 无输出。

它机械编为 `HAND_UNRESOLVED`，并参与 actor CE。模型预测的 ambiguous side 可以作为 prediction，但不能回写 annotation 的 `AMBIGUOUS` GT。

## 3. Compiler admission：部分字段可以学习，不完整 packet 不能冒充 GT

### 3.1 两层输出必须分开

对每个 chunk 编译两类产物：

```text
planner supervision:
  per-field target/mask/confidence/boundary likelihood
  可保留局部正 edge，即使其它手未审

oracle action packet:
  只有完整 hand roster + 完整 reference + grammar-valid packet
  才允许作为 event_tube_v2_oracle condition
```

这解决了最危险的情况：左手 contact 已知、右手未审时，左手字段仍能训练 planner，但 oracle action branch 不会看到一个假的 `LEFT_ONLY` 完整 packet。

### 3.2 完整性 gate

一个 event 可进入 oracle action packet，当且仅当：

1. `hand_roster_coverage=EXHAUSTIVE`，或 event ontology 明确 `object_only/none` 且已审；
2. 每个 task-relevant hand edge 的 actor 至少为 `LEFT/RIGHT/HAND_UNRESOLVED`；
3. 每个 edge 的 primary node 可解析且在 present node union；
4. present edges 为 canonical prefix、严格排序、无重复，最多两条；
5. 未出现“确认第二只手可能参与但 edge 未复核”的状态；
6. boundary、event kind、focus 与 effect 达到 standard-r1 required target；
7. packet validator 返回 `OK/NULL`，不是 overflow/invalid。

不完整样本的 action condition mask 不是“逐字段用 prediction 补 GT 后 merge”。它是整个 packet inadmissible。主 predicted policy forward 永不接收 GT packet；oracle 是独立 route/checkpoint。

## 4. 单手、交替、同步双手与 unknown 的有限状态机

### 4.1 action packet 的机械判定

| event hand topology | canonical edge bundle | 语义 |
|---|---|---|
| no-hand/object-only | 无 hand actor；可有 `OBJECT_OBJECT` | 只有 exhaustive review才确认无 hand edge |
| left single-hand | present hand edges 的 actor 仅 `LEFT`，且 roster确认 right not involved | 左手单手 event |
| right single-hand | present hand edges 的 actor 仅 `RIGHT`，且 roster确认 left not involved | 右手单手 event |
| bimanual | 同一 macro event 的 bundle 同时含至少一个 `LEFT` 和一个 `RIGHT` edge | 并发/重叠双手；不建两个同 onset event |
| side unresolved | 至少一条 `HAND_UNRESOLVED` edge | hand contact已知，literal side/assignment未知 |
| incomplete | 任一可能参与 hand 未审，或 known edge不能形成完整 packet | 不产生 oracle packet；control-plane `REOBSERVE/ABSTAIN` |

单手与 bimanual 是 **完整性结论**，不是看到几条正 edge后的默认分类。

### 4.2 alternating 不是第六种 actor code

交替由 event clock 表达：

```text
LEFT-only event -> RIGHT-only event
RIGHT-only event -> LEFT-only event
```

若两手 contact interval 有重叠，重叠段必须是一个 bimanual macro event：

```text
LEFT-only -> BIMANUAL -> RIGHT-only
```

不能把同一 event 内 actor 从 left 瞬时改成 right 而不切 boundary；否则 duration、contact relation和 effect无法被唯一解释。短 handover 仍可用 interval boundary，边界不确定时用 interval/censor likelihood。

### 4.3 状态转换

令每个 event 的编译后 hand topology 为：

```text
N = no hand
L = left only
R = right only
B = bimanual
U = one or more HAND_UNRESOLVED edges
X = incomplete, action-invisible control state
```

允许的 observation/compiler transition：

```text
N -> N | L | R | B | U | X
L -> L | B | N | U | X
R -> R | B | N | U | X
B -> B | L | R | N | U | X
U -> U | L | R | B | N | X
X -> X | N | L | R | B | U   # 只有新证据/新 observation 才能退出 X
```

这不是新增 action field；它由 edge bundle与 completeness receipt机械派生，用于 validator、episode audit和 safe reobserve。`X` 永不送 action。

## 5. Action-visible 与 action-invisible 的冻结边界

### 5.1 进入 finite grammar

保持 standard-r1 的 node/event/edge字段，只扩 actor：

```text
node:
  observability, center_cell, scale_band, orientation_2d

event:
  kind, focus_node, onset, duration, object_effect, outcome

edge:
  actor = ABSENT | LEFT | RIGHT | OBJECT_OBJECT | HAND_UNRESOLVED
  primary_node, primary_part, primary_surface_cell,
  secondary_node, relation_before, relation_after, contact_mode
```

其中：

- `ABSENT` 仍由 `actor=0` 表示；
- `HAND_UNRESOLVED` 是 present edge，所有 required payload照常满足；
- `RIGHT_CENSORED` duration与 active onset仍按 v2现有语义；
- field-specific reviewed unknown（part/cell/relation/mode/effect等）仍用现有 UNKNOWN code；
- event、edge与node的 MISSING 不进入 wire。

### 5.2 只作 loss mask/provenance

以下字段永不送 action、packet embedding或 action cache：

- hand observation state、tracker state、hand detector score；
- annotator/reviewer、confidence、coverage、evidence tier；
- hand/local track ID、MANO/mesh/joints、continuous wrist/landmark；
- imputed side/pose/contact、teacher logits/hidden/version embedding；
- missingness propensity、PU class prior、MNAR latent、inverse-propensity weight；
- pair ID、CAPOT coupling、human frame、raw event table；
- boundary lower/upper exact frame与censoring receipt。

这些可以产生 planner-only auxiliary、loss weight或审计指标，但不得经 state token、KV/cache或 hidden concatenation绕过 hard packet。

## 6. Loss 与 mask contract

### 6.1 通用字段 loss

对字段 `f` 定义：

```text
m_f = 1  only if field is KNOWN or REVIEWED_UNKNOWN with a schema UNKNOWN code
m_f = 0  for NOT_REVIEWED, inapplicable, inadmissible oracle target
w_f = m_f * confidence_for_loss_only
```

每字段独立归一：

```text
L_f = sum_i w_if * loss_if / max(1, sum_i w_if)
L_event = mean over fields with positive denominator
```

不得用所有 dense cell 的总元素数归一，否则常见的 node/part字段会压过稀有 bimanual、outcome和censor。

### 6.2 actor / presence

- present positive edge：只要 hand edge存在且 primary reference已解析，presence可监督；
- actor known left/right/object-object：普通 CE；
- side reviewed ambiguous：target=`HAND_UNRESOLVED`，普通 CE；
- side not reviewed：actor mask=0，不用 `HAND_UNRESOLVED`补；
- absent edge negative：仅当 event edge coverage与 hand roster都 `EXHAUSTIVE`；
- `POSITIVE_ONLY`/`NOT_REVIEWED` tail：presence mask=0。

因此 `HAND_UNRESOLVED` 是“已知正例的未知类别”，MISSING 是“没有 target”，ABSENT 是“经完整复核的负例”。三者梯度不同。

### 6.3 boundary 与 censor

在固定 discrete time bins 上：

- exact boundary：对对应 bin 做 event-time NLL；
- interval `[l,u]`：`-log sum_{b=l..u} p(T=b)`；
- right censor at `c`：`-log P(T>c)`；
- left censor at `c`：`-log P(T<=c)`，action compiler只把当前 event写成 `ACTIVE`；
- boundary unknown/not-reviewed：对应 boundary mask=0；
- visibility interruption不自动生成 censor；只有 annotation coverage 边界截断真实 onset/offset才是 censor。

[Nnet-survival](https://arxiv.org/abs/1805.00917) 展示了 discrete-time hazard 对 right-censored event time 的最大似然训练。这里借用 censor-aware likelihood原则，不主张生存分析方法新颖性。

### 6.4 grip proxy loss 保持原 contract

`hand_status` 只训练 dimensions 18/19 的 open/closed proxy：

```text
left grip weight  = left_active[target]  * 0.25
right grip weight = right_active[target] * 0.25
inactive grip weight = 0
```

它不能参与：

- contact presence/mode；
- hand involvement；
- side ambiguity；
- acquire/release boundary；
- bimanual event truth。

`ego_hand_active_transition_pair` 仍只作 record-only，除非未来有独立、预注册的 proxy-transition auxiliary。

### 6.5 missingness auxiliary 的唯一合法位置

可以给 planner 加 action-invisible observation/missingness head，预测：

```text
observation_state / reviewability / packet_abstain probability
```

用途仅限 calibration、reobserve policy与分层诊断。其 hidden/logits不进入 action。若要用 propensity/IPW/PU risk改善主 field loss，必须先在 fully observed audit subset证明：

1. propensity 有 overlap，极小概率不过度放大；
2. SCAR/SAR/MNAR假设至少有可检验支持；
3. calibration split 与 policy test隔离；
4. 结果对 clipping、class prior与selection model稳定。

在这些条件之前，主推荐是 masked supervised likelihood，不是 propensity correction。

## 7. 容量影响：+2.493144 bit，不增加 wire bytes/tokens

### 7.1 actor 扩展后的 exact standard grammar

保持 standard-r1 的其它字典、node union、两 edge canonical set、kind path与 onset order不变。给定 `n` 个 present nodes，primary字段基数仍为：

```text
n primary nodes * 9 parts * 17 cells * 6 modes = 918*n
```

原 actor bundle含两种 hand actor与一种 object-object actor：

```text
E_n = 918*n * [2*(1+81*(n-1)) + 81*(n-1)]
    = 918*n * [2 + 243*(n-1)]
```

新增 `HAND_UNRESOLVED` 后，hand-like actor为三种：

```text
E'_n = 918*n * [3*(1+81*(n-1)) + 81*(n-1)]
     = 918*n * [3 + 324*(n-1)]

E'_n for n=1..4
= [2,754, 600,372, 1,792,854, 3,580,200]
```

继续使用：

```text
B'_n = 1 + E'_n + C(E'_n,2)
A'_n = 88*n*B'_n
```

并沿 schema memo 的 kind path、onset nondecreasing、node union inclusion-exclusion公式复算，得到：

```text
S_action_missing_hands_r1
= 1947227281526202347439908176679818858538657207150616941292393985

log2(S_action_missing_hands_r1)
= 210.24289126401328 bit

ceil(log2 S) = 211 bit
```

相对原 standard-r1 `207.74974691951957 bit`：

```text
delta = +2.4931443444937145 bit
```

Q8 planner-state grammar同样复算为：

```text
S_Q8_missing_hands_r1
= 147057422931374437915990191856335851511669256420591174150894993638617557614238987660713696861689227787196409144611799755275052821874959097369899521

log2(S_Q8_missing_hands_r1)
= 485.5578814616901 bit

ceil(log2 S_Q8) = 486 bit
delta_Q8 = +6.648491902354294 bit
```

### 7.2 工程宽度不变

actor 本来就是一个 `uint8` code；其 alphabet 从4个 wire codes扩为5个，不增加字段数：

```text
node_code  [4,4] = 16 bytes
event_code [3,6] = 18 bytes
edge_code  [3,2,8] = 48 bytes
total                 82 bytes
tokens                13
```

需要改变的是 actor embedding/head输出宽度、validator、capacity receipt、rank/unrank与 equal-bit permutation—not wire layout。

若未来再新增 `hand_scope`、visibility、hand count、actor instance或confidence action field，上述整数立即失效，必须重新枚举。当前唯一推荐明确禁止这些追加。

## 8. canonical `e212afec` 的未来插入面（本轮不实现）

### 8.1 Schema / compiler

- 新建的 `world_action_model/event_tube_schema.py` 中把 edge actor固定为五码；`HAND_UNRESOLVED` 是 present actor。
- `validate_event_tube_packet_v2()` 检查 absent zero、edge prefix/sort/unique、node reference、complete packet receipt与 `HAND_UNRESOLVED` payload。
- `compile_action_event_tube_v2()` 同时读取 event/edge targets和 hand-roster completeness；partial fields只生成 planner masks，不生成 oracle packet。
- 旧 `world_action_model/contact_transition_schema.py:65-105` 的 v1.1 三 actor冻结，不原地升级；22D baseline保持 bitwise兼容。

### 8.2 Dataset / sidecar

- 沿 `world_action_model/datasets/wam_lerobot_dataset.py:285-325,741-819` 的 strict SHA sidecar模式新增 episode sparse `event_hand_observations_v1` 与编译 receipt。
- 现有 hand-active loader `:326-367,821-868` 保持独立；不得把 tracker mask join成 event contact truth。
- 每个 replan anchor从 sparse episode events机械编 target；不把全 episode hand table、track ID或imputation复制成 frame-level action输入。

### 8.3 Planner / action route

- `transformer_wa_casual_mot.py:599-631,703-870`：v2 opt-in config、actor head/embedding vocabulary与receipt；v1 disabled路径不变。
- `:1253-1285` 邻近的 v2 planner输出 factorized actor logits与 packet-abstain auxiliary；planner hidden不导出。
- `:1287-1403`：predicted route做 hard decode/validate；invalid/incomplete不 soft fallback，返回 control-plane reobserve。
- `:1405-1495`：13 tokens只来自 canonical packet；missingness reason、confidence、tracker state与imputer hidden不进入 action。
- `:1531-1590`：GT只传 loss侧或显式 oracle route；禁止现有 `:1287-1333` 式 field-wise oracle/prediction merge用于 v2 predicted policy。
- `:127-194,1806-1822`：strict v2继续切断 action/state/packet对 reference与 image cross-attention旁路。
- `:2231-2499`：action cache只存 canonical packet/receipt与允许的 state/text/action KV；不存 missingness logits、imputed pose或 raw tracker values。

### 8.4 Trainer

- 沿 `wa_casual_trainer_mot.py:111-189` 的 complete-group CE原则新增 actor五类 CE、event boundary survival NLL、per-field normalization与 packet-admission metrics。
- `:957-1055` 的 batch/loss入口增加 hand-roster coverage、review state、observation reason与censor target；predicted action forward不接 GT condition。
- `hand_status` action MSE继续走现有 per-side active weight，不和 event actor loss相乘或互相补全。

## 9. 必须比较的 baseline

所有 WAM route共享 backbone、数据、split、action horizon、token/FLOP预算；v1/v2/F-PCD/continuous route互斥。

| baseline | 目的 | 允许/禁止 |
|---|---|---|
| `complete-case-only` | 检查只用双手可见样本的 selection bias与样本损失 | 只训练 fully reviewed；必须报告 coverage shift |
| `mask-only-r1` | 原 actor无 unknown，ambiguous side全部 actor mask | 不把 ambiguous变 absent |
| `missing->ABSENT` | 负面对照，量化常见错误 | 明确标为错误 baseline，不能成为生产方法 |
| `last-observation-hold` | 负面对照，量化时序填充偏差 | imputed值标 `MODEL_PSEUDO`，不称 GT |
| `model-impute-pseudo` | 检验 HandOccNet/VLM/MANO proposal是否只提高表面coverage | teacher/version隔离；只作 weak auxiliary |
| `nnPU / SAR-PU` | presence selection-bias敏感性 | 只在假设/propensity receipt下运行；不 author GT |
| `reason-coded mask` | 本 memo 推荐，但不加 `HAND_UNRESOLVED` | 检验 observation reason auxiliary本身 |
| `reason-coded + HAND_UNRESOLVED` | 唯一推荐 contract | exact 210.242891-bit route |
| `equal-bit-210.242891` | 匹配新 grammar容量的无语义对照 | 全 packet rank/permutation；NULL固定 |
| `sensor/mocap oracle` | fully observed upper bound | HOT3D/ARCTIC或自采传感 truth；域差异单报 |
| `continuous hand trajectory` | 信息上界 | 互斥 action route，不与 SM-FET concat |

关键比较不是“imputation提高了多少标签量”，而是：在相同 observed truth 下，是否降低 false absence、改善 actor/bimanual calibration，并最终改善机器人策略且不靠伪标签泄漏。

## 10. Benchmark：从数据契约到动作结果

### 10.1 B0：现有 66 Ego 只做 observation audit

已有 hand-active sidecar用于：

- 报告 per-side raw/current/future validity与 episode-zero集合；
- 交叉表 `tracker_state × observation_state × event phase × adjudicated participation`；
- 检查 tracker inactive 是否在 acquire/hold/manipulate、side、object/viewpoint上非均匀；
- 证明 inactive不等于 not involved。

它不能给 contact/side/bimanual GT，也不能因 66 段数较大而替代人工 adjudication。

### 10.2 B1：3 calibration + 12 blind human-pour

三段 calibration 必须覆盖：

1. 清晰单手；
2. overlap/bimanual或左右交替；
3. object occlusion + out-of-frame + interval/censor + tracker inactive 的 missingness困难段。

12 blind由 A/B 独立标 `events/node/edges/event_hand_observations/pose-probe`，第三人裁决。除既有 D1 指标外新增：

| 指标 | 门槛 |
|---|---:|
| participation macro F1 | `>=0.80` |
| side三类 κ（left/right/ambiguous） | `>=0.70` |
| observation-state κ | `>=0.70` |
| bimanual event F1 | `>=0.80`，若支持不足则 `NOT_EVALUABLE` |
| false-ABSENT under non-exhaustive coverage | `0` |
| censor-vs-occlusion confusion | `<5%` |
| actor/hand roster incomplete packet误 admission | `0` |
| critical reviewed unknown | `<=40%` |
| unresolved/invalid/overflow 各自 | `<=5%`；类别支持不足时另报 |

单任务 3+12 只解锁实际出现字段；没有足够 bimanual/ambiguous样本时不能用“零错误”宣称通过。

### 10.3 B2：missingness benchmark 最小规模

若要把本 contract作为独立 benchmark，最低建议沿 portfolio 的 Idea D 预算：

```text
>=300 adjudicated events
>=50 confirmed bimanual events
>=30 production groups
sensor/mocap-confirmed fully observed subset
自然 occlusion/out-of-frame/inactive strata
```

需要两种测试：

1. **natural missingness**：按真实 observation reason、side、phase、object、viewpoint和group分层；
2. **controlled hiding**：只在 fully observed sensor/mocap subset人工遮挡/crop/censor，真值来自原 fully observed capture，而不是模型补全。

controlled hiding用于因果 stress，不替代 natural MNAR评估。

### 10.4 B3：36 human + 36 target-robot 同任务 pilot

沿 final architecture 的 pick-place / stabilize-twist-cap / pour-transfer 三 family、12 cells 收集。每个 family必须同时含：

- confirmed left-only、right-only、bimanual与 alternating sequence；
- role reversal或 infeasible assignment；
- occluded/out-of-frame但由 sensor/另一视角确认的子集；
- no-hand/object-only与 hard negative；
- target robot native contact/outcome truth。

human side不字面映射 robot side；Idea B 的 allocator读取 functional event、target proprio与hardware capability。`HAND_UNRESOLVED` 不授权盲目执行：若目标 allocator不能唯一求解，必须 safe reobserve/hold。

### 10.5 报告指标

监督层：

- per-field coverage、KNOWN/REVIEWED_UNKNOWN/NOT_REVIEWED率；
- actor五类 confusion、macro F1、PR-AUC；
- exhaustive-negative false omission与 false-absence rate；
- bimanual event F1、alternating sequence edit/F1；
- boundary interval NLL、censor calibration、interval coverage与 width；
- risk-coverage curve：abstain比例对 actor/contact错误；
- 按 observation reason、tracker state、phase、side、group的所有指标。

策略层：

- oracle packet vs predicted packet；
- task/phase success、first-contact target、release timing、unintended contact；
- single-hand/bimanual/alternating分层成功率；
- missingness reason分层的 safe hold/reobserve、watchdog与恢复率；
- proposed相对 mask-only、equal-bit、continuous upper bound的cluster CI；
- fixed packet替换 reference、missingness metadata或imputer输出时，pre-safety action应保持 `max_abs<=1e-6`。

## 11. CPU-only contract tests（未来实现前）

本轮不写测试代码；未来至少冻结以下不变量：

1. `ABSENT/NOT_INVOLVED/MISSING/HAND_UNRESOLVED` 四者 round-trip不互换；
2. occluded/out-of-frame/inactive 不改变已知 semantic target，只改变 reason/mask；
3. non-exhaustive tail永不产生 presence negative；
4. left exhaustive不授权 right negative，反之亦然；
5. ambiguous reviewed actor产生 `HAND_UNRESOLVED` CE，unreviewed actor梯度为零；
6. known left + unreviewed right不能生成 oracle `left-only` packet；
7. 同 event left+right edge派生 bimanual；左右连续且无 overlap派生 alternating；有 overlap必须有 bimanual中间段；
8. occlusion不自动变 censor；chunk coverage截断才产生 censor；
9. tracker inactive只影响 grip proxy weight，不影响 contact/event actor；
10. model proposal/imputation带 teacher provenance，且不能出现在 GT receipt；
11. actor五码、`S_action`、`210.24289126401328`、Q8容量与receipt固定；
12. equal-bit rank/unrank覆盖新 reachable alphabet且NULL固定；
13. 82-byte wire与13-token layout不变；
14. incomplete/invalid packet不 soft fallback，只产生 safe control code；
15. action/cache中不存在 observation reason、tracker、confidence、imputer hidden/logits；
16. predicted route改变任意 GT annotation bytes不改变 action；oracle route必须显式命名。

## 12. 停止条件

### 12.1 立即停止 annotation/schema 扩量

以下任一发生，协议最多修订一次；仍失败则 SM-FET human-hand contract No-Go：

- 3+12 participation F1 `<0.80`，side/observation κ `<0.70`；
- false-ABSENT、non-exhaustive negative或 incomplete-packet admission不为零；
- censor与occlusion混淆 `>=5%`；
- critical reviewed unknown `>40%`；
- unresolved/invalid/edge overflow持续 `>5%`；
- 双标者无法稳定区分 `NOT_INVOLVED` 与 `NOT_REVIEWED`；
- 两手未审样本仍必须靠人工猜 side 才能满足 validator。

### 12.2 停止 actor 扩展

- `HAND_UNRESOLVED` 在有足够样本的 gold set上 κ/F1不达0.70/0.80；
- 它主要吸收 `NOT_REVIEWED` 而非 reviewed ambiguity；
- allocator看到该码后只能等价随机选 side，且 reobserve不能降低风险；
- mask-only-r1在相同 coverage下不劣于它，且错误/abstain calibration无改善。

此时删除 `HAND_UNRESOLVED`，回到 strict mask+abstain；不能用更大 unknown latent补救。

### 12.3 停止 finite action route 主张

沿 final architecture 的 D3/D4：

- oracle v2不优于22D-only，或相对 full trajectory下降超过3 pp；
- 只有 imputed/continuous MANO/pose action-visible才有效；
- equal-bit-210.242891与semantic route持平且利用率相当；
- predicted route不能恢复至少70% oracle gain；
- fixed packet后 action仍随 reference、missingness reason或imputer hidden改变；
- bimanual/alternating子集上 incomplete packet导致 unsafe action而非可校准 abstain。

这些都应停止 headline claim，不事后增加 visibility、hand count、confidence或latent code保护结果。

## 13. 论文主张边界

可以主张：

> 在严格有限 WAM route 中，将 confirmed absence、reviewed side ambiguity、field missingness、tracker validity与boundary censor分离，能否减少 egocentric bimanual supervision 的 false negatives，并在保持可审计容量的同时改善 calibrated policy behavior。

不能主张：

- 首次检测 left/right/bimanual affordance；
- HandOccNet/MANO/VLM补出了真实被遮挡手；
- MNAR模型识别了不可识别的 contact truth；
- `hand_active=0` 表示 inactive hand或无物理接触；
- 模型预测、插值、PU posterior或pseudo-label是 ground truth；
- 现有66 Ego已经提供连续双手 contact/pose truth。

## 14. 最终一句话

Ego “缺一只手”不是一个值，而是一个需要原因、coverage、review state与时间边界的观测过程。SM-FET 的唯一可守 contract 是：**对确认不存在才给 ABSENT，对确认存在但侧别不可判才给有限的 `HAND_UNRESOLVED`，对遮挡/出画/inactive/未审只给 mask与provenance，对 chunk边界给 censor likelihood，对所有不完整 roster拒绝 oracle packet；任何模型补全永不升级为 GT。**
