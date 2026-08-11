# SM-FET 连续事件标注与部分人机配对协议 v1

- 日期：2026-08-12
- 工作树：`codex/temporal-event-pairing-meta-20260812`
- 协议状态：设计稿；未实现 UI/CSV checker，未生成 GT，未训练，未调用 GPU
- 目标：用最小人工字段验证“连续多姿态事件 + partial human–robot pairing”是否可标、可复核、可拒绝错误配对
- 非目标：修改 `contact_segment_v1_1`、构造 MANO/透明物体 3D 真值、训练 CAPOT/WAM、证明目标硬件收益

## 0. 决定摘要

本协议采用一个**跨任务固定的八状态骨架**：

```text
approach -> precontact -> acquire -> hold/manipulate
                                -> regrasp -> manipulate
                                -> release -> verify_outcome
```

固定的是事件语义、边界规则和字段字典；任务条件化的是允许的 skip/loop、角色、部位、主关系谓词和预期 effect，不能为每个任务新造 event kind。`actuate` 统一标为 `manipulate`，实际效果写入 `object_effect`；`effect` 本身不是一个任意时长的动作状态，效果已发生后的检查阶段标为 `verify_outcome`。这样既保持 F/meta 的 Semi-Markov 主线，也吸收 H 的 pre/post relation 与 effect，而不再建立第二套图模块。

最小人工产物是四张 versioned 表：

1. `events_v1.csv`：episode 级事件、边界、duration 可推导量、active hand、effect/outcome、observability；
2. `event_edges_v1.csv`：每个事件至多两条 contact/relation edge，引用但不替换现有 contact segment；
3. `pose_keys_2d_v1.csv`：事件入口/出口及必要 turning point 的稀疏 2D functional pose；
4. `event_pair_v1.csv`：human event 与 robot event 的 reviewed 多对多边、显式 unmatched/null，以及与 GT 分离的 solver mass。

当前结论是：

- **Go**：3 段指南校准、12 段盲法 A/B+裁决、旧数据 hard-negative rejection 设计、36+36 同任务补采设计；
- **Conditional Go**：只有标注 Gate A 通过且 36+36 的 group/outcome provenance 完整，才建立 `event_pair_v1` gold benchmark；
- **No-Go**：把自动 proposal、OT coupling、MANO、`hand_status`、透明液体猜测当 GT；把现有 human-pour 与旧 robot-grab 当正 pair；当前训练/GPU/真机。

## 1. 与现有资产的兼容边界

本协议以以下只读事实为约束：

- F/H/meta 已把主粒度收敛为 episode event clock 上的最多八个 macro event；continuous pose 只允许 training-only auxiliary/probe。只读输入分别位于 sibling worktree `../gwp-agent-idea-f-temporal-tube/docs/research/idea_f_temporal_contact_pose_tube_review_20260812.md`（commit `20421c7`）、`../gwp-agent-idea-h-event-transition/docs/research/idea_h_event_state_transition_wam_review_20260812.md`（commit `8f0432a`）和本分支 `docs/research/temporal_event_pairing_wam_meta_synthesis_20260812.md`。
- G 已明确 partial/monotonic/OT 本身不是 novelty，且旧 66 Ego pour 与 88 dual-Inspire grab 没有 positive overlap；只读输入为 `../gwp-agent-idea-g-partial-pairing/docs/research/idea_g_partial_human_robot_alignment_review_20260812.md`（commit `9f60f04`）。
- 现有正式 A/B packet 使用 processed 20 Hz 半开 contact interval、field-level unknown/censor、`coverage_mode` 和第三人裁决；其 12 段是空白任务而非 GT，见 `track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/README.md`、`LABEL_DICTIONARY.md` 与 `track_a/contact_labels_v1_1/ADJUDICATION_PROTOCOL.md`。
- op34 UI 是 `contact_segment_v1_1` 的精确 7 文件 packet 填写器，HTML 和 preparer 都锁死 header、episode 集、allowlist、UI build 和 packet schema；直接往当前 packet 加 CSV 会 fail-closed，见 `tools/contact_annotation_ui/contact_annotation_ui.html:413-436` 与 `scripts/contact_labels/prepare_offline_contact_annotation_ui.py:27-76,323-366,369-512`。
- op29 证明现有 66 段缺 `subject/session/operator/source instance/target instance/liquid/batch`，四个 continuity block 只可作 temporal diagnostic，不能作 production group GT；补录必须有 authoritative provenance 与 attestation，见 `docs/audits/ego_66ep_group_metadata_recovery_20260811.md` 与 `world_action_model/ego_group_metadata_recovery.py:18-67,209-390`。

因此本协议是**新的 sibling annotation packet 设计**，不向 v1.1 CSV 静默加列，不把当前 UI 称为已支持 SM-FET，也不改 canonical。

## 2. 事件本体：固定骨架，任务条件化谓词

### 2.1 八类固定 event kind

| `event_kind` | 进入边界 | 退出边界 | 接触要求 | 关键排歧 |
|---|---|---|---|---|
| `approach` | active effector 开始持续、任务定向地接近 primary role entity | 进入可执行接触前对准区，或直接首次接触 | 应为无接触；不确定则 contact 字段 unknown | 普通走动/抬手不是 approach；只有可判断目标角色后才标 |
| `precontact` | 已进入接触前对准/微调区 | 首次可确认 task-relevant contact，或放弃接触 | confirmed no-contact | 2D overlap 不能自动算接触；短暂停顿仍可属于 precontact |
| `acquire` | 首次 task-relevant contact | 建立稳定、可操控的 grasp/engagement，或 acquisition 失败 | 至少一条 contact edge 从无到有 | `hand_status=closed` 不是进入边界；impact-only 任务可 skip |
| `hold` | 已稳定接触但尚未制造主任务 relation/effect change | 开始 task-relevant actuation、regrasp 或 release | 通常 sustained/stick | 单纯稳定、等待、对准属于 hold；有主效果变化则转 `manipulate` |
| `manipulate` | contact configuration 开始驱动预注册任务效果 | 效果停止/达成、进入 regrasp 或 release | 可为 stick/slide/roll/impact/object-object | 用户语义中的 `actuate`；effect 写字段，不新建 `actuate_*` 类 |
| `regrasp` | 原稳定接触拓扑开始有意解除、迁移或重新分配 | 新接触拓扑重新稳定，或失败转 release/NULL | 接触可能短暂消失；edge/part/mode 应有变化 | 仅腕姿变化、未改变接触拓扑，不算 regrasp；允许重复 |
| `release` | 开始有意终止最后一条 task-relevant contact | confirmed separation 或 coverage 结束 | 从 sustained 到 separated；边界可右删失 | effect 已发生不等于 release；失败掉落需 outcome=failure |
| `verify_outcome` | 主效果已可观察，且执行进入检查/稳定等待 | outcome 判据满足、失败被确认或 episode 结束 | active hand 可为 none；可保留支撑 contact | effect 可在 manipulate 中发生；本事件只表示结果可被判断 |

固定 transition graph 允许：

- skip：如 episode 进入画面时已经 hold，可从 left-censored `hold` 开始；impact 任务可跳过 `acquire`；
- self-loop：长 hold/manipulate 不按固定窗口重复造 event；仍保留一个 global `event_id`；
- repeat：`manipulate -> regrasp -> manipulate` 可重复；
- early termination：失败/放弃可从 precontact/acquire/regrasp 进入 release 或 episode-end NULL；
- 双手并发：一个 macro event 用 `active_hand=both`，并在最多两条 edge 中分别记录左右接触；不为左右手各造一条互相重叠的宏状态序列。

禁止做法：按任务发明 `pour/turn/insert/wipe` event kind；按每个 48-frame window 重编号事件；把 preparation、等待、镜头调整强行塞进 `approach`；为不可见边界补一个“最像”的帧。

### 2.2 任务条件化内容

每个 `task_family` 在标注前冻结一个 `task_ontology_version`，只定义：

- 允许/要求的 transition 与合法 skip；
- primary/secondary functional role 和 part 字典子集；
- 该任务的主 `relation_before -> relation_after` 判据；
- 可判的 `object_effect`、`outcome` 与传感/人工判据；
- 哪些字段在该任务中必须 reviewed，哪些可始终 mask。

任务条件文件不能改变八种 event kind、unknown/censor 语义和 A/B 门槛。它是 coordinator-provided dictionary，不是模型 proposal，也不能在 test 结果出来后修改。

### 2.3 边界、duration、coverage 与 unknown

所有事件使用 source episode 的稳定时钟。每行先保存一个保守、半开的 confirmed-positive support interval
`[support_start_frame,support_end_frame)`；它表示这些帧可确认属于该事件，不声称两端就是真实 transition。一个物理边界不是强制单帧，而是：

```text
start boundary support = [start_frame_lo, start_frame_hi]
end boundary support   = [end_frame_lo, end_frame_hi]
```

边界状态固定为：

- start：`exact | interval | left_censored | unknown`；
- end：`exact | interval | right_censored | unknown`。

规则：

1. `exact` 时 `lo=hi`；`interval` 表示真实 transition 可确定在闭区间内，不能挑中点冒充 exact；
2. `left_censored` 只表示真实 onset 在 coverage 起点前；`right_censored` 只表示真实 termination 在 coverage 终点后；遮挡不等于 censor；
3. `unknown` 表示现有证据无法给真实边界 support，此时对应 boundary `lo/hi` 留空，但保守 confirmed-positive support interval 仍保留；相应 boundary likelihood/mask 为零，已知 kind/edge/effect 不被清除；
4. event duration 不由标注员再填一个 float。两端 observed 时机械推导：

```text
duration_lower = max(0, end_frame_lo - start_frame_hi)
duration_upper = end_frame_hi - start_frame_lo
```

5. censored duration进入 interval/survival 统计；不得把 coverage 边界当真实 duration；
6. `coverage_mode=exhaustive` 才允许未标事件/edge成为 absence negative；`positive_only` 的空白仍是 unknown；
7. `visibility`、`existence`、`boundary censor` 与 `field unknown` 四者分开。看不见不是不存在，unknown 不是 NULL。

### 2.4 Active hand、contact、pre/post relation 与 effect

`active_hand` 固定为 `left | right | both | object_only | none`，未知留空。它描述该 macro event 的任务主动方，不直接进入 human–robot OT cost；human left/right 与 RH56/OmniPicker left/right 不做字面对齐。

每个 event 至多两条 reviewed edge。edge 记录：

- `effector_side`：`left/right/object_object`；
- primary entity/role/part/surface cell；
- 可选 secondary entity/role/part/surface cell；
- `contact_mode`：空值或 `none/stick/slide/roll/impact`；
- `relation_before/relation_after`：`separate/near/touching/overlap_2d/inside/contains/aligned_to_rim/supported_by`，未知留空；
- 现有 `contact_segment_v1_1.segment_id` 引用；没有合格 contact row 时留空，不能反向伪造；
- edge visibility、evidence tier、confidence。

`object_effect` 固定为 `none/translate/rotate/articulate/flow/deform/transfer`，未知留空；`outcome` 固定为 `success/partial/failure`，未知留空。二者都必须附 task-specific observable/sensor criterion：

- RGB 只能确认可观察关系变化；
- 秤、液位、编码器、触觉/力等可作 sensor-confirmed outcome；
- 透明液体体积、流速、物理接触面不可判时留空；
- `release` 不自动推导 success，`effect observed` 也不自动推导 intended outcome。

## 3. 最小字段表

### 3.1 `events_v1.csv`

| 字段组 | 必填字段 | 规则 |
|---|---|---|
| 身份 | `schema_version,dataset_id,episode_id,coverage_id,event_id,event_order` | `event_id` 全 campaign 唯一；`event_order` 按 adjudicated support/onset 后机械生成 |
| 任务 | `task_family,task_ontology_version,event_kind,active_hand` | event kind 固定八类；active hand 未知留空 |
| 时间与边界 | `support_start_frame,support_end_frame,start_frame_lo,start_frame_hi,start_boundary,end_frame_lo,end_frame_hi,end_boundary` | support 为保守 positive 半开区间；exact/interval/censor/unknown 约束如上；duration 不重复人工填写 |
| 状态 | `object_effect,outcome,event_observability` | observability=`visible/partial/occluded/out_of_frame`；未知留空 |
| 证据 | `confidence,evidence_scope,evidence_tier` | scope=`rgb_direct/rgb_temporal/sensor`；tier=`human_independent/human_adjudicated/sensor_confirmed/model_pseudo` |
| 工作流 | `annotation_role,annotator_id,review_status,label_origin,parent_event_ids,notes` | A/B ID 隔离；proposal ID 不得进入 human-independent row |

`duration_lower/upper`、event count、Q overflow、transition validity 全部由 checker 生成，不作为人工源字段。

### 3.2 `event_edges_v1.csv`

| 字段组 | 必填字段 | 规则 |
|---|---|---|
| 身份 | `edge_id,event_id,edge_order` | `edge_order` 只允许 0/1；超过 2 条标 event-level overflow，不静默截断 |
| 参与者 | `effector_side,primary_entity_id,primary_role,primary_part,primary_surface_cell` | instance/track 不明可留空，但 role/part 不得凭任务名猜 |
| 第二关系 | `secondary_present,secondary_entity_id,secondary_role,secondary_part,secondary_surface_cell` | absent 时 secondary 字段 literal empty；不能藏字符串 |
| 接触/状态差 | `contact_segment_id,contact_mode,relation_before,relation_after` | contact row 与 event row 是引用关系，不复制/改写 v1.1 contact truth |
| 缺失与证据 | `edge_visibility,confidence,evidence_scope,evidence_tier` | 每条 edge 独立 unknown/mask |
| 工作流 | `annotation_role,annotator_id,review_status,label_origin,parent_edge_ids` | 与 event 表同一身份/session receipt |

### 3.3 `pose_keys_2d_v1.csv`

每个存在 active hand 且手可定位的 observed event，对每只 active hand 默认标两个稀疏姿态：`entry` 和 `exit`；仅当 event 内存在方向反转、最大开合、contact part/mode 改变或 regrasp 时增加一个 `turn`。`active_hand=none/object_only` 不要求 hand pose。持续不足 3 帧且无法分出两姿态时只标 `anchor`，不得复制同一点充数。

| 字段组 | 必填字段 | 规则 |
|---|---|---|
| 身份/时间 | `pose_key_id,event_id,pose_role,frame_index,hand_side` | role=`entry/turn/exit/anchor`；frame 必须是实际查看的 source-clock 帧 |
| 参照物 | `reference_entity_id,box_x0_norm,box_y0_norm,box_x1_norm,box_y1_norm` | bbox 不可定位时留空，对象局部量 mask |
| 7 个 functional landmarks | `wrist,palm_center,thumb_tip,index_tip,middle_tip,ring_tip,little_tip` 各自的 `x_norm,y_norm,visibility` | visibility=`visible/self_occluded/object_occluded/out_of_frame/motion_blur`；不可见坐标留空，禁止想象补点 |
| 人工离散姿态 | `aperture_band,grasp_family,palm_axis_bin` | aperture=`closed/narrow/medium/wide`；family=`open/power/pinch/lateral/hook/support/tool_precision/other`；axis=8方向 bin；未知留空 |
| 证据 | `confidence,evidence_scope,annotation_role,annotator_id,review_status,parent_pose_key_ids` | tracker/MANO 只能另记 `model_pseudo` proposal，不能覆盖 human row |

不额外人工标“连续轨迹”。以下连续变化全部由同一 event 的 ordered pose keys 机械计算，避免多填相互矛盾字段：

- image-normalized 和 object-box-normalized wrist/palm `Δx,Δy`；
- fingertip centroid、finger spread 与 thumb-index aperture 的变化；
- palm axis 的最小环向变化；
- 入口→turn→出口的折线长度、净位移、曲折率与方向反转；
- grasp-family/aperture/contact-part/contact-mode 的离散 transition；
- 可见 landmark 比例和 coordinate support mask。

这些量足以检查“event-aligned 多姿态”是否比 single grasp/uniform K 保留过程变化，不依赖 MANO、metric depth、透明物体重建或逐帧 21 joints。自动 optical flow/tracker 可用于第二套 proposal-only benchmark，但盲标 A/B 不可见。

### 3.4 `event_pair_v1.csv`

一行是一条 candidate episode pair 内的 event edge；同一 human event 或 robot event 可出现在多行，因此天然 many-to-many。NULL 不建一个假事件 ID，而用 one-sided unmatched row。

| 字段组 | 必填字段 | 规则 |
|---|---|---|
| pair 身份 | `schema_version,pair_family_id,candidate_pair_id,split_id,task_family,candidate_label,pair_reason_code` | label=`candidate_positive/hard_negative/easy_negative/unknown`；pair 在 split 后生成且不跨 split |
| human 端 | `human_dataset_id,human_episode_id,human_event_id,human_rank` | `robot_unmatched` row 时 human event 为空；其它相应必填 |
| robot 端 | `robot_dataset_id,robot_episode_id,robot_event_id,robot_rank` | `human_unmatched` row 时 robot event 为空；两端不得同时为空 |
| edge 语义 | `edge_type,gold_edge_label,effect_match,monotone_status` | type=`event_event/human_unmatched/robot_unmatched`；label=`match/nonmatch/unmatched/ambiguous/unreviewed`；effect=`yes/no/unknown`；monotone=`compatible/crossing/not_applicable` |
| 两种权重 | `review_weight,solver_mass` | review weight 是 reviewed 可靠性/训练权重；solver mass 是模型/算法输出，**永远不是 GT** |
| 置信/来源 | `edge_confidence,evidence_tier,annotation_role,annotator_id,adjudication_status,provenance_id` | solver row 也必须指向 solver receipt；人工与自动来源不可混写 |

验证规则：

1. `event_event` 两端均有 ID；unmatched row 恰有一端 event ID且 `gold_edge_label=unmatched`；绝不使用 `event_id=NULL` 伪造普通 event；
2. reviewed `match` 可以 one-to-many/many-to-one，但所有正边必须满足 non-crossing：若 `h_i<h_k`，不得有 `r_j>r_l` 的两条正边；并列 rank 允许；
3. `nonmatch` 的 `solver_mass` 必须为 0；`ambiguous/unreviewed` 不进 gold PR/F1，也不当 negative；
4. `review_weight` 不被解释为真实 transport probability。推荐由 adjudicated edge confidence 与两端 field coverage 机械生成，并保留未归一化值；
5. `solver_mass` 的 row/column 和可小于端点质量，剩余质量写 one-sided unmatched row；它只用于 stop-gradient 训练或算法诊断；
6. `effect_match` 只在两端 adjudicated effect/outcome 均 known 时机械派生，否则为 `unknown`；
7. 每个 CSV 配一个 receipt，至少绑定 human/robot event table SHA-256、source clip/rollout SHA-256、split receipt、task ontology、cost config、solver/version/seed、生成 commit。行内 `provenance_id` 引用该 receipt。

## 4. A/B 双标、裁决与一致性门槛

### 4.1 两阶段独立性

**事件轮：** A/B 各自在隔离 packet 内看相同原视频，独立填 coverage、contact segment、event、edge 和 2D pose。不能看 hand status、timing hint、模型 proposal、对方 CSV 或旧 GT。事件 ID 前缀、annotator/session 必须不同。

**pair 轮：** 先完成事件裁决并冻结两端 event manifest；A/B 再独立观看成对视频与 adjudicated event timelines，对预先冻结的 candidate edge universe 标 `match/nonmatch/ambiguous` 或 one-sided unmatched。pair UI 不显示 OT/DTW 分数、solver mass、模型 embedding 或另一标注者选择。

pair 轮使用共同的 adjudicated event ID 不削弱 edge 判断的独立性；它避免把“事件切分分歧”和“是否匹配分歧”混成一个不可解释指标。

### 4.2 3 段校准与 12 段盲标

3 段 calibration 必须与 12 段 blind 不重叠，也不进入 G0 数值或 test：

1. 一段边界清楚、无 regrasp 的标准成功；
2. 一段含长 hold、双手或 regrasp 的高事件复杂度样本；
3. 一段含遮挡、interval/censor、effect 不可判或 partial/failure 的 missingness 样本。

现有 12 段 formal minimum set 保持 episode `43,50,51,58,34,38,9,54,36,53,12,0`，不因看到视频后换样本。calibration 从其余 54 段选；选择规则和 ID 在 A/B 开始前冻结。episode 0 的历史 timing hint 继续隐藏。

A/B 完成 3 段后可共同讨论，协调员最多修订一次 dictionary 与 task ontology，然后冻结 protocol hash。12 段盲标期间不再改定义。盲标汇总 gate 未通过时，不允许直接靠裁决“洗掉”系统性分歧；当前预算判 No-Go，若继续需新 protocol version 和新的 disjoint calibration。

注意：现有 3+12 全是 pour，只能验证单任务 ontology 与可标性，不能证明八类/多 effect 的跨任务 coverage。跨任务字段必须在 36+36 中另过 coverage gate。

### 4.3 A/B 匹配与指标

同一 episode 的 A/B macro events 先按 confirmed-positive support interval 的 temporal IoU 最大化做一对一 bipartite matching；候选要求 `tIoU>=0.5`，不能用 event kind 相同作为匹配前提，否则 kind agreement 会被高估。接着只在 matched event 与双方 known 字段上算分类一致性；真实 boundary 的 lo/hi 分歧另算，不能由 support tIoU 代替。

| 层级 | 指标 | Gate A 阈值 |
|---|---|---:|
| coverage | weighted Cohen κ | `>=0.70` |
| event existence | event precision/recall/F1 | `F1>=0.80` |
| interval | matched median tIoU | `>=0.70` |
| boundary | 双方 observed 的 P50 absolute error | `<=2 processed frames` 或 `<=100 ms`；robot 域用时间再换算帧 |
| event kind | Cohen κ | `>=0.70` |
| active hand / role / primary part | Cohen κ | `>=0.70` |
| contact mode | Cohen κ | `>=0.60` |
| relation before/after、effect、outcome | Cohen κ | `>=0.70` |
| visibility / censor status | Cohen κ + confusion matrix | `>=0.70` |
| 2D pose | 双方 visible landmark 的 image-diagonal NME P50；visibility κ | `NME<=3%`；`κ>=0.70` |
| capacity | `Q_episode>8`、node>4、edge/event>2 | 各 `<=5%` reviewed episodes/events |
| missingness | 关键 effect/outcome 双侧 unknown | `<=40%` |

κ 只有在至少两个类别有支持且有足够 comparable items 时才解释；单类或极稀疏字段报告 prevalence、positive/negative agreement 和 bootstrap CI，并标 `not_evaluable`，不能用 100% agreement 冒充过门。任何计划进入 action packet 但在 blind set 中 `not_evaluable` 的字段保持 Conditional No-Go，直到 36+36 跨任务样本提供支持。

pair 轮另报：

- `match/nonmatch/ambiguous` weighted κ `>=0.70`；
- reviewed positive edge F1 `>=0.80`；
- human/robot unmatched F1 各 `>=0.80`；
- effect-match κ `>=0.70`；
- adjudicated positive crossing rate必须为 0；若任务确有顺序反转，该 pair 应标 sequence hard negative或另定义 task family，而不是放宽 monotonic truth。

### 4.4 第三人裁决

只有 aggregate agreement gate 通过后，第三人才能同时查看 A/B 与原视频：

- 新建 adjudicated event/edge/pose/pair ID；`parent_*_ids` 完整引用 A/B；
- 单方事件或单方 edge 必须记录 unilateral reason；
- 不可判字段保持空/unknown，candidate edge 可降为 ambiguous/unreviewed；
- 不得把 `positive_only` 升级为 exhaustive，也不得将 interval/censor 压成 exact；
- adjudicated tables 才能成为 gold；A/B 原 CSV 与 disagreement report永久保留 hash 绑定。

## 5. Partial / unbalanced OT 的主 cost 与 GT 边界

### 5.1 Candidate gate

先按 `task_family × functional roles × intended effect/outcome` 建 candidate episode graph，再在同一 candidate pair 内建 event cost。task/effect 明显不相容的 pair 进入 hard-negative stress set，而不是让 OT 靠轨迹外观猜。

主 cost 冻结为：

```text
C_ij = 0.20*d_event_kind
     + 0.20*d_contact_mode
     + 0.15*d_role_part
     + 0.30*d_relation_effect_outcome
     + 0.10*d_local_order
     + 0.05*d_object_local_geometry
```

| 分量 | 只允许的输入 | 禁止替代 |
|---|---|---|
| `d_event_kind` | fixed eight-state kind 的 exact/预注册兼容矩阵 | learned embedding 距离、task name 相似度 |
| `d_contact_mode` | known edge mode；两条 edge 用固定 role-aware minimum assignment | `hand_status`、mask overlap、接触分类器 logit |
| `d_role_part` | functional role、primary/secondary part；不比较 human/robot literal side | object appearance/instance embedding |
| `d_relation_effect_outcome` | relation before/after、effect、outcome；为最高权重 | VLM 生成“应该成功”文本 |
| `d_local_order` | 各自 event rank 的归一化差；另加 crossing penalty | raw frame index、同速假设 |
| `d_object_local_geometry` | 4×4 surface/center bin、2D orientation band | metric 3D、MANO、dense trajectory、learned ROI feature |

duration、active-hand 字面侧别和 global wrist path不进入主 cost；可作为预注册 ablation。人/robot 时长不同不是 negative。

对 unknown field，删除该分量并对剩余已知权重重新归一化；已知总权重低于 `0.45` 时该 edge 只允许 unmatched/ambiguous，不得把 unknown 当零距离。端点质量可取：

```text
a_i = q_event * q_coverage * q_effect
b_j = q_event * q_coverage * q_effect
```

其中 human-adjudicated/oracle 路线的 `q` 来自冻结的 coverage/evidence/confidence band；predicted 路线的 `q` 必须在 validation 做 calibration。NULL cost 随 evidence confidence 增加，使高置信 task event 更难被随意丢弃；具体常数只能用 calibration/validation 冻结。

离线求解允许 row/column mass 小于端点质量，并加 crossing penalty或 banded support；求解后做 deterministic maximum-mass non-crossing projection。最终 sparse `solver_mass` 写 `event_pair_v1`，完整 cost config、marginals、epsilon/tau、projection 与 hash 写 solver receipt。

### 5.2 哪些是 GT，哪些绝不是

可成为 GT 的只有：

- 隔离 A/B 后第三人裁决的 event/edge/pose/candidate pair label；
- 有明确 contract 的 native simulator、触觉/力、编码器、秤/液位等 sensor-confirmed 字段；
- 从 adjudicated known 字段机械派生的 rank、duration interval、effect-match 和 non-crossing status。

以下全部只能是 proposal、weak target、diagnostic 或 model output，不能用于 gold 指标：

- optical flow/tracker 关键点、MANO/单目 3D、mesh proximity、VLM relation/effect；
- `hand_status/hand_active`、gripper command、mask overlap；
- event detector kind/boundary/confidence、learned feature distance；
- DTW/Drop-DTW/GMS-SDTW/OT/CAPOT pair、`solver_mass`、null mass与 transported mass；
- 从旧 task 名、文件序号、时间空档、视频外观推断的 operator/session/object/liquid/batch；
- 对透明液体接触、流速、体积或成功状态的无传感猜测。

benchmark 必须分 `oracle-field pairing` 与 `predicted-field pairing`。前者只是 solver upper bound，不可作为部署/自动配对结果。

## 6. 36 human + 36 robot 补采预算

### 6.1 三个任务族

选择任务族的原则是用三个互补的 transition signature 覆盖 acquisition/release、持续作用/regrasp、effect verification，而不是堆三个外观不同的 pick-place：

| family | 目标 transition | 硬件角色 | outcome truth |
|---|---|---|---|
| `pick_place` | `separate -> held -> inside/on -> released` | 右 OmniPicker 主抓放；左手默认不参与，作为单末端基线 | object 是否稳定 inside/on；人工+俯视/位姿传感 |
| `stabilize_twist_cap` | `right stabilize + left acquire/rotate/regrasp -> cap open/closed` | 右 OmniPicker 稳定瓶体，左 RH56 扭盖；专门检验不对称双手与 regrasp | cap encoder/已知旋转、人工视频双证据 |
| `pour_transfer` | `held -> aligned_to_rim -> flow/transfer -> upright -> release/verify` | 左 RH56 持 source，右 OmniPicker 稳定 target；实际可行性仍受 hardware gate | 秤/液位差为主，视频只作可见辅助；spill另记 |

若硬件安全/可达性审查表明上述左右分工不可执行，只能在采集前改 `task_ontology_version` 并重新冻结；不能采完后按成功样本重定义 family。

### 6.2 12 个 cell 与每 cell 三种样本

```text
3 task families
* 2 physical object-instance conditions
* 2 session/day groups
= 12 cells

每 cell：3 human clips + 3 target-robot rollouts
总计：36 human + 36 robot
```

每 cell 的三条 human clip 固定覆盖：

1. 标准成功；
2. 额外 preparation 或速度变化，但结果成功；
3. regrasp、partial 或失败之一。

每 cell 的三条 robot rollout 固定覆盖：两个成功/可恢复样本 + 一个 partial/failure。不能因为失败“不好看”而重录替换；重试是新 rollout，并保持失败源记录。

对 pour family，采集前必须记录 op29 已证明缺失的七字段：

```text
subject_id, session_id, operator_id,
source_object_instance_id, target_object_instance_id,
liquid_id, collection_batch_id
```

对非液体 family，通用 manifest 仍要求 `subject/session/operator/source-role instance/target-role instance/collection batch`，并用显式 `field_applicability=false` 表示 `liquid_id` 不适用；它与 unknown/missing 不同。不能把 `N/A` 填进 op29 v1 supplement，因为现有 validator 会把它当占位符拒绝。robot 侧另记录 `collector_id,embodiment_id,calibration_id,camera_id,safety_config_id`；所有数据记录 source hash、起止 UTC、frame/fps/time-sync、task ontology、intended effect、observed outcome、spill/drop/collision。每个非空 group 值必须有 capture record/experiment log/collector attestation provenance，不能事后从视频猜。

### 6.3 这个预算能证明什么

12 cells 可作 G1/G2 pilot，但每 family 只有 `2 object × 2 session`。它不足以同时建立 train/val/test 三套互斥的 object/session OOD 轴，也不支撑 production OOD 论文。

预注册的诚实 split 是 cell-heldout：每 family 2 cells train、1 cell validation、1 cell test，共 `6/3/3` cells，即每域 `18/9/9` clips。一个 cell 的 3 human + 3 robot 及其所有 events/windows/pairs 必须同 split。由于 object/session 值仍会在其它 cell 重现，结果只能写“held-out object-session cell”，不能写“novel object 且 novel session”。

若最终要声称独立 object/session OOD，需要另采至少第三组、最好更多 instance/session，并令相应 identity 在各 split 间完全不相交；不在本次 36+36 预算内。

### 6.4 Pair gold 的抽样量

所有 72 段做单人 event anchor review；至少 20%（向上取 15 段，且每 family/human-robot 域均覆盖）做第二人 blind review与全分歧裁决。每 family 至少双标并裁决：

- 20 条 `match` event edges；
- 20 条 one-sided unmatched/NULL；
- 20 条 hard-negative event edges。

因此最少是 180 条 A/B pair decisions。candidate universe 在看到 solver score 前按 family/role/effect 与预注册难度分层冻结；未抽中的 edge 是 `unreviewed`，不是 negative。

如果“3 段校准”被用于新 36+36，而不是现有 Ego 指南校准，应把它解释为三个 **paired calibration units**（每 family 一对 human+robot，实际 6 个视频）。只校准 3 个单视频不能同时检查 human 与 robot 域的边界/可见性差异。

## 7. 旧 human-pour / robot-grab 的 hard-negative 用法

现有 66 Ego pour 与旧 88 dual-Inspire grab 只能进入一个独立的 `legacy_task_mismatch_stress` split：

1. episode candidate 标 `hard_negative`，理由固定为 `task/effect/hardware mismatch`；不生成 66×88 笛卡尔积，只从 source-level split 中按长度/可见性分层抽固定 pair，避免伪样本量；
2. event 级只标 effect-bearing hard negatives，例如 human `manipulate: transfer/flow` 对 robot `hold/manipulate: translate-without-transfer`，以及 human `verify_outcome: transfer success` 对 robot `verify_outcome: bottle grasp/lift`；
3. 通用 `approach/acquire/release` 不因 episode task 不同自动标 nonmatch，它们可能是真正共享的低层语义；保持 `unreviewed` 或 one-sided unmatched；
4. 主要测试是完整路径方法是否强迫配对：CAPOT 应把绝大多数 task-relevant effect mass送入 unmatched，而不是找一条看似平滑的完整 path；
5. 预注册 stress 指标：reviewed positive edge 数为 0、episode false-accept rate `<=5%`、transported mass / `min(total_h,total_r) <=0.10`；阈值只在 calibration/validation 冻结；
6. 该 split 不进入同任务 positive retrieval、policy transfer或目标硬件结论，不用来训练一个靠 task 名就能轻易区分的主 benchmark classifier。

## 8. Split、group、泄漏与 benchmark

### 8.1 Split 顺序

唯一合法顺序是：

```text
source clip/rollout inventory + authoritative group metadata
-> split/group receipt
-> event/pose/contact annotation or proposal
-> within-split episode candidate graph
-> event_pair_v1 reviewed edges / solver mass
-> windows, augmentation, cached features and training samples
```

严禁先枚举 pair/window 再随机切分。若一个 human clip 连多个 robot rollout，或反之，bipartite candidate graph 的整个 connected component 必须同 split。相同 source episode 的裁剪、不同 fps 版本、增强、pose keys、contact/event rows、solver features 和 hard-negative副本全部同 split。

### 8.2 Group 约束

- 现有 66 Ego 在七字段补录完成前只允许 episode-heldout/temporal-diagnostic，不允许 OOD claim；op29 continuity block 不等于 session/batch。
- 新采集 cell 是本 pilot 的最小 group；同 physical instance、session/day、calibration 与 capture batch 的数据不得通过不同 pair edge跨 split。
- calibration clips/paired units 不进 blind agreement数字、validation/test或阈值调参；可在 protocol 冻结后仅进入 train，并单独标记。
- pair threshold、NULL cost、cost compatibility matrix、candidate sampler 与 confidence calibration只能在 calibration/validation冻结一次；test 不调参。
- event-table、source、task ontology、split 与 solver receipt 的 SHA 不匹配时 fail-closed。

### 8.3 Benchmark 分层

| 层 | 数据 | 主问题 | 指标 |
|---|---|---|---|
| L0 标注 | 3 calibration + 12 blind Ego | 事件、边界、2D pose、unknown 是否可重复 | Gate A 全指标、unknown/overflow、人时 |
| L1 配对 | 36+36 同任务 | partial/many-to-many 是否胜完整路径并拒绝 NULL | edge P/R/F1、NULL AUPRC/F1、effect consistency、crossing、mass/calibration |
| L1-negative | 固定 human-pour/robot-grab stress | 是否拒绝 task/effect mismatch | false accept、transported mass、错误 effect edge |
| L2 oracle | future native/sensor robot data | oracle event packet是否比22D-only保留控制信息 | task/event completion、correct contact/release、safety |
| L3 predicted/OOD | 未来扩充 group/hardware | predicted packet是否恢复 oracle gain | group-OOD success与cluster CI、causal route audit |

L1 必须包含：normalized-time/strict 1:1、standard/soft-DTW、Drop-DTW、GMS-SDTW/subsequence DTW、EgoBridge-style balanced OT、partial OT、无 monotonic UOT、CAPOT 与 oracle manual edge。所有算法使用同一 candidate universe和同一 known-field masks。

pair edge和滑窗不是独立样本。对 event pairing 用 human clip 与 robot rollout 的交叉 cluster bootstrap；对 36+36 pilot至少以 cell 为 cluster报告 95% CI。三 seed不能替代真实 clip/rollout数。

## 9. 如何扩展现有 UI/CSV（本轮不改代码）

不能直接修改 op34 packet：当前 HTML 的 `PACKET_FILES/COVERAGE_FIELDS/SEGMENT_FIELDS` 与 Python preparer 的 exact allowlist/header 会拒绝任何新增文件或列。未来实现应：

1. 新建 `sm_fet_dual_annotation_packet_v1` 与独立 `UI_BUILD_ID`，复用但不修改 `contact_dual_assignment_packet_v1_1`；
2. 原样保留 `coverage.csv` 和 `segments.csv`，新增 sibling `events_v1.csv`、`event_edges_v1.csv`、`pose_keys_2d_v1.csv`；pair 轮使用独立 `event_pair_v1.csv` packet；
3. packet/媒体/state 三根目录、无网络 CSP、精确 allowlist、owner-only 权限、identity/session lock、SHA-256、autosave到外部state、staged输出和原子提交全部沿用 op34；
4. UI 分成三次 pass：event interval → contact/relation/effect → pose keys。timeline 同时显示 contact segment 与 macro event，但任何字段都不预填；
5. event boundary 支持 lo/hi 两端 click；pose canvas只允许点击七个 functional landmarks与对象 bbox，遮挡时一键设置 visibility并清空坐标；
6. pair UI 左右并列播放 human/robot，显示已裁决 event timeline与 candidate edge列表，但隐藏 solver/model score；显式按钮为 match/nonmatch/ambiguous/human-unmatched/robot-unmatched；
7. 新 checker验证 transition graph、interval/censor、duration derivation、edge<=2、pose visibility/coordinate mask、A/B ID隔离、pair non-crossing和所有 receipt；
8. 继续禁止 notes 自由输入敏感路径；CSV unknown 使用空字符串，JSON receipt拒绝 NaN/Inf/重复 key；
9. proposal-only 模式必须输出不同 schema/tier，不得由同一按钮升级为 human submission；
10. adjudication 使用第三套 identity/session 与独立 checker phase，不让 A/B 单方面覆盖。

未来代码位置应是新文件，而非本轮修改：

- `tools/sm_fet_annotation_ui/sm_fet_annotation_ui.html`；
- `scripts/contact_labels/prepare_offline_sm_fet_annotation_ui.py`；
- `scripts/contact_labels/check_formal_sm_fet_dual_annotation_v1.py`；
- `track_a/sm_fet_labels_v1/` 下的 schema、packet template、dictionary 与 receipt。

只有本 memo 已创建；以上代码和目录当前均不存在。

## 10. 人时估算

### 10.1 当前 3+12 标注 pilot

| 工作 | 人时 |
|---|---:|
| dictionary 熟悉、3 段 A/B calibration、讨论与一次冻结 | 3–4 |
| 12 段 blind，A/B 各做 event/contact/relation/2D key pose | 每人 3–4.5，共 6–9 |
| 第三人 event/pose/contact 裁决 | 2–3 |
| coordinator packet、收件、agreement/receipt 与抽查 | 0.75–1.25 |
| **合计** | **约 12–17 人时** |

这是 integrated pass 的估算，不应与 op17 contact-only 的 5.5–8.5 人时简单相加。若每 event 都需要多于一个 turn key、透明/遮挡使回看显著增加，按上界再加 20%。

### 10.2 36+36 补采与 gold pilot

| 工作 | 人时 |
|---|---:|
| human capture、cell setup、metadata/outcome检查 | 6–9 |
| robot capture、reset、安全观察、传感/标定检查 | 12–18 |
| 72 段单人 event/pose标注 + 15段第二标 | 18–28 |
| 180 条 A/B pair decisions + 裁决 | 5–8 |
| coordinator、hash/group/split/QA | 3–5 |
| **合计** | **约 44–68 人时** |

不含 driver/mapper/calibration/watchdog 开发，也不含任何模型训练。若 robot rollout要求两人同时在场，上表 robot capture 已按安全观察人计入；实际场地流程应以本地安全 SOP 再校准。

## 11. Go / No-Go gates

### Gate A：schema/annotation

3 段 calibration 后冻结 protocol，12 段 blind 满足第 4.3 节所有可评估门槛，且关键 effect/outcome unknown `<=40%`、各 capacity overflow `<=5%`。

- 通过：允许把 adjudicated v1 tables作为单任务 annotation/compression gold；
- 失败：最多一次新 protocol version；当前数据不扩到 66，不实现 action packet；
- 字段 `not_evaluable`：该字段不可宣称通过，也不可进入 action-visible schema，等待跨任务补采。

### Gate B：true overlap / pairing

至少完成 36+36 中同任务 group metadata、outcome truth与 180 条 reviewed pair decisions。要求：

- NULL AUPRC `>=0.80`、unmatched F1 `>=0.80`；
- positive edge F1相对 Drop-DTW/GMS-SDTW/EgoBridge-style最优者 `>=+5 pp`；
- effect consistency `>=0.80`；
- projected crossing `<1%` 且无 all-null collapse；
- legacy mismatch false accept `<=5%`、normalized transported mass `<=0.10`。

未通过时 CAPOT只能作为内部数据工具，删除 matcher novelty与 paired-transfer claim。

### Gate C：oracle sufficiency（不在本轮执行）

需要 native/sensor-confirmed robot contact/outcome。oracle event packet相对 oracle 22D-only在 outcome-sensitive task success至少 `+8 pp` 或 event completion `+10 pp`，相对 full trajectory下降不超过 `3 pp`，safety不退化。

未通过时 event/pair协议仍可作为 benchmark，但 SM-FET action route No-Go；若只有 continuous trajectory/MANO/learned ROI action-visible 才有效，停止 finite claim。

## 12. 最终结论

最小且足以证伪本想法的协议，不是 dense hand trajectory，也不是把所有阶段压成一个 contact segment，而是：固定八状态 + interval/censor + 至多两条 role/contact/relation edge + 每事件 2–3 个可见 2D functional pose keys + 显式 effect/outcome + reviewed sparse many-to-many pair edge。

当前可以批准 3+12 的 annotation Gate A。当前不能批准任何正向 human–robot alignment 结论，因为现有 human-pour 与 robot-grab 没有同 task/effect overlap，group provenance也不完整。只有 36+36 按本协议补采、先 split 后建 pair、Gate B 通过，才值得继续做 oracle event packet；在 Gate C 前仍不应训练 SM-FET-WAM。
