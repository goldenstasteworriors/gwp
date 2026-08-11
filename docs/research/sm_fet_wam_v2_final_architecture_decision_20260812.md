# SM-FET-WAM v2 最终架构裁决：事件锚定、局部运动监督与非同步人机配对

- 日期：2026-08-12
- 审阅性质：独立、只读交叉审阅；本 memo 之外未改代码、数据或训练配置
- canonical 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 审阅输入：
  - `sm_fet_wam_v2_schema_architecture_20260812.md`，commit `190f302`
  - `sm_fet_annotation_and_pairing_protocol_20260812.md`，commit `66f2857`
  - `human_observational_tool_learning_evidence_20260812.md`，commit `52e763c`
  - `temporal_event_pairing_wam_meta_synthesis_20260812.md`，commit `8939659`（背景参考）
- 范围：schema、事件本体、容量、标注、训练期配对与认知叙事；未训练、未调用 GPU、未改 canonical

## 0. 唯一裁决

唯一推荐路线仍是 **SM-FET-WAM v2-min**，但它不是把原 825-float 草案直接实现，也不是现在立刻训练：

```text
causal robot observation/history + proprio + text
                  |
       Semi-Markov episode planner (Q<=8)
                  |
 deterministic validated compiler
                  |
 4 role nodes + active/next_1/next_2 + <=2 edges/event
        82 uint8 codes / 13 hard tokens
                  |
         strict WAM action route
                  |
     48-step prediction, execute 8, replan
```

其训练期可以使用两类 action-invisible 信息：

1. 人类/机器人事件内的 `entry -> optional turn -> exit` 稀疏手姿态与接触变化，作为 event encoder/planner 的 auxiliary 与压缩 probe；
2. 离线、stop-gradient 的 CAPOT sparse pair table，只连接同任务、同功能角色/效果的非同步 human/robot macro events。

部署时两者都不进入 action：没有 human video、pair ID、OT coupling、MANO、连续 hand trajectory 或 planner hidden 的直达路径。动作侧所需的“操作前手状态”应来自机器人 proprio（必须包含相关末端/手指状态），操作目标接触来自 hard event edge；人类手姿态只作为识别事件与检验多姿态假设的训练证据，**不做 human joint 到 robot joint 的字面配对**。

当前结论为：

- **Go**：修订 schema—annotation 契约、做 3+12 标注可行性；
- **Conditional Go**：36 human + 36 target-robot 同任务数据与 180 条 pair gold 通过后，才允许离线配对 benchmark；
- **No-Go**：现在实现/训练 WAM、使用 GPU、声称 paired transfer、目标硬件收益，或声称人类认知机制已经证明 SM-FET。

## 1. 207.749746919520-bit 独立审计

### 1.1 算术结论

在 `sm_fet_hard_grammar_v2_standard_r1` 的全部约束不变时，schema memo 的整数和对数可以独立复算：

```text
S_action
= 345864043392188289642033658185203030253953267411599196753571073

log2(S_action)
= 207.74974691951957 bit

ceil(log2(S_action))
= 208 bit
```

复算同时得到：

```text
E_n = [1836, 449820, 1343952, 2684232]

single-event weighted states
= [1780803552,
   213669437226432,
   2861033982456672,
   15217179888890496]

weighted kind-path counts, p=1..8
= [12, 39, 108, 289, 759, 2013, 5408, 14562]
```

因此 `207.749746919520 bit` 是正确的 **r1 validator 可达 action packet grammar alphabet**。它不是经验熵、互信息、模型利用的 bit 数，也不是 82-byte wire 的存储宽度。原 meta 的 `281.932700038062 bit` 只能保留为 independent-slot syntactic upper bound，不能再称 codec-exact reachable capacity。

### 1.2 与“八类事件”是否一致

有条件一致，但当前两份 memo 的措辞必须统一：

```text
8 semantic event kinds
= approach, precontact, acquire, hold,
  manipulate, regrasp, release, verify_outcome

+ 1 UNKNOWN sentinel
+ 1 ABSENT code
```

`UNKNOWN` 不是第九种动作语义。它只表示 event 已知存在、但 kind 不可判；在 r1 grammar 中它必须终止当前 tube。`ABSENT` 表示 slot 不存在。`MISSING/NOT_REVIEWED` 没有监督，不能自动编译成 UNKNOWN。

容量计数把 `UNKNOWN` sentinel 纳入可达 packet，因此和“八个语义类”不矛盾；但如果论文写“八状态 wire alphabet”，就是错误的。准确写法应是“八类跨任务语义事件，外加 UNKNOWN/ABSENT 控制状态”。

### 1.3 何时 207.7497 不再成立

以下任一变化都会使该数字失效，必须重新枚举：

- 允许 protocol 当前写出的额外 skip/early-release transition；
- 允许非 `verify_outcome` event 携带 action-visible outcome；
- 将 entry/turn/exit hand pose、confidence、depth 或 secondary part/cell 送入 action；
- 改 `N=4, K=3, R=2`、node state dictionary 或 edge order/union 规则；
- 把 task-conditioned transition 当成运行时可变 grammar，却仍引用全局 r1 数字。

因此本裁决选择：**保持 r1 action wire 与固定全局 grammar，不为适配尚未产生的标签扩大 schema；先修改标注协议使其成为该 grammar 的子集。**

## 2. 阻塞性冲突与处理决定

| 冲突 | 严重性 | 裁决 |
|---|---:|---|
| annotation 四张表没有 `focus_node`，也没有四 role node 的 current geometry/observability；因此无法编译 schema 的 node/event packet | 阻塞 | 在任何盲标前增加 `focus_role/focus_entity_id`，并定义独立 `event_node_snapshots_v1`；否则 208-bit schema 只有数学定义，没有 gold 编译路径 |
| protocol 允许 impact 跳过 acquire、precontact 直接 release；r1 grammar 不允许 | 阻塞 | r1 global graph 为规范。impact 仍标一个可短时/interval 的 acquire；无接触放弃从 precontact 进入 episode NULL/失败 sidecar，不伪造 release |
| protocol 暗示 release 可直接 `outcome=failure`；schema 只有 `verify_outcome` 能携带 outcome | 阻塞 | event action field 的 outcome 只属于 `verify_outcome`；release 不推 outcome。即时失败写 episode outcome sidecar，若后续有可观察确认则建立 verify event |
| protocol 的 unknown 使用空值，schema 有 action-visible UNKNOWN code | 高 | checker 必须按 `event exists + reviewed but undecidable -> UNKNOWN`、`not reviewed -> loss mask/no target`、`absent under exhaustive coverage -> ABSENT` 三分编译 |
| `event_edges_v1` 要求 secondary part/cell，但 standard wire 已删除它们 | 中 | 从 core required 字段移出；只作 full/oracle geometry probe。不能让 standard Gate A 被不会进入主模型的字段卡住 |
| `active_hand` 与 edge actor 部分重复，且 human/robot 左右侧不能字面对齐 | 中 | annotation 中保留为 edge-less approach/verify 的 QA/可见性字段，action wire 不保留；有 edge 时与 actors 做一致性校验，CAPOT cost 不用 literal side |
| pose 表有 7 landmarks 与 entry/turn/exit，但主 action packet完全没有 hand pose | 高 | 不把认知动机写成“action packet显式保存多手姿”。pose 是独立 auxiliary/probe；必须做 `single / event-multipose / local-window / dense` 消融。若只有 pose action-visible 才有效，finite route停止 |
| relation_before/after 描述对象关系，不等于手接触部位 before/after | 高 | 不混写这两个概念。机器人 before-state 来自当前 proprio/接触状态，target contact 来自 active event edge；人类 entry/exit pose/contact 只监督事件。如果 robot proprio缺末端接触/手指状态，strict v2 不具备可执行语义，直接 No-Go |
| annotation task roles 与固定四 node roles 没有 frozen mapping | 高 | 每个 task ontology 在标注前给出一对一 role mapping；无法映射或同 role 同时多实例的样本记 overflow，不由标注员自由改 slot |
| 3+12 全是 human pour，却被用于所有 action-visible 字段 Gate A | 高 | 3+12 只能证明单任务边界/事件/pose可重复。未出现两个类别或未覆盖的字段记 `not_evaluable`，必须等 36+36 跨任务补采，不能宣称 schema 已通过 |
| 36+36 只有 12 个 cell，却有 point-threshold 的 pairing 胜幅 | 中 | 只称 feasibility pilot；除 point `+5 pp` 外必须给 cell/clip cluster bootstrap 95% CI。CI 不支持正优势就删除 CAPOT superiority claim |
| 人类 touching/untouching、MEP/fMRI、眼动被用作架构机制证据 | 阻塞叙事 | 只能作为候选锚点与消融动机；永不写成必要机制、固定八阶段证明或机器人收益证据 |

## 3. v2-min 的唯一字段契约

### 3.1 action-visible：保留

保持 schema standard r1 的 82 个 canonical `uint8` codes 与 13 个 tokens：

```text
node_code  [4,4]
  observability/presence, center_cell, scale_band, orientation_2d

event_code [3,6]
  kind/presence, focus_node, onset, duration, object_effect, outcome

edge_code  [3,2,8]
  actor/presence, primary_node, primary_part, primary_surface_cell,
  secondary_node, relation_before, relation_after, contact_mode
```

理由不是这些字段已被人类认知证明，而是它们构成一个可反序列化、可做 hard intervention、同时保留当前对象位置、事件顺序/时长、接触目标与效果的最小可执行候选。它必须与 v1 22D、F-PCD 和 direct/full-trajectory 路线互斥。

### 3.2 action-visible：删除或禁止追加

以下字段不进入主 action packet：

- `confidence_band`、annotator confidence、coverage/evidence tier；
- depth、连续 bbox/UV、mask/flow、track embedding、MANO/SMPL/3D joints；
- secondary part/cell；
- `active_hand` 独立副本；
- event/track/pair ID、source hash、CAPOT mass/cost/hidden；
- entry/turn/exit raw pose、planner logits/hidden/posterior；
- v1 22D packet的拼接副本。

### 3.3 presence 与重复字段的合并

- node presence 合并进 `observability==ABSENT`；
- event presence 合并进 `kind==ABSENT`；
- edge presence 合并进 `actor==ABSENT`；
- `active_hand` 不进 action；有 edge 时由 actor bundle校验，无 edge 的 approach/verify只作 annotation QA；
- duration 由 boundary interval机械生成，人工不重复填写 duration float；
- effect 与 outcome 不合并：effect 是对象变化，outcome 是任务判据，后者只在 verify event action-visible；
- relation before/after 不再另造 delta 字段，delta 由二者机械计算。

### 3.4 annotation 必须补齐但 action 不新增的字段

在现有四表设计之外，必须补一个可编译层：

```text
events_v1:
  + focus_entity_id
  + focus_role

event_node_snapshots_v1:
  event_id, snapshot_role(entry|turn|exit|anchor), frame_index,
  node_role, entity_id, observability,
  center_cell, scale_band, orientation_2d,
  binding_provenance, annotation/evidence fields
```

`event_node_snapshots_v1` 用于验证 role binding 与 node quantization 是否可标；未来 chunk-level oracle packet必须来自 native robot object state或有 receipt 的 causal tracker，不能把稀疏人标插值冒充逐 replan GT。

`pose_keys_2d_v1` 保留为单独的多姿态研究表，但它不决定 208-bit core schema 是否过 Gate A。核心字段与 pose probe 应分别报告 agreement/unknown/工时。

## 4. 模型结构

### 4.1 Episode planner

planner 只读当前机器人 observation、合法 causal history、proprio、text 与上一个 hard event clock，预测最多八个 Semi-Markov events。它学习：

- eight semantic kinds + explicit unknown/absence；
- interval/censored onset 与 duration/termination hazard；
- fixed role binding 与 current node snapshot；
- contact/relation edge、effect 与 verify outcome；
- overflow/null probability，但 overflow code只进 control log。

连续或多姿态信息可以进入 planner 的训练监督，也可以作为 causal history evidence；不能通过 planner hidden/KV 绕过 hard compiler直达 action。

### 4.2 Action compiler 与 WAM

compiler 每 8 steps 取 `active + next_1 + next_2`，且 future 只保留 onset 在未来 96 frames 内的事件。它做 prefix、reference union、canonical sort、transition、unknown terminal、positive-zero 与 overflow 检查；invalid 时 safe hold/reobserve，不做 soft fallback。

WAM action head只读：

```text
robot proprio/state（含所需末端/手指当前状态）
instruction text
diffusion timestep/noisy action
13 hard event-tube tokens
```

它仍预测 48-step chunk，只执行前 8 steps再重观测。v2 disabled 时旧 checkpoint、参数 key 与 forward 必须保持不变。

### 4.3 为什么不另建 ETG、graph matcher 或 human-motion action branch

- object `relation_before/after + effect/outcome` 已吸收 ETG 最有价值的 state delta；ETG current-delta-only 留作 baseline；
- CAPOT 是训练数据工具，不是 runtime network；
- human continuous pose 与 robot morphology不应逐 joint 对齐；domain-specific pose auxiliary只帮助事件识别/压缩评估；
- 如果主 route必须加 human trajectory、learned ROI 或 continuous pose才能工作，方法应诚实降级为 trajectory-conditioned WAM，而不是继续扩大有限 schema来保护原 claim。

## 5. 训练期人机配对路径

当前只批准协议，不批准执行训练。未来满足数据门槛后的唯一路径是：

```text
human video -------------------> human event encoder/planner
  | events/edges/effect labels        | pose-change auxiliary only
  |                                   v
  +---------------- reviewed semantic event features
                                      |
                         offline CAPOT / non-crossing projection
                                      |
                         sparse stop-gradient event_pair_v1
                                      |
robot video + state/action ----> robot event encoder/planner
                                      |
                         matched-event distillation/alignment
                                      |
robot hard event packet + robot action only
                                      |
                              strict WAM policy loss
```

硬规则：

1. 先按 source/group 切 split，再在 split 内建 candidate graph；
2. candidate gate 先匹配 `task family × functional roles × intended effect/outcome`；
3. CAPOT 主 cost只用 adjudicated/predicted 的八类 event、contact mode、role/part、relation/effect/outcome、local order 与粗 object-local geometry；
4. human/robot duration、literal left/right、global wrist path和 3D hand pose不进主 cost，只作预注册 ablation；
5. unknown 分量 mask 后重归一，known weight `<0.45` 只能 unmatched/ambiguous；
6. coupling、solver mass 与 pair ID stop-gradient且不可进 action；
7. human clip没有 robot action label，不能通过伪 retarget action参与 policy loss；
8. 必做 `robot-only/no-pair` 对照。若配对监督不优于它，删除 human–robot pairing贡献。

该路径实现的是**事件级、部分、多对多且跨时长**的共享监督，不是从拿取前开始的全程一一同步，也不是 human trajectory 到 robot trajectory 的直接模仿。

## 6. 推理路径

部署时固定为：

```text
robot current image + causal robot history + proprio + text
-> robot event planner
-> hard decode / validate / active+next2 compile
-> 13 event-tube tokens + proprio/text/noisy action
-> WAM action chunk
-> execute 8 steps
-> reobserve / clock update / cache invalidation
```

推理时明确不存在：human video、human pose、CAPOT、pair table、GT events、future visual、continuous trajectory action token、soft planner distribution或 semantic graph runtime module。

## 7. 必须的 baseline 与消融

### 7.1 Action representation

- direct current-reference WAM；
- `contact_segment_v1_1` 22D-only；
- F-PCD-only；
- ETG current-delta-only；
- SM-FET active-only、active+next1、active+next2；
- SM-FET global equal-bit rank/permutation control；
- full/local continuous trajectory upper bound。

v1、F-PCD、v2 与 continuous route必须互斥，不能 concat 后仍声称某一有限表示有效。

### 7.2 时间信息与“多姿态”假设

同 token/参数/FLOPs 比较：

- single contact snapshot；
- fixed entry/exit poses；
- entry/turn/exit event-multipose；
- event-centered local windows；
- uniform K、random K、motion-peak K；
- full dense trajectory。

另做 leave-one-event-out、matched-motion deletion、phase shuffle/time reversal、oracle boundary vs predicted boundary。这样检验的是事件锚点与局部运动的贡献，而不是引用人类文献替代实验。

### 7.3 Pairing

- normalized-time strict 1:1；
- DTW/soft-DTW；
- Drop-DTW；
- GMS-SDTW/subsequence DTW；
- EgoBridge-style balanced OT；
- partial OT；
- UOT without monotonicity；
- CAPOT；
- oracle manual event edges；
- no-human-pair/robot-only。

所有方法使用相同 candidate universe、known-field mask与 source-level split。

### 7.4 字段归因

逐项移除 duration、future events、contact mode/part、relation delta、effect/outcome、pose auxiliary；再做 field-local intervention。equal-bit 持平时不能声称人类可读字段语义必要；pose auxiliary无增益时删除 hand-multipose training claim，而不是把 pose偷偷送进 action。

## 8. 数据门槛与停止条件

### D0：契约一致性，训练前必须完成

要求：

- annotation row 到 hard packet 的每个 action-visible字段都有唯一来源；
- 八语义类、UNKNOWN、ABSENT、MISSING 可机械区分；
- transition/outcome规则只存在一个规范版本；
- packet round-trip、canonicalization与 207.749746919520-bit receipt 一致；
- node/focus mapping与 overflow 可完整报告。

任一不满足：停止 UI 扩展与模型实现。

### D1：3 calibration + 12 blind human-pour

只验证单任务 annotation feasibility：event F1 `>=0.80`、kind/role/relation/effect κ `>=0.70`、contact mode κ `>=0.60`、boundary P50 `<=2 processed frames`、matched event tIoU `>=0.70`、关键 unknown `<=40%`、各 overflow `<=5%`。单类/稀疏字段记 `not_evaluable`，不能用 100% agreement 过门。

失败：协议最多修订一次；仍失败则停止 human-auditable v2，不扩到 66 episodes。

### D2：36 human + 36 target-robot 同任务 pilot

固定三任务族、12 cells、每 cell 3 human + 3 robot；所有 72 段单人 event review，至少 15 段跨 family/domain A/B；至少 180 条双标并裁决 pair decisions，包含每 family 各 20 match、20 unmatched、20 hard negative。

最低 pairing gate：

- NULL AUPRC与 unmatched F1均 `>=0.80`；
- CAPOT positive-edge F1相对强对齐 baseline 最优者 point gain `>=5 pp`，且 cell/clip cluster bootstrap 95% CI 支持正优势；
- effect consistency `>=0.80`；
- projected crossing `<1%` 且无 all-null collapse；
- legacy mismatch false accept `<=5%`、normalized transported mass `<=0.10`。

失败：CAPOT最多作为内部数据工具，删除 matcher novelty与 paired-transfer claim。现有 66 human-pour + 88 robot-grab永远不作 positive pair。

### D3：robot oracle sufficiency

36+36 是 alignment feasibility，不足以自然推出 policy 训练样本量或 production OOD。只有 native/sensor-confirmed robot contact/outcome 数据到位，并预先按 task variance 做 power analysis后，才运行 oracle route：

- v2 相对 oracle 22D-only：outcome-sensitive success `>=+8 pp` 或 event completion `>=+10 pp`；
- 相对 full trajectory下降不超过 `3 pp`；
- safety不退化，cluster CI不支持零优势；
- regrasp/精细接触子集必须单独报告。

失败：SM-FET action route No-Go；event/pair protocol仍可作为 benchmark。

### D4：predicted/causal route

只有 D0–D3 全过才检查 predicted planner：恢复至少 `70%` oracle gain；strict v2 优于 auxiliary-direct、22D-only与 matched equal-bit；group-OOD success有预注册正增益和 CI；field intervention具 phase locality，swap/cache probe不过则对应 causal claim删除。

以下为直接停止条件：

- 只有 continuous pose/trajectory action-visible 才有效：停止 finite representation claim；
- equal-bit与 semantic v2持平且利用率相当：停止 semantic field necessity claim；
- no-pair与 CAPOT相当：停止 human–robot pairing contribution；
- event windows不优于同预算 uniform/random：停止 human-inspired event selection claim；
- fixed八类在跨任务中 overflow/unknown持续超门槛：不要事后加类别保护结果，先判当前 ontology No-Go。

## 9. 认知主张边界

认知 evidence 的正确地位是 **Conditional rationale，不是 architecture proof**：

- Pomp 等支持 touching/untouching 是高信息候选边界，但多数 contact motif并未成为主观事件边界；
- Bunday、Roberts 与教学动画研究支持动态/局部轨迹可能不能被单帧普遍替代；
- goal/outcome、前瞻注视与跨效应器结果支持功能层级和稀疏注意，但不证明观察者只保存关键帧；
- regrasp 作为独立观察学习阶段的证据最弱，保留它是工具控制工程选择；
- 未发现容量匹配的 `single snapshot vs event multipose vs full trajectory` 人类工具执行三臂实验。

论文可写：

> 人类事件分段与观察学习研究提示 contact/release/effect 是候选事件锚点，同时动态运动仍可能携带不可替代信息。SM-FET 因而是“事件语义 + 局部运动监督”的可证伪工程归纳偏置。

论文不可写：

> 人类已被证明用固定八阶段或若干手姿态替代连续轨迹；mirror system证明该编码；因此该模型会提高机器人控制。

不论未来机器人结果多好，biological/mechanistic claim 仍是 No-Go，除非另有直接的人类因果实验。

## 10. 最终结论

当前最值得保留的不是“把人类动作完整映射给机器人”，而是三层分工：

1. **事件层**：用 contact、release、relation effect 与 duration 切出任务相关操作区间；
2. **域内运动层**：human 与 robot 各自保留 event 内 entry/turn/exit 或局部窗口，避免单帧手姿假设；
3. **跨域配对层**：只在共享功能事件上做 partial/many-to-many、允许 NULL 的离线配对，不对齐全程和字面手关节。

v2-min 的 208-bit hard packet是这一想法最干净、最容易被否证的 action route候选；但在 node/focus 标注缺口、transition/outcome冲突修好并通过 D0/D1 前，它仍只是设计契约，不是已验证模型。下一步应止于协议修订与标注，不应继续训练、仿真或 GPU 测试。
