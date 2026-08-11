# 生化实验 WAM Idea Portfolio 与唯一主线架构

- 日期：2026-08-12
- 唯一代码基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 目标硬件：Unitree G1 + 左手 RH56DFTP + 右手 OmniPicker
- 当前数据：66 段 Ego 倒液；88 段旧 dual-Inspire `grab_red_bottle`
- 文档性质：Zotero-first 研究证据汇总、架构裁决与数据/实验路线图
- 非本轮内容：代码实现、模型训练、GPU、仿真、服务器或真机操作
- 总体 `idea_id`：`b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`
- 本轮 critic `discussion_id`：`240c8a16-b6a2-4119-918f-711590af88ea`

> 本文中的 A–L 指十二份独立 idea memo，而不是 op22 dossier 内较早的 A–E 章节编号。op22 是历史总览；若其结论与后续 SM-FET standard-r1 裁决冲突，以后者为准。

## 0. 最终决定

唯一 headline 是：

> **SM-FET-WAM v2-min：将 Ego 视频中的工具操作编译为可审计的 Semi-Markov 有限事件管道，以当前角色节点、active/next 两个近期事件、接触/关系变化和结果作为当前图像到 WAM 动作分支的唯一视觉桥；用事件内多姿态监督避免单帧假设，用训练期离线部分人机配对避免全程轨迹复刻。**

它不是 A–L 的机械拼接。首版主模型只保留四个不可再拆的原则：

1. **事件而非单帧或完整人手轨迹**：episode planner 最多规划八个事件；每次动作只读取 active + next 1 + next 2。
2. **显式“在哪里操作”**：用固定角色 node、focus node、对象 part、part-local 4×4 surface cell 与对象关系表示，而不是一个自由连续 location token。
3. **严格有限路由**：82 个 canonical `uint8` codes 编成 13 个 hard tokens；standard-r1 的精确可达容量为 `207.749746919520 bit`，不是旧草案的 `281.932700038062 bit` 松上界。
4. **人机只对齐共享功能事件**：CAPOT 只离线生成 stop-gradient 的 sparse pair table；部署时没有 human video、pair ID、OT coupling、MANO 或完整人手轨迹。

当前裁决是：

- **现在 Go**：冻结 ontology/schema、做 D0 契约检查设计和 3 calibration + 12 blind 人工标注。
- **数据到位后 Conditional Go**：36 human + 36 目标机器人同任务 pilot 与 180 条 pair gold。
- **现在 No-Go**：实现/训练大 WAM、使用服务器/GPU、透明 3D 重建、把现有 66+88 当正配对、声称目标硬件或论文收益。

## 1. 用户四个问题如何进入唯一主线

### 1.1 Visual embodiment gap 与“操作哪里”

不增加一个不受约束的 `operation_location_token`。它容易退化为高带宽 ROI hidden，并重新形成 visual bypass。v2-min 把“哪里”拆成可人工复核的有限字段：

```text
四个固定 node roles
  0 manipulated_or_source
  1 tool
  2 receptacle_or_destination
  3 fixture_or_support

event.focus_node
edge.primary_node
edge.primary_part
edge.primary_surface_cell  # part-local 4x4
edge.secondary_node
edge.relation_before / relation_after
node.center_cell / scale_band / orientation_2d / observability
```

这组字段同时回答：当前任务对象是谁、目标部件在哪里、哪只机器人末端要作用、工具与目标在操作前后应形成什么关系。标注不是直接点机器人抓取位姿，而是：

1. 为每个 task ontology 冻结实体到四个 role slots 的一对一映射；
2. 在 event 的 entry/turn/exit/anchor 上记录四个 role node snapshot；
3. 对 task-relevant edge 标 primary part 与 part-local 4×4 cell；
4. `VISIBLE/PARTIAL/OCCLUDED_TRACKABLE/OUT_OF_VIEW/UNKNOWN/ABSENT` 明确分开；
5. human 的 literal left/right finger identity 不映射成 robot joint；先 canonicalize 成角色和效果，再由目标硬件 executor 分配给左 RH56 或右 OmniPicker。

该设计继承了 Idea B 的 morphology feasibility，但不把 B 的大拓扑包再拼进 v2。目标机器人当前手型/接触状态必须来自 proprio/tactile；不能用 human entry pose 代替机器人当前状态。

### 1.2 Ego 双手真值不持续存在

missingness 不是负例，也不是全零手势。主契约要求逐字段区分：

```text
KNOWN              -> 正常 categorical target
REVIEWED_UNKNOWN   -> 显式 UNKNOWN code，参与相应 CE
NOT_REVIEWED       -> loss mask = 0
ABSENT             -> 仅在 EXHAUSTIVE coverage 下成为负例
LEFT/RIGHT_CENSOR  -> interval/survival boundary target
```

对左右手、relation 和字段分别维护 coverage；左手 exhaustive 不会授权右手空槽为 negative。`hand_status` 仍只表示 open/closed grip proxy，`hand_active` 只是这个 proxy 的 validity covariate，二者永远不是 physical contact GT。若某只手出画、遮挡或 tracker invalid，允许另一个手和对象事件继续提供部分监督，不补造双手轨迹。

这由 Idea D 提供数据契约，由 SM-FET standard-r1 将 UNKNOWN/ABSENT/MISSING/censor 机械编译为不同语义。

### 1.3 透明容器/液体无法可靠重建

主线明确不依赖透明物体 3D 重建，也不从当前 RGB-D 制造伪 depth、mesh、flow 或液体体积 GT。

- 动作可见的空间信息只用可复核的 2D node cell、part-local cell、orientation、relation 与 visibility。
- 杯体 boundary、source/target rim 在可见时可作为 training-only auxiliary；连续 mask/heatmap 不进入 action。
- `flow/transfer` 可以是 event effect，但 transparent liquid 的体积、流速和接触界面在无传感证据时必须 unknown/not-reviewed。
- pour outcome 应由秤、液位或其它同步传感器确认；视频只作为可见辅助。
- ClearGrasp/透明 depth、PourIt!/液流分割等只做上界或 baseline，不成为主方法依赖。

因此 op22 中的 Transparent-Liquid Reconstruction / dense Flow World Model 当前是 **No-Go**；Idea C 只保留 visibility-gated 2D auxiliary。

### 1.4 用连续手姿态与接触变化教机器人

不把“一帧 grasp pose”当工具使用，也不把完整 human MANO 轨迹逐帧 retarget 给机器人。每个 observed event 只标：

```text
entry pose -> optional turn pose -> exit pose
```

每个 pose 使用 wrist、palm center 与五指 tip 七个 2D functional landmarks，加 aperture/grasp family/palm axis；不可见点留空并记录原因。由这些稀疏姿态机械计算 aperture、spread、方向、regrasp 与相对位移变化。

这些多姿态信息的唯一合法作用是：

- event encoder/planner 的 auxiliary supervision；
- single snapshot / multipose / local window / dense trajectory 的压缩与可证伪比较；
- annotation proposal 或连续轨迹上界。

它们不进入 208-bit action packet。机器人动作分支读的是自身 proprio + hard event tube，而不是 human joint angles。若只有 action-visible 连续 pose/MANO 才有效，有限事件 headline 失败，应诚实改为 trajectory-conditioned WAM。

认知证据只支持较弱的 rationale：touching/untouching 是高信息候选边界，动态运动通常比单一终态更有信息，goal/effect 可以跨效应器保持；没有直接人类实验证明固定八阶段或三张姿态替代完整轨迹。

## 2. 唯一 WAM 数据流

### 2.1 训练期

```text
Human video
  -> human event encoder/planner
  -> event/node/edge/effect prediction
  -> entry/turn/exit pose auxiliary only
                    \
                     -> offline CAPOT / non-crossing projection
                    /       -> sparse stop-gradient event_pair_v1
Robot video + native state/contact/outcome
  -> robot event encoder/planner
  -> matched-event distillation/alignment

Robot current image + causal history + proprio + text
  -> Semi-Markov episode planner (Q_episode <= 8)
  -> deterministic hard compiler
  -> 4 role nodes + active/next1/next2 + <=2 edges/event
  -> 82 uint8 codes / 13 hard tokens
  -> strict WAM action expert
  -> predict 48 steps, execute 8, reobserve and replan
```

human clip 没有 robot action label，不得通过 pseudo retarget action进入 policy loss。CAPOT、pair ID、solver mass、human pose、continuous track 和 planner hidden 均 stop-gradient/action-invisible。

### 2.2 部署期

```text
robot current image + causal robot history + proprio + instruction
  -> robot event planner
  -> validate + active/next2 compiler
  -> 13 hard event tokens
  -> WAM action chunk
  -> execute 8 / reobserve
```

部署期明确不存在 human video、human pose、pair table、OT、GT event、future visual、continuous trajectory token 或 soft planner distribution。

### 2.3 action-visible standard-r1 contract

```text
node_code  [4,4]
  observability/presence, center_cell, scale_band, orientation_2d

event_code [3,6]
  kind/presence, focus_node, onset, duration, object_effect, outcome

edge_code  [3,2,8]
  actor/presence, primary_node, primary_part, primary_surface_cell,
  secondary_node, relation_before, relation_after, contact_mode
```

八个语义事件为：

```text
approach, precontact, acquire, hold,
manipulate, regrasp, release, verify_outcome
```

另有 `UNKNOWN` sentinel 与 `ABSENT` code。UNKNOWN 不是第九类动作，ABSENT 不是事件，MISSING/NOT_REVIEWED 不进入 wire。

### 2.4 为什么是 v2-min，而不是“所有 idea 都打开”

这里的 `v2-min` 指**研究组合最小化**，其 wire 采用 schema memo 推荐的 `event_tube_v2_standard`，不是 schema 表里 81.88-bit 的 `minimal` 变体。首轮只启用：

- Semi-Markov event clock；
- standard-r1 hard compiler；
- strict WAM route；
- core event/node/edge监督；
- pose auxiliary 的开/关消融。

首轮不启用 CAPOT、CET-WM、RE-Q、SCEMC、forward-effect cycle、continuous pose action branch 或额外 graph network。它们各有独立 gate；没有证据前不能成为默认模型。

## 3. Portfolio 分级规则

- **HEADLINE**：论文主问题与主模型，只能有一个。
- **SUPPORTING**：不增加第二条主 action route；通过独立 gate 后可增强数据可信性、训练或硬件执行。
- **BASELINE**：必须作为互斥对照，不能与 headline 拼接后再把收益归给主方法。
- **NO-GO**：当前证据下不实现、不训练或不作论文 claim；未来若前置 gate 改变，可重新审查，但不默认复活。

## 4. A–L Idea Portfolio

### A. Auditable Finite Contact-Transition WAM

- **角色：BASELINE / inherited contract。** v1 22D/Q12 是已实现的严格有限接触基线；其“唯一 action-visible visual bridge、精确容量、hard intervention、equal-bit control”原则被 v2继承。
- **一句核心：** 把当前图像到 action 的视觉信息限制为有限、可反序列化的 contact segment grammar。
- **最强冲突：** BridgeACT、VLAff、ST-VLA/CoA-VLA、GeCCo，以及 GigaWorld Policy/GWP-0.5 已分别覆盖 contact/affordance conditioning、显式中间表示和 joint WAM；contact token本身不新。
- **novelty 残余：** 严格路由 + exact finite alphabet + matched direct/soft/hard/equal-bit controls + intervention 的受审计交集；现在由更完整的 v2 event tube承载。
- **所需数据：** reviewed contact/event labels、group-safe split、target robot contact/outcome；现有空白 12 段 packet不能证明收益。
- **WAM 插入点：** 现有 `contact_transition_schema.py`、contact planner/transport、22→action encoder；v2不得与其 concat，只把它作为 22D-only baseline。

### B. Asymmetric Contact Topology / Feasibility Allocation

- **角色：SUPPORTING，目标硬件层。** 不再建第二个大 topology packet，只保留 canonical role 到左 RH56/右 OmniPicker 的可行分配、split decoder 和 tactile residual。
- **一句核心：** 同一功能事件可以由六主动通道的灵巧左手与单 aperture 通道的右夹爪以不同执行路径实现，不能字面对齐 human fingers。
- **最强冲突：** KITE 已做 contact intent + embodiment decoder；TopoRetarget、Transferring Contact Not Just Motion、UniDex、CrossDex、AsymDex 已覆盖 contact retargeting、共享 latent 与异构双手。
- **novelty 残余：** 在 `6+1` 极不对称双手上检验 finite event semantics 的 sufficiency/feasibility gap，并用目标硬件 tactile/current 只修正接触状态。
- **所需数据：** RH56/OmniPicker driver、reverse mapper、标定、watchdog、安全链；目标硬件 task demos、触觉/电流和 held-out rollouts。
- **WAM 插入点：** hard v2 packet之后、25D action decoder之前的 embodiment allocator；输出仍遵守 `left wrist 9 + right wrist 9 + left hand 6 + right aperture 1`。

### C. Visibility-Gated 2D Boundary/Rim Auxiliary

- **角色：SUPPORTING，training-only。** 不进入 action route。
- **一句核心：** 只监督可见的杯边、杯口、node observability 和 censored event，不为透明液体伪造3D。
- **最强冲突：** ClearGrasp/TransCG/ClearPose 已覆盖透明深度与边界；PourIt! 和透明液体分割覆盖 flow/stream；Mask World Model、VLAff、CoA-VLA 已覆盖 mask/affordance auxiliary。
- **novelty 残余：** `不存在 / 存在但不可定位 / 可定位` 的 observation contract，以及证明连续 spatial auxiliary 不绕过 finite route。
- **所需数据：** 先对 boundary/source rim/target rim 做正式 A/B 标注；transparent stream/contact interface 没有可标性或传感 truth 前保持 unknown。
- **WAM 插入点：** shared visual/planner 的 training-only head与 masked loss；输出不能拼入 packet/action/cache。

### D. Censored Bimanual Contact-Event Benchmark

- **角色：SUPPORTING，数据与科学有效性层。** 它是主线必须有的监督契约，不是独立 headline。
- **一句核心：** 把 per-side coverage、物理事件 ID、interval/censor、字段 missingness 和 group split统一成可审计训练/评测协议。
- **最强冲突：** Ego4D/VISOR/100DOH/EgoLoc 已覆盖 contact frame或边界；ARCTIC提供动态几何接触；EgoTouch/TouchAnything提供连续 pressure map；censor/MNAR 方法本身也有成熟 survival/PU 文献。
- **novelty 残余：** 在 strict finite WAM 中联合检验 missingness calibration 与机器人 contact behavior，而非“首次标接触”。
- **所需数据：** 3+12 先过标签 gate；若独立 benchmark，至少约 300 adjudicated events、50 bimanual events、30 production groups与 sensor-confirmed subset；主线首轮不要求立即达到独立论文规模。
- **WAM 插入点：** sidecar loader、field masks、interval/survival loss、grouped evaluator；不增加 action token。

### E. Contact–Action Effect Consistency

- **角色：SUPPORTING，后置 causal/faithfulness audit。** D3 oracle之前不实现。
- **一句核心：** 预测 packet还不够；由 robot state+generated action推断的 realized contact/effect必须随可行字段干预而变化。
- **最强冲突：** DexMani、VERA、AMPLIFY、GWP、EVA与 action-conditioned contact world models已覆盖 action-effect guidance、forward/inverse consistency和可执行性评分。
- **novelty 残余：** 不读取 packet/text的 robot forward-effect evaluator + same-noise `swap/delay/reorder` intervention，用于证明动作确实消费了 event semantics。
- **所需数据：** native robot action/contact/effect、成功和失败/off-policy coverage、sensor-confirmed outcome；66 Ego与旧 robot grab均不够。
- **WAM 插入点：** 训练侧独立 forward-effect head/loss和 intervention evaluator；不新增 action-visible字段。

### F. Temporal Contact-Pose Tube

- **角色：HEADLINE，经裁决后成为 SM-FET-WAM v2-min。**
- **一句核心：** 以 Semi-Markov 事件时钟表达多阶段手姿态、接触变化、持续时间和效果，动作每次只读取 active+next2 hard packet。
- **最强冲突：** TOCH、ARCTIC、LatentAct、CAMS、VLAff、Being-H0、EgoVLA、ImMimic、VideoManip 已覆盖完整 hand/contact sequence；PPI/Keypose Exploration已覆盖 keypose接口。
- **novelty 残余：** 单帧与全轨迹之间的稀疏 event granularity + explicit duration/censor + exact finite WAM route + 多姿态仅辅助的可证伪组合。
- **所需数据：** 3+12 检验 ontology/pose可标性；36+36检验跨域事件；native robot oracle data检验 action sufficiency。
- **WAM 插入点：** 新 event schema/planner/compiler，13 hard tokens替代而不是追加到22D tokens。

### G. Event-Anchored Partial Human–Robot Pairing

- **角色：SUPPORTING，offline training data module。** partial OT本身不是 headline。
- **一句核心：** 只匹配 human/robot 共享的 contact/relation/effect events，允许 many-to-many、内部 NULL、两端 unmatched与不同时长。
- **最强冲突：** EgoBridge 已做 DTW-guided joint OT；Drop-DTW 已做保序部分匹配；ImMimic/GMS-SDTW处理 many-to-many和长视频子序列；HARP-VLA使用 paired strict temporal alignment。
- **novelty 残余：** 在极不对称双手上，用可审计 contact/effect字段和 missingness-aware sparse coupling证明对目标策略有独立价值。
- **所需数据：** 至少同任务 human/target-robot overlap；推荐36+36、12 cells、180条pair decisions。当前66 pour+88 grab只作NULL/hard-negative stress。
- **WAM 插入点：** `event_pair_v1` sidecar、matched-event distillation；CAPOT coupling stop-gradient且推理时完全删除。

### H. Event-Transition Graph WAM

- **角色：BASELINE。** 最有价值的 relation-before/after、effect/outcome已被v2字段吸收，不再加独立graph network。
- **一句核心：** 用 pre/during/post 的对象关系状态差与结果，而不是完整人手轨迹，表示工具操作。
- **最强冲突：** Ego4D PRE/CONTACT/PNR/POST、SWIM、Action Genome/EASG、BridgeACT、VRB和state-change表征均已覆盖 event/state graph、affordance与效果。
- **novelty 残余：** 有限、角色化、可干预的 state delta进入strict WAM route；该残余已并入SM-FET，不再单独主张。
- **所需数据：** pre/post relation、effect/outcome、contact/censor的 reviewed labels与robot oracle outcome。
- **WAM 插入点：** 作为 `ETG current-delta-only` 互斥 baseline，与v2比较；不能与v2 concat。

### I. Causal Event-Triggered Window Memory

- **角色：SUPPORTING，条件式 planner evidence front-end。** 不成为独立 WAM，也不进入首版。
- **一句核心：** 在线只从历史截取少量事件邻域短窗给planner，并用 keep/delete/exchange证明其动作充分性，避免一帧与全程两个极端。
- **最强冲突：** SKIP已覆盖event-preserving sparse embodied world model；SWEET、BridgeACT、VLAff/ST-VLA与视频token pruning覆盖关键帧/短窗/稀疏计划。
- **novelty 残余：** 严格在线因果历史、短窗口而非孤立帧，以及对action的operational ε-sufficiency干预而非重建指标。
- **所需数据：** 现有66 Ego可做无机器人结论的offline selector pilot；策略结论需同任务robot数据与closed-loop rollout。
- **WAM 插入点：** 只在event planner之前提供causal history evidence；selected raw windows、selector score/hidden不能进入action。

### J. Factorized Pose–Contact Derivative

- **角色：BASELINE / challenger。** 不与v2拼接。
- **一句核心：** 用少量2D姿态—接触变化原子而非绝对MANO或完整轨迹，检验更低容量是否足以跨embodiment。
- **最强冲突：** DMP/ProMP、LatentAct hand-contact codebook、UniT、DexGrasp-Zero、TopoRetarget与跨手latent已覆盖trajectory primitive、离散codebook和morphology alignment。
- **novelty 残余：** 2D可复核 derivative-only + exact `75.445813886781 bit` + asymmetric `6+1` 的受控比较。
- **所需数据：** 12段 pose/contact变化双标，随后同任务robot oracle与dynamic regrasp/rotation子集。
- **WAM 插入点：** `f_pcd_v1_challenger` 互斥 action route，或仅training auxiliary；不能作为v2新增字段。

### K. Relational Event Quotient Supervision

- **角色：SUPPORTING，后置training-only grouping。** 首轮不启用。
- **一句核心：** 将 `pre-relation -> 1–3 operation events -> post-effect` 相同的不同human/robot执行组成多正例组，而不要求动作轨迹相似。
- **最强冲突：** Affordance Equivalence已形式化不同agent/action产生同effect；MIR、FUNCTO、MimicFunc、RHyME、XSkill/UniSkill覆盖effect、functional frame、mismatched execution与skill prototype。
- **novelty 残余：** finite reviewed relation/effect projection + counterfactual wrong-effect negatives + strict v2 route；action-visible增加0 bit。
- **所需数据：** 36+36中可靠的pre/post relation、effect/outcome与多正例/hard-negative group；CAPOT-only必须是强baseline。
- **WAM 插入点：** candidate gate、sampler、contrastive/ranking loss；quotient ID、matcher hidden或group ID不进action。

### L. Sparse Interaction Motif Dictionary

- **角色：NO-GO（当前首轮和独立headline）。** 只有event recurrence和matched-output planner gate通过后，才可重新审查为episode-level challenger。
- **一句核心：** 从长度2–4的contact-effect event subsequence挖掘可复用macro，再确定性展开成v2 event tube。
- **最强冲突：** MOTIF 2026直接覆盖cross-embodiment action motif；XSkill、CompILE、LOTUS、PRISE覆盖skill/codebook；Drop-DTW、StepFormer、GMS-SDTW、REALIGN覆盖稀疏/部分子序列。
- **novelty 残余：** contact/relation/effect atoms组成的有限macro compiler，且不增加action bits；但当前只是窄组合，不足以抵消直接冲突。
- **所需数据：** 至少72段跨domain事件，macro effect purity、跨group复现、prep invariance与强alignment baseline均通过。
- **WAM 插入点：** 若未来复活，只能作episode planner prior并确定性展开为现有v2；learned motif latent不得直达action。

## 5. 明确 No-Go 清单

以下不属于“先试试看”，而是当前研究设计直接拒绝：

1. 透明物体/液体 dense 3D reconstruction、伪depth或无传感flow/volume GT。
2. 把单张 hand grasp pose当完整工具使用表示。
3. 把完整human MANO/wrist trajectory逐帧retarget为robot action主线。
4. 从拿取前到结束强制human–robot完整1:1或complete-DTW正配对。
5. 将现有66 human pour与88 robot grab组成正pair或做paired transfer。
6. 将`hand_status`、tracker valid、mask overlap或gripper command命名为physical contact truth。
7. 将v1 22D、v2、F-PCD、continuous pose、ETG token全部concat后做一个“大模型”。
8. 将CAPOT、RE-Q、SCEMC、CET-WM或forward-effect全部默认打开。
9. 当前启动5B/GPU/server训练或目标硬件部署。
10. 声称人类已被证明用固定八阶段/三姿态替代完整轨迹，或用mirror system为模型机制背书。

## 6. 为什么不能机械拼接

### 6.1 无法做科学归因

如果同时增加event clock、pose tube、graph、partial OT、motif、effect cycle、visibility head与hardware allocator，最终success变化无法归因。每个模块还改变标签量、参数量、action-visible bits和训练数据，任何正结果都可能只是“更多监督”。

### 6.2 会破坏有限通道claim

22D、v2、F-PCD、ROI/pose/graph latent一旦并行进入action，`207.7497 bit`就失去意义。planner hidden、heatmap、CAPOT coupling或human memory更会形成不可枚举的视觉旁路。

### 6.3 多套时钟会互相矛盾

F的Semi-Markov event、H的ETG、I的关键窗口和L的macro如果各自定义事件边界，就会产生四套phase pointer。最终只保留SM-FET clock；CET-WM只提证据，SCEMC只能编译到同一clock。

### 6.4 source morphology不是target control

human finger identity、MANO pose和hand-side不能直接映射到左RH56/右OmniPicker。共享层只能保留角色、接触部位、关系变化和效果；执行层必须读机器人本体状态并满足硬件可行性。

### 6.5 当前标签不支持全部模块

66 Ego没有完整contact/event/outcome/group truth，88 robot数据task/hardware又不匹配。现在堆模块只会训练proposal bias，不能验证human-to-robot skill transfer。

## 7. 数据路线与 benchmark

### 7.1 现在可做：不等新机器人数据

1. 冻结 `sm_fet_hard_grammar_v2_standard_r1`、四role mapping、UNKNOWN/ABSENT/MISSING/censor规则。
2. 准备D0所需的annotation→82-code机械映射、round-trip与overflow检查设计；未实现checker前不得称D0通过。
3. 完成3段A/B calibration，再对12段human-pour做盲法双标+第三人裁决。
4. 每段标 `events_v1`、`event_node_snapshots_v1`、`event_edges_v1` 和action-invisible `pose_keys_2d_v1`。
5. 报告逐字段PASS/FAILED/NOT_EVALUABLE、unknown、overflow、边界误差与人时；协议最多修订一次。
6. 将旧66 pour与88 grab只设计成`legacy_task_mismatch_stress`，不生成66×88伪样本量。

预计当前3+12人工成本约 `14.5–20` 人时。它只验证单任务可标性，不能解锁完整schema、配对或模型。

### 7.2 等目标硬件同任务数据后做

推荐最小数据为：

```text
3 task families
* 2 object-instance conditions
* 2 session/day groups
= 12 cells

每cell：3 human clips + 3 target-robot rollouts
总计：36 human + 36 robot
```

三任务族：

1. `pick_place`：右OmniPicker主抓放，作为单末端基线；
2. `stabilize_twist_cap`：右夹爪稳瓶、左RH56扭盖/regrasp；
3. `pour_transfer`：建议左RH56持source、右OmniPicker稳target，最终左右分工必须先过可达性/安全审查。

所有72段做单人event review；至少15段跨family/domain双标；至少180条A/B pair decisions，每family各20 match、20 unmatched、20 hard negative。预计采集+标注约 `50–76` 人时，不含driver/mapper/watchdog开发。

配对 benchmark必须含：strict/normalized time、DTW/soft-DTW、Drop-DTW、GMS-SDTW/subsequence DTW、EgoBridge-style balanced OT、partial OT、non-monotonic UOT、CAPOT与oracle manual edges。

### 7.3 数据通过后才允许的 oracle 阶段

需要 native/sensor-confirmed robot current state/contact/outcome。先比较互斥route：

- direct current-reference WAM；
- 22D-only；
- F-PCD-only；
- ETG current-delta-only；
- SM-FET active-only / active+next1 / active+next2；
- SM-FET equal-bit；
- local/full continuous trajectory upper bound。

D3门槛：v2相对oracle 22D-only在outcome-sensitive success `>=+8 pp` 或event completion `>=+10 pp`，相对full trajectory下降不超过`3 pp`，安全不退化且cluster CI支持正优势。失败则SM-FET action route No-Go，不通过扩大模型挽救。

### 7.4 服务器训练后才做

只有D0–D3全部通过，才在服务器进入predicted/causal route：

1. predicted planner恢复至少70% oracle gain；
2. strict v2优于auxiliary-direct、22D-only与matched equal-bit；
3. group-OOD有预注册正增益和cluster CI；
4. field intervention呈phase locality；
5. cache/reference swap/side-channel审计通过；
6. 之后才依次审查E effect-cycle、I CET-WM、K RE-Q；L SCEMC仍需独立复活gate。

服务器阶段也不能跳过目标硬件driver、canonical-to-device reverse mapper、calibration、watchdog、E-stop、bench/robot test和数据receipt。旧dual-Inspire数据不能解锁这些gate。

## 8. 预注册停止条件

- 3+12 的event F1 `<0.80`、关键κ不达门槛、unknown/overflow持续过高：停止v2 ontology扩量。
- CAPOT不比Drop-DTW/GMS-SDTW/EgoBridge-style最优者高至少5 pp且CI不支持正优势：删除pairing novelty，最多作数据工具。
- oracle v2不优于22D或显著差于full trajectory：停止headline action route。
- 只有continuous pose/MANO/ROI action-visible才有效：停止finite sufficiency claim。
- equal-bit与semantic v2持平且利用率相当：停止semantic field necessity claim。
- robot-only/no-pair与CAPOT相当：停止human–robot pairing contribution。
- event windows不优于同预算uniform/random：停止human-inspired event selection claim。
- effect forward model不优于no-action baseline：停止contact-action cycle。
- fixed八类事件跨任务unknown/overflow超门槛：不事后加类别保护结果，先判r1 ontology No-Go。

## 9. canonical `e212afec` 代码位置与未来插入面

以下位置是当前实现事实或未来任务书，不表示v2已实现：

### 9.1 已存在的v1 strict contact基线

- `world_action_model/contact_transition_schema.py:18-217,270-413,624-655`：v1/v1.1 schema、hard validator、容量、ordered segment编码。
- `world_action_model/models/transformer_wa_casual_mot.py:1253-1285`：当前contact planner；planner hidden不导出。
- `world_action_model/models/transformer_wa_casual_mot.py:1358-1403`：contact route与finite controls。
- `world_action_model/models/transformer_wa_casual_mot.py:1405-1495`：当前22→action-width的唯一planner-to-action投影。
- `world_action_model/models/transformer_wa_casual_mot.py:127-194,1806-1822`：MoT attention mask与strict image旁路风险点。
- `world_action_model/datasets/wam_lerobot_dataset.py:285-325,711-819`：reviewed contact sidecar preflight和逐episode加载。
- `world_action_model/trainer/wa_casual_trainer_mot.py:111-189,555-598,957-1055`：grouped contact loss、route receipt和batch/loss入口。
- `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py:460-473`：当前`state/ref/text planner -> hard 22D -> action` contract与capacity receipt。

### 9.2 v2未来新增位置

- 新建 `world_action_model/event_tube_schema.py`：node/event/edge code、clock、compiler、validator、rank/unrank、capacity与receipt。
- 新建 `world_action_model/models/event_tube_planner.py`：factorized heads、13-token encoder与hard transport。
- 在 `transformer_wa_casual_mot.py:599-631,703-870` 加opt-in v2 config/module；默认disabled且与v1互斥。
- 在 `:1253-1403` 邻近增加event planner/route；在`:1405-1495`用factor embedding生成13 tokens，不复用22D linear。
- 在`:1531-1590`返回factor logits、canonical packet和receipt；GT只到loss或显式oracle route。
- trainer新增event categorical/survival loss，但predicted action forward不得收到GT packet。
- dataset沿现有strict sidecar模式新增episode级sparse event sidecar；coupling不复制成frame-level action输入。

### 9.3 不对称25D硬件边界

- `world_action_model/asymmetric_action_schema.py:20-166`：canonical 25D schema、pack/split/pad/unpad与mask。
- `world_action_model/asymmetric_hardware_manifest.py:172-710`：manifest、标定、动作与receipt校验。
- `docs/asymmetric_action_schema.md:12-24`：`left wrist 9 | right wrist 9 | RH56 6 | OmniPicker 1`。
- `docs/asymmetric_action_schema.md:136-140`：仓库没有OmniPicker driver、reverse mapper、endpoint、watchdog和bench/robot闭环，部署No-Go。
- `docs/asymmetric_action_schema.md:158-164`：旧88段数据是dual-Inspire，不是目标硬件。

## 10. 人类观察学习证据的正确边界

可作为设计动机：

- Pomp et al. 2021：touching/untouching与主观事件边界高度相关，但多数contact motif并不成为边界。
- Bunday et al. 2016：动态抓取视频比单张接触帧更能诱发抓型特异运动共振。
- Roberts et al. 2015：自然/恒速轨迹产生观察学习，endpoint-only没有；精确生物运动学不是必要条件。
- goal/outcome、前瞻注视、跨效应器结果研究支持功能层级和稀疏注意。

不能写：

- 人类被证明只看固定几个阶段；
- 固定八事件是认知状态机；
- regrasp是普适人类观察阶段；
- mirror system证明机器人应使用同构hand pose code；
- 认知文献本身证明SM-FET会改善robot policy。

目前没有发现容量匹配的三臂人类工具学习实验直接比较 `single snapshot vs event multipose vs full trajectory`。因此论文只能把SM-FET称为**受人类事件分段与动态观察证据启发的、可证伪工程归纳偏置**。

## 11. 证据 memo 与 idea IDs

### 11.1 历史总览与最终规范

- op22 dossier：`/home/ykj/project/WM/gwp-agent-op22-idea-synthesis/docs/research/contact_transition_lab_wam_dossier_20260811.md`，commit `ce56584`。
- temporal meta：`/home/ykj/project/WM/gwp-agent-temporal-event-meta/docs/research/temporal_event_pairing_wam_meta_synthesis_20260812.md`，`idea_id=b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`，`discussion_id=4ccb56ec-e102-4d76-b542-080aa0ea3280`。
- schema：`/home/ykj/project/WM/gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_schema_architecture_20260812.md`。
- annotation/pairing：`/home/ykj/project/WM/gwp-agent-temporal-event-meta/docs/research/sm_fet_annotation_and_pairing_protocol_20260812.md`。
- final architecture：`/home/ykj/project/WM/gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_final_architecture_decision_20260812.md`。
- cognitive evidence：`/home/ykj/project/WM/gwp-agent-human-observational-learning/docs/research/human_observational_tool_learning_evidence_20260812.md`，`idea_id=f3dfb6e9-d244-4188-95f7-6aca1c537947`，`discussion_id=5c525c2a-5a71-4ab8-9b22-c17a87667783`，commit `52e763c`。

### 11.2 A–L

| Idea | evidence memo | idea_id | discussion_id | commit |
|---|---|---|---|---|
| A | `/home/ykj/project/WM/gwp-agent-idea-a-finite-contact-wam/docs/research/idea_a_finite_contact_wam_review_20260811.md` | `2fd0a8a6-0efa-4038-9aeb-93a8378e41c0` | `44cf5abe-05c6-477d-93ce-c3dad8a6a3eb` | `279ddf4` |
| B | `/home/ykj/project/WM/gwp-agent-idea-b-asymtopology/docs/research/idea_b_asymmetric_contact_topology_review_20260811.md` | `24a3b591-7839-4593-8cef-169e01acf1d6` | `5e9ca133-76e3-4818-8322-fa6e60185a61` | `07821f0` |
| C | `/home/ykj/project/WM/gwp-agent-idea-c-vis-contact/docs/research/idea_c_visibility_contact_aux_review_20260811.md` | `5757566f-4c34-4cf1-aecf-0dfac76a8ab9` | `e1e27a11-cb4b-4ac8-bef8-28dd3e51f634` | `de3cbc9` |
| D | `/home/ykj/project/WM/gwp-agent-idea-d-censored-contact/docs/research/idea_d_censored_contact_benchmark_review_20260811.md` | `15d89a15-c958-4a33-9735-a9547d45fa27` | `dbf2e3a4-9d4d-436a-9e74-4d727ca8d037`; librarian `90187152-d080-45c4-b5ef-b9f12f1a63a0` | `cd6629f` |
| E | `/home/ykj/project/WM/gwp-agent-idea-e-contact-action/docs/research/idea_e_contact_action_consistency_review_20260811.md` | `8f386f3f-50be-45b5-970d-4bca4bd1a062` | `bba6e215-6a61-4203-9c03-696c17a42bf5` | `1534ede` |
| F | `/home/ykj/project/WM/gwp-agent-idea-f-temporal-tube/docs/research/idea_f_temporal_contact_pose_tube_review_20260812.md` | `a78673af-f56e-4ad1-9974-beee1edb765b` | `8562848c-0b71-485e-abff-064f0900159c` | `20421c7` |
| G | `/home/ykj/project/WM/gwp-agent-idea-g-partial-pairing/docs/research/idea_g_partial_human_robot_alignment_review_20260812.md` | `e6561894-fa5a-4e44-912f-24524aac7764` | `4cbab6ef-7c05-412b-a600-3797f393370e` | `9f60f04` |
| H | `/home/ykj/project/WM/gwp-agent-idea-h-event-transition/docs/research/idea_h_event_state_transition_wam_review_20260812.md` | `f759ccbb-735b-45e4-8efe-158b79a3ce90` | `2ca05e6f-75c5-49df-bbc9-2c8a68d292c9` | `8f0432a` |
| I | `/home/ykj/project/WM/gwp-agent-idea-i-causal-critical-windows/docs/research/idea_i_causal_critical_windows_review_20260812.md` | `a0da9ceb-7748-4ef5-910c-e57b7cf041dd` | `08578342-f03a-4e9d-a454-0fe1f373b2ef`; follow-up `a32a9415-7fc6-464d-9bc3-97ee10c0ccf8` | `0b67bee` |
| J | `/home/ykj/project/WM/gwp-agent-idea-j-pose-contact-change/docs/research/idea_j_pose_contact_change_signature_review_20260812.md` | `383b218b-5c66-4503-b977-530c7f7b3094` | `611d78e5-f36f-4332-b2b5-abebb699925a`; librarian `4781abe9-bac9-40cd-8408-31acd30deff2` | `85dc53c` |
| K | `/home/ykj/project/WM/gwp-agent-idea-k-effect-equivalence/docs/research/idea_k_effect_equivalent_interaction_quotient_review_20260812.md` | `2823cc3b-c066-4e44-ac77-b445f260a683` | `4749b58a-5b7c-4ff9-ab5b-87c30ea6b95d` | `23e3834` |
| L | `/home/ykj/project/WM/gwp-agent-idea-l-motif-dictionary/docs/research/idea_l_sparse_interaction_motif_dictionary_review_20260812.md` | `76f0e0a1-cc36-4e72-88ed-fb063f752297` | `6707f93d-9734-4ca1-8858-fa9b24cd6ec1` | `f2059a0` |

## 12. 最终一句话

这项工作的论文价值不在“又加一个contact token”，而在于证明或证伪：**相较单帧姿态和全程人机轨迹复刻，一个带显式位置、持续时间、接触/关系变化、效果、missingness与部分配对的有限事件管道，能否成为Ego视觉到异构双手WAM控制的唯一、足够且可干预的桥。** 在同任务目标硬件数据和oracle证据出现前，最有价值的工作仍是把这个命题做成可标、可编译、可拒绝错误配对的实验契约，而不是先训练更大的模型。
