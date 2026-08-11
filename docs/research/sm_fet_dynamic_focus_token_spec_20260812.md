# SM-FET dynamic focus 规格：不增加连续位置 token 的跨具身“操作哪里”契约

- 日期：2026-08-12
- 独立审查分支：`codex/sm-fet-focus-token-20260812`
- canonical 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 上位规范：`event_tube_v2_standard / sm_fet_hard_grammar_v2_standard_r1`
- 研究范围：original issue 1——visual embodiment gap 下，human/robot 不对齐时机器人如何知道操作对象、部位与位置
- 非本轮内容：代码、UI/checker、训练、GPU、服务器、仿真或真机
- 复用总 `idea_id`：`b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`
- 本轮 `discussion_id`：`092fc9ae-f268-4aa1-b79e-69cb44e6afe9`

## 0. 唯一裁决

**加入“操作位置”语义，但不新增 wire 字段、不新增 action token，也不传一个 human 像素点。**

唯一推荐是把 standard-r1 已有的 `focus_node + current node geometry + primary part/cell + secondary relation` 正式定义为一个确定性的 **Dynamic Event Focus handle**：

```text
FOCUS_BINDING_VERSION = "sm_fet_dynamic_event_focus_v1"

active/next event
  -> event.focus_node                         # 当前事件关注哪个功能角色
  -> referenced current node token            # 它在机器人当前视图哪里、是否可见
  -> deterministic anchor edge                # 它的哪个部件/局部格是作用位置
  -> secondary node + relation/effect          # 对谁作用、预期形成什么关系/效果
```

它不是第 14 个 token，而是 `PackedEventTubeEncoder` 对现有 13 个 hard tokens 做的确定性引用绑定。action 仍只看：

```text
robot proprio/state + allowed text + noisy action/diffusion timestep
+ 4 node tokens + 3 event tokens + 6 edge tokens
```

因此：

- standard payload 仍为 `82 uint8 codes`；
- action tokens 仍为 `13`；
- exact reachable action alphabet 仍为
  `345864043392188289642033658185203030253953267411599196753571073`；
- exact action-visible capacity 仍为 `207.74974691951957 bit`，固定容器宽 `208 bit`；
- `FOCUS_BINDING_VERSION`、anchor-edge index、诊断用像素投影和 tracker receipt 都属于 control plane，不进 action。

一句话回答原问题：**机器人不是从 human 图像坐标猜抓点；human 与 robot 只共享“功能角色—部件—关系/效果”，目标机器人在自己的当前相机中重新定位这个语义焦点，再用自己的 proprio、运动学和闭环控制执行。**

## 1. 先把三个问题拆开

“human 和 robot 不对齐，机器人怎么知道抓哪里”其实包含三种不同量，不能压成一个二维点：

| 层 | 回答的问题 | 推荐表示 | 是否跨域共享 |
|---|---|---|---|
| 功能语义 | 操作谁、哪个部件、为了什么效果 | role、part、relation、effect、event kind | 是；这是 human/robot 共享层 |
| 当前视图定位 | 这个实体在 robot 当前相机哪里 | observability、node center/scale/orientation、part-local cell | 否；各域独立估计，robot 部署时重定位 |
| 目标硬件控制 | 哪只末端以什么姿态/力/路径到达 | robot proprio、kinematics、tactile/current、action decoder | 否；只属于目标 embodiment |

错误做法是把第一层和第三层都替换成 human 图像中的 `(u,v)`：同一把手在不同相机、裁剪、焦距、物体姿态和机器人本体遮挡下没有同一个像素位置；即使 human 点本身正确，也不是 RH56/OmniPicker 的可执行控制量。

## 2. 文献给出的结论与不能照搬的部分

### 2.1 “哪里”确实重要

- [VLAff](https://arxiv.org/abs/2608.05215) 从大规模 egocentric 视频联合学习视觉热图、抓取姿态和轨迹 affordance，并明确把它们解释为 where/how-to-grasp/how-to-move；其消融中视觉 affordance 对 manipulation learning 很关键。但它依赖 3D SfM、hand mesh、heatmap、grasp pose、trajectory 和 target-scene 3D，不是有限 action channel。
- [CoA-VLA](https://arxiv.org/abs/2412.20451) 在 action 前显式推理 object、grasp part、placement space 和 collision-free movement affordance，说明只说物体名不够；其 visual/text co-injection 是连续表示，不能直接拿来证明 finite token sufficiency。
- [Point-VLA](https://arxiv.org/abs/2512.18933) 用 bbox 视觉指称解决 clutter/OOD 中语言歧义，支持“语言不够，必须有视觉 grounding”；它解决 object-level reference，不等于精细 contact location，更不建立 human pixel 到 robot control 的对应。
- [RoboPoint](https://arxiv.org/abs/2406.10721) 直接预测语言条件 image keypoint affordance，证明单点是有竞争力的 locator baseline；它仍需下游机器人模块把图像点转成可执行动作。

### 2.2 静态点/框不足以覆盖动态操作

- [VRB](https://arxiv.org/abs/2304.08488) 的表示是 contact points 加 post-contact trajectories，而不是孤立接触点，并指出字面 human motion 会因 morphology 差异失去泛化。
- [ATM](https://arxiv.org/abs/2401.00025) 预测任意图像点的未来轨迹，用少量 action-labeled robot data 学 policy；它支持 trajectory 作为强上界，但点轨迹是高带宽、相机相关的 action guidance。
- [Track2Act](https://arxiv.org/abs/2405.01527) 从 web/human/robot 视频预测 2D point tracks，再估计 rigid transforms，并用少量 embodiment-specific demonstration 学闭环 residual；这恰好说明 2D track 不能省掉 target-embodiment 转换器。
- [3PoinTr](https://arxiv.org/abs/2603.08485) 用 visibility-aware dense 3D point tracks 从 unconstrained human video 预训练，再用 20 条 action-labeled robot demos 学闭环策略；它是 event-conditioned track upper bound，而不是零成本的共享控制量。
- [JOPAT](https://arxiv.org/abs/2605.23856) 在 world-action model 中联合预测 latent pixels、2D tracks、visibility 和 actions，长时遮挡任务收益更大；它支持 point-track auxiliary 的价值，但其 track hidden/action joint route 明显超过本项目的 strict finite channel。
- [TraceVLA](https://arxiv.org/abs/2412.10345) 把 state-action trajectory 画成视觉 trace prompt，能增强时空理解；它是图像级提示，不具备本项目所需的可枚举容量与 bypass 审计。

### 2.3 3D/4D affordance 是上界，不是当前主路线

- [ST-VLA](https://arxiv.org/abs/2603.13788) 把 2D guidance 转成 3D trajectory，并用 smooth spatial masks 提供 4D 时空上下文与在线重规划。它直接说明仅有 2D 静态 token 可能丢失深度和时间一致性，但也说明要获得该能力需开放连续 3D/4D 表示。
- [BridgeACT](https://arxiv.org/abs/2604.23249) 把 human-to-robot 分解为 where-to-grasp 与 how-to-move：先在 target scene grounding affordance region，再预测 3D motion affordance，最后由 grasp module 和 closed-loop controller 映射成 robot action。主线应继承“target scene 重新 grounding + embodiment-specific executor”，而不是继承其连续 3D channel。
- [KITE](https://arxiv.org/abs/2606.22113) 把 contact-pattern interaction intent 与 embodiment-specific action decoder 分开，进一步支持“共享交互意图、分离运动控制”的架构边界；其 learned latent 不能在本项目中绕过 hard schema。

### 2.4 独立判断

这些论文共同否定了两个极端：

1. 只有语言或一个静态 object ID，通常不足以解决 clutter、部件和动态关系；
2. human point/trajectory 也不能无需转换地直接控制异构机器人。

它们没有验证“207.75-bit finite event packet 足够”。因此本 memo 的 Dynamic Event Focus 是要由互斥 baseline 和 stop condition 证伪的工程假设，不是由相关工作自动证明的结论。

## 3. 五种候选的严格比较

| 候选 | 优点 | 决定性缺陷 | 在本项目中的角色 |
|---|---|---|---|
| 语言指称 | 易跨视角，能表达角色/属性 | clutter 中歧义；难精确表达 part/cell；text 已合法直达 action，容量不可与视觉 route混算 | 必做 baseline，不作唯一 locator |
| 单点 token | 容量低、易标、RoboPoint/VRB 有前例 | 视角相关；不能表示对象范围、姿态、遮挡、第二对象关系；一个点不含接触后的移动 | 互斥 finite baseline |
| bbox/part token | object reference 比单点稳定；part 有功能语义 | bbox 仍是相机坐标；整框不等于接触位置；连续框易形成高带宽旁路 | `Point-VLA` 风格 baseline；主路只保留粗 node geometry + part |
| event-conditioned point/track | 表达方向、时长和动态 interaction | 高容量；对相机/重建/track drift 敏感；需要 target embodiment residual/controller | continuous upper bound 或 planner auxiliary |
| **Dynamic Event Focus** | event-conditioned、角色化、part-local、可重定位、可枚举、能表达双对象关系 | 空间分辨率有限；滑动/插入等连续接触可能不够；依赖 target robot 当前感知 | **唯一主路线** |

不是因为 Dynamic Event Focus 表达力最强，而是因为它是唯一同时满足以下约束的候选：

- human/robot 共享的是功能语义，不是相机坐标；
- robot 部署时用自己的当前 observation 重定位；
- action 只读可反序列化 hard codes；
- 仍保留 part-level contact 与 object-object relation；
- 其不足可以由 track/3D/continuous upper bound直接否证。

## 4. `sm_fet_dynamic_event_focus_v1` 精确契约

### 4.1 不新增字段；冻结使用以下已有 vocab

#### Event focus

```text
event.focus_node:
  0 PAD
  1 manipulated_or_source
  2 tool
  3 receptacle_or_destination
  4 fixture_or_support
```

`focus_node` 的语义改为精确的一句话：**当前事件中，action 必须优先重定位且其误定位最可能导致事件失败的 task-role node。**

它不是“画面最显眼对象”，也不必永远是手正在抓住的对象。例如：

```text
acquire bottle       -> manipulated_or_source
manipulate pour/rim  -> receptacle_or_destination
verify transfer      -> receptacle_or_destination
twist cap            -> tool（若 task ontology 将 cap 绑定为 tool）
```

同一 active event 内 `focus_node` 固定；变化必须发生在合法 event transition。这样 dynamic 指“随事件状态切换”，不是逐帧抖动的 attention peak。

#### 当前 robot 视图中的 node locator

```text
node.observability:
  ABSENT | VISIBLE | PARTIAL | OCCLUDED_TRACKABLE |
  OUT_OF_VIEW | UNKNOWN

node.center_cell:
  PAD | 4x4 image cells (1..16) | UNKNOWN

node.scale_band:
  PAD | SMALL | MEDIUM | LARGE | UNKNOWN

node.orientation_2d:
  PAD | 8 x 45-degree bins | UNKNOWN
```

这些字段回答“该 role node 在当前 robot image 里大致哪里”。它们每次 replan 从 robot 当前 observation 重编译；human node coordinates 不复制给 robot。

#### 部件与作用位置

```text
edge.primary_part:
  PAD | BODY | HANDLE | LID_CAP | OPENING | RIM_EDGE |
  TRIGGER_BUTTON | TOOL_TIP | OTHER | UNKNOWN

edge.primary_surface_cell:
  PAD | part-local 4x4 cells (1..16) | UNKNOWN
```

`OTHER` 是已知部件不在词表；`UNKNOWN` 是已复核但不可判；`NOT_REVIEWED` 不编 target。`primary_surface_cell` 是当前域、当前对象部件局部框中的粗格，不是 human/robot 共享的绝对 `(u,v)`。

#### 第二对象与目标关系

```text
edge.secondary_node:
  NONE | four fixed roles

edge.relation_before / relation_after:
  N/A | SEPARATE | NEAR | TOUCHING | OVERLAP_2D |
  INSIDE | CONTAINS | ALIGNED_TO_RIM | SUPPORTED_BY | UNKNOWN

event.object_effect:
  PAD | NONE | TRANSLATE | ROTATE | ARTICULATE |
  FLOW | DEFORM | TRANSFER | UNKNOWN
```

这些字段回答“作用位置相对哪个对象、要产生什么变化”。对 pour 等任务，目标 rim 可作为 `OBJECT_OBJECT` edge 的 primary，source 为 secondary；手对 source 的接触由另一条 edge 表达。

### 4.2 anchor edge 的唯一确定性规则

不增加 `focus_edge_index`。对每个 present event slot `e`：

```text
C_e = all present edges of event e whose primary_node == event.focus_node

anchor_edge(e) = first edge in the existing canonical edge order, if C_e nonempty
               = NONE, otherwise
```

由此机械派生、但不序列化：

```text
NODE_ONLY   = anchor_edge is NONE
PART_ONLY   = anchor_edge present and primary_surface_cell == UNKNOWN
PART_CELL   = anchor_edge present and primary_surface_cell in 1..16
```

该派生状态只用于 encoder gather、日志、评测分层和 fail analysis，不是 action 类别。若两个 edge 的 primary 都等于 focus，现有 canonical sort 保证选择唯一；action 仍可读取另一条 edge，不会丢失双手/双对象信息。

### 4.3 动态更新规则

```text
episode level:  event.focus_node 随合法 event transition 切换
active event:   focus_node 与 anchor primary part/cell 在 commit 后冻结
every 8 steps:  node observability/center/scale/orientation 从 robot 当前视图刷新
future event:   focus/part/cell 是 provisional，可在晋升 active 前更新
cache:          node bytes、event transition、binding receipt 或 packet bytes变化即失效
```

active event 中 relation/contact observation 可以更新，但不能把一个逐帧 moving pixel 偷塞进 focus。若任务确实需要同一 maximal-duration `manipulate` 内沿表面连续滑动多个 cell，event-conditioned track/full trajectory 应赢；这是 finite route 的预注册失败模式，不以事后增加更多 cell掩盖。

### 4.4 编码器中的“位置 token”究竟是什么

逻辑上每个 event token 形成：

```text
event factor embedding
+ gather(node_token[event.focus_node])
+ gather(edge_token[anchor_edge(event)]) if present
```

这是对 canonical hard codes 的确定性 gather/fusion；不是 planner hidden、cross-attention ROI、heatmap pooled feature或 raw box embedding。node/edge tokens 仍保留为独立 token，便于 action 同时看到第二对象关系和另一只手。

该 fusion 可以增加 encoder 参数/FLOPs，但不增加 observation alphabet。公平实验必须给所有 finite locator baseline 相同 token 数、embedding width、fusion module参数量与 action FLOPs。

## 5. human 与 robot 两域如何对应

### 5.1 共享与不共享的字段

| 字段 | human/robot 是否可对齐 | 规则 |
|---|---|---|
| event kind、focus role | 对齐 | 任务 ontology 固定 role→slot；不按图像中左右位置匹配 |
| primary part | 对齐 | 同一功能部件词表；OTHER/UNKNOWN严格区分 |
| relation/effect/outcome | 对齐 | 需要相同 task/effect semantics 与可观察判据 |
| literal left/right actor | 不对齐 | human hand side 不决定 RH56/OmniPicker 分配 |
| global node center/scale/orientation | 不对齐 | 每个相机独立编译；不能作为 human→robot 坐标 target |
| part-local cell | 条件对齐 | 只有冻结了同一 entity/part canonicalization 且双域可复核时才进 pairing cost；否则 mask |
| continuous point/bbox/track | 不对齐 | 只作各域 auxiliary、proposal或 upper bound |
| robot current hand/contact state | robot only | 必须来自 proprio/tactile/current；human pose不能补 |

### 5.2 human annotation

沿修订 protocol：

1. `events_v1.csv` 写 `focus_role/focus_entity_id`；focus 是 event-level semantic binding。
2. `event_node_snapshots_v1.csv` 在 `entry/turn/exit/anchor` 上固定四 role 行，记录 human 当前视图的 observability/center/scale/orientation。
3. `event_edges_v1.csv` 写 primary role/part/part-local cell、secondary role、relation与effect；最多两条 edge。
4. `pose_keys_2d_v1.csv` 的 hand landmark/reference bbox仍只作 auxiliary/probe，不是 focus wire 来源。

为审计动态 focus，未来 sibling packet 需要新增一张 **annotation-only** 的 `event_focus_anchors_v1.csv`，而不是给 action 增字段：

```text
schema_version, dataset_id, episode_id, event_id,
focus_anchor_id, anchor_role, frame_index,
focus_role, focus_entity_id,
primary_part, primary_surface_cell,
focus_observability,
part_review_state, cell_review_state, observability_review_state,
evidence_scope, evidence_tier,
annotation_role, annotator_id, review_status, parent_focus_anchor_ids
```

约束：

- `anchor_role=entry|turn|exit|anchor`，与 node/pose 的实际查看帧一致；
- `focus_role/entity` 必须与 event row及任务 ontology一致；
- cell 在该帧该域的 part-local 4×4 中标，不保存为跨域坐标；
- `KNOWN/REVIEWED_UNKNOWN/NOT_REVIEWED` 明确分离；遮挡不可补点；
- 它用于检查 focus 是否在 event 内稳定、part/cell 是否可标、是否需要 track challenger；
- human sparse anchors不得插值成 robot chunk-level oracle。

`events_v1 + node snapshots + event_edges` 仍是 standard packet 的 schema来源；`event_focus_anchors_v1` 是对它们的时间一致性审计和 planner auxiliary，不改变 208-bit core。

### 5.3 robot annotation/compilation

robot 域必须独立生成当前 packet：

```text
robot current RGB / causal history
+ native object state or receipt-bound causal tracker
+ robot proprio/tactile/current
-> robot event planner
-> robot-view node codes + semantic focus/edge codes
-> hard compiler / validator
-> Dynamic Event Focus gather
-> action
```

oracle 阶段的 current node/focus truth应来自 native/sensor-confirmed object state或经正式 A/B 审核的 robot-view annotation；不能把 human cell warp 到 robot image当 GT。predicted 阶段由 robot planner从自己的 current observation预测。

目标硬件的 actor/末端分配在 hard packet之后由 embodiment allocator决定，并受 RH56/OmniPicker 可达性、proprio和安全约束。共享 focus 只表达“作用哪个功能部件以造成什么效果”，不表达 robot joint、wrist pose、gripper aperture或力。

### 5.4 对现有 CAPOT cost 的必要修正

修订 annotation protocol 的 `d_object_local_geometry` 当前允许 human/robot node `center/scale/orientation` 进入主 cost。跨相机、跨视角时这会把视图布局误当功能对应，必须收窄：

```text
CAPOT main cross-domain geometry:
  allowed = primary part
          + part-local cell only when canonicalization receipt is valid
          + relation/effect/outcome

  masked  = global image center
          + bbox/scale
          + image-plane orientation
          + raw point/track
```

node center/scale/orientation仍可用于同域 planner loss和 robot action packet，但不作为默认 human–robot pairing distance。若未来有标定到同一 object-centric frame的可靠 receipt，可把 canonical part cell作为预注册 ablation恢复；仍不得比较 raw pixels。

## 6. planner/action 可见性与防旁路规则

### Planner 可见

```text
robot current image
causal robot history
robot proprio/state
instruction text
previous hard EpisodeClockState
```

训练期 planner可有 stop-gradient 的 heatmap、keypoint、bbox、point-track、pose或3D辅助目标；部署时这些连续 teacher不进入 action。

### Compiler 可见

只接收 planner categorical outputs、clock和 current robot node binding，执行 canonicalization、reference union、anchor-edge派生、overflow和 receipt检查。

### Action 可见

只接收 robot state/text/noisy action与13个hard tokens。Dynamic Event Focus fusion只能 gather这些hard tokens。

### Action 禁止

```text
human frame / human pixel / human track
raw robot RGB/reference token
continuous point/bbox/mask/heatmap/ROI feature
future visual / 3D trajectory / point-track hidden
planner logits/hidden/confidence/posterior
tracker feature/KV or CAPOT pair/coupling
```

strict route继续要求 state、packet、action rows对 reference不可达，并拒绝 image cross-attention。固定 packet/state/text/noise后替换 robot reference，pre-safety action应保持 `max_abs<=1e-6`；改变 human视频、pair table或连续 focus auxiliary时 action必须 bit-equal。

## 7. 容量与为什么不另加 token

### 7.1 主 route

Dynamic Event Focus 完全由 standard-r1 packet机械派生，所以：

```text
delta wire bytes        = 0
delta action tokens     = 0
delta reachable bits    = 0
standard exact bits     = 207.74974691951957
```

这不是“位置免费”：focus、node geometry、part/cell原本已经被计入 exact grammar。新规格只是冻结这些字段如何共同回答“where”，避免后续再加一个隐式 ROI latent。

### 7.2 明确拒绝的扩容

- 不加连续 `(u,v)`、bbox四元组、heatmap、mask、ROI pooled embedding；
- 不加 `focus_edge_index`、`locator_kind`、tracker confidence；
- 不把 entry/turn/exit point track拼进 packet；
- 不把 human target pixel写进 text绕开视觉容量；
- 不把 v1 22D、F-PCD或continuous trajectory与v2 concat。

若未来增加任何 action-visible field，`207.7497 bit` 立即失效，必须升级 schema/grammar/version、重算 exact reachable alphabet，并重做 equal-bit control；不能只报 tensor width。

## 8. WAM 插入位置（未来任务书，本轮不改代码）

### 8.1 新 v2 文件

- `world_action_model/event_tube_schema.py`：在既定 `NodeCodeV2/EventCodeV2/EdgeCodeV2`、compiler和validator旁加入 control-plane `FOCUS_BINDING_VERSION` 与纯函数 `derive_dynamic_event_focus_v1()`；函数只返回 event→node/edge indices与诊断状态，不产生新 wire code。
- `world_action_model/models/event_tube_planner.py`：在 `PackedEventTubeEncoder` 内实现 hard gather/fusion；不得读取 RGB、planner hidden或ROI feature。

### 8.2 `transformer_wa_casual_mot.py`

- `:599-631,703-870`：v2 opt-in配置/module；v1 disabled行为不变，route互斥。
- `:1253-1403` 邻近：robot planner hard decode后派生 focus handle；invalid reference fail closed。
- `:1405-1495`：13-token packed factor encoder中做 event→node/anchor-edge gather，不复用 `22 -> action_width` linear，不加第14 token。
- `:1531-1590`：返回 canonical packet、focus receipt与aux logits；GT/continuous focus只到loss侧。
- `:127-194`：`num_plan_tokens` mask必须保持 action/state/packet对reference不可达。
- `:1806-1822`：strict v2拒绝 image-text concat造成的 image cross-attention旁路。
- `:2240-2499`：cache只存 packet/focus index/receipt及由同packet产生的KV；任何 node bytes、event或binding变化都失效。

### 8.3 dataset/trainer

- `world_action_model/datasets/wam_lerobot_dataset.py:285-325,711-819`：未来沿 strict sidecar模式读取 sparse event/focus anchor；human anchors不展开成robot action condition。
- `world_action_model/trainer/wa_casual_trainer_mot.py:111-189,555-598,957-1055`：focus role/part/cell的field-level masks与aux loss；普通 predicted action forward不得接收GT focus packet。

### 8.4 v1 不动

`world_action_model/contact_transition_schema.py:18-217,270-413` 的 v1/v1.1常量、validator、capacity与equal-bit语义冻结。Dynamic Event Focus只属于互斥的v2 route。

## 9. 公平 baseline 与实验矩阵

### 9.1 主 locator 比较

所有 finite 条件使用相同 backbone初始化、robot数据、split、13 token slots、embedding width、action head、参数/FLOPs和execute-8/replan时钟：

1. `language_ref_only`：保留同一 instruction budget，location fields canonical masked；检验语言是否足够。
2. `single_point`：每event一个robot-view量化点，代替role/part locator；不允许连续坐标或human点复制。
3. `bbox_part`：robot-view coarse bbox/object role + part，Point-VLA风格；bbox必须量化并报告exact alphabet。
4. `static_focus`：episode只允许一个focus role/cell，检验“有位置”与“随event切换”差异。
5. **`dynamic_event_focus_v1`**：本规格唯一主方案。
6. `event_track_action_visible`：VRB/ATM/Track2Act/3PoinTr/ST-VLA风格连续或3D track；明确标为高容量upper bound。
7. `direct_current_reference_wam`：raw image直达action的性能上界/旁路对照。

`single_point/bbox_part` 必须使用独立schema version并给出exact valid state count。若无法做到与standard同exact alphabet，就只能称 representation baseline，不能称 equal-capacity evidence。

### 9.2 真正的容量匹配 control

唯一严格 equal-capacity control仍是 standard-r1 全packet enumerative rank上的可逆非语义置换：

```text
semantic dynamic focus
vs
event_tube_v2_equal_bit
```

两者共享 `S_action`、13 tokens、planner/aux、参数/FLOPs与cache policy。不能逐field乱序，因为会破坏node reference、presence和grammar依赖。

### 9.3 必做干预

- same-noise swap `focus_node`，其余packet固定；动作变化应在对应event phase最大；
- swap `primary_part/cell`，focus role固定；只应改变局部接近/接触，不应无关地改变整条任务语义；
- freeze focus为episode首节点，与dynamic event focus比较；
- shuffle node global cell但保留role/part，检验action是否真的使用robot-view location；
- 改human像素/track、保持robot packet不变，action必须不变；
- 改robot current node cell、保持semantic role/part不变，action应产生可解释的空间响应；
- oracle boundary/focus与predicted causal focus分开报告。

### 9.4 指标

- task success、event completion、correct-contact part/cell、release/verify success与safety；
- focus role/part/cell accuracy、unknown/overflow、invalid-reference和safe-reobserve rate；
- event-transition处focus switch latency，active event内focus flicker必须为0；
- per-phase intervention effect与无关action维度变化；
- cluster bootstrap 95% CI，以cell/clip/rollout而不是frame或point为独立单位；
- 各route exact action-visible alphabet、经验entropy、token利用率和实际参数/FLOPs。

## 10. annotation / data gates

### F0：机械契约，任何模型实现前

必须证明：

- 每个present event有唯一focus role来源；
- `anchor_edge(e)` 对每个合法packet确定且round-trip后不变；
- human/robot的global image geometry不进入cross-domain pair cost；
- active focus绑定与EpisodeClockState一致；
- focus receipt、packet receipt、capacity receipt SHA一致；
- focus handle derivation不改变82 bytes、13 tokens与`S_action`。

任一失败：停止UI/model实现。

### F1：3 calibration + 12 blind human-pour

沿现有D1门槛，focus-specific至少要求：

- focus role κ `>=0.70`；
- primary part κ `>=0.70`；
- primary part-local cell exact agreement `>=0.70`，并报告Manhattan confusion；
- focus observability κ `>=0.70`；
-关键focus part/cell 双侧 `REVIEWED_UNKNOWN <=40%`；
- focus/reference invalid或edge overflow各 `<=5%`；
- entry/exit focus role不一致但没有合法event transition的比例为0。

3+12仍只验证human-pour可标性，不解锁robot-view grounding或cross-embodiment action claim。

### F2：36 human + 36 target-robot同任务

- human/robot共享比较只用role/part/relation/effect；global pixel字段全mask；
- part-cell只有canonicalization receipt有效时才加入配对ablation；
- robot domain独立双标至少覆盖三task families和遮挡/regrasp；
- legacy human-pour/robot-grab仍只作NULL/hard-negative stress。

F2通过也只说明语义对应与robot-view可标，不说明action sufficiency。

### F3：oracle locator sufficiency

在上位D3 gate之外，dynamic focus必须：

- 相对 `static_focus` 与最佳 `single_point/bbox_part` finite baseline，outcome-sensitive success point gain至少 `+5 pp`，且cluster CI支持正优势；
- 相对 action-visible event track/full trajectory上界下降不超过 `3 pp`；
- correct-part/cell与safety不退化；
- focus intervention在对应phase有局部响应，human-pixel intervention为零响应。

未通过时不进入predicted route。

## 11. 预注册停止条件

以下任一发生，立即收缩或停止对应claim：

1. `single_point`或`bbox_part`与dynamic focus差异小于5 pp且CI不支持正优势：删除“event-dynamic focus必要”claim；保留更简单locator。
2. 只有action-visible 2D/3D tracks、heatmap、ROI或continuous pose有效，且finite route距upper bound超过3 pp：停止finite location sufficiency claim，改称trajectory-conditioned WAM。
3. human global pixel、bbox或image orientation进入robot action后才有收益：停止cross-embodiment claim；该结果只说明paired-view calibration有效。
4. robot current-view node/part定位达不到F1门槛、critical unknown持续`>40%`或invalid/reobserve持续`>5%`：停止strict focus route，不以VLM伪标签补真值。
5. active event内focus role必须频繁变化才能完成任务：当前eight-event ontology/granularity No-Go；不增加逐帧focus index保护结果。
6. equal-bit与semantic route持平且利用率相当：停止“human-readable focus semantics必要”claim。
7. dynamic focus优于static但不优于robot-only/no-human：停止human-to-robot focus transfer贡献，只保留robot-side grounding。
8. part-local 4×4 在sliding/regrasp/插入子集不足：先让event-track/F-PCD challenger赢；不得直接扩成continuous ROI主线。

## 12. 与上位架构的最终一致性

本裁决不推翻final architecture和portfolio，反而把其中“显式在哪里操作”补成可执行的引用规则：

- role node回答“谁”；
- event focus回答“本阶段优先关注谁”；
- robot current node code回答“它现在在本机器人视图哪里”；
- primary part/cell回答“作用哪个局部”；
- secondary relation/effect回答“相对谁、造成什么”；
- target embodiment state/decoder回答“本机器人如何到达”。

因此 unique recommendation 是：**保留 standard-r1 的13-token/208-bit finite route，正式增加 `sm_fet_dynamic_event_focus_v1` 的确定性绑定语义与annotation-only focus anchors；不增加新的action-visible operation-location token。**

如果后续论文写“我们加入一个位置token”，准确表述应是：

> We expose an event-conditioned, role-and-part grounded focus handle by deterministically binding existing finite node, event, and edge codes; no source-domain pixels, continuous ROI features, or extra action-visible token are introduced.

## 13. idea-discussion 登记

本轮按 Zotero-first workflow 复用总idea card：

```text
idea_id       = b4ec9e69-e9aa-4025-8d79-e36df14e0a5e
discussion_id = 092fc9ae-f268-4aa1-b79e-69cb44e6afe9
memory_updated = true
```

个人库命中以 RoboBrain 2.0、RT-1、RoboBrain、FAST、RoboBrain 2.5 笔记为主；自动外部scout偏向failure recovery，与本问题不匹配，因此未用其生成裁决。随后只用原始arXiv/CVF/作者项目页补核 VLAff、CoA-VLA、ST-VLA、BridgeACT、VRB、RoboPoint/Point-VLA、ATM/Track2Act/3PoinTr/JOPAT和KITE。该检索结果支持“where需要显式grounding、目标机器人需自行转换/闭环”的边界，但不替代F0–F3实验。
