# SM-FET-WAM v2-min：可投稿但诚实的 Paper Blueprint

- 日期：2026-08-12
- 文档性质：论文叙事、图表与实验蓝图；不是实现说明、训练报告或结果声明
- 独立 worktree / branch：`gwp-agent-sm-fet-paper-story` / `codex/sm-fet-paper-blueprint-20260812`
- canonical 实际基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 目标方法：`SM-FET-WAM v2-min / event_tube_v2_standard / sm_fet_hard_grammar_v2_standard_r1`
- 当前事实：v2 schema、planner、compiler 与训练均未实现；D0–D4 均未产生通过结果
- idea-discussion：复用 `idea_id=b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`；本轮唯一一次登记为 `discussion_id=7f20f4b9-bd73-4d6f-a414-ba94d05d598f`，`memory_updated=true`

> 本文把“可投稿”理解为：先冻结一条审稿人能够清楚否证的中心命题、相应的公平实验和失败后的降级路径。它不把尚未执行的设计写成已完成方法，也不预写正结果。

## 0. Executive decision

这篇论文不应讲成：

> 我们把 Semi-Markov、contact、partial OT、三姿态、effect grouping、missingness 和 WAM 拼成了一个新模型。

这会得到合理的 `weak reject`：每个组件都有直接或强相邻前例，组合本身也没有证明为何必要。

唯一可守的 paper story 是一个**经严格路由和容量对照检验的实证问题**：

> 对 action-centered WAM，训练期人类视频中的哪些信息值得迁移？一个由角色、粗位置、接触、对象关系变化、持续时间和效果组成的严格有限、可逐字段干预的事件接口，能否在部署时完全移除人类视频、密集人手轨迹和跨域配对身份，同时保留对目标机器人控制真正有用的信息？

如果实验成立，论文贡献不是“首次提出事件、接触或 partial OT”，而是：

> **在相同机器人数据、backbone、优化预算和动作可见容量下，用因果路由审计证明或证伪：有限、人工可读的 role–contact–relation–effect event contract 是否比软 latent、单帧姿态、对象地址、旧 contact packet 和无语义等容量码更适合作为训练期 human-to-robot WAM transfer 的接口。**

当前推荐的投稿形态是**方法假设 + 受控实证论文**，不是算法组件论文；CAPOT、entry/turn/exit pose、透明场景 auxiliary 和 missingness contract都是支撑中心检验的训练/数据工具，不分别申报 novelty。

## 1. 候选标题

按推荐顺序：

1. **What Should a Robot Copy from Human Tool Use? Auditing Finite Event Interfaces for World-Action Models**
2. **Copy the Effect, Not the Motion? Finite Event Bottlenecks for Human-to-Robot World-Action Transfer**
3. **Events, Not Trajectories? A Causal Audit of Human Video Supervision for World-Action Models**
4. **An Auditable Role–Contact–Effect Interface for Training World-Action Models from Human Video**
5. **SM-FET-WAM: Testing Finite Event Sufficiency for Human-to-Robot Tool Use**

标题 1 最稳妥：它把论文定位成“回答应该复制什么”的实证研究，不先宣称 finite event 已经充分。标题 2 只有在 dense trajectory 上界差距不超过预注册门槛时使用；标题 5 只有在方法名已稳定且 D3/D4 结果成立时使用。

避免使用 `Human-Inspired`、`Cognitively Grounded`、`Brain-Like`、`First Contact-Aware Alignment`、`First Cross-Embodiment Event WAM` 等标题措辞。

## 2. 三个一句话

### Problem

人类第一视角视频包含视角、人体形态、手部轨迹和任务外片段，而目标机器人需要从自己的当前观测与本体状态生成动作；尚不清楚训练时应迁移哪一小组功能信息、又如何证明动作没有经软视觉旁路消费更多信息。

### Insight

跨域共享的候选不应是 human 像素、手指关节或完整轨迹，而应是可在机器人当前视图重新定位的功能角色、作用部位、接触/对象关系转折、持续时间和效果；连续手姿与人机对应只作为训练期证据，部署时删除。

### Contribution

我们提出并审计一个固定语法、精确可枚举、可逐字段干预的有限事件接口，并用等容量、等预算、无旁路对照检验它是否在固定机器人数据下保留了人类视频对 held-out 工具/效果有用的控制信息；贡献成立与否由预注册的 oracle、predicted、pairing 和 transfer 结果共同决定。

## 3. 论文真正要回答的科学问题

### 3.1 主问题

设机器人动作分支始终获得 `proprio/state + text + diffusion state`。新增的人类视觉知识只能通过一个受限接口进入训练。主问题是：

> 在相同动作可见容量和训练预算下，**语义化的有限事件接口**是否比无结构等容量码、软跨具身 latent、单帧/多帧姿态、对象地址和连续轨迹更能改善目标机器人对 held-out 工具与效果的控制？

这里的“有限”只表示 frozen grammar 的可达 alphabet 有精确上界；“接口”表示动作头只能读 hard packet；“充分”必须是相对指定任务族、指标和连续上界的经验结论，不是信息论最小充分统计量。

### 3.2 三个次问题

1. **时间粒度：** `entry/turn/exit` 或事件邻域是否比同帧数 single/uniform/motion-peak 更好，并接近完整 event trajectory？
2. **跨域监督：** fixed robot data 下，人类 event supervision 是否相对 robot-only/no-pair 和最强 prior aligner产生稳定增益？
3. **可信度：** missingness、遮挡、透明容器和 unresolved actor 被诚实 mask/UNKNOWN/censor/fail-closed 后，有限接口还能否保持足够 coverage 和安全行为？

### 3.3 贡献层级，不能混写

| 层级 | 论文可检验内容 | 成立所需证据 | 失败后的处理 |
|---|---|---|---|
| C0 契约 | 82-code packet唯一可编译、无旁路、容量 receipt正确 | D0 100% round-trip/fail-closed、mask/cache/gradient audit | 论文停止，不训练 |
| C1 表示 | finite semantic packet对控制有价值 | D3 胜22D/等容量码且接近continuous upper bound | 删除finite sufficiency，保留benchmark |
| C2 预测 | causal planner能恢复oracle价值 | D4恢复至少70% oracle gain，旁路审计通过 | 降为oracle/representation study |
| C3 人类监督 | human data在固定robot budget下有增益 | held-out tool/effect上胜robot-only和unpaired/prior aligner | 删除human-transfer claim |
| C4 配对 | reviewed event-level partial pairing有独立价值 | CAPOT胜最强对齐baseline且下游增益一致 | CAPOT降为内部数据工具 |

论文摘要只写实际通过的最高层级。例如 C0–C2 通过但 C3 失败，应写“finite event interface for robot WAM”，不能继续写“human-to-robot transfer”。

## 4. 方法叙事：一个接口，两个域，三条防线

### 4.1 Action-visible standard-r1

每次 action chunk 只读取 13 个 hard tokens，由 82 个 canonical `uint8` codes组成：

```text
node_code  [4,4]
  observability/presence, center_cell, scale_band, orientation_2d

event_code [3,6]
  kind/presence, focus_node, onset, duration, object_effect, outcome

edge_code  [3,2,8]
  actor/presence, primary_node, primary_part, primary_surface_cell,
  secondary_node, relation_before, relation_after, contact_mode
```

四个固定角色是：

```text
manipulated_or_source
tool
receptacle_or_destination
fixture_or_support
```

八类跨任务语义事件是：

```text
approach, precontact, acquire, hold,
manipulate, regrasp, release, verify_outcome
```

另有 `UNKNOWN` sentinel 和 `ABSENT` code。`UNKNOWN` 不是第九类动作；`ABSENT` 不是事件；`MISSING/NOT_REVIEWED` 不进入 wire。

在 `sm_fet_hard_grammar_v2_standard_r1` 的完整约束下：

```text
reachable packet states
= 345864043392188289642033658185203030253953267411599196753571073

log2(states)
= 207.74974691951957 bit

minimum enumerative index width
= 208 bit
```

论文必须称其为**frozen grammar 下 conditional current-visual route 的理论可达容量**。它不是 82-byte 存储宽度、经验熵、互信息、最小充分性或语义压缩证明。

### 4.2 模块 A：任务本体与 role binding

每个任务在数据进入标注或训练前冻结：四角色到实体的一对一映射、允许的 r1 transition 子图、primary part 子词表、relation/effect/outcome 判据和传感来源。任务本体只能收窄全局 grammar，不能在 test 后增添 event kind、skip 或新 role。

它解决“操作谁”，不负责目标机器人执行。human literal left/right、finger identity和 wrist path不跨域共享。

### 4.3 模块 B：域内 event encoder / planner

human 与 robot 各自从本域视频提取：

- macro event kind、interval/censor 和 duration；
- focus role 与四个 node 的当前粗 2D 状态；
- 每事件最多两条 contact/object-relation edge；
- object effect 与 verify-only outcome。

human/robot 可以共享部分语义头或做 matched-event distillation，但不能要求像素、手关节和全程速度同步。planner最多维护 `Q_episode<=8` events；动作编译只取 `active+next_1+next_2`。

### 4.4 模块 C：Dynamic Event Focus，无新增 token

“在哪里操作”不是额外连续位置 token，而是现有 hard codes 的确定性绑定：

```text
event.focus_node
-> 当前机器人视图中的 node observability/cell/scale/orientation
-> primary part + part-local 4x4 cell
-> secondary node + relation/effect
```

anchor edge由 canonical edge order机械选取；encoder只 gather已有 node/edge tokens，不读取 ROI、heatmap或 planner hidden。因此增量 wire、token和 reachable bits均为0。human坐标不复制到robot图像；机器人必须从自己的当前观测重新定位功能焦点。

### 4.5 模块 D：Semi-Markov clock 与 hard compiler

episode clock保存 action-invisible 的 event ID、绝对帧、elapsed time和 receipt。每执行8步后重新观测：

1. planner基于 `x_{<=t}`、proprio、text和旧hard clock预测Q8 tube；
2. compiler取 active+未来96帧内最近两个event；
3. 收集focus/edge引用node union，量化为82 codes；
4. 检查prefix、reference、canonical order、transition、UNKNOWN terminal和overflow；
5. 只有`OK/NULL`产生action packet；invalid/unresolved时hold/reobserve，不soft fallback。

WAM预测48步action，只执行前8步，然后使旧packet/cache失效并replan。

### 4.6 模块 E：Packed event encoder 与 strict WAM action expert

13个token由factor embeddings产生，不能复用一个任意 `82 floats -> hidden` 软投影来绕过字段语义。action expert只读：

```text
robot proprio/state
allowed instruction text
diffusion timestep/noisy action
13 hard event-tube tokens
```

它不能读raw RGB/reference token、future visual、human video、continuous pose/box/depth/flow、planner logits/hidden、pair ID、CAPOT coupling或tracker KV。text仍可直达action，所以论文不能称整个策略只有208 bit，只能限定当前视觉桥。

### 4.7 模块 F：训练期 auxiliary 与 offline pairing

两个支撑模块必须画成虚线、并在部署图中删除：

1. `entry -> optional turn -> exit` 的七个2D functional landmarks及局部窗口，只监督event encoder/planner，并用于single/S3/local/full压缩比较；不进入packet。
2. CAPOT只对同任务、同功能role/effect的reviewed macro events生成stop-gradient sparse pair table，允许NULL、多对多和速度差；它是配对协议，不声称为新OT算法。

首个v2-min oracle/representation实验可以关闭CAPOT，先隔离hard interface本身；只有D2配对benchmark通过后，D4 human-transfer实验才开启CAPOT。

### 4.8 防线 1：missingness不是负例

逐字段、逐侧区分：

```text
KNOWN             -> 普通target
REVIEWED_UNKNOWN  -> 有该字段UNKNOWN code时参与CE
NOT_REVIEWED      -> loss mask=0
ABSENT            -> 只在EXHAUSTIVE coverage下为负例
LEFT/RIGHT_CENSOR -> interval/survival target
```

actor side没有UNKNOWN code。确认hand edge存在但left/right仍不可判时，保留confirmed presence和其它已知planner supervision，actor mask为0，整个oracle/action packet inadmissible。部署hold/reobserve，或在目标机器人allocator给出唯一、有receipt的side后重编译为LEFT/RIGHT。当前不引入`HAND_UNRESOLVED` r2，因此仍保持207.7497-bit r1口径。

### 4.9 防线 2：透明场景不补造不可见3D

透明容器profile只允许直接可复核的coarse 2D role、part、relation、event和sensor-confirmed effect/outcome进入packet。visible-only rim/boundary/status可以做training-only auxiliary；completed depth、mesh、NeRF、liquid volume/flux、stream/contact-interface geometry和CAM不得成为GT或action旁路。

无秤、液位或native truth时，不能由倾斜手腕或task name推断FLOW/TRANSFER/success。2D overlap不能写成3D touching/inside/aligned。若只有reconstruction或action-visible continuous spatial route有效，non-reconstruction finite claim失败。

### 4.10 防线 3：跨域共享功能，不共享形态

human与robot可对齐event kind、focus role、primary part、relation/effect/outcome；默认不对齐literal hand side、global image cell、bbox、orientation、wrist path和joint pose。目标执行层必须使用robot proprio、kinematics、tactile/current和硬件可行性为RH56DFTP/OmniPicker分配动作。

## 5. Training 与 deployment 必须明确分开

### 5.1 Training-only dataflow

```text
Human video
  -> human event encoder/planner
  -> event/node/edge/effect prediction
  -> entry/turn/exit or local-window auxiliary only
                         \
                          -> offline reviewed pairing/CAPOT
                         /      -> sparse stop-gradient event_pair_v1
Robot video + native state/contact/outcome
  -> robot event encoder/planner
  -> matched-event distillation/alignment

Robot current causal inputs
  -> predicted or explicit oracle hard packet
  -> strict WAM action expert
  -> robot action loss + action-conditioned future visual loss
```

训练硬规则：

- human clip没有robot action label，不做pseudo-retarget action loss；
- 先按source/group切split，再建windows、candidate graph和pair；
- coupling、pair ID、pose、continuous geometry和confidence均stop-gradient/action-invisible；
- oracle route是独立checkpoint/route，不能在同一predicted forward按字段混GT与预测；
- finite routes互斥，不能把22D、F-PCD、SM-FET和continuous trajectory concat后归因给SM-FET。

### 5.2 Deployment dataflow

```text
robot current image + causal robot history + proprio + instruction
-> robot event planner
-> hard decode / validate / active+next2 compiler
-> 13 hard event tokens
-> WAM predicts 48 action steps
-> execute 8
-> reobserve / update clock / invalidate cache / replan
```

部署时明确不存在：human video、human pose、human pixel、pair table、OT、GT event、future visual、continuous trajectory token、soft planner distribution、annotation confidence或missingness reason。

### 5.3 摘要中应有的一句隔离声明

> Human videos and cross-domain correspondences supervise event prediction only during training; at deployment, the action model consumes robot observations, proprioception, language, and a validated finite event packet, with no human reference or dense trajectory input.

## 6. 三张核心 Figure

### Figure 1：`One finite interface; training-only human evidence; human-free deployment`

**目的：** 一眼说明论文不是human trajectory retargeting，也没有第二条视觉旁路。

建议四个panel：

- **(a) Human/robot domain-specific evidence：** human视频上画entry/turn/exit、event interval和role/contact/effect；robot侧画当前图像+proprio/native outcome。
- **(b) Offline supervision：** sparse reviewed event pairing，NULL/many-to-many边；CAPOT框用灰色虚线并标`training only / not a new OT claim`。
- **(c) Finite compiler：** 四role nodes、active/next2、每event两edge，输出`82 uint8 / 13 tokens / 207.7497 bits under r1`；画出UNKNOWN、MISSING和invalid的不同出口。
- **(d) Deployment：** robot-only闭环，predict48/execute8/reobserve；在人类、pose、OT、ROI、planner hidden到action之间画红色阻断线。

图注必须写：认知文献只启发event anchor选择；图中模块不是人类认知机制模型。

### Figure 2：`Matched representation frontier and causal route audit`

**目的：** 让审稿人看到论文在回答“语义事件是否必要”，不是“加了token是否有用”。

建议四个panel：

- **(a) Performance vs action-visible capacity/bandwidth：** direct RGB、22D、F-PCD、SM-FET semantic、SM-FET equal-bit、continuous local/full；连续路线用实际tokens/bytes/FLOPs，不伪称equal-bit。
- **(b) Oracle-to-predicted waterfall：** direct → oracle22D → oracleSM-FET → predictedSM-FET，标注oracle gain与恢复比例。
- **(c) Field interventions：** same-noise交换focus、part/cell、relation/effect、onset/duration，显示只在对应phase产生局部动作/效果变化。
- **(d) Leakage controls：** fixed packet/state/text/noise下换reference、human clip、pair ID或aux logits，pre-safety action变化应约为0。

只有semantic packet显著胜自身global equal-bit、field shuffle和强representation baseline，才能谈semantic necessity。

### Figure 3：`Where human supervision helps—and where the finite interface fails`

**目的：** 把主正结果、数据效率和诚实失败放在同一张图，防止只报平均success。

建议四个panel：

- **(a) Human-data scaling：** 固定robot demos，横轴human clips或reviewed pairs，纵轴held-out effect/tool success；画robot-only、unpaired、strongest prior aligner、CAPOT、oracle pair。
- **(b) Temporal evidence frontier：** single-repeat3、entry/exit、S3-semantic、uniform3、motion-peak3、local window、full-to-same-packet、action-visible dense upper bound。
- **(c) Coverage–risk：** missingness/transparent strata下compile coverage、abstain、wrong-contact与success；必须包含unresolved actor和unlocalizable rim。
- **(d) Per-family failures：** pick-place、twist/regrasp、pour/transparent分别报告；不让容易任务掩盖精细接触与液体失败。

如果人类数据只改善seen object、增加等量robot data后优势消失，或只在pick-place有效，图3应直接显示并收缩claim。

## 7. 四张核心 Table

### Table 1：Data, supervision and provenance

每个数据源一行，列为：domain、hardware、task/effect families、clips/rollouts、operator/session/object/tool groups、native/sensor outcome、contact truth、transparent strata、annotation coverage、允许用途、禁止用途。

必须把现有`66 human pour`和`88 legacy dual-Inspire grab`分开，并明确联合用途只有ontology/hard-negative stress，positive pair数为0。36+36标成feasibility pilot，不写production OOD。

### Table 2：Main representation and control results

行：direct robot-only、22D、F-PCD、ETG current-delta、SM-FET active-only、+next1、+next2、SM-FET global equal-bit、continuous local/full。列：exact capacity或实际带宽、parameters/FLOPs、task success、event completion、correct first contact/release、spill/drop/collision、latency、95% cluster CI。

主比较必须是同robot data、backbone、checkpoint init、optimizer steps、seeds和safety filter。22D与v2原生容量不同，不能把它们单独称equal-capacity；各自semantic vs自身equal-bit才是干净语义对照。

### Table 3：Human supervision, temporal evidence and pairing

分三个block：

- robot-only / unpaired human / forced 1:1 / strongest prior aligner / CAPOT / oracle pair；
- no pose / single-repeat3 / entry-exit / S3 / uniform3 / motion-peak3 / local / full-to-same-packet；
- alignment edge F1、NULL AUPRC/F1、hard-negative FAR、effect consistency、downstream success与data efficiency。

不要把完整二维笛卡尔积全跑后挑最好结果。先在validation预注册每个轴的finalist，再做`pairing × temporal evidence × action representation`的小型核心factorial。

### Table 4：Audit, safety, limitations and failure strata

列：route ID、packet validity、compile coverage、UNKNOWN/MISSING/ABSENT confusion、overflow、unresolved actor、cache/reference swap、gradient leakage、field-intervention locality、safe reobserve、wrong-side/unintended contact、transparent/depth dependency、selection shift。

这张表是论文可信度的核心，不应放到supplement。任何旁路审计失败时，对应causal/finite claim直接删除。

## 8. 分阶段实验矩阵

### 8.1 为什么不能一次跑“大模型大全”

SM-FET同时涉及标签、时间表示、跨域配对、action route和hardware control。把所有模块一次打开会同时改变监督量、参数量、visible bits和训练样本，无法归因。实验必须按D0–D4逐层解锁。

### 8.2 总矩阵

| Stage | 数据与输入 | 核心比较 | Primary endpoint | 解锁条件 | 失败含义 |
|---|---|---|---|---|---|
| D0 契约 | synthetic legal/illegal fixtures + 各域只读sample | semantic packet vs global equal-bit codec；round-trip/validator | 100% unique round-trip、fail-closed、capacity/receipt一致 | 全部通过 | 不能实现或训练v2 |
| D1 可标性 | 3 calibration + 12 blind human-pour A/B+裁决 | core event/node/edge；pose独立probe | event F1、tIoU、boundary、κ、unknown/overflow、人时 | core字段达到门槛；其它NOT_EVALUABLE | 见第13节降级 |
| D1-M/T | D1 gold + controlled hiding；fresh transparent audit | reason-coded mask vs错误填充；visible-only vs pseudo geometry | false absence、calibration、rim/boundary可标性 | 不造GT且风险可校准 | missing/transparent仅作限制 |
| D2-R | 36 human + 36 target-robot / 12 cells | single/S3/uniform/motion/local/full | event/effect macro-F1、retrieval、成本 | S3胜single且接近full | 删除multi-pose claim |
| D2-P | 180 reviewed pair decisions + mismatch stress | forced/DTW/Drop-DTW/GMS/EgoBridge/REMAP-UOT/CAPOT/oracle | edge F1、NULL、FAR、effect consistency | 胜最强prior且CI支持 | CAPOT降data tool |
| D3 oracle | powered target-robot native truth | direct/22D/F-PCD/ETG/SM-FET/equal-bit/continuous | rollout success、event completion、安全 | v2胜22D且接近full | finite action route失败 |
| D4 predicted/transfer | expanded group-safe human+robot | robot-only/unpaired/prior/CAPOT/oracle pair；oracle→predicted | held-out tool/effect success与human-data scaling | predicted恢复oracle gain且human增益稳定 | 收缩predicted或transfer claim |

### 8.3 D1 门槛

核心门槛沿已冻结协议：

- event existence F1 `>=0.80`；
- matched median tIoU `>=0.70`；
- boundary P50 `<=2 processed frames`或`<=100 ms`；
- kind/focus/role/part/node/relation/effect/outcome κ通常`>=0.70`；
- contact mode κ `>=0.60`；
- key effect/outcome双侧unknown `<=40%`；
- Q>8、node>4、edge/event>2和invalid reference各`<=5%`；
- false-ABSENT与incomplete oracle packet误admission必须为0；
- pose NME/visibility只作为probe，不可补救core失败。

单类或支持不足必须写`NOT_EVALUABLE`，不能以100% agreement通过。

### 8.4 D2 数据与split

feasibility pilot：

```text
3 task families
× 2 physical object conditions
× 2 session/day groups
= 12 cells

每cell 3 human + 3 target-robot
= 36 human + 36 robot
```

任务族：`pick_place`、`stabilize_twist_cap`、`pour_transfer`。每个cell包含标准成功、prep/速度变化、regrasp/partial/failure；失败不能重录删除。

split必须先于annotation-derived windows和pair：每family `2 train / 1 val / 1 test cells`，candidate graph connected component不跨split。这个设计只能称held-out object-session cell；要声称novel object/session OOD，必须扩更多独立group。

### 8.5 D2-R 时间证据arms

| Arm | 输入 | 公平性解释 |
|---|---|---|
| no-pose | 只有core event监督 | pose auxiliary是否必要 |
| S1-contact-repeat3 | 单个contact anchor重复三次 | 与S3同位置数的single主对照 |
| S1-next-repeat3 | 最有信息的next pose重复 | 避免故意选差单帧 |
| E2 | entry + exit | 是否需要turn |
| S3-semantic | entry + optional turn + exit | 主候选 |
| U3 | event内均匀三帧 | 同帧数非语义对照 |
| M3 | 三个motion peaks | 强动态challenger |
| W-local | anchor邻域固定总时长窗口 | 局部运动challenger |
| F-full | interaction event完整轨迹 | 信息上界 |
| F-episode | prep到结束完整episode | 检验全程噪声 |

先用统一小probe回答信息问题；D3/D4中所有arms最终编译到相同packet，回答training inductive bias问题。只有action-visible dense route另作高带宽upper bound。

### 8.6 D2-P pairing arms

主线必须相对强baseline，而不是只胜forced 1:1：

```text
normalized-time forced 1:1
DTW / soft-DTW
Drop-DTW
GMS-SDTW / subsequence DTW
EgoBridge-style balanced OT
partial OT
UOT without monotonicity
REMAP-style partial alignment
CAPOT
oracle manual edges
no-pair / robot-only
```

所有方法共享candidate universe、known-field masks、normalization、调参预算和split。CAPOT主cost不能用human/robot raw pixels、literal side、global wrist path或dense pose shortcut。

最低gate：NULL AUPRC和unmatched F1均`>=0.80`、effect consistency`>=0.80`、crossing`<1%`、无all-null collapse；CAPOT positive-edge F1相对最强prior至少`+5 pp`且cell/clip cluster bootstrap 95% CI支持正优势。legacy mismatch false accept`<=5%`，normalized transported mass`<=0.10`。

### 8.7 D3/D4 representation arms

主矩阵：

```text
R0 direct current-reference WAM
R1 contact_segment_v1_1 22D-only
R2 F-PCD-only
R3 ETG current-delta-only
R4 SM-FET active-only
R5 SM-FET active+next1
R6 SM-FET active+next2
R7 SM-FET global equal-bit
R8 local/full continuous trajectory upper bound
```

locator/focus在validation内另比较language-only、single-point、bbox-part、static-focus、dynamic-event-focus和event-track upper bound。只有dynamic focus相对最佳finite locator有稳定优势，才强调event-conditioned focus；否则保留更简单locator并删除该claim。

### 8.8 Missingness 与透明场景arms

Missingness：complete-case、mask-only-r1、错误的missing→ABSENT、last-observation-hold、model-impute-pseudo、reason-coded-mask-r1、sensor/mocap oracle。主要看false absence、risk–coverage和策略安全，不看“补了多少标签”。

Transparent：RGB-only SM-FET、RGB+raw depth、completed-depth action baseline、visible-only rim/boundary train auxiliary、action-visible 2D affordance ceiling、continuous trajectory ceiling。若只有completion或spatial ceiling提升，诚实结论是finite non-reconstruction不足。

### 8.9 统计纪律

- rollout而不是frame/action chunk是控制实验的独立单位；
- annotation以episode聚类，pairing至少以cell聚类并做human/robot交叉cluster bootstrap；
- 3 seeds不能替代独立clip/rollout数；
- primary endpoint、阈值、candidate universe、NULL cost、solver budget、early stopping和排除规则在validation前冻结；
- 报per-family与macro，透明、regrasp、bimanual/alternating和failure strata单列；
- D3正式样本量由pilot rollout variance做power analysis，36 robot feasibility clips不能直接充当policy论文样本量。

## 9. 决定性结果必须长什么样

### 9.1 结果形态 A：语义事件，而不只是多一个通道

论文成立需要同时看到：

1. oracle SM-FET相对oracle 22D在outcome-sensitive success至少`+8 pp`或event completion至少`+10 pp`；
2. 相对full continuous trajectory success差距`<=3 pp`且安全不退化；
3. SM-FET semantic显著胜自身global equal-bit、字段置乱和matched locator/control；
4. focus/contact/relation/effect swap产生phase-local、语义一致的动作变化；
5. fixed packet下换reference/human/pair/aux不改变action。

如果只有第1项成立，收益可能来自容量、时域或额外监督，不能写semantic event necessity。

### 9.2 结果形态 B：稀疏事件证据，而不是预设三姿态正确

需要：S3相对最佳single至少`+5 pp`且CI支持正优势；相对F-full的effect/contact macro-F1差距`<=5 pp`，在policy的full-to-same-packet对照中差距`<=3 pp`；S3不低于U3/M3最优者。

若S3只和uniform持平，claim改为“少量帧足够”；若dense显著更好，删除“三姿态近似充分”并让trajectory-conditioned challenger获胜。

### 9.3 结果形态 C：人类信息，而不只是更多数据

固定robot demos、模型、steps和seed后：CAPOT/事件监督相对robot-only、unpaired human和最强prior aligner在held-out tool/effect上至少有预注册`+5 pp`关键指标优势，CI支持正优势；增加等量robot data后仍报告human marginal value；predicted planner恢复至少`70%` oracle正增益。

如果收益只在seen object、容易的pick-place或某个operator出现，论文不能说cross-embodiment/general transfer。

### 9.4 决定性负结果同样有价值

以下任一形态应直接改变paper结论：

- equal-bit≈semantic：有限通道可能有用，但人类可读factorization不是必要；
- 22D>=SM-FET：event annotation可用于benchmark，v2 action route不成立；
- continuous≫finite：应改为trajectory-conditioned WAM；
- no-pair≈CAPOT：删除pairing contribution；
- robot-only≈+human：删除human-transfer headline；
- oracle好、predicted差：定位为representation ceiling/planner challenge；
- compile coverage低且abstain集中在关键phase：保守contract可能造成不可接受selection bias。

## 10. Related work 的诚实定位

### 10.1 Human video 与跨具身技能

必须正面对比 XSkill、UniSkill、EgoBridge、ImMimic、MOTIF、MimicFunc，以及dense human trajectory/affordance路线。不能说“首次从人类视频学习机器人”“首次不需严格配对”或“首次迁移功能而非动作”。

区分点不是任务首创，而是待验证的四条件交集：

1. action-visible接口有固定语法与精确可达容量；
2. human RGB、pose、pair identity和软跨域latent不直达action；
3. matching仅离线、事件级、允许NULL/多对多；
4. 部署完全无human reference。

### 10.2 Contact、affordance 与“在哪里操作”

CoA-VLA、VLAff、BridgeACT、VRB、Contact-Anchored Policies、DexMani、KITE、RoboPoint/Point-VLA等已覆盖object/part/contact/trajectory/affordance conditioning。不能说首次显式表达对象、抓取位置或接触变化。

本论文应写：这些工作多使用连续3D/4D、heatmap、trajectory或learned intent latent；我们检验更窄的问题——一个可枚举、由目标机器人当前视图重定位的role/part/relation/effect handle是否足够，并以continuous route作为上界而非贬低它。

### 10.3 WAM、event memory 与离散表示

GigaWorld Policy/GWP-0.5是底座，不是本论文贡献；WorldScape Policy 2.0已覆盖event memory/progress，OA-WAM已覆盖object-addressable WAM，VQ-BeT及大量skill tokenizer已覆盖离散code。不能把“给WAM加event/object/token”写成novelty。

区分点只可能是：**strict hard route + exact grammar accounting + equal-capacity semantic control + field intervention + human-free deployment**共同形成的实证审计。

### 10.4 Procedure learning 与partial alignment

TCC、Drop-DTW、GMS-SDTW、EgoBridge-style OT、REMAP/REALIGN、regularized GW-OT、StepFormer、STEPs和invariant HSMM已经覆盖关键步骤、partial/unbalanced/subsequence alignment与Semi-Markov时序。CAPOT应称为`reviewed semantic event pairing protocol`，除非未来确有不同的目标函数、求解性质和标准benchmark优势，否则不称新OT方法。

REMAP与REALIGN是同一arXiv条目的版本演化，不应列成两个独立baseline。

### 10.5 Missingness 与透明感知

Mohan–Pearl missingness graph、PU/MNAR方法、100DOH、HandOccNet、HOT3D、ARCTIC已有missing/occluded hand建模与truth资源；ClearGrasp、TransCG、ClearPose、A4T、PourIt!已有透明depth/pose/liquid感知路线。本文不声称首次建模缺手或首次不用重建，而是把false absence、reviewed unknown、censor和pseudo-GT隔离作为finite action route的审计条件。

### 10.6 Cognitive evidence只放在动机，不放在贡献

Pomp等支持touching/untouching是候选边界，但多数contact motif并未成为主观边界；Bunday、Roberts和教学动画研究说明动态信息不能普遍被单帧替代；goal/outcome和前瞻注视支持功能层级，但不证明观察者只保留关键帧。regrasp作为独立人类观察阶段的证据最弱。

推荐措辞：

> Human event-segmentation and observational-learning studies motivate contact/release/effect-centered hypotheses while also warning that local motion can be indispensable. We therefore treat SM-FET as a falsifiable engineering bias, not a model of human cognition.

## 11. Claim ledger

### 11.1 当前设计文档可以说

- 我们定义了一个待验证的strict finite event interface和相应的公平审计方案。
- 在冻结r1 grammar下，合法action packet的理论可达容量是`207.74974691951957 bit`。
- 设计上human pose、CAPOT、continuous geometry和missingness metadata不进入deployment action route。
- 现有66 human pour和88 legacy robot grab没有同task/effect/hardware positive overlap。
- D0–D4尚未通过，因此当前没有policy、transfer、hardware或透明任务收益可报告。

### 11.2 只有结果通过后可以说

- “在本文测试的任务族、robot budget和baselines下，SM-FET比自身等容量无语义control及强representation baselines更有效。”
- “动作对focus/contact/relation/effect intervention呈phase-local响应，且fixed-packet swap审计未发现所测旁路。”
- “训练期human event supervision在固定robot data下改善held-out tool/effect表现；部署不需要human reference。”
- “在测试的目标硬件与role-swap条件下观察到domain transfer。”只有至少两个末端/具身条件经过独立评估时才可谨慎写cross-embodiment。
- “visible-only auxiliary在不增加action-visible bits的前提下改善透明stratum。”只有真实group-safe结果支持时可写。

### 11.3 永远不能靠当前设计直接说

1. 首次从第一视角或人类视频学习机器人操作。
2. 首次contact/affordance/event/object-address/skill token/partial OT。
3. 首次无需配对、无需robot data或完全端到端。
4. CAPOT是新OT算法。
5. 82 bytes或207.75 bits证明最小、充分、可解释或信息瓶颈。
6. 人类被证明只看固定八阶段或entry/turn/exit三姿态。
7. mirror system、人类眼动或脑活动证明SM-FET机制正确。
8. 现有66+88证明paired transfer。
9. 单硬件结果证明广义cross-embodiment泛化。
10. 解决透明物体/液体操作、双臂灵巧操作或missing hand恢复。
11. 模型/VLM/MANO/depth completion补出了真实GT。
12. `hand_active`、closed gripper或2D overlap等于physical contact。

## 12. 推荐摘要骨架（不预填结果）

1. **问题句：** Human videos offer abundant demonstrations, but transferring dense human motion into robot policies entangles task-relevant interaction structure with viewpoint and embodiment-specific trajectories.
2. **缺口句：** Existing approaches already cover skill latents, contact/affordance conditioning, object addresses, dense trajectories, and partial alignment; what remains unclear is whether a strictly limited and auditable interface can retain the useful information.
3. **方法句：** We formulate SM-FET-WAM, where the action expert receives only a validated finite role–contact–relation–effect packet, while human pose and event correspondences remain training-only and are absent at deployment.
4. **实验句：** We compare the interface against direct vision, legacy contact, object/pose/trajectory, strong alignment, and reversible equal-capacity controls under matched robot data and route audits.
5. **结果句：** 只填实际结果，例如`[SM-FET improves X by Δ with CI, remains within δ of dense trajectory, and passes/fails specified intervention audits]`。
6. **结论句：** 用实际最高claim；若仅benchmark成立，就写`Our results expose when finite event supervision is observable and when it is insufficient.`

不能在实验完成前把第5句写成未来时的预期结论放进摘要草稿。

## 13. D1 / D2 失败时如何降级为 benchmark paper

### 13.1 D1 部分失败：稳定字段仍可形成窄 benchmark

如果event existence/boundary/focus和部分role字段通过，但contact mode、relation/effect、transparent或bimanual字段不稳定/NOT_EVALUABLE：

- 删除SM-FET-WAM方法与控制充分性claim；
- 只保留可重复字段，不降低阈值、不用VLM/pseudo label填空；
- 将论文改成 **Censored Event Observability Benchmark for Egocentric Tool Use**；
- 贡献是reason-coded missingness、interval/censor、PASS/FAILED/NOT_EVALUABLE报告、annotation cost和强proposal baselines；
- 扩展到至少约300 adjudicated events、50 confirmed bimanual events、30 production groups及sensor/mocap-confirmed subset后再考虑正式benchmark投稿；
- 主要结果是coverage–agreement–risk frontier，而不是robot success。

候选标题：

> **What Is Actually Observable in Egocentric Tool Use? A Censored Event Annotation Benchmark**

### 13.2 D1 核心失败：不能强行做benchmark paper

如果经过一次预注册协议修订后，event existence F1、boundary、role binding等核心仍失败，12段pilot本身不足以成为完整benchmark论文。诚实路径是：

1. 宣告固定r1 ontology/action route No-Go；
2. 发布内部negative audit或workshop note，说明哪些语义在单目Ego中不可稳定复核；
3. 若继续做benchmark，必须重新采集多视角/传感truth和更大group-safe数据，再设计更窄标签；
4. 不得用自动proposal“修好”agreement，也不得把失败字段留在action schema。

换言之，D1核心失败时不是“换一个paper title就能投稿”，而是需要新的数据资产；这条边界应预先写进项目决策。

### 13.3 D2-R失败：降为观测与时间表示benchmark

如果S3不胜single/uniform/motion-peak或显著落后full：

- 删除“三姿态近似充分”和人类式event sampling叙事；
- benchmark保留single、semantic-K、uniform-K、motion-K、local-window、full的统一协议；
- 主结论可以是“语义关键帧未优于同预算动态采样”或“连续局部运动在regrasp/pour中不可删”；
- 若数据规模与group覆盖足够，这个诚实负结果比保护三姿态更有价值。

候选标题：

> **Snapshots, Event Anchors, or Trajectories? A Controlled Benchmark for Tool-Use Supervision**

### 13.4 D2-P失败：降为partial event alignment benchmark

如果reviewed pair gold可靠，但CAPOT不胜Drop-DTW/GMS-SDTW/EgoBridge/REMAP最强者：

- CAPOT明确降为内部数据编译器，删除算法/配对superiority claim；
- 发布`event_pair_v1` gold、NULL/many-to-many/hard-negative strata、candidate/split receipts和强baseline复现；
- 论文主结论是哪些preparation、regrasp、failure、速度差与遮挡会击穿现有aligner；
- 不训练或不声称human-to-robot policy transfer，除非最强已有aligner本身带来稳定下游收益。

候选标题：

> **Align What Matters? A Gold Benchmark for Partial Human–Robot Event Correspondence**

### 13.5 D2整体失败：数据合同论文，而非transfer论文

如果36+36的跨域agreement、outcome provenance或同任务overlap本身不合格：

- 不做positive pairing；
- 把旧66+88只保留为task-mismatch rejection stress；
- 论文只能聚焦数据合同、false-pair防线和可观测性，不写cross-domain transfer；
- 如果数据量/类别支持仍不足，降为dataset report/workshop而非夸大主会benchmark。

### 13.6 D1/D2通过但D3失败

虽然题目重点是D1/D2，仍应预注册这个最可能的降级：event/pair gold与完整benchmark可以保留，但若oracle finite packet不胜22D或距continuous超过3 pp，论文应题为`When Finite Event Interfaces Are Not Enough`一类的representation benchmark/negative result；不能继续称SM-FET是控制方法。

## 14. Failure modes、limitations、ethics

### 14.1 预期failure modes

- **空间分辨率不足：** part-local 4×4无法表达插入、滑动、液流或精细rim alignment。
- **事件粒度错误：** active event内focus频繁变化、超过2 edges或8 events，说明固定ontology太粗。
- **missingness selection shift：** fail-closed提升precision但关键phase compile coverage过低。
- **actor ambiguity：** r1 hold/reobserve造成延迟；若allocator无法唯一分配，不能安全执行。
- **transparent ambiguity：** RGB无法判FLOW/TRANSFER/outcome，sensor缺失使关键字段unknown。
- **semantic shortcut：** text/task name诱导FLOW/SUCCESS而不是视觉证据。
- **pairing collapse：** all-null、all-match或只按duration/task identity匹配。
- **planner bottleneck：** oracle强、predicted弱；问题来自事件识别而非action representation。
- **route leakage：** image cross-attention、state→reference、cache logits或aux hidden绕过hard packet。
- **hardware mismatch：** human functional role无法映射到RH56/OmniPicker的可达、安全执行。

### 14.2 论文限制段必须主动承认

- 固定四role、八event、两edge和4×4 cell是工程假设，不是普适操作本体；
- exact capacity只属于r1 grammar，不代表实际利用或充分性；
- 36+36只做feasibility，正式policy/OOD结论需要更大、power-analyzed robot数据；
- 单个G1+两种末端最多支持目标系统内domain/role transfer，不自动支持多机器人cross-embodiment；
- 人工event ontology、reviewed pairing和sensor outcome意味着系统不是无条件end-to-end；
- transparent non-reconstruction减少伪GT，但可能以更高abstain和较低精度为代价；
- fail-closed结果可能偏向容易、可见样本，必须报告coverage与selection shift；
- 认知文献只提供消融动机，不提供人类机制或机器人收益证明。

### 14.3 Ethics 与安全

- human视频采集需有知情同意、用途/保留期限、去身份化和退出机制；避免公开脸、声音、场地标识与无关个人信息；
- 标注者劳动量、分歧和不确定性应公开，不把裁决后gold描述成客观无噪声真理；
- model proposal、MANO、VLM、depth completion和sim/native truth必须有provenance，不得混入human direct GT；
- target robot实验必须有driver/mapper/calibration/watchdog/E-stop及本地安全SOP，失败rollout不能删除；
- 液体/生化场景应从安全替代液体开始，报告spill/collision和污染处置，不能由模型paper授权危险材料部署；
- finite、auditable不等于安全或可解释；它只扩大可检查范围，实际安全需独立验证。

## 15. 论文结构建议

### 1. Introduction

从“机器人到底应复制human tool use中的什么”切入；先承认dense trajectory、skill latent、contact affordance、object address和partial alignment均已有前例，再提出缺失的是严格可审计的因果比较。结尾给出可否证假设，不写`we are the first`。

### 2. Related Work

按第10节五条线组织。每节最后用一行说明“本工作测试的不同问题”，避免把前例写成稻草人。

### 3. Problem and Information Contract

形式化training-only human evidence、deployment inputs、action-visible channel、grammar capacity、missingness和fail-closed。这里先定义claim scope，再讲网络。

### 4. SM-FET-WAM

依次讲role binding、event planner、dynamic focus、hard compiler、13-token encoder、strict WAM、training-only pose/pairing。CAPOT放在最后并明确不是headline算法。

### 5. Benchmark and Causal Evaluation

先讲数据/split/provenance，再讲D0–D4、matched controls、interventions和statistics。不要把annotation协议全部藏在supplement，因为它决定GT和claim是否合法。

### 6. Results

顺序固定：D1/D2可信度 → D3 oracle representation → route audit → D4 predicted/human transfer → failure strata。这样不能用最终平均success掩盖标签或旁路问题。

### 7. Limitations and Negative Findings

主动报告未过gate、连续路线优势、coverage下降、single/uniform持平、CAPOT无优势等；根据实际最高claim改标题和摘要。

## 16. 基于 canonical 实际 HEAD 的代码边界

本蓝图以实际HEAD `e0ebc6929ab25bd5793ef64939b4b831e15b12c2` 建worktree，并重新核对当前代码，而不是假设旧memo所用`e212afec`仍是HEAD。

当前已存在的v1/基础边界：

- `world_action_model/contact_transition_schema.py:168`：unstructured capacity metadata；
- `world_action_model/contact_transition_schema.py:340`：finite contact transport capacity metadata；
- `world_action_model/contact_transition_schema.py:417`：现有contact segment/event数据结构；
- `world_action_model/models/transformer_wa_casual_mot.py:624-865`：当前contact配置、route、capacity与condition encoder初始化；
- `world_action_model/models/transformer_wa_casual_mot.py:1253-1402`：当前contact planner/route；
- `world_action_model/models/transformer_wa_casual_mot.py:1405-1481`：当前contact/action token编码；
- `world_action_model/models/transformer_wa_casual_mot.py:1727-1807`：strict image input拒绝与image cross-attention风险边界；
- `world_action_model/datasets/wam_lerobot_dataset.py:326-367`：hand-active sidecar明确要求`physical_contact_truth=false`；
- `world_action_model/datasets/wam_lerobot_dataset.py:729-868`：现有sidecar加载入口；
- `world_action_model/trainer/wa_casual_trainer_mot.py:555-598`：现有route validation/receipt；
- `world_action_model/trainer/wa_casual_trainer_mot.py:958-999`：当前contact target/condition入口；
- `world_action_model/asymmetric_action_schema.py:92-166`：asymmetric 25D pack/split/pad边界；
- `world_action_model/asymmetric_hardware_manifest.py:177-589`：target hardware fail-closed contract；
- `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py:205-215,468-473`：当前22D contact route与capacity配置。

当前HEAD中`world_action_model/event_tube_schema.py`不存在，说明SM-FET v2仍只是研究契约。未来方法若实现，应新建独立v2 schema/planner并保持v1 disabled行为不变；本轮没有授权或执行这些修改。

## 17. 证据来源与本轮登记

蓝图完整读取并综合了以下独立memo：

- final portfolio：`gwp-agent-final-idea-portfolio/docs/research/wam_biochemical_manipulation_idea_portfolio_20260812.md`；
- novelty adversarial review：`gwp-agent-wam-novelty-review/docs/research/sm_fet_wam_novelty_adversarial_review_20260812.md`；
- benchmark roadmap：`gwp-agent-wam-benchmark-roadmap/docs/research/sm_fet_wam_benchmark_and_data_roadmap_20260812.md`；
- schema：`gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_schema_architecture_20260812.md`；
- final architecture：`gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_final_architecture_decision_20260812.md`；
- annotation/pairing：`gwp-agent-temporal-event-meta/docs/research/sm_fet_annotation_and_pairing_protocol_20260812.md`；
- cognitive evidence：`gwp-agent-human-observational-learning/docs/research/human_observational_tool_learning_evidence_20260812.md`；
- dynamic focus：`gwp-agent-sm-fet-focus-token/docs/research/sm_fet_dynamic_focus_token_spec_20260812.md`；
- missing hands：`gwp-agent-sm-fet-missing-hands/docs/research/sm_fet_missing_hands_contract_20260812.md`；
- missingness/capacity reconciliation：`gwp-agent-sm-fet-missing-hands/docs/research/sm_fet_missingness_capacity_reconciliation_20260812.md`；
- transparent non-reconstruction：`gwp-agent-sm-fet-transparent/docs/research/sm_fet_transparent_nonreconstruction_contract_20260812.md`。

本轮按`idea-discussion`工作流对总idea card做了恰好一次续写登记。自动检索返回了若干低相关结果，因此没有用它们扩张related-work claim；唯一吸收的通用审稿建议是把novelty按**任务、方法、数据、评测**四维分别核查。最终related-work边界以已有Zotero-first adversarial review和上述原始memo为主。

## 18. 最终投稿裁决

当前不是“方法已准备投稿”，而是“论文问题已经被收束为可拒绝的实验命题”。建议保留标题1和以下一句作为全项目north star：

> **SM-FET-WAM的价值不在于发明了事件、接触、partial OT或三姿态，而在于用严格有限的信息路由、等容量对照和字段干预，证明或否证机器人从human tool use中是否只需复制可重定位的角色—接触—关系—效果转折；若证据不支持，就把失败转化为可复核的观测、对齐或表示benchmark，而不是继续加组件保护故事。**

在D0/D1前，不写模型实现；在D2前，不写positive human–robot pairing；在D3前，不写finite control sufficiency；在D4前，不写predicted human-to-robot transfer。无论机器人结果多好，都不写人类固定八阶段/三姿态机制已被证明。
