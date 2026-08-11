# SM-FET 极简 action packet 对生化工具控制的充分性独立审查

- 日期：2026-08-12
- 独立工作树：`/home/ykj/project/WM/gwp-agent-sm-fet-minimal-control`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 审查对象：D0 reduction 提出的 `minimal-r1a` action-visible 候选
- 目标硬件：Unitree G1；左 RH56DFTP（6 个主动通道）；右 OmniPicker（1 个 aperture 通道、两个物理夹面）
- 范围：只做控制充分性、反例与 oracle 规格审查；未写代码、未训练、未使用 GPU、未启动仿真或真机
- `idea-discussion`：`idea_id=ca118643-f979-44e8-9526-50957b0ca236`，`discussion_id=64674e9a-a2cb-4868-8eeb-e57934f31176`，`memory_updated=true`

## 0. 裁决

**否。** 如果 strict WAM 的 action-visible 输入只剩：

```text
四个功能 role namespace
+ 被 active interaction 引用 role 的 robot-view 4x4 center cell
+ active phase kind / object effect
+ 一条 primary-role / part / secondary-role / target-relation
```

那么它不是倒液、移液/插入、旋盖、擦拭四类任务的通用充分控制状态。失败不是“字段少所以容量小”，而是存在可构造的**编译碰撞**：robot 当前视频中的两个不同执行状态，经该 reduction 后产生相同 packet；action expert 又按 strict route 看不到 raw RGB，于是它接收相同的 packet、proprio 与 text，但安全动作集合互不相交。

两个阻塞缺口已经可以由反例证明：

1. **4×4 object center 不是执行位姿。** 它无法区分同一 image cell 内的两个 well/port/rim/污渍，也不提供插入轴、杯口法向、擦拭方向、目标区域范围或允许误差。`primary_part` 只能回答“哪个功能部件”，不能回答“该部件上具体从哪里、沿什么方向执行”。因此不能把 `primary_surface_cell`/连续 locator 一概删除；更准确地说，必须保留一种达到任务公差的 **target-domain execution locator**。standard-r1 的 part-local 4×4 cell 也尚未被证明达到该要求，不能因为本审查否决 reduction 就反向宣布 standard site 已充分。
2. **一条 interaction record 不能表达两个同时独立受控的约束。** 旋盖、插入和擦拭都可同时要求“右 OmniPicker 稳定 body”与“左 RH56 在 cap/connector/wipe 上操作”；单 record 若表达 cap↔bottle、tip↔opening 或 wipe↔surface，就没有位置再表达 support/stabilize 约束。只有当第二约束已经由经过验证的 proprio/contact state 与持久 controller memory 完整承载时，才可不重复发送。当前硬件契约没有证明这种条件对所有任务成立，尤其 OmniPicker 的独立 tactile/current 接口尚未核实。

但这不支持把 standard-r1 全部字段原样恢复。逐项必要性裁决是：

- **future / onset / duration / outcome：不进入最小 action packet。** 在 `execute 8 -> reobserve`、任务目标由 text/active goal给出、阶段切换由当前证据触发的前提下，它们没有不可替代性；future outcome 还会构成后见泄漏。
- **human literal actor：不进入共享 packet。** 人类 LEFT/RIGHT 不能直接映射左 RH56/右 OmniPicker。目标机器人却必须在 action 产生前得到一个唯一 `effector assignment`；它可以是 capability-aware allocator 的确定性输出，不必是 shared event field。若 allocator 输出不进入 split decoder/control path，极简系统仍不充分。
- **site：不能整体删除。** 必须以 target-domain execution locator 的形式保留；它可能是经标定的 object/part-local region、target frame/axis 或其它有限表示，具体编码要由 oracle 公差试验决定。
- **mode：不默认保留独立字段。** 若冻结的 task ontology 能从 `phase + effect + role/part + target relation` 唯一编译出抓持、滑动、插入、旋转等 controller profile，mode 是机械派生。只有出现相同 retained inputs、相同 text/current state，却要求不同阻抗/接触模式的反例，才升为 action-visible。
- **second edge：对四任务通用主张不能删除。** 最小语义应允许最多两条**当前并发独立约束**，或提供等价且有 receipt 的 active-constraint controller state；不是为每个事件无条件占满两条 edge。
- **历史 `relation_before`：不必恢复。** 动作需要的是 current relation/error，而不是事件 entry 时可能已经过时的 relation。若 current relation/error 不能由 proprio/controller state得到，就必须由当前视频经 compiler送入 execution locator/constraint record。

因此本审查的唯一建议是：**拒绝把 `minimal-r1a` 直接升为四任务主路线；改为“active goal + execution locator + 最多两条并发约束 + 目标硬件 allocator”的最小系统契约。** 先做下文的无训练 paired-world oracle。不能以 D0 三段 human pour 的可标性、字段存储宽度或 schema 枚举代替控制充分性。

## 1. 审查口径：什么时候字段才是“必要”

### 1.1 strict route 下的真实 action-side 输入

上位架构规定 strict action expert 只读：

```text
robot proprio/state
allowed instruction text
hard event packet
diffusion timestep / noisy action
```

raw current RGB 只供 planner/编译器使用，不能被 action rows、state rows、cache 或 image cross-attention 读取。因此必须区分两句话：

- “变量可从 robot 当前视频看到”：说明 planner 有潜在证据；
- “变量可由 action expert 恢复”：只有 compiler 将必要差异写入 hard packet，或另一个允许的控制状态明确承载它，才成立。

当前 canonical 的 attention mask已有可切换的 reference 路径，位置在 `world_action_model/models/transformer_wa_casual_mot.py:127-194`；现有 planner从 current reference产生 contact semantic，位置在 `:1531-1590`。v2 strict 主张要求关闭 action/reference 旁路。因此本审查不允许用“网络也许从图像自己看出来”替极简 packet解围。

### 1.2 必要性的反例定义

令：

```text
P_t = 目标机器人 proprio、已验证 contact/force/current 与 controller memory
T   = 部署时允许的 instruction text
V_t = 当前 raw video / causal history，只对 planner 可见
M_t = 极简 compiler 输出的 hard packet
A*(s) = native/sensor-confirmed state s 下能安全推进当前事件的参考动作集合
```

被删信息 `z` 只有在以下条件同时成立时，才判为 action 侧必要：

1. 存在两个 admissible state `s1,s2`，使 `P_t`、`T`、controller configuration 与 `M_t` 相同；若是视觉量，允许 `V_t` 不同但被极简 compiler 压成同一 `M_t`；
2. `A*(s1) ∩ A*(s2) = ∅`，即不存在一段共同的、非abstain且能推进任务的安全 first-execute action；纯hold另按risk/coverage与time-to-resolution评估；
3. `z` 不能由 `P_t`、`T`、冻结 task ontology、hardware manifest 或确定性 allocator机械推出；
4. 只增加 `z` 后可区分这对状态，并且 swap `z` 只改变相关腕/手/阶段动作。

反过来，只要存在有 receipt 的确定性派生函数，字段就不进入 event packet。不能用“可能有用”“提高预测分数”作为保留理由。

### 1.3 两类碰撞

本审查使用两类严格反例：

1. **视觉编译碰撞。** 两个 raw robot view 中的执行局部几何/支撑关系不同，但 role、4×4 center、part、phase、effect、target relation相同；action expert没有 raw video，因此收到相同输入。
2. **计划编译碰撞。** 当前物理状态相同，但 planner 选择了不同且不等价的目标硬件分配/并发约束；若 packet 与 allocator output均未表达该决定，split 25D action head无从选择该驱动哪些腕/手通道。

第二类不能用来保留 human actor；它只要求 robot execution assignment 在 action/control路径中存在。

## 2. 哪些信息可从哪里得到

| 信息 | robot proprio / controller | instruction text | current video / history | 必须进入 hard packet 的条件 |
|---|---|---|---|---|
| 双腕当前位姿、RH56 六通道、Omni aperture | **可得**，前提是 target hardware已标定 | 否 | 可作视觉校验 | 不重复进入 event packet |
| 当前 grasp/contact/slip/load | RH56 只有经标定 tactile/force才可得；Omni 目前未证明有可靠独立 tactile，aperture本身不等于接触 | 否 | 可见运动一致性只能作证据，不等于力/接触真值 | 若 controller state不能可靠承载，active constraint/current relation必须进 packet或 fail closed |
| 任务 family、tighten/loosen、目标剂量、目标 well 名 | 否 | **只有文本明确写出才可得** | marker/label可辅助grounding | 文本不含的 goal argument必须进入 planner goal/packet，不能靠任务常识猜 |
| 当前 phase | 局部接触状态可辅助，但不完整 | 通常不更新阶段 | causal video/history可判 | 已由 retained `active kind` 承载 |
| object/part identity与粗位置 | 不可从25D proprio得到 | 可给语义名，不能完成视觉grounding | **可得但受遮挡/透明影响** | role binding与robot-view locator必须编进 packet |
| well/port/rim/污渍的精细位置、轴、范围 | 仅在已建立且标定的 object-to-hand transform下部分可得 | 文字可指称，不能给当前相机位姿 | 可由当前RGB/native state估计 | strict action不看RGB，因此必须进入 execution locator或由外部视觉伺服直接闭环；后二者与“packet唯一视觉桥”不兼容 |
| fixture/support、当前 object-object relation | 一般不可从手状态推出 | 通常不写实时状态 | 当前视频/native state可得 | 若改变当前安全动作，进入 current constraint/state；不能只给 terminal relation |
| effector assignment | proprio只给可达性输入，不自动给planner决定 | human left/right不应决定 | 几何可参与allocator | 确定性 allocator输出必须进入 split decoder/control；allocator多解则hold/reobserve |
| contact mode / impedance profile | controller可以由冻结 ontology编译 | 任务动词可提供约束 | 视频只能观察结果，不能给期望力 | 可唯一派生时不进 packet；不能唯一派生时该 profile成为required field |
| event outcome / transfer量 /清洁完成 | 力、秤、液位等传感器可给；当前仓库未普遍具备 | 只给目标，不给实际结果 | 可见结果可给证据；透明flow/volume常不可判 | 进入 supervisor/next phase，不作为当前 future outcome condition |
| future events / coarse duration | controller clock与每8步replan可更新 | 完整任务目标可给远期意图 | history可预测 | first-execute动作能由active goal决定时不进 packet |

### 2.1 25D proprio 的边界

canonical 目标动作/状态布局已在 `world_action_model/asymmetric_action_schema.py:18-28,92-163` 冻结：

```text
left wrist 9
+ right wrist 9
+ left RH56DFTP 6
+ right OmniPicker aperture 1
= 25 physical dimensions，pad 到 32
```

这 25 维能告诉 action expert“机器人当前是什么姿态/开度”，不能告诉它“目标 well 在哪里”“容器是否由 fixture支持”“目标 relation 是否已建立”。RH56/Omni 指令是 absolute open-fraction target，腕为 local relative transform；human grip proxy也不能直接当 Omni command。

另外，当前仓库只返回 canonical 25D，没有目标 G1 OmniPicker driver、canonical-to-device reverse mapper、endpoint、watchdog和bench/robot闭环，见 `docs/asymmetric_action_schema.md:126-164`。所以本 memo 的 oracle 是未来 contract test，不是当前硬件执行许可。

## 3. 四类任务的反例

### 3.1 倒液

#### P1：同一 4×4 cell 内的不同 target-rim 位姿

固定：

```text
proprio       = source 已由同一末端以同一 wrist/hand state持有
text          = “向目标烧杯倒液”
active kind   = MANIPULATE
effect        = TRANSFER
primary       = destination / OPENING_OR_RIM
secondary     = source
target relation = ALIGNED_TO_RIM
center cell   = 两个场景相同
```

只把 target rim 在该 cell 内横移，或改变其可见椭圆轴/开口法向；位移选到让两个安全 pouring corridor不相交。极简 packet完全相同，正确腕平移/倾角起点不同。proprio只知道手里 source 的姿态；text不给 target 当前位姿；raw current video有证据，但 strict action看不到它。

**必要结论：** packet必须承载 target-domain rim execution locator，至少区分安全 corridor所需的局部位置与approach/tilt axis。只保留 object center不充分。

#### P2：自由 target 与已稳定 target

两个场景保持 source/target role、center、part、transfer goal相同；一个 target 被 rack/fixture可靠支持，另一个只是松放且在预注册 pour流量/碰撞下会倾倒。fixture状态在 raw view/native state可见，但不在单 record中。

- 已支持：另一末端不应无故接近，避免碰撞；
- 未支持：必须先/同时建立 target stabilization，或hold/reobserve；
- 单 record若写 source↔target alignment，就没有第二个并发 support constraint。

**必要结论：** 对通用 pour，要么 controller state已带有经传感确认的 target support，要么 packet允许第二条当前约束；不能从“桌面可见”或 aperture值推定已稳定。

#### P3：继续倒还是停止

目标剂量固定后，继续/停止应由可见液位、秤/flow/volume sensor或明确 supervisor判据决定。`duration`和future `outcome`只是预测，不能替代实际 transfer evidence。透明流不可见时必须hold/reobserve或使用传感闭环。

**删除结论：** 该反例不支持恢复 duration/outcome；它支持把结果/流量放在 sensor/supervisor，而不是 action future field。

### 3.2 移液 / 插入

#### I1：同 cell 多 well / 多 port

把两个目标 well/connector port放在同一 robot-view 4×4 cell，且共享：

```text
role = receptacle_or_destination
part = OPENING_OR_RIM
effect = TRANSFER 或 TRANSLATE
target relation = INSIDE
```

text可以只说“移入标记孔”；不同场景由视觉 marker指定不同孔。planner可看见marker，但极简 compiler只输出相同 role/part/cell/relation。两孔间距大于安全插入公差时，first approach方向和末端位置不同，安全动作集合不相交。

**必要结论：** `primary_instance + part-local execution region` 必须达到孔/井级分辨率。四 role固定slot不能让多个同role实例静默共享一个binding；多解时必须overflow/hold，而不是挑最近中心。

#### I2：同中心、不同插入轴或剩余深度

保持 port center落在同一 cell，改变 port轴向；或者保持 wrist proprio相同，让 receptacle在cell内发生小位移，使 tip到目标深度的剩余误差不同。active kind都可处于同一个 maximal `MANIPULATE`/插入事件，target relation均为 `INSIDE`。

- 状态A要求继续沿轴推进；
- 状态B已接近底部，继续同样推进会碰撞，应减速/停止/转入下一阶段。

`target_relation=INSIDE`只描述终端类别，不能提供current relative error。旧 `relation_before`又是event-entry历史，不是所需量。

**必要结论：** execution locator/current constraint必须包含插入轴、容差以及当前相对误差的可执行表示；可由native pose+proprio机械求出时无需另字段，否则必须经packet传递。

#### I3：单手插入与“稳定 socket + 插入”

同样的 connector↔opening goal，在 socket已固定与会随推力移动两种情况下，左 RH56的connector动作可以相同语义，但右 OmniPicker是否必须先稳定socket不同。单 record不能同时表达 tip↔opening和right-effector↔socket support。

**必要结论：** 对非固定 receptacle，第二并发约束或等价verified controller state是必要条件。

### 3.3 旋盖

#### C1：cap↔bottle 旋转与 body stabilization 同时存在

目标为 `ROTATE(cap relative to bottle)`：

- 左 RH56 的多指/拇指旋转能力适合操作 cap；
- 右 OmniPicker 的1-DoF coupled jaws适合夹持/stabilize bottle body；
- 一条 object relation只能表达 cap↔bottle；它不能同时说明 bottle body必须由右末端维持稳定。

构造瓶体被fixture锁定与未锁定的 matched pair。raw view/native support状态不同但极简 packet相同。前者只需左侧旋转；后者必须右侧先建立/维持稳定，两个25D action的右腕/aperture分量不同。

**必要结论：** 至少需要两个当前约束，或一个已验证的persistent support state。该结论来自独立物理约束数，不来自“standard-r1本来有两条edge”。

#### C2：human actor不必要，robot assignment必要

若 hardware ontology与reachability给出唯一分配：

```text
precision/roll cap -> left RH56
stabilize body     -> right OmniPicker
```

则 shared packet不需要 human LEFT/RIGHT。allocator必须读取 functional constraints、target geometry、25D proprio与capability manifest，输出唯一 target assignment并让split decoder可见。若 assignment多解且后果不同，系统必须hold/reobserve；不能让WAM从数据相关性暗选。

**删除/保留结论：** 删除 human actor；保留 robot-side assignment receipt/control signal。没有后者的“actor-free WAM”不充分。

#### C3：tighten / loosen 与 contact mode

若 text明确写 `tighten`或`loosen`，且 `phase=MANIPULATE, effect=ROTATE, part=LID_CAP`由冻结ontology唯一映射为stick-grasp + signed rotation controller，则 rotation方向和mode都可机械派生。

**删除结论：** 该任务不构成独立 `contact_mode` 字段的必要性证据。若部署text不保证方向参数，缺的是 goal argument，不是把所有历史 contact mode恢复。

### 3.4 擦拭

#### W1：同一 surface/cell 内不同污渍

在同一大工作面、同一4×4 cell内放两个不重叠污渍区域；text只说“擦除标记污渍”，raw image的marker决定目标。两场景的：

```text
primary role/part = fixture_or_support / BODY(or surface)
secondary role    = tool
effect            = TRANSLATE（当前词表没有独立CLEAN）
target relation   = TOUCHING
center cell       = 相同
```

极简 packet相同，但起始接触点与第一段切向动作不同。一个共同轨迹会遗漏一处或触碰禁止区。

**必要结论：** target region不能退化成object center；必须有part-local区域与允许/禁止边界。

#### W2：同中心、不同擦拭方向/范围

让两个细长污渍具有相同中心但正交主轴。角色、part、effect、relation和center全部相同；安全参考动作分别沿两条正交切向路径。`TRANSLATE`没有方向/extent，timing也不能给方向。

**必要结论：** execution locator必须能表达局部path axis/extent或等价运动目标。若只有action-visible event track/continuous trajectory能满足公差，就应停止finite locator充分性主张，而不是把方向藏进planner hidden。

#### W3：workpiece已固定与需另一手稳定

在已夹紧tray与未夹紧tray上擦同一位置。未固定时，右 OmniPicker需先/持续稳定，左 RH56持wipe并滑动；已固定时不需要右手接近。单 interaction record无法同时表达 wipe↔surface slide和effector↔tray stabilize。

**必要结论：** 这是第二并发约束的直接反例；若proprio/controller state能经native sensor确认tray已稳定，则该约束可以机械派生，但必须有receipt，不能由“右aperture当前较小”猜测。

## 4. 被删字段的最终必要性裁决

| 候选信息 | 裁决 | 必要性论证 / 允许的替代来源 |
|---|---|---|
| 4-role namespace | **保留** | 是跨任务功能binding的最小共同语义；同role多实例仍需instance binding或fail closed |
| robot-view 4×4 node center | **保留作粗routing，但不充分** | 可区分大范围对象位置；P1/I1/W1证明无法承担执行局部几何 |
| active phase kind | **保留** | 同物理场景的approach/acquire/manipulate/release安全动作不同；不能只由单帧proprio稳定推出 |
| active object effect | **保留** | 相同role/part可执行translate/rotate/transfer；text未必逐phase展开 |
| primary role/part | **保留** | 决定功能对象与部件；object center或文本对象名不能代替part |
| secondary role + target relation | **保留** | 插入、倒液、旋盖需要相对目标；单object目标不足 |
| `future next1/next2` | **删除默认action可见性** | execute-8/replan与完整task text下，尚无不同first-execute安全动作的必要反例；用anticipatory grasp oracle继续证伪 |
| onset / duration | **删除** | 当前phase、proprio、current relation和反馈决定动作；粗duration不是flow/torque/接触真值 |
| human literal actor | **删除shared字段** | human laterality不映射异构末端；由target allocator替代 |
| target robot effector assignment | **系统级必须可见** | 同一functional plan写入25D的不同slice；可作为allocator输出而非shared event field |
| old `primary_surface_cell` | **不能无条件删除，也不能原样宣布充分** | P1/I1/W1证明需要execution locator；实际分辨率/轴/extent由任务公差oracle决定 |
| node scale/orientation | **不独立保留** | 只有对execution locator不可机械派生的轴/尺度才保留；避免把相机局部装饰字段整体恢复 |
| historical `relation_before` | **删除** | 动作需要current relation/error；event-entry历史可过时 |
| current relation/error | **条件必需** | proprio/controller能确定则外置；否则由当前视频编译进constraint/locator |
| contact mode | **默认机械派生** | 由phase/effect/part/relation映射到versioned controller profile；映射多解时fail closed或promotion |
| outcome | **移到supervisor** | current action不应读取future result；可见/传感结果触发next phase/recovery |
| second active edge/record | **保留到最多2条，或等价active-constraint state** | P2/I3/C1/W3证明存在两个独立并发约束；只在第二约束已由verified control memory承载时省略 |

## 5. 建议的最小必要系统契约

这不是新 schema 的 wire 定稿，也不包含任何容量数字。它只冻结控制上不可继续删除的语义。

### 5.1 Embodiment-neutral active goal packet

```text
active_phase_kind
active_effect_goal

constraints[1..2]:
  primary_role
  primary_instance_binding
  primary_part
  execution_locator
    - target-domain local region/frame
    - task-required approach/path axis
    - tolerance/extent class when not fixed by ontology
  secondary_role | NONE
  secondary_instance_binding | NONE
  current_relation | DERIVED_FROM_VERIFIED_STATE
  target_relation | N/A
```

约束：

1. `constraints[1]`只有在第二个独立物理约束不存在，或已由verified controller memory承载时才合法；
2. `execution_locator`必须从 robot 当前 observation/native state生成，human像素不复制到robot；
3. locator达到哪种空间分辨率由任务安全公差决定，不预先承诺standard 4×4 site、连续ROI或3D pose；
4. required instance/locator/current relation未知时fail closed；不能把unknown center当可执行目标；
5. task text必须有冻结接口，明确哪些goal arguments（剂量、tighten/loosen、指定well）由text提供；未提供者进入goal packet，不准隐式从dataset名推断。

### 5.2 Target-robot execution binding

在 shared packet之后、split action decoder之前，必须有：

```text
TargetEffectorAllocator(
  active constraints,
  robot proprio/contact state,
  execution locator geometry,
  versioned hardware capability manifest,
  safety/reachability constraints
)
-> per-constraint assignment:
     LEFT_RH56 | RIGHT_OMNIPICKER | OBJECT_ONLY
   + infeasible / ambiguous flag
   + receipt
```

assignment不是human actor的别名。它决定同一语义写入：

- 左腕9D + RH56六通道；
- 右腕9D + Omni aperture一路；
- 或两侧协同。

若唯一 assignment可机械求出，它不增加 shared event字段；但 action decoder/control必须读取其hard输出。若多个assignment后果不同且无唯一安全解，返回infeasible/ambiguous并hold/reobserve。

### 5.3 可留在 action 外的状态

```text
task text:
  task family, named target, dose, tighten/loosen, success criterion

robot proprio/controller:
  current wrists/hands, calibrated contact/load/current,
  held-object transform, persistent established constraints,
  safety limits and active controller state

supervisor:
  outcome, transfer/clean completion, recovery, terminal decision

planner-only:
  future event tube, onset/duration forecast, confidence,
  observability/missingness/provenance
```

这些状态只有在source与更新规则被版本化、可测且有receipt时才能作为“字段可删”的依据。把它们写在架构图上但不给action/control实际读取，不构成替代。

## 6. 无训练、可证伪的 paired-world oracle

### 6.1 总体程序

不用模型训练。为每个候选字段构造native/sensor-confirmed matched pair：

1. 冻结 task text、25D proprio、controller memory、hardware manifest、replan anchor与安全约束；
2. 只改变一个候选变量，例如rim sub-cell pose、port axis、fixture support或stain path；
3. 让极简 compiler分别编译，要求两边得到byte-identical的 retained semantic values；
4. 用确定性几何/力/接触 reference controller求每边的安全 first-8-step action set，而不是用另一个learned policy当oracle；
5. 若两个安全集合不相交，记录为minimal packet collision；
6. 只加入一个候选field或确定性外部状态，重新编译；它必须消除碰撞；
7. swap该field，其余输入和noise固定，动作变化必须局限在预期腕/手/阶段；
8. 同时报告hold/reobserve。如果一个共同的hold动作可以无限拖延但不能完成任务，它不算sufficiency；若一次reobserve能从允许通道消除歧义，则按safe abstention另报。

判定单位是“安全动作集合是否可同时满足”和“闭环任务是否完成”，不是字段分类F1、packet容量或注释一致率。

### 6.2 四组核心 oracle pair

| Oracle | 唯一改变 | 极简字段为何相同 | 参考动作必须不同 | 通过后说明 |
|---|---|---|---|---|
| `O-LOC-POUR` | target rim在同center cell内横移/转轴 | role/part/effect/relation/cell均相同 | wrist平移、对准/倾倒轴不同 | execution locator必要；若一条动作在全部公差内都安全，则该分辨率必要性被证伪 |
| `O-LOC-INSERT` | 同cell的well/port或port axis | 同一destination/opening/INSIDE | approach位置/方向不同 | instance/site/axis必要；若text+proprio能唯一机械求出则可外置 |
| `O-LOC-WIPE` | 同中心的污渍区域/主轴 | 同surface/tool/TRANSLATE/TOUCHING | 接触点、路径方向/extent不同 | finite locator必须表达path goal，或让continuous track challenger获胜 |
| `O-CONSTRAINT` | fixture locked vs unlocked，几何中心不变 | 单record只表达主操作关系 | 是否建立/维持另一末端support不同 | second constraint必要；若verified proprio/controller memory已区分并成功，则可省第二record |

### 6.3 对“已删除字段”的反证 oracle

这些测试用于防止因为本 memo 否决 extreme reduction，又无证据地把字段全部加回：

| Oracle | 操作 | 必要性成立条件 | 否则裁决 |
|---|---|---|---|
| `O-FUTURE` | 固定current state/active goal，交换next1/next2 | first-8安全动作集合必须不相交，且差异不能由text/terminal goal推出 | future保持planner-only |
| `O-TIMING` | 固定phase/current error，交换duration bin | reference速度/停止决策必须不同，且传感反馈不能替代 | duration保持planner/control-only |
| `O-ACTOR` | 固定functional constraints，交换human left/right | target allocator输出和reference action不应变化 | human actor永久移出shared packet |
| `O-ASSIGN` | 固定semantic constraints，交换唯一target assignment receipt | 只应交换/改变对应left6/right1 action slice | allocator输出必须进入execution path |
| `O-MODE` | 固定phase/effect/roles/part/relation/text/current state，构造两种必需controller profile | 只有安全模式集合不相交且ontology不能唯一派生时成立 | mode保持机械派生 |
| `O-OUTCOME` | current evidence固定，交换future/success label | verify前action必须不变 | outcome留supervisor；若变化则判后见泄漏 |

### 6.4 极简路线的 Go / No-Go

只有全部满足才可把修订后最小包称为四任务oracle-sufficient：

1. 所有 `O-LOC-*` pair不再发生相同packet、异安全动作碰撞；
2. `O-CONSTRAINT`中一条或两条约束的admission由机械规则决定，不静默漏掉support；
3. target allocator对每个合法packet给唯一assignment或明确abstain，且split action只响应对应assignment；
4. future/timing/mode/outcome消融不造成独立oracle gap；若造成，只promotion已通过paired-world test的字段；
5. 相对full/native trajectory reference，finite locator在插入、rim alignment与wipe path的安全公差内成立；若不成立，停止finite sufficiency claim；
6. action固定packet/proprio/text/assignment后替换raw reference、planner hidden、human视频与idea-discussion metadata，pre-safety action不变。

任一阻塞碰撞仍存在：`minimal-r1a` No-Go。不能用更长训练、更大模型、unknown token或容量描述掩盖不可辨识性。

## 7. 对现有 D0 与 standard-r1 的含义

### 7.1 D0 human pour没有证明 reduction 的控制充分性

三段 episode 43/50/58 的只读审计只证明source/target role、粗center、部分event/effect/relation存在proposal evidence。它同时报告：

- `primary_surface_cell` 在3/3不可可靠评估；
- rim/contact interface在关键turn受遮挡；
- 另一只手全程画外，完整hand roster在0/3可证明；
- outcome、flow、volume和success不可评估；
- tool/fixture正例与second-edge/bimanual覆盖不足。

这些事实支持“不要把不可标字段冒充GT”，不支持“该字段对target robot控制可删”。可标性失败与控制非必要是两个不同命题。

### 7.2 standard-r1也仍只是挑战者

本审查不接受以下反向推理：

```text
minimal-r1a 不充分
=> standard-r1 的 future/timing/actor/site/mode/outcome/second edge 全都必要
```

standard-r1 的优点是明确给了node、future、两edge和part/cell的可干预位置；但：

- 4×4 part-local cell未证明满足well/port/rim/wipe公差；
- `orientation_2d`不是metric insertion/pour axis；
- `relation_before`可能是过时entry状态；
- literal actor混合human laterality和robot assignment；
- outcome/duration/future仍缺当前action必要性反例。

所以正确比较是：

```text
extreme minimal-r1a
vs
本 memo 的 minimal necessary system
vs
standard-r1
vs
full/native trajectory or direct visual servo ceiling
```

字段只按paired-world oracle逐项升降，不按schema来源、存储整齐度或“信息上界”投票。

## 8. Idea-discussion 登记与证据边界

本轮按用户要求运行 Zotero-first `idea-discussion --mode critic`：

```text
idea_id        = ca118643-f979-44e8-9526-50957b0ca236
discussion_id  = 64674e9a-a2cb-4868-8eeb-e57934f31176
memory_updated = true
zotero_synced  = 100 records
```

自动检索返回 Rubik's Cube benchmark、ARMBench，以及明显误召回的 image-manipulation detection benchmark；Zotero近邻为通用 RoboBrain 2.0 note。它们没有提供本题“极简事件包对异构双手生化工具控制是否充分”的字段级证据，因此不用于保留/删除裁决。本文的结论来自项目内strict route、目标硬件contract、D0可观测性事实和四类paired-world counterexample。

## 9. 只读证据来源

本轮完整读取或定向核对以下版本：

- D0 observability：`gwp-agent-sm-fet-d0-observability/docs/research/sm_fet_d0_three_episode_observability_audit_20260812.md`，commit `664dfc6c5baa10a709b2996178a7238f48247f20`；
- D0 reduction：`gwp-agent-sm-fet-d0-schema-reduction/docs/research/sm_fet_d0_schema_reduction_review_20260812.md`，commit `fdac965a181e7e3acef3510e788ae4661931197f`；
- standard schema：`gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_schema_architecture_20260812.md`，commit `190f3023da6ea6e0ee8251da22e2f21efea17bcc`；
- final architecture：`gwp-agent-temporal-event-meta/docs/research/sm_fet_wam_v2_final_architecture_decision_20260812.md`，commit `6bc51b90002f7f533d7b1ff523caf562949eab64`；
- dynamic focus：`gwp-agent-sm-fet-focus-token/docs/research/sm_fet_dynamic_focus_token_spec_20260812.md`，commit `2452e60bbaf6edb4714fed7d1397aea6ba9e4111`；
- missing hands：`gwp-agent-sm-fet-missing-hands/docs/research/sm_fet_missing_hands_contract_20260812.md`，commit `ca7d641703749a15ff30c644250f5147b5125e72`；
- missingness reconciliation：`gwp-agent-sm-fet-missing-hands/docs/research/sm_fet_missingness_capacity_reconciliation_20260812.md`，commit `9f6eeff6761168048b49101cb5839170ae6b84bc`；
- transparent non-reconstruction：`gwp-agent-sm-fet-transparent/docs/research/sm_fet_transparent_nonreconstruction_contract_20260812.md`，commit `f795c4f615595d72219e79a0cb43d5adef12b0a0`；
- asymmetric25 hardware：`gwp-agent-asym25-hardware/docs/asymmetric_action_schema.md` 与 `docs/audits/grab_red_bottle_asymmetric25_audit.md`，commit `6d7deeac076f96f4ad72f2ab3b2180e3a5516926`；
- asymmetric topology/allocator：`gwp-agent-human-robot-dataset-survey/docs/research/idea_b_asymmetric_contact_topology_review_20260811.md`，commit `8e2e13e8953548415e509a603c3e114b22b817de`；
- canonical code：`world_action_model/asymmetric_action_schema.py:18-28,92-163` 与 `world_action_model/models/transformer_wa_casual_mot.py:127-194,1531-1590`，基线 `e0ebc692`。

## 10. 最终一句话

`minimal-r1a` 删除的不是一组“也许有帮助”的字段，而是混合了可删项与控制必要项：future、粗timing、human actor、outcome没有必要性证据；但把执行site压成object 4×4 center、把多个并发物理约束压成单record，会在倒液、插入、旋盖和擦拭上制造相同action输入对应互不相容安全动作的反例。最小可守方案必须让目标机器人看到**达到任务公差的 execution locator、当前relation/error、最多两条并发约束，以及allocator给出的唯一RH56/OmniPicker assignment**；其余字段只有通过paired-world oracle才允许promotion。
