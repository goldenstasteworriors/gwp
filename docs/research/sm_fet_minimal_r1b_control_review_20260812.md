# SM-FET `minimal-r1b` 四任务控制充分性独立复审

- 日期：2026-08-12
- 独立工作树：`/home/ykj/project/WM/gwp-agent-sm-fet-minimal-control`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 被审 memo：`sm_fet_minimal_r1b_schema_capacity_20260812.md`，commit `e31208172d194511e7fc95acd6075635c249c335`
- 复审依据：上一轮倒液、移液/插入、旋盖、擦拭 paired-world counterexamples
- 范围：schema语义、target-domain locator、双约束、目标末端分配、capacity口径与allocator防旁路；未写代码、未训练、未使用GPU、未执行仿真或真机

## 0. 总裁决

**当前版 No-Go for schema freeze / 四任务充分性；Conditional Go for 修订后的 CPU contract 与无训练 oracle。**

`minimal-r1b` 正确修复了 `minimal-r1a` 的两个主要缺口：

1. 不再用 full-object 4×4 center冒充执行site，而是加入target-robot camera-frame locator；
2. 不再把所有场景压成单interaction，而是允许两条constraint，并把目标机器人assignment写入wire。

它也正确拒绝恢复没有必要性证据的future、duration、human laterality和outcome。allocator显式只输出assignment enum的方向是对的。

但当前契约仍有五个阻塞问题：

1. **locator只有一个axis。** PATH必须同时约束擦拭切向和接触面法向；roll-sensitive connector、pipette、带key/flat的cap也需要axis以外的roll或第二轴。`POINT/AXIS/PATH`当前不能表达完整局部frame。
2. **两条constraint没有功能标签。** `LEFT_RH56/RIGHT_OMNIPICKER`只回答“谁驱动”，不回答该constraint是主操作、稳定/维持还是被动监测。全局phase/effect会被同时施加到两条constraint，不能机械区分“左手旋cap、右手保持bottle”或“左手抬起wipe、右手继续稳定tray”。
3. **禁止same-effector double会拒绝真实移液。** 一个RH56可同时需要保持pipette tip的位姿/插入约束并用拇指按plunger；把两者合成一个frame/path会丢掉独立手指目标。现有part词表还没有`PLUNGER/ACTUATOR`和`WORKING_FACE/PAD`。
4. **profile与continuous state边界没有闭合。** 相同 locator bytes 在不同 `[z_min,z_max]` / extent profile下代表不同metric目标；25D proprio也不含任意held-object transform、camera pose或contact validity。若这些量经config/controller私下改变动作，便形成未计入的action通道。
5. **`81.47790257814123 bit` 的算术未见错误，但“exact reachable”措辞过强。** role–part compatibility、task-profile locator kind/tolerance、assignment reachability与resolution admission仍会拒绝大量被公式计入的tuple。它是当前**profile-agnostic structural grammar upper bound**；不是任一固定部署profile的exact reachable action alphabet，也不是总action bottleneck。

因此，r1b不能按原文“冻结为唯一折中”。必须至少修改：

```text
每constraint新增 constraint_function
  = OPERATE | STABILIZE | MONITOR

locator新增可选第二方向量
  = roll / surface_normal / second_axis
  # PATH至少需要 tangent + normal，roll-sensitive AXIS需要axis+roll

part补齐
  = ACTUATOR_OR_PLUNGER
  = WORKING_FACE_OR_PAD

允许经hardware profile证明可联合控制的 LEFT_RH56 + LEFT_RH56 双constraint

冻结全局metric量化表，或把profile_id变成显式action context并改容量口径

明确扩展robot state allowlist；若held transform/camera pose/contact state进入action，
81.4779 bit只能称conditional finite visual-goal route capacity
```

修订后也只能得到 **Conditional Go**：64×64×32、方向/extent的每个task profile仍必须用native paired-world公差试验逐项通过。透明rim/深度、细pipette port、roll-sensitive插入和曲线擦拭中任一失败，该profile应转向direct visual servo、F-PCD或continuous trajectory，而不是继续扩hidden通道。

## 1. 分项裁决

| 审查项 | 裁决 | 核心理由 |
|---|---|---|
| 64×64×32 camera anchor | **Conditional Go，target robot only** | 比4×4能消除部分碰撞，但精度与coverage依赖camera/depth/profile；不可作为human GT或跨域共享坐标 |
| 512 directed axis | **Conditional Go for roll-invariant tasks** | 单axis可覆盖直线插入/approach；不能覆盖绕axis roll或同时需要切向+法向的任务 |
| PATH axis + 8 extent | **No-Go for generic wiping** | 只表示直线中心线；没有surface normal、宽度、曲率、禁止区或多段path |
| 两条constraint | **Conditional Go for one operation + one support** | cap/insert/wipe基本拓扑可覆盖；same-effector compound、异步release与三独立约束仍overflow |
| L/R/MONITOR assignment | **方向正确，但当前不足** | 末端身份显式且已计入；缺constraint function，MONITOR不是active stabilization的替代 |
| phase/effect | **Conditional** | 能表达粗active goal；global effect无法独立标记support constraint，方向/剂量仍依赖冻结text goal args |
| role/part/relation | **No-Go as current closed vocabulary** | 缺plunger/actuator、wipe working face；primary与locator目标语义仍需明确，粗relation不足以承载current error |
| 81.4779-bit口径 | **Go for给定结构公式；No-Go for deployed exact/headline total bottleneck** | profile/admission继续删状态；text、profile、proprio/held transform是条件通道 |
| allocator防旁路 | **Conditional Go** | assignment-only接口良好；profile选择、held transform/contact/camera state、controller lookup仍可旁路 |
| 四任务通用充分性 | **No-Go current r1b** | pipette compound、wipe frame、support intent与profile/channel边界尚未闭合 |

## 2. Locator：64×64×32不是“过精或足够”的单一答案

### 2.1 对 target-robot oracle：不是过精，仍可能过粗

64×64在640×480图像上约对应10×7.5 pixel cell。这个分辨率相对r1a 4×4确实能区分更多rim/well/stain，但“每轴多16倍”不是控制充分性证据。安全误差要经camera model转换到task metric frame：

- 同样的pixel bin在不同depth上对应不同横向metric误差；
- depth的32 bins由profile workspace范围决定，workspace放宽会直接降低深度精度；
- port、pipette tip、microtube opening的公差可能小于一个uv/depth bin；
- rim被透明、反光、手部遮挡时，支持区域可能跨多个bin，最近bin不是GT；
- 量化边界附近会产生locator flicker，必须报告admission coverage和reobserve rate。

所以正确结论是：64×64×32可作为**候选上限固定的target-domain oracle locator**，每个profile按metric误差准入。它既没有因字段数多而“过精”，也没有因分辨率高而自动“足够”。

### 2.2 对human annotation：No-Go

D0 human pour已经报告：关键turn中的rim、contact interface和part-local site被遮挡，standard的part-local 4×4 cell在三段中都不可可靠评估。让同一human Ego视频标64×64 anchor、metric depth、512方向和8档extent只会把不可见性伪装成高精度标签。

r1b称自己是target robot action packet，因此可避开这个矛盾，但必须冻结：

```text
human/robot pairing共享：phase, effect, role, part, target relation
human side不监督：robot uv/depth/axis/extent, robot assignment
robot locator GT来源：native state、校准marker/stereo/depth或独立robot-view审阅
```

如果未来训练把human点击、completed depth、human-to-robot warp或VLM方向当r1b locator GT，locator route直接No-Go。

### 2.3 跨域：不共享坐标，只有语义共享

camera anchor是目标域当前视图量，不能进入CAPOT或human–robot主pair cost。即使human与robot都使用64×64 code，同一个整数也不代表同一物理位置。

因此论文/架构只能写：

> r1b在共享功能事件之上附加target-robot local grounding。

不能写：

> 81.4779-bit packet是human与robot共享的完整语义瓶颈。

其中大部分locator与assignment状态是robot-domain control grounding，不是跨域语义。

### 2.4 单axis缺失完整frame

`AXIS`给一个有向单位向量；`PATH`给同一向量和extent。以下状态仍会碰撞：

1. **roll-sensitive insertion：** 两个connector有相同anchor和插入轴，但key/flat绕轴旋转90°；正确wrist roll不同。
2. **pipette orientation：** tip轴相同，但plunger朝向需要落在RH56拇指可达侧；同axis不同roll对应不同手姿。
3. **擦拭：** path tangent相同，但workpiece surface normal不同；必须改变腕姿与法向载荷方向。
4. **tool working face：** pad/刷头中心和运动方向相同，但工作面朝向不同；只有一个axis不能同时表示运动方向和接触面朝向。

这些pair在当前r1b得到相同locator，proprio/text/global relation不能机械恢复差异。

**必须修改：** locator增加可选第二方向量。最低语义不是“再加一个也许有用的orientation”，而是：

```text
POINT: anchor
AXIS_ROLL_INVARIANT: anchor + directed axis
FRAME: anchor + directed axis + roll/orthogonal second axis
PATH_ON_SURFACE: anchor + tangent + surface normal + extent
```

若task profile能证明surface normal或roll由固定fixture/held transform唯一派生，可以用`DERIVED`并不编码；否则required field缺失即fail closed。新增字段会改变wire和capacity，不能继续沿用当前81.4779数值。

### 2.5 PATH不覆盖通用擦拭区域

直线axis+length只覆盖窄直stroke。它不表达：

- 污渍/允许区域宽度；
- 曲线或多段path；
- 需要避开的边界/禁触区；
- overlap/coverage pattern；
- 多pass或往返方向。

对预注册的直线单pass擦拭profile，PATH可Conditional Go；对“擦净一块任意形状区域”的通用任务，当前PATH No-Go。若必须加入mask、track或多段waypoints，应该把该route另名为spatial/trajectory challenger并重新计容。

## 3. 两条constraint与三种assignment是否足够

### 3.1 一主操作 + 一稳定：基本可覆盖

在最简任务分解下：

| 任务 | constraint A | constraint B |
|---|---|---|
| 倒液 | move held source/opening toward target rim | monitor fixture或另一末端稳定target |
| 插入 | tool/connector toward opening | 另一末端稳定socket/receptacle |
| 旋盖 | RH56操作cap | OmniPicker稳定bottle body |
| 擦拭 | RH56/wipe沿surface path | OmniPicker稳定tray/workpiece |

因此“最多两条”比r1a单record有明确必要性，并能覆盖一类受限任务。

### 3.2 assignment不能替代constraint function

当前每条constraint只有：assignment、primary role/part、locator、secondary relation。它没有回答：

```text
这是造成全局effect的主操作？
这是应该保持不动/保持接触的stabilizer？
还是只观察一个passive relation？
```

`MONITOR_ONLY`只能表达无直接effector动作的被动关系，不能表达右OmniPicker主动夹住并稳定bottle/tray。`RIGHT_OMNIPICKER + BODY + N_A`在以下阶段payload相同：建立夹持、持续夹持、开始释放。global phase又同时作用于左/右两条constraint。

四任务都有异步阶段反例：

- 旋盖完成时左RH56释放cap，右Omni继续稳定bottle；
- connector插入/拔出时一侧运动，另一侧保持socket；
- wipe抬起时左侧终止surface contact，右侧仍保持tray；
- pour arm回正/释放source时，support arm可能继续稳定target。

**必须修改：** 每constraint增加最小`constraint_function`：

```text
OPERATE       # 直接实现global effect/target motion
STABILIZE     # 建立/保持反作用约束，默认不跟随global effect
MONITOR       # 无effector动作，只向action暴露已确认passive relation
```

global phase对三类功能的机械解释必须冻结，例如`RELEASE`只终止OPERATE，STABILIZE保持到下一显式event；若需要同时释放support，则下一anchor删除/切换该constraint。这样不必先增加完整per-constraint phase，但若paired-world仍出现异步建立/释放碰撞，就必须promotion为constraint-local lifecycle。

### 3.3 pipette反例否决same-effector全面禁止

典型单手移液器操作同时需要：

1. wrist/多指保持pipette body，使tip维持目标well内的位姿；
2. RH56拇指对plunger施加独立absolute finger target，执行aspirate/dispense。

当前grammar禁止`{LEFT,LEFT}`，并要求同一effector多个目标合并成一个frame/path。这个合并做不到：tip pose是腕/抓持约束，plunger actuation是手内独立通道，两者不是同一3D frame或直线路径。

若只写tip constraint，packet没有plunger target；若只写plunger part，packet丢tip/well约束；若依赖private controller profile同时生成二者，profile成为未计入的复合动作字典。

**必须修改：**

- part词表增加`ACTUATOR_OR_PLUNGER`；
- validator允许硬件profile明确批准的双`LEFT_RH56` constraints，要求不同function/part且能映射到互不冲突的wrist/hand subspaces；
- 不为OmniPicker开放通用`RIGHT+RIGHT`，因为它只有一个active aperture；除非两个constraint被证明是同一aperture的机械等价目标；
- 仍禁止任意same-effector堆叠；允许集合必须来自版本化capability manifest。

这会改变bundle grammar与capacity，当前数值必须重算。

### 3.4 part词表还缺wipe working face

擦拭的interaction part是pad/cloth/brush的工作面，不是`TOOL_TIP`，也不应长期塞进`OTHER`。`OTHER`只能表示已知但非词表部件；如果四任务核心正例稳定使用它，词表就没有完成任务ontology。

**必须增加或重命名：**

```text
WORKING_FACE_OR_PAD
ACTUATOR_OR_PLUNGER
```

`TRIGGER_BUTTON`可与后者合并。这样role/part intervention才能区分tip、plunger和wipe face。

### 3.5 两条constraint仍是可证伪上限

两条不是普适物理上限。以下样本必须保留为overflow stress：

- RH56 tip pose + plunger actuation + passive receptacle support；
- 两末端同时操作且另有必须监控的fixture relation；
- 三点稳定或多tool contact；
- bimanual handover中两个active contact加object-object target relation且无法合并。

若新增`constraint_function`后，secondary relation能在一条constraint中诚实吸收object relation，两条可能够用；若仍有第三个独立安全约束，就判r1b profile No-Go，不静默drop MONITOR或relation。

## 4. Effect、role、part、relation的充分性

### 4.1 role：足够作功能namespace，不足以单独绑定实例

四个role对这四任务基本够用。多well/多瓶场景依赖locator+task goal唯一绑定实例；这在action执行上可行，但必须有可检测的唯一性谓词。不能让planner在多个相同role/part、同locator bin实例中任意选一个。

`ROLE_INSTANCE_OVERFLOW`应在两个实例的task-relevant support重叠到同一execution tolerance时触发，而不是仅看离散tuple是否相同。

### 4.2 effect：粗动作类别够用，但不自带方向/剂量/完成判据

- `ROTATE`不区分tighten/loosen；
- `TRANSFER/FLOW`不携带目标剂量；
- `TRANSLATE`不表达wipe direction，后者由PATH提供；
- 当前词表没有`CLEAN/REMOVE_SURFACE_MATERIAL`，因此wipe的任务结果只能来自text/supervisor。

这不自动要求扩大effect。若允许text接口明确、结构化地给出signed direction、dose、named target与success criterion，当前effect可保留。否则这些goal arguments必须成为显式action context；不能从dataset/task name或controller profile隐式推断。

必须冻结一张表：

```text
goal_arg source = TEXT_EXPLICIT | PACKET_FIELD | FIXED_PROFILE_CONSTANT
```

同一公开key若仍允许两种不相容目标，compiler fail closed。

### 4.3 relation：terminal relation可保留，但不能冒充current error

`SEPARATE/NEAR/TOUCHING/INSIDE/CONTAINS/ALIGNED_TO_RIM/SUPPORTED_BY`足够表达四任务的大部分粗terminal relation，但：

- cap partially threaded与fully tightened可能都为TOUCHING；停止依赖torque/turn/supervisor；
- wipe是否clean不是几何relation；
- current insertion depth/error不是`INSIDE`一个类别；
- fixture support是否真实建立需要sensor/native证据。

所以relation只作intended target，current error必须由locator+明确allowlisted robot state机械求出。不能把粗relation当feedback替代品。

### 4.4 primary与locator语义必须消除歧义

pour存在两个合理描述：

1. primary=destination rim，locator是rim当前位置；
2. primary=source opening，locator是source opening应到达的目标位置，secondary=destination。

assignment实际驱动持有source的effector。若schema不冻结是哪一种，标注/编译和action解释会分叉。

建议统一为：

> `primary_role/part`是assignment直接控制或携带的实体/部件；locator是该primary part的**目标camera-frame pose/path**；secondary relation定义其相对目标。

于是pour写source opening→destination aligned，insert写tool tip→destination inside，wipe写tool working face→surface touching。当前primary part的位置可由proprio+verified held transform得到；目标locator来自destination/current scene。若不是这个语义，需要新增`controlled_role`与`goal_anchor_role`，不能继续混用一个primary。

## 5. 81.4779 bit：算术可接受，科学口径必须改名

### 5.1 未发现结构公式的算术错误

在memo自己冻结的结构grammar下——给定locator alphabet、part/relation词表、禁止same-effector double以及列出的assignment bundle——单/双constraint计数与全局phase/effect block拼接内部一致。本复审不重复公式。

### 5.2 不是固定部署profile的exact reachable alphabet

memo同时要求下列情况在action前拒绝：

- role–part或task ontology不匹配；
- locator kind不适用于profile；
- profile metric/tolerance不通过；
- locator resolution insufficient；
- assignment不可达、ambiguous或infeasible；
- held transform、contact、camera calibration不可用。

但当前count把所有role×part×locator×relation×assignment结构组合计入。若这些admission规则是validator的一部分，许多计入状态永远不会到达WAM。因此准确名称应是：

```text
profile-agnostic structural grammar upper bound
```

固定profile的exact reachable capacity必须对该profile的role/part/relation/locator kind/assignment allowlist重新枚举；多profile联合route则计数它们的有标签union并处理相同wire在不同profile下的语义冲突。

这与纯算术错误不同：数字可以对其结构集合精确，但不能继续被解释为所有fail-closed部署状态的exact reachable action alphabet。

### 5.3 不是总action bottleneck

r1b action仍可读：

- continuous robot proprio/state；
- allowed instruction text；
- task profile / calibration semantics；
- diffusion noise/timestep；
- 确定性safety/controller状态。

所以该数字最多描述：

> conditional finite robot-view goal packet capacity, given text, robot state, immutable profile and controller.

它不是总任务信息、总控制带宽或human–robot共享语义容量。尤其当held-object transform、contact/load或movable-camera pose加入state时，应显式列出其连续维度与source，避免把有限locator叫完整动作瓶颈。

### 5.4 profile ID是未闭合的语义选择器

同一个depth/extent code在不同profile范围下代表不同metric值。memo说profile由instruction/config公开确定但不进入action wire；这只在以下二选一时成立：

1. 一个checkpoint/route只允许一套全局metric bins和controller table；或
2. `profile_id`作为action-visible、版本化context被WAM/decoder明确读取，并将capacity声明写成“conditional on profile”。

如果同一checkpoint支持多profile，而profile只在外部controller私下决定decode/gain，profile就是未计入旁路。推荐优先使用跨四任务冻结的全局metric bin表；确实需要多profile时，显式发送profile ID并重新定义route receipt。

## 6. Allocator防旁路：已闭合部分与剩余漏洞

### 6.1 已闭合

以下设计应保留：

- target assignment写入canonical wire/token；
- human side不参与robot assignment；
- allocator对同公开输入必须确定性；
- ambiguous/infeasible只允许no-packet + hold/reobserve；
- logits、cost、candidate order、continuous target、hidden、KV、receipt不进入action；
- split gate只由assignment enum决定；
- allocator不能通过action维度重排、gain或mask传信。

这些规则解决了上一轮“actor删掉后allocator可能暗选末端”的主要问题。

### 6.2 held transform / camera pose / contact state仍未闭合

memo的current-error公式依赖：

```text
decoded locator
+ camera extrinsics
+ current 25D proprio
+ sensor-confirmed held-object/tool transform
```

但25D只有双腕、RH56六通道与Omni aperture；它没有任意held-object transform、camera/head/base pose、contact validity、load或slip。必须明确三种处理之一：

1. **进入robot state allowlist：** action expert直接读这些量；则capacity明确是conditional packet capacity；
2. **固定机械常量：** 例如已标定rigid tool mount，所有episode同一transform；保存profile hash；
3. **不可得则fail closed：** 不允许allocator/controller计算continuous error后把修正动作私下加到WAM输出。

尤其禁止以下伪闭环：WAM只读离散locator，但外部controller同时读取连续target pose/held transform并产生主要轨迹，再把性能归功于81-bit packet。该controller应作为direct visual/geometry servo baseline单报。

### 6.3 controller lookup仍可能是字典型动作旁路

`task profile + text args + phase/effect + constraints -> controller mode/tolerance/gain`只有在输出是固定、公开、小型的controller选择时可视为机械派生。若lookup包含每实例trajectory、continuous target、gain schedule或tool-specific motion template，它本身就是action channel。

必须冻结allowlist：

```text
lookup可输出：controller enum、固定安全阈值、固定profile constants
lookup不可输出：sample-specific pose/path、trajectory、continuous residual、learned hidden
```

同key多解必须拒绝，不让text embedding或allocator score暗选。

### 6.4 constraint presence与reobserve timing

constraint数、MONITOR与assignment均已计入packet；allocator可以通过它们表达有限决定。failure reason不进action也合理。但必须检查allocator是否通过不同hold时长、reobserve次数或cache保留状态向下一次action传递连续score。

当packet最终相同、proprio/text相同且没有新可观察证据时，不同allocator内部历史不得导致不同pre-safety action。

### 6.5 equal-bit control与allocator/safety存在冲突

whole-packet permutation会同时改变assignment、role、locator和relation。置换后的packet虽满足结构grammar，却可能违反原场景reachability、task profile与support truth。此时“不改变allocator/failure set”和“用置换后的assignment门控split decoder”不能同时自然成立：

- 若重新跑allocator/feasibility纠正置换packet，semantic信息被重新引入，control不再是纯双射；
- 若不纠正，置换route会因wrong effector/invalid locator产生额外安全失败，不是只改变语义编码；
- 若split gate仍用原assignment，则原assignment成为旁路。

因此当前equal-bit control不能直接作为安全控制充分性的唯一容量匹配证据。应把它限定为encoder-inductive-bias诊断，或设计保持task/profile/assignment feasibility strata的分层置换；后者的exact alphabet需另算。

## 7. 四任务复审结论

### 7.1 倒液：Conditional Go

r1b可消除同4×4 cell不同rim位置的部分碰撞，前提是：

- primary/locator语义冻结为source opening的目标pose或另一唯一规则；
- rim depth/normal/tilt axis有direct/native证据；
- source-to-hand transform在state allowlist或固定fixture中；
- 第二constraint有`STABILIZE`或`MONITOR`功能，而不是仅靠part猜support；
- dose/stop来自text+sensor/supervisor，不恢复duration/outcome。

若透明rim axis/depth无法达到coverage，或需要完整pour frame而非单axis，该profile No-Go。

### 7.2 移液/插入：当前 No-Go

对简单roll-invariant connector insertion，FRAME修订后可Conditional Go。对移液器，当前schema阻塞：

- 缺plunger part；
- 禁止双LEFT constraint；
- 单axis不保证plunger朝向/roll；
- tip insertion current error依赖held transform；
- dose与aspirate/dispense方向依赖明确goal args。

完成`ACTUATOR_OR_PLUNGER + FRAME + capability-gated LL + explicit goal args`后再跑oracle。

### 7.3 旋盖：Conditional Go after constraint-function fix

两constraint与L/R assignment符合左RH56操作cap、右Omni稳定body的硬件分工。但必须：

- 标明OPERATE vs STABILIZE；
- text/goal arg明确tighten或loosen；
- roll/axis和cap grasp feature达到公差；
- thread completion由torque/rotation supervisor判定；
- 左release时右support的异步规则机械确定。

没有constraint function时，当前版No-Go。

### 7.4 擦拭：当前 No-Go for generic task

直线、已知平面、单pass、固定width的profile可在增加working-face part并由profile固定surface normal后Conditional Go。通用wipe当前No-Go，因为PATH缺normal/width/curvature/forbidden region，多数清洁结果也不是`TRANSLATE` effect或几何relation能验证。

若最终必须传mask、多段point track或dense affordance，应停止r1b generic wipe claim。

## 8. 必须修改的最小字段与契约

### 8.1 Wire/action semantics

必须修改：

```text
constraint_function:
  OPERATE | STABILIZE | MONITOR

primary_part新增:
  ACTUATOR_OR_PLUNGER
  WORKING_FACE_OR_PAD

locator orientation扩展:
  POINT
  AXIS_ROLL_INVARIANT
  FRAME(axis + roll/second_axis)
  PATH_ON_SURFACE(tangent + normal + extent)

assignment bundles:
  保留 LEFT / RIGHT / MONITOR
  增加 capability-gated LEFT+LEFT 双constraint
  不开放任意same-effector duplicates
```

可能不必增加，但必须冻结来源：

```text
signed rotate/translate direction
dose/target quantity
named well/port
success criterion
```

这些若由structured text goal args明确给出，就不重复进packet；否则成为required action context。

### 8.2 State/profile contract

必须明确：

- camera固定还是camera/head/base pose进入state；
- held-object/tool transform是固定mount、sensor state还是required missing；
- RH56 contact/load/slip、Omni aperture/current/force哪些真实可用；
- 同一checkpoint是否只接受一套global metric bins；
- controller lookup只输出哪些有限enum/常量；
- pre-safety WAM action与post-safety/controller executed action分别报告。

### 8.3 Capacity wording

当前数字只保留为：

```text
exact size of the profile-agnostic structural r1b grammar as written
```

修改字段后重新枚举。固定task profile另报profile-valid alphabet；若profile作为action context，显式写“conditional on profile/text/state”。禁止继续写成total action bottleneck或cross-domain shared capacity。

## 9. 无训练 oracle 增补

在原`O-LOC-POUR/INSERT/WIPE`、`O-CONSTRAINT`与`O-ASSIGN`之外，必须新增：

| Oracle | matched pair | current r1b预期 | 修订通过条件 |
|---|---|---|---|
| `O-ROLL-INSERT` | anchor/axis相同，connector roll不同 | 当前碰撞 | FRAME/derived roll区分，first action满足各自key orientation |
| `O-PATH-NORMAL` | path tangent/extent相同，surface normal不同 | 当前碰撞 | PATH normal或fixed surface profile区分腕姿/法向载荷 |
| `O-FUNCTION-CAP` | 左cap operate + 右body stabilize，交换哪条执行global effect | 当前无功能标签 | 只OPERATE响应ROTATE，STABILIZE保持 |
| `O-ASYNC-RELEASE` | 左release、右maintain vs 两侧release | global RELEASE碰撞 | function/lifecycle规则产生不同右侧action |
| `O-PIPETTE-COMPOUND` | tip保持一致，plunger目标不同 | 当前无可执行编码 | LL constraints只改变RH56相关finger slice，wrist/tip保持 |
| `O-PROFILE` | bytes相同、profile ranges不同 | 当前metric语义冲突 | 单全局表或显式profile context消除冲突 |
| `O-HELD-STATE` | wrist相同、held transform不同 | current error碰撞 | allowlisted held state区分，或packet fail closed |
| `O-ALLOC-BYPASS` | packet/assignment/state/text相同，改cost/logits/history | action应相同 | pre-safety action严格不变 |

判据仍是native/sensor-confirmed reference controller的可推进安全动作集合，不用learned policy、不用注释F1、不用容量大小代替结果。

## 10. 最终 Go / Conditional / No-Go

### Go

- 保留active-only、删除future/timing/human actor/outcome的方向；
- target robot locator与assignment不参与human跨域pair cost；
- assignment enum进入wire/token；
- allocator logits/cost/continuous target/hidden不进action；
- `81.4779`作为**当前结构集合**大小的内部算术结果，未发现公式错误。

### Conditional Go

- 64×64×32用于有native/marker/stereo truth且量化误差通过的target-robot profile；
- 单axis用于roll-invariant插入/approach；
- 直线PATH用于已知平面、固定normal/width的单pass wipe；
- 两constraint用于一个OPERATE加一个STABILIZE/MONITOR；
- phase/effect/relation在structured text goal args和sensor supervisor补齐方向、剂量、完成判据时使用。

### No-Go until fixed

- 当前schema freeze、模型实现、训练、真机与“四任务通用充分”主张；
- 当前移液器compound control与generic wiping；
- 用human Ego生成64×64×32/axis/extent GT；
- 把单axis用于roll-sensitive connector/tool；
- 继续禁止所有same-effector double；
- 没有constraint function却声称双约束语义闭合；
- profile/held transform/contact/camera state私下进入controller；
- 把81.4779称为固定部署profile exact reachable、总action bottleneck或cross-domain shared capacity；
- 用whole-packet equal-bit permutation的额外physical infeasibility证明semantic necessity。

## 11. 最终一句话

`minimal-r1b`已经从“过度删除”走到了一个值得做oracle的target-robot finite geometry候选，但尚未成为四任务最小充分状态：64×64×32只解决位置量化的一部分，单axis/PATH没有完整局部frame；两条constraint只有末端身份，没有OPERATE/STABILIZE功能；禁止双LEFT会直接拒绝“保持pipette tip + 按plunger”；profile、held transform与controller lookup仍可能绕过有限packet。修完这些字段和边界后是Conditional Go；当前版本应No-Go for freeze，而81.4779 bit只保留为原结构grammar的精确大小，不作为部署reachable或总信息瓶颈主张。
