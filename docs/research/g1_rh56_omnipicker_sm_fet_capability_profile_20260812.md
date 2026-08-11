# G1 + 左 RH56DFTP + 右 OmniPicker：SM-FET 有限语义 capability profile

- 日期：2026-08-12
- 性质：event constraint / capability registry 语义设计；不实现、不训练、不使用 GPU、不做驱动、标定、部署或真机操作
- canonical 代码基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 目标硬件（用户冻结范围）：Unitree G1 + 左 RH56DFTP + 右 OmniPicker
- 目标 action contract：`sonic_g1_rh56dftp6_omnipicker1_v1`
- 被审 event route：`sm_fet_hard_grammar_v2_minimal_r1b`
- 本文输出：registry 的有限语义、准入规则与 hash 语义；**不是一份带伪标定值的可部署 profile**

## 0. 裁决

需要新增一个与 hardware manifest、calibration capture 分离的纯语义 registry：

```text
sm_fet_event_capability_registry/v1
```

它只回答五类有限问题：

1. 哪些 target-robot effector ID 存在；
2. 每条 event constraint 是 `OPERATE`、`STABILIZE` 还是 `MONITOR`；
3. 哪些 task part、effector working surface 与 locator family 可以形成公开、有限的 binding；
4. 两条 constraint 是否竞争同一腕、手指、aperture、held object 或全身安全资源；
5. 在固定 registry 下，哪些 wire tuple 属于静态 grammar，哪些必须 fail closed。

该 registry **不能**保存 serial、endpoint、raw 极性、腕 frame、工作区、力/位移阈值、相机参数、任务实例 pose、轨迹、ROI、gain、per-bin residual 或 held-tool transform 数值。那些分别属于 hardware manifest、calibration、当前 verified robot state 或 controller/safety contract。

当前可冻结的是“型号拓扑和语义边界”，不是四任务实机能力：

- RH56DFTP 的五指、6 个主动通道以及指尖/指腹/掌心触觉面属于型号级 `HARDWARE_KNOWN`；这不证明该台左手的驱动、触觉、抓持或移液器拇指操作已可用。
- OmniPicker 的双夹爪、1 个主动 aperture 以及简单开合抓取属于型号级 `HARDWARE_KNOWN`；这不证明当前 G1 已有对应驱动、力/掉落状态或可靠稳定瓶体能力。
- 左右腕的 25D 软件 slice 是 `REPO_CONTRACT_KNOWN`；目标实机腕 frame、极性、时序、可达域和安全 envelope 仍为 `NEEDS_CALIBRATION`。
- 倒液、插入/移液、旋盖、擦拭的所有具体 allocation 都只能作为有限候选规则；在 task oracle 与 calibration receipt 前不得进入部署可达 grammar。

因此当前状态为：**Go for semantic registry memo；No-Go for executable profile instance、exact deployed reachability、训练或部署。**

## 1. 证据层级与“已知”的严格含义

### 1.1 状态枚举

registry 中每条 capability/binding 必须使用下列四态之一；不能用一个模糊 `supported=true`：

| 状态 | 含义 | 能否进入固定部署 profile 的 static grammar |
|---|---|---:|
| `HARDWARE_KNOWN` | 厂商资料或冻结硬件拓扑确认离散结构存在；不等于任务能力已通过 | 只能作为拓扑前提，不能单独放行任务 tuple |
| `NEEDS_CALIBRATION` | 物理上有候选通道，但需要目标实机 frame、端点、接触、可达性或 paired-world oracle receipt | 否 |
| `UNSUPPORTED_BY_TOPOLOGY` | 与离散资源拓扑直接矛盾，例如一个 aperture 同时接受两个独立目标 | 否，且不能靠标定升级 |
| `UNKNOWN_NOT_ADMITTED` | 官方资料/现有代码不能确认，或未做任务级验证 | 否；unknown 不是 `OTHER`、`MONITOR` 或默认 true |

`NEEDS_CALIBRATION` 只有在外部 receipt 绑定 exact hardware manifest、calibration、task oracle 与 registry hash 后才可被新 revision 升级；运行时不得原地改状态。

### 1.2 当前可确认事实

| 事实 | 状态 | 边界 |
|---|---|---|
| 目标组合是 G1 + 左 RH56DFTP + 右 OmniPicker | 用户冻结 scope | 不等同于本文看到设备 serial 或安装验收 |
| RH56DFTP 为 12 关节、6 主动自由度，六路语义为 little/ring/middle/index/thumb-bend/thumb-rotate | `HARDWARE_KNOWN` | 厂商 raw 与 repo canonical open fraction不同，必须另标定 |
| RH56DFTP 有压力/触觉传感器，触觉面分布于指尖、指腹、掌心 | `HARDWARE_KNOWN` | 只证明型号配置；当前 G1 数据路径和可靠 contact/load state 未确认 |
| OmniPicker 为双夹爪、1 个主动自由度的自适应通用夹爪 | `HARDWARE_KNOWN` | 只证明 coupled aperture 拓扑，不证明任意对象抓稳 |
| AimDK 文档把 OmniPicker `position=1.0` 定义为全开，并公开 position/effort/state/fault 语义 | 厂商 X2 接口参考 | 不证明当前 G1 endpoint、raw 极性或这些状态已接入 |
| repo 的 asymmetric action layout 为 left wrist 9 + right wrist 9 + RH56 6 + Omni aperture 1 | `REPO_CONTRACT_KNOWN` | 是软件 schema，不是真机 capture |
| 88 段 `grab_red_bottle` 是双 Inspire6，不是目标硬件 | repo 审计已确认 | 永远不能给本 profile 的 task/calibration receipt |
| RH56/Omni serial、firmware、driver、腕 frame、端点、安全限幅、时序、接触有效性 | `UNKNOWN_NOT_ADMITTED` / `NEEDS_CALIBRATION` | hardware manifest 模板当前为空且 fail closed |

官方来源：

- [因时机器人 RH56DFTP 官方用户手册](https://www.inspire-robots.com/d/file/p/2025/01-16/%E5%9B%A0%E6%97%B6%E6%9C%BA%E5%99%A8%E4%BA%BA%E7%81%B5%E5%B7%A7%E6%89%8B--%E8%A7%A6%E8%A7%89%E6%89%8B%E7%94%A8%E6%88%B7%E6%89%8CV1.0.0cn.pdf)：6 个微型执行器、压力与触觉传感器、12 关节/6 自由度以及六路位置接口。
- [智元 OmniPicker 官方手册](https://x2-aimdk.agibot.com/zh-cn/v0.8.1/_downloads/9179a383c97b0f37b1f0fbf51ed2cb35/omni_picker_introduction_for_x2.pdf)：1 个主动自由度、自适应通用夹爪与主动力控的型号级描述。
- [AimDK 官方末端执行器接口](https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/endeffector.html)：OmniPicker 单 command/state entry、开合 position 语义和状态字段；这是 X2/AimDK 参考，不是 G1 集成证据。

本地代码/审计来源：

- `world_action_model/asymmetric_action_schema.py:18-39,56-67,92-128`：25D split 与六路/一路语义；
- `configs/hardware/asymmetric25_hardware_manifest.template.json:14-109`：所有实机接口、frame、timing、safety 与 calibration 值仍为空；
- `docs/audits/grab_red_bottle_asymmetric25_audit.md:1-73`：旧 88 集 No-Go 和目标硬件补采缺口。

## 2. 三层 contract 必须分开

```text
Semantic capability registry（本文）
  finite IDs / functions / parts / surfaces / resource conflicts / static predicates

Hardware + calibration binding（已有 manifest/capture contract，当前为空）
  exact device identity / channel mappings / frames / timing / limits / safety

Per-anchor verified robot state（本文只定义引用边界）
  proprio/contact validity / held-tool transform / fixture support / fresh receipts
```

任何一层缺失都不允许用另两层补造：

- registry 不保存当前场景 geometry；
- manifest 不宣称某个 task tuple 已可执行；
- current state 不得临时改 registry 或选择另一个 profile；
- allocator 不得输出 continuous correction、gain、trajectory 或 held transform。

## 3. Event constraint 的最小修订

### 3.1 `constraint_function` 必须显式 action-visible

`minimal-r1b` 当前每条 constraint 有 assignment、role/part、locator、secondary relation，但没有功能。仅靠全局 phase/effect 无法区分“左手旋 cap、右手稳定 bottle”。因此每条 present constraint 必须新增：

```text
constraint_function = OPERATE | STABILIZE | MONITOR
```

语义冻结为：

| function | 唯一含义 | 是否响应全局 effect | 是否打开动作 slice |
|---|---|---:|---:|
| `OPERATE` | 直接实现当前 global effect 或目标部件运动 | 是 | 必须是物理 effector |
| `STABILIZE` | 建立/保持反作用、姿态或支撑约束 | 默认否 | 必须是物理 effector |
| `MONITOR` | 只暴露已由当前 sensor/native evidence 确认的被动关系 | 否 | 否，只能配 `MONITOR_ONLY` |

强制规则：

1. `MONITOR` 不等于 epistemic unknown，也不等于继续用旧 packet hold；证据未知时 no-packet + `HOLD/REOBSERVE`。
2. `STABILIZE` 不得伪装为 `MONITOR`。右 OmniPicker 正夹持瓶体时是 active stabilization，即使 aperture 变化很小。
3. 非 `VERIFY` 事件至少有一条 `OPERATE`；只有稳定而没有主操作的独立阶段必须用冻结 lifecycle 规则或另立 event，不可暗用 global effect。
4. `RELEASE` 默认只释放 `OPERATE` constraint；`STABILIZE` 是否保持或删除由下一 anchor 的显式 constraint set 决定。若同一公开输入仍有两种 lifecycle，需 promotion，不能藏进 profile。
5. function 是 constraint wire/token 的字段；registry 只判断组合是否合法。若把 function 藏在 profile lookup，原编译碰撞仍然存在。

### 3.2 Effector ID

冻结三种 assignment ID；不引入 human side：

| effector/assignment ID | action path | 可用 function | 当前状态 |
|---|---|---|---|
| `LEFT_RH56` | left wrist 9D + RH56 6D | `OPERATE`, `STABILIZE` | 拓扑 `HARDWARE_KNOWN`；task binding均待标定 |
| `RIGHT_OMNIPICKER` | right wrist 9D + aperture 1D | `OPERATE`, `STABILIZE` | 拓扑 `HARDWARE_KNOWN`；task binding均待标定 |
| `MONITOR_ONLY` | 不打开任何动作 slice | `MONITOR` only | 语法 ID；每 anchor 仍需 verified evidence |

`MONITOR_ONLY` 是 assignment sentinel，不是假想第三只末端。RH56/Omni 的传感器可以为 supervisor 提供 evidence，但“读取传感器”本身不生成一个物理 effector assignment。

## 4. Finite part 与 working-surface registry

### 4.1 Task part vocabulary

四任务 executable profile 不应继续让核心部件落入 `OTHER`。建议冻结：

```text
N_A                         # VERIFY only
BODY
HANDLE_OR_GRASP_FEATURE
LID_CAP
OPENING_OR_RIM
TOOL_TIP
ACTUATOR_OR_PLUNGER
WORKING_FACE_OR_PAD
```

规则：

- `ACTUATOR_OR_PLUNGER` 与 `WORKING_FACE_OR_PAD` 是 control review 已证明缺失的核心正例，必须成为 known part。
- `OTHER` 可留在通用 annotation ontology，但必须从 `biochem4` executable profile 的 static allowlist 中删除；它不能吸收 unknown 或稳定出现的核心部件。
- `primary_part` 表示 assignment 直接控制或携带的 operation frame，不必等于手实际接触的位置。例如倒液可控制 source `OPENING_OR_RIM` 的目标 frame，而 RH56 实际通过 source handle/body 建立 held binding。
- secondary 目标面没有独立 part 字段时，locator family 必须唯一说明目标是 rim、port 或 surface path；若仍多解，应增加显式 goal-part binding，而不是让 registry 猜。

### 4.2 Effector working surfaces

registry 只列有限 surface class，不写 mesh、pose、压力阈值或接触点：

| surface ID | owner | 型号级事实 | task 使用状态 |
|---|---|---|---|
| `RH56_FINGERTIP_SET` | `LEFT_RH56` | 手册确认指尖触觉面 | `NEEDS_CALIBRATION` |
| `RH56_FINGERPAD_SET` | `LEFT_RH56` | 手册确认指腹触觉面 | `NEEDS_CALIBRATION` |
| `RH56_PALM_SURFACE` | `LEFT_RH56` | 手册确认掌心触觉面 | `NEEDS_CALIBRATION` |
| `RH56_THUMB_ACTUATION_CONTACT` | `LEFT_RH56` | thumb bend/rotate 通道存在 | plunger 操作 `NEEDS_CALIBRATION` |
| `OMNIPICKER_COUPLED_JAW_PAIR` | `RIGHT_OMNIPICKER` | 双夹爪、单主动 aperture | `NEEDS_CALIBRATION` |
| `HELD_TOOL_TIP` | 外部 held tool | 不是裸 effector surface | 需要 verified held binding |
| `HELD_TOOL_ACTUATOR` | 外部 held tool | 不是裸 effector surface | 需要 verified held binding |
| `HELD_TOOL_WORKING_FACE` | 外部 held tool | 不是裸 effector surface | 需要 verified held binding |

“型号有触觉面”不等于“当前 action/state 中已有可用 tactile”。在目标实机传感接口和 calibration receipt 前，surface class 只参与候选 binding，不得把 contact 写成 known。

同一 `(effector, function, primary_part, held_binding_kind)` 若允许多个会导致不同动作的 working surface，必须二选一：

1. registry 将其冻结成唯一 surface；或
2. 把 `working_surface_id` promotion 为 action-visible finite field。

禁止 allocator 用 hidden score逐样本选择指尖、指腹或掌心。

## 5. Resource 与互斥规则

### 5.1 有限资源 ID

```text
G1_SHARED_WBC_BALANCE
G1_SHARED_SELF_COLLISION

LEFT_WRIST_POSE
LEFT_RH56_LITTLE
LEFT_RH56_RING
LEFT_RH56_MIDDLE
LEFT_RH56_INDEX
LEFT_RH56_THUMB_BEND
LEFT_RH56_THUMB_ROTATE

RIGHT_WRIST_POSE
RIGHT_OMNIPICKER_APERTURE

HELD_OBJECT_BINDING(<effector>)
MONITOR_EVIDENCE(<relation>)
```

这里的 ID 是冲突图节点，不是驱动名称。`G1_SHARED_*` 的 predicate 需要目标实机可达/安全验证；本文不填任何数值 envelope。

### 5.2 静态冲突规则

1. 每个 present 物理 constraint必须声明一个有限 `resource_claim_set`；未声明即 profile invalid。
2. 两条 constraint 若同时独占同一 wrist pose resource，则冲突，除非二者是同一个公开 frame/path 的可证明合并；合并后只能留下一个 constraint。
3. `RIGHT_OMNIPICKER_APERTURE` 是单一 coupled resource。两个独立 aperture target 为 `UNSUPPORTED_BY_TOPOLOGY`；相同目标应 canonical deduplicate，不形成 `RIGHT+RIGHT`。
4. RH56 六个主动通道是可区分资源，但“通道不同”不自动证明可并发完成两项任务；抓持通常同时占用多个 finger resource。
5. 任一左右并发 bundle还必须通过 `G1_SHARED_WBC_BALANCE` 与 `G1_SHARED_SELF_COLLISION`；没有 verified predicate 时 no-packet，不可只按手部资源判可行。
6. 同一 object instance/part 不得同时收到互不相容的 `OPERATE` 与 `STABILIZE` 目标。cap 可被左侧 OPERATE，bottle body 可被右侧 STABILIZE；不能让同一 cap 同时“转动且固定”。
7. `MONITOR` 不占动作资源，但必须占一个 fresh `MONITOR_EVIDENCE` receipt；两个相同 monitor payload 是 duplicate。
8. event grammar 仍最多两条独立 constraint；第三条为 `CONSTRAINT_OVERFLOW`，不得静默丢弃 MONITOR 或 support。

### 5.3 Same-effector double

#### `LEFT_RH56 + LEFT_RH56`

不是通用允许，也不能通用禁止。唯一候选是 profile 明列的 compound bundle，例如移液 dispense 阶段：

```text
constraint A:
  function = STABILIZE
  controlled_part = TOOL_TIP
  claims = LEFT_WRIST_POSE + calibrated grasp-channel set

constraint B:
  function = OPERATE
  controlled_part = ACTUATOR_OR_PLUNGER
  claims = calibrated thumb actuation subset
```

只有同时满足以下条件才可在未来 registry revision 中 admitted：

- exact part/function pair 在 allowlist 中，不允许任意 `LEFT+LEFT`；
- 两条 resource claim经实机标定为可同时满足，不竞争同一独占目标；
- held pipette binding与 tip/plunger frame均为 fresh verified state；
- `O-PIPETTE-COMPOUND` 证明改变 plunger目标只改变允许的RH56手内 slice，tip/wrist保持在公差内；
- G1共享安全 predicate通过。

当前裁决：`NEEDS_CALIBRATION`，**不进入当前 executable grammar**。

#### `RIGHT_OMNIPICKER + RIGHT_OMNIPICKER`

两个独立右侧 constraint都会竞争单一 aperture，通常还竞争同一 wrist pose。因此：

```text
independent RIGHT + RIGHT = UNSUPPORTED_BY_TOPOLOGY
identical/equivalent goals = merge or duplicate rejection, not two constraints
```

不能因为两个 jaw 是两个物理接触面，就把它们当成两个可独立控制的执行器。

#### `MONITOR_ONLY + MONITOR_ONLY`

只在 `VERIFY` 或另一个明确允许纯检查的 event 中候选合法；payload必须不同且各自有 fresh evidence。它不解决三条物理约束 overflow。

## 6. Held-tool transform：外部 verified state，不是 action channel

registry 只允许声明：

```text
requires_external_state = VERIFIED_HELD_TOOL_BINDING
```

外部 state schema 至少需要概念上绑定：

```text
effector_id
tool_instance_id
grasp_contract_id
controlled_part_id
T_effector_to_controlled_part
contact_validity
source / timestamp / uncertainty support
hardware_manifest_hash / calibration_hash / receipt_hash
```

本文不填任何 transform、阈值或 source 值。规范边界如下：

1. transform 不进入 event wire、constraint token、profile table、text 或 allocator output；
2. 它不能包含 target-scene pose、target-relative residual、well/rim/path geometry；
3. 它只能来自固定已标定 mount，或当前 sensor-confirmed held state；二者都必须有 receipt；
4. 每个 replan anchor按其 validity policy重新确认；last-known、planner hidden或“aperture较小”不能证明仍持有；
5. missing/expired/multi-hypothesis 时 required constraint不编译；
6. 若 controller/action expert实际读取该连续 state，实验必须把它列为所有 route共享的 external robot-state condition。有限 packet capacity只是在给定该 state时的条件容量，不能称为总 action bottleneck；
7. 禁止外部 controller同时读取 continuous target pose再替 r1b完成主要视觉伺服，却把收益归给 finite packet。

因此“非 action channel”准确指：它不是 planner/allocator可写入的逐样本动作消息，也不是 profile 选择器；它是公开、allowlisted、可替换审计的外部 verified robot state。

## 7. 四任务的有限 binding 表

表中 `CAL` 表示 `NEEDS_CALIBRATION`，不是已支持。

| task/stage | OPERATE constraint | STABILIZE constraint | MONITOR 候选 | same-effector | 当前裁决 |
|---|---|---|---|---|---|
| 倒液 | `LEFT_RH56` 携带 source，控制 `OPENING_OR_RIM` 的 frame/tilt；实际 grasp在 handle/body held binding | `RIGHT_OMNIPICKER` 以 jaw pair稳定 target `BODY` | target已由fixture锁定且有native evidence时可用 | 不需要 | 双侧任务 binding均 `CAL`；透明 rim/depth/normal未知则 no-packet |
| 简单插入 | `LEFT_RH56` 携带 connector/pipette，控制 `TOOL_TIP` 进入 opening | `RIGHT_OMNIPICKER` 稳定 socket/receptacle `BODY`，若其并非固定 | fixture support有verified evidence时可替代 active support | 不需要 | roll-invariant可候选；roll-sensitive需要 `FRAME`，当前 r1b单axis不足 |
| 移液 dispense | `LEFT_RH56` 操作 `ACTUATOR_OR_PLUNGER` | 同一 `LEFT_RH56` 保持 `TOOL_TIP`/pipette pose；必要时右侧另稳 receptacle会产生第三约束压力 | 只能监测已知fixture relation | 候选 `LEFT+LEFT` | 当前 No-Go；compound与两约束上限均未通过 |
| 旋盖 | `LEFT_RH56` 通过 fingertip/fingerpad grasp 操作 `LID_CAP` | `RIGHT_OMNIPICKER` 以 jaw pair稳定 bottle `BODY` | bottle已固定且fresh verified时可用 | 不需要 | function fix后为 `CAL`；tighten/loosen由显式goal args给出 |
| 限定直线擦拭 | `LEFT_RH56` 持 wipe handle，控制 `WORKING_FACE_OR_PAD` 沿 surface path | `RIGHT_OMNIPICKER` 稳定 tray/workpiece `BODY` | workpiece固定且有fresh evidence时可用 | 不需要 | fixed plane/normal/width才可 `CAL` |
| 通用擦拭 | 同上，但需要曲线、宽度、禁止区或动态normal | 同上 | 同上 | 不解决locator缺口 | 当前 finite r1b `UNSUPPORTED/UNKNOWN`；应让continuous/direct-servo challenger获胜 |

补充边界：

- 右 OmniPicker 用于“简单抓放”是官方拓扑支持的 task family 候选，但具体对象/姿态仍需 `CAL`。不能据此自动允许其倒液、旋cap或持wipe。
- 左 RH56 的多指/拇指拓扑使 cap/plunger成为合理候选，不构成安全抓持、力控或成功率证据。
- 任何 reversed allocation（例如右侧旋cap、左侧稳瓶）若没有预注册 task oracle，应为 `UNKNOWN_NOT_ADMITTED`，而不是 allocator 的备用选择。
- dose、flow stop、thread torque、clean completion属于显式 text goal args + sensor supervisor，不进入 capability registry，也不能从 task name推断。

## 8. Registry 的规范对象

本文不生成实例 JSON，但冻结未来对象必须包含的有限字段：

```text
registry_schema_id
semantic_profile_id
revision
target_hardware_class

effector_ids[]
constraint_functions[]
task_parts[]
working_surface_ids[]
resource_ids[]

effector_function_allowlist[]
part_surface_binding_allowlist[]
resource_claim_templates[]
same_effector_bundle_allowlist[]
mutex_rules[]
task_family_rules[]
locator_family_requirements[]
external_state_requirements[]

status_per_entry:
  HARDWARE_KNOWN | NEEDS_CALIBRATION |
  UNSUPPORTED_BY_TOPOLOGY | UNKNOWN_NOT_ADMITTED

source_contract_refs[]
canonicalization_rule
```

禁止字段：

```text
serial / firmware / endpoint
raw endpoints / gains / forces / workspace numbers
camera calibration / wrist frames
object instance pose / ROI / map / spline / trajectory
per-scene affordance or learned residual
held-tool transform value
allocator score / candidate order / fallback policy
```

## 9. Profile ID 与 hash 语义

### 9.1 Logical ID

建议的稳定逻辑 ID 是：

```text
sm_fet_capability/g1-l_rh56dftp-r_omnipicker/biochem4/v1
```

它表示语义 family，不表示当前已有可部署实例。任何 part、function、surface、same-effector 或 mutex 语义变化都必须升 revision；不能同 ID 改解释。

### 9.2 Content hash

authoritative identity 是内容 hash，不是文件名：

```text
semantic_profile_hash = SHA256(
  RFC8785_canonical_JSON(semantic_registry_object)
)
```

hash 输入必须包含所有有限 allowlist、状态、冲突规则、外部 state requirement、source contract ref 与 canonicalization version；不包含注释、生成时间、文件路径或 per-anchor state 值。

未来运行时 binding 另建 envelope：

```text
semantic_profile_id
semantic_profile_hash
hardware_manifest_hash
calibration_capture_hash
camera/metric_profile_hash
controller_lookup_hash
```

本文没有真实 hardware/calibration binding，因此**不生成一个假的 64-hex profile hash**。

同一 logical ID 对应不同 bytes/hash必须 fail closed。registry 在 data split前冻结、test期间只读。planner/allocator不得逐样本选择 profile。

### 9.3 多 profile 不是免费条件

- 一个 checkpoint/route只接受一个冻结 profile时，packet capacity可表述为“conditional on profile”。
- 同一 checkpoint接收多个 runtime profile时，profile ID就是额外 action context；必须显式给 action path、按profile分层报告，或把 tagged union计入总 grammar。
- profile ID若编码指定well、rim坐标、path、tool pose或per-scene controller，就是隐藏 action channel，直接失去 finite headline。

## 10. 对 grammar 与 exact reachability 的影响

### 10.1 原 r1b 数字只能保留为什么

原 `minimal-r1b` 的：

```text
3,367,384,122,335,042,437,906,433 states
81.47790257814123 bit
```

是原字段、原 part词表、无 `constraint_function`、禁止所有same-effector double时的 **profile-agnostic structural grammar size**。control review已确认算术本身未见错误，但它不是固定 G1/RH56/Omni task profile 的 exact reachable alphabet。

本文至少引入三项会改变结构 grammar 的语义：

1. 每constraint新增三值 `constraint_function`，且 function/assignment bundle受约束；
2. part新增 `ACTUATOR_OR_PLUNGER` 与 `WORKING_FACE_OR_PAD`；
3. future profile可能只允许一类经验证的 `LEFT+LEFT` compound，而继续拒绝 `RIGHT+RIGHT`。

因此不能继续复用原状态数，也不能简单乘 `3^constraints`：function、phase、assignment、part、same-effector 和 task family不是独立笛卡尔积。必须在修订 wire/canonical order后重新做 constrained rank/unrank。

### 10.2 三种集合必须分别命名

```text
G_struct
  修订字段的全部canonical结构 grammar；只检查字节/枚举/排序。

G_profile(profile_hash)
  对 G_struct 应用固定 semantic registry 的有限静态谓词：
  function/effector/part/surface/resource/same-effector/task-family allowlists。

R_runtime(profile_hash, manifest_hash, calibration_hash, state_receipt)
  再应用当前可达性、接触、held binding、camera/locator公差和安全 predicate。
```

只有 `G_profile` 在 registry 实例完整且所有 admitted entry已升级后，才有可复现的 profile-valid exact finite count。`R_runtime` 随当前连续robot state变化，不应伪装成一个跨episode常数 alphabet。

当前没有可部署 semantic registry instance，也没有 hardware/calibration/state receipts，所以本 memo **不报告 `|G_profile|` 或 runtime reachable 数值**。

### 10.3 Status 如何进入计数

- `HARDWARE_KNOWN` 只允许建立拓扑节点，不自动把 task tuple计入 `G_profile`；
- `NEEDS_CALIBRATION`、`UNKNOWN_NOT_ADMITTED`、`UNSUPPORTED_BY_TOPOLOGY` 一律从 `G_profile` 排除；
- calibration不能修改原对象；它产生新 revision/hash，将经验证 entry显式升级；
- no-packet/HOLD/REOBSERVE 是 control-plane结果，不是把 unknown编码成新的 action state。

## 11. 确定性 allocator / validator 规则

```text
input:
  revised functional constraints with explicit function
  discrete target-robot locator
  fixed semantic_profile_hash
  verified manifest/calibration refs
  allowlisted current robot-state validity

static validation:
  role/part/function/locator family admitted?
  effector/function/working-surface binding admitted?
  resource claims complete?
  same-effector bundle explicitly admitted?
  mutex / duplicate / overflow clear?

runtime validation:
  unique role instance and locator support?
  held binding fresh where required?
  reachability / WBC / collision / contact predicate verified?

output visible to action:
  canonical finite constraint fields only

output invisible:
  reason / score / receipt / hash
```

若两个 effector assignment仍都合法且会产生不同动作，allocator必须 `AMBIGUOUS -> no packet`，不能用成本、历史成功率或 candidate order暗选。reversed allocation只有在 registry 静态允许且当前唯一可行时才可能输出；当前 `biochem4` 候选表未提供这种许可。

## 12. 必须通过的无训练审计

1. **Function swap：** 固定其它字段，交换 OPERATE/STABILIZE，只允许主操作侧响应global effect；MONITOR永不打开动作slice。
2. **Same-effector：** 未在 exact bundle allowlist的 `LEFT+LEFT`稳定拒绝；任意 `RIGHT+RIGHT`稳定拒绝或先合并为单constraint。
3. **Resource conflict：** 只改 resource claim使两个constraint竞争同一 wrist/aperture时，结果从admit稳定变为mutex；allocator score不能改变结果。
4. **Held-state swap：** fixed packet/profile、只替换verified held transform时，若action使用该state必须按external-state contract变化并对所有baseline一致；缺失时必为no-packet，不能用旧transform。
5. **Profile swap：** fixed packet下未选 registry条目变化不影响action；同profile ID不同hash必拒绝。
6. **Task oracle：** 保留 control review 的 `O-LOC-POUR/INSERT/WIPE`、`O-CONSTRAINT`、`O-ROLL-INSERT`、`O-PATH-NORMAL`、`O-FUNCTION-CAP`、`O-ASYNC-RELEASE`、`O-PIPETTE-COMPOUND`、`O-HELD-STATE`、`O-ALLOC-BYPASS`。
7. **Unknown：** 任一 required part/surface/contact/held binding/reachability为unknown，清空旧packet/cache并HOLD/REOBSERVE；不得降级成`OTHER`、`MONITOR`或最近bin。

这些 gate只验证 contract；本文不实现、运行或声称通过。

## 13. 四任务最终 Go / No-Go

| 范围 | 当前裁决 |
|---|---|
| 型号拓扑 registry：三种 assignment、三种 function、task part、surface class、resource ID | **Go for semantic freeze** |
| `constraint_function` 留在 profile lookup而不进wire | **No-Go** |
| 任意 `LEFT+LEFT` | **No-Go** |
| 精确列出的pipette `STABILIZE tip + OPERATE plunger` | **Needs calibration/oracle；当前不admit** |
| 任意独立 `RIGHT+RIGHT` | **Unsupported by one-aperture topology** |
| 倒液、简单插入、旋盖、限定直线擦拭 | **Conditional candidate；当前均未过实机profile gate** |
| 移液compound、roll-sensitive insertion、generic wiping | **当前 No-Go** |
| 把 held transform 写入profile、text、allocator或event packet | **No-Go** |
| 将 held transform作为公开external verified state并条件化capacity | **契约上允许，仍待真实source/calibration** |
| 继续把81.4779 bit称为固定部署exact reachable或总action bottleneck | **No-Go** |

## 14. Idea-discussion 登记

本轮按要求复用 control review 的同一 idea card，以 critic 模式登记：

```text
idea_id        = ca118643-f979-44e8-9526-50957b0ca236
discussion_id  = 5eabd03f-c496-4d06-b538-762b92bae3df
memory_updated = true
zotero_synced  = 100 records
```

自动 scout 返回 Rubik's Cube manipulation benchmark、ARMBench，以及明显误召回的 image-manipulation detection benchmark；Zotero近邻仍是通用 RoboBrain note。它们没有给出本题的 function/part/resource/same-effector/hash 字段级证据，因此不参与 registry 裁决，只保留为 idea memory。

## 15. 版本化只读来源

- canonical code：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`；
- asymmetric25 hardware contract与审计：commit `6d7deeac076f96f4ad72f2ab3b2180e3a5516926`；
- `minimal-r1b` schema/capacity：commit `e31208172d194511e7fc95acd6075635c249c335`；
- `minimal-r1b` control review：commit `144d050ddfcb3d4a3fa0dc5f299754177c63f04b`；
- `minimal-r1b` cross-domain/profile review：commit `49667be32d15bd95fd2bde610b3e251c2fcc8474`；
- 生化任务 portfolio：commit `3c545aa31469272be68a9c94ca51fa7c17aa3331`。

## 16. 最终一句话

G1 + 左 RH56DFTP + 右 OmniPicker 的 capability profile 应是一个**冻结、有限、只做准入的语义冲突图**：`OPERATE/STABILIZE/MONITOR`显式进入每条constraint，registry枚举effector、part、working surface、资源与唯一可允许的compound；RH56的六通道只提供候选并发拓扑，不能免标定放行pipette双LEFT，OmniPicker的单aperture则直接否决独立双RIGHT。held-tool transform只作为公开external verified robot state，缺失即no-packet；profile ID/hash不能携带场景几何。修订后的profile-valid grammar必须在真实registry revision完成后重新exact enumerate，原81.4779 bit只保留为旧r1b结构集合大小。
