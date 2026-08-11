# SM-FET `minimal-r1c`：共享功能语义与目标机器人有限局部 frame 的最终候选

- 日期：2026-08-12
- 性质：综合 r1a/r1b、控制、跨域、局部 frame 与 G1 asymmetric capability 审查的最终 schema candidate；不实现、不训练、不使用 GPU、不修改 canonical
- canonical 代码基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- schema ID：`sm_fet_hard_grammar_v2_minimal_r1c_struct_v1`
- 目标硬件 family：Unitree G1 + left RH56DFTP + right OmniPicker

## 0. 最终裁决

推荐把 `minimal-r1c` 冻结为**结构与信息边界的最终候选**，取代 r1a/r1b 作为下一步 CPU contract/oracle 的唯一主候选：

```text
active-only:
  compact phase + known intended current effect

1..2 concurrent constraints:
  constraint_function = OPERATE | STABILIZE | MONITOR
  primary role/part                      # primary is focus
  robot-only finite execution frame/path
  optional secondary role + intended target relation
  explicit target-robot assignment
```

它继续删除 future、onset、duration、human actor、outcome、historical relation_before 和自由 contact mode；修复 r1b 的五个已证缺口：

1. 每条 constraint 显式区分 `OPERATE/STABILIZE/MONITOR`；
2. part新增 `ACTUATOR_OR_PLUNGER`、`WORKING_FACE_OR_PAD`，executable grammar不再用`OTHER`吸收核心部件；
3. locator从单axis升级为 `64×64×32` translation、`512 axis×32 roll` full frame，以及 `LINE/ARC_LEFT/ARC_RIGHT/RASTER_3 × 8 extent` path；
4. 最多两条 constraint，允许**profile精确放行**的 LEFT `OPERATE+STABILIZE` same-effector compound；独立 RIGHT+RIGHT 因单aperture拓扑永久拒绝；
5. target assignment仍显式进入packet；capability profile与held transform只作verified external receipts，不得成为未计容动作旁路。

全局、未按task/capability裁剪的 canonical structural alphabet 可精确计算：

```text
canonical field wire = 32 bytes
action token slots   = 5
|G_struct|           = 126,149,773,623,130,877,098,697,687,041
log2 |G_struct|      = 96.67105236912121 bit
minimum index width  = 97 bit
compact container    = 13 bytes
```

这个数字只叫 **global structural grammar exact size**，不是 G1 biochem4 可部署 profile 的 exact reachable states。当前 capability memo 明确所有具体倒液、插入、移液、旋盖、擦拭 binding仍为 `NEEDS_CALIBRATION`，没有 authoritative registry hash、真实 resource claims、metric depth/extent centers、held-state来源或 task oracle receipt。因此本 memo只给 `|G_profile|` 的精确公式，当前目标 profile 数值与 bits 均 **No-Go / undefined**；不把全局结构数伪装成部署容量。

冻结范围分两级：

- **Go for structural freeze**：字段、vocab、wire、tokens、global grammar、rank/unrank、共享/robot-only边界与profile计数公式；
- **No-Go for executable-profile freeze**：在真实 hardware/calibration/capability entries、global metric codebook和paired-world oracle到位前，不生成profile hash、不实现/训练、不称四任务充分。

## 1. 三个集合与容量口径

r1c 必须始终区分：

```text
G_struct
  本 memo 的全局canonical结构grammar。
  只检查字段依赖、assignment/function拓扑、排序与最多2条constraint。

G_profile(profile_hash)
  在G_struct上应用一个冻结semantic capability registry：
  role/part/function/assignment/working-surface/locator-kind/resource/
  same-effector/task-family allowlists与symmetry canonicalization。

R_runtime(profile_hash, manifest_hash, calibration_hash, state_receipts)
  再应用当前reachability、collision、fresh contact、held binding、camera support、
  locator tolerance与safety predicates。
```

`|G_struct|` 是固定整数。只有存在完整、immutable、全部 admitted entry 已校准的 profile bytes/hash时，`|G_profile|` 才是可复现整数。`R_runtime` 随当前连续robot state变化，不报告跨episode常数 bits；它只报告每anchor admission/abstention。

r1c capacity的准确表述是：

> 给定allowed text、25D proprio、单一冻结 target profile、public calibration语义和通过receipt验证但不被action读取的held binding时，有限robot execution-goal packet的条件结构容量。

它不是总动作带宽、human–robot共享语义容量或机器人全部状态信息。

## 2. Human-shared semantic key

### 2.1 共享字段

Human 与 robot 只共享：

```text
compact_phase
intended_current_effect
constraints[1..2]:
  constraint_function
  primary_role                 # focus
  primary_part
  secondary_role | NONE
  intended_target_relation | N_A
```

不共享：human LEFT/RIGHT、human pixel/center/depth、bbox/track/pose/direction、robot locator、robot assignment、instance ID、working-surface choice、held transform、outcome、duration、confidence、CAPOT mass或pair ID。

`constraint_function` 是跨具身功能语义：human/robot都可有主操作、稳定和被动监测；`assignment` 是robot-only执行绑定。human observed actor不得决定robot assignment。

### 2.2 Canonical semantic key

每个domain-neutral constraint按：

```text
(constraint_function,
 primary_role, primary_part,
 secondary_role, intended_target_relation)
```

升序；允许数组中存在两个字节相同的semantic entries以保留“两个功能相同但未来分给不同robot resources”的multiplicity，但任何human side不进入排序。canonical object：

```text
{
  "pair_key_version": "sm_fet_r1c_semantic_pair_v1",
  "task_ontology_version": <frozen string>,
  "task_family": <frozen enum>,
  "compact_phase": <enum>,
  "intended_effect": <enum>,
  "constraints": [<1 or 2 sorted semantic entries>]
}
```

`semantic_key=SHA256(RFC8785_canonical_JSON_UTF8)`。key equality只准入pair候选，不自动成为positive。CAPOT strict cost只读上述semantic fields、known masks与local order；human/robot geometry、literal side、duration、appearance、outcome和continuous auxiliary全部屏蔽。

### 2.3 Eight-kind 投影

```text
APPROACH/PRECONTACT -> APPROACH_ALIGN + NONE
ACQUIRE             -> ACQUIRE + NONE
HOLD                -> CONTROLLED_CONTACT + NONE
MANIPULATE          -> CONTROLLED_CONTACT + frozen intended effect
REGRASP             -> REGRASP + NONE
RELEASE             -> RELEASE + NONE
VERIFY_OUTCOME      -> VERIFY + NONE
```

HOLD+TRANSLATE不允许；可执行TRANSLATE必须进入MANIPULATE语义。observed multi-effect/outcome另存annotation/supervisor，不从事后结果反写intended effect/relation。required shared字段 unknown/missing时不生成semantic key或action packet。

## 3. Core vocabularies

### 3.1 Role

| code | role |
|---:|---|
| 0 | `PAD/NONE`，按字段语境 |
| 1 | `MANIPULATED_OR_SOURCE` |
| 2 | `TOOL` |
| 3 | `RECEPTACLE_OR_DESTINATION` |
| 4 | `FIXTURE_OR_SUPPORT` |

多实例必须由robot current geometry、named goal和tolerance唯一解；否则`ROLE_INSTANCE_OVERFLOW`。instance/track ID不进wire。

### 3.2 Compact phase 与 effect

| phase code | phase | legal effect |
|---:|---|---|
| 0 | `PAD` | PAD only |
| 1 | `APPROACH_ALIGN` | `NONE` |
| 2 | `ACQUIRE` | `NONE` |
| 3 | `CONTROLLED_CONTACT` | any 7 known effects |
| 4 | `REGRASP` | `NONE` |
| 5 | `RELEASE` | `NONE` |
| 6 | `VERIFY` | `NONE` |

Effect codes保持：0 PAD；1 NONE；2 TRANSLATE；3 ROTATE；4 ARTICULATE；5 FLOW；6 DEFORM；7 TRANSFER。无action-visible UNKNOWN。

### 3.3 Constraint function

| code | function | global effect | opens action slice |
|---:|---|---:|---:|
| 0 | `PAD` | 否 | 否 |
| 1 | `OPERATE` | 直接实现 | 必须是physical effector |
| 2 | `STABILIZE` | 默认不响应 | 必须是physical effector |
| 3 | `MONITOR` | 不响应 | 否 |

`MONITOR`只暴露fresh direct/sensor/native-confirmed passive relation，不等于unknown、hold-last或inactive stabilizer。右Omni夹住瓶体是STABILIZE，不是MONITOR。非VERIFY至少一条OPERATE；VERIFY禁止OPERATE，只允许STABILIZE/MONITOR。

### 3.4 Executable part

| code | part |
|---:|---|
| 0 | `PAD` |
| 1 | `N_A`，VERIFY only |
| 2 | `BODY` |
| 3 | `HANDLE_OR_GRASP_FEATURE` |
| 4 | `LID_CAP` |
| 5 | `OPENING_OR_RIM` |
| 6 | `TOOL_TIP` |
| 7 | `ACTUATOR_OR_PLUNGER` |
| 8 | `WORKING_FACE_OR_PAD` |

非VERIFY必须用七个known parts；VERIFY只N_A。`OTHER`可留annotation ontology，但不能编入biochem executable grammar；unknown也不能借N_A。

### 3.5 Secondary relation

Relation codes：0 N_A；1 SEPARATE；2 NEAR；3 TOUCHING；4 INSIDE；5 CONTAINS；6 ALIGNED_TO_RIM；7 SUPPORTED_BY。

secondary=NONE当且仅当relation=N_A。非零secondary必须不同primary且relation为1..7，方向固定 `primary -> secondary`。OVERLAP_2D继续排除。

## 4. Robot-only finite local frame

### 4.1 Translation：`64×64×32`

```text
uv_code:
  0 PAD
  1..4096 = 1 + 64*row + col
  row,col in 0..63 on normalized current robot image

depth_code:
  0 PAD
  1..32 = one globally fixed camera-z bin

P = 4096 * 32 = 131,072
```

同一 camera hardware/codebook hash只有一套global depth centers；task profile只能删cell，不能重定义同一码。当前材料没有target hardware的真实 `[z_min_global,z_max_global]`、bin centers或metric covering error receipt，所以结构cardinality可冻结，executable metric codebook不可伪造。

### 4.2 Axis 与 roll full frame

```text
axis_code:
  0 PAD
  1..512 = frozen directed-axis center table

roll_code:
  0 PAD
  1..32; r=code-1, angle=2*pi*r/32

D=512, R=32, F=D*R=16,384
```

512 axis table必须物化/hash。对每个axis center `z_d` 的zero-roll基：从camera x/y/z选择与`z_d`最不平行的seed（最小absolute dot；tie `x<y<z`），`b1=normalize(seed×z_d)`、`b2=z_d×b1`；roll确定右手frame `[x,y,z]`。

AXIS只允许profile/oracle证明绕z的roll对first-8 safe action等价的task。FRAME中，surface语义为 `z=outward normal, x=directed tangent`；connector/tool中z是functional/insertion axis，x是key/flat/plunger/pour-plane reference。symmetry必须选唯一canonical roll，不允许多个wire codes解成相同控制目标。

### 4.3 Path templates

```text
primitive_code:
  0 PAD
  1 LINE
  2 ARC_LEFT
  3 ARC_RIGHT
  4 RASTER_3

extent_code:
  0 PAD
  1..8 globally fixed log-spaced metric scale centers
```

PATH使用完整frame：z为surface normal、x为initial tangent；primitive是global table中的固定无量纲polyline，extent只统一缩放。RASTER_3的三程、turn shape、相对间距和aspect ratio全部固定。禁止per-scene width、curvature、mask、spline、waypoint、continuous residual或private progress。四模板不满足目标region/obstacle tolerance时generic wipe profile No-Go。

### 4.4 Locator kind 与 exact count

| code | kind | required fields |
|---:|---|---|
| 0 | `PAD` | all locator payload zero |
| 1 | `POINT` | translation；axis/roll/primitive/extent PAD |
| 2 | `AXIS` | translation+axis；其它PAD |
| 3 | `FRAME` | translation+axis+roll；primitive/extent PAD |
| 4 | `PATH` | translation+axis+roll+primitive+extent |

每translation的结构variants：

```text
1 POINT
+ 512 AXIS
+ 512*32 FRAME
+ 512*32*4*8 PATH
= 541,185
```

因此global structural locator数：

```text
L = 131,072 * 541,185
  = 70,934,200,320
```

这不是固定profile的locator reachable数。固定profile必须枚举各semantic key下admitted POINT/AXIS/FRAME/PATH tuples；task tolerance、reachability、symmetry与transparent evidence通常不独立，不能把mask cardinalities机械相乘。

### 4.5 Locator的执行语义

Primary role/part仍是共享的受控功能部件。robot compiler用当前目标几何与fresh verified held binding，把“primary part相对secondary的目标frame/path”组合为assigned effector的**目标execution frame/path**，再量化成r1c locator。由此，held transform对动作的全部影响必须先压入hard locator或admission；downstream WAM/controller不读取连续held transform。

若两个held transforms在相同proprio/text下编成相同packet，但native safe action集合不相交，则该profile/分辨率No-Go；不得让controller读取receipt中的transform补continuous residual。

## 5. Robot assignment 与 capability boundary

### 5.1 Assignment codes

| code | assignment | legal function |
|---:|---|---|
| 0 | `PAD` | absent only |
| 1 | `LEFT_RH56` | OPERATE, STABILIZE |
| 2 | `RIGHT_OMNIPICKER` | OPERATE, STABILIZE |
| 3 | `MONITOR_ONLY` | MONITOR only |

assignment是target robot执行字段并计入capacity；human actor永久删除。Constraint function不能由assignment/profile机械隐藏。

### 5.2 One/two-constraint structural grammar

用缩写 `LO/LS/RO/RS/ZM` 分别表示 LEFT-OPERATE、LEFT-STABILIZE、RIGHT-OPERATE、RIGHT-STABILIZE、MONITOR_ONLY-MONITOR。

非VERIFY：

```text
single:
  LO | RO

double:
  LO+RO
  LO+RS
  LS+RO
  LO+ZM
  RO+ZM
  LO+LS        # only structurally representable;
               # target profile must explicitly admit exact compound
```

每个bundle至少一条OPERATE。禁止STABILIZE-only、MONITOR-only、RR、LL的OO/SS与任意其它function/assignment组合。

VERIFY：

```text
single:
  LS | RS | ZM

double:
  LS+RS | LS+ZM | RS+ZM | ZM+ZM
```

ZM+ZM的payload必须不同。VERIFY不读outcome；supervisor根据current evidence终止/recover。

### 5.3 Same-effector gate

`LO+LS`进入 `G_struct` 只表示wire能表达，不表示G1 profile可执行。它进入 `G_profile` 当且仅当：

- exact `(OPERATE part, STABILIZE part, held_binding_kind)` 在immutable `same_effector_bundle_allowlist`；
- resource claims不竞争同一exclusive wrist/finger target；
- fresh held binding、working surfaces、WBC balance/self-collision均有receipt；
- paired-world oracle证明两个目标可同时满足。

当前pipette候选 `OPERATE ACTUATOR_OR_PLUNGER + STABILIZE TOOL_TIP` 仍是 `NEEDS_CALIBRATION`，不进入当前 executable profile。独立 `RIGHT+RIGHT` 竞争Omni单aperture和wrist，属于topology unsupported，连 `G_struct` 都不包含；等价目标先merge/deduplicate。

### 5.4 Working surface不另加action field的条件

固定profile必须让 `(assignment,function,part,held_binding_kind)` 唯一映射到一个working-surface class和resource claim template。若RH56 fingertip/fingerpad/palm的选择会改变安全动作且同key多解，profile不得admit该tuple；必须新schema promotion显式working_surface，而不是让allocatorhidden选择。本r1c capacity不包含working-surface field。

## 6. Profile 与 held transform：external receipt而非旁路

### 6.1 Receipt envelope

每个packet bundle在action-invisible envelope绑定：

```text
semantic_profile_id + semantic_profile_hash
hardware_manifest_hash
camera_metric_codebook_hash
calibration_capture_hash
controller_lookup_hash
held_binding_receipt_hash, if required
compiler/validator version + anchor timestamp
```

这些值只用于选择**单一预冻结评测stratum**、验证source/freshness与拒绝packet，不作为WAM/action/cache输入。

### 6.2 Profile不形成通道

- 一个checkpoint/route/eval stratum只接受一个profile hash；planner/allocator不能逐样本选profile；
- profile只含有限allowlists、global codebook references、symmetry、resource conflicts与固定controller enum/constants；
- 禁止per-scene pose/ROI/map/path、sample-specific gain、continuous target/residual、learned hidden或trajectory lookup；
- profile只能删 `G_struct` states，不能重解释同一code；
- 同profile ID不同bytes/hash一律拒绝；未选registry entry变化不得影响action。

若未来一个checkpoint同时接收多个profile，profile ID必须成为显式context或按tagged union重新计容；本r1c headline不覆盖该路线。

### 6.3 Held binding不形成通道

Held receipt可包含连续 `T_effector_to_controlled_part`、contact validity与uncertainty，但：

1. source只能是固定calibrated mount或当前sensor-confirmed state，不能是planner/allocator写入；
2. 每anchor freshness、hardware/calibration hash和uncertainty必须验证；missing/expired/multihypothesis即no packet；
3. transform只在action前用于构造/验证hard execution locator和assignment feasibility；
4. WAM、split decoder、post-WAM controller、cache和text均不得读取transform或receipt payload；
5. allocator不能输出continuous correction/gain/path；
6. 同packet/proprio/text下替换held receipt不得改变pre-safety action。若safe action本应不同，说明hard locator发生collision，profile直接No-Go；
7. external controller若读取continuous target/held transform产生主要轨迹，必须作为direct geometry servo baseline，不归功于r1c。

所以 held transform 是**编译证据**，不是action condition。其影响要么改变hard locator/assignment，要么导致abstain；不存在第三条连续路径。

## 7. Canonical order、PAD 与 fail-closed

Constraint sort key：

```text
(assignment, function,
 primary_role, primary_part,
 locator_kind, uv, depth, axis, roll, primitive, extent,
 secondary_role, relation)
```

单constraint写slot0，slot1全PAD；双constraint严格升序。LO+LS因function code不同可唯一排序；ZM+ZM payload必须严格不同。duplicate、第三条、非法same-effector或profile不admit均不截断。

全局NULL是全wire零，且只由EXHAUSTIVE confirmed no-active授权。其它规则：

- required known字段使用known code；
- REVIEWED_UNKNOWN与NOT_REVIEWED/MISSING原因不同，但都no packet；
- OCCLUDED/OUT_OF_FRAME/TRACKER_INACTIVE本身不编码；无fresh独立证据即no packet；
- transparent 2D overlap不推出depth/normal/TOUCHING/INSIDE；completed depth/mesh/NeRF/teacher fill不作oracle；
- support跨会导致不同安全动作的bins时不取nearest/mean；
- 任一constraint required field未知，整个packet拒绝；不能drop stabilization/monitor后执行第一条；
- prior packet/cache立即失效，control plane执行verified `HOLD/REOBSERVE`，不last-value hold；
- `CONTROLLED_CONTACT+NONE` 是known hold-like task state，不是epistemic abstention。

Stable reasons至少包括：

```text
ROLE_INSTANCE_OVERFLOW
CONSTRAINT_OVERFLOW / DUPLICATE_CONSTRAINT
SAME_EFFECTOR_NOT_ADMITTED / RESOURCE_MUTEX
ASSIGNMENT_AMBIGUOUS / INFEASIBLE
LOCATOR_UNKNOWN / RESOLUTION_INSUFFICIENT
HELD_BINDING_MISSING_OR_EXPIRED
WORKING_SURFACE_AMBIGUOUS
PROFILE_OR_CODEBOOK_HASH_MISMATCH
REQUIRED_SEMANTIC_UNKNOWN
```

这些reason/envelope不进action alphabet。

### 7.1 Active-only clock

Planner仍可在action外维护 `Q_plan_max=8` 的full event/boundary/censor/future supervision；r1c action固定 `Q_action_max=1`、future `N=0`，没有onset/duration bytes。Phase图为：

```text
APPROACH_ALIGN -> ACQUIRE
ACQUIRE -> CONTROLLED_CONTACT
CONTROLLED_CONTACT -> REGRASP | RELEASE
REGRASP -> CONTROLLED_CONTACT | RELEASE
RELEASE -> VERIFY
VERIFY -> NULL
```

每个execute-8 anchor重观测并完整重编1..2条current constraints；phase可自保持，CONTROLLED effect可更新。STABILIZE/MONITOR是否继续只由下一anchor fresh constraint set表达，不能靠private lifecycle或旧cache。WAM可预测48步但只执行前8步；任一receipt/constraint失效立即清packet/cache并进入安全控制面。

## 8. Exact 32-byte wire 与 five tokens

### 8.1 Wire

所有多字节字段为unsigned little-endian。Header：

| offset | type | field |
|---:|---|---|
| 0 | `uint8` | phase |
| 1 | `uint8` | effect |

每条constraint 15 bytes；slot0 base=2，slot1 base=17：

| base+offset | type | field | present range |
|---:|---|---|---|
| +0 | `uint8` | function | 1..3 |
| +1 | `uint8` | assignment | 1..3 |
| +2 | `uint8` | primary_role | 1..4 |
| +3 | `uint8` | primary_part | 1..8，phase-dependent |
| +4 | `uint8` | locator_kind | 1..4 |
| +5..6 | `uint16_le` | uv | 1..4096 |
| +7 | `uint8` | depth | 1..32 |
| +8..9 | `uint16_le` | axis | 0..512，kind-dependent |
| +10 | `uint8` | roll | 0..32，kind-dependent |
| +11 | `uint8` | primitive | 0..4，kind-dependent |
| +12 | `uint8` | extent | 0..8，kind-dependent |
| +13 | `uint8` | secondary_role | 0..4 |
| +14 | `uint8` | target_relation | 0..7 |

唯一NULL是32字节全零；absent slot必须15字节全零。schema/profile/hash/timestamp/provenance/receipt只在envelope。

### 8.2 Tokens

```text
token 0: EVENT(phase,effect)
token 1: CONSTRAINT_0(function,assignment,primary_role,part,
                      secondary_role,relation)
token 2: LOCATOR_0(kind,uv,depth,axis,roll,primitive,extent)
token 3: CONSTRAINT_1(...)
token 4: LOCATOR_1(...)
```

单constraint有3个non-PAD tokens，双constraint有5个；NULL只保留零EVENT sentinel。Constraint/locator token必须成对present/mask。Action只读这5 tokens、25D proprio/state、allowed structured text、diffusion timestep/noisy action；不读human、raw reference、planner/locator/allocator hidden、profile/receipt、continuous geometry或future。

## 9. Global structural alphabet：exact integer

### 9.1 Locator 与 payload

```text
P = 64*64*32 = 131,072
D = 512
R = 32
K = 4
E = 8

locator variants per translation
= 1 + D + D*R + D*R*K*E
= 541,185

L = P*541,185
  = 70,934,200,320
```

固定一个assignment/function，VERIFY constraint payload：

```text
V = 4 primary roles * 1 N_A part * L * 22 secondary bundles
  = 6,242,209,628,160
```

非VERIFY有七个known parts：

```text
N = 7*V
  = 43,695,467,397,120
```

### 9.2 Bundle count

非VERIFY有2个single categories与6个double categories：

```text
B_nonverify
= 2*N + 6*N^2
= 11,455,763,226,316,748,597,640,560,640
```

VERIFY有3个single categories、3个different-type pair categories，以及不同ZM payload的无序pair：

```text
B_verify
= 3*V + 3*V^2 + C(V,2)
= 136,378,133,646,642,524,651,520,000
```

非VERIFY phase/effect blocks共11个；VERIFY 1个；再加唯一NULL：

```text
|G_struct|
= 1 + 11*B_nonverify + B_verify
= 126,149,773,623,130,877,098,697,687,041

log2 |G_struct|
= 96.67105236912121 bit

ceil(log2 |G_struct|)
= 97 bit
```

97-bit index装入13 bytes；104-bit container高7位必须0，且rank `< |G_struct|`。32-byte field wire的存储宽度不是256-bit semantic capacity。

### 9.3 Constructive rank/unrank

Translation rank：`p=((uv-1)*32+(depth-1)) in [0,P-1]`。Locator variant：

```text
POINT: q=0
AXIS:  q=1+(axis-1)
FRAME: q=1+512+(axis-1)*32+(roll-1)
PATH:  q=1+512+16384
         + (((axis-1)*32+(roll-1))*4+(primitive-1))*8
         + (extent-1)

locator_rank=p*541185+q                       # 0..L-1
```

Secondary rank：NONE/N_A为0；否则从升序roles删除primary，secondary local index `j in [0,2]`，`s=1+j*7+(relation-1)`，范围0..21。

固定assignment/function的payload按 `(part_index,primary_role,locator_rank,secondary_rank)` 混合进制排名，得到VERIFY `0..V-1` 或nonVERIFY `0..N-1`。

NonVERIFY bundle blocks依次为：

```text
single LO, single RO,
pair LO-RO, LO-RS, LS-RO, LO-ZM, RO-ZM, LO-LS
```

前两块各N，后六块各N²。VERIFY blocks依次为：

```text
single LS, single RS, single ZM,
pair LS-RS, LS-ZM, RS-ZM, pair ZM-ZM
```

最后一块要求 `q0<q1`，combinadic rank：

```text
q0*(2*V-q0-1)/2 + (q1-q0-1)
```

覆盖0..C(V,2)-1。11个nonVERIFY phase/effect blocks与VERIFY block最后拼接，NULL rank=0。逐层divmod/combinadic unrank唯一恢复wire，证明计数未把function、same-effector、PAD或canonical pair当独立dense fields误乘。

## 10. Fixed target-profile exact formula；当前 No-Go 数值

### 10.1 Profile sets

对固定 authoritative profile `p` 和每个phase/effect block `h`，定义已应用全部**静态** allowlists、唯一working surface、global codebook mask、symmetry和resource templates后的payload集合：

```text
Omega_h_LO, Omega_h_LS,
Omega_h_RO, Omega_h_RS,
Omega_h_ZM
```

元素包含合法 `(primary role,part,locator,secondary,relation)`；VERIFY sets自动只含N_A，nonVERIFY只含七known parts。对每个semantic key，profile还必须唯一冻结 controller enum、working surface、resource template与held-binding kind；若同一wire tuple可由两个会产生不同动作的external binding解释，该tuple不是“两次可达”，而是profile歧义并必须删除。

令：

```text
C_p(A,B) = sum_{x in A, y in B} I(profile_static_compatible(x,y))
U_p(A)   = sum_{x<y; x,y in A} I(profile_static_compatible(x,y))
```

兼容谓词必须包含resource mutex、same-effector exact allowlist、role-instance/part冲突、working-surface唯一性和task-family规则；不得依赖runtime allocator score。

### 10.2 Exact profile formula

对每个nonVERIFY block：

```text
B_h(p)
= |Omega_h_LO| + |Omega_h_RO|
  + C_p(Omega_h_LO, Omega_h_RO)
  + C_p(Omega_h_LO, Omega_h_RS)
  + C_p(Omega_h_LS, Omega_h_RO)
  + C_p(Omega_h_LO, Omega_h_ZM)
  + C_p(Omega_h_RO, Omega_h_ZM)
  + C_p(Omega_h_LO, Omega_h_LS)       # profile-gated LEFT compound
```

VERIFY：

```text
B_verify(p)
= |Omega_v_LS| + |Omega_v_RS| + |Omega_v_ZM|
  + C_p(Omega_v_LS, Omega_v_RS)
  + C_p(Omega_v_LS, Omega_v_ZM)
  + C_p(Omega_v_RS, Omega_v_ZM)
  + U_p(Omega_v_ZM)
```

所以固定profile的 exact structural reachable alphabet：

```text
|G_profile(p)|
= 1 + sum_{h in 11 admitted nonVERIFY phase/effect blocks} B_h(p)
    + B_verify(p)

bits_profile(p)=log2 |G_profile(p)|
```

若profile禁用某phase/effect，相关集合为空。这个公式可以由canonical enumeration/adjacency audit复算，不假设locator、part、assignment和资源独立。

### 10.3 为什么当前不报G1数值

现有 capability memo只是semantic registry设计，不是可部署实例：RH56/Omni task binding、working surfaces、resource claims、LEFT compound、WBC/collision、camera metric codebook、held state、driver/contact与四任务oracle全部仍是 `NEEDS_CALIBRATION` 或 unknown，且没有 authoritative `semantic_profile_hash`。

因此当前：

```text
|G_profile(g1-rh56-omni-biochem4)| = UNDEFINED / NO-GO
bits_profile                       = UNDEFINED / NO-GO
```

不能把所有 `NEEDS_CALIBRATION` 当true计算“候选profile”，也不能把它们全删后报告只有NULL的1-state alphabet；后者会把“profile尚不存在”误写成“一个合法但什么也不做的部署profile”。

Runtime `R_runtime`更不报告固定整数；每anchor只记录从已冻结 `G_profile` 中被fresh receipts实际admit的packet或abstain。

## 11. Equal-bit controls

### 11.1 Global structural codec control

只用于CPU rank/codec与encoder inductive-bias诊断。令：

```text
M=|G_struct|-1
 =126,149,773,623,130,877,098,697,687,040
```

定义：

```text
pi(0)=0
pi(u)=1+((65537*(u-1)+104729) mod M), u in 1..M
```

`gcd(65537,M)=1`，模逆元为 `89710184315419345683536084993`。它在global structural states上全域可逆、NULL固定，使用同一32-byte wire/5-token encoder。它**不准直接上安全控制**：置换可能把场景packet改成另一assignment/function/geometry，即使结构合法也不再是该场景truth。

### 11.2 Target-profile safe equal-bit

只有 `G_profile(p)` 已精确枚举后才定义。安全stratum固定：

```text
z=(phase,effect,constraint_count,
   ordered (assignment,function) tuple)
```

在每个stratum的profile-admitted payload set `Omega_z` 内rank并冻结双射 `pi_z`；置换发生在admission之后，不重新跑allocator，不改变sample/failure set、split action gate或MONITOR行为。Semantic与equal route共享profile、receipts、assignment/function routing、token/参数/FLOPs和abstention。

这个control匹配每stratum的residual capacity，检验role/part/frame/relation编码；它有意不随机化安全routing fields，不能单独证明assignment/function语义必要。whole-packet global permutation若被用于robot rollout，会制造额外物理不可行性，不能拿其失败支持r1c。

当前 profile未定义，因此 target-profile `Omega_z`、permutation与equal-bit数值同样 No-Go。

## 12. 与 r1a、r1b、standard-r1 对比

| 项 | r1a | r1b | **r1c** | standard-r1 |
|---|---|---|---|---|
| temporal | active | active | **active** | active+next1+next2 |
| timing/outcome | 无/外置 | 无/外置 | **无/外置** | timing + verify outcome |
| human actor | 无 | 无 | **无** | LEFT/RIGHT混合语义 |
| constraint function | 无 | 无 | **O/S/M显式** | 无独立function |
| parts | 6 known | 6 known | **7 known，含plunger/pad，无OTHER** | 8 known+UNKNOWN |
| locator | full-object 4×4 center | 64×64×32 + point/axis/line extent | **64×64×32 + AXIS/FRAME + 4 PATH×8** | node粗geometry + part-local4×4 |
| orientation | 无 | 512 axis | **512 axis×32 roll full frame** | image-plane orientation |
| constraints | exactly1 | 1..2；禁same-effector | **1..2；profile-gated LEFT O+S** | ≤2 edges/event |
| robot assignment | 不显式 | L/R/M显式 | **L/R/M显式** | actor四码，不等于target assignment |
| field wire | 10 bytes | 26 bytes | **32 bytes** | 82 bytes |
| tokens | 5 | 5 | **5** | 13 |
| global structural states | 1,445,057 | 3,367,384,122,335,042,437,906,433 | **126,149,773,623,130,877,098,697,687,041** | `345864043392188289642033658185203030253953267411599196753571073` |
| global structural bits | 20.46269497000547 | 81.47790257814123 | **96.67105236912121** | 207.74974691951957 |
| target-profile bits | 未定义 | 未定义 | **公式已冻结；当前No-Go数值** | 未做G1 task-profile枚举 |

r1c相对r1a增加 `76.20835739911574 bit`，相对r1b增加 `15.193149790979987 bit`，仍比standard全局grammar少 `111.07869455039835 bit`。这些是完整structural grammar的log ratio，不能分摊成字段独立贡献。

兼容性：四者wire/codec/checkpoint互不兼容。r1b不能生成r1c roll/frame/function/new parts/LEFT compound；必须重新读取robot evidence编译。standard的human actor、4×4 site、image orientation也不能生成robot frame或assignment。r1c可投影到r1a/r1b作diagnostic ablation，但丢失信息后不得复用为oracle action。

## 13. CPU-only invariants 与 oracle gates（本轮不实现）

### 13.1 Wire/codec

1. 32字节全零是唯一NULL；nonzero header与零constraint或反之拒绝；
2. slot0必须present；single时slot1十五字节全零；constraint/locator tokens成对mask；
3. phase/effect严格通过11 nonVERIFY + 1 VERIFY block；
4. nonVERIFY part只七known，VERIFY只N_A；OTHER/unknown不可编；
5. function/assignment只允许LO/LS/RO/RS/ZM匹配；
6. nonVERIFY与VERIFY bundle只允许第5.2节列出的categories；
7. POINT/AXIS/FRAME/PATH的axis/roll/primitive/extent PAD规则逐字执行；
8. locator rank恰覆盖70,934,200,320 states；axis/roll frame table全码唯一；
9. secondary NONE iff relation N_A，secondary!=primary；
10. canonical sort严格；duplicate、third constraint、RR、非法LL稳定拒绝；
11. global wire↔rank覆盖恰好`0..|G_struct|-1`，无hole/collision；
12. 13-byte rank高7位非零或rank越界拒绝；
13. global equal-bit affine/inverse全域双射、NULL固定；
14. exact integer、96.67105236912121、97-bit、32-byte/5-token receipt逐字固定。

### 13.2 Ownership/missingness/transparent

15. semantic key中不存在human side/pixel/pose/geometry、robot locator/assignment、outcome/pair ID；
16. human geometry改变不改变semantic key/CAPOT strict cost；
17. robot locator只由current robot direct/native evidence生成；human warp/completion不进oracle；
18. reviewed unknown、missing、occlusion、tracker inactive不变成NULL/PAD/nearest bin；
19. transparent无direct depth/normal时packet消失并清旧cache；
20. EXHAUSTIVE no active才授权NULL；censor只在planner clock/loss；
21. fixed packet/proprio/text/noise时替换raw reference、human、pair、planner/locator/allocator hidden，pre-safety action `max_abs<=1e-6`。

### 13.3 Profile/held-state/allocator

22. 一个route只接受一个profile/hash/codebook；未选条目变化不影响action；
23. profile只删global states、不重解释code、不含per-scene geometry/gain/path；
24. held transform只影响hard compile/admission；同packet下receipt swap不改变action；
25. downstream读取held transform、continuous target或receipt必须被attention/API审计拒绝；
26. allocator可见输出只有wire assignment enum；cost/logits/order/history/receipt swap不改变packet/action；
27. assignment swap只改变对应split action slice；MONITOR不开slice；
28. working surface同key多解时profile拒绝，不让allocator暗选；
29. LEFT+LEFT只有exact profile allowlist/resource/oracle通过才进入G_profile；任意RIGHT+RIGHT拒绝；
30. capability status为NEEDS_CALIBRATION/UNKNOWN/UNSUPPORTED的entry不计入G_profile。

### 13.4 Paired-world oracle

31. `O-LOC-POUR/INSERT/WIPE`消除r1a粗center collision；
32. `O-ROLL-INSERT/O-POUR-PLANE`同axis不同roll编成不同FRAME并产生局部正确动作；
33. `O-PATH-NORMAL`同tangent不同normal编成不同frame；
34. `O-PATH-SHAPE`四primitive可区分；`O-WIPE-OUTSIDE`必须No-Go而非nearest template；
35. `O-FUNCTION-CAP`只有OPERATE响应global effect，STABILIZE保持；
36. `O-ASYNC-RELEASE`由fresh constraint set区分，不靠profile private lifecycle；
37. `O-PIPETTE-COMPOUND`在profile admitted后，plunger变化只影响允许RH56 finger slice且tip/wrist在公差；
38. `O-HELD-STATE`不同transform必须形成不同hard locator或abstain；不得同packet不同动作；
39. `O-ALLOC-BYPASS/O-RESIDUAL-BYPASS`保持action不变；
40. relative to full/native trajectory，quantized translation/frame/path满足每task预注册安全公差；否则该profile finite claim停止。

## 14. Freeze recommendation 与停止条件

### 14.1 现在冻结什么

建议现在冻结：

- `minimal-r1c` 32-byte/5-token structural schema；
- O/S/M functions、7-part executable vocab与assignment bundle grammar；
- 64×64×32 + 512×32 frame + four-path×8 cardinality和canonical code construction；
- human-shared semantic key / robot-only geometry-assignment边界；
- global structural exact integer/rank/equal-bit diagnostic；
- `G_profile`计数公式、profile/held receipt anti-channel和missing/transparent fail-closed。

这是一份**设计级 structural freeze**，只解锁CPU checker spec和无训练oracle准备。

### 14.2 当前不能冻结什么

当前明确不冻结/不声称：

- G1 biochem4 executable profile/hash或任何profile bits；
- global metric depth/extent centers与task tolerance；
- RH56/Omni working surface、resource claims、contact/load、driver与reachability；
- pipette LEFT compound admission；
- roll-sensitive insertion、transparent pour、generic wipe的控制充分性；
- 模型实现、训练、仿真、真机或论文headline。

### 14.3 Stop

任一发生则停止相应r1c finite profile，不继续添加hidden字段：

- 64×64×32、512×32 frame或8 extent covering error超过task tolerance；
- four path templates无法覆盖目标region/obstacle corridor；
- full/native trajectory相对finite locator有预注册安全gap；
- transparent成功只在completion/reconstruction/continuous rim pose action-visible时出现；
- same-effector compound资源无法实机并发；
- profile必须逐scene改decoder/range/gain/path；
- held transform/continuous residual必须被downstream读取；
- allocator必须靠cost/hidden暗选assignment/working surface；
- third independent constraint持续必要；
- target-profile exact enumeration与admission/failure set无法匹配equal-bit control。

发生这些情况，让 F-PCD、direct visual servo、standard或continuous trajectory作为互斥winner；不要把它们concat进r1c后仍引用96.6711-bit结构口径。

## 15. 最终一句话

`minimal-r1c` 是目前唯一同时守住“不过度恢复standard字段”和“不给控制必要信息做碰撞压缩”的 final candidate：human只共享 `phase/effect/function/role/part/relation`，robot在当前视角独立生成达到公差的translation/full-frame/path并显式写入RH56/Omni assignment；profile与held transform只验证、编译或拒绝，不能被action读取形成连续旁路。其global structural grammar精确为 `126,149,773,623,130,877,098,697,687,041` states / `96.67105236912121 bit`，但真实 G1 profile因为所有task capability仍待标定，当前只允许公式与No-Go，不允许伪造reachable integer。建议冻结结构，不冻结可执行profile；下一步止于CPU contract与无训练paired-world oracle。
