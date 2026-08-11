# SM-FET `minimal-r1b`：执行 locator、双约束与显式 robot assignment 的唯一折中

- 日期：2026-08-12
- 性质：基于 minimal-r1a 与独立控制充分性审查的 schema/codec 裁决；不实现、不训练、不使用 GPU、不修改 canonical
- canonical 代码基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- schema ID：`sm_fet_hard_grammar_v2_minimal_r1b`
- 对照：`minimal-r1a`、`sm_fet_hard_grammar_v2_standard_r1`

## 0. 唯一裁决

`minimal-r1a` 不能作为倒液、插入/移液、旋盖、擦拭四任务的通用执行包：robot-view object 4×4 center 会把同 cell 内不同 rim/well/port/stain 编成同一输入；单 interaction 又不能同时表达“执行主操作”和“另一末端/fixture稳定对象”。但这不支持恢复 standard-r1 的 future、timing、human actor 或 outcome。

唯一折中定为 **`minimal-r1b`**：

```text
active-only global goal:
  compact phase + known current effect

1 or 2 concurrent execution constraints:
  explicit target-robot effector assignment
  primary role + coarse functional part       # primary is focus
  target-domain camera-frame execution locator
  optional secondary role + intended target relation
```

明确删除：future、onset、duration、human LEFT/RIGHT、future/current outcome、historical relation_before、独立 contact mode、continuous pose/track。明确恢复：达到预注册公差的 robot-view execution locator，以及最多两条并发约束。robot assignment 不再藏在 allocator adapter、action维度置换或hidden中，而是进入 canonical wire、token和 exact capacity。

全局 validator grammar 的精确结果是：

```text
canonical field wire = 26 bytes
action token slots   = 5
reachable states     = 3,367,384,122,335,042,437,906,433
exact capacity       = 81.47790257814123 bit
minimum index width  = 82 bit
compact container    = 11 bytes
```

这是一个**可证伪的 finite candidate**，不是“64×64必然满足所有机器人任务”的承诺。每个 task profile必须证明位置、深度、轴和路径extent量化误差小于该任务安全公差；不满足时该 profile对 r1b 为 No-Go，并比较 F-PCD/direct visual servo/full trajectory，不能继续加连续 hidden后沿用81.4779-bit口径。

## 1. 为什么是 r1b，而不是回到两端

独立控制审查构造了四类 packet collision：

1. target rim在同4×4 cell内横移/转轴，pour安全corridor不相交；
2. 同cell内不同well/port或不同插入轴，安全approach不相交；
3. 同中心不同污渍主轴/extent，安全wipe path不相交；
4. fixture locked/unlocked保持原r1a packet相同，但是否需要第二末端稳定不同。

因此 r1b 必须比 r1a 多两类信息：执行局部frame/path，以及最多两条独立当前约束。审查同时没有给出以下字段的 first-8 safe-action 必要反例：future、duration、human laterality、outcome、历史 relation_before。r1b不恢复它们。

r1b 是**目标机器人 action packet**，不是 human annotation packet。human/robot共享与配对的只是 role/part/phase/effect/target relation；robot camera locator和robot effector assignment在目标域当前 replan anchor重新生成，绝不复制 human pixel/side。

## 2. 固定 role、phase、effect、part 与 relation

### 2.1 Role

| code | role |
|---:|---|
| 0 | `PAD/NONE`，按字段语境解释 |
| 1 | `MANIPULATED_OR_SOURCE` |
| 2 | `TOOL` |
| 3 | `RECEPTACLE_OR_DESTINATION` |
| 4 | `FIXTURE_OR_SUPPORT` |

task profile必须把每条 constraint 的 role唯一绑定到目标域实体。同 role 多实例只有在 `(role, part, locator)` 与 task goal能唯一解引用时才合法；否则 `ROLE_INSTANCE_OVERFLOW`。r1b不加入跨episode instance ID，避免把tracker identity变成隐藏语义通道。

### 2.2 Compact active phase

| code | phase |
|---:|---|
| 0 | `PAD`，只用于全局NULL |
| 1 | `APPROACH_ALIGN` |
| 2 | `ACQUIRE` |
| 3 | `CONTROLLED_CONTACT` |
| 4 | `REGRASP` |
| 5 | `RELEASE` |
| 6 | `VERIFY` |

仍无 action-visible `UNKNOWN`。reviewed unknown 与 missing保留不同原因，但required phase任一未知均不编 packet。

### 2.3 Current effect

| code | effect |
|---:|---|
| 0 | `PAD`，只用于NULL |
| 1 | `NONE` |
| 2 | `TRANSLATE` |
| 3 | `ROTATE` |
| 4 | `ARTICULATE` |
| 5 | `FLOW` |
| 6 | `DEFORM` |
| 7 | `TRANSFER` |

phase/effect 的合法组合保持 r1a 的去冗余规则：

```text
APPROACH_ALIGN      -> NONE
ACQUIRE             -> NONE
CONTROLLED_CONTACT  -> any of 7 known effects
REGRASP             -> NONE
RELEASE             -> NONE
VERIFY              -> NONE
```

`CONTROLLED_CONTACT+NONE` 是 HOLD-like，non-NONE 是 MANIPULATE-like。FLOW/TRANSFER必须来自 causal goal或直接/sensor/native evidence，不能从task name、腕倾角或事后结果补GT。

### 2.4 Primary part

| code | part |
|---:|---|
| 0 | `PAD`，只用于NULL/absent constraint |
| 1 | `N_A` |
| 2 | `BODY` |
| 3 | `HANDLE_OR_GRASP_FEATURE` |
| 4 | `LID_CAP` |
| 5 | `OPENING_OR_RIM` |
| 6 | `TOOL_TIP` |
| 7 | `OTHER` |

每条非VERIFY constraint必须用六个known parts；VERIFY只允许`N_A`。`BODY`加PATH locator可表达已知surface/stain region；`OTHER`是已知其它功能部位，不吸收unknown。

### 2.5 Intended target relation

| code | relation |
|---:|---|
| 0 | `N_A`，仅允许secondary为NONE |
| 1 | `SEPARATE` |
| 2 | `NEAR` |
| 3 | `TOUCHING` |
| 4 | `INSIDE` |
| 5 | `CONTAINS` |
| 6 | `ALIGNED_TO_RIM` |
| 7 | `SUPPORTED_BY` |

方向固定为 `primary_role -> secondary_role`。secondary为NONE时relation必须N_A；secondary非零时必须不同primary且relation为1..7。`OVERLAP_2D`仍被排除；它不能证明TOUCHING/INSIDE。

## 3. Robot-view execution locator

### 3.1 选择

r1b 不用 r1a 的 full-object 4×4 center，也不机械恢复 standard 的 part-local 4×4 site。每条 constraint使用一个校准 robot-camera frame中的有限 locator：

```text
anchor_uv     : 64 x 64 normalized robot-view grid
depth         : 32 metric bins in profile workspace
locator_kind  : POINT | AXIS | PATH
axis          : 32 azimuth x 16 elevation bins, when required
extent        : 8 log-spaced metric bands, PATH only
```

选择理由：

- 64×64把4×4每轴细化16倍、面积细化256倍；在640×480输入上约对应10×7.5 pixel cell，可区分审查中的同4×4-cell目标；
- 32个profile-bounded depth bins让camera anchor成为3D执行目标；只给2D cell无法从proprio唯一求当前平移误差；
- 32×16有向axis使插入、rim approach/tilt、wipe tangent不再依赖planner hidden；bin宽分别11.25°，从bin中心量化的组合角误差上界保守记为8°；
- POINT不为不需要axis的任务浪费多重N/A编码，AXIS表达approach/insertion/tilt，PATH在同一axis上再给8档extent；
- tolerance不做自由action字段，而由 `(task_profile, phase, effect, role, part, relation, locator_kind)` 的冻结controller表唯一给出。若同一 retained key需要不同tolerance/controller profile，profile拒绝或另立版本。

这个选择只定义候选分辨率。task profile admission必须用paired-world/native geometry证明：最大位置、depth、axis、extent量化误差分别不超过冻结的 `epsilon_pos/depth/axis/extent`。任何一项失败，不能把目标硬塞进最近bin；应判 `LOCATOR_RESOLUTION_INSUFFICIENT`。

### 3.2 精确 code

`anchor_uv` 用 `uint16`：

```text
0      = PAD
1..4096:
  row = floor((code-1)/64)
  col = (code-1) mod 64
  code = 1 + 64*row + col
```

输入点采用 normalized robot image `(u,v)`，`row=min(63,floor(64*v))`、`col=min(63,floor(64*u))`。locator必须来自当前目标域因果观测/native state；human cell不可复制。

`depth_code` 用 `uint8`：0为PAD，1..32是在 task profile冻结 `[z_min,z_max]` 内的等宽camera-z区间。超出工作区、跨多个不可唯一bin或没有direct/sensor/native depth evidence均fail closed。透明物体的completed depth、mesh/NeRF或teacher fill不能升级为oracle GT。

`axis_code` 用 `uint16`：0为PAD，1..512。对camera-frame有向单位向量：

```text
azimuth bin   a in 0..31 over [-pi, pi)
elevation bin e in 0..15 over [-pi/2, pi/2]
axis_code = 1 + 32*e + a
```

每个code解码为对应bin中心。严格竖直输入的azimuth canonical取0；量化support跨越多个不满足task angular tolerance的bin时拒绝，不由标注者挑最近方向。

`extent_code` 用 `uint8`：0为PAD，1..8。task profile冻结正数 `[d_min,d_max]`，按log-distance等宽分8档；PATH表示从anchor沿有向axis的目标path长度/region major extent。目标超范围或跨档不满足任务公差时拒绝。

`[z_min,z_max]`、`[d_min,d_max]`、camera intrinsics/extrinsics与tolerance表必须来自有限、版本化的 target-hardware profile registry；同一 profile 在所有样本/route中逐字不变，不能由planner或allocator逐样本选择。profile ID必须由部署允许的instruction/config公开确定，并与semantic/equal-bit route共享；若action端不知道profile或同一ID对应多套量化表，packet inadmissible。这样profile不是allocator可调的连续旁路。

`locator_kind` 用 `uint8`：

| code | kind | axis | extent |
|---:|---|---|---|
| 0 | `PAD` | 0 | 0 |
| 1 | `POINT` | 0 | 0 |
| 2 | `AXIS` | 1..512 | 0 |
| 3 | `PATH` | 1..512 | 1..8 |

每个非PAD locator都必须有 `anchor_uv in 1..4096` 与 `depth in 1..32`。因此 locator没有“给了axis但kind=POINT”或“N/A axis的多个写法”。

### 3.3 Current error 的机械来源

r1b不新增历史relation或自由current-error字段。当前平移/方向误差必须由以下公开、版本化函数机械得到：

```text
decoded camera-frame target anchor/axis
+ calibrated camera extrinsics
+ current 25D robot proprio
+ sensor-confirmed held-object/tool transform when applicable
-> current target-relative error
```

若 held transform、camera calibration、depth、current contact或目标frame任一不可用，required constraint不编译。allocator不得私下计算一条连续target pose直接传给WAM；action expert只收到离散locator。若离散locator达不到安全公差，则finite route失败，不能用私有continuous correction补救。

### 3.4 Locator alphabet

位置部分：

```text
P = 4096 uv cells * 32 depth bins = 131,072
```

每个位置的locator variant：

```text
POINT = 1
AXIS  = 512
PATH  = 512 * 8 = 4,096
variants = 4,609
```

所以：

```text
L = 131,072 * 4,609 = 604,110,848 execution locators
```

## 4. 显式 target-robot effector assignment

### 4.1 Wire vocabulary

| code | assignment | action path |
|---:|---|---|
| 0 | `PAD` | absent constraint only |
| 1 | `LEFT_RH56` | left wrist 9D + RH56 6D |
| 2 | `RIGHT_OMNIPICKER` | right wrist 9D + Omni aperture 1D |
| 3 | `MONITOR_ONLY` | passive fixture/object relation；不直接驱动effector |

这些是目标硬件assignment，不是human LEFT/RIGHT。cross-domain pairing不比较该字段；robot compiler在当前target locator、25D proprio、hardware capability manifest和安全/可达性约束上确定它。

### 4.2 防 allocator 隐藏 action channel

Allocator接口必须冻结为：

```text
input:
  validated functional constraints
  discrete execution locators
  current versioned robot proprio/contact validity
  immutable hardware capability + reachability + safety tables

output visible to action:
  exactly one 2-bit-enough enum per constraint:
    LEFT_RH56 | RIGHT_OMNIPICKER | MONITOR_ONLY

output not visible to action:
  status/reason/receipt/hash only
```

不可协商规则：

1. assignment byte写入26-byte packet，并进入constraint token；它已计入81.4779-bit容量；
2. allocator logits、score、continuous target、cost、candidate order、hidden、KV、receipt和failure reason不得进入action/state/cache；
3. allocator不得通过重排25D state/action维度、修改locator、controller gain、action mask以外的连续值或选择不同decoder偷偷传信息；
4. split control只按公开assignment enum门控：未assigned侧保持verified hold；LEFT+RIGHT bundle同时打开两侧；MONITOR_ONLY不打开动作slice；
5. 同一公开输入必须byte-identical输出；ambiguous/infeasible返回no packet + `HOLD/REOBSERVE`，不能随机选择；
6. 持久support/grasp若影响当前动作，必须是第二条显式constraint，或由独立传感器每anchor重新确认；allocator/controller私有memory不能替代它；
7. semantic与equal-bit route使用完全相同allocator实现、输入和failure set。

因此 r1b 不再声称“assignment在共享语义外所以免费”。它是目标执行packet的显式、已计容字段；被删除的只有human laterality。

## 5. Constraint grammar

### 5.1 单条 constraint

每条 present constraint恰含：

```text
effector_assignment
primary_role             # 同时是focus
primary_part
execution_locator:
  kind, anchor_uv, depth, axis, extent
secondary_role | NONE
target_relation | N_A
```

Secondary bundle计数：

```text
1                                   # NONE + N_A
+ 3 roles other than primary * 7 relations
= 22
```

relation是 intended terminal/current control target，不是事后 observed relation_after。当前error由第3.3节函数计算；历史relation_before不进action。

### 5.2 一条或两条并发约束

全局NULL外，constraint 0必须present；constraint 1可PAD。不存在零constraint non-NULL event。

对非VERIFY phase：

```text
one constraint:
  LEFT or RIGHT

two constraints, allowed assignment sets:
  {LEFT, RIGHT}
  {LEFT, MONITOR_ONLY}
  {RIGHT, MONITOR_ONLY}
```

禁止 `{LEFT,LEFT}` 或 `{RIGHT,RIGHT}`：同一effector的多个目标必须由task profile编成一个frame/path constraint；无法合并即 `SAME_EFFECTOR_CONSTRAINT_OVERFLOW`。禁止单条MONITOR_ONLY，因为非VERIFY active goal必须有可执行effector。

对VERIFY：

```text
one constraint:
  LEFT | RIGHT | MONITOR_ONLY

two constraints:
  {LEFT, RIGHT}
  {LEFT, MONITOR_ONLY}
  {RIGHT, MONITOR_ONLY}
  {MONITOR_ONLY, MONITOR_ONLY}
```

双MONITOR约束必须payload不同；它允许同时检查两个已知对象条件，不允许重复同一状态增加表示数。VERIFY仍不读取outcome；supervisor根据当前evidence决定是否终止/recover。

### 5.3 Canonical order 与 overflow

Constraint sort key固定为：

```text
(assignment,
 primary_role, primary_part,
 locator_kind, anchor_uv, depth, axis, extent,
 secondary_role, target_relation)
```

两条constraint按该tuple严格升序写slot 0/1；单条永远写slot0，slot1十二字节全零。duplicate、第三条constraint或无法合并的same-effector目标全部fail closed，不取排序前两项。stable reasons至少包括：

```text
ROLE_INSTANCE_OVERFLOW
CONSTRAINT_OVERFLOW
SAME_EFFECTOR_CONSTRAINT_OVERFLOW
DUPLICATE_CONSTRAINT
ASSIGNMENT_AMBIGUOUS / ASSIGNMENT_INFEASIBLE
LOCATOR_UNKNOWN / LOCATOR_RESOLUTION_INSUFFICIENT
REQUIRED_PART_OR_RELATION_UNKNOWN
TASK_PROFILE_MISMATCH
```

这些reason不进action wire；部署统一 `HOLD/REOBSERVE`。

## 6. Exact wire 与 tokens

### 6.1 26-byte canonical field wire

所有多字节整数为 unsigned little-endian。header：

| offset | type | field |
|---:|---|---|
| 0 | `uint8` | phase |
| 1 | `uint8` | effect |

每条constraint固定12 bytes；slot `i=0,1` 的base分别为2和14：

| base+offset | type | field | valid present range |
|---:|---|---|---|
| +0 | `uint8` | assignment | 1..3 |
| +1 | `uint8` | primary_role | 1..4 |
| +2 | `uint8` | primary_part | 1..7，受phase约束 |
| +3 | `uint8` | locator_kind | 1..3 |
| +4..5 | `uint16_le` | anchor_uv | 1..4096 |
| +6 | `uint8` | depth | 1..32 |
| +7..8 | `uint16_le` | axis | 0..512，受kind约束 |
| +9 | `uint8` | extent | 0..8，受kind约束 |
| +10 | `uint8` | secondary_role | 0..4 |
| +11 | `uint8` | target_relation | 0..7 |

唯一全局NULL是26字节全零。absent slot也必须12字节全零。任何reserved bit/range、非canonical order、非法PAD、phase/effect/part/locator/assignment bundle都拒绝。schema ID、task profile、camera calibration、anchor time、provenance和receipt在envelope中，不进入action。

### 6.2 五个固定 tokens

```text
token 0: EVENT(phase, effect)
token 1: CONSTRAINT_0(assignment, primary_role, part,
                      secondary_role, target_relation)
token 2: LOCATOR_0(kind, uv, depth, axis, extent)
token 3: CONSTRAINT_1(...)
token 4: LOCATOR_1(...)
```

单constraint packet有3个non-PAD tokens；双constraint有5个；NULL只保留全零EVENT sentinel，其余mask。constraint/locator必须成对present，不能只mask其中一个。Action expert只读这5 tokens、机器人proprio/state、allowed text、diffusion timestep/noisy action；不读raw reference、planner/allocator hidden、future、continuous geometry或receipt。

## 7. Episode/chunk clock 与 missing/transparent contract

### 7.1 Clock

`Q_plan_max=8` 的完整事件、boundary/censor、future与confidence仍只在planner/control面。r1b的action-visible `Q_action_max=1`、future `N=0`。

phase图与r1a一致：

```text
APPROACH_ALIGN -> ACQUIRE
ACQUIRE -> CONTROLLED_CONTACT
CONTROLLED_CONTACT -> REGRASP | RELEASE
REGRASP -> CONTROLLED_CONTACT | RELEASE
RELEASE -> VERIFY
VERIFY -> NULL
```

每8步reobserve/recompile；当前phase可自保持，CONTROLLED effect可更新。每个anchor必须重算locators、验证contact/held transform并重新确认assignment；旧locator或私有controller memory不能last-value hold。WAM仍可预测48步，但只执行前8步。

### 7.2 Missingness

| source truth | wire/action result |
|---|---|
| EXHAUSTIVE confirmed no active event | 唯一NULL |
| required known field | known code |
| REVIEWED_UNKNOWN | no packet；reason保留planner/annotation |
| NOT_REVIEWED/MISSING | no packet；不变成UNKNOWN/NULL/PAD |
| OCCLUDED/OUT_OF_FRAME/TRACKER_INACTIVE | 本身不编码；有独立证据才可known，否则fail closed |
| boundary censor | planner likelihood/clock；不进wire |
| human side ambiguous/known | 不进入robot assignment；目标allocator独立求解 |
| model/VLM/MANO/reconstruction proposal | 不可作为oracle/GT；只作proposal/auxiliary |

### 7.3 Transparent/non-reconstruction

- 透明RGB中的2D overlap不能编TOUCHING/INSIDE；
- completed depth、mesh、NeRF、inpaint或teacher fill不能给oracle locator depth/axis；
- direct calibrated stereo/depth、marker、force/tactile、native state或人工时序可复核evidence可以；
- FLOW/TRANSFER和target relation无直接/时序/sensor/native依据时packet inadmissible；
- reconstruction或continuous spatial route可作互斥baseline，但它们不再是r1b，也不能引用81.4779-bit容量。

## 8. Exact reachable capacity

### 8.1 单constraint payload

固定一个assignment，VERIFY的单constraint payload数为：

```text
V = 4 primary roles
  * 1 N_A part
  * 604,110,848 locators
  * 22 secondary bundles
  = 53,161,754,624
```

非VERIFY有6个known parts：

```text
N = 6 * V
  = 318,970,527,744
```

### 8.2 Constraint bundle

非VERIFY单constraint有LEFT/RIGHT两类，双constraint有LR/LM/RM三类；不同assignment使pair有固定身份，不做组合除2：

```text
B_nonverify
= 2*N + 3*N^2
= 305,226,592,708,495,562,244,096
```

VERIFY单constraint有L/R/M三类，双constraint有LR/LM/RM三类，以及两个不同MONITOR payload的无序组合：

```text
B_verify
= 3*V + 3*V^2 + C(V,2)
= 9,891,602,541,591,253,221,376
```

非VERIFY phase/effect blocks恰有11个：APPROACH 1、ACQUIRE 1、CONTROLLED 7、REGRASP 1、RELEASE 1。VERIFY另有1个。因此：

```text
S_r1b
= 1                                      # global NULL
  + 11 * B_nonverify
  + B_verify
= 3,367,384,122,335,042,437,906,433

log2(S_r1b)
= 81.47790257814123 bit

ceil(log2(S_r1b))
= 82 bit
```

compact enumerative index需要11 bytes；88-bit container的高6位必须为0，且整数rank必须小于 `S_r1b`。26-byte是可审计field wire的存储宽度，不是208-bit语义容量。

### 8.3 构造式 rank/unrank 复核

先定义 locator rank。位置 `p=((uv-1)*32+(depth-1)) in [0,131071]`。variant：

```text
POINT: vloc = 0
AXIS:  vloc = 1 + (axis-1)                           # 1..512
PATH:  vloc = 1 + 512 + (axis-1)*8 + (extent-1)    # 513..4608

locator_rank = p*4609 + vloc                        # 0..L-1
```

secondary rank：NONE为0；否则从升序roles删除primary，secondary局部编号 `j in [0,2]`，`s=1+j*7+(relation-1)`，得到0..21。

固定assignment的constraint payload按 `(part_index, primary_role, locator_rank, secondary_rank)` 混合进制排名，VERIFY得到0..V-1，非VERIFY得到0..N-1。

bundle blocks按以下顺序拼接：

```text
non-VERIFY:
  single L, single R, pair LR, pair LM, pair RM

VERIFY:
  single L, single R, single M,
  pair LR, pair LM, pair RM, pair MM
```

前六类pair直接用两个assignment-specific payload的笛卡尔rank。MM要求 `q0<q1`，用：

```text
comb_rank(q0,q1)
= q0*(2*V-q0-1)/2 + (q1-q0-1)
```

覆盖0..C(V,2)-1。最后按11个非VERIFY phase/effect blocks、1个VERIFY block与global NULL拼接：

```text
rank(NULL)=0
rank(nonverify h,b)=1 + h*B_nonverify + b      # h=0..10
rank(verify,b)=1 + 11*B_nonverify + b
```

逐层divmod/组合unrank唯一恢复wire。这一双射独立证明第8.2节没有把dense slots相加，也没有因canonical sort双计pair。

## 9. Equal-bit control

令：

```text
M = S_r1b - 1
  = 3,367,384,122,335,042,437,906,432
```

冻结whole-packet permutation：

```text
pi(0)=0

for rank u in 1..M:
  v = u-1
  pi(u)=1 + ((65537*v + 104729) mod M)
```

`gcd(65537,M)=1`，其模逆元是 `1717372581975471236775937`。因此NULL固定，所有非NULL合法状态全域双射。baseline必须执行 valid wire→rank→pi→unrank→同一5-token encoder，且与semantic route共享：

- allocator输入/输出与fail set；
- token数、embedding/head参数、attention masks、FLOPs；
- proprio/text/noise、split decoder gate、cache、数据与训练预算；
- locator/task-profile admission与invalid/overflow处理。

不得做逐字段shuffle，因为那会破坏locator kind、secondary relation与assignment bundle语法。r1a、r1b、standard必须分别使用自身 exact alphabet 的equal-bit control。

## 10. r1a、r1b 与 standard-r1

| 项 | minimal-r1a | **minimal-r1b** | standard-r1 |
|---|---|---|---|
| action horizon semantics | active-only | active-only | active+next1+next2 |
| current timing | 无 | 无 | active/future onset+duration |
| locator | referenced role full-object 4×4 center | 每constraint 64×64×32 anchor + POINT/AXIS/PATH | node center/scale/orientation + part-local 4×4 site |
| concurrent records | exactly1 | 1或2 | 每event最多2 edges |
| focus | primary role | primary role | independent focus with anchor rules |
| actor | 无；allocator adapter可能成隐形通道 | **explicit robot assignment，已计容** | LEFT/RIGHT/OBJECT_OBJECT，混合human/robot literal语义 |
| part | 6 known；VERIFY N/A | 同r1a | 8 known + UNKNOWN |
| relation | one intended 7-class | one intended perconstraint 7-class | before+after，含OVERLAP_2D/UNKNOWN |
| contact mode | 无/派生 | 无/版本化派生 | action-visible |
| outcome | supervisor | supervisor | VERIFY action-visible |
| future | planner-only | planner-only | action-visible next2 |
| field wire | 10 bytes | **26 bytes** | 82 bytes |
| tokens | 5 | **5** | 13 |
| exact states | 1,445,057 | **3,367,384,122,335,042,437,906,433** | `345864043392188289642033658185203030253953267411599196753571073` |
| exact bits | 20.46269497000547 | **81.47790257814123** | 207.74974691951957 |
| min index | 21 bits | **82 bits** | 208 bits |

r1b比r1a多 `61.01520760813575 bit`，主要来自可执行locator、双约束与assignment；仍比standard少 `126.27184434137834 bit`。这些差值是完整grammar的log ratio，不能解释成单字段独立贡献。

兼容关系：

- r1a→r1b不可无损：r1a没有depth/axis/extent、第二约束和公开assignment；只能重新读取robot current evidence再编译；
- standard→r1b也不是截字段：standard的4×4 site不能生成r1b locator，human actor不能生成robot assignment；需目标域重定位和allocator；
- r1b→r1a是many-to-one诊断投影，可丢第二约束/locator细节/assignment，但该投影不允许用于执行；
- 三者schema ID、codec、validator、token embedding、checkpoint互不兼容；v1.1/standard disabled path继续冻结。

## 11. Planner、auxiliary、controller 与 action boundary

### 11.1 Action-visible

仅以下信息进入r1b WAM：

```text
phase, effect
1..2 constraints:
  robot assignment
  primary/focus role, part
  discrete execution locator
  optional secondary role, target relation

+ existing robot proprio/state
+ allowed instruction text
+ diffusion timestep/noisy action
```

### 11.2 Action-invisible

- full eight-kind annotation、future Q≤8、onset/duration/censor；
- human literal actor、human locator/pose/trajectory；
- node observability reason、scale/orientation原字段；
- historical relation_before、observed outcome；
- continuous target pose/ROI/mask/rim/flow/depth completion；
- allocator cost/logits/hidden/receipt；
- confidence、review state、coverage、teacher/proposal、CAPOT/pair ID；
- planner/auxiliary hidden和raw reference。

Contact/controller mode由versioned lookup table从公开 `task profile + text goal args + phase/effect + constraint fields` 唯一派生。若lookup多解，compiler拒绝；不能让allocator或WAM hidden暗选mode。只有paired-world oracle证明同一公开key确实需要两种不相容controller profile时，才另立schema promotion。

### 11.3 Continuous pose/contact 与 challengers

entry/turn/exit pose、local motion window、continuous hand/tool/object轨迹和contact change只作planner/backbone auxiliary或compression probe，stop-gradient且不进action/cache。F-PCD、direct visual servo、standard-r1、22D与full/local continuous trajectory必须是独立route/checkpoint，不能concat到r1b后仍声称81.4779-bit finite route。

## 12. CPU-only 可验证不变量（本轮不实现）

1. 26字节全零是唯一NULL；nonzero header配零constraint或反之均拒绝；
2. slot0 present、slot1 absent时后12字节全零；constraint与locator tokens必须成对mask；
3. phase/effect严格通过12-block表；VERIFY只NONE，其它phase规则不变；
4. 非VERIFY part为六known，VERIFY只N_A；unknown/missing不借N_A/OTHER；
5. POINT/AXIS/PATH分别严格执行axis/extent zero/nonzero规则；
6. uv/depth/axis/extent所有边界值round-trip且reserved values拒绝；
7. locator rank恰覆盖604,110,848 states，无多个wire映同一locator；
8. secondary NONE当且仅当relation N_A；secondary不能等于primary；
9. nonVERIFY assignment bundle只允许L、R、LR、LM、RM；VERIFY另允许M、MM；
10. MM payload必须严格不同并按payload rank升序；其它双constraint按完整sort key升序；
11. 第三constraint、duplicate、same-effector double、role instance多解稳定overflow，不截断；
12. assignment byte进入wire/token；交换L/R只打开对应split action slices，MONITOR不打开slice；
13. 固定公开输入时allocator重复运行byte-identical；ambiguous/infeasible永不随机选择；
14. allocator logits/cost/continuous target/candidate order/receipt改变不能改变action；只有公开assignment byte可改变split path；
15. allocator不能重排state/action维度或修改continuous gains传信；审计hash覆盖公开接口；
16. fixed packet/proprio/text/noise时替换raw image、planner/allocator/aux hidden、human video，pre-safety action `max_abs<=1e-6`；
17. current error只由decoded locator+公开calibration+proprio+verified held transform机械复算；两端实现逐bit/容差一致；
18. missing/reviewed unknown/occlusion/out-of-frame/tracker inactive不变成NULL/PAD/最近bin；
19. completed depth/reconstruction/proposal不能出现在oracle locator provenance；
20. EXHAUSTIVE known no-event才授权NULL；censor只改planner clock/loss；
21. exact `S_r1b`、81.47790257814123、82-bit minimum、26-byte/5-token receipt逐字固定；
22. field wire↔rank覆盖恰好 `0..S_r1b-1`；11-byte超范围rank和高6位非零拒绝；
23. equal-bit pi/inverse全域双射、NULL固定，semantic/equal route allocator和mask完全相同；
24. schema/checkpoint IDs不交叉接受；v1.1、r1a、standard disabled route bitwise不变；
25. `O-LOC-POUR/INSERT/WIPE` paired-world中安全动作不相交的state必须编成不同r1b locator，否则profile No-Go；
26. `O-CONSTRAINT` locked/unlocked必须分别形成monitor/single与active second constraint，或由公开verified sensor state证明相同安全动作；
27. `O-ASSIGN`交换assignment只能改变对应left/right action slice；human actor swap不改变robot packet；
28. O-FUTURE/O-TIMING/O-OUTCOME swap在fixed current evidence下不改变r1b action；否则停止对应删除主张；
29. task profile量化误差超过预注册公差时必须resolution insufficient，不得最近bin继续执行；
30. 若只有continuous correction/action-visible reconstruction能通过oracle，r1b finite sufficiency判失败。

## 13. 当前风险与 Go/No-Go

### 13.1 主要风险

1. **64×64×32仍可能不足**：细port、pipette depth与tight cap轴公差可能小于量化误差；这是直接No-Go条件，不由更大模型解决。
2. **axis只有方向，没有完整6D orientation**：有roll-sensitive工具时同axis不同roll仍会collision；若proprio/held transform和part ontology不能派生roll，需独立paired-world promotion或continuous challenger。
3. **PATH是直线axis+extent**：弯曲wipe region、多段轨迹或避障corridor不能表达；F-PCD/direct servo可能必要。
4. **最多两条约束**：三点稳定、双手+fixture+object relation可能overflow；不得删除这些test cases。
5. **depth evidence coverage**：透明/遮挡场景若没有direct sensor/native depth，r1b会保守拒绝；不可用completion提高假compile coverage。
6. **MONITOR_ONLY误用**：不能把未知support写成monitor；必须每anchor有sensor/native证据。
7. **allocator泄漏**：assignment虽显式，若allocator还控制continuous target/gain/order，容量主张仍失效。

### 13.2 Gate

当前只批准 CPU contract 与无训练 paired-world oracle设计，不批准模型训练或真机执行。r1b成为四任务oracle candidate必须同时满足：

- `O-LOC-POUR/INSERT/WIPE` 中所有原r1a collision均被不同locator消除；
- 量化后的reference first-8 action仍满足预注册位置/轴/extent公差；
- `O-CONSTRAINT`对single/double/MONITOR admission零静默遗漏；
- allocator每条合法constraint唯一assignment或明确abstain，且无hidden channel；
- 相对full/native trajectory的安全动作差距在预注册容差内；
- future/timing/mode/outcome没有独立oracle gap；若有，只promotion通过对应paired-world test的字段。

任一核心collision仍存在，r1b No-Go。此时保留event/constraint annotation benchmark，比较standard、F-PCD/direct visual servo与continuous route，不追加unknown、confidence、continuous residual或第三constraint保护headline。

## 14. 最终建议

冻结 `sm_fet_hard_grammar_v2_minimal_r1b` 为 r1a 与 standard-r1 之间的唯一可实现折中：active-only、最多两条并发constraint、64×64×32 camera anchor、POINT/AXIS/PATH locator、primary即focus、known part/effect/target relation，以及**显式、已计容的robot effector assignment**。其 exact action alphabet 为 `3,367,384,122,335,042,437,906,433` states / `81.47790257814123 bit`，26-byte inspectable wire、82-bit minimum index与5 tokens。

它修复的是已被反例证明的r1a执行碰撞，不恢复没有必要性证据的future/timing/human actor/outcome。最重要的审计边界是：allocator只能输出wire里的硬assignment enum，不能再通过continuous target、维度置换、gain、hidden或cache成为第二条动作通道。下一步止于CPU rank/grammar审计与无训练paired-world oracle；在定位公差、双约束和assignment gate通过前，不实现WAM、不训练、不声称控制充分。
