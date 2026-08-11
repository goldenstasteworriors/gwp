# SM-FET `minimal-r1d` structural freeze addendum

- 日期：2026-08-12
- 作用：关闭 `minimal-r1d` 最终独立复审剩余六项 exact-contract 缺口
- 依赖主合同：`docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`，commit `5019c1828cf17e169c51ecf6772ad16b23d51c2d`
- 性质：只补规范和未来 CPU golden vectors；不实现、不训练、不使用 GPU、不修改 canonical

## 0. 规范地位与不变量

本 addendum 对主合同中的歧义具有规范优先级，但**不新增或删除任何 core field、token、phase/effect block、constraint topology、locator kind或code**。因此以下常量保持逐字不变：

```text
wire bytes                 = 32
hard token slots           = 5
P                          = 131,072
L_nonverify                = 70,934,200,320
L_verify                   = 2,214,723,584
N                          = 43,695,467,397,120
V                          = 194,895,675,392
B_nonverify                = 14,319,704,032,895,892,051,583,303,680
B_verify                   = 151,937,297,146,405,128,765,440
|G_struct_r1d|             = 157,516,896,299,151,958,972,545,105,921
log2 |G_struct_r1d|        = 96.99142134161319 bit
minimum integer width      = 97 bit
compact container          = 13-byte unsigned little-endian
|G_profile(g1,... )|       = UNDEFINED / NO-GO
profile bits               = UNDEFINED / NO-GO
```

若未来实现发现必须改变上述任一alphabet或topology，本 addendum不授权静默修改；必须升schema并重新计容。

## 1. Global rank/unrank：唯一 canonical codec

### 1.1 三层canonical order

只有三层次序进入global rank：

1. **event/header order**：唯一NULL先于所有positive event；positive event按主合同固定的11个nonVERIFY phase/effect blocks `h=0..10`，最后是VERIFY；
2. **topology block order**：按§1.3和§1.4逐项顺序；
3. **constraint order**：different-type pair严格按block名字从左到右放`constraint0,constraint1`；same-type pair按payload rank `q0<q1`。

这是active-only、`Q_action=1`的codec；一个packet没有多个event可再排序。Planner的episode queue不进32-byte wire，也不参与global rank。若同一anchor存在两个互斥的phase/effect解释或超过两条required constraints，结果是no packet，不得靠locale、confidence、allocator score或arrival order选一个。

Header order固定为：

```text
h=0   APPROACH_ALIGN+NONE
h=1   ACQUIRE+NONE
h=2   CONTROLLED_CONTACT+NONE
h=3   CONTROLLED_CONTACT+TRANSLATE
h=4   CONTROLLED_CONTACT+ROTATE
h=5   CONTROLLED_CONTACT+ARTICULATE
h=6   CONTROLLED_CONTACT+FLOW
h=7   CONTROLLED_CONTACT+DEFORM
h=8   CONTROLLED_CONTACT+TRANSFER
h=9   REGRASP+NONE
h=10  RELEASE+NONE
h=11  VERIFY+NONE
```

不存在其它phase/effect pair的rank。

### 1.2 Payload与locator rank/unrank

令`p=((uv-1)*32+(depth-1))`。Wire UV严格为：

```text
uv = 1 + 64*row + col, row,col in 0..63
depth in 1..32
```

Locator variant rank：

```text
POINT q = 0
AXIS  q = 1+(axis-1)
FRAME q = 1+512+(axis-1)*32+(roll-1)
PATH  q = 1+512+512*32
          +(((axis-1)*32+(roll-1))*4+(primitive-1))*8
          +(extent-1)
```

nonVERIFY取`locator_rank=p*541185+q`；VERIFY只允许POINT/AXIS/FRAME并取dense `locator_rank=p*16897+q`。Unrank先用`divmod(locator_rank,541185或16897)`得到`p,q`，再用`divmod(p,32)`得到`uv-1,depth-1`；PATH按`extent,primitive,roll,axis`由低到高连续`divmod`。任何商或余数越界均`INVALID_LOCATOR_RANK`，不截断、不取模。

Secondary rank：NONE/N_A为0。否则从升序roles `[SOURCE,TOOL,DESTINATION,SUPPORT]`删除primary，secondary在剩余三项中的index为`j=0..2`，relation code为`r=1..7`：

```text
secondary_rank = 1+j*7+(r-1)       # 1..21
```

Payload rank为：

```text
payload_rank = (((part_index*4+primary_role_index)*L_h
                 +locator_rank)*22
                +secondary_rank)
```

其中nonVERIFY `part_index=0..6,L_h=L_nonverify`；VERIFY `part=N_A,part_index=0,L_h=L_verify`。Unrank严格反向连续`divmod(...,22)`、`divmod(...,L_h)`、`divmod(...,4)`；最终part商越界即`INVALID_PAYLOAD_RANK`。

### 1.3 nonVERIFY bundle-local rank

令`n=N`。所有区间均左闭右开；`local`从0开始：

| block | offset | size | local |
|---|---:|---:|---|
| LO | `0` | `n` | `q_LO` |
| RO | `n` | `n` | `q_RO` |
| LO-RO | `2n` | `n²` | `q_LO*n+q_RO` |
| LO-RS | `2n+n²` | `n²` | `q_LO*n+q_RS` |
| LS-RO | `2n+2n²` | `n²` | `q_LS*n+q_RO` |
| LO-ZM | `2n+3n²` | `n²` | `q_LO*n+q_ZM` |
| RO-ZM | `2n+4n²` | `n²` | `q_RO*n+q_ZM` |
| LO-LS | `2n+5n²` | `n²` | `q_LO*n+q_LS` |
| RO-RS | `2n+6n²` | `n²` | `q_RO*n+q_RS` |
| RO-RO | `2n+7n²` | `C(n,2)` | `comb_rank(q0,q1;n)` |

`b=offset+local`，且必须`0<=b<B_nonverify`。Different-type unrank对`local`做`divmod(local,n)`，商对应block左侧constraint、余数对应右侧constraint。

Same-type：

```text
A(i;n)=i*(2*n-i-1)/2
comb_rank(q0,q1;n)=A(q0;n)+(q1-q0-1)
```

Unrank不得用浮点根公式：在整数域二分查找唯一`q0 in [0,n-2]`满足`A(q0;n)<=local<A(q0+1;n)`，再令`q1=q0+1+local-A(q0;n)`。若不存在唯一解或不满足`q0<q1<n`，返回`INVALID_COMB_RANK`。

生产常量下每个block的inclusive local `b`端点为：

| block | first b | last b |
|---|---:|---:|
| LO | 0 | 43,695,467,397,119 |
| RO | 43,695,467,397,120 | 87,390,934,794,239 |
| LO-RO | 87,390,934,794,240 | 1,909,293,871,052,864,258,719,088,639 |
| LO-RS | 1,909,293,871,052,864,258,719,088,640 | 3,818,587,742,105,641,126,503,383,039 |
| LS-RO | 3,818,587,742,105,641,126,503,383,040 | 5,727,881,613,158,417,994,287,677,439 |
| LO-ZM | 5,727,881,613,158,417,994,287,677,440 | 7,637,175,484,211,194,862,071,971,839 |
| RO-ZM | 7,637,175,484,211,194,862,071,971,840 | 9,546,469,355,263,971,729,856,266,239 |
| LO-LS | 9,546,469,355,263,971,729,856,266,240 | 11,455,763,226,316,748,597,640,560,639 |
| RO-RS | 11,455,763,226,316,748,597,640,560,640 | 13,365,057,097,369,525,465,424,855,039 |
| RO-RO | 13,365,057,097,369,525,465,424,855,040 | 14,319,704,032,895,892,051,583,303,679 |

### 1.4 VERIFY bundle-local rank

令`v=V`：

| block | offset | size | local |
|---|---:|---:|---|
| LS | `0` | `v` | `q_LS` |
| RS | `v` | `v` | `q_RS` |
| ZM | `2v` | `v` | `q_ZM` |
| LS-RS | `3v` | `v²` | `q_LS*v+q_RS` |
| LS-ZM | `3v+v²` | `v²` | `q_LS*v+q_ZM` |
| RS-ZM | `3v+2v²` | `v²` | `q_RS*v+q_ZM` |
| ZM-ZM | `3v+3v²` | `C(v,2)` | `comb_rank(q0,q1;v)` |
| RS-RS | `3v+3v²+C(v,2)` | `C(v,2)` | `comb_rank(q0,q1;v)` |

生产常量下inclusive local `b`端点：

| block | first b | last b |
|---|---:|---:|
| LS | 0 | 194,895,675,391 |
| RS | 194,895,675,392 | 389,791,350,783 |
| ZM | 389,791,350,784 | 584,687,026,175 |
| LS-RS | 584,687,026,176 | 37,984,324,287,088,521,379,839 |
| LS-ZM | 37,984,324,287,088,521,379,840 | 75,968,648,573,592,355,733,503 |
| RS-ZM | 75,968,648,573,592,355,733,504 | 113,952,972,860,096,190,087,167 |
| ZM-ZM | 113,952,972,860,096,190,087,168 | 132,945,135,003,250,659,426,303 |
| RS-RS | 132,945,135,003,250,659,426,304 | 151,937,297,146,405,128,765,439 |

Unrank与§1.3相同，只把`n`替换为`v`。

### 1.5 Global rank、wire canonicality与invalid rank

```text
rank(NULL)=0
rank(nonVERIFY h,b)=1+h*B_nonverify+b, h in 0..10
rank(VERIFY,b)=1+11*B_nonverify+b
```

Global unrank先拒绝`r<0`或`r>=|G_struct_r1d|`。`r=0`唯一解码成32个零bytes。其余令`x=r-1`；若`x<11*B_nonverify`，`divmod(x,B_nonverify)`得到`h,b`；否则固定VERIFY并取`b=x-11*B_nonverify`。之后按唯一half-open block interval反解topology、payload和wire。

13-byte输入必须长度恰13，按unsigned little-endian解释。`byte[12]&0xFE != 0`先报`RESERVED_BITS`；保留位合法但整数大于等于结构状态数时报`INVALID_GLOBAL_RANK`。禁止modulo、saturation、fallback NULL或best-effort decode。32-byte field wire只有通过field range、PAD、phase/effect、locator kind、function/assignment、bundle、secondary/relation、sort/duplicate检查后才可rank；`rank(unrank(r))=r`且`unrank(rank(w))=w`必须逐byte成立。

## 2. POINT/AXIS与VERIFY STABILIZE的精确语义

### 2.1 Locator是目标流形，不一律是完整SE(3)

令`T=[R,t] in SE(3)`，`e_z=(0,0,1)`：

```text
projection_POINT(T) = t
projection_AXIS(T)  = (t, R*e_z)
projection_FRAME(T) = T
```

POINT wire只指定`g_POINT=p_C*`；AXIS指定`g_AXIS=(p_C*,z_C*)`；FRAME才指定唯一`T_C_P*`。因此主合同的完整变换方程只直接适用于FRAME/PATH。POINT/AXIS合法目标集合分别是：

```text
M_POINT = {T_C_P | projection_POINT(T_C_P)=g_POINT}
M_AXIS  = {T_C_P | projection_AXIS(T_C_P)=g_AXIS}
```

若P通过公开`T_E_P`绑定到effector E，则effector目标流形为：

```text
M_E(K)={T_C_E | projection_K(T_C_E @ T_E_P)=g_K}, K in {POINT,AXIS}
```

`T_E_P`只能来自immutable calibrated part transform或`public_execution_state_v1`的held transform。缺失、多解binding或超过uncertainty阈值时runtime拒绝。POINT缺orientation、AXIS缺roll不是UNKNOWN，因为这些自由度按kind明确不受约束；不得由allocator、receipt metadata、pair geometry或private completion补入。

Action expert可在流形自由度内产生动作，但只能读取主合同允许的hard tokens、complete public state、structured text、noise/timestep。若一个task的安全动作需要额外orientation/roll，static profile必须改用FRAME或删除该tuple；不能把POINT/AXIS当作压缩的hidden FRAME。

Function语义固定为：

- OPERATE：允许从流形外接近并实现目标effect；
- STABILIZE：anchor时流形已满足，只对扰动做保持/恢复，不发起新的task relation transition；
- MONITOR：观测对应流形或relation，own actuator slice恒零，可条件化同bundle physical slices。

不改变structural alphabet的kind/function矩阵为：

| event | OPERATE | STABILIZE | MONITOR |
|---|---|---|---|
| nonVERIFY | POINT, AXIS, FRAME, PATH | POINT, AXIS, FRAME, PATH | POINT, AXIS, FRAME, PATH |
| VERIFY | illegal | POINT, AXIS, FRAME | POINT, AXIS, FRAME |

这张表只说明wire structural legality；bundle legality仍由LO/LS/RO/RS/ZM topology决定，executable legality还必须通过static profile。NonVERIFY STABILIZE+PATH只表示随同bundle OPERATE在同一公开path上维持已建立约束，MONITOR+PATH只观察该公开path corridor；若没有唯一公开chunk progress或first-8安全映射，profile删除，而不是改用private progress。

### 2.2 VERIFY的`N_A+STABILIZE`

VERIFY没有controlled part，故不得调用held-part方程。对VERIFY中的LS/RS，locator直接约束assigned effector的公开control frame：

```text
POINT -> maintain effector origin at p_C*
AXIS  -> maintain effector origin and directed z-axis at (p_C*,z_C*)
FRAME -> maintain full effector frame T_C_E*
```

其唯一控制模式是`HOLD_EXISTING`。Runtime validator必须用同一anchor的public state、camera calibration和profile tolerance证明目标在anchor时已满足；不满足、无法由public state复算或需要held object/part-specific transform时，tuple拒绝而不是把VERIFY变成OPERATE。Profile tolerance是immutable、hash-matched的validator常量，不作为逐sample动作输入。

VERIFY中的ZM同样只允许POINT/AXIS/FRAME，但它只观察role-level事实且own slice为零。合法topology仍恰为：

```text
single: LS | RS | ZM
pair:   LS-RS | LS-ZM | RS-ZM | ZM-ZM(q0<q1) | RS-RS(q0<q1)
```

每条constraint可独立选择POINT/AXIS/FRAME。RS-RS还必须具有不同payload、不同或可共享的finite resource claims并通过static compatibility。VERIFY PATH、VERIFY OPERATE、LS-LS、三条constraints、unsatisfied STABILIZE均拒绝。以上只裁剪`G_profile/R_runtime`，不改变`G_struct_r1d`。

## 3. RIGHT resource target identity

### 3.1 Target identity不是assignment或continuous target

`RIGHT_OMNIPICKER`只表示effector assignment，不是resource target。Static profile对每条physical OPERATE/STABILIZE constraint和每个相关resource生成唯一`resource_claim_v1`：

```text
resource_id: finite uint16 profile enum
claim_mode: EXCLUSIVE_TARGET | HOLD_SAME_TARGET | READ_ONLY
target_namespace:
  APERTURE_ACTUATION | WRIST_POSE | HELD_ENTITY | STABILIZED_ENTITY
  | MONITORED_ENTITY | SHARED_WBC | SELF_COLLISION
target_selector:
  CONTROLLER_ENUM | LOCATOR_POINT | LOCATOR_AXIS | LOCATOR_FULL
  | ENTITY_LOCATOR | CONSTANT
target_key: finite byte string produced by the selected rule
```

每个selector的输出逐字固定：

```text
CONTROLLER_ENUM = uint16_le controller_target_enum
LOCATOR_POINT   = uint32_le translation rank p
LOCATOR_AXIS    = uint32_le p || uint16_le axis
LOCATOR_FULL    = uint64_le locator rank
ENTITY_LOCATOR  = primary_role:u8 || part:u8 || secondary:u8 || relation:u8
                  || kind:u8 || uint64_le locator rank
CONSTANT        = uint16_le immutable profile constant
```

`resource_target_id`是 `(resource_id,target_namespace,target_selector,target_key)` 的bytewise tuple。映射函数`F(p,t,canonical wire constraint,resource_id)`只能读immutable profile `p`、fixed task/text stratum`t`和canonical wire payload；返回恰一个上述tuple或`UNKNOWN_NOT_ADMITTED`。它不得读scene instance、continuous controller goal、receipt、reachability、contact、WBC值、allocator score或action output。Selector、controller enum表、resource enum表都进入static profile hash。

### 3.2 Aperture actuation与entity identity严格分域

- 只有确实命令或保持OmniPicker aperture actuator的constraint才声明`target_namespace=APERTURE_ACTUATION`；其identity必须由有限`CONTROLLER_ENUM`给出。若目标需要未离散的continuous aperture值，当前profile拒绝。
- 两个不同`APERTURE_ACTUATION resource_target_id`恒mutex；相同identity合并为一个resource claim。相同claim不自动删除两个语义不同的constraints，只有整条canonical payload完全相同才触发wire `DUPLICATE`。
- RIGHT上的STABILIZED_ENTITY/HELD_ENTITY identity描述被保持的role/part/locator，不等于aperture actuation target。它可以同时另声明wrist或aperture HOLD claim，但不能用entity identity伪装aperture target。
- MONITOR_ONLY的MONITORED_ENTITY只声明READ_ONLY `MONITOR_EVIDENCE`，永不声明RIGHT aperture/wrist actuator ownership。
- 不同WRIST_POSE target默认mutex；只有profile证明resource channel disjoint且列入有限compatibility table时才兼容。Runtime reachability/collision只能继续删除，不能改写`F`。

每条OPERATE必须至少有一个`EXCLUSIVE_TARGET` actuator claim；每条STABILIZE必须至少有一个`HOLD_SAME_TARGET`或`EXCLUSIVE_TARGET` actuator claim。全NO_CLAIM、仅READ_ONLY或UNKNOWN template均不admit physical constraint。由此RO-RS、RO-RO、RS-RS的结构可表示性不会被allocator变成隐藏action channel，真实G1 compound仍保持UNDEFINED/No-Go直到profile标定。

## 4. Human shared semantic JSON与structured text exact schema

### 4.1 非NULL shared key

一个非NULL JSON value必须是object，且top-level key set**恰好**为：

```text
schema_id, task_ontology_id, task_family, phase, effect,
event_focus_role, constraint_roster_coverage, constraints
```

类型与enum：

```text
schema_id = exact string "sm_fet_r1d_shared_semantic_key_v1"
task_ontology_id = string matching ^[a-z0-9._/-]{1,64}$
task_family in {"POUR","INSERT","PIPETTE","CAP","WIPE"}
phase in {"APPROACH_ALIGN","ACQUIRE","CONTROLLED_CONTACT",
          "REGRASP","RELEASE","VERIFY"}
effect in {"NONE","TRANSLATE","ROTATE","ARTICULATE",
           "FLOW","DEFORM","TRANSFER"}
event_focus_role in {"MANIPULATED_OR_SOURCE","TOOL",
                     "RECEPTACLE_OR_DESTINATION","FIXTURE_OR_SUPPORT"}
constraint_roster_coverage = exact string "EXHAUSTIVE"
constraints = array length 1..2
```

每个constraint必须是object，key set恰好为：

```text
purpose, primary_role, primary_part, secondary_role, target_relation
```

其enum：

```text
purpose in {"EFFECT_GOAL","SUPPORT_REQUIREMENT","OBSERVATION_REQUIREMENT"}
primary_role in {"MANIPULATED_OR_SOURCE","TOOL",
                 "RECEPTACLE_OR_DESTINATION","FIXTURE_OR_SUPPORT"}
primary_part in {"BODY","HANDLE_OR_GRASP_FEATURE","LID_CAP",
                 "OPENING_OR_RIM","TOOL_TIP","ACTUATOR_OR_PLUNGER",
                 "WORKING_FACE_OR_PAD","N_A"}
secondary_role in {"NONE","MANIPULATED_OR_SOURCE","TOOL",
                   "RECEPTACLE_OR_DESTINATION","FIXTURE_OR_SUPPORT"}
target_relation in {"N_A","SEPARATE","NEAR","TOUCHING","INSIDE",
                    "CONTAINS","ALIGNED_TO_RIM","SUPPORTED_BY"}
```

每个字段恰选一个完整string；含`|`的pipe-string不是enum简写而是非法值。禁止extra key、numeric enum、null JSON value和Unicode同形替代。Phase/effect、part、secondary/relation、constraint sort、duplicate、focus和EXHAUSTIVE规则沿用主合同。

### 4.2 NULL shared key

NULL仍使用相同top-level exact key set，且逐字段满足：

```text
schema_id = "sm_fet_r1d_shared_semantic_key_v1"
task_ontology_id = one actual valid ontology identifier
task_family = one actual member of the five-string enum
phase = "NULL"
effect = "N_A"
event_focus_role = "NONE"
constraint_roster_coverage = "EXHAUSTIVE"
constraints = []
```

这里`"NULL"`、`"N_A"`和`"NONE"`只在NULL schema分支合法；它们不是非NULL phase/effect/focus候选。Unknown、missing、occluded、partial roster不能使用NULL分支。

### 4.3 Canonical bytes与真实示例

Schema validation、constraint ordinal sort、focus复算和duplicate检查均通过后才做RFC 8785 canonicalization，再对UTF-8 bytes做SHA-256。以下是一个真实单值示例；其中没有任何pipe-string：

```json
{"constraint_roster_coverage":"EXHAUSTIVE","constraints":[{"primary_part":"BODY","primary_role":"MANIPULATED_OR_SOURCE","purpose":"EFFECT_GOAL","secondary_role":"RECEPTACLE_OR_DESTINATION","target_relation":"INSIDE"}],"effect":"TRANSFER","event_focus_role":"MANIPULATED_OR_SOURCE","phase":"CONTROLLED_CONTACT","schema_id":"sm_fet_r1d_shared_semantic_key_v1","task_family":"POUR","task_ontology_id":"biochem4.v1"}
```

上述bytes已按RFC 8785 key order书写；实现仍必须canonicalize而不是依赖输入key顺序。Robot/human geometry、assignment、instance、outcome和profile仍禁止进入key与pair cost。

该示例的expected canonical UTF-8长度为412 bytes，expected SHA-256为：

```text
dafc4e176b925bb553fded9b1f9fcd1bc96542b3c8c5288df1be641ea0278fab
```

### 4.4 Structured text同步消歧

Text object exact key set为：

```text
schema_id, task_family, goal_symbol, quantity_class,
signed_direction, success_criterion
```

```text
schema_id = "sm_fet_r1d_text_goal_v1"
task_family in {"POUR","INSERT","PIPETTE","CAP","WIPE"}
goal_symbol matches ^[a-z0-9._/-]{1,64}$ and must resolve in the
  immutable ontology registry to semantic_only=true
quantity_class in {"N_A","LOW","MEDIUM","HIGH"}
signed_direction in {"N_A","POSITIVE","NEGATIVE"}
success_criterion in {"RELATION","DOSE","TORQUE","CLEAN","POSE"}
```

Pipe-string、extra key、unregistered goal、或registry中可解析为pixel/depth/axis/roll/path/assignment/trajectory的alias均拒绝。Ontology registry hash属于fixed text/task stratum receipt；它不能逐sample选择profile或controller。

## 5. FRAME/PATH metric codebook补充合同

### 5.1 W/H唯一来源

Metric manifest exact key set新增明确值而非自然语言占位：

```text
schema_id = "sm_fet_r1d_metric_codebook_manifest_v1"
generator_version = "sm_fet_r1d_codebook_generator_v1"
locator_width_px: uint32, >0
locator_height_px: uint32, >0
source_calibration_sha256: 64 lowercase hex chars
preprocess_id: string matching ^[a-z0-9._/-]{1,64}$
camera_convention = "OPENCV_RIGHT_DOWN_FORWARD"
pixel_center_rule = "HALF_PIXEL_NORMALIZED_V1"
uv_grid = [64,64]
depth_bins = 32
axis_bins = 512
roll_bins = 32
path_primitives = 4
extent_bins = 8
edge_tie_rule = "INTERNAL_TO_HIGH_LAST_RIGHT_CLOSED_V1"
float_policy = "IEEE754_BINARY64_RNE_NO_FMA_V1"
angle_quantizer = "CORRECTLY_ROUNDED_BINARY64_ATAN2_ASIN_V1"
depth_unit = "METER_CAMERA_Z"
extent_unit = "METER"
axis_generator = "r1d_axis_az32_el16_equal_angle_v1"
axis_vertical_epsilon_squared_hex = "0x1p-80"
roll_generator = "r1d_roll32_seed_min_absdot_v1"
path_generator = "r1d_line_arc90_raster3_v1"
table_layout = "sm_fet_r1d_tables_le_f64_v1"
```

`W=locator_width_px,H=locator_height_px`是进入locator encoder的**预处理后raster**尺寸，不是任意raw sensor尺寸。`preprocess_id`唯一规定crop/resize/undistort及其顺序；`tables.bin`中的`fx,fy,cx,cy`必须已经对应这个同一raster。任一不匹配或只有calibration hash而无显式W/H均`CODEBOOK_W_H_MISSING_OR_MISMATCH`。W/H不改变64×64 normalized alphabet，故不改结构容量。

### 5.2 Metric ranges与边界

Depth表含严格递增finite binary64 `z_edge[0..32]`和32个center；admitted range为闭区间`[z_edge[0],z_edge[32]]`。内部cell为`[edge_i,edge_{i+1})`，最后一格右闭；精确落内部edge取高index，range外、NaN、Inf和negative zero均UNKNOWN/no packet。Center必须满足对应cell，且只有codebook center用于decode。

Extent同理：严格递增positive binary64 `d_edge[0..8]`、8 centers，range为`[d_edge[0],d_edge[8]]`，内部edge取高档、最后右闭。真实G1数值缺失，因此当前metric artifact/hash与executable profile继续UNDEFINED；本 addendum不伪造range。

UV输入使用pixel center：`u=(x+0.5)/W,v=(y+0.5)/H`。Normalized range为`[0,1]`，64格内部edge取高档、1归最后一档；实际合法pixel center严格位于(0,1)。越界或support跨越多个会改变safe action的cell即UNKNOWN。

### 5.3 Axis epsilon与全部tie rule

Axis输入先验证finite且非零，再按manifest float policy正规化为binary64单位向量`(x,y,z)`。非vertical角度使用manifest指定的correctly-rounded binary64 `atan2/asin`；实现不满足该数学合同就必须通过materialized golden vectors证明逐码一致，否则不conformant。Vertical判定不用未定义的`hypot`：按binary64 round-to-nearest-ties-to-even、禁止FMA，计算：

```text
rho2 = round64(round64(x*x)+round64(y*y))
epsilon2 = exact binary64 0x1p-80
vertical iff rho2 <= epsilon2
```

恰等于epsilon走vertical分支；取`a=0`，e由z符号取顶部或底部elevation cell。非vertical使用主合同azimuth/elevation edges：azimuth wrap到`[-pi,pi)`，`pi`归`-pi`；内部edge取高index；elevation内部edge也取高index。Materialized axis centers而非运行时三角函数结果进入`tables.bin`并用于decode。

Roll seed比较`abs(dot(ex,z_d)),abs(dot(ey,z_d)),abs(dot(ez,z_d))`，最小者胜，exact tie为`ex<ey<ez`。Roll wrap到`[0,2pi)`，内部edge取高index，`2pi`归0。PATH vertex tangent、final tangent、extent edge和RASTER segment tie沿用主合同；任何无法唯一应用的symmetry/tie为UNKNOWN而不是自选。

### 5.4 Artifact bytes/checksum

Manifest必须恰含§5.1逐项列出的keys，不允许extra key或同义duplicate。`tables.bin`固定为little-endian binary64顺序：

```text
fx,fy,cx,cy;
z_edge[33],z_center[32];
azimuth_edge[33],elevation_edge[17];
axis_center[512][3];
roll_edge[33];
d_edge[9],d_center[8];
path_pi_over_2;
raster_vertices[6][2] in the main-contract traversal order
```

`path_pi_over_2`和所有table值均为materialized binary64 bytes；LINE端点及ARC正负号由`path_generator`常量定义，不另存可冲突副本。ARC运行时sin/cos遵循与angle quantizer相同的correctly-rounded binary64要求；RASTER只读materialized vertices。

Checksum保持：

```text
SHA256(RFC8785(manifest UTF-8) || 0x00 || tables.bin)
```

W/H、epsilon2、float policy、metric端点或preprocess任一改变均改变hash并要求新receipt；禁止沿用旧hash。

## 6. 未来CPU golden gates与expected结果

本节冻结**未来测试输入与expected**，本轮不写checker或测试脚本，也不把未执行说成已通过。

### 6.1 Tiny topology exhaustive

只对bundle/global层建立toy cardinality `n=3,v=2`，不改变production schema：

```text
B_nonverify_toy = 2*3 + 7*3^2 + C(3,2) = 72
B_verify_toy    = 3*2 + 3*2^2 + 2*C(2,2) = 20
G_toy           = 1 + 11*72 + 20 = 813
```

Expected nonVERIFY local intervals：LO 0..2；RO 3..5；七个pair依次6..14、15..23、24..32、33..41、42..50、51..59、60..68；RO-RO为69..71，且`(0,1)->69,(0,2)->70,(1,2)->71`。

Expected VERIFY：LS 0..1；RS 2..3；ZM 4..5；LS-RS 6..9；LS-ZM 10..13；RS-ZM 14..17；ZM-ZM 18；RS-RS 19。Global：NULL=0；h0=1..72；h1=73..144；h10=721..792；VERIFY=793..812。穷举必须无洞、无重、共813 states，且每一state左右round-trip。

### 6.2 Production block/global boundary vectors

§1.3/§1.4的每个first/last b都必须分别unrank后再rank回原整数；其相邻`first-1,last+1`必须落到精确前/后block或越界。Global expected：

| meaning | rank | 13-byte LE hex |
|---|---:|---|
| NULL | 0 | `00000000000000000000000000` |
| h0 first | 1 | `01000000000000000000000000` |
| h0 last | 14,319,704,032,895,892,051,583,303,680 | `000038771cef3b1747fb442e00` |
| h1 first | 14,319,704,032,895,892,051,583,303,681 | `010038771cef3b1747fb442e00` |
| h10 last | 157,516,744,361,854,812,567,416,340,480 | `0000681f394693ff0dccf6fc01` |
| VERIFY first | 157,516,744,361,854,812,567,416,340,481 | `0100681f394693ff0dccf6fc01` |
| global last | 157,516,896,299,151,958,972,545,105,920 | `0000c8e0932a25893aecf6fc01` |

Expected invalid：rank=`|G_struct|`报`INVALID_GLOBAL_RANK`；`2^97-1`同报该错；任一`byte[12]&0xFE!=0`报`RESERVED_BITS`；12或14 bytes报`LENGTH`；不得decode成NULL。

### 6.3 Locator/payload boundary vectors

至少覆盖：每个kind的q首尾、translation `p=0`与`p=P-1`、每个secondary rank 0..21、part/role首尾。Expected q区间：POINT=0；AXIS=1..512；FRAME=513..16,896；PATH=16,897..541,184。VERIFY q=16,897立即`VERIFY_PATH_FORBIDDEN`。Production rank/unrank对全block boundary加固定seed大整数样本运行，不使用浮点combinadic inverse。

### 6.4 Bad wire/profile vectors

固定expected error priority和结果：

| vector | expected |
|---|---|
| positive constraint出现uv=0/depth=0 | `FIELD_RANGE_OR_PAD` |
| uv=4097、depth=33、axis=513、roll=33、primitive=5、extent=9 | `FIELD_RANGE` |
| POINT带非零axis；AXIS带非零roll；FRAME带primitive；PATH缺extent | `LOCATOR_KIND` |
| VERIFY+PATH | `VERIFY_PATH_FORBIDDEN` |
| LEFT_RH56+MONITOR或MONITOR_ONLY+OPERATE | `FUNCTION_ASSIGNMENT` |
| different-type pair交换wire slots | `SORT` |
| same-type q0=q1 | `DUPLICATE` |
| same-type q0>q1 | `SORT` |
| secondary NONE但relation非N_A，或secondary=primary | `SECONDARY_RELATION` |
| 非法phase/effect pair | `HEADER_BLOCK` |
| physical O/S没有actuator claim | `PROFILE_NO_ACTUATOR_CLAIM` |
| claim template/target identity UNKNOWN | `PROFILE_UNKNOWN_NOT_ADMITTED` |
| distinct APERTURE_ACTUATION target IDs并发 | `PROFILE_RESOURCE_MUTEX` |
| runtime receipt尝试改static target ID | `RECEIPT_LAYER_VIOLATION` |

### 6.5 JSON/text alias golden vectors

以下均expected reject：pipe-string enum、numeric enum、extra key、constraint未canonical sort、duplicate tuple、focus不等于派生值、partial roster伪NULL、text goal未注册、goal registry条目带robot locator/trajectory resolver。推荐固定错误码：`JSON_ENUM`、`JSON_EXTRA_KEY`、`JSON_SORT`、`JSON_DUPLICATE`、`JSON_FOCUS`、`NULL_NOT_EXHAUSTIVE`、`TEXT_UNKNOWN_GOAL`、`TEXT_LOCATOR_ALIAS`。

同一合法JSON随机重排object keys后，RFC 8785 bytes/hash必须相同；constraints array次序不是object-key次序，未按ordinal排序必须拒绝而非自动偷偷排序。Human/robot geometry互换而semantic fields固定时，semantic key与strict pair cost必须不变。

### 6.6 Public-state/matched-control golden vectors

以下canonical binary变体均expected reject：NaN、Inf、negative zero、非canonical quaternion sign、held_valid=0但后续payload非零、FIXED camera mode但transform非零、stale timestamp、hash不符。错误码分别固定为`STATE_NONFINITE`、`STATE_NEGATIVE_ZERO`、`STATE_QUATERNION_CANONICAL`、`STATE_PAD`、`STATE_CAMERA_MODE`、`STATE_STALE`、`STATE_HASH`。

Matched invariant future test：固定canonical packet、public-state bytes/hash、structured text bytes/hash、noise/timestep，交换receipt metadata、allocator score/order/private history，expected learned pre-safety action逐bit或按预注册determinism tolerance相同；改变public held transform时允许action变化，但所有route必须收到同一新state/hash。MONITOR token可改变其它physical slice，own actuator ownership mask始终全零。

### 6.7 Capacity/checksum constants gate

Future checker必须assert§0所有整数、`bit_length(|G|-1)=97`、`gcd(65537,|G|-1)=1`及inverse：

```text
138,514,980,096,989,891,931,988,754,433
```

Axis table必须512个finite、unit-tolerance内、互异center；FRAME正交右手；manifest W/H与intrinsics raster一致；metric edges严格递增；checksum按§5.4复算。真实metric tables不存在时expected不是一个假hash，而是`CODEBOOK_UNDEFINED`并阻止任何G1 executable packet。

## 7. Freeze裁决边界

本 addendum关闭的是书面structural contract的六个局部歧义：global codec成为左右可逆的唯一排序；POINT/AXIS是精确partial-goal manifold；VERIFY STABILIZE只保持anchor时已满足的公开effector目标；RIGHT aperture action target与被监测/稳定entity分域；human/text JSON逐enum、逐key、NULL无歧义；metric artifact绑定W/H、range、epsilon和tie；未来CPU vectors有固定expected。

因此在**不声称本轮已实现或运行CPU checker**的前提下，`minimal-r1d`的书面合同可标为：

```text
structural specification: FREEZE CANDIDATE, exact contract complete
CPU conformance:          PENDING
G1 executable profile:    NO-GO
G1 profile states/bits:   UNDEFINED
```

只有未来CPU gates实际通过后才升级为`Freeze structural / CPU-conformant`。真实camera/metric tables、G1 static capability/resource profile、runtime receipts与task admission仍未标定，不能用本addendum的结构freeze替代部署证据。
