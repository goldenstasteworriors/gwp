# SM-FET `minimal-r1d`：不增核心字段的最终 schema contract 修订

- 日期：2026-08-12
- 性质：依据 r1c final No-Go 的合同修订；不新增 core packet fields，不实现、不训练、不使用 GPU、不修改 canonical
- canonical 代码基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- schema ID：`sm_fet_hard_grammar_v2_minimal_r1d_struct_v1`
- hardware family：Unitree G1 + left RH56DFTP + right OmniPicker

## 0. 裁决

`minimal-r1d` 保留 r1c 的32-byte wire、5-token布局、active-only、O/S/M function、七个known parts、full finite frame/path和显式robot assignment；不增加任何模型字段。它只修正合同：

1. RIGHT+RIGHT按**有限资源目标**而非assignment名字判冲突；结构grammar加入profile-gated `RO+RS`、不同payload `RO+RO`和VERIFY `RS+RS`，两个独立aperture targets仍mutex；
2. held transform成为完整、显式、所有route共享的 `public_execution_state_v1`，不再假装action-invisible physical fact；
3. human shared key改用domain-neutral purpose，禁止exact duplicate，机械导出唯一focus，并要求EXHAUSTIVE constraint roster；
4. static capability profile、metric codebook、runtime resource receipt和held/public-state receipt严格分层；
5. structured text、canonical25 proprio与其它public state都显式hash并在equal-capacity causal实验中固定；
6. FRAME/PATH坐标、bin edge、tie-break、path transport、controlled-part/effector变换与checksum规则闭合；真实metric ranges缺失时profile仍No-Go；
7. MONITOR保持action-visible，自己的split action slice恒为零，但允许条件化其它physical slices；同时收紧VERIFY：`primary_part=N_A`时只允许POINT/AXIS/FRAME，禁止暗含motion command的PATH。

RIGHT topology变化和VERIFY locator收紧使r1c容量失效，必须重算。r1d hardware-family structural alphabet为：

```text
field wire            = 32 bytes
hard token slots      = 5
|G_struct_r1d|        = 157,516,896,299,151,958,972,545,105,921
log2 |G_struct_r1d|   = 96.99142134161319 bit
minimum index width   = 97 bit
compact container     = 13 bytes
```

这仍只是 **G1/RH56/Omni hardware-family structural grammar exact size**。当前没有已标定、authoritative 的 biochem4 capability profile、metric codebook、working-surface/resource table或held-state source，所以：

```text
|G_profile(g1-rh56-omni, task stratum)| = UNDEFINED / NO-GO
profile bits                              = UNDEFINED / NO-GO
```

推荐冻结 r1d 的**合同、布局、结构grammar与计数**，作为CPU checker与无训练oracle的final candidate；不冻结 executable G1 profile，不批准实现、训练或真机。

## 1. 不变的 core packet fields

### 1.1 Header

```text
compact phase:
  APPROACH_ALIGN | ACQUIRE | CONTROLLED_CONTACT |
  REGRASP | RELEASE | VERIFY

known intended effect:
  NONE | TRANSLATE | ROTATE | ARTICULATE |
  FLOW | DEFORM | TRANSFER
```

合法phase/effect blocks仍是11个nonVERIFY加1个VERIFY：

```text
0  APPROACH_ALIGN + NONE
1  ACQUIRE + NONE
2  CONTROLLED_CONTACT + NONE
3  CONTROLLED_CONTACT + TRANSLATE
4  CONTROLLED_CONTACT + ROTATE
5  CONTROLLED_CONTACT + ARTICULATE
6  CONTROLLED_CONTACT + FLOW
7  CONTROLLED_CONTACT + DEFORM
8  CONTROLLED_CONTACT + TRANSFER
9  REGRASP + NONE
10 RELEASE + NONE
11 VERIFY + NONE
```

### 1.2 Constraint fields

每个present constraint仍恰含：

```text
constraint_function: OPERATE | STABILIZE | MONITOR
assignment: LEFT_RH56 | RIGHT_OMNIPICKER | MONITOR_ONLY
primary_role                       # controlled/focus candidate
primary_part
locator: POINT | AXIS | FRAME | PATH
secondary_role | NONE
intended_target_relation | N_A
```

Function/assignment唯一匹配：

```text
LO = LEFT_RH56 + OPERATE
LS = LEFT_RH56 + STABILIZE
RO = RIGHT_OMNIPICKER + OPERATE
RS = RIGHT_OMNIPICKER + STABILIZE
ZM = MONITOR_ONLY + MONITOR
```

不存在 LEFT/RIGHT+MONITOR，也不存在 MONITOR_ONLY+OPERATE/STABILIZE。

### 1.3 Role、part、relation

Role codes不变：0 PAD/NONE，1 SOURCE，2 TOOL，3 DESTINATION，4 SUPPORT。

Part codes不变：

```text
0 PAD
1 N_A                         # VERIFY only
2 BODY
3 HANDLE_OR_GRASP_FEATURE
4 LID_CAP
5 OPENING_OR_RIM
6 TOOL_TIP
7 ACTUATOR_OR_PLUNGER
8 WORKING_FACE_OR_PAD
```

Relation codes不变：0 N_A；1 SEPARATE；2 NEAR；3 TOUCHING；4 INSIDE；5 CONTAINS；6 ALIGNED_TO_RIM；7 SUPPORTED_BY。Secondary NONE iff relation N_A；非零secondary必须不同primary。

## 2. RIGHT+RIGHT 与完整资源冲突合同

### 2.1 修订原则

“OmniPicker只有一个aperture”只推出：**两个不同、同时要求的 aperture actuation targets 冲突。** 它不推出任意两个RIGHT constraints按assignment全称冲突。r1d因此把以下bundle加入结构可表示集，再由静态resource templates和runtime facts裁剪：

```text
nonVERIFY:
  RO + RS
  RO + RO, payloads distinct

VERIFY:
  RS + RS, payloads distinct
```

完全相同目标必须merge/deduplicate，不形成双constraint。两个独立aperture target永久mutex。其它RIGHT compound是否admit取决于profile中唯一、有限的resource claims；未知claim不是compatible。

### 2.2 Resource claim modes

每条physical constraint由static profile唯一映射到有限claim entries：

```text
resource_id
claim_mode =
  EXCLUSIVE_TARGET(target_class)
  HOLD_SAME_TARGET(target_class)
  READ_ONLY
  NO_CLAIM
```

`target_class`是有限controller enum，不含pose、aperture数值、gain或scene ID。若同一wire semantic key可能映射两套不同claim templates，profile删除该tuple。

### 2.3 资源冲突表

| resource | static rule | runtime rule | 结果 |
|---|---|---|---|
| `RIGHT_OMNIPICKER_APERTURE` | 两个不同 `EXCLUSIVE_TARGET` 永久mutex；相同目标merge；`HOLD_SAME_TARGET`只可与同target兼容 | 当前aperture/contact validity由runtime receipt确认 | 不由RIGHT名字全称拒绝 |
| `RIGHT_WRIST_POSE` | 两个不同pose-target classes默认mutex；一个constraint为NO_CLAIM/READ_ONLY时可候选 | 当前reachability/collision再判 | 可允许wrist-dominant + aperture-only |
| `LEFT_WRIST_POSE` | 同上 | 当前reachability/collision | LEFT compound仍需exact allowlist |
| `LEFT_RH56_FINGER_*` | 相同独占finger target mutex；disjoint channel claims仅为候选 | 当前held/contact与oracle receipt | pipette O+S仍待标定 |
| `G1_SHARED_WBC_BALANCE` | profile只声明需要该predicate，不放当前值 | 每anchor WBC receipt | runtime，不进G_profile兼容值 |
| `G1_SHARED_SELF_COLLISION` | 同上 | 每anchorcollision receipt | runtime |
| `HELD_OBJECT_BINDING` | profile声明binding kind/所需part | freshness与concrete transform在public state | runtime |
| `MONITOR_EVIDENCE` | 相同monitor payload duplicate；不同payload可候选 | 每条fresh evidence receipt | 无动作slice、可影响其它slice |

Static pair compatible当且仅当所有有限claim templates不存在已知mutex；`UNKNOWN_NOT_ADMITTED`、缺claim或多模板均false。Runtime resource变化只能改变admission/assignment，不修改static profile bytes。

### 2.4 Revised bundle grammar

NonVERIFY至少一条OPERATE：

```text
single:
  LO | RO

different-type pairs:
  LO-RO, LO-RS, LS-RO,
  LO-ZM, RO-ZM,
  LO-LS,                 # profile-gated LEFT compound
  RO-RS                  # profile-gated RIGHT compound

same-type pair:
  RO-RO with distinct payload ranks
```

不允许LS/RS/MONITOR-only nonVERIFY bundle。VERIFY禁止OPERATE：

```text
single:
  LS | RS | ZM

different-type pairs:
  LS-RS | LS-ZM | RS-ZM

same-type pairs:
  ZM-ZM with distinct payloads
  RS-RS with distinct payloads and profile compatibility
```

不加入LS-LS：当前capability memo没有同LEFT双stabilization的正候选；未来若证明必要，必须新grammar/recount。MONITOR组合已经由LO/RO+ZM、VERIFY LS/RS+ZM和ZM+ZM显式允许；它从未因为“无action slice”而与RIGHT资源冲突。

## 3. MONITOR：action-visible、zero own slice

`MONITOR_ONLY+MONITOR` 仍形成constraint token和locator token，计入packet alphabet并参与action attention。它不打开left/right任何新split slice：

```text
gate(MONITOR_ONLY) = all-zero actuator ownership mask
```

但“zero own slice”不等于action expert忽略它。Fresh passive support、fixture lock或relation evidence可以改变同bundle中OPERATE/STABILIZE slice的速度、方向、安全corridor或是否允许推进。

必须满足：

1. 固定OPERATE constraint，在support-confirmed MONITOR与support-absent/active-STABILIZE matched worlds中，packet/admission或physical action按预注册语义变化；
2. MONITOR missing/expired时整个required bundle拒绝，不能drop它后执行OPERATE；
3. MONITOR不表示unknown、not-reviewed、last-known或inactive effector；
4. function/assignment联合干预使用完整合法bundle，不能对单byte做非法swap并要求“只变对应slice”；
5. 若实验证明MONITOR永不影响任何physical slice，只用于supervisor admission，则下一schema应移出action并重新计容；r1d当前把它保留为可干预action condition。

## 4. Public execution state：held transform 的正确位置

### 4.1 唯一不变量

正确隔离式为：

```text
fixed canonical packet
+ fixed complete public_execution_state bytes/hash
+ fixed structured text bytes/hash
+ fixed timestep/noise
-> same learned pre-safety action
```

改变真实held transform、contact或camera pose就是改变public state，允许并通常要求action变化。所有semantic/equal-bit/baseline routes必须得到相同public state bytes；matched causal实验要么固定hash，要么显式把该state作为唯一干预变量。

Receipt metadata、allocator score、candidate order或两张证明同一physical state的等价receipt变化，在public state和packet固定时不得改变action。

### 4.2 `public_execution_state_v1`

这是packet外的显式条件输入，不增加32-byte core fields。Canonical binary field order：

```text
magic/version:                 ASCII `SMFETPES1` + uint16_le value 1
anchor_timestamp_ns:           int64_le
canonical_proprio:             float32_le[25]
  left wrist 9
  right wrist 9
  left RH56 6
  right Omni aperture 1

camera_mode:                    uint8 FIXED=0 | MOVING=1
T_base_camera:                 float32_le[7]
  translation xyz meters
  unit quaternion wxyz
  FIXED mode -> canonical all-zero payload; transform comes from calibration hash

for effector in [LEFT_RH56, RIGHT_OMNIPICKER]:
  held_valid:                   uint8 0|1
  held_binding_kind:            uint8 finite profile enum
  controlled_part:             uint8 r1d part code or 0
  T_effector_controlled_part:   float32_le[7]
  translation_uncertainty_m:    float32_le
  angular_uncertainty_rad:      float32_le
  invalid -> all payload after held_valid is positive-zero
```

Quaternion canonicalization：unit norm在manifest tolerance内；`w>0`，若`w==0`则第一个非零`x,y,z`为正；否则state invalid。NaN、Inf、negative zero、noncanonical padding或stale timestamp拒绝。Exact state bytes计算：

```text
public_state_hash = SHA256(canonical_binary_bytes)
```

当前 schema**不允许 learned action读取**额外contact/load/slip连续量；它们只可供post-WAM safety hold/clamp/retract。若某task主动作需要这些反馈，必须建立新 public-state revision、对所有route匹配并将capacity继续表述为conditional；当前G1 profile保持No-Go。

### 4.3 Controlled-part 与 effector frame方程

Frame convention：`T_A_B`把B-frame坐标变换到A-frame。Wire locator编码 controlled primary part 的目标 `T_C_P*` 或PATH `T_C_P*(s)`，其中C是anchor时刻current robot camera，P是primary controlled part。

Public state给 `T_E_P(t)`，即controlled part在assigned effector E中的current held transform。目标effector frame机械为：

```text
T_C_P*(s) = T_C_E*(s) @ T_E_P(t)

therefore

T_C_E*(s) = T_C_P*(s) @ inverse(T_E_P(t))
```

Action expert/公开deterministic decoder可以读取hard locator与public state计算该目标；不存在action-invisible transform旁路。若held state missing、stale、多hypothesis或uncertainty超过profile tolerance，runtime拒绝packet。

## 5. Human-shared semantic key v1d

### 5.1 Domain-neutral purpose，不强制O/S/M equality

Shared key不直接使用robot action function，而使用：

```text
EFFECT_GOAL
SUPPORT_REQUIREMENT
OBSERVATION_REQUIREMENT
```

Robot binding规则：

```text
EFFECT_GOAL          -> OPERATE
SUPPORT_REQUIREMENT  -> STABILIZE, or MONITOR when fresh passive support is verified
OBSERVATION_REQUIREMENT -> MONITOR
```

所以human第二手STABILIZE与robot fixture MONITOR可以共享同一个SUPPORT requirement；robot packet仍显式发送最终O/S/M。Purpose只存在pairing/annotation sidecar，不新增model packet field。

### 5.2 Exact JSON object

非NULL object必须**恰有**以下keys，不允许额外key；所有enum均用这里给出的uppercase string literal，不用数值code：

```json
{
  "schema_id": "sm_fet_r1d_shared_semantic_key_v1",
  "task_ontology_id": "<frozen ASCII identifier>",
  "task_family": "POUR|INSERT|PIPETTE|CAP|WIPE",
  "phase": "APPROACH_ALIGN|ACQUIRE|CONTROLLED_CONTACT|REGRASP|RELEASE|VERIFY",
  "effect": "NONE|TRANSLATE|ROTATE|ARTICULATE|FLOW|DEFORM|TRANSFER",
  "event_focus_role": "MANIPULATED_OR_SOURCE|TOOL|RECEPTACLE_OR_DESTINATION|FIXTURE_OR_SUPPORT",
  "constraint_roster_coverage": "EXHAUSTIVE",
  "constraints": [
    {
      "purpose": "EFFECT_GOAL|SUPPORT_REQUIREMENT|OBSERVATION_REQUIREMENT",
      "primary_role": "MANIPULATED_OR_SOURCE|TOOL|RECEPTACLE_OR_DESTINATION|FIXTURE_OR_SUPPORT",
      "primary_part": "BODY|HANDLE_OR_GRASP_FEATURE|LID_CAP|OPENING_OR_RIM|TOOL_TIP|ACTUATOR_OR_PLUNGER|WORKING_FACE_OR_PAD|N_A",
      "secondary_role": "NONE|MANIPULATED_OR_SOURCE|TOOL|RECEPTACLE_OR_DESTINATION|FIXTURE_OR_SUPPORT",
      "target_relation": "N_A|SEPARATE|NEAR|TOUCHING|INSIDE|CONTAINS|ALIGNED_TO_RIM|SUPPORTED_BY"
    }
  ]
}
```

`task_ontology_id`必须先冻结，匹配ASCII `[a-z0-9._/-]{1,64}`；不能含episode/object/scene/locator信息。

NonVERIFY shared entry的part只能是七个known executable parts；VERIFY entry只允许`N_A`。Secondary NONE iff relation N_A，非零secondary不能等于primary。Phase/effect必须通过第1.1节block表。

### 5.3 Canonical constraints、focus 与 completeness

Constraint sort使用以下ordinal表而非locale/string order：

```text
purpose: EFFECT_GOAL=1, SUPPORT_REQUIREMENT=2, OBSERVATION_REQUIREMENT=3
role: MANIPULATED_OR_SOURCE=1, TOOL=2,
      RECEPTACLE_OR_DESTINATION=3, FIXTURE_OR_SUPPORT=4, NONE=0
part: BODY=1, HANDLE_OR_GRASP_FEATURE=2, LID_CAP=3, OPENING_OR_RIM=4,
      TOOL_TIP=5, ACTUATOR_OR_PLUNGER=6, WORKING_FACE_OR_PAD=7, N_A=8
relation: N_A=0, SEPARATE=1, NEAR=2, TOUCHING=3, INSIDE=4,
          CONTAINS=5, ALIGNED_TO_RIM=6, SUPPORTED_BY=7
```

Sort tuple为 `(purpose,primary_role,primary_part,secondary_role,target_relation)`。Exact duplicate tuple禁止；embodiment-specific fan-out到两个robot constraints只发生在robot binding层，不复制human shared entry。

NonVERIFY必须有至少一个EFFECT_GOAL；VERIFY可以没有。`event_focus_role`机械为：先取sorted constraints中第一个EFFECT_GOAL的primary；若VERIFY无EFFECT_GOAL，取sorted constraints第一个entry的primary。JSON值必须与派生值一致，否则key invalid。这一规则正式取代“每条primary都是event focus”；action packet不新增focus field。

只有 `constraint_roster_coverage=EXHAUSTIVE`、1..2条shared requirements、所有required fields known且primary/secondary roster完整时才生成key。partial positive、另一手/constraint未审样本可做field-level planner supervision，但不生成完整pair key或oracle packet。

### 5.4 NULL key

只有EXHAUSTIVE确认当前没有active event时允许：

```json
{
  "schema_id": "sm_fet_r1d_shared_semantic_key_v1",
  "task_ontology_id": "<frozen ASCII identifier>",
  "task_family": "POUR|INSERT|PIPETTE|CAP|WIPE",
  "phase": "NULL",
  "effect": "N_A",
  "event_focus_role": "NONE",
  "constraint_roster_coverage": "EXHAUSTIVE",
  "constraints": []
}
```

Unknown/missing/occluded不生成NULL key。

### 5.5 Bytes/hash 与 pair cost

验证schema/enum/sort/focus/completeness后，使用 RFC 8785 canonical JSON UTF-8：

```text
semantic_key = SHA256(canonical_json_bytes)
```

Pairing/CAPOT strict cost仅可读取：task family、phase、effect、purpose、role、part、target relation、known masks与local order。永久禁止：

```text
human or robot uv/depth/axis/roll/path
global center/scale/orientation
bbox/mask/track/pose/appearance
human actor / robot assignment
instance ID / held transform / profile or resource state
duration/outcome/confidence/pair ID/CAPOT mass
```

Geometry改变不得改变semantic key、candidate gate或strict pair cost。CAPOT只产生stop-gradient sparse semantic pair weight，不产生locator/assignment/action target。

## 6. 四类 receipt 必须分离

| receipt | immutable/current | 内容 | 是否action-readable | 影响哪个集合 |
|---|---|---|---:|---|
| `static_capability_profile_receipt_v1` | split前immutable | function/effector/part/working-surface/resource templates、static mutex、same-effector allowlists、task rules、controller enums | profile hash固定为route条件，不逐sample读取 | `G_profile(p,t)` |
| `metric_codebook_receipt_v1` | calibration后immutable | camera convention/intrinsics、uv/depth/axis/roll/path/extent tables与checksum | decoder按固定hash解释codes；不可逐sample切换 | code semantics与profile mask |
| `runtime_resource_receipt_v1` | per anchor | concrete role instances、reachability、WBC、collision、current monitor evidence、resource availability | metadata/reason不进action；其结果可改变admission/assignment packet | `R_runtime` |
| `held_public_state_receipt_v1` | per anchor | public state中held transform/validity/uncertainty的source、freshness、manifest/calibration hashes | physical values已在public state；receipt metadata不进action | `R_runtime`与public-state validity |

Static profile不能含current instance、freshness、contact、reachability、WBC/collision values。Offline oracle可以授权产生一个新的immutable profile revision，但不作为per-anchor profile predicate。Metric codebook不能混入task/scene geometry。Runtime receipts不能改profile/codebook或写continuous correction。

## 7. Structured text、proprio 与 matched controls

### 7.1 Exact allowed text object

Action-visible text先解析成恰有以下keys的structured object，再用RFC8785 bytes/hash；free-form原文不直达strict action route：

```json
{
  "schema_id": "sm_fet_r1d_text_goal_v1",
  "task_family": "POUR|INSERT|PIPETTE|CAP|WIPE",
  "goal_symbol": "<frozen ontology symbol>",
  "quantity_class": "N_A|LOW|MEDIUM|HIGH",
  "signed_direction": "N_A|POSITIVE|NEGATIVE",
  "success_criterion": "RELATION|DOSE|TORQUE|CLEAN|POSE"
}
```

`goal_symbol`匹配 `[a-z0-9._/-]{1,64}`，只指功能目标，不允许通过lookup映射到current robot pixel/depth/axis/roll/path、assignment、held transform、trajectory或per-scene controller。Named well/port若使用symbol，改变场景位置时symbol保持不变，具体instance/geometry必须由current robot observation解析。

禁止text携带：数字坐标/pose/depth、robot locator code或别名、human coordinates、waypoint/spline、assignment、profile selector、gain、action chunk或instance-specific trajectory。

```text
text_hash = SHA256(RFC8785_canonical_JSON_UTF8)
```

### 7.2 Proprio/public-state boundary

Canonical25严格是 `float32_le[25]`：left wrist9 + right wrist9 + RH56 six + Omni aperture one，使用冻结 `sonic_g1_rh56dftp6_omnipicker1_v1` units/frame/normalization。它不是可任意扩展的“state”占位词。

Camera pose和held transform只按第4节public state schema加入；当前 learned route不读contact/load/slip。任何state channel revision必须升route/schema ID、更新state hash、给所有baseline相同输入并继续把packet capacity称为conditional。

### 7.3 Equal-capacity causal experiment

Semantic、equal-bit、standard/F-PCD/continuous matched route比较必须固定或显式成对干预：

```text
text_hash
public_state_hash
static_profile_hash
metric_codebook_hash
hardware_manifest/calibration hash
noise/timestep/anchor timestamp
```

禁止一条route看到robot locator alias、held transform或camera pose而另一条没有。若text/public state变化，不能把action差异归因于packet semantic bits。

## 8. FRAME/PATH codebook contract

### 8.1 Camera coordinate convention

使用右手 OpenCV camera frame：`+x` image right，`+y` image down，`+z` optical forward；`x cross y = z`。Pixel-center normalization：

```text
u=(pixel_x+0.5)/W
v=(pixel_y+0.5)/H
```

UV bin edges恰为 `i/64, i=0..64`。Intervals为左闭右开，最后一档右闭；恰在内部edge归入较高index。合法pixel center不会等于0或1；support越界或跨越会改变safe action的多个bins时UNKNOWN/no packet。

Depth使用camera-z米制值。Executable codebook必须给33个严格递增 `float64` edges `z_edge[0..32]` 与32 centers；不能由task profile重新定义。内部edge同样归较高bin，range外UNKNOWN。当前真实range/edges缺失，所以metric codebook hash未定义，G1 profile No-Go。

给定decoded normalized bin center `(u_c,v_c)`、depth center `z_c`、image size与camera intrinsics `(f_x,f_y,c_x,c_y)`，translation center唯一解码为：

```text
pixel_x = u_c*W - 0.5
pixel_y = v_c*H - 0.5
x_c = (pixel_x-c_x)*z_c/f_x
y_c = (pixel_y-c_y)*z_c/f_y
p_C = (x_c,y_c,z_c)
```

### 8.2 Directed axis table与quantization

Generator version：`r1d_axis_az32_el16_equal_angle_v1`。

```text
azimuth edges:   -pi + 2*pi*a/32, a=0..32
elevation edges: -pi/2 + pi*e/16, e=0..16

center(a,e):
  az=-pi + 2*pi*(a+0.5)/32
  el=-pi/2 + pi*(e+0.5)/16
  d=(cos(el)cos(az), cos(el)sin(az), sin(el))

axis_code=1+32*e+a
```

Azimuth先wrap到`[-pi,pi)`；恰在internal edge取较高index；`pi`映`-pi`。严格竖直或`hypot(x,y)`低于codebook epsilon时canonical `a=0`；e按z方向取端档。512个decoded centers必须float64 materialize、逐项finite/unit-norm/distinct检查并进入checksum。Support跨task-safe axis cells时UNKNOWN，不nearest average。

### 8.3 Roll 与 zero-roll frame

对axis center `z_d`，从camera basis `ex,ey,ez`选择 `abs(dot(seed,z_d))`最小者，精确tie顺序 `ex<ey<ez`：

```text
b1=normalize(seed cross z_d)
b2=z_d cross b1
theta_r=2*pi*r/32, r=0..31
x_r=cos(theta_r)b1+sin(theta_r)b2
y_r=z_d cross x_r
R_C_F=[x_r,y_r,z_d]
roll_code=1+r
```

Roll support先投影到该zero-roll basis，再wrap `[0,2pi)`；edges为 `2*pi*r/32`，internal edge归较高index，`2pi`映0。Task symmetry必须在static profile选唯一canonical representative；否则profile删除该tuple。

### 8.4 Path primitives与orientation transport

在decoded anchor frame F的local xy plane，参数 `s in [0,1]`：

```text
LINE:
  gamma(s)=(s,0)

ARC_LEFT:
  gamma(s)=(sin(pi*s/2), 1-cos(pi*s/2))

ARC_RIGHT:
  gamma(s)=(sin(pi*s/2), -(1-cos(pi*s/2)))

RASTER_3 polyline vertices:
  (0,0)->(1,0)->(1,1/2)->(0,1/2)->(0,1)->(1,1)
  parameterized by normalized cumulative arc length
```

Extent codebook必须给9个严格递增positive `float64` edges和8 centers，单位meter；内部edge归高档。Decoded center `d_k`统一缩放gamma。LINE/ARC的orientation tangent用normalized analytic derivative；RASTER每segment为constant tangent，恰在vertex取outgoing segment，final vertex取incoming。Surface normal `z_F`在整条path固定：

```text
p_C_P*(s)=p_anchor + R_C_F * [d_k*gamma_x(s), d_k*gamma_y(s), 0]^T
z_C(s)=R_C_F*ez
x_C(s)=normalize(R_C_F*[tangent_x(s),tangent_y(s),0]^T)
y_C(s)=z_C(s) cross x_C(s)
R_C_P*(s)=[x_C(s),y_C(s),z_C(s)]
```

然后按第4.3节用public `T_E_P(t)`求 `T_C_E*(s)`。这冻结了PATH是在controlled-part frame生成、normal固定、tangent随path更新；没有private spline/progress。每anchor重新从s=0执行first-8 reference；跨chunk progress若必要必须成为公开有限state的新revision，当前profile否则No-Go。

### 8.5 POINT/AXIS/FRAME/PATH 与 UNKNOWN

```text
POINT: uv+depth；axis=roll=primitive=extent=0
AXIS:  uv+depth+axis；roll=primitive=extent=0
FRAME: uv+depth+axis+roll；primitive=extent=0
PATH:  uv+depth+axis+roll+primitive+extent全部present
```

POINT不因normal未知拒绝；AXIS不因roll未知拒绝；只有kind实际required field未知才拒绝。Required support跨会产生不同safe actions的bins、out-of-range、transparent无direct depth、frame symmetry多解或codebook hash不符均no packet。没有action-visible UNKNOWN code，不能nearest/last-value/completion fill。

VERIFY的`primary_part=N_A`表示对role-level point/axis/frame事实进行STABILIZE或MONITOR，并不指定新的controlled part。故VERIFY只允许POINT、AXIS、FRAME；PATH会规定沿曲线运动且必须绑定controlled part/effector transform，在`N_A`下语义不完整，validator以`VERIFY_PATH_FORBIDDEN`拒绝。nonVERIFY仍允许四种locator。这是字段不变但grammar收紧，必须反映在structural count和rank中。

### 8.6 Codebook artifact/checksum

`metric_codebook_receipt_v1` 必须绑定：

```text
manifest.json: RFC8785 canonical JSON
  schema/generator versions
  camera convention and calibration hash
  W,H normalization rule
  z/extent units and ranges
  bin-edge/tie rules
  axis/roll/path/interpolation versions
tables.bin:
  fixed field order, float64 little-endian
  camera intrinsics
  33 depth edges + 32 centers
  512 axis centers[xyz]
  9 extent edges + 8 centers
  canonical primitive vertices/constants
```

```text
codebook_hash = SHA256(
  RFC8785_manifest_UTF8 || 0x00 || tables_bin_bytes
)
```

NaN、Inf、negative zero、重复axis center、非递增edge、center不在对应bin或hash不符均拒绝。当前未有真实G1 calibration/metric ranges，故本memo不伪造checksum；只能冻结生成与验证合同、wire cardinality和structural count。

## 9. Static profile、runtime state 与 exact profile formula

### 9.1 Pure sets

```text
G_struct_r1d:
  hardware-family canonical wire grammar。

G_profile(p,t):
  对固定immutable capability profile p和fixed task/text stratum t，
  仅应用finite static allowlists、resource templates/mutex、symmetry、
  controller enum唯一性与codebook masks后的profile-valid wire set。

R_runtime(p,t,state_receipts):
  再应用current role instances、held/contact freshness、reachability、
  WBC/collision、monitor evidence与safety predicates。
```

`G_profile`的compatibility函数只能读 `(p,t,wire payloads)`；不读scene、current instance、receipt、score或WBC值。Runtime facts不改变profile count。

### 9.2 Formula

对每个 `(p,t,h)` 定义static admitted payload sets：

```text
Omega_LO, Omega_LS, Omega_RO, Omega_RS, Omega_ZM
```

`C(A,B)`是两个different-type sets的static compatible ordered pairs；`U(A)`是按canonical payload rank `x<y` 的static compatible unordered distinct pairs。

NonVERIFY：

```text
B_h(p,t)
= |LO|+|RO|
 +C(LO,RO)+C(LO,RS)+C(LS,RO)
 +C(LO,ZM)+C(RO,ZM)
 +C(LO,LS)
 +C(RO,RS)
 +U(RO)
```

VERIFY：

```text
B_v(p,t)
= |LS|+|RS|+|ZM|
 +C(LS,RS)+C(LS,ZM)+C(RS,ZM)
 +U(ZM)+U(RS)
```

因此：

```text
|G_profile(p,t)|
= 1 + sum_{h in admitted 11 nonVERIFY blocks} B_h(p,t)
    + B_v(p,t)
```

同一checkpoint若服务多个task/text strata，必须逐`t`报告。若要报告跨task wire set，只能取canonical wire bytes的并集并去重；不能相加，因为同wire在不同text条件下可有不同控制语义。

### 9.3 当前G1 profile No-Go

Static capability entries、working surfaces、resource claims、RIGHT/LEFT compounds、metric codebook、driver/contact与task oracle仍未标定或为UNKNOWN/UNSUPPORTED；没有authoritative `p` 或codebook hash。因此profile integer/bits保持UNDEFINED。不能把NEEDS_CALIBRATION当true，也不能全删后报告“1 state profile”。

## 10. Exact wire、tokens 与 revised codec

### 10.1 Wire/tokens不变

Header 2 bytes：phase、effect。每constraint仍15 bytes：

```text
function:u8, assignment:u8, primary_role:u8, part:u8, kind:u8,
uv:u16_le, depth:u8, axis:u16_le, roll:u8,
primitive:u8, extent:u8, secondary_role:u8, relation:u8
```

Slot0 base2，slot1 base17；总32 bytes。唯一NULL全零。五tokens仍为EVENT、CONSTRAINT0、LOCATOR0、CONSTRAINT1、LOCATOR1。MONITOR token正常attention，assignment gate为zero own slice。

### 10.2 Structural recount

字段布局不变，但VERIFY禁止PATH，所以nonVERIFY与VERIFY使用不同locator alphabet：

```text
P = 64*64*32 = 131,072

L_nonverify
= P*(1 + 512 + 512*32 + 512*32*4*8)
= 70,934,200,320

L_verify
= P*(1 + 512 + 512*32)
= 2,214,723,584

N = 7*4*L_nonverify*22 = 43,695,467,397,120
V = 1*4*L_verify*22    = 194,895,675,392
```

新增RIGHT blocks后：

```text
B_nonverify_r1d
= 2*N + 7*N^2 + C(N,2)
= 14,319,704,032,895,892,051,583,303,680

B_verify_r1d
= 3*V + 3*V^2 + 2*C(V,2)
= 151,937,297,146,405,128,765,440

|G_struct_r1d|
= 1 + 11*B_nonverify_r1d + B_verify_r1d
= 157,516,896,299,151,958,972,545,105,921

log2 = 96.99142134161319 bit
ceil  = 97 bit
```

相对r1c增加 `0.32036897249198 bit`；wire仍32 bytes、compact仍13 bytes。这里不能说“字段没变所以容量不变”：VERIFY的合法topology减少是严格grammar变化；RIGHT compound增加则是严格grammar放宽，二者已共同计入。

### 10.3 Rank/unrank

Wire `uv=1+64*row+col`，其中`row,col in 0..63`；translation rank为 `p=((uv-1)*32+(depth-1))`。Locator variant rank严格为：

```text
POINT: q=0
AXIS:  q=1+(axis-1)
FRAME: q=1+512+(axis-1)*32+(roll-1)
PATH:  q=1+512+512*32
         +(((axis-1)*32+(roll-1))*4+(primitive-1))*8
         +(extent-1)
```

nonVERIFY variant order为POINT、AXIS、FRAME、PATH；PATH中axis→roll→primitive→extent，extent最低位，每个translation共 `1+512+512*32+512*32*4*8=541,185` variants：

```text
locator_rank_nonverify = p*541185 + variant_rank
```

VERIFY variant order只含POINT、AXIS、FRAME，每个translation共 `1+512+512*32=16,897` variants：

```text
locator_rank_verify = p*16897 + variant_rank
```

二者使用同一wire codes，但VERIFY不能使用有PATH空洞的global rank，否则`V`不是dense alphabet。

Payload exact mixed radix：

```text
payload_rank
= (((part_index*4 + primary_role_index)*L_h + locator_rank_h)*22
   + secondary_rank)
```

`secondary_rank`最低位；`part_index`最高。NonVERIFY取`L_h=L_nonverify`、part index按codes2..8映0..6；VERIFY取`L_h=L_verify`、N_A固定0。

`secondary_rank`定义为：NONE/N_A取0；否则从升序primary role codes `SOURCE,TOOL,DESTINATION,SUPPORT` 删除当前primary，secondary在剩余三项中的local index为`j in 0..2`，relation code为`r in 1..7`，取 `1+j*7+(r-1)`，恰覆盖0..21。

Bundle block order：

```text
nonVERIFY:
  single LO, single RO,
  LO-RO, LO-RS, LS-RO, LO-ZM, RO-ZM, LO-LS, RO-RS,
  RO-RO(q0<q1)

VERIFY:
  single LS, single RS, single ZM,
  LS-RS, LS-ZM, RS-ZM,
  ZM-ZM(q0<q1), RS-RS(q0<q1)
```

Same-type pair使用：

```text
comb_rank(q0,q1;n)=q0*(2*n-q0-1)/2+(q1-q0-1)
```

不同type pair的wire顺序严格等于上述block名字的左到右顺序，例如RO-RS必为constraint0=RO、constraint1=RS；禁止交换编码。Same-type按payload rank升序放入constraint0/1。Decoder重编码不相等即`SORT`；完全相同payload即`DUPLICATE`。

Overall rank：NULL=0；对第1.1节order的nonVERIFY `h=0..10`，`1+h*B_nonverify+b`；VERIFY为`1+11*B_nonverify+b`。

Compact container是13-byte unsigned little-endian integer；`byte[12]&0xFE`必须0，rank必须 `<|G_struct_r1d|`。Validation error priority固定：

```text
LENGTH/RESERVED_BITS
-> NULL_CANONICALITY
-> FIELD_RANGE/PAD
-> LOCATOR_KIND
-> FUNCTION_ASSIGNMENT/BUNDLE
-> SECONDARY_RELATION
-> SORT/DUPLICATE
-> STATIC_PROFILE
-> RUNTIME_RECEIPT
```

### 10.4 Equal-bit

Global CPU codec diagnostic令：

```text
M=|G_struct_r1d|-1
 =157,516,896,299,151,958,972,545,105,920

pi(0)=0
pi(u)=1+((65537*(u-1)+104729) mod M)
```

`gcd(65537,M)=1`，inverse=`138514980096989891931988754433`。令置换输出rank为`y`，逆式：

```text
u=1+((
  138514980096989891931988754433
  * ((y-1-104729) mod M)
) mod M)
```

Global permutation只作CPU/encoder诊断，不上安全rollout。Target-profile equal-bit只有在 `G_profile(p,t)` 完整枚举后，按：

```text
(phase,effect,constraint_count,ordered assignment/function tuple)
```

安全stratum内置换residual admitted payload。它保持split routing、MONITOR zero-slice、sample/failure set；当前profile不存在，因此数值与permutation均No-Go。

## 11. Missingness、transparent、clock 与 action boundary

- Required field没有action UNKNOWN；reviewed unknown与missing原因保留，但均no packet；
- EXHAUSTIVE no-active才编全零NULL；partial roster不编single oracle；
- POINT需要uv+depth，不需要normal；AXIS再需要axis；FRAME再需要roll；PATH再需要primitive+extent；
- transparent 2D overlap不能给depth/normal/TOUCHING/INSIDE；completion/reconstruction只作独立baseline；
- 任一required constraint unknown/expired，整个packet与旧cache失效，进入verified HOLD/REOBSERVE；
- action为`Q_action=1,N_future=0`；planner可在action外维护Q≤8与boundary/censor；
- 每execute-8 anchor重新生成semantics、geometry、resource/held receipts与packet；WAM可预测48但只执行前8；
- Action只读5 hard tokens、public execution state、structured text、noise/timestep；不读raw reference、human、pair、planner/allocator hidden或receipt metadata；
- Safety plane可读allowlistedcurrent validity/contact做clamp/hold/retract，但若产生主要task trajectory，必须另报direct-controller baseline；同时报告pre-safety和executed action。

## 12. CPU-only invariants / gates（本轮不实现）

### Grammar/capacity

1. r1d新增RO-RS、distinct RO-RO、VERIFY distinct RS-RS；两个独立aperture targets在profile兼容图中恒mutex；
2. identical RR targets merge/deduplicate；same-type pair严格`q0<q1`；
3. MONITOR_ONLY只配MONITOR，own slice恒zero，但token可改变其它slice；
4. VERIFY的N_A只接受POINT/AXIS/FRAME；任一VERIFY PATH以`VERIFY_PATH_FORBIDDEN`拒绝；
5. wire/rank覆盖恰好 `0..157516896299151958972545105920`；
6. exact integer、96.99142134161319、97-bit、32-byte/5-token逐字固定；
7. affine/inverse与13-byte LE边界/golden vectors round-trip；
8. profile公式与enumerated finite compatibility graph逐项一致，runtime receipt变化不改变`|G_profile|`。

### Public state/text/receipts

9. fixed packet+public_state_hash+text_hash+noise/timestep，metadata/hidden swap后pre-safety action不变；
10. held transform/public state变化允许action变化，并在所有matched route使用相同bytes/hash；
11. held receipt metadata变化但physical public state相同，不改变packet/action；
12. static profile、metric codebook、runtime resource、held state receipts不能互相替代或改层；
13. text拒绝locator code/alias、numeric geometry、assignment、profile selector、trajectory/gain；
14. canonical25与public state NaN/Inf/negative-zero/stale/noncanonical quaternion全部拒绝；
15. allocator fixed public inputs时score/order/private history变化不改变hard output；physical state变化按validator重算。

### Semantic/pairing

16. JSON exact keys/string enums/RFC8785/hash复算一致，extra key或numeric enum拒绝；
17. exact duplicate shared requirement拒绝；event focus按规则唯一派生；
18. nonEXHAUSTIVE roster不生成semantic key/oracle packet；
19. S↔M embodiment变化保持相同SUPPORT_REQUIREMENT key；robot action function仍显式不同；
20. human/robot geometry、assignment、instance、outcome改变不进入pair cost；
21. NULL只来自EXHAUSTIVE no-active，unknown/missing不hash成NULL。

### Codebook/control

22. 512 axis centersfinite/unit/distinct；FRAME正交右手；symmetry canonical唯一；
23. uv/depth/roll/extent internal edge tie取高bin，wrap/tie规则跨实现一致；
24. path analytic/polyline、arc-length、vertex tangent与orientation transport golden vectors一致；
25. codebook artifact/hash缺失或metric ranges未定时任何profile packet拒绝；
26. `O-RIGHT-RESOURCE`只改aperture/wrist claim，compatible↔mutex按表变化；
27. `O-MONITOR-SUPPORT`中MONITOR presence可条件化operate slice，own gate仍zero；
28. `O-HELD-STATE` fixed packet但public held transform变化时允许正确action变化；fixed complete state时receipt swap不变；
29. `O-ROLL-INSERT/O-PATH-NORMAL/O-PATH-SHAPE`消除对应collision；
30. `O-WIPE-OUTSIDE`、transparent missing-depth、third constraint稳定No-Go；
31. relative full/native trajectory超过预注册safe-action gap时相应finite profile停止。

## 13. Freeze recommendation

### Freeze now

- 32-byte/5-token core field layout；
- revised O/S/M × L/R/M bundle topology和resource-level RR原则；
- hardware-family structural integer、rank/unrank与global codec diagnostic；
- exact shared semantic JSON、purpose projection、unique focus、duplicate/roster/NULL规则；
- public execution state/text hash与matched-control boundary；
- 四receipt分层、pure `G_profile(p,t)`公式和MONITOR zero-slice contract；
- codebook generation/validation/checksum schema。

### Still No-Go

- 真实metric depth/extent ranges、tables.bin与codebook hash；
- authoritative G1 capability/profile instance、working surfaces/resource claims与profile bits；
- RH56/Omni driver/contact/load、camera/held state capture和WBC/collision receipts；
- RIGHT/LEFT compound的task admission；
- D0/D1/H1/R0/P0/B0、paired-world oracle、模型实现/训练/仿真/真机；
- transparent pour、pipette compound、roll-sensitive insertion、generic wipe充分性。

因此 r1d 可作为**最终待验证schema contract**冻结，但不是可部署profile或已证明minimal sufficient representation。

## 14. 最终一句话

`minimal-r1d` 不加任何core model字段，却关闭了r1c No-Go的合同漏洞：RIGHT compound由aperture/wrist/finger资源目标决定而非assignment全称删除；held transform进入完整public execution state，正确隔离变成“fixed packet + fixed public state + fixed text → same action”；human key改用domain-neutral purpose、唯一focus、无duplicate且EXHAUSTIVE；profile/codebook/runtime/held四类receipt不混层；text/proprio/state全部hash并禁止locator alias；FRAME/PATH从camera坐标、bin edges、axis/roll、polyline、orientation transport到checksum均有唯一规则；MONITOR action-visible但own slice为零；VERIFY的N_A只表达role-level point/axis/frame事实，不再携带无controlled part的PATH命令。RIGHT topology放宽与VERIFY topology收紧共同把hardware-family structural count更新为 `157,516,896,299,151,958,972,545,105,921` states / `96.99142134161319 bit`。真实G1 profile仍未标定，所以profile integer/bits继续UNDEFINED，而不是用结构数冒充部署reachable。
