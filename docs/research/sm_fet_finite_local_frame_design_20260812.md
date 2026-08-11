# SM-FET action packet：有限局部 frame / locator 独立设计

- 日期：2026-08-12
- 独立工作树：`/home/ykj/project/WM/gwp-agent-sm-fet-local-frame`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 分支：`codex/sm-fet-finite-local-frame-20260812`
- 范围：只比较并裁决有限 robot-view 局部 frame / locator；不实现代码、不训练、不运行 GPU/仿真/真机
- 目标硬件语境：Unitree G1；左 RH56DFTP 六个主动通道；右 OmniPicker 一个 aperture 通道、两个物理夹面

## 0. 裁决

当前 `minimal-r1b` 的 `POINT | AXIS | PATH(axis+extent)` **不能冻结**：单一 axis 会把“同轴、不同 roll”以及“同路径切向、不同表面法向”的精细操作状态编码为同一个 locator。最小可审计修订不是任意 two-axis、连续 6D pose 或逐样本 SO(3) residual，而是一个有明确物理语义的有限局部 frame：

```text
translation:
  64 x 64 normalized robot-view uv
  x 32 globally fixed camera-depth bins

orientation:
  POINT                         # 无方向
  AXIS                          # 512 个有向主轴；只给 roll-invariant 任务
  FRAME                         # 512 个主轴 x 32 个 roll = 16,384 frames

surface path:
  FRAME                         # z = surface normal，x = initial tangent
  x 4 fixed dimensionless path templates
  x 8 globally fixed metric extent bands
```

四个 path template 冻结为：

```text
LINE | ARC_LEFT | ARC_RIGHT | RASTER_3
```

这四个模板只支持直线、单曲率弧和固定三程覆盖；不声称覆盖任意污渍、障碍 corridor、多段轨迹或自由 spline。`extent` 只缩放模板，不携带逐样本宽度、曲率、waypoint 或控制增益。

推荐把这一候选命名为 `sm_fet_finite_local_frame_v1_candidate`。它是对 r1b locator 的修订建议，不是对现有 26-byte wire 或 `81.4779 bit` 数字的兼容解释。**本 memo 只给 locator 的精确结构计数因子，刻意不计算新的整包 alphabet/capacity。** constraint function、移液器同一 RH56 的双约束、part 词表和 held-state 边界仍由 control review 的其它修订处理；局部 frame 不能替它们兜底。

task profile 可以且应该裁剪 vocabulary，但只能用 split 前冻结的 allowlist **删除**不适用状态；不能改变同一 code 的几何含义，不能逐样本选择 profile，不能保存场景 pose/path，不能附连续 residual。透明或遮挡下 required translation/frame/path 无直接证据时，唯一结果是 no-packet + hold/reobserve，而不是 nearest code、last-value 或 reconstruction fill。

## 1. 审查输入与证据边界

本轮读取了以下规范版本：

| 输入 | commit | 本轮采用的约束 |
|---|---|---|
| `sm_fet_minimal_r1b_schema_capacity_20260812.md` | `e31208172d194511e7fc95acd6075635c249c335` | 64×64×32、512 axis、8 extent、target-robot current grounding、严格有限计数 |
| `sm_fet_minimal_r1b_control_review_20260812.md` | `144d050ddfcb3d4a3fa0dc5f299754177c63f04b` | 单 axis 的 roll/normal 反例、PATH 不足、profile/held-state 旁路、四任务 No-Go 点 |
| `sm_fet_dynamic_focus_token_spec_20260812.md` | `2452e60bbaf6edb4714fed7d1397aea6ba9e4111` | human/robot 只共享 role/part/relation/effect；robot 在当前视角重新 grounding |
| `sm_fet_transparent_nonreconstruction_contract_20260812.md` | `f795c4f615595d72219e79a0cb43d5adef12b0a0` | 透明/遮挡的 UNKNOWN/MISSING、direct/native GT allowlist、completion/reconstruction 互斥 |
| `sm_fet_minimal_r1b_crossdomain_review_20260812.md` | `49667be32d15bd95fd2bde610b3e251c2fcc8474` | locator/assignment robot-only、oracle/predicted provenance、profile/text/allocator 防旁路 |

本轮结论属于设计推断，不把相关工作当作充分性证明。尤其：

- 离散动作已有大量前例，不等于这里的局部 frame vocabulary 足够；
- 6D rotation、点云抓取、连续轨迹已有成熟表示，不等于它们符合 strict finite packet；
- human affordance、point/track 可以帮助语义或训练期辅助，不得直接成为 target-robot geometry；
- 512 directions、32 roll、4 templates、8 extent 都是预注册候选，最终仍由 paired-world/native tolerance oracle 决定 Go/No-Go。

## 2. 六类表示的独立比较

| 候选 | 有限性与计数 | 能表达什么 | 关键碰撞/风险 | 本轮角色 |
|---|---|---|---|---|
| 单 `axis + extent` | 最小；当前位置有 `512×8` path variants | 直线方向和长度 | 同轴不同 roll；同 tangent 不同 normal；曲线/覆盖 path | **拒绝作四任务通用 frame**；只留 roll-invariant AXIS baseline |
| `axis + roll` | `D×R`，无非法平行轴组合 | 完整有向局部 frame；可按轴对称性降级成 AXIS | 需规范零 roll、极区基和 symmetry canonicalization | **推荐的 wire factorization** |
| 离散 two-axis frame | naive 为 `D²`，但真实可达数不是 `D²` | 两个显式方向，语义直观 | 平行/近平行非法；Gram–Schmidt 后多码同 frame；若不枚举去重则容量错误 | **拒绝直接上 wire** |
| 离散 SO(3) codebook | 直接 `F` 个 rotation code；可做等体积/低 dispersion | 全局均匀完整旋转 | 不显式区分 roll-invariant axis 与 surface normal/tangent；task mask和错误解释较差 | **16,384-state 等 cardinality baseline** |
| `surface normal + tangent` | 若 tangent 以绕 normal 的 roll 编，恰为 `D×R` | 擦拭法向载荷、工作面朝向和运动切向 | normal/tangent 的来源在透明/遮挡下可能 unknown | **推荐的物理语义**；wire 仍编码 axis+roll |
| path primitive | `FRAME×K×E`；不是 orientation 替代品 | 在局部 surface plane 上表达有限 path family | mask、任意 spline、障碍 corridor、宽度/曲率自由度仍缺失 | **只作 FRAME 的有限附加项** |

### 2.1 为什么不是 raw two-axis

若把两个 512-direction codes 独立相乘，先得到 `512²` 个 wire pairs；但平行/近平行 pair 不能形成稳定 frame，非正交 pair 必须投影，多个 second-axis code 可能在投影、量化或 symmetry quotient 后落到同一 rotation。于是：

```text
reachable frame count != 512 * 512
```

除非预先枚举所有合法 pair、canonicalize、去重并冻结 lookup，不能把 `D²` 写进 capacity。这样做最后仍等价于另造一个不透明 SO(3) codebook，失去 surface normal/tangent 的结构优势。

### 2.2 为什么推荐 normal/axis + roll

一个 directed primary axis `z` 加绕轴角 `roll` 已唯一决定一个右手 frame；surface task 中把 `z` 解释为外向 normal，把零 roll 方向旋转后的 `x` 解释为有向 tangent。它同时满足：

1. AXIS task 可以不支付 roll 状态，不为圆柱形 tip/cap制造冗余码；
2. FRAME task能区分 connector key、pipette plunger 朝向、cap flat 和 wipe working face；
3. PATH天然落在 frame 的局部 `xy` 平面，normal 不再藏在 controller；
4. 每个 factor 都有可干预含义，容易做 axis/roll/normal/tangent matched pairs；
5. 与同 cardinality 的 opaque SO(3) codebook 可直接比较。

### 2.3 为什么 SO(3) codebook只作 baseline

Hopf-fibration 一类 deterministic SO(3) grid 能给均匀性、dispersion 和邻接结构，是非常合适的 full-orientation baseline。但本任务频繁出现 quotient structure：圆柱工具只需 `S²` axis，表面任务需要 normal+tangent，二/四重对称夹具还应 canonicalize roll。统一 SO(3) codebook会给这些任务分配无效或重复状态；task profile再去裁剪时，也很难解释每个删去的 rotation 为什么物理等价。

因此 equal-cardinality baseline 使用 `F=16,384` 个冻结 SO(3) codewords；主方案使用 `512×32=16,384` 个 factorized frames。两者 cardinality相同，但主方案保留 axis/roll 与 normal/tangent 的因果干预结构。

## 3. `finite_local_frame_v1` 的具体契约

### 3.1 Translation grid：保留 `64×64×32`，取消 task-specific 重解释

```text
uv_code:       64 x 64 normalized current target-robot image
depth_code:    32 camera-z bins
P:             4096 x 32 = 131,072 translation cells
```

与原 r1b 相比，关键修正不是增加分辨率，而是冻结唯一 metric 解释：

- 同一 checkpoint/camera hardware registry 只有一套全局 `[z_min_global, z_max_global]` 和 32 个 depth centers；
- task profile只可选择允许的 cell subset，不得重新定义 `[z_min,z_max]`；
- camera intrinsics/extrinsics和 movable camera pose来自固定 calibration或显式 public robot-state allowlist；
- 每个 profile单独验证 uv/depth cell 在其工作区内的 metric covering error；超过 task tolerance 即 No-Go；
- 64×64×32不是 human annotation target。它只在目标机器人当前 replan anchor生成。

当前材料没有足够的 target-hardware 标定与任务公差来诚实填写米制 bin centers；本 memo 固定的是 alphabet cardinality与“全局一义解码”原则。metric centers必须在 oracle 前由 calibration receipt 冻结，不能由 dataset统计、planner或 controller逐样本更新。

### 3.2 Directed-axis vocabulary：`D=512`

第一版保留 r1b 的 512 个有向 axis states，避免同时改变位置、方向分辨率和 frame维度：

```text
axis_code = 32 azimuth bins x 16 elevation bins
D = 512
```

axis table必须物化为 512 个 distinct unit-vector centers并 hash。编码按冻结 cell/support rule；严格竖直、边界和 support 跨 bin都用 canonical rule或拒绝，不由标注者主观挑码。另做同 cardinality 的近等面积 `S²` codebook baseline；若后者显著改善 covering error或 oracle success，再升级 schema，不在本轮偷偷替换 code semantics。

### 3.3 Frame vocabulary：`D=512, R=32, F=16,384`

对每个 axis center `z_d`，确定性构造零 roll 基：

```text
seed axis = camera x/y/z 中与 z_d 最不平行者
            # abs(dot) 最小；tie-break 固定为 x < y < z
b1(d)     = normalize(seed_axis cross z_d)
b2(d)     = z_d cross b1(d)

roll r    = 2*pi*r/32,  r in 0..31
x(d,r)    = cos(roll)*b1(d) + sin(roll)*b2(d)
y(d,r)    = z_d cross x(d,r)
frame     = [x, y, z_d]
```

选择 32 个 roll bins是为了让 roll 的半 bin误差 `5.625°` 与现有 512-axis table的方向量化尺度同阶；它不证明满足 fine insertion。每个 task profile仍要对完整 SO(3) geodesic error和任务安全动作集合做 oracle admission。

语义约定：

| locator use | `z` | `x` |
|---|---|---|
| keyed connector / pipette | 插入或 tool functional axis | key、flat、plunger-to-thumb roll reference |
| cap | cap/thread axis | grasp feature或zero-roll reference |
| pour | opening/rim normal或受控 functional axis | pour-plane/tilt tangent |
| wipe | outward surface normal | initial directed wipe tangent |
| Omni stabilization | jaw closing/approach normal或profile定义的主轴 | jaw-face tangent；只有硬件几何可证时使用 |

`AXIS` 只用于 task/profile已经证明绕 `z` 的所有 roll 对 first-8 safe action 等价的场景。完全 roll-invariant 时不用 FRAME 后再允许32个等价码。若存在二重/四重物体或夹爪对称性，profile必须冻结 canonical representative；不能让多个 wire codes解码成同一控制目标并仍宣称它们是不同 reachable semantics。

### 3.4 Path vocabulary：`K=4, E=8`

PATH只能与完整 surface frame一起出现：

```text
PATH(frame_code, primitive_code, extent_code)

primitive_code:
  1 LINE
  2 ARC_LEFT
  3 ARC_RIGHT
  4 RASTER_3

extent_code:
  1..8 globally fixed log-spaced metric scale centers
```

每个 primitive是 registry 中一条固定、无量纲、从原点出发、初始切向为 `+x`、位于 `xy` 平面的 canonical polyline；`extent` 对它作唯一的统一尺度变换。固定插值器可以把 polyline变成 controller reference，因为它是 hard code、global table和公开 robot state的确定函数。以下内容一律不在 profile或 decoder中暗藏：

- per-scene spline/control points；
- 自由 curvature、width、aspect ratio、pass count或obstacle waypoint；
- 由 raw mask/heatmap/track动态生成的 path；
- continuous endpoint residual或 path phase hidden；
- 从上一 chunk保留的 private progress。

`RASTER_3` 的三程数、turn shape、相对间距和aspect ratio都固定。任意污渍/区域不能由这四个模板在 task tolerance内覆盖时，generic wipe profile直接 No-Go，转向 action-visible track、direct visual servo或continuous trajectory；不能给 registry逐场景新增模板。

`extent` 只表示空间 path scale。cap的旋转角/方向、pour的剂量/停止判据、pipette的aspirate/dispense量不能借用这个字段；它们必须来自冻结 structured goal args、sensor supervisor或另立明确字段。

### 3.5 Canonical kind / PAD 规则

```text
POINT:  translation present; axis=roll=primitive=extent=PAD
AXIS:   translation+axis present; roll=primitive=extent=PAD
FRAME:  translation+axis+roll present; primitive=extent=PAD
PATH:   translation+axis+roll+primitive+extent all present
```

每个非适用字段只有一个 canonical PAD 写法。禁止“FRAME但roll缺失”“PATH用AXIS然后从profile取normal”“POINT附一个不参与控制的nuisance roll”等多码同义状态。

## 4. 精确 locator 状态因子；不计算整包容量

记：

```text
P = 64 * 64 * 32 = 131,072       # translation
D = 512                          # directed axes
R = 32                           # roll bins per axis
F = D * R = 16,384               # frames
K = 4                            # path templates
E = 8                            # path extent bands
```

在**未做 task/profile/capability 裁剪、且假设所有 Cartesian combinations 都合法**时，每个 translation cell的 syntactic variants 是：

```text
POINT = 1
AXIS  = D
FRAME = D * R
PATH  = D * R * K * E

V_struct
= 1 + D + D*R + D*R*K*E
= 1 + 512 + 16,384 + 524,288
= 541,185
```

所以全局 non-PAD locator 的**结构上界**为：

```text
L_struct = P * V_struct
         = 131,072 * 541,185
         = 70,934,200,320 locator states
```

这不是固定 deployment profile 的 exact reachable count，更不是整包 action capacity。profile、semantic key、effector capability、symmetry、metric tolerance和当前 public robot state会继续删状态。

对固定 profile `p`、semantic/controller key `g` 与允许的 public robot-state stratum `s`，准确计数必须直接枚举四个 canonical set：

```text
Omega_point(p,g,s) = admitted translation cells for POINT
Omega_axis(p,g,s)  = admitted (translation, axis) tuples
Omega_frame(p,g,s) = admitted (translation, axis, roll) tuples
Omega_path(p,g,s)  = admitted
                     (translation, axis, roll, primitive, extent) tuples

L(p,g,s)
= |Omega_point| + |Omega_axis| + |Omega_frame| + |Omega_path|
```

只有当所有 masks都被证明与 translation、semantic key和state独立时，才可把它简写成若干 cardinality 的乘积；通常 reachability、碰撞、视野和 symmetry 并不独立，不能擅自相乘。整包还要处理 phase/effect、role/part/relation、constraint function、assignment与单/双constraint grammar，应对各 semantic block求和而不是拿 `L_struct`机械代入旧公式；本 memo不做这一步。

## 5. task profile：允许裁剪，不允许改义

唯一推荐是“**global codebook + frozen profile mask**”：

### profile允许包含

- 合法 semantic key、locator kind、effector和code tuple的有限 allowlist；
- 固定 task tolerance与安全阈值；
- object/tool symmetry order及canonical representative规则；
- RH56/OmniPicker capability约束；
- 四个全局 path template中的有限 subset；
- controller family enum与固定机械常量。

### profile禁止包含

- 另一套 depth/extent centers或同码不同 metric解释；
- per-episode/per-object/per-scene pose、ROI、map、trajectory、spline或affordance field；
- sample-specific controller gain、continuous target、bin residual或learned offset；
- 由planner/allocator分数选择的profile、codebook或decoder；
- test期间新增模板、更新code center或修改symmetry；
- 人类图像坐标、human pose/track或pair ID。

建议的四任务 mask不是容量结论，而是 oracle 起点：

| profile | RH56 OPERATE locator | OmniPicker STABILIZE locator | PATH |
|---|---|---|---|
| pour | FRAME；只有经证明roll不影响pour-plane时降为AXIS | POINT/AXIS/FRAME按jaw surface geometry | 默认禁用 |
| pipette/insert | FRAME；tip axis + plunger/key roll | 若稳定receptacle则AXIS/FRAME | 禁用 |
| cap | keyed/flat cap用FRAME；圆对称cap可AXIS | bottle body稳定用AXIS/FRAME | 禁用；twist不是surface path |
| wipe | PATH；normal+tangent+template+extent | tray/body稳定用AXIS/FRAME | 必需；不允许Omni执行自由path |

若一个 checkpoint支持多个 profile，必须按 profile分层报告 `L(p,g,s)`，并把 packet capacity写成 conditional on profile/text/state。profile ID若可逐样本变化且影响解码或controller，就必须作为显式 context计入系统信息边界；不能藏在配置文件里继续称同一个 finite route。

## 6. 标注、预测与 provenance 的唯一所有者

### 6.1 Human端

human可提供、经 A/B裁决后跨域共享的只有：

```text
phase / effect / primary role / part / secondary role / intended relation
```

human端禁止提供：

```text
robot uv/depth
robot axis/roll/surface normal/tangent
robot path primitive/extent
robot effector assignment
human hand side -> robot side mapping
human pixel/track/pose -> robot locator warp
```

human的bbox、point、track、entry/turn/exit pose可留在独立 human branch作事件/语义 auxiliary；跨域边界在 hard semantics处终止。不能因为 human与robot都使用512 directions，就认为同一个code跨相机具有同一几何含义。

### 6.2 Robot oracle来源

每个 translation/orientation/path field必须在 action-invisible source receipt中逐字段记录。可作为 oracle/GT 的来源只包括：

- synchronized simulator/native object、tool、surface和fixture state；
- calibration marker或直接测量得到的 robot-camera geometry；
- calibrated direct depth/stereo，且透明/反光条件下有有效性 receipt；
- 正式 target-robot-view A/B annotation；
- 固定 CAD/tool/fixture frame，前提是其**当前 pose**由上述直接来源确认；
- 由以上来源和冻结 codebook机械量化得到的 code。

`robot_locator_source_v1` 至少要绑定：anchor timestamp、camera/calibration hash、source tier、原始 support、量化 code、per-field review state、temporal skew、compiler/validator receipt和父 artifact hashes。

以下只能是 proposal/debug/baseline，不能升级为 oracle：

- monocular depth/normal completion；
- ClearGrasp式 completed depth、mesh、NeRF、inpaint或teacher fill；
- VLM猜测的rim normal、tangent、roll或path；
- human point warp、human motion direction或retarget pose；
- 无fresh receipt的 tracker/last-known frame；
- action planner自己的continuous target、logits或confidence。

### 6.3 Robot predicted来源

predicted route只能从目标机器人当前因果输入生成 hard codes：

```text
current robot RGB / valid direct depth or stereo
+ causal robot history
+ public proprio/contact/camera state
-> robot semantic resolver
-> categorical translation / axis / roll / primitive / extent heads
-> support-aware hard compiler
-> canonical locator packet
```

oracle与predicted必须分报。predicted confidence、logits、continuous surface fit、point cloud、normal map和path proposal只用于loss、校准和abstention；action/cache看不到它们。

### 6.4 透明、遮挡与 unknown

r1b required locator field没有 action-visible UNKNOWN，因此规则必须 fail closed：

1. transparent RGB只有2D overlap，不能推出depth、surface normal、rim frame、TOUCHING或INSIDE；
2. completed depth/normal可作 reconstruction challenger，不能填 oracle code；
3. occluded tracker只有在当前anchor有fresh causal receipt，且全部support落入同一task-safe code region时才可known；
4. support跨多个会产生不同安全动作的translation/axis/roll/path bins时，不取均值或最近码；
5. required field为 `REVIEWED_UNKNOWN` 或 `NOT_REVIEWED/MISSING` 时均不编 packet，control plane执行hold/reobserve并清除相关cache；
6. 全零NULL只表示 exhaustive review后确实没有 active event，不表示unknown、遮挡或sensor invalid；
7. 每 execute-8/replan anchor重算frame与path；不做last-value hold。

透明任务最诚实的结果可能是compile coverage下降。如果只有reconstruction、continuous rim pose或direct visual servo能达到安全公差，应让相应 challenger获胜，并停止 finite-frame transparent sufficiency claim。

## 7. 防止隐形 continuous residual

strict finite route的核心审计不是 tensor dtype，而是 action可读信息集合。以下逐项禁止：

- `hard code + continuous xyz/so3 residual`；
- codebook邻居插值权重、soft-argmax坐标或VQ commitment residual进入action；
- surface fit point/normal/tangent、path spline或waypoint hidden进入action/cache；
- planner/allocator logits、candidate order、score、confidence或ROI feature改变pre-safety action；
- controller读取一个未在packet/public state声明的continuous target pose；
- per-code learned metric offset、per-scene profile、sample-specific gain或trajectory lookup；
- held-object transform伪装成controller私有memory；若需要，它必须成为所有baseline共享的public robot-state input，容量只作conditional表述；
- unknown时复用上一locator、上一frame、上一path phase或cache KV；
- text携带pixel/depth/axis/roll/path数值或instance-specific trajectory。

允许的连续计算只有：

```text
hard code的固定中心/模板
+ immutable calibration/profile constants
+ 明确allowlisted public robot state
-> deterministic metric frame/path/current error
```

固定 packet、profile、public state、text和noise时，替换 raw robot reference、human视频、pair table、planner/allocator hidden、continuous proposal/residual或aux target，pre-safety action必须在预注册数值容差内不变。若外部 deterministic controller而不是WAM实际产生主要轨迹，则它必须作为独立 direct-geometry controller路线报告，不能把收益归给 finite packet。

## 8. RH56 / OmniPicker 四任务反例

| 任务 | matched counterexample | 单axis为什么碰撞 | 推荐locator与仍未解决的问题 |
|---|---|---|---|
| 倒液 | anchor和opening normal相同，但source handle/rim feature使安全倾倒平面绕normal旋转 | axis不知道绕轴roll；RH56腕姿和倾倒起始方向不同 | FRAME：`z=opening/rim normal, x=pour-plane tangent`。透明rim frame无direct证据则no-packet；液量/停止仍需sensor/supervisor |
| 移液/插入 | tip位置、插入轴相同，但plunger朝向或connector key绕轴相差90° | 同axis对应不同RH56拇指可达侧/腕roll | FRAME：`z=insertion axis, x=plunger/key direction`。仍需`ACTUATOR_OR_PLUNGER`和capability-gated LEFT+LEFT约束；frame本身不解决compound control |
| 旋盖 | cap中心和thread axis相同，但key/flat或可抓取feature绕轴不同 | RH56需要不同roll/grasp，右Omni仍需稳定bottle | keyed用FRAME，真正圆对称且oracle证明roll等价时用AXIS。OPERATE/STABILIZE功能、tighten/loosen和torque stop不由frame表达 |
| 擦拭 | anchor、path tangent和extent相同，但surface normal不同 | 腕姿和法向载荷方向不同 | PATH：`z=normal, x=tangent`。同frame/extent但曲线、障碍或区域shape不同仍可能碰撞；四模板外直接No-Go |

硬件不对称必须进入 profile mask，而不是用human side先验：

- 左 RH56六个主动通道适合 precision/roll、多指与plunger类操作，但真实 tactile/load/slip仍须单独标定；
- 右 OmniPicker只有一个 aperture action channel，适合body/tray稳定；两个物理夹面不等于两个独立控制自由度；
- OmniPicker未核实的 tactile、driver、canonical-to-device mapper和bench receipt不能在frame设计里假定已存在；
- 同一个 local frame能否由某末端到达由target capability/reachability检查决定，多解或不可达时abstain；allocator不能修改frame或发continuous correction。

## 9. Baseline、equal-bit 与干预

### 9.1 表示 baseline

所有路线共享 robot数据、planner supervision、execute-8/replan、public state/text、action head、参数/FLOPs预算和 failure set：

1. `axis_extent_r1b`：当前单axis+8 extent；暴露 roll/normal collision。
2. `axis_roll_frame`：本 memo 主方案，512×32 factorized frames。
3. `two_axis_enumerated`：先离线去非法/去重并报告真实count；不得按512²宣称容量。
4. `so3_16384`：冻结16,384个Hopf/grid SO(3) codewords，与主方案 orientation cardinality一致。
5. `normal_tangent_frame`：与主方案同wire，但以surface semantics训练/干预；检查语义是否带来收益。
6. `path_line_only` 与 `path_4template`：区分frame收益和path-shape收益。
7. `direct_current_visual_servo / F-PCD / full_continuous_trajectory`：高带宽控制充分性上界。
8. `reconstruction_action`：透明场景的completed depth/pose baseline；与strict route互斥。

RT-1式逐维binning、FAST式动作块压缩和Diffusion Policy式连续action sequence属于不同动作边界。它们是合理 control/upper bound，但不能与本locator concat后继续引用有限packet count。

### 9.2 equal-bit baseline

不能逐字段独立shuffle：那会产生非法 axis/roll/PATH/PAD组合，也会改变assignment/capability failure。建议在每个固定 `(profile, semantic key, locator kind, effector/function stratum)` 中：

```text
1. 枚举该stratum的 exact admitted locator set Omega
2. canonical rank: Omega <-> {0, ..., |Omega|-1}
3. split前冻结一个全域双射 pi_stratum
4. semantic route读取factorized canonical code
5. equal-bit route读取pi_stratum(rank)对应的opaque relabel
6. 两route共享同一admission、sample、assignment、failure、token/parameter/FLOP预算
```

置换发生在admission之后；不能重新运行allocator或用重排后的几何字段触发额外physical rejection。assignment、constraint function、presence和kind作为共同的安全/路由stratum保持不变，因此这个 control回答的是“局部frame的可解释factorization/邻接是否有用”，不是“整个packet所有语义是否有用”。若要做整包 equal-bit，必须在最终修订 grammar上重新枚举，不能复用旧r1b的 `pi` 或 `81.4779 bit`。

equal-bit路线是 encoder inductive-bias诊断，不替代控制充分性比较。finite main route必须另外接近 full/native trajectory 的安全动作集合；否则即使胜过opaque relabel也不能证明frame足够。

### 9.3 必做 paired-world干预

- `O-ROLL-INSERT`：translation/axis固定，只改roll；RH56 wrist/hand相关动作应改变。
- `O-PATH-NORMAL`：translation/tangent/extent固定，只改normal；腕姿/法向load方向应改变。
- `O-CAP-SYMMETRY`：圆对称cap的roll swap应动作等价；keyed cap的roll swap应不等价。
- `O-POUR-PLANE`：opening axis固定，只改pour-plane tangent；安全tilt corridor必须被区分。
- `O-PIPETTE-THUMB`：tip frame固定，只改plunger constraint；finger slice变化而tip/wrist保持在公差内。该项同时检查frame外的双LEFT grammar。
- `O-PATH-SHAPE`：anchor/frame/extent固定，LINE/ARC/RASTER对应不同安全path。
- `O-WIPE-OUTSIDE`：目标shape不属于四模板；compiler必须No-Go，不能nearest-template。
- `O-TRANSPARENT`：移除direct depth/normal证据；oracle packet必须消失，completion不能维持compile coverage。
- `O-RESIDUAL-BYPASS`：hard packet固定，替换continuous proposal/residual/profile private state；pre-safety action不变。

## 10. Go / No-Go 与停止条件

### Conditional Go：只批准 contract 与无训练 oracle

- 采用 `64×64×32 + AXIS/FRAME/PATH` 的候选结构；
- FRAME用 `512 axis × 32 roll`，surface语义固定为normal+tangent；
- PATH只用四个固定模板和八个全局extent bands；
- task profile只删状态，不改code语义；
- human只给共享语义，robot当前视角独立生成geometry；
- oracle/predicted/proposal分源，透明/遮挡unknown fail closed；
- locator结构计数按本 memo公式审计，整包容量等最终grammar完成后再算。

### No-Go：当前不能宣称

- 现有单axis r1b已满足倒液、移液、旋盖、擦拭；
- 16,384 frames或四个path templates天然满足精细操作公差；
- human视频能给robot depth/normal/tangent/roll/path GT；
- completed depth/normal是transparent oracle；
- task profile可逐样本改range、模板或controller；
- bin后加continuous residual仍是同一finite route；
- 本 locator修订已经解决constraint function、plunger part、双LEFT或held-state；
- `70,934,200,320`是整包states或部署reachable capacity；它只是未裁剪locator结构上界。

### 立即停止对应 finite claim

出现以下任一情况即停止，不继续堆字段保护结果：

1. translation、axis或roll covering error超过任一核心 task tolerance；
2. `O-ROLL-INSERT/O-PATH-NORMAL/O-POUR-PLANE`仍发生packet collision；
3. 四个path templates相对full/native trajectory的安全成功差距超过预注册阈值；
4. generic wipe需要mask、障碍map、多段track或自由waypoints；
5. transparent/occluded成功只在completion/reconstruction action-visible时出现；
6. predicted route必须把confidence、soft code、residual或planner hidden送给action；
7. profile裁剪后同一code出现多套metric解码，或profile由scene动态选择；
8. equal-bit control无法匹配admission/assignment/failure set；
9. RH56/OmniPicker真实hardware tolerance、driver、calibration或state source无法形成receipt。

## 11. 最终建议

对 SM-FET minimal-r1b，最小值得做 oracle 的局部表示是：**目标机器人当前视角的 `64×64×32` translation，加 `512 directed axis × 32 roll` 的规范局部 frame；roll-invariant任务显式降级为AXIS；surface motion再加 `LINE/ARC_LEFT/ARC_RIGHT/RASTER_3 × 8 extent`。** 物理语义写成 normal+tangent，wire factorization写成 axis+roll；不用raw two-axis，不把opaque SO(3) codebook当主线，但以相同16,384 orientation states作baseline。

task profile必须是global codebook上的冻结删除mask，而不是第二套decoder。human端止于role/part/phase/effect/relation，robot端在每个当前anchor用direct/native evidence重建 hard locator；透明/遮挡不确定就不编 packet。任何continuous residual、soft插值、per-scene path/profile或last-value cache都会使finite claim失效。

这一选择恰好修复单axis最直接的两类反例：绕轴roll和surface normal；它不假装用四个path templates解决所有擦拭，也不掩盖pipette compound、constraint function与held-state等独立schema阻塞。下一步只应做 source receipt、profile mask枚举和上述 paired-world oracle；未通过前不改模型、不训练、不上真机、不重算整包headline capacity。

## 12. Zotero-first / idea-discussion 登记

本轮按 `idea-discussion` 先同步个人 Zotero并创建独立 idea card：

```text
idea_id        = b6764f3e-92e1-41f1-bc95-2ea6709c43ac
discussion_id  = 22cac0b2-03d8-4e4d-8c0e-d4f9c52e0733
memory_updated = true
```

个人库命中以 FAST、RT-1、π0.5 笔记为主：

- `/home/ykj/Documents/ObsidianVault-dailypaper-skills/PaperNotes/多模态/FAST.md`
- `/home/ykj/Documents/ObsidianVault-dailypaper-skills/PaperNotes/多模态/RT-1.md`
- `/home/ykj/Documents/ObsidianVault-dailypaper-skills/PaperNotes/具身智能/Pi05.md`

它们分别支持“action tokenization需明确量化/重建边界”“逐维有限bin可用于真机策略”“高层离散预训练与连续action expert必须区分”，但都不直接回答局部frame选择。自动scout混入 IMDL/ActivityForensics 等明显低相关条目，因此这些结果只保留为检索审计，没有用于本 memo 的几何裁决。具体技术比较随后只核对原始论文/官方来源。

## 13. 原始参考

- Brohan et al., [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817), 2022. 每个动作维度离散为有限 bins；是离散控制先例，不是local-frame充分性证据。
- Pertsch et al., [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747), 2025. DCT+BPE压缩连续动作块；是高频动作challenger，不是固定语义packet。
- Shridhar et al., [Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation](https://arxiv.org/abs/2209.05451), CoRL 2022. 使用RGB-D voxel与离散6-DoF action（原实现按Euler轴5°离散）；支持finite spatial action precedent，但依赖3D observation/planner。
- Zeng et al., [Transporter Networks: Rearranging the Visual World for Robotic Manipulation](https://arxiv.org/abs/2010.14406), CoRL 2020. 以像素空间位移参数化操作并利用空间对称性；支持current-view keypoint baseline，不覆盖深度与完整frame。
- Zhou et al., [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035), CVPR 2019. 两个向量/6D可连续表示3D rotation；本设计借用“两个轴可定frame”的几何事实，但拒绝连续action residual。
- Yershova et al., [Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration](https://doi.org/10.1177/0278364909352700), IJRR 2010. 提供deterministic、低dispersion、近等体积SO(3) grids；用于16,384-state orientation baseline。
- Sundermeyer et al., [Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes](https://arxiv.org/abs/2103.14127), ICRA 2021. 将6-DoF grasp锚定到观测接触点并降低预测维度；支持local contact frame价值，但其depth point cloud与连续grasp不是本路线GT。
- Mo et al., [Where2Act: From Pixels to Actions for Articulated 3D Objects](https://openaccess.thecvf.com/content/ICCV2021/html/Mo_Where2Act_From_Pixels_to_Actions_for_Articulated_3D_Objects_ICCV_2021_paper.html), ICCV 2021. 预测actionable pixel及interaction orientation/trajectory；说明point alone不足，但属于RGB-D/continuous proposal路线。
- Chi et al., [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023. 连续、多模态、receding-horizon action-sequence上界。
- Sajjan et al., [ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation](https://doi.org/10.1109/ICRA40945.2020.9197518), ICRA 2020. 证明透明深度completion是成熟而独立的路线；completed depth/normal应作baseline，不能洗成strict finite oracle。

