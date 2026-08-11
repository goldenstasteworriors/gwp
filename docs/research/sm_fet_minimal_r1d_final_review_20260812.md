# SM-FET `minimal-r1d` 最终独立复审

- 日期：2026-08-12
- 唯一裁决：**Conditional Freeze（structural only）**
- 被审文档：`docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`
- 被审提交：`5019c1828cf17e169c51ecf6772ad16b23d51c2d`
- 审阅基线：上一轮 `minimal-r1c` 最终独立审阅及其 No-Go blockers
- 审阅方式：只读逐条复核、Python 标准库 CPU 小整数复算；未实现、未训练、未使用 GPU、未修改被审 worktree，也未创建永久测试脚本

## 0. 唯一结论

**`minimal-r1d` 的当前裁决是 Conditional Freeze，而且只适用于 hardware-family structural contract。**

r1d 已经实质关闭 r1c 的五类原则性 No-Go：RIGHT compound 不再按 assignment 名字全称删除；held transform 成为显式 public execution state；human shared key 改用 domain-neutral purpose 并机械派生唯一 focus；static profile、metric codebook、runtime resource、held state 四类 receipt 分层；structured text、proprio/public state、MONITOR 与 VERIFY topology 都有明确边界。

独立整数复算也全部通过。当前书面 grammar 的结构状态数确为：

```text
157,516,896,299,151,958,972,545,105,921
```

其 `log2` 为 `96.99142134161319 bit`，最小宽度仍为 97 bit。nonVERIFY 与 VERIFY 的新增/删除 block 没有漏计或重计。

但文档尚不能按原文直接作 unconditional Freeze。剩余问题均为**局部 exact-contract 缺口**，不要求新增 core field，也不要求在当前 topology 不变的前提下重算容量：

1. bundle-local rank `b` 没有定义 block offset、不同类型 pair 的局部 rank及 inverse/unrank；
2. POINT/AXIS 是部分约束，§4.3 却把所有 locator 写成完整 `SE(3)` 的 `T_C_P*`；VERIFY `part=N_A` 下 STABILIZE 的动作语义也未机械闭合；
3. RIGHT resource 表的 `target_class` 不含数值目标，但“same target merge / different target mutex”的 target identity 函数未定义；
4. 两个所谓 exact JSON 示例把枚举候选写进一个带 `|` 的 JSON string，hash 语义仍有两种合法读法；
5. metric codebook artifact 没有逐字绑定实际 `W,H` 与 axis vertical epsilon；
6. 文档自己列出的 codec/golden-vector CPU gates 尚未执行。

修正以上六项并通过对应 CPU checker 后，可以给 **Freeze structural**。无论这些条件是否完成，真实 metric tables、authoritative G1 capability/resource profile、runtime receipts 和任务 admission 尚不存在，因此：

> **G1 executable profile 仍是 No-Go；其整数/bits 仍为 UNDEFINED。**

## 1. 对 r1c No-Go blockers 的逐条复核

| r1c blocker | r1d 结果 | 本轮裁决 |
|---|---|---|
| held transform 被错误要求在同 packet 下永不改 action | §4.1 改为 fixed packet + fixed complete public state + fixed text/noise/timestep 才要求同一 learned pre-safety action；真实 held transform 改变会改变 public state，允许且通常要求 action 改变 | **关闭**；部分帧到 effector goal 的机械语义仍需补一条精确定义 |
| 任意 `RIGHT+RIGHT` 永久拒绝缺乏依据 | 新增 `RO-RS`、distinct `RO-RO` 与 VERIFY distinct `RS-RS`；冲突下沉到 aperture/wrist 等 resource claims | **原则关闭，exact target identity 有条件** |
| human key 混入 robot O/S/M、重复 multiplicity、每条 primary 都称 focus、roster 不完整 | 使用 `EFFECT_GOAL/SUPPORT_REQUIREMENT/OBSERVATION_REQUIREMENT`；duplicate 禁止；fan-out 在 robot binding；唯一 focus 机械派生；EXHAUSTIVE gate | **关闭**；enum JSON 需消除字面歧义 |
| `G_profile` 混入 fresh receipt、WBC/collision 与 current instance | §6/§9 将 static capability、metric codebook、runtime resource、held public state 分开，`G_profile(p,t)` 只读 immutable profile/task/wire payload | **关闭** |
| text/proprio/state/action boundary 与 codebook/codec 不 exact | structured text、canonical25、public state hash、denylist、camera/axis/roll/path生成规则和13-byte container均已列出 | **大部关闭**；rank `b`、W/H、epsilon 与 golden vectors 是冻结前条件 |
| MONITOR 是否应 action-visible | token/action attention 可见，own actuator gate 恒零；可改变同 bundle 其它 physical slice；missing/expired 时整包拒绝 | **关闭** |
| D0/CPU observability gates 未通过 | r1d 列出 31 条 gate，但明确写“本轮不实现” | **未关闭**；至少 exact codec/codebook/public-state 的 CPU gate 应在 structural Freeze 前通过，真实 profile/人机实验可延后 |

因此不再维持 r1c 的整体 No-Go，但也不能把文档 §13 的 “Freeze now” 当作已经完成验证的无条件结论。

## 2. CPU-only 容量与 topology 独立复算

本轮只使用 Python 标准库即时计算，没有写测试文件。基础量：

```text
P = 64*64*32
  = 131,072

L_nonverify
= P*(1 + 512 + 512*32 + 512*32*4*8)
= 70,934,200,320

L_verify
= P*(1 + 512 + 512*32)
= 2,214,723,584

N = 7*4*L_nonverify*22
  = 43,695,467,397,120

V = 1*4*L_verify*22
  = 194,895,675,392
```

### 2.1 nonVERIFY topology

每个 nonVERIFY phase/effect block 恰有：

```text
2 singles:
  LO, RO                         -> 2*N

7 different-type ordered pairs:
  LO-RO, LO-RS, LS-RO,
  LO-ZM, RO-ZM, LO-LS, RO-RS     -> 7*N^2

1 same-type unordered pair:
  RO-RO, q0<q1                   -> C(N,2)
```

所以：

```text
B_nonverify
= 2*N + 7*N^2 + C(N,2)
= 14,319,704,032,895,892,051,583,303,680
```

`RO+RS` 没有重复落入其它 block；`RO+RO` 只取 distinct payload 且 `q0<q1`，与 identical-target merge/dedup 不重叠。`LO+LO`、`LS+LS` 和其它未列 same-type pair 均未暗中计入。

### 2.2 VERIFY topology

VERIFY 的 `part=N_A` 且禁止 PATH，因此 payload alphabet 必须使用 dense `L_verify`，不能复用带 PATH 空洞的 nonVERIFY rank。每个 VERIFY block 恰有：

```text
3 singles:
  LS, RS, ZM                     -> 3*V

3 different-type ordered pairs:
  LS-RS, LS-ZM, RS-ZM            -> 3*V^2

2 same-type unordered pairs:
  ZM-ZM, RS-RS, q0<q1            -> 2*C(V,2)
```

所以：

```text
B_verify
= 3*V + 3*V^2 + 2*C(V,2)
= 151,937,297,146,405,128,765,440
```

VERIFY PATH 的排除已经同时反映在 `L_verify`、`V`、block count 和 dense locator rank中，未发现 topology 漏计。

### 2.3 总状态数、bit 与 affine diagnostic

11 个 nonVERIFY phase/effect blocks、1 个 VERIFY block及唯一 NULL 给出：

```text
|G_struct_r1d|
= 1 + 11*B_nonverify + B_verify
= 157,516,896,299,151,958,972,545,105,921

log2(|G_struct_r1d|)
= 96.991421341613192050699458...

bit_length(|G_struct_r1d|-1)
= 97
```

文档保留 14 位小数后的 `96.99142134161319` 正确；相对 r1c 增量为 `0.32036897249197693 bit`，文档的 `0.32036897249198` 也正确。

进一步复核：

```text
M = |G_struct_r1d|-1
  = 157,516,896,299,151,958,972,545,105,920

gcd(65537,M) = 1

65537^(-1) mod M
= 138,514,980,096,989,891,931,988,754,433
```

对 rank 端点与中点做 affine/inverse round-trip均通过；用 toy `n=23` 穷举 combinadic rank恰覆盖 `0..C(23,2)-1`。按当前 block sizes计算的最大 overall rank恰为 `|G_struct_r1d|-1`。

结论：**整数、96.99142134161319 bit、97-bit宽度、VERIFY topology 和 affine inverse可以条件冻结；当前 Conditional 不是算术否决。**

## 3. 必须修改 1：把 bundle rank `b` 定义成逐字唯一 codec

§10.3 已经定义 locator rank、payload rank、block顺序和 same-type combinadic rank，但 overall rank公式直接使用未定义的 `b`：

```text
1 + h*B_nonverify + b
1 + 11*B_nonverify + b
```

至少缺少：

- 每个 block 的累计 offset；
- single 的 local rank；
- different-type pair 的 row-major local rank；
- same-type pair 的 combinadic local rank如何加 offset；
- 由 `b` 反解 block、`q0`、`q1` 的 exact unrank；
- 13-byte LE 的 golden vectors，包括所有 block 边界。

冻结前应逐字补成等价于：

```text
single A:
  local = q0

different-type A-B:
  local = qA*nB + qB

same-type A-A:
  local = comb_rank(q0,q1;nA), q0<q1

b = cumulative_size_of_preceding_blocks + local
```

其中 nonVERIFY 的 payload cardinality用 `N`，VERIFY 用 `V`；block offset必须按文档现有顺序明确列出。Unrank必须证明是上述 rank 的左右逆。否则两个实现仍可能分别采用 `qA*N+qB` 与 `qB*N+qA`，都满足当前文字的 wire顺序，却得到不同 compact rank。

这不改变 grammar 或状态整数，但在补齐前不能冻结“rank/unrank 与 global codec”。

## 4. 必须修改 2：闭合 POINT/AXIS、VERIFY N_A 与 held transform 的控制语义

### 4.1 public execution state 不变量已正确

§4.1 的不变量现在具有正确作用域：

```text
fixed canonical packet
+ fixed complete public_execution_state bytes/hash
+ fixed structured text bytes/hash
+ fixed timestep/noise
-> same learned pre-safety action
```

它明确允许真实 held transform/contact/camera pose变化导致 public state和正确 action变化，也把 receipt metadata 与 physical state分开。这与正确闭环控制不再冲突，r1c 最重要的 blocker 已关闭。

四类 receipt 的表也保持纯分层：static profile 不含 current values；metric codebook 不混 task/scene geometry；runtime receipt 可改 admission/assignment而不改 profile；held physical values经 public state action-visible，receipt metadata不可见。未发现 allocator、profile 或 receipt 再次形成 hidden correction channel。

### 4.2 剩余的是 locator manifold，而不是 held-state 原则

§4.3 把 wire locator统称为完整 controlled-part target `T_C_P*`，再计算：

```text
T_C_E* = T_C_P* @ inverse(T_E_P)
```

这个方程对 FRAME/PATH 成立；POINT没有orientation，AXIS没有roll，因此两者只定义目标 manifold/部分约束，不能直接代入唯一 `SE(3)` transform。

冻结前应明确二选一，并写入 deterministic/profile contract：

1. POINT/AXIS 保持部分约束，action expert把它们解释为 goal manifold，不宣称唯一 `T_C_E*`；或
2. 给出只依赖 frozen profile + complete public state 的 deterministic completion，例如 POINT保留当前orientation、AXIS保留由当前state机械计算的roll，并规定退化/tie/uncertainty时 abstain。

不得由 allocator hidden state、receipt metadata或private geometry补 orientation/roll。

VERIFY 的 `part=N_A` 同样需要一条动作语义：MONITOR 可以只验证 role-level fact且 own slice为零；但 LS/RS 会打开 effector slice。规范必须说明它们是在保持哪个公开定义的目标/当前 pose，或者声明所有尚无 deterministic action mapping 的 VERIFY STABILIZE tuples 在 executable profile 中删除。当前禁止 VERIFY PATH 是正确且已计数，但仅禁止 PATH 还不足以唯一解释 `N_A + STABILIZE + POINT/AXIS/FRAME`。

这两项都不要求修改 structural alphabet；它们决定 profile admission与decoder，不补齐就不能称 semantic contract完全冻结。

## 5. 必须修改 3：RIGHT target identity 要能机械决定 compatible/mutex

r1d 对 r1c 的修正方向正确：单一 OmniPicker aperture只证明两个不同 aperture actuation targets 冲突，不证明任意 RIGHT pair 都冲突。结构层加入 `RO-RS`、distinct `RO-RO` 与 VERIFY distinct `RS-RS` 是有依据的；static profile再依据 resource compatibility取子集也与 `G_profile(p,t)` 分层一致。

剩余 exact 缺口在于 §2.2 规定 `target_class` 不含 pose/aperture数值，而 §2.3 又要求：

```text
same target -> merge/deduplicate
different targets -> mutex or profile-compatible
```

同一个 `target_class` 下的两个不同 locator/payload 仍可能要求不同 wrist pose或aperture target。若 compatibility只比较 class，会误把它们当“same target”；若每个实现自行从 continuous controller goal比较，又会破坏 finite、static、exact profile。

冻结前应定义一个有限、确定的 resource target identity，例如：

```text
resource_target_id = F(p, t, canonical wire payload, resource_id)
```

并规定：

- `F` 只能读 immutable profile/task stratum和canonical wire payload；
- `F` 的有限 codomain、canonical equality与 UNKNOWN规则写入 profile；
- identical `resource_target_id` 才可 merge/`HOLD_SAME_TARGET`；
- distinct aperture targets恒mutex；distinct wrist targets默认mutex，除非 profile显式证明 disjoint compatible subspace；
- runtime reachability/contact/WBC/collision只能继续裁剪，不可回写 `F` 或 `G_profile`；
- 每条 physical O/S constraint必须至少声明一个与其动作相关的 resource claim，不能以 `NO_CLAIM` 绕过 same-effector检查。

这只收紧 profile-compatible relation，不改变 structural count。真实 RIGHT compound 是否可执行仍应保持 G1 profile No-Go，不能由结构可表示性反推已支持。

## 6. 必须修改 4：human shared key与 text JSON 消除 `|` 字面歧义

human shared key的 domain-neutral purpose、S↔M projection、duplicate prohibition、robot-layer fan-out、唯一 focus与 EXHAUSTIVE roster已经闭合 r1c blocker。geometry、assignment、instance、held transform和pairing outcome也都明确不得进入 key/cost。

但 §5.2、§5.4 与 §7.1 在“Exact JSON object”中写了：

```json
"task_family": "POUR|INSERT|PIPETTE|CAP|WIPE"
```

其它 enum字段也使用同样写法。按 JSON 本身，这表示一个包含竖线的单一 string；按自然语言，它又像表示五个候选字符串。由于随后对 RFC 8785 bytes做 SHA256，二者会得到不同 hash，所以不能留给实现猜测。

冻结前必须用 JSON Schema `enum`、逐项 ordinal表或明确的 one-of 规范表示“恰选一个完整字符串”，并说明示例中的值是一个实际候选而不是候选集合。NULL key也应给一个真实 task family示例或 schema-level metavariable。structured text object同步修正。

这不改变 packet容量，但属于 exact semantic/text hash冻结前的必修项。

## 7. 必须修改 5：FRAME/PATH codebook artifact补齐 W/H 与 epsilon

现有 FRAME/PATH 组合方向正确：

- camera frame、UV/depth tie rule、directed axis、roll zero basis与symmetry处理已给出；
- PATH 使用完整 frame，`z` 为固定 surface normal，`x` 随解析/折线路径 tangent transport；
- RASTER vertex tangent、final vertex和extent缩放均有规则；
- PATH 在 controlled-part frame生成，再通过公开 held transform得到 effector goal；
- VERIFY PATH因 `part=N_A` 而结构拒绝。

这里没有发现 axis/roll/primitive/extent组合计数错误。不过 metric artifact schema仍少两个会影响跨实现逐字一致的值：

1. translation decode显式使用 `W,H`，manifest当前只说“W,H normalization rule”，`tables.bin`只列 camera intrinsics；必须绑定 exact positive integer `W,H` 或绑定一个包含它们的 immutable calibration record/hash，并明确读取字段；
2. axis quantizer对严格竖直/`hypot(x,y)`小于 `codebook epsilon` 走 `a=0`，但 epsilon的数值、类型和bytes未列入 manifest/tables；必须冻结为 exact float/比较规则并纳入 checksum。

真实 depth/extent edges、axis materialized table和G1 calibration hash可以延后到 executable profile；但用于产生/验证工件的 schema字段本身应先补齐。

## 8. MONITOR、missing/transparent 与旁路复核

### 8.1 MONITOR zero-slice闭合

MONITOR 不是 private supervisor bit：它占合法 constraint/locator token、进入 action attention并可改变同 bundle其它 slice；其 own actuator ownership mask恒零。Fresh evidence缺失或过期时整包拒绝，不能静默 drop MONITOR后继续 OPERATE。这同时满足“action-visible”与“自身不直接占执行器”的要求。

因此本轮不要求给 MONITOR 独立动作 slice，也不要求把它移出 packet。只有未来干预证明它从不改变任何 physical slice、只影响 admission时，才应在下一版另行重构并重计。

### 8.2 missing/transparent与完整 roster

EXHAUSTIVE no-active才允许 NULL；partial roster不编伪 single oracle；required field unknown/expired清 packet和旧 cache；transparent 2D overlap不补 depth/normal/relation；completion/reconstruction只可作独立 baseline。POINT不要求normal、AXIS不要求roll的 field-level missing规则与其 locator kind一致。

这些规则已经关闭 r1c 的 incomplete-roster与transparent旁路问题。

### 8.3 text/proprio/held-transform旁路

action route只读 hard tokens、complete public execution state、structured text、noise/timestep。canonical25字节布局固定；camera/held state有显式revision/hash；text禁止数字 geometry、locator alias、assignment、profile selector、trajectory/gain；allocator hidden score/order与 receipt metadata不可见。Safety plane只可 clamp/hold/retract，若产生主要 task trajectory必须另报 direct-controller baseline。

除第6节指出的 enum字面问题外，未发现新的 text/proprio/held-transform private control channel。

## 9. Structural Freeze 前的最小验收清单

在不改字段与本轮 topology 的前提下，以下项目完成后可把本裁决升级为 **Freeze structural**：

1. 补全 bundle `b` 的 block offsets、不同类型 pair rank、same-type rank、完整 inverse/unrank；
2. 明确 POINT/AXIS 的 partial-goal manifold或 deterministic completion，并闭合 VERIFY `N_A + STABILIZE` 的动作/profile拒绝规则；
3. 定义 finite `resource_target_id` 与 exact RIGHT merge/mutex/compatibility判定，禁止 physical O/S 以空 claim绕过；
4. 把 shared semantic key和structured text的 pipe-string改成无歧义 one-of enum规范；
5. 在 metric codebook artifact schema中绑定 exact `W,H` 与 axis epsilon；
6. 实际运行 CPU-only rank/unrank、13-byte LE、block边界、public-state/text canonicalization、axis/frame/path与 checksum golden vectors，并保存可复核结果。

若上述修订改变 bundle topology、payload cardinality或 VERIFY locator family，则必须重新计数；若只按本 memo补 exact定义，当前结构整数和97-bit宽度不变。

## 10. 可以延后的项目

以下项目不应被误报为已通过，但可以延后到 executable-profile/实验阶段，不阻塞上述窄义 structural contract在条件满足后的冻结：

- 真实 G1 camera calibration、depth/extent edges、materialized tables与metric codebook hash；
- authoritative G1 capability profile、working surfaces、resource claim templates、RIGHT/LEFT compound allowlist和 profile integer/bits；
- held/contact capture、RH56/Omni driver、WBC/collision/runtime monitor receipts；
- `G_profile(p,t)` 的实际枚举、target-profile equal-bit permutation和task-family逐层报告；
- D0/D1/H1/R0/P0/B0、paired-world oracle、human可标性、仿真、训练与真机验证；
- transparent pour、pipette compound、roll-sensitive insertion和generic wipe的 task sufficiency。

其中任何一项若未来证明现有字段/grammar不足，必须升 schema revision；不能以“structural frozen”掩盖新的 collision。

## 11. 最终裁决

**唯一裁决：Conditional Freeze（structural only）。**

`minimal-r1d` 的状态整数、96.99142134161319 bit、97-bit宽度、RIGHT/VERIFY bundle topology、public-state不变量、human shared purpose/focus/roster、四receipt分层、MONITOR zero-slice和主要旁路边界均通过本轮复核。必须先补齐 bundle rank/unrank、partial-frame/VERIFY STABILIZE、RIGHT finite target identity、JSON enum、W/H/epsilon，并让 CPU golden gates通过，之后才能写成 unconditional **Freeze structural**。

即使升级为 Freeze structural，也仍必须逐字保留：

> **G1 executable profile = No-Go；`|G_profile(p,t)|` 与其 bits = UNDEFINED。**
