# SM-FET `minimal-r1c` 最终独立审阅

- 日期：2026-08-12
- 唯一裁决：**No-Go**
- No-Go 范围：当前 `minimal-r1c` 文本的“最终 structural freeze”；不是否定其作为下一版候选，也不是说已复算的整数算错
- 被审文档：`docs/research/sm_fet_minimal_r1c_schema_capacity_20260812.md`
- 被审提交：`39bc2b4b1c4fa822cc5565e8d23081f628723151`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 审阅方式：只读对照文档与 CPU 小整数复算；未实现、未训练、未使用 GPU、未修改被审 worktree

主要规范对照：

| 文档 | 提交 |
|---|---|
| `sm_fet_minimal_r1b_control_review_20260812.md` | `144d050ddfcb3d4a3fa0dc5f299754177c63f04b` |
| `sm_fet_minimal_r1b_crossdomain_review_20260812.md` | `49667be32d15bd95fd2bde610b3e251c2fcc8474` |
| `g1_rh56_omnipicker_sm_fet_capability_profile_20260812.md` | `febccbb209d429f523562ac0119d06a3c421a6df` |
| `sm_fet_finite_local_frame_design_20260812.md` | `39adb119b1e039736e5404150ed9f3fa1e10d358` |
| `sm_fet_d0_observability_independent_review_20260812.md` | `189dfa8a2ca24b1d9e333addb2d393af90704ac6` |
| `sm_fet_dynamic_focus_token_spec_20260812.md` | `2452e60bbaf6edb4714fed7d1397aea6ba9e4111` |
| `sm_fet_missing_hands_contract_20260812.md` | `ca7d641703749a15ff30c644250f5147b5125e72` |
| `sm_fet_transparent_nonreconstruction_contract_20260812.md` | `f795c4f615595d72219e79a0cb43d5adef12b0a0` |
| `sm_fet_wam_v2_schema_architecture_20260812.md`（standard-r1） | `190f3023da6ea6e0ee8251da22e2f21efea17bcc` |
| `sm_fet_annotation_and_pairing_protocol_20260812.md` | `288ef5dcf4e6299871dc50d33caa603e46345fcd` |

## 0. 最终结论

**当前唯一结论是 No-Go。**

`minimal-r1c` 已经把 r1b 的主要结构缺口——显式 function、plunger/pad part、full frame、surface path、显式 target assignment 和 profile-gated LEFT compound——放进了一个相当接近可审计规范的候选中。以下部分通过本轮独立复核：

- 32-byte field wire 与 5-token slot 数自洽；
- 在文档当前列出的 bundle grammar 下，locator、payload、single/double bundle、总状态数、`log2`、97-bit 最小索引宽度和 affine 逆元均复算一致；
- `PATH` 强制携带完整 frame，`z=surface normal, x=initial tangent` 的组合方向正确；
- human geometry 与 target-robot geometry/assignment 分层、unknown/missing/transparent fail-closed、单一冻结 profile、allocator 只输出 hard assignment 的方向正确；
- 当前不报告 G1 executable profile 整数和 bits 是正确且必要的。

但当前不能冻结为最终结构，原因有五个会触及核心 schema 或信息边界的 blocker：

1. `held transform` 的 swap invariant 被写成过强的系统级控制不变量，与既有 capability/control contract 对 variable held state 的允许方式冲突；
2. “任意/永久 `RIGHT+RIGHT` 拒绝”没有由“单 aperture”推出 assignment-level 全称命题，修订可能改变 bundle grammar 和总容量；
3. human-shared key 把 `STABILIZE/MONITOR`、重复 constraint multiplicity 和“每条 primary 都是 focus”直接冻结，未解决跨具身执行差异、唯一 event focus 与 incomplete roster；
4. `G_profile` 与 `R_runtime` 在 same-effector gate 中混入 fresh receipt、当前 WBC/collision 和 role instance，导致所谓 exact profile formula 的集合边界不纯；
5. action-visible public state、text、camera pose、contact/load 与 safety controller allowlist 未闭合，且 axis/path/codebook 与 compact rank 的“exact”工件尚未物化。

这些不是可在冻结后补注释的小问题。第 2 项可能改总整数；第 1、3、4、5 项会改变同一 packet 所条件化的动作函数或 pairing admissibility。因此本轮不能给 `Conditional Freeze`；必须先修订规范、必要时重算，再做一次独立 final review。

## 1. 分项裁决

| 检查项 | 裁决 | 说明 |
|---|---|---|
| 字段布局、32 bytes、5 tokens | **Pass with exact-spec fixes** | byte offsets自洽；compact index byte order、完整 block order和 mixed-radix公式未完全冻结 |
| 当前 grammar 下容量算术 | **Pass** | 未发现整数、bits、ceil 或 affine inverse 错误 |
| `constraint_function` action-visible | **Pass** | O/S/M 必须进 hard token；`MONITOR`不开自己的动作 slice，但仍可条件化其它 slice |
| profile-gated LEFT O+S | **Conditional Pass** | exact allowlist/resource/oracle方向正确；static profile 与 runtime receipts 必须拆开 |
| assignment-level 永久 `RIGHT+RIGHT` 拒绝 | **No-Go** | 单 aperture只直接证明两个独立 aperture target冲突，尚未证明所有 RO+RS 都冲突 |
| FRAME/PATH 组合与结构计数 | **Pass with artifact gate** | kind/PAD/cardinality正确；axis centers、path polylines、extent/depth centers与hash未齐 |
| human semantic / robot geometry 分层 | **Direction Pass, key No-Go** | geometry/assignment分层正确；function、multiplicity、focus和roster规则未闭合 |
| missing / transparent | **Conditional Pass** | fail-closed与清 cache正确；缺完整 constraint roster gate，normal规则需按 locator kind写 |
| allocator/profile 防旁路 | **Mostly Pass** | hard assignment与单profile正确；receipt swap范围、controller lookup和public state仍需收紧 |
| text/proprio/state 防旁路 | **No-Go** | `allowed structured text`和`25D proprio/state`没有字节级 allowlist；camera/contact/load边界不明 |
| held-transform 防旁路 | **No-Go as written** | downstream隔离可以成立；把 variable physical state一律要求 same-action 不成立 |
| `G_profile` exact formula | **Formula shape Pass, set semantics No-Go** | 求和形状正确；static/runtime predicate混层且“reachable”措辞错误 |
| standard-r1 对照 | **Pass with wording fixes** | standard整数/bits复核一致；不能由此称 r1c 为“唯一 final candidate” |
| D0 / human 可标性 | **Not passed** | D0 checker不存在，D1未执行；新 function/focus/constraint roster更未被A/B验证 |

## 2. CPU-only 容量复算

本轮用 Python 标准库的一次性命令行整数计算复算，没有创建测试脚本或临时文件。按被审文档当前 grammar：

```text
P = 64*64*32
  = 131,072

q = 1 + 512 + 512*32 + 512*32*4*8
  = 541,185

L = P*q
  = 70,934,200,320

V = 4*1*L*22
  = 6,242,209,628,160

N = 7*V
  = 43,695,467,397,120

B_nonverify = 2*N + 6*N^2
            = 11,455,763,226,316,748,597,640,560,640

B_verify = 3*V + 3*V^2 + C(V,2)
         = 136,378,133,646,642,524,651,520,000

S = 1 + 11*B_nonverify + B_verify
  = 126,149,773,623,130,877,098,697,687,041

log2(S) = 96.67105236912121
bit_length(S-1) = 97
gcd(65537, S-1) = 1
65537^(-1) mod (S-1)
  = 89710184315419345683536084993
```

相对数值也一致：

```text
r1c - r1a      = 76.20835739911574 bit
r1c - r1b      = 15.193149790979987 bit
standard - r1c = 111.07869455039835 bit
```

另以 `V_toy=17` 穷举 `q0<q1` 检查 combinadic rank，恰覆盖 `0..C(17,2)-1`；locator variant端点为 `q=0..541,184`；对 rank边界与中间样本做 affine/inverse round-trip也全部通过。

因此不能把本轮 No-Go 解释为算术否决。准确结论是：

> `126,149,773,623,130,877,098,697,687,041` 是当前书面 grammar 的 exact syntactic/structural code count；只有 bundle grammar 保持不变时，这个整数才能继续冻结。

如果修订后增加 profile-gated `RO+RS`、改变 VERIFY part/locator规则、改变 MONITOR payload family，必须从 `B_nonverify/B_verify` 开始重算，不能只改文字。

## 3. Blocker 1：held transform 不变量必须改写

### 3.1 当前文本的两个不同层次

被审文档 §4.5（L282–286）给出一条可以成立的有限编译路线：

```text
primary controlled-part target frame/path
+ fresh T_effector_to_controlled_part
-> assigned-effector target execution frame/path
-> finite r1c locator or abstain
```

如果该映射、量化 support 和 task tolerance全部闭合，held transform 的作用可以在 action expert 之前被压入 hard locator。此时 downstream WAM 不读 receipt 是合理的。

但 §6.3（L378–390）、invariant 24/25/38（L766–767、L783）又把它扩大成：同 packet/proprio/text 下替换任何 held receipt都不得改变 pre-safety action，post-WAM controller也不得读 transform；若正确动作应变就直接判 packet collision。这个表述没有区分：

- receipt 的 hash/provenance 元数据；
- receipt 所证明的当前物理 transform/contact state；
- 编译前 packet/admission；
- 编译后的 learned action expert；
- 需要实时反馈的 safety/low-level control plane。

### 3.2 与既有 contract 的冲突

`g1_rh56_omnipicker_sm_fet_capability_profile_20260812.md` §6（commit `febccbb`，L263–294）允许 held transform 作为公开、allowlisted、所有 baseline共享的 external robot-state condition；若 action expert/controller实际读取它，容量必须明确写成 conditional，而不是伪装成 packet 内 bits。其 held-state audit（L483–490）进一步要求：若 action 使用该 state，transform变化应按 external-state contract改变 action。

`sm_fet_finite_local_frame_design_20260812.md` §7（commit `39adb119`，L359–382）同样允许：

```text
hard code center/template
+ immutable calibration/profile constants
+ explicitly allowlisted public robot state
-> deterministic metric goal/current error
```

`sm_fet_minimal_r1b_control_review_20260812.md` §6.2 和 `sm_fet_minimal_r1b_crossdomain_review_20260812.md` §10.3 也给出三选一：进入 public state、固定机械常量、或 fail closed。r1c 可以选择更窄的“先编译进 effector goal”路线，但必须明确声明这是对 variable-held-state route 的收缩，而不是把“正确控制可依赖公开 current state”重新定义为旁路。

### 3.3 对“same packet 不得因 held transform 改变 action”的直接回答

**作为 downstream action-module 隔离测试，它正确；作为端到端控制系统的不加条件不变量，它错误。**

正确作用域应是：

```text
给定：
  fixed canonical packet
  fixed complete declared public robot state
  fixed allowed text/noise/timestep
  successful compile/admission

替换：
  action-invisible receipt metadata，或不改变上述公开物理状态的等价receipt

要求：
  learned pre-safety action不变
```

而不是：

```text
只固定25D和packet，任意改变当前真实held transform/contact/load，
仍强迫整个控制链输出相同动作。
```

对 variable held binding 必须冻结三种互斥路线：

1. **Rigid constant**：transform是同一 stratum 的固定标定常量；
2. **Compile-and-quantize**：transform在编译前完全映入 effector goal locator，且整个 receipt support 在同一 code 内的 safe-action intersection非空并通过 task tolerance；否则 abstain；
3. **Public-state conditional**：若正确反馈仍需当前 transform/contact/load，则把它列为所有路线共享的显式 public robot-state input，容量写成 conditional；若这会让外部几何 servo产生主要轨迹，则单列 direct-geometry baseline，不归功于 finite packet。

第 2 条中，held state改变可以改变 packet或 admission；第 3 条中，同 packet下 action可以随已声明 public state正确变化。只有在“packet与完整 public state都固定”时才要求 receipt swap action-invariant。

## 4. Blocker 2：`RIGHT+RIGHT` 的依据不足以支持全称结构拒绝

被审文档 L30、L321、L344、L746、L771 把任意独立 `RIGHT+RIGHT` 从 `G_struct` 永久删除，理由是 OmniPicker只有一个 aperture。

capability memo确实支持下列窄命题：

- `RIGHT_OMNIPICKER_APERTURE` 是一个 coupled exclusive resource；
- 两个独立 aperture target 为 `UNSUPPORTED_BY_TOPOLOGY`；
- 等价目标应 merge/deduplicate，而不是形成两条constraint。

但同一 memo §5.1 同时把 `RIGHT_WRIST_POSE` 与 `RIGHT_OMNIPICKER_APERTURE` 列为两个不同 resource（L200–201），其 §5.2 的规则是按 `resource_claim_set` 判断冲突（L211–218）。§5.3 对 RR 的原句也是“两个独立右侧 constraint都会竞争单一 aperture，**通常**还竞争 wrist”（L248–257）；它没有证明每个结构上合法的 `RO` 和 `RS` 都必然同时独占 aperture。

因此“单 aperture”能直接否决的是 resource-level 双 aperture target，不是尚未声明 resource claims 的 assignment-level 所有 `RO+RS`。例如从当前书面字段无法排除“wrist-dominant OPERATE + aperture-hold STABILIZE”是否可合并、是否冗余、或是否需要一个 exact compound；这必须由规范证明，不能由 assignment名字推断。

冻结前只能二选一：

1. 给出规范性拓扑定理：每个 `RIGHT_OMNIPICKER` physical constraint无例外地独占哪些资源，证明任意两个非等价 payload必 mutex；保留当前 grammar 与整数；或
2. 把可能的 `RO+RS` 放入 structural representability，再由 exact right-compound allowlist/resource compatibility裁剪；随后重算 `B_nonverify`、`G_struct` 与 `G_profile` 公式。

在完成其中一个之前，当前 `6*N^2` double block和总整数不能被称为最终冻结值。

## 5. Blocker 3：human-shared key、focus 与 roster 未闭合

### 5.1 O/S/M 对 robot action 必要，不等于三者天然跨具身同值

capability memo §3.1（L100–122）已经充分说明 `constraint_function` 必须 action-visible：否则“左旋 cap、右稳 bottle”发生碰撞。被审文档把 function放进 constraint token是正确的。

但被审文档 L82–109 又把 `OPERATE/STABILIZE/MONITOR` 原样放入 human/robot exact semantic key。这里漏掉一个跨具身反例：同一功能上的“target 被支撑”在 human 中可能由第二只手主动 `STABILIZE`，在 robot 中可能由锁定 fixture的 fresh evidence表示为 `MONITOR`；反向情况也可能发生。`STABILIZE/MONITOR` 描述的是目标 embodiment 的执行实现和资源占用，不总是 domain-neutral task semantics。

冻结前必须选择：

- shared key使用更抽象、域中立的 `constraint_purpose`，robot compiler再把它解析为 action-visible O/S/M；或
- 保留 O/S/M exact equality，但给出机械 human annotation/projection规则和 A/B gate，明确哪些跨具身 S↔M pair应被拒绝，且证明不会系统性删除有效 pairing。

目前 D0 review（commit `189dfa8`，L16–21、L160–206、L277–289）明确说 event、focus、两条 edge 与完整 roster都没有通过 A/B，D0 checker也不存在；新增的 O/S/M 更没有可标性证据。因此“human/robot共享 function 已冻结”过强。

### 5.2 重复 semantic entry 泄漏 target-resource multiplicity

被审文档 L109 允许两个完全相同的 semantic entries，理由是未来可分给不同robot resources。这个理由本身把 target-robot resource multiplicity带进了所谓 domain-neutral human key：human看不到未来 assignment，也无法仅凭两个完全相同、无 instance ID 的 tuple判断应写一次还是两次。

必须规定重复的可观察身份和 completeness receipt；否则：

- 单条与双重复 key 可表示同一 human语义却 hash不同；
- robot需要两个资源时会要求human凭 target topology复制语义；
- duplicate rejection、multiplicity与 pair equality失去机械可复现性。

推荐默认禁止完全相同 shared entries；若确有两个不同功能约束，应由 role/part/relation/purpose至少一个字段区分。若确实需要 embodiment-specific duplication，应放在 robot binding层，不进入 exact shared key。

### 5.3 “每条 primary 都是 focus”与唯一 event focus冲突

`sm_fet_dynamic_focus_token_spec_20260812.md`（commit `2452e60`）把 focus定义为每个 active event唯一的 role（L113–135），并给出 anchor edge唯一派生规则（L186–205）。r1c却在每条constraint上写 `primary_role # focus`；双constraint因而可能有两个不同 primary，也可能有两个 OPERATE，无法恢复唯一 event focus。

冻结前必须显式声明：

- r1c正式废弃 event-level dynamic focus，只保留 per-constraint controlled primary；此时删除“primary is focus”措辞并说明对 focus spec 的 supersession；或
- 给出从 1..2 constraints机械导出唯一 event focus的规则，并处理 `LO+RO` 双OPERATE冲突；若需新增字段则重算容量。

### 5.4 删除 human side 不等于删除完整 roster gate

missing-hands contract（commit `ca7d641`）要求只有完整 roster + 完整 reference + grammar-valid packet才能进入 oracle action branch（L165–195），并明确“known left + unreviewed right”不能形成假的 single constraint（L589–595）。

r1c L407–413 只检查 present constraint 的 required fields，未要求 human constraint roster/exhaustive coverage。删除 anatomical side可以避免错误映射左右，但不能证明没有未观察到的第二条 stabilization/support constraint。

shared semantic key admission必须新增 action-invisible completeness gate，例如：

```text
constraint_roster_coverage = EXHAUSTIVE
or event ontology explicitly object_only/none and reviewed
```

否则 partial positive semantics可用于 planner field loss，但不得冒充完整 1-constraint pair key或 oracle packet。

## 6. `MONITOR` 应不应该 action-visible

**应该。** 前提是它表示一个会改变当前操作策略的、fresh verified passive constraint，而不是 unknown、missing、hold-last或 inactive stabilizer。

被审文档已经把 `MONITOR` 放进 hard constraint token，因此它是 action-visible；“不开动作 slice”只表示没有第三个物理执行器，不能解释成 action expert忽略该 token。一个 monitor可以通过支撑/fixture事实改变现有 `OPERATE` slice的安全动作。

必须补两个不变量：

1. 固定 OPERATE constraint，在“被动支撑已确认”与“没有该支撑/必须 active stabilize”两个 matched worlds中，packet/admission或 operating action必须按预注册语义变化；
2. 若 MONITOR只供 supervisor admission、从不允许影响任一 physical slice，则它不应计入 action alphabet，应移到 receipt/control plane并重算。

此外，`MONITOR_ONLY` 不开 slice与“交换 assignment只改变对应 slice”的 invariant不能混写。function/assignment联合 swap可能改变其它 slice的条件上下文；测试必须固定完整合法 bundle，而不是对单个 byte作无语义 swap。

## 7. Blocker 4：`G_profile` 与 `R_runtime` 混层

被审文档 §1 对三个集合的顶层定义是正确方向：

```text
G_struct  = canonical structural grammar
G_profile = frozen finite static profile predicates
R_runtime = current state/receipt dependent admission
```

但 same-effector gate L337–342 把“fresh held binding、WBC balance/self-collision receipt”写成进入 `G_profile` 的 iff 条件；profile compatibility L620 又包含 role-instance conflict。freshness、当前实例、当前 reachability/WBC/collision/contact都随 anchor变化，应属于 `R_runtime`，否则 `|G_profile|` 不再是只由 immutable profile bytes复算的整数。

必须改成：

```text
G_profile static:
  exact part/function/binding-kind allowlist
  finite working-surface/resource-claim templates
  static mutex graph and symmetry
  task-family rule
  calibration/oracle evidence that authorizes a new immutable revision

R_runtime per anchor:
  concrete role-instance binding
  held receipt freshness/support
  contact/load validity
  reachability, WBC, collision and safety predicates
```

paired-world oracle可以成为升级 profile revision的离线证据，但不能作为每个 state membership计算中的隐含变量。`C_p/U_p` 的兼容谓词必须是纯有限函数 `f(p,x,y)`；任何 scene/instance/receipt输入都不得进入。

相应地，L648 的“exact structural reachable alphabet”应改为 **exact profile-valid structural alphabet**。物理 reachable集合已经由 `R_runtime` 表示，不能复用“reachable”一词。

### 7.1 task family / text 也必须进入公式的条件索引

当前 `Omega_h_*` 只按 profile `p` 和 phase/effect block `h` 定义，但 `profile_static_compatible` 又要检查 task-family rules；同一 `biochem4` profile覆盖多个task family，而 task family只出现在 semantic key/allowed text，不在32-byte wire中。此时“同一wire tuple在哪个task中admitted、映射哪个controller enum”不能只由 `(p,h,x,y)` 唯一决定。

必须选择并写进公式：

- 固定 task/text stratum，定义 `Omega_{p,t,h,*}` 与 `G_profile(p,t)`；或
- 定义跨 task 的 canonical wire-byte set并集并去重；如果 task tag同时直达 action并改变同一wire的控制语义，则它是显式条件，不能把各 task cardinality直接相加或仍只写 `G_profile(p)`。

这与“text是合法条件输入”一致，但必须把条件变量写在 exact count 的下标里，否则所谓 exact profile alphabet缺少一个自由变量。

## 8. Blocker 5：FRAME/PATH 与 exact code artifacts 尚不足以冻结语义

### 8.1 已通过部分

以下组合正确：

- POINT：translation only；
- AXIS：translation + directed axis；
- FRAME：translation + axis + roll；
- PATH：translation + full frame + primitive + extent；
- PATH的 `z=normal, x=tangent`；
- 每translation `1 + 512 + 512*32 + 512*32*4*8 = 541,185`。

这修复了 r1b 的 roll、surface normal和path shape collision方向，且没有把 human geometry带进robot locator。

### 8.2 尚缺的冻结工件

被审文档 L232 自己要求 512 axis table物化/hash，但当前没有 table bytes/hash；四条 path也只有名字与定性约束，没有 canonical polyline坐标、采样顺序和 interpolation version；depth/extent centers明确尚未给出。L791–798 却同时建议冻结“canonical code construction”。两者不能同时成立。

必须二选一：

- 把本轮范围收窄为 **wire/layout/cardinality candidate**，明确 axis/path/depth/extent code semantics尚未冻结；或
- 在 schema freeze中提交并 hash：512 distinct directed centers、zero-roll实现版本、四条无量纲polyline、插值规则、8 extent centers、32 depth centers、camera convention和边界量化规则。

metric centers可等硬件 calibration后再定，但在那之前只能冻结 code数量，不能声称相同 code已有全局唯一物理解释。若 axis table有重复或 path template解码别名，结构 code count仍是整数，但不能称 distinct control semantics。

PATH还必须冻结 orientation transport：frame是否在整条primitive上保持为anchor frame，还是 `x` 随局部切向更新、`z` 固定或并行运输；ARC/RASTER转角如何插值。仅说“polyline位于xy平面”不足以唯一决定 wrist orientation trajectory。

### 8.3 controlled-part frame 与 effector execution frame必须写成方程

r1b control review曾把 locator定义为 primary controlled part的目标 camera-frame pose/path。r1c L284 又说它最终量化为 assigned effector的 target execution frame/path。这是一个实质语义变更，不只是实现细节。

必须冻结：

- wire locator究竟编码 `T_camera_to_primary_goal` 还是 `T_camera_to_effector_goal`；
- `T_effector_to_controlled_part` 的方向、组合次序和时间戳；
- PATH是先在 controlled-part frame生成再整体刚体变换，还是直接在 effector frame解释；
- primary part与secondary relation如何在 decode后保持可验证。

没有这些定义，同一 32 bytes可被两个 controller作不同解释，profile hash也无法修复 schema-level歧义。

## 9. Text、proprio、allocator、profile 与 controller 旁路

### 9.1 已闭合部分

以下规则可以保留：

- assignment显式进入wire/token；
- allocator不能输出continuous correction/gain/path；
- ambiguous/infeasible必须abstain；
- 一个route/checkpoint/eval stratum只接受一个profile hash；
- profile不能有per-scene pose/ROI/map/path、sample gain、residual或trajectory lookup；
- 未选择的 registry entry变化不影响 action；
- human/pair/raw reference/planner hidden不进 action/cache。

### 9.2 `allowed structured text` 未定义

r1c L74、L489把 text列为合法直达 action 的条件，但没有重述 r1b cross-domain §10.4 的 denylist。只要 text能携带 numeric pixel/depth/axis/roll/path、instance-specific trajectory、assignment或held transform，96.671-bit packet就可被绕过。

必须冻结 text schema/template/hash和字段 allowlist。允许 task family、named goal、dose、tighten/loosen、success criterion；禁止 current-scene geometry、human coordinates、continuous path/pose和 allocator/controller指令。

### 9.3 `25D proprio/state` 含义不唯一

“25D proprio/state”可能表示严格25个 action-layout值，也可能表示一个可扩展 state object。当前25D不自动包含 camera/head/base pose、contact/load/slip或held transform。finite-frame memo L110–116要求 movable camera pose来自 fixed calibration或显式 public state；r1c没有选择。

必须列出 action-visible state 的 exact channels、单位、frame、sample time、freshness和hash，并明确：

- camera是否固定；若可动，pose放哪里；
- contact/load/slip仅由 safety plane使用，还是也进入 learned action；
- held state采用上一节三条路线中的哪一条；
- state维度变化如何改变 route ID、baseline matching与容量措辞。

### 9.4 receipt swap invariant需要固定 public input

r1c invariant 26 写“cost/logits/order/history/receipt swap不改变packet/action”。其中 cost/logits/order/private history应不可影响；但一张 receipt若证明的 current reachability、contact或held binding真的变化，正确结果可以是 assignment变化或 abstain。

准确测试应是：固定 allocator 的全部 public semantic/geometry/state输入，只替换 score、candidate order、非语义receipt metadata，hard output不变；改变 verified physical facts时必须按 validator规则重算，而不是强迫输出不变。

### 9.5 safety与性能归因必须分开

“post-WAM controller不得读 held transform”不能扩张为“安全控制面不得读 current contact/load/validity”。安全 hold/retract可以使用显式 allowlisted current state；如果它产生主要任务轨迹，则结果属于 direct controller baseline，不能归功于 r1c。需要同时报告 learned pre-safety action和executed post-safety action。

## 10. Missingness、transparent 与 D0

### 10.1 可保留

r1c 的以下收缩是自洽的：

- required field没有 action-visible UNKNOWN；reviewed unknown与missing原因不同，但均不编 packet；
- `EXHAUSTIVE no active`才授权全零NULL；
- transparent 2D overlap不推出3D `TOUCHING/INSIDE`；
- completed depth/mesh/NeRF/teacher fill不作oracle；
- support跨安全 bins不取nearest/mean；
- 任一required constraint未知时清旧packet/cache并 HOLD/REOBSERVE。

它比 standard-r1 更易 abstain，但这是 minimal route的合法设计选择，前提是单报 coverage/selection shift并与 standard/reconstruction/direct-servo互斥比较。

### 10.2 必须修正

- “transparent无direct depth/normal时packet消失”应按 locator kind写：depth对每个present locator required；normal/axis/roll只在 AXIS/FRAME/PATH需要。缺一个实际required field才拒绝，不能让 POINT因不需要的normal消失。
- human侧必须先过完整 constraint roster gate；只检查已出现constraint的字段不够。
- D0 review已明确：D0未通过、D1未执行、focus/edge/roster只是proposal。r1c新 semantic projection、O/S/M和双constraint completeness必须进入新的 X0/H1 checker/A-B gate，不能以旧 `1 V / 13 P / 4 U` prescreen支持冻结。

## 11. Exact wording 与 codec 仍需修订

以下措辞必须在下一版逐字改正：

1. `G_struct` 不是无条件“global” grammar；它已经嵌入 G1/RH56/Omni assignment与拟议RR拓扑。应称 **target-hardware-family structural grammar**，或明确定义 global仅指“该hardware family内未按task profile裁剪”。
2. L44和L832不能说所有具体 capability都只是 `NEEDS_CALIBRATION`；capability memo还含 `UNKNOWN_NOT_ADMITTED` 与 `UNSUPPORTED_BY_TOPOLOGY`，generic wiping/roll-sensitive cases也有表示层 No-Go。
3. 当前根本没有 executable profile instance，L344“不进入当前 executable profile”应改为“不进入任何当前可声明的 `G_profile`；profile instance尚不存在”。
4. `G_profile` 应叫 profile-valid，不叫 runtime reachable。
5. “唯一 final candidate”“不给控制必要信息做碰撞压缩”属于未经过 oracle/D0/H1/R0/B0 支持的结论，应改为“当前待验证主候选”。

如果文档继续声称 constructive rank/unrank已经冻结，还必须补齐：

- 13-byte compact index的 byte order；
- 11个 nonVERIFY phase/effect block 的逐项顺序；
- `(part, role, locator, secondary)` 中谁是最高/最低位的显式公式；
- overall NULL和各 block offset；
- `U_p(A)` 中 `x<y` 所用的 canonical payload rank；
- affine inverse的完整公式与边界/golden vectors；
- unrank失败、reserved bits与noncanonical wire的唯一错误顺序。

建议明确 compact rank 为 little-endian unsigned uint104；此时 byte 12 只能使用最低1 bit，即 `byte[12] & 0xFE == 0`。仅写“高7位为0”而不写byte order，两个实现仍可能产生不同容器。

semantic key也尚未达到 hash-exact：JSON中的 `<enum>` 没有规定使用数值码还是字符串字面值，也没有冻结必选键、禁止额外键和完整 enum spelling。RFC 8785只能规范同一 JSON value 的序列化，不能替规范选择 JSON value；这些必须进入 pair-key version。

另有一个 grammar语义需要明确：VERIFY强制 `primary_part=N_A`，却允许 `LS/RS` physical stabilization和所有 locator kind。若 VERIFY中的STABILIZE真要维持特定 controlled part，N_A可能不足；若它只是维持role-level已建约束，则必须给出唯一解释并限制不适用 PATH。若因此改 part/locator allowlist，需重算 `V/B_verify`。

## 12. 必须修改项

以下全部完成后才可重新申请 final freeze：

1. **重写 held-state contract**：区分 fixed constant、compile-and-quantize、public-state conditional三条互斥路线；把 same-action invariant限定为 fixed packet + complete public state后的 downstream模块。
2. **冻结 locator坐标语义**：controlled-part goal与effector execution goal给出明确变换方程、frame方向和PATH组合规则。
3. **解决 RR grammar**：提供 assignment-level全称资源证明，或加入 profile-gated right compound并完整重算。
4. **纯化三个集合**：所有 current instance/freshness/contact/reachability/WBC/collision进入 `R_runtime`；`C_p/U_p`只用有限静态谓词。
5. **修订 shared key**：解决 S↔M跨具身映射、禁止或证明duplicate multiplicity、定义唯一focus、加入 exhaustive constraint-roster admission。
6. **补强 MONITOR contract**：明确其可条件化其它physical slice，并增加 support-present/support-absent matched oracle；否则移出action并重算。
7. **冻结 action-input allowlist**：exact 25D/public state、camera pose、contact/load/safety边界和 text denylist逐字给出。
8. **收窄或物化 codebook freeze**：提交 axis/path/codebook工件与hash，或只称layout/cardinality candidate。
9. **补完 exact codec**：compact endian、block order、mixed-radix与profile ordering；澄清 VERIFY N_A/locator语义。
10. **补齐条件索引与hash序列化**：明确 `G_profile(p,t)` 或跨task并集语义，冻结 semantic JSON enum/key规则。
11. **改正 exact wording**：hardware-scoped structural count、profile-valid count、capability状态与“候选而非唯一final”。
12. **修订后先做 CPU D0**：wire/grammar/rank/unrank/toy enumeration/receipt机械通过，再请求独立 final review；在此之前仍不实现模型、不训练。

## 13. 可延后项

以下不会阻止“修订后的结构候选”再次送审，但在对应 gate前仍保持 No-Go：

- G1 biochem4 authoritative registry实例、profile hash、`|G_profile|` 与 profile bits；
- RH56/OmniPicker真实 driver、serial、端点、腕frame、working surface、resource claim、contact/load/slip、WBC和reachability标定；
- pipette LEFT compound的实机 admission；
- task-specific depth/extent metric centers与安全 tolerance，前提是文档不先冻结其物理语义；
- roll-sensitive insertion、transparent pour、generic wipe的 task sufficiency；
- four-path覆盖率、full/native trajectory gap和third-constraint overflow率；
- target-profile safe equal-bit permutation及其exact数值；
- D1 human A/B、R0 robot locator、P0 pairing、B0 bypass与无训练paired-world oracle的实际结果；
- 模型实现、训练、仿真、真机和论文 headline。

这些可延后项不能被 profile、text、proprio扩展、receipt或controller lookup偷偷补回 action。

## 14. 最终一句话

`minimal-r1c` 当前的**算术候选是成立的，最终结构冻结不成立**：现有 grammar 下 `126,149,773,623,130,877,098,697,687,041` 与 `96.67105236912121 bit` 复算正确，MONITOR应作为无自身slice但可影响其它slice的 action-visible finite constraint，FRAME/PATH结构也正确；然而 held-state same-action规则、assignment-level永久RR拒绝、human shared function/multiplicity/focus/roster、static/runtime profile边界以及 text/proprio/codebook exact contract尚未闭合。它们修订后可能改变 packet解释、pair key、admission，甚至总容量，所以本轮唯一裁决是 **No-Go**，不是 Conditional Freeze。
