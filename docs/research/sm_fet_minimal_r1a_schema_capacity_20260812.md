# SM-FET `minimal-r1a`：D0 后的精确 schema、codec 与容量裁决

- 日期：2026-08-12
- 性质：D0 后的独立 schema/容量审计；只设计，不实现、不训练、不使用 GPU、不修改 canonical
- canonical 代码基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 新候选 schema ID：`sm_fet_hard_grammar_v2_minimal_r1a`
- 上位对照：`sm_fet_hard_grammar_v2_standard_r1`

## 0. 裁决

冻结一个**可打包、可枚举、但尚未证明控制充分**的 `minimal-r1a` 候选。它只把目标机器人当前视图中、当前 active interaction 所需的有限功能状态送给 action expert：

```text
4-role fixed namespace
+ active references 的 target-domain current 4x4 center
+ 1 compact active phase
+ 1 known object effect
+ exactly 1 primary/focus interaction
+ optional 1 secondary role and 1 intended target relation
```

它的 canonical field wire 是 **10 bytes**，模型侧是 **5 个固定 token slots**，其中合法非 NULL packet 实际只有 2 或 3 个非 PAD tokens。精确 validator 可达状态为：

```text
S_minimal_r1a = 1,445,057
log2(S_minimal_r1a) = 20.46269497000547 bit
ceil(log2(S_minimal_r1a)) = 21 bit
compact enumerative container = 3 bytes
```

这个数不是从 standard-r1 删字段后相减出来的，而是按本 memo 的依赖语法重新计数并由 rank/unrank 构造复核。`281.932700038062 bit` 不是任何当前 canonical grammar 的 exact reachable capacity：它只是旧 dense independent-slot 上界；未修改 standard-r1 的正确值是 `207.74974691951957 bit`，本候选的正确值是 `20.46269497000547 bit`。

推荐地位是 **D0/D1 最小候选，非训练许可**。三段 D0 human pour 只支持 source/target、粗 2D center、单个可见 interaction、六个粗 phase 与粗 `TRANSFER`；它们没有验证目标机器人、双手、精细接触、物理 3D relation、flow、outcome、regrasp 或 verify。因此：

- minimal-r1a 是默认先证伪候选；
- standard-r1 是互斥的高表达 finite challenger；
- F-PCD、22D 与 continuous trajectory 是互斥 challenger/upper bound；
- 若只有 second edge、surface cell、contact mode、orientation 或连续 pose action-visible 才能执行，就判 minimal-r1a 不充分，不向它静默追加字段。

## 1. 证据边界

本裁决以以下 D0 与信息流契约为约束：

1. `gwp-agent-sm-fet-d0-observability/.../sm_fet_d0_three_episode_observability_audit_20260812.md`：3 个 proposal-only Ego episode 中，node center、primary/secondary node可见；scale、orientation、surface cell、outcome不可支持；contact/3D relation/flow只可 proposal 或 unknown。
2. `gwp-agent-sm-fet-d0-schema-reduction/.../sm_fet_d0_schema_reduction_review_20260812.md`：standard-r1 不能继续称 minimal；action core应降为 active-only、one interaction、known required fields。
3. `gwp-agent-sm-fet-focus-token/.../sm_fet_dynamic_focus_token_spec_20260812.md`：focus 必须由当前 interaction anchor机械派生；human locator不可复制给 robot；continuous ROI/track不进 action。
4. `gwp-agent-sm-fet-missing-hands/.../sm_fet_missing_hands_contract_20260812.md` 与 capacity reconciliation：遮挡、出画、tracker inactive和未复核不等于 absence；required actor/field unresolved必须 fail closed。
5. `gwp-agent-sm-fet-transparent/.../sm_fet_transparent_nonreconstruction_contract_20260812.md`：2D overlap不推出 touching/inside；flow/transfer/outcome必须有直接、时序、sensor 或 native truth；reconstruction/proposal不可升级为 GT。

其中 missing-hands 的“不要把未知写成不存在”保留；其 standard-r1 actor 四码不机械搬入 minimal-r1a。minimal-r1a 把 human literal actor 从共享 packet 删除，目标 effector 由 target adapter/allocator唯一解析。若无法唯一解析，仍然 `HOLD/REOBSERVE`，不让 WAM 暗选。

## 2. 固定 role 与 task profile

### 2.1 Role code

| wire code | role |
|---:|---|
| 0 | `PAD`，只允许在未引用 locator 或全局 NULL |
| 1 | `MANIPULATED_OR_SOURCE` |
| 2 | `TOOL` |
| 3 | `RECEPTACLE_OR_DESTINATION` |
| 4 | `FIXTURE_OR_SUPPORT` |

四个 role 是 schema namespace，不代表每个场景都有四个实体。每个 task profile 必须在标注/部署前冻结：

```text
task_profile_id
role -> exactly zero or one task entity binding
phase -> exactly one primary/focus anchor rule
allowed target relation predicate and direction
target-effector allocation rule
```

同 role 同时有多个不可唯一绑定实例时是 `ROLE_BINDING_OVERFLOW`。未引用 role 不发送 geometry；其 locator 为 PAD 不表示该物体物理不存在。

### 2.2 Target-domain center

每个被 active record 引用的 role 有一个当前目标域 locator：

| wire code | 含义 |
|---:|---|
| 0 | `PAD`；只允许该 role 未被引用 |
| 1..16 | image-plane 4×4 row-major cell |

对归一化当前目标图像坐标 `(x,y) in [0,1]^2`：

```text
row = min(3, floor(4*y))
col = min(3, floor(4*x))
center_code = 1 + 4*row + col
```

这里的 center 是 task profile 指定实体在**机器人当前因果观测**中的粗中心，不是 human frame 的 cell，也不是 bbox、ROI、depth 或 metric pose。若 entity binding、中心证据或量化支持不能唯一给出一个 cell，required locator unresolved，compiler拒绝 packet。

## 3. 精确 vocabulary

### 3.1 Compact phase

| code | phase | standard-r1 语义投影 |
|---:|---|---|
| 0 | `PAD` | 只用于 NULL |
| 1 | `APPROACH_ALIGN` | `APPROACH` 或 `PRECONTACT` |
| 2 | `ACQUIRE` | `ACQUIRE` |
| 3 | `CONTROLLED_CONTACT` | `HOLD` 或 `MANIPULATE` |
| 4 | `REGRASP` | `REGRASP` |
| 5 | `RELEASE` | `RELEASE` |
| 6 | `VERIFY` | `VERIFY_OUTCOME`，但 outcome 不进 action |

没有 action-visible phase `UNKNOWN`。reviewed unknown 和 not-reviewed 保留不同 annotation reason，但两者都不能形成 minimal-r1a packet。

### 3.2 Object effect

| code | effect |
|---:|---|
| 0 | `PAD`，只用于 NULL |
| 1 | `NONE` |
| 2 | `TRANSLATE` |
| 3 | `ROTATE` |
| 4 | `ARTICULATE` |
| 5 | `FLOW` |
| 6 | `DEFORM` |
| 7 | `TRANSFER` |

`FLOW/TRANSFER` 是 causal plan 的 intended current effect，或由当前直接/sensor/native证据确认的 effect；不能从 task name、wrist tilt 或脚本完成反推为观测 GT。没有 evidence 时不编码 UNKNOWN，而是拒绝 required-effect packet。

### 3.3 Primary part

| code | part |
|---:|---|
| 0 | `PAD`，只用于 NULL |
| 1 | `N_A` |
| 2 | `BODY` |
| 3 | `HANDLE_OR_GRASP_FEATURE` |
| 4 | `LID_CAP` |
| 5 | `OPENING_OR_RIM` |
| 6 | `TOOL_TIP` |
| 7 | `OTHER` |

`OTHER` 表示已知的其它功能部位，不是 unknown。为消除同一语义的多种编码，`N_A` **只允许 `VERIFY`**；其它 phase 必须使用六个 known parts。若非 VERIFY 的 part 不可判，则 packet inadmissible。

### 3.4 Secondary role 与 intended target relation

`secondary_role` 使用同一 role code；code 0 在该字段表示 `NONE`。非零 secondary 必须不同于 primary。

| relation code | target relation |
|---:|---|
| 0 | `N_A`；仅允许 `secondary_role=NONE` |
| 1 | `SEPARATE` |
| 2 | `NEAR` |
| 3 | `TOUCHING` |
| 4 | `INSIDE` |
| 5 | `CONTAINS` |
| 6 | `ALIGNED_TO_RIM` |
| 7 | `SUPPORTED_BY` |

关系方向固定为 `primary_role -> secondary_role`。`INSIDE` 与 `CONTAINS` 不做无 receipt 的自动换向。`OVERLAP_2D` 被排除，因为它是视角相关投影而非跨域执行目标。secondary 存在时 relation 必须是 1..7；若不需要 target relation，则 secondary 必须为 NONE。这样不存在“secondary 已给出但 relation=N/A”的重复编码。

## 4. Canonical wire 与 token layout

### 4.1 10-byte field wire

所有字段为 canonical `uint8`：

| byte offset | field | valid range |
|---:|---|---|
| 0 | `center[MANIPULATED_OR_SOURCE]` | 0..16 |
| 1 | `center[TOOL]` | 0..16 |
| 2 | `center[RECEPTACLE_OR_DESTINATION]` | 0..16 |
| 3 | `center[FIXTURE_OR_SUPPORT]` | 0..16 |
| 4 | `phase` | 0..6 |
| 5 | `effect` | 0..7 |
| 6 | `primary_role` | 0..4 |
| 7 | `primary_part` | 0..7 |
| 8 | `secondary_role` | 0..4 |
| 9 | `target_relation` | 0..7 |

全局 NULL 的唯一编码是 10 个零。任何其它 packet 必须满足第5节语法；不存在第二个 NULL 编码。schema/version、task profile、clock anchor、hash、allocator receipt和 invalid reason属于 envelope/control-plane metadata，不是这10字节，也不进入 action capacity。

### 4.2 五个固定 action token slots

```text
token 0..3: ROLE_CENTER(role_id, center_code)
token 4:    ACTIVE_INTERACTION(phase, effect, primary_role,
                               primary_part, secondary_role,
                               target_relation)
```

- primary center token必须非 PAD且可见；
- secondary 存在时对应 center token也必须非 PAD且可见；
- 其它 role center tokens为 PAD并被 attention mask；
- interaction token始终保留；全零时它是唯一 NULL sentinel；
- 合法单对象 packet有2个非 PAD tokens，双 role relation packet有3个非 PAD tokens；物理 slot 数仍固定为5，便于参数/FLOPs匹配。

Action expert 只读这5个 hard tokens、机器人 proprio/state、instruction text、diffusion timestep/noisy action。它不读 raw/current reference、planner hidden/logits、future visual、human video、pair/CAPOT、missingness、continuous pose或 auxiliary hidden。

## 5. 完整 grammar、排序与 overflow

### 5.1 NULL 与 non-NULL

```text
NULL:
  all 10 bytes == 0

NON_NULL:
  phase          in 1..6
  effect         in legal table below
  primary_role   in 1..4
  primary_part   in legal table below
  center[primary_role] in 1..16
```

phase/effect/part 的唯一合法组合为：

| phase | legal effect | legal part | 组合数 |
|---|---|---|---:|
| `APPROACH_ALIGN` | `NONE` | 6 known parts | 6 |
| `ACQUIRE` | `NONE` | 6 known parts | 6 |
| `CONTROLLED_CONTACT` | 7 known effects | 6 known parts | 42 |
| `REGRASP` | `NONE` | 6 known parts | 6 |
| `RELEASE` | `NONE` | 6 known parts | 6 |
| `VERIFY` | `NONE` | `N_A` only | 1 |

因此 `CONTROLLED_CONTACT + NONE` 是 HOLD-like control，`CONTROLLED_CONTACT + non-NONE` 是 MANIPULATE-like control。其它 phase 不允许携带对象变化，以避免 `APPROACH+TRANSFER`、`VERIFY+FLOW` 等重复或后见编码。VERIFY 的结果只在 supervisor/benchmark 中。

Secondary bundle 恰有两种形式：

```text
NO_SECONDARY:
  secondary_role = 0
  target_relation = 0
  only primary center is non-PAD

WITH_SECONDARY:
  secondary_role in the other 3 roles
  target_relation in 1..7
  center[secondary_role] in 1..16
  all other unreferenced centers = 0
```

任何 unreferenced center 非零、referenced center 为零、primary=secondary、非法 phase/effect/part、secondary/relation不一致或非 NULL positive-zero违反，均为 validator error。

### 5.2 Canonical order

1. node locator 永远按四个固定 role slot 排列；不按检测顺序、实体 ID 或屏幕位置排序；
2. primary 是 task profile 预先声明的当前 control anchor，同时就是 focus，不做数值 role 交换；
3. relation 永远按 primary→secondary方向；
4. 一个 action packet只有一个 record，因此没有 action-visible edge/event sort；
5. compiler 对 annotation 中的候选先标准化为 `(primary_role, primary_part, secondary_role, target_relation)`，再按该 tuple 字典序排序，仅用于稳定 duplicate/overflow receipt，**绝不取前一条静默截断**。

### 5.3 Overflow 与 fail-closed

| 情况 | stable result |
|---|---|
| 同一 role 多个不可唯一绑定实体 | `ROLE_BINDING_OVERFLOW` |
| task profile无法指定唯一 primary/focus anchor | `PRIMARY_OVERFLOW` |
| 当前动作同时需要两条不可合并 interaction | `INTERACTION_OVERFLOW` |
| 同一 primary-secondary 同时需要多个 target relations | `RELATION_OVERFLOW` |
| target effector allocator无唯一安全解 | `ACTOR_UNRESOLVED` |
| required field unknown/missing或 current center不可得 | 对应 `REQUIRED_*_UNRESOLVED` |

这些是 control-plane reason，不是 wire code。结果统一为不产生 action packet，并执行 `HOLD/REOBSERVE`；不得取排序第一项、补默认值、last-value hold或借用 NULL。

## 6. PAD、ABSENT、UNKNOWN、MISSING 与 censor 的唯一映射

| annotation/control truth | minimal-r1a wire | 说明 |
|---|---|---|
| 经 EXHAUSTIVE review 确认当前没有 active event | 全零 NULL | 这是唯一允许把“无事件”编成 NULL的情况 |
| role 未被 active record 引用 | 对应 center=PAD(0) | 不声称实体物理 absent |
| secondary 经 task profile确认不需要 | `secondary=0, relation=0` | NONE/N_A，不是 missing |
| required field `KNOWN` | 对应 known code | 正常编译 |
| required field `REVIEWED_UNKNOWN` | 无 packet | annotation/planner可监督 unknown；action fail closed |
| required field `NOT_REVIEWED/MISSING` | 无 packet | reason 与 reviewed unknown保持不同，不变成 NULL/UNKNOWN |
| node/hand `OUT_OF_FRAME/OCCLUDED/TRACKER_INACTIVE` | 本身不编码 | 有其它充分证据则编 known；否则 required unresolved |
| confirmed hand contact但 human side unknown | 不编码 human side | target allocator唯一解后才可编；否则 no packet |
| left/right boundary censor | 不编码 timing | planner clock保留 likelihood；active phase已知时可编，phase未知时拒绝 |
| model/VLM/MANO/reconstruction proposal | 永不作为 GT code | 只可 proposal/auxiliary，并带 provenance |

`ABSENT` 不再是 action field vocabulary：event absence映成唯一 NULL；未引用 role/secondary只用结构 PAD/NONE。annotation 中的 confirmed absence仍必须保留，以授权 NULL/NONE；missing不能伪装成 confirmed absence。

## 7. Episode clock、chunk clock 与 visibility boundary

### 7.1 Episode planner state

Episode planner 最多维护 `Q_plan_max=8` 个完整事件及 boundary/censor/unknown/missing监督。它可以保留 standard 的八类事件、relation before/after、second edge、mode、site、outcome和 future plan，但这些都不序列化为 minimal-r1a action wire。

minimal-r1a action phase的 episode-level有向图为：

```text
APPROACH_ALIGN -> ACQUIRE
ACQUIRE        -> CONTROLLED_CONTACT
CONTROLLED_CONTACT -> REGRASP | RELEASE
REGRASP        -> CONTROLLED_CONTACT | RELEASE
RELEASE        -> VERIFY
VERIFY         -> NULL
```

在连续 replan anchors间允许当前 phase自保持。`CONTROLLED_CONTACT` 内 effect可在 `NONE` 与 non-NONE间变化，表示 hold/manipulate control state变化，不新造 kind。新 episode正常从 `APPROACH_ALIGN` 开始；left-censored clip或有审计的 recovery可从任意已知 phase初始化。其它逆序、跳转或未知 phase不进入 action packet。

### 7.2 Chunk clock

```text
at anchor t:
  causal planner updates episode state
  target adapter rebinds roles and recomputes referenced current centers
  target allocator resolves a unique logical effector or abstains
  compiler emits exactly active packet or fail-closed reason
  WAM predicts 48 steps
  controller executes first 8 steps
  at t+8: reobserve, update phase/effect/centers, invalidate old packet cache
```

minimal-r1a 的 action-visible `Q_action_max=1`、future `N=0`。`Q_plan_max=8` 是 planner/control 数据结构而不是共享 action channel，因此本 memo不把 planner内部 rich state伪装成一个可计的 action alphabet，也不报告“8个独立 packet相加/相乘”的假容量。未来若把 planner queue hard-serialize给任何 action/cache consumer，必须另立 schema并重新计容。

### 7.3 Target effector 不得形成隐形容量

Target allocator必须在 WAM 前给出唯一、安全、有 receipt 的 effector。首选 target adapter把该 effector的 state/action维度确定性置换到逻辑 active-effector位置，WAM输出后再逆置换；minimal与其 equal-bit control使用完全相同的 adapter。若显式 effector ID/token、allocator logits或hidden进入 WAM，它就是新增 action-visible信息，必须另计容量，不能继续使用20.4627-bit主张。

## 8. 精确容量推导与独立复核

### 8.1 不能把 dense entries 相加

本 grammar 的主要依赖是：

- 只有 primary 和 optional secondary 的 center非 PAD；
- secondary role不能等于 primary；
- relation有无由 secondary决定；
- phase约束 effect/part；
- NULL只有一个；
- unknown、missing、censor、overflow和 allocator reason不进 wire。

所以 `10 bytes × 8 bit`、逐字段 log 相加、或从 standard-r1 减位数都不是 reachable capacity。

### 8.2 Constructive count

phase/effect/part 合法组合：

```text
G = 6 + 6 + 42 + 6 + 6 + 1 = 67
```

固定 primary 后，secondary bundle：

```text
B = 1                                      # no secondary
  + 3 remaining roles * 16 centers * 7 relations
  = 337
```

非 NULL active event：

```text
S_nonnull
= G * 4 primary roles * 16 primary centers * B
= 67 * 4 * 16 * 337
= 1,445,056
```

加入唯一全局 NULL：

```text
S_minimal_r1a
= 1 + S_nonnull
= 1,445,057

log2(S_minimal_r1a)
= 20.46269497000547 bit

ceil(log2(S_minimal_r1a))
= 21 bit
```

因此：

| 口径 | exact states | bits / width |
|---|---:|---:|
| 一个 non-NULL active event | 1,445,056 | `log2=20.462693972...`（仅比较规模，不含 NULL） |
| action packet，含 NULL | **1,445,057** | **20.46269497000547 bit** |
| minimal enumerative index | 1,445,057 valid ranks | **21-bit minimum；3-byte container** |
| canonical inspectable field wire | 同一 alphabet | 10 bytes；不是80-bit语义容量 |
| physical action tokens | 固定5 slots | 2/3 non-PAD；NULL为全零 interaction sentinel |

上表第一行的非 NULL log 只作辅助；论文 headline只用含 NULL 的 exact值。

### 8.3 第二种复核：rank 是双射

把67个合法 `(phase,effect,part)` 按 `phase -> effect -> part` 升序编号为 `g in [0,66]`。primary code转零基 `p in [0,3]`，primary center转 `c_p in [0,15]`。

secondary bundle编号：

```text
b = 0                                      if NONE
b = 1 + ((j*16 + c_s)*7 + r)              otherwise
```

其中 `j in [0,2]` 是“从升序 roles 去掉 primary 后”的 secondary局部编号，`c_s in [0,15]`，`r in [0,6]`。全局 rank：

```text
rank(NULL) = 0

rank(nonnull)
= 1 + (((g*4 + p)*16 + c_p)*337 + b)
```

它机械覆盖且只覆盖 `1..1,445,056`；反向依次对337、16、4做 divmod 即可唯一恢复字段。这一构造独立证明计数没有 collision、hole或重复表示。3-byte compact codec把 rank 按 unsigned little-endian存储；`rank >= 1,445,057`、高3位非零或 decode后 grammar-invalid都拒绝。

### 8.4 纠正 standard 与旧 281.9327 口径

| schema | exact reachable states | exact bits | minimum index |
|---|---:|---:|---:|
| standard-r1 | `345864043392188289642033658185203030253953267411599196753571073` | `207.74974691951957` | 208 bits |
| minimal-r1a | `1,445,057` | `20.46269497000547` | 21 bits |

两者相差 `187.2870519495141 bit`，但这只是两个完整 grammar 的 log-capacity差，不是可以分摊给被删字段的独立 bit贡献。旧 `281.932700038062 bit` 把不可同时出现的 dense slot当独立，违反 reference union、prefix、event path、edge bundle和 field dependency；它只能标成 syntactic upper bound，不能称 exact codec capacity。

## 9. Equal-bit baseline

minimal-r1a 必须有自己的 whole-packet equal-bit control，不能借用 standard-r1 的207.7497-bit control。

令 `M=1,445,056`，对第8.3节 rank 定义预注册 permutation：

```text
pi(0) = 0

for u in 1..M:
  v = u - 1
  pi(u) = 1 + ((65537*v + 104729) mod M)
```

`gcd(65537,M)=1`，乘法逆元为 `225985`，因此它在所有非 NULL states上可逆且 NULL固定。baseline流程是：合法 semantic packet → rank → `pi` → unrank → 同一10-byte/5-token encoder。它必须与 semantic route匹配：

- 同一 backbone、数据、split、optimizer和训练步数；
- 同一5 token slots、embedding/head参数量、attention mask和FLOPs；
- 同一 target allocator、proprio/text/noisy action与cache规则；
- permutation在实验前冻结，不按结果挑 seed；
- 同报 rank利用率、field/token利用率与 invalid/abstain rate。

standard-r1 challenger仍使用其自身全 packet、`207.74974691951957-bit` equal control。minimal semantic 对 standard equal-bit、或 standard semantic 对 minimal equal-bit，都不是容量匹配比较。

## 10. 与 standard-r1 的兼容关系

### 10.1 不是 bitwise/checkpoint compatible

minimal-r1a 与 standard-r1：

- schema ID、wire、token数、head/embedding、validator、rank/unrank和 capacity receipt均不同；
- 不允许拿 standard的82-byte packet截字段后冒充minimal；
- 不允许在同一 checkpoint中静默切换；
- v1.1 22D 与 standard-r1 checkpoint/key/disabled path继续冻结。

未来实现必须是显式 opt-in route和独立 checkpoint bundle。只可做离线 semantic migration，不能做二进制兼容。

### 10.2 Standard → minimal 的部分投影

只有 standard packet满足以下条件时才可确定性投影：

1. 只取 active；future/timing丢到 planner；
2. task profile能从 standard focus/edges唯一选出一个 control anchor；否则 overflow；
3. primary role与 current target-domain center已重新定位，不能复制 human center；
4. part可映入六类，VERIFY映 N/A；required unknown拒绝；
5. relation_after中恰有一个 intended target relation，映入七类；relation_before、OVERLAP_2D和unknown不投影；
6. phase映射为：APPROACH/PRECONTACT→APPROACH_ALIGN，HOLD→CONTROLLED+NONE，MANIPULATE→CONTROLLED+non-NONE；其它一一对应；非法 effect组合拒绝；
7. actor、surface cell、contact mode、outcome、第二edge不进入minimal；actor必须由target allocator重新解析。

这个投影是 many-to-one且只对 standard子集定义；minimal状态的 standard lift也不唯一。因此两者不能共用 codec或把 minimal gain解释成 standard字段的逐位消融。

## 11. Planner/auxiliary 与 action 的边界

### 11.1 降为 auxiliary/control 的字段

| 原 standard 信息 | minimal-r1a 去向 |
|---|---|
| node observability/absence reason | compiler admission、reobserve、visibility auxiliary |
| node scale/orientation | target-local auxiliary或独立 challenger |
| full 8-kind event | planner supervision；action只看6 compact phases |
| onset/duration/interval/censor | Semi-Markov planner、episode clock、boundary likelihood |
| future next1/next2/Q8 tube | planner-only |
| independent focus | 机械等于 primary role |
| LEFT/RIGHT/OBJECT_OBJECT actor | annotation truth与target allocator/control receipt |
| surface cell | target-local grounding auxiliary/challenger |
| relation before/完整 after集合 | annotation/planner；action只看一个 intended target relation |
| contact mode/force/tactile | target sensor/planner/controller |
| second edge、hand roster、bimanual topology | planner/standard challenger；minimal overflow gate |
| verify outcome | supervisor、termination/recovery与benchmark |
| unknown/missing/coverage/confidence | loss mask、provenance、risk calibration |
| exact bbox/ROI/rim/stream/depth/mesh | auxiliary或独立 spatial/reconstruction baseline |

### 11.2 连续 pose/contact change

Human 与 robot 各自的 entry/optional-turn/exit 2D/3D pose、连续 hand trajectory、contact transition和local window只允许：

- 训练视觉 encoder/planner的 masked auxiliary；
- 检验 single snapshot / event-multipose / local window / dense trajectory 的 compression probe；
- target robot current proprio/contact作为既有低层状态输入。

它们的 raw值、decoder hidden、heatmap、logits、teacher proposal、pair ID与cache不得进入 minimal action route。若 test-time必须把连续 pose/contact geometry送给action才有效，应将结果命名为独立 continuous/spatial route，并停止20.4627-bit finite sufficiency claim。

### 11.3 F-PCD 是独立 challenger

F-PCD 不与 minimal-r1a concat，也不借 planner hidden回流。它使用独立 route ID/checkpoint，报告自己的 action-visible连续带宽、参数/FLOPs和cache。应比较：

```text
22D-only
minimal-r1a
minimal-r1a equal-bit
standard-r1 + standard equal-bit
F-PCD-only
full/local continuous trajectory
```

若 F-PCD 或 standard只在 bimanual、精细接触、regrasp等预注册 strata显著获益，可据此 promotion相应字段；不能在看结果后把其hidden拼回minimal。

## 12. 当前可执行性判断与主要风险

本 memo**不宣称 minimal-r1a 已足以执行**。D0只给出“可以稳定写出较小候选”的证据，没有目标机器人闭环证据。

主要风险按严重性排序：

1. **single-record bottleneck**：双手稳定、一个effector持物且另一个对象关系同时重要时会 overflow；D0没有第二手真值，无法支持删除第二edge。
2. **coarse locator insufficiency**：4×4 center可能不足以对rim、handle、小部件做first-contact；surface cell/orientation可能在target robot上有独立价值。
3. **actor/control laundering**：allocator若不唯一或把hidden/logits送给WAM，20.4627-bit主张失效。
4. **effect evidence不足**：透明pour的FLOW、TRANSFER无法由腕部倾斜推断；缺sensor/native truth时compile coverage可能很低。
5. **phase coverage不足**：3个episode没有REGRASP/VERIFY正支持；六phase只是可实现ontology，不是D0全类通过。
6. **N/A/OTHER吸收错误**：N/A不能吸收unknown，OTHER不能吸收未复核part；否则表面compile coverage会虚高。
7. **current-only短视**：execute-8下future可能不必要，但尚无robot证据；若active+next确有收益，应以独立版本promotion，不直接复活standard全部字段。

因此当前状态：

- **Go**：冻结本 memo、做CPU codec/validator设计审计、用3+12复核字段来源和overflow；
- **Conditional Go**：有target robot native current center、effector allocator、contact/effect/outcome truth后做oracle比较；
- **No-Go**：现在训练WAM、把D0 proposal当GT、声称跨域控制充分或删除standard/F-PCD challenger。

## 13. CPU-only 可验证不变量（本轮不实现）

未来实现前至少冻结以下测试；不得为了让样本通过而修改期望：

1. 全零是唯一NULL；任一非零payload配phase=0都拒绝；
2. non-NULL 的 primary center必为1..16，其它非引用 center必须为0；
3. secondary=0当且仅当relation=0且无secondary center；
4. secondary非零时必须不同primary，且其center非零、relation为1..7；
5. phase/effect/part严格通过67组合表；VERIFY只允许NONE+N_A；其它phase禁止N_A；
6. `CONTROLLED_CONTACT+NONE/non-NONE`分别唯一映HOLD/MANIPULATE-like语义；
7. role center row-major quantize/dequantize边界固定，human cell不能出现在robot compiler receipt；
8. unknown、missing、occluded、out-of-frame、tracker-inactive不变成NULL/PAD/NONE；
9. 只有EXHAUSTIVE known no-active授权NULL；
10. censor只改变planner clock/loss，不能改成遮挡或进入10-byte wire；
11. role/primary/interaction/relation overflow稳定拒绝且不截断排序首项；
12. target allocator多解稳定`ACTOR_UNRESOLVED`；唯一解只做确定性adapter/receipt；
13. field wire→rank→wire与rank→wire→rank覆盖恰好1,445,057 states；
14. ranks必须恰为0..1,445,056，无collision/hole；大于等于1,445,057的3-byte值拒绝；
15. exact integer、20.46269497000547-bit、21-bit minimum与10-byte/5-token receipt逐字固定；
16. equal-bit permutation全域双射、NULL固定，逆元225985复原原rank；
17. minimal equal-bit与semantic route token mask、参数、FLOPs、allocator和cache完全一致；
18. fixed packet/state/text/noise时替换raw reference、future events、aux target/logits/hidden、missingness metadata，pre-safety action保持`max_abs<=1e-6`；
19. action/cache attention allowlist不存在image/reference/future/planner-hidden/pose/F-PCD/CAPOT；
20. standard→minimal投影对不兼容packet fail closed，不填默认part/relation/effect；
21. v1.1、standard-r1与minimal-r1a schema/checkpoint IDs互不接受；disabled路径bitwise不变；
22. task profile缩小合法子集时只降低task-conditioned utilization，不改变全局1,445,057 headline；
23. bimanual/多关系样本若minimal overflow，standard/F-PCD challenger仍保留该样本，不把它删出公平test；
24. continuous pose/contact auxiliary只改变loss；同一forward修改其label bytes不能改变action输出。

## 14. 最终建议

把 `sm_fet_hard_grammar_v2_minimal_r1a` 冻结为D0后的唯一**最小 action 候选**：10-byte canonical field wire、5个固定token slots、active-only、一个interaction、exact `1,445,057` states / `20.46269497000547 bit`。它诚实地把 unknown/missing/censor、future、actor、timing、outcome和高带宽geometry留在planner/control/auxiliary，避免用epistemic token扩大动作字母表。

但不要把“更小”写成“已足够”。D0没有目标机器人与双手证据；single record、4×4 center和去掉contact mode/site可能直接导致控制失败。下一步只能是CPU contract审计与3+12可标性复核；之后用native target-robot oracle在minimal、standard、F-PCD、22D和continuous互斥route上决定promotion。若minimal不能达到预注册success/safety门槛，接受它被证伪，而不是向其静默加回字段后仍沿用20.4627-bit口径。
