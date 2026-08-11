# SM-FET-WAM v2：精确 Schema 与可实现架构契约

- 日期：2026-08-12
- 审计基线：canonical `e212afec9db5198a89930c7ff896e3e49a0c8249`
- 输入：`temporal_event_pairing_wam_meta_synthesis_20260812.md` 与 Idea F/G/H/I/J memo
- 范围：只定义机器契约、容量与未来插入点；未训练、未调用 GPU、未改 canonical 代码或数据
- 推荐：**standard `event_tube_v2`**，不是原 825-float 草案的机械实现

## 0. 决定与最重要的纠正

推荐把 SM-FET v2 固定为一个独立、互斥于 v1 的 packed categorical contract：

```text
episode planner: Q_episode_max = 8
action compiler: active + next_1 + next_2
node roles:      4 个固定 role slots
edges:           每个 action-visible event 最多 2 条
wire payload:    82 个 uint8 categorical codes，而不是 825 个 float entries
action tokens:   4 node + 3 event + 6 edge = 13 tokens
```

原 synthesis 的 `281.932700038062 bit` 需要改名而不是简单改小数：

- 算术正确。`36865^4 * 156673^3 * 318512641^6` 的确等于原文整数，`log2` 为 `281.9327000380615`。
- 语义结论不正确。这个乘积允许 future slot 在 active 缺失时存在、edge 指向 absent node、secondary 与 primary 为同一 node、重复/乱序 edge、unused present node、非法 phase transition 和非单调 onset。它是 **independent-slot syntactic upper bound**，不是原文声称的 “codec-exact reachable upper bound”。
- 原草案还把 `confidence_band` 作为 action-visible 四值字段；即使说它由熵编译，它仍是一个可由 planner 操纵的额外视觉通道。v2 主 route 删除该字段。

本 memo 的 standard grammar 将上述依赖写入 validator，并用包含–排除与有限状态路径计数得到：

```text
exact reachable action packet states
= 345864043392188289642033658185203030253953267411599196753571073

exact action-visible capacity = 207.749746919520 bit
fixed binary container width = 208 bit
```

这里的 `exact` 只针对下文逐条冻结的 standard hard grammar；它不是互信息、经验熵或模型实际利用率。

## 1. 版本、wire 类型与互斥规则

### 1.1 版本常量

```text
EVENT_TUBE_SCHEMA_VERSION = "event_tube_v2_standard"
EVENT_TUBE_GRAMMAR_VERSION = "sm_fet_hard_grammar_v2_standard_r1"
EVENT_TUBE_CODEC_VERSION = "sm_fet_enumerative_codec_v1"
EVENT_TUBE_EQUAL_BIT_VERSION = "sm_fet_global_affine_permutation_v1"

FPS = 20
H_ACTION = 48
R_ACTION = 8
LOOKAHEAD_FRAMES = 96
Q_EPISODE_MAX = 8
K_ACTION_VISIBLE = 3
N_NODE_ROLES = 4
R_EDGE_MAX = 2
```

`contact_segment_v1_1`、22D、Q12、B16、`48673` states/query 和 `186.850008943084 bit` 全部冻结。v2 不改旧常量、不复用旧 schema version，也不得与 v1 packet concat。每个 checkpoint 的 route 必须且只能是：

```text
none | contact_segment_v1_1 | event_tube_v2_standard |
event_tube_v2_equal_bit | f_pcd_v1_challenger
```

### 1.2 wire layout

所有值先编码为 canonical `uint8`，再由 factor embedding 编成 action token；sidecar 中不保存 one-hot 展开。standard payload 是：

```text
node_code  [4, 4]  # observability/presence, center, scale, orientation
event_code [3, 6]  # kind/presence, focus, onset, duration, effect, outcome
edge_code  [3, 2, 8]
                       # actor/presence, primary node/part/cell,
                       # secondary node, relation before/after, contact mode
```

即 `16 + 18 + 48 = 82` 个 byte codes。schema ID、grammar receipt、overflow/error code、packet length、CRC/SHA 属于 control-plane metadata，**不得成为 action tensor**。

## 2. Standard schema：每一个 action-visible 字段

### 2.1 Node：四个固定 role slots

slot 顺序固定，不传自由 object ID：

```text
0 manipulated_or_source
1 tool
2 receptacle_or_destination
3 fixture_or_support
```

每个 slot 最多绑定一个 task-relevant instance。track ID、mask、box、实例名与 digest 只进 sidecar/receipt，不进 action。

| wire 字段 | code 集合 | canonical 约束 |
|---|---|---|
| `observability` | `0 ABSENT, 1 VISIBLE, 2 PARTIAL, 3 OCCLUDED_TRACKABLE, 4 OUT_OF_VIEW, 5 UNKNOWN` | `ABSENT` 是 node 不存在；`UNKNOWN` 是 node 被任务引用但观测状态未知，二者绝不合并 |
| `center_cell` | `0 PAD, 1..16 = 4×4 cells, 17 UNKNOWN` | `VISIBLE` 必须为 1..16；`PARTIAL/OCCLUDED` 可为 1..17；`OUT_OF_VIEW/UNKNOWN` 必须为 17 |
| `scale_band` | `0 PAD, 1 SMALL, 2 MEDIUM, 3 LARGE, 4 UNKNOWN` | 阈值只用 train split 冻结；`OUT_OF_VIEW/UNKNOWN` 必须为 4 |
| `orientation_2d` | `0 PAD, 1..8 = 8×45° bins, 9 UNKNOWN` | `OUT_OF_VIEW/UNKNOWN` 必须为 9；角度边界采用左闭右开，最后一 bin 包含 wrap |

`ABSENT` 时后三字段必须为 literal positive zero。一个 present node 的合法几何状态数为：

```text
VISIBLE:             16 * 4 * 9 = 576
PARTIAL/OCCLUDED: 2 * 17 * 4 * 9 = 1224
OUT_OF_VIEW/UNKNOWN:                  2
G_standard = 1802 present states/role
```

standard 不把 monocular `depth_band` 送 action；它只可作为 auxiliary。full 候选才增加该字段。

### 2.2 Event：active + next two

| wire 字段 | code 集合 | canonical 约束 |
|---|---|---|
| `event_kind` | `0 ABSENT, 1 APPROACH, 2 PRECONTACT, 3 ACQUIRE, 4 HOLD, 5 MANIPULATE, 6 REGRASP, 7 RELEASE, 8 VERIFY_OUTCOME, 9 UNKNOWN` | event presence 由 `kind!=0` 表示；present slots 必须是 prefix |
| `focus_node` | `0 PAD, 1..4` | present event 必须引用一个 present node；无法解析 focus 的 proposal 不进入 action packet |
| `onset_code` | `0 PAD, 1 ACTIVE, 2..17 FUTURE_BIN_0..15, 18 UNKNOWN` | slot 0 只能 `ACTIVE`；future bin `b` 表示当前帧后 `[1+6b, 6+6b]` 帧；future onset 非降序 |
| `duration_code` | `0 PAD, 1..8 finite bins, 9 GT96, 10 RIGHT_CENSORED, 11 UNKNOWN` | active 表示 remaining duration，future 表示 total duration；`GT96` 与删失必须分开 |
| `object_effect` | `0 PAD, 1 NONE, 2 TRANSLATE, 3 ROTATE, 4 ARTICULATE, 5 FLOW, 6 DEFORM, 7 TRANSFER, 8 UNKNOWN` | unknown 是 action-visible 的诚实不确定值，不是 loss mask |
| `outcome` | `0 NOT_APPLICABLE_OR_PAD, 1 SUCCESS, 2 PARTIAL, 3 FAILURE, 4 UNKNOWN` | 只有 `VERIFY_OUTCOME` 可为 1..4；其它 kind 必须为 0 |

duration 的八个 finite bins（processed 20 Hz frames）固定为：

```text
[1,2], [3,4], [5,8], [9,16],
[17,24], [25,32], [33,48], [49,96]
```

`UNKNOWN kind` 仍是一个 present、带 focus/duration/effect 的事件，但只能是当前 tube 的最后一个事件；它与 `ABSENT/NULL` 不同。action packet 的唯一全局 NULL 是三个 event 全 absent、四个 node 全 absent、六个 edge 全 absent。

### 2.3 Edge：每事件最多两条

| wire 字段 | code 集合 | canonical 约束 |
|---|---|---|
| `actor` | `0 ABSENT, 1 LEFT, 2 RIGHT, 3 OBJECT_OBJECT` | edge presence 由 `actor!=0` 表示；每个 event 内 present edge 为 prefix |
| `primary_node` | `0 PAD, 1..4` | 必须引用 present node |
| `primary_part` | `0 PAD, 1 BODY, 2 HANDLE, 3 LID_CAP, 4 OPENING, 5 RIM_EDGE, 6 TRIGGER_BUTTON, 7 TOOL_TIP, 8 OTHER, 9 UNKNOWN` | `OTHER` 是已知但不在词表；`UNKNOWN` 是无法判断 |
| `primary_surface_cell` | `0 PAD, 1..16 = part-local 4×4, 17 UNKNOWN` | 不传连续 UV、mask、ROI feature |
| `secondary_node` | `0 NONE, 1..4` | 若非零，必须 present 且不能等于 primary；无法识别 secondary 时不伪造 edge |
| `relation_before` | `0 N/A, 1 SEPARATE, 2 NEAR, 3 TOUCHING, 4 OVERLAP_2D, 5 INSIDE, 6 CONTAINS, 7 ALIGNED_TO_RIM, 8 SUPPORTED_BY, 9 UNKNOWN` | secondary 为 0 时必须为 0；否则必须 1..9 |
| `relation_after` | 同上 | 与 `before` 独立监督；delta 由二者机械生成，不直接让标注者写自由 delta |
| `contact_mode` | `0 PAD, 1 NONE, 2 STICK, 3 SLIDE, 4 ROLL, 5 IMPACT, 6 UNKNOWN` | `NONE` 是确认无物理接触但有 relation；`UNKNOWN` 是接触模式不可判 |

`OBJECT_OBJECT` 必须有 secondary；`LEFT/RIGHT` 可以无 secondary，表示 effector 对 primary 的接触。两条 edge 按完整整数 tuple 排序且禁止重复，所以 edge bundle 数不是 `E^2`。

standard 有意删除原 825 草案的 secondary part/cell。主接触点已经由 primary part/cell定位；secondary node 与 pre/post relation 足以表达 tool/source 对 target/receptacle 的效果。若 secondary 精细接触点在 oracle gate 中确有独立价值，只进入 full 候选。

### 2.4 annotation-only 状态：unknown、censor 与 missing

episode sidecar 的每个 event 另有 action-invisible 字段：

```text
event_id: stable uint32 within episode
onset_lower_frame, onset_upper_frame
offset_lower_frame, offset_upper_frame
onset_status:  OBSERVED_EXACT | OBSERVED_INTERVAL | LEFT_CENSORED | UNKNOWN
offset_status: OBSERVED_EXACT | OBSERVED_INTERVAL | RIGHT_CENSORED | UNKNOWN
coverage:      EXHAUSTIVE | POSITIVE_ONLY | NOT_REVIEWED
evidence_tier: SENSOR_CONFIRMED | HUMAN_ADJUDICATED | GEOMETRY_PROXY | MODEL_PSEUDO
field_known_mask, confidence_for_loss_only, provenance_sha256
```

三类情况严格分开：

1. `ABSENT`：只有相应 coverage 为 `EXHAUSTIVE` 时才是 supervised negative；所有 payload literal `+0`。
2. `UNKNOWN`：reviewer/planner确认实体或字段存在，但值不可判；使用上表显式 UNKNOWN 类。
3. `MISSING/NOT_REVIEWED`：没有证据；`field_known_mask=0`，不产生 CE/BCE，也不自动编成 UNKNOWN。

left/right censor 进入 survival/interval likelihood。action 只看 `ACTIVE` 与 remaining duration，或明确的 `RIGHT_CENSORED` duration；它不看 annotator、confidence、coverage、evidence tier 或 boundary bounds。

## 3. Grammar、排序与 overflow

### 3.1 phase transition grammar

相邻相同 kind 必须先合并成一个 maximal-duration event，因此 event 序列没有 self-loop。合法 successor 固定为：

```text
APPROACH       -> PRECONTACT | ACQUIRE
PRECONTACT     -> ACQUIRE
ACQUIRE        -> HOLD | MANIPULATE | REGRASP | RELEASE
HOLD           -> MANIPULATE | REGRASP | RELEASE
MANIPULATE     -> REGRASP | RELEASE | VERIFY_OUTCOME
REGRASP        -> HOLD | MANIPULATE | RELEASE
RELEASE        -> VERIFY_OUTCOME | APPROACH
VERIFY_OUTCOME -> APPROACH
any known kind -> UNKNOWN   # UNKNOWN 必须终止 tube
```

`RELEASE→APPROACH` 与 `VERIFY→APPROACH` 支持同 episode 多个 subgoal；同步双手接触由同 event 的两条 edge 表达，不以同 onset 的两个 event 表达。

### 3.2 canonical order

- node：固定 role slot；同 role 多实例的 binding key 为 `(first_task_relevant_frame, track_sha256)`。
- event：slot 0 是 active；future 按 `(onset_code, kind, focus_node, edge_bundle_digest)` 排序。量化到同 onset 且能合并者合并为一个双 edge event；不能合并则 packet invalid，不用 query index暗藏顺序。
- edge：按 `(actor, primary_node, primary_part, primary_cell, secondary_node, relation_before, relation_after, contact_mode)` 严格升序；重复 edge 拒绝。
- present node 集合必须**恰好**等于所有 event focus 与 edge references 的 union；禁止没有被任何事件引用的“免费视觉 node”。

### 3.3 overflow 是 control error，不是新 action bit

validator 返回：

```text
OK | NULL | EVENT_OVERFLOW | NODE_OVERFLOW | EDGE_OVERFLOW |
INVALID_REFERENCE | INVALID_ORDER | INVALID_GRAMMAR | NONCANONICAL_ZERO
```

- reviewed training sample 只接受 `OK/NULL`；其它状态进入独立 overflow 统计并从主 loss/action condition 移除。
- inference 不静默截断。任何 overflow/invalid 使 compiler 不产生 packet，controller 执行预注册的 safe hold/reobserve，并记录 receipt。
- `Q_episode>8`、同一时刻 task-relevant role instance >1、edge/event >2 的 empirical rate 各自报告；任一超过 5% 时先修 schema/任务分解，不启动主训练。
- overflow code 只进日志，不送 action，避免它成为隐藏高带宽类别。

## 4. Episode clock 与 action-chunk clock

### 4.1 Episode event clock

运行时状态是一个明确的纯数据结构：

```text
EpisodeClockState:
  episode_uid_sha256
  fps_num=20, fps_den=1
  observation_frame:uint32
  mode: IDLE | ACTIVE | TERMINAL
  active_event_id:uint32 | NONE
  active_kind:uint8
  active_focus_node:uint8
  active_onset_frame:uint32 | LEFT_CENSORED
  elapsed_frames:uint32
  last_committed_offset:uint32
  last_committed_kind:uint8 | NONE
  next_event_counter:uint32
  node_binding_receipt_sha256
  planner_packet_receipt_sha256
```

初始化与每次 replan 更新必须是确定性的：

1. episode UID 变化时 reset；首个非 NULL active event 获得 `event_id=0`。
2. 同 episode 必须满足新 `observation_frame > old`；否则拒绝 stale/out-of-order observation。
3. planner 只用 `x_{≤t}`、state、text 和上一个 hard clock state产生 Q≤8 tube。
4. 新 slot 0 的 `(kind, focus_node)` 与已 active 相同则保持 event ID，`elapsed += Δframe`；edge/relation 的观测更新不新建 event。
5. 若不同且 successor 合法，则在当前 observation frame关闭旧 event、counter 加一并打开新 event；从 IDLE 打开时用 `last_committed_kind` 做同一 successor 检查。若 successor 非法则整个 plan fail closed，clock 保持旧 committed state。
6. 无 active、但 Q8 tube 中仍有 future event 时进入 `IDLE`；本次 action packet 为全局 NULL，后续 event 可按 onset 晋升。只有显式 episode end/terminal判据才进入 `TERMINAL`，terminal 后仅允许新 episode reset。
7. future slots 是 provisional，没有稳定 ID；只有晋升为 slot 0 时才分配 ID。event ID、clock mode 与绝对 frame/time 都不进 action。

训练 sidecar 的稳定 `event_id` 来自 reviewed episode，不按 48-frame window 重编号。预测 clock 的边界分辨率受 `R_ACTION=8` 限制，必须与 annotation boundary error 分开报告。

### 4.2 Planner Q8 clock 与 compiler

planner 在 `ACTIVE` mode 的 slot 0 为 active；`IDLE` mode 可以只有 future events。future onset 内部 alphabet 是：

```text
16 bins within next 96 frames | FAR_GT96 | UNKNOWN
```

future onset 非降序。`FAR_GT96` 与 `UNKNOWN` 保留在 Q8 planner/clock，不进入本次 action packet。compiler 步骤：

1. 取 active；按顺序扫描 future，只保留 onset 被判断在 `(t,t+96]` 内的前两个。没有 active 时，严格遵守“active+next”接口并输出全局 NULL，不把 future-only计划伪装成 active。
2. active 编码 `onset=ACTIVE`、`duration=remaining`；future 编码 `onset=1..96` 的 16 bins、`duration=total`。
3. 收集三个 event 的 focus/edge node union，按固定 role slots生成当前 observation 的 node snapshot。
4. 运行 grammar/reference/order/overflow validator；仅 `OK/NULL` 可序列化。
5. action 预测仍为 48 steps，但只执行前 8 steps；到 `t+8` 重新观测、更新 clock、重编译 packet并失效旧 cache。

### 4.3 action 到底能看到什么

主 strict route 的 action 可读：

```text
proprio/state
allowed instruction text
diffusion timestep/noisy action
13 个由 hard packet 产生的 node/event/edge tokens
```

唯一 current-image-derived 内容是：当前四个 role node 的 2D cell/scale/orientation/observability，以及 planner 编译出的有限 event/edge/effect字段。action 不可读：raw RGB/reference token、历史窗口 token、future visual、continuous box/depth/pose/flow、ROI/track embedding、planner logits/hidden/confidence、posterior、CAPOT coupling/pair ID 或 human frame。

Idea I 的 CET-WM 若未来启用，只向 planner 提供 causal history evidence；selected raw windows 仍不可直接进 strict action。text 继续合法直达 action，所以容量仅称 **conditional current-visual route capacity**，不称总任务信息瓶颈。

## 5. 精确有限状态数与容量

### 5.1 为什么不是把 82 byte 或 one-hot 宽度相加

容量按 validator 接受的 packet 数 `S` 计算 `log2(S)`。`uint8` 存储宽度、one-hot dense width、embedding dim 与理论 alphabet 是三件不同的事。下面计数纳入：presence prefix、node reference、node union、edge unique sort、phase grammar、UNKNOWN terminal、outcome/kind依赖和 onset 单调性。

### 5.2 standard 的 edge/event 计数

设 packet 中恰有 `n` 个 present role nodes。一个 present edge 的合法数为：

```text
E_n = n * (9 primary parts) * (17 primary cells) * (6 modes)
      * [2 hand actors * (1 + (n-1)*9*9)
         + 1 object actor * ((n-1)*9*9)]
    = 918*n*(2 + 243*(n-1))
```

两条 edge canonical sort、禁止重复，所以每 event 的 edge bundle 数为：

```text
B_n = C(E_n,0) + C(E_n,1) + C(E_n,2)
```

event 的 base payload含 `focus n × duration 11 × effect 8 × edge bundle B_n`：

```text
A_n = 88*n*B_n
```

七个非-VERIFY known kind 的 outcome 固定 N/A；VERIFY 有四个 outcome；UNKNOWN kind 权重一且只能 terminal。因此单个 active event、给定至多 n 个 node 的状态数为 `12*A_n`。数值为：

| n | `E_n` | `B_n` | 单 event states `12*A_n` | bits |
|---:|---:|---:|---:|---:|
| 1 | 1,836 | 1,686,367 | 1,780,803,552 | 30.729881229389 |
| 2 | 449,820 | 101,169,241,111 | 213,669,437,226,432 | 47.602373891474 |
| 3 | 1,343,952 | 903,104,161,129 | 2,861,033,982,456,672 | 51.345458056949 |
| 4 | 2,684,232 | 3,602,552,057,029 | 15,217,179,888,890,496 | 53.756550535562 |

最后一行是 **per-event maximum conditional alphabet**，不是把三 event 直接立方后的整个 packet；共享 node、grammar 和 onset order 仍需继续扣除。

### 5.3 kind path、node union 与 action exact count

对长度 `p=1..8` 的合法 kind path，按 VERIFY 的四种 outcome加权，路径数是：

```text
C_p = [12, 39, 108, 289, 759, 2013, 5408, 14562]
```

action 的 `p` 个 event 中，第一个 onset固定 ACTIVE，其余 `p-1` 个 onset从 17 个有序值（16 bins + UNKNOWN）非降序选择：

```text
T_action(p) = C(17+p-2, p-1)
```

若只允许引用某个大小为 `n` 的 node subset、但不要求全部用到，非空 K-event packet 数为：

```text
F_K(n) = sum_{p=1..K} T_action(p) * C_p * A_n^p
```

present node 集合必须恰好等于 references union，因此用包含–排除：

```text
H_K(n) = sum_{j=0..n} (-1)^(n-j) * C(n,j) * F_K(j),  F_K(0)=0
```

standard action packet 的 exact alphabet：

```text
S_action = 1(NULL)
         + sum_{n=1..4} C(4,n) * 1802^n * H_3(n)

= 345864043392188289642033658185203030253953267411599196753571073

log2(S_action) = 207.74974691951957 bit
ceil(log2(S_action)) = 208 bit
```

这比 v1 Q12 的 186.8500 bit 高约 20.8997 bit，而不是原独立槽草案所暗示的高约 95.08 bit。

### 5.4 Qmax=8 planner-state exact upper bound

Q8 planner 使用 18 个 future onset values（action 的 17 个值再加 `FAR_GT96`）。它既允许 active-led tube，也允许 IDLE 时的 future-only tube。分别有：

```text
T_active(p) = C(18+p-2, p-1)
T_future_only(p) = C(18+p-1, p)

S_Q8 = 1 + sum_{n=1..4} C(4,n) * 1802^n
             * IE_n(sum_{p=1..8}
                 [T_active(p)+T_future_only(p)]*C_p*A_n^p)

= 1465856518066138790269135634824611452184694655527621382890063156278454756789622415135820384526316544093611241215440752563073698020313423453248257

log2(S_Q8) = 478.9093895593358 bit
fixed container width = 479 bit
```

这是 Q8 **coarse planner packet** 的 exact grammar bound；reviewed sidecar 的 uint32 exact boundary intervals、provenance和置信度是 supervision database，不是 action-visible channel，也不混进该数字。

### 5.5 equal-bit control：必须全 packet 排名，不能逐 slot 置乱

原 synthesis 提议对 node/event/edge 各 slot独立 affine permutation。由于 v2 有跨 slot dependencies，这会把合法 packet 变成 edge 指 absent node等非法状态，或不能均匀覆盖 reachable grammar。standard control 改为全 packet enumerative codec：

```text
rank(packet)   -> u in [0, S_action)
u_control      = (a*u) mod S_action, gcd(a,S_action)=1
packet_control = unrank(u_control)
```

使用乘法置换且不加 `b`，使唯一 NULL `u=0` 保持 NULL；`a!=1`，并保存 `a`、模逆、ranking grammar SHA、big-endian limb layout 和 test vectors。`rank/unrank` 的顺序严格对应上述求和：NULL → node subset lexicographic → node states → event length/path/onset → focus/edge bundle。

训练时 hard forward 使用 `packet_control`，backward 只用显式 ST surrogate；planner、aux labels、13-token action encoder、参数/FLOPs、cache policy均与 semantic v2 相同。这个 control 是可逆的同 alphabet 表示归纳偏置 control，不是删信息实验。必须另报经验 entropy/code usage；equal-bit 持平时删除“人类字段语义必要”claim。

## 6. 三个复杂度版本

三者共享相同 missingness、grammar、enumerative counting与严格 route；差异只在明确列出的 maxima/字段，不能运行时随样本改变 schema。

| 版本 | 固定配置 | wire | exact action states / bits | Qmax states / bits | 角色 |
|---|---|---:|---:|---:|---|
| **minimal** | `N=3, K=2, Q=6, R=1`；node只含 observability+center；part缩为6类 | 34 B | `4,451,687,442,164,227,332,862,081` / `81.880626580338` | `6,307,265,436,130,146,861,242,190,910,107,091,440,072,082,743,966,097,703,480,750,977` / `211.938484627037` | 低成本 oracle/标签 pilot；会丢 scale/orientation、fixture和双接触 |
| **standard（推荐）** | `N=4, K=3, Q=8, R=2`；本 memo 第2节 | 82 B | 上述 `S_action` / `207.749746919520` | 上述 `S_Q8` / `478.909389559336` | 足以表达双手、tool-target relation、duration/effect；仅比 v1 多约20.9 bit |
| **full** | `N=4, K=4, Q=12, R=2`；node加 `near/mid/far/unknown` depth；edge加同词表的 secondary part/cell | 124 B | `569597333259431493426655724139003548025680829467289001685030314995047915693155264497188008028137729` / `328.058895690873` | exact formula同上，`log2=873.028486350756` | 只作 upper bound；容量、标签和头部复杂度过大，不建议主跑 |

full 的 Q12 exact state integer极长，未来 `capacity_metadata()` 必须以十进制字符串和 SHA保存；memo 不把超长整数当可读表格。推荐 standard，因为它保留原 insight 的 active+next2、四 role 与双 edge，同时删除最可疑的 confidence、depth和secondary geometry自由度。

## 7. Continuous pose/contact 与 F-PCD 的唯一位置

连续 2D landmarks、MANO、metric 3D wrist/tool pose、mask/flow、continuous contact map、spline/DMP/ProMP系数只能用于：

1. planner/backbone 的 masked auxiliary target；
2. event-aligned vs uniform/dense compression probe；
3. soft/continuous action-visible upper-bound baseline；
4. annotation proposal，并保留 teacher/version/error/provenance。

它们不得进入 standard action packet、CAPOT主 cost、action KV/cache或 equal-bit codec。主动作 loss不得通过 continuous auxiliary decoder hidden回传一条额外 action path。

F-PCD 保持 **独立 challenger**，不追加到 standard event 字段。Idea J 的计数可复算为：

```text
S_FPCD_slot = 1 + 3*9*5*7*8*7*9 = 476281
4 slots = 75.445813886781 bit
```

公平比较是 mutually exclusive `v1-only / F-PCD-only / SM-FET-standard-only / full-trajectory`。F-PCD 也可只作 training-only derivative auxiliary，此时 action-visible capacity仍是 207.7497 bit。若只有 `SM-FET + action-visible F-PCD/continuous pose` 有效，必须报告新联合容量并停止“small finite event schema足够”的主张。

## 8. Attention、mask、cache、teacher forcing 与 missingness

### 8.1 attention contract

逻辑 stream：

```text
action stream: [state | 13 hard packet tokens | noisy action]
visual stream: [causal/current reference | future visual target]
```

- planner queries可读 state、text、current reference、合法 causal history和上次 hard clock；不可读 noisy/GT action、future visual、GT current/future event、posterior、human frame/CAPOT。
- state rows在 strict v2 中不可读 reference，否则 action 可经 state→action旁路取图。
- hard packet rows在 joint transformer 中不可再读 reference；它们只能由外部 planner的 canonical bytes编码。
- action rows可读 state、packet、action和 text cross-attention；不可读 reference/history/future/posterior/raw planner。
- future visual rows可读 action/state/packet/reference以维持 action-conditioned future prediction，但其输出/hidden不能回流 action。
- strict route必须拒绝 `encoder_hidden_states_image`；否则 line 1806 的 image-text concat 会绕过 self-attention mask。

### 8.2 cache contract

cache allowlist：

```text
schema/grammar/route receipt
canonical 82-byte packet + packet SHA
EpisodeClockState 的离散 action-safe子集
由同一 packet产生的 packet/action-prefix KV
allowed text/state/action KV
observation_frame + observation_sha256
```

禁止 raw logits、soft distributions、planner/reference/history/future/posterior KV、continuous geometry/MANO、pair matrix、human memory。以下任一条件必须 invalidation：新 observation、`frame += 8`、event transition、node binding receipt变化、packet bytes变化、schema/route变化。cached/uncached同 dtype 输出需 bit-equal；若底层 kernel不保证，则预注册 `max_abs<=1e-6`。

### 8.3 teacher forcing

主 predicted route 的 action forward **永不**接收 GT event packet。为训练 episode recurrence，可保留两个明确分离的 forward：

1. `planner_tf_aux`：上一个 reviewed hard clock可作为输入，只计算 planner field/boundary loss；不产生 action、stop-gradient到 action branch。
2. `policy_predicted`：上一个状态来自 causal predicted rollout；编译 predicted hard packet并计算 action/visual loss，完全匹配部署。

oracle variant 是独立 route/checkpoint，只能显式命名 `event_tube_v2_oracle`。禁止像现有 masked oracle merge 那样在同一 batch按字段混合预测与 GT 后再把它叫 predicted policy；字段混合会产生部署时不存在的合法性组合。

### 8.4 loss 与 missingness

- categorical field：完整 known group 才做 CE；adjudicated UNKNOWN 是正常 category，not-reviewed mask 为零。
- presence：只有 EXHAUSTIVE coverage才监督 negative；positive-only尾槽不是零标签。
- boundary：exact/interval/left/right-censored用 discrete survival NLL；不把窗口边缘当真实 onset/end。
- event/edge canonical assignment由稳定排序完成；若 reviewed 同 onset无法合并或 overflow，样本不进入 action condition。
- node/edge/effect/outcome各 loss独立归一，防大量 node cell维度压过稀有 outcome。
- CAPOT coupling、continuous pose teacher和 proposal confidence均 `stop_gradient`；confidence仅乘 loss weight，不是 planner input或action字段。
- `hand_status` 仍只可作 grip/open proxy或弱 proposal，不能监督 contact presence/mode/onset/release/effect。

## 9. canonical `e212afec` 的具体插入位置

以下是未来实现任务书，不是本轮改动。

### 9.1 新文件与类

新增 `world_action_model/event_tube_schema.py`，至少包含：

```text
NodeCodeV2 / EventCodeV2 / EdgeCodeV2 dataclasses
EpisodeClockState
validate_event_tube_packet_v2()
canonicalize_event_tube_packet_v2()
compile_action_event_tube_v2()
update_episode_clock_v2()
rank_event_tube_packet_v2() / unrank_event_tube_packet_v2()
event_tube_capacity_metadata_v2()
permute_equal_bit_event_tube_v2()
packet_receipt_v2()
```

新增 `world_action_model/models/event_tube_planner.py`：

```text
EventTubePlannerHeads       # factorized node/event/edge/boundary/overflow heads
PackedEventTubeEncoder      # field embeddings sum成13 tokens
EventTubeHardTransport      # canonical hard/ST；不导出 planner hidden
```

不要修改 `world_action_model/contact_transition_schema.py:18-217,270-413` 的 v1 常量、validator、equal-bit或capacity语义。

### 9.2 `transformer_wa_casual_mot.py`

- `:599-631` constructor：加 opt-in `event_tube_*` config；默认全部 disabled，且 route与 `contact_*`互斥。
- `:703-870` contact module初始化旁：实例化 `EventTubePlannerHeads/PackedEventTubeEncoder`，新参数只用 `event_tube_` namespace。
- `:1253-1285` `_plan_contact_logits` 邻近新增 `_plan_event_tube_logits`；planner输入只取 state/current ref/text/previous clock。
- `:1287-1403` transport/route：新增 `plan_event_tube_route`，先 hard decode/validate/compile再给 action；invalid不做 soft fallback。
- `:1405-1495` `encode_action_tokens`：不能复用 `22 -> action_width` linear；改由 packed factor embeddings产生固定13 tokens。
- `:1531-1590` `encode_joint_streams`：返回 factorized aux logits、canonical packet和receipt；GT只传 loss侧或显式 oracle route。
- `:127-194` mask：建议泛化参数名为 `num_plan_tokens`，保留旧 `num_contact_tokens` wrapper保证 v1 bitwise behavior。
- `:1806-1822`：strict v2 fail closed拒绝 image cross-attention，state/packet/action对 reference均按第8.1节切断。
- `:2240-2499` cache：新增独立 v2 validator/allowlist；现有实现会在部分 finite route缓存 `contact_logits`，v2 明确禁止对应 `event_logits`。

### 9.3 trainer

`world_action_model/trainer/wa_casual_trainer_mot.py`：

- `:13-30` import旁引入独立 v2 schema/loss，不改 v1 imports。
- `:111-189` grouped v1 loss旁新增 `grouped_event_tube_loss_v2`、`boundary_survival_nll_v2`、`masked_complete_group_ce_v2`。
- `:555-598` route validation/receipt：验证 v1/v2互斥、teacher-forcing双 forward、schema/capacity SHA。
- `:632-647` checkpoint load：v2只允许新增 `event_tube_*` missing keys；其它 missing/unexpected仍 fail closed。
- `:957-1000` batch入口：读取 `event_*_target/mask/confidence/coverage/censor`；普通 predicted action forward不得传 GT condition。
- `:1007-1055` loss：分开记录 kind/onset/duration/node/edge/relation/effect/outcome/boundary/overflow与 action/visual loss，不用一个 dense BCE。

dataset 未来沿 `world_action_model/datasets/wam_lerobot_dataset.py:285-325,741-819` 的 strict sidecar模式新增 `event_tube_sidecar_dir`；每个 episode只存一次 sparse event表，chunk target由 compiler机械生成，禁止把 coupling或全 episode表复制成 action输入。

### 9.4 checkpoint 兼容与 v1冻结

1. `event_tube_enabled=False` 时不注册任何 v2 parameter，旧 state dict key/shape/forward应 bitwise不变。
2. v2 从 e212afec backbone初始化时使用 allowlisted migration：允许缺失 `event_tube_*`；不得把旧 `contact_query_tokens/contact_decoder/contact_condition_encoder` reshape后冒充初始化。
3. v1 checkpoint只能以 v1 route加载；v2 checkpoint必须带 schema/grammar/capacity/ranking receipt SHA，不匹配即拒绝。
4. 保存时记录 base checkpoint SHA、v1 frozen schema SHA、v2 packet receipt、route互斥值和所有新参数key列表。
5. `strict=False` 不能继续只打印任意 missing/unexpected keys；v2 loader必须比较精确 allowlist，否则现有 `load_pretrained_weights():1075-1133` 会把真实不兼容静默成 warning。

## 10. CPU-only、暂不实现的验证不变量

未来实现前先把以下测试写入正式 test suite；若为了一次审计创建临时脚本，测试后删除。

1. **enum/version**：枚举数值、顺序、duration/onset边界与 schema SHA固定。
2. **positive zero**：拒绝 NaN/Inf/`-0.0`；absent node/event/edge payload必须 byte-zero。
3. **round trip**：`pack(unpack(bytes))==bytes`，canonicalize幂等。
4. **unknown/absent/censor**：三者不互相映射；GT96与RIGHT_CENSORED round trip不同。
5. **reference integrity**：focus/primary/secondary不能指 absent node，secondary不能等于primary，OBJECT_OBJECT必须有secondary。
6. **prefix/order/duplicate**：event/edge presence为prefix；permutation后canonical bytes相同；重复edge拒绝。
7. **node union**：unused present node拒绝；删掉唯一引用后 node必须自动变 absent或packet invalid。
8. **grammar**：枚举所有 kind pair检查 adjacency；UNKNOWN非末尾、TERMINAL后同 episode复活均拒绝；IDLE/NULL 后合法 future晋升必须接受。
9. **overflow**：Q/N/R每种 overflow返回指定control error，不产生截断 packet。
10. **bin edges**：逐一检查 frame `1,2,3,4,5,8,9,16,17,24,25,32,33,48,49,96,97`。
11. **clock**：stale frame拒绝、continuation保持ID、合法transition递增ID、episode reset清空、每8帧cache失效。
12. **capacity toy proof**：对缩小的 N/K/R/vocab穷举，与包含–排除公式逐项一致。
13. **capacity constants**：standard big integer、`207.74974691951957`、Q8 big integer和 old `281.9327000380615`审计值逐字匹配 receipt。
14. **rank/unrank**：边界 rank `0/S-1`、随机大整数、toy全穷举双向一致；affine逆置换正确且NULL固定。
15. **mask graph**：CPU构造 attention adjacency，证明 action/state/packet对 reference/future/image-cross-attn不可达，而planner对current ref可达。
16. **teacher forcing**：改变当前/future GT label bytes但不用 oracle forward时，pre-safety action严格不变。
17. **fixed packet swap**：固定 packet/state/text/noise替换 reference，action `max_abs<=1e-6`；对raw planner hidden/logits梯度为零。
18. **cache**：allowlist外任一 key拒绝；cached/uncached packet bytes、receipt与action一致。
19. **missingness loss**：not-reviewed字段梯度为零；adjudicated UNKNOWN有CE；positive-only padding不产生presence negative。
20. **checkpoint**：v1 disabled模式state keys/forward一致；v2 migration只有 `event_tube_*` missing；任一 receipt不符拒绝。

这些测试都可在 CPU、小 tensor和纯整数上完成；不需要 conda环境、模型训练、CUDA或大数据写入。

## 11. 推荐、风险与止损

最终推荐 **standard**：它比 v1 多约20.9 bit，却保留 active+next2、显式 duration、四角色、两条接触/关系边与 effect/outcome；82-byte packed schema也显著小于825-float草案的工程表面积。

最主要风险按优先级是：

1. **oracle sufficiency**：208-bit hard packet可能仍缺连续倾角/路径；必须先做 oracle vs 22D/full trajectory/F-PCD。
2. **标签可靠性**：relation/effect/outcome和2D cell在遮挡/透明液体上可能大量 unknown；unknown不是用VLM补真值的许可。
3. **Q/N/R overflow**：四固定role和两edge是可证伪假设，不是事实；先做12段双标统计。
4. **route leakage**：现有 text直达合法，但 image cross-attention、state→reference、cache logits和masked oracle merge都需v2 fail closed。
5. **equal-bit解释**：可逆置换持平不说明语义无用，collapse也不自动说明语义必要；必须结合利用率、field intervention和matched auxiliary。
6. **F-PCD竞争**：75.45-bit derivative-only可能在旋转/regrasp更高效。它应独立挑战standard；在oracle证据前不扩主schema。

因此当前仍是：只批准 schema/annotation/capacity 的 CPU 审计；不训练 WAM/5B，不用 GPU，不改 canonical，不把现有 66 Ego + 88 task-mismatched robot数据当作 human–robot positive pair。
