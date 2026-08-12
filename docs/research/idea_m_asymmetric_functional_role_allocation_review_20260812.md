# Idea M 独立调研：Morphology-Aware Functional Role Allocation

- 日期：2026-08-12
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 隔离分支：`codex/idea-m-asymmetric-functional-role-allocation-20260812`
- 目标硬件：Unitree G1 + 左 RH56DFTP + 右 OmniPicker
- 目标接口：SM-FET `minimal-r1d` + WAM；本文只做文献核查、结构与实验协议，不实现、不训练、不使用 GPU、不操作真机
- idea-discussion：`idea_id=27c8dbd6-9b9d-4858-9418-9c7ad0536359`，`discussion_id=54ad556d-5d94-4305-9447-e67daa7f1447`，`memory_updated=true`

## 0. 结论先行

**当前裁决：Conditional Go as an auditable SM-FET supporting compiler；No-Go as a standalone “functional role allocation / many-to-many retargeting” novelty claim。**

用户指出的问题是真问题：把人类左、右手轨迹分别送给机器人左、右末端，是把 human actor identity 错当成 task semantics。对于左 RH56 灵巧手、右 OmniPicker 单 aperture 夹爪，正确中间量不是 `human_left -> robot_left`，而是一个不含 human literal side 的功能需求集合：

```text
EFFECT_GOAL             任务效果必须由谁实现
SUPPORT_REQUIREMENT     哪个对象/部件必须被稳定或提供反力
OBSERVATION_REQUIREMENT 哪个关系只需被 fresh evidence 确认
```

目标侧再依据固定 capability profile，把这些需求编译为 `OPERATE / STABILIZE / MONITOR` constraint bundle。这样会自然出现 actor 数量不守恒：

- human 一只手同时持稳 pipette tip 并按 plunger，源 actor 数是 1，robot packet 可以是同一个 `LEFT_RH56` 的两条显式 constraint；
- human 两只手分别操作与支撑，目标场景若已有经验证 fixture，则 robot 可以只有一个 physical effector，再以 `MONITOR_ONLY+MONITOR` 表达 passive support；
- human 两手动作很像，但目标 RH56/Omni 能力不对称，最终 assignment 仍可不同；
- 无唯一可行 binding 时必须 abstain/no-packet，不能用 learned score、candidate order 或 profile lookup 暗选。

但这条大方向已经被多条强前例分割：

1. Alissandrakis 等 2007 年已用 correspondence matrices 表达跨不同 embodiment 的 partial、one-to-many、many-to-one 与 many-to-many body mapping；
2. BUDS 已显式提出 `stabilizing arm + acting arm` 角色分解；PA-BiCoop 已动态把 primary/auxiliary roles 分配给左右臂；
3. XSkill/UniSkill 已学习跨具身 skill representation；Affordance Equivalence 已以不同 actor/action 的相同 effect 做跨机器人迁移；
4. DexMachina 已展示同一 human reference 在不同 dexterous hand morphology 上产生不同的双手策略；SoftAct 已按功能力分布进行非一一 human-finger→robot-finger allocation；
5. Human2Any 已把 human object–object interaction prior 与 robot-specific grasp/reachability/collision feasibility 组合；多臂 TAMP 也早已按异构 gripper/suction capability 分配任务。

所以不能再写“首次不做 1:1 retarget”“首次动态分配操作/稳定角色”“首次 many-to-many 跨具身对应”或“首次依据 morphology 重分配功能”。当前真正可守的增量很窄：

> 将 human actor grouping 与 domain-neutral functional requirements 分离；只用冻结的 finite capability/resource registry 和公开 runtime receipts，把需求编译为 `minimal-r1d` 已有的显式 O/S/M + assignment packet；部署 action 看不到 human video、allocator logits、pair ID、mapping graph 或自由 latent；并用 fixture/monitor、same-effector compound、heterogeneous-effector counterfactual 明确否证它。

这足以成为 SM-FET 的关键 compiler、数据协议和 benchmark axis；在只有一个 RH56+Omni 目标 profile 时，不足以成为一篇独立 morphology-aware algorithm。独立晋级必须至少增加三个真实 morphology/profile、leave-one-morphology-out、同一功能需求的多种真实 allocation，以及强对照下的下游因果收益。

## 1. Zotero-first 结果与检索质量

### 1.1 idea-discussion

本轮先按 `idea-discussion` 同步默认 Zotero，更新了上面的 idea card。自动 librarian 返回 RoboBrain 系列通用笔记，自动 scout 还误召回 image-manipulation detection；这些结果不支持本题 novelty，未被当作证据。

随后对本地 Zotero 数据库做 exact title/keyword 复查，真正相关的个人库条目包括：

| Zotero key | 条目 | 本题用途 |
|---|---|---|
| `V7LLF6VJ` | XSkill: Cross Embodiment Skill Discovery | shared skill prototype 强基线 |
| `EK2CTN7J` | UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations | unpaired/shared representation 强基线 |
| `8WMVVJAL` | ImMimic: Cross-Domain Imitation from Human Videos via Mapping and Interpolation | 轨迹映射、DTW/GMS-SDTW 基线 |
| `A89APIR3` | EgoBridge | joint latent-action OT / cross-domain co-training 基线 |
| `JEZ26KWY` | One-Shot Imitation under Mismatched Execution | mismatched execution 与 sequence retrieval |
| `CSMS2VV2` | TopoRetarget | interaction/contact topology preserving retarget 基线 |

Zotero 中没有 exact 命中 BUDS、PA-BiCoop、DexMani、DexMachina、SoftAct、Human2Any 与 Affordance Equivalence，因此按 Zotero-first 规则再查原始论文/作者项目页。`arxiv-search` CLI 本身未打印返回值，直接调用其函数后又遇到 arXiv API HTTP 429；最终只采用 arXiv paper page、PMLR、IEEE/KIT、作者项目页等第一方来源逐项核验，不采用自动摘要的相关性判断。

### 1.2 搜索范围

定向检索覆盖：

```text
bimanual role assignment / active-passive or primary-auxiliary hands
heterogeneous end-effectors / gripper+suction allocation
cross-embodiment retargeting and mismatched execution
tool-use functional correspondence / affordance equivalence
task-and-motion allocation / feasibility-aware planning
many-to-many correspondence mapping
DexMani / DexMachina / SoftAct / XSkill / Human2Any
```

检索结论是“没有看到完全同构于 `finite SM-FET role compiler + fixture MONITOR + hidden-channel firewall` 的系统”，但各组成思想几乎都有强前例；因此 novelty 只能来自严格组合与评测，而不是概念名词。

## 2. 最强前例与可守边界

| 工作 | 已经覆盖什么 | 对 Idea M 的直接冲突 | 仍未覆盖的窄边界 |
|---|---|---|---|
| [Correspondence Mapping Induced State and Action Metrics, 2007](https://doi.org/10.1109/TSMCB.2006.886947) | 跨不同 DOF/morphology 的 correspondence matrix；明确支持 one/many-to-many | many-to-many body mapping 不是新概念 | 它映射 state/action aspects，不是 finite functional-requirement→resource-validated event compiler |
| [A Bimanual Manipulation Taxonomy](https://doi.org/10.1109/LRA.2022.3196158) | 以 hand role、symmetry、contact graph 分类 human bimanual action | human hand roles/contact graph 不是新 taxonomy | 不负责异构目标末端、fixture 与 action bottleneck |
| [Stabilize to Act / BUDS](https://arxiv.org/abs/2309.01087) | stabilizing arm 保持环境、acting arm 执行任务；四个真机任务 | `OPERATE/STABILIZE` 二分与降维动机直接相撞 | 角色结构基本固定，目标是同构双 UR16e；没有 human→heterogeneous grounding、fixture MONITOR 或 many-to-many actor cardinality |
| [PA-BiCoop](https://arxiv.org/abs/2606.28192) | 动态 primary/auxiliary 分工，并自动映射到左右臂 | “动态主辅臂分配”不能 claim | 仍把输出组织成两臂连续动作；没有有限 capability registry、passive fixture、strict no-hidden-channel contract |
| [XSkill](https://proceedings.mlr.press/v229/xu23a.html) / UniSkill | 从 human/robot video 学跨具身 prototype/skill representation | embodiment-agnostic skill abstraction 不是新 | learned prototype 可成为自由 latent；未显式审计 O/S/M、资源互斥与 fixture substitution |
| [Affordance Equivalence](https://arxiv.org/abs/2404.15648) | object/action/effect shared latent；不同 actor/action 产生相同 effect 的跨机器人迁移 | functional/effect equivalence 与跨 actor grounding 不是新 | 没有 `minimal-r1d` finite wire、显式 resource claims 和 action-channel 审计 |
| [MimicFunc](https://arxiv.org/abs/2508.13534) | function frame 与 keypoint-based functional correspondence；novel tools | 工具功能对应不是新 | 主要是 tool/target 几何对应和动作生成，不是 actor-role cardinality / heterogeneous effector allocation |
| [ScrewMimic](https://robin-lab.cs.utexas.edu/ScrewMimic/) | human 两腕+接触点到 bimanual screw action | bimanual human video transfer 强轨迹基线 | 仍依赖两手相对运动；不允许 fixture 替代或同 effector compound |
| [DexMachina](https://project-dexmachina.github.io/) | 同一 human object-state reference 在六种 hand designs 上学习不同可行策略；示例中小手会改变双手用法 | “morphology 迫使不同策略/角色”已有直接实证 | 仿真 RL 功能 retarget，仍保留 hand/contact/motion guidance；没有有限功能需求 compiler |
| [SoftAct](https://arxiv.org/abs/2604.01224) | 从 human contact-force distribution 到非 anthropomorphic soft hand；按力分配 robot fingers | 非一一 functional allocation 与 extreme embodiment mismatch 已被覆盖 | 单手内 finger/force allocation，不是双末端+fixture 的 event-role compiler；需要 dense VR force supervision |
| [DexMani](https://arxiv.org/abs/2608.00554) | human contact-conditioned manipulability evolution 指导多种 robot hand rotation | contact transition、morphology 与后续可持续性结合不是新 | 单手连续旋转；不做双末端角色/fixture 分配；其动态信息也是 coarse role schema 的反例 |
| [Human2Any](https://arxiv.org/abs/2606.28813) | human object–object interaction prior + robot agent–object grasp/reach/collision feasibility + planning；跨 Franka/RBY-1 | “人类功能约束 + 机器人可行性组合”高度直接冲突 | 用连续 object motion、factor graph 与 planning；没有有限 packet、actor grouping、MONITOR 或 side-channel proof |
| [Cooperative TAMP for Multi-Arm Assembly](https://arxiv.org/abs/2203.02475) | MILP 分配 tasks，再做 collision-free motion；含异构 grippers/suction plates | capability-aware heterogeneous allocation 不是新 | 不是从 human event requirements 学习，也不受 SM-FET action capacity 约束 |
| [Task-driven Perception with Heterogeneous End-Effectors](https://arxiv.org/abs/2006.15503) | vacuum + adaptive finger hand、handoff/regrasp、task-driven feasibility | 异构双臂按能力改变操作序列已有真机前例 | 单一 constrained placement 系统，不学习 human functional roles |
| [Learning Symbolic and Subsymbolic Temporal Task Constraints](https://arxiv.org/abs/2403.16953) | 从 human bimanual demos 学 precedence/overlap/duration并同步 primitives | human bimanual constraint graph 不新 | 主要是 temporal synchronization，不决定哪个异构 effector/fixture承担功能 |

### 2.1 三条明确不能写的 claim

```text
No-Go: first many-to-many cross-embodiment mapping
No-Go: first learned/dynamic operate-stabilize or primary-auxiliary assignment
No-Go: first morphology-aware functional retargeting from human demonstrations
```

### 2.2 唯一可守的 claim

```text
Conditional:
an auditable finite requirement-to-constraint compiler for an asymmetric target,
where source actor cardinality is discarded, passive fixtures are first-class,
the final allocation is fully serialized in minimal-r1d,
and no allocator latent or score reaches the action model.
```

这首先是 systems/representation claim。只有第 13 节的独立晋级 gate 全过，才可把它提升为 morphology-aware allocation method。

## 3. 问题定义：不要映射 hands，要 grounding requirements

### 3.1 Source-side functional requirement set

对 human macro event `e`，先编译不含 literal left/right 的共享需求集合：

```text
Q(e) = {q_i}_{i=1..2}

q_i = (
  purpose,                 # EFFECT_GOAL / SUPPORT_REQUIREMENT / OBSERVATION_REQUIREMENT
  primary_role,
  primary_part,
  secondary_role,
  target_relation
)
```

这正好复用 `minimal-r1d` 的 `sm_fet_r1d_shared_semantic_key_v1`。human hand identity 只留在 annotation sidecar：

```text
source_actor_group = H0 | H1 | BOTH | UNRESOLVED
```

它只用于检查“一只手承担几个 requirement”和“两只手是否被目标 fixture 吸收”，不进入 semantic key、pair cost 或 action。

关键解释是：**many-to-many 发生在 source actors、functional requirements 与 target physical resources 三层之间，不需要把 human hand ID 塞入 packet。**

例如 human 单手移液并不是“一个不可分解 action 映射到两个 robot action”。它应标成同一 `source_actor_group` 下的两条需求：

```text
EFFECT_GOAL            ACTUATOR_OR_PLUNGER
SUPPORT_REQUIREMENT    TOOL_TIP / held pipette pose
```

然后两者分别绑定到同一个 `LEFT_RH56` 的 `OPERATE` 与 `STABILIZE` constraint。相反，human 两手旋盖是两个 source actor groups，但目标瓶体若被 verified fixture 锁定，support requirement 可以绑定为 `MONITOR_ONLY+MONITOR`，所以只有一个 physical robot effector动作。

### 3.2 Target-side capability/resource profile

固定 profile `P` 只能包含有限、版本化的静态事实：

```text
effector IDs
allowed O/S/M functions
task parts and working-surface classes
finite resource claim templates
static mutex and same-effector allowlists
required runtime predicate names
profile ID + content hash
```

对于当前目标：

```text
LEFT_RH56
  wrist pose + six active hand channels
  candidate fingertip/fingerpad/palm/thumb-actuation surfaces

RIGHT_OMNIPICKER
  wrist pose + one coupled aperture
  independent RIGHT+RIGHT aperture targets are impossible

MONITOR_ONLY
  zero own actuator slice
  requires fresh passive relation evidence
```

profile 不得包含 task instance pose、轨迹、held transform、scene ID、allocator score、当前可达性或 per-sample profile selector。held tool、fixture、WBC、collision、contact freshness 分别来自显式 public state/runtime receipts。

### 3.3 Allocation as finite constrained grounding

给定 `Q`、固定 profile `P` 和公开 runtime admissibility `R`，生成所有 canonical candidate bundles：

```text
C(Q,P,R) = {
  (binding_edges, canonical minimal-r1d bundle, finite reason codes)
}
```

每条 binding edge 只做：

```text
EFFECT_GOAL             -> OPERATE
SUPPORT_REQUIREMENT     -> STABILIZE or fresh-fixture MONITOR
OBSERVATION_REQUIREMENT -> MONITOR
```

候选必须满足：

1. `minimal-r1d` 1..2 constraint grammar；
2. function/assignment 唯一合法组合；
3. part、surface、locator 与 controller enum 在 profile 中唯一；
4. wrist/finger/aperture/held-object resource 不冲突；
5. same-effector compound 在 exact allowlist；
6. fixture、held state、reachability、WBC、collision receipts fresh；
7. required evidence unknown 时无 candidate，而不是降级成 MONITOR/OTHER。

最终状态只有：

```text
UNIQUE(candidate)
AMBIGUOUS(no packet)
INFEASIBLE(no packet)
UNKNOWN(no packet / reobserve)
```

如果两种 binding 会产生不同动作，但当前有限输入无法区分，必须 `AMBIGUOUS`。这条 abstention 不是保守附加项，而是防 allocator 变成暗通道的核心定义。

## 4. 模型结构：Finite Role Grounder（FRG）

### 4.1 推荐两级版本

**M0：确定性 compiler，推荐先做。**

```text
human/robot event planner
        -> shared finite requirements Q
        -> enumerate P-valid bindings
        -> static/runtime validator
        -> unique canonical minimal-r1d bundle or abstain
```

M0 没有 learned allocator。它先证明 schema、profile 与 benchmark 是否真的需要 allocation，而不是让模型掩盖 ontology/profile 缺口。当前项目应把 M0 作为 supporting compiler 的正式路线。

**M+：learned finite scorer，只用于独立晋级实验。**

当多个静态候选存在、且公开 finite candidate/runtime features 已经给出可区分证据、同时有 adjudicated allocation gold 时，可加入小型 set-to-set factor scorer。若 canonical finite inputs 完全相同却出现不同 gold，标签应判为 `AMBIGUOUS` 或 schema 缺字段，不能要求 learned score 暗分：

```text
requirement nodes: finite q_i fields
target nodes: LEFT_RH56 / RIGHT_OMNIPICKER / MONITOR_ONLY
factor nodes: working surface, resource template, fixture/held validity

2-layer finite relational message passing or DeepSets
  -> edge feasibility logits
  -> bundle score
  -> calibrated ABSTAIN logit

hard profile mask -> hard argmax/abstain -> validator -> canonical packet
```

输入只允许 frozen enum embeddings、known masks 与 finite runtime predicate outcomes。M+ 不读 raw human video、continuous human/robot trajectory、pixels、depth residual、held transform 数值、scene ID、task outcome history或 WAM hidden。几何仍由 `minimal-r1d` planner/locator 和公开 controller state处理；allocator只回答谁承担哪个显式 constraint。

建议 score 分解为：

```text
S(b) = sum requirement-to-binding compatibility
     + sum capability/working-surface compatibility
     + sum pairwise resource compatibility
     + finite runtime availability terms
```

不使用一个 unrestricted MLP 直接从视觉输出 assignment。否则 assignment token本身虽是有限的，allocator仍可把视觉/trajectory hidden压进 candidate choice，破坏“功能分配”解释。

### 4.2 与 WAM 的唯一接口

WAM 不新增 allocator branch。推理路径固定为：

```text
robot current image/history + canonical25 proprio + structured text
  -> robot event/requirement planner
  -> M0 or frozen M+ grounder
  -> hard validate / canonicalize
  -> 5 minimal-r1d hard tokens + public execution state
  -> WAM predicts 48 steps
  -> execute first 8
  -> reobserve, invalidate receipts, re-ground
```

action head看不到：

```text
human video / human actor side / source_actor_group
candidate list / candidate count / candidate order
allocator logits / margin / hidden / attention
binding edge IDs / allocation receipt / reason code
pair table / OT coupling / pair ID
profile selector / profile bytes / task instance lookup
```

final assignment/function/part/locator 已全部进入 canonical `minimal-r1d` packet；这就是允许传给 action 的完整 allocation message。

### 4.3 与 CAPOT、RE-Q、WAM 的分工

```text
CAPOT / shared key:
  训练期连接同功能 human/robot events，不用 literal hand side

RE-Q / effect supervision:
  检查 pre/post relation 与 effect 是否等价，不决定 target effector

FRG (Idea M):
  把 shared requirements grounding 到 target O/S/M + assignment

minimal-r1d:
  序列化最终有限 constraint bundle

WAM:
  在 hard bundle + robot state 下生成连续动作
```

FRG 不应重新发明 cross-domain matcher，也不应预测 trajectory。否则会和 Idea G/K、Human2Any、DexMachina、ImMimic 的问题重叠，同时丢掉 SM-FET 的可审计边界。

## 5. Training：先监督 allocation，禁止 action loss 反向发明通道

### 5.1 三段式训练，不能端到端混写

1. **Requirement supervision**
   - human 与 robot event 各自预测 shared finite requirements；
   - human `source_actor_group` 只做 annotation QA/actor-cardinality probe；
   - literal human left/right 不参与 cross-domain loss。
2. **Allocation supervision**
   - 对固定 `Q,P,R` 枚举全部候选；
   - oracle/专家只标 `unique positive / feasible equivalent / infeasible / ambiguous`；
   - 用 bundle CE、edge BCE、hard-negative margin 与 abstention calibration 训练 M+；
   - profile mask、resource mutex 与 unknown rejection 始终是 hard validator，不让模型学习绕过。
3. **WAM training**
   - grounder冻结；只把 gold 或 predicted-and-validated hard packet送入 WAM；
   - WAM loss不反传到 allocator；
   - human clips没有 robot action label，不产生 retargeted pseudo action。

核心损失可写成：

```text
L_FRG = L_bundle
      + lambda_edge * L_edge
      + lambda_cf   * L_capability_counterfactual
      + lambda_abs  * L_abstain_calibration
```

不加 `L_WAM -> allocator` 梯度。否则 allocator可能通过同样合法但细微不同的 assignment/locator code把动作信息编码给 WAM，训练指标会上升但 scientific claim 已失败。

### 5.2 必须构造的 counterfactuals

每个 positive allocation 至少配以下 hard negatives：

- swap `OPERATE/STABILIZE` 但保持其它字段；
- 将 RH56-required thumb actuation 改绑 OmniPicker；
- 让两个独立 target竞争 OmniPicker aperture；
- 删除 fresh fixture evidence却仍输出 MONITOR；
- 把同一 human hand group强制映成单constraint；
- 把 human 两手强制映成两个 physical effectors，即使 fixture已验证；
- profile hash不变但暗改 working surface/resource claim；
- current held binding过期但沿用旧 packet；
- candidate order随机打乱；
- capability mask去掉一个 effector，检查是否正确 reallocate 或 abstain。

### 5.3 human–robot pairing 不需要同步

正 pair 只要求相同 task cell 与 shared requirement key，不要求 trajectory、时长或 hand side 对齐。一个 human event可连接多个 robot execution regimes：

```text
active right stabilization
verified fixture + MONITOR
same-left compound
infeasible/abstain under capability removal
```

因此 pair table是 group supervision，不是 human_i→robot_i 唯一对应。split必须先按 human actor/session、robot operator/session、tool/container instance分组，再在 split 内建立 pair/candidate graph。

## 6. Inference：human 完全退出，allocation 每 8 steps 重审

部署时没有 human prompt video。唯一允许路径：

```text
1. robot causal planner 从当前 observation/history 预测 Q
2. 读取固定 profile hash 与 allowlisted public/runtime receipts
3. 枚举 candidate bundle；hard mask static/resource invalid candidates
4. M0 unique-select，或 frozen M+ 在有限输入上 select/abstain
5. validator canonicalize 成 32-byte / 5-token minimal-r1d packet
6. WAM 预测48步，仅执行8步
7. 重观测；fixture/contact/held/reachability过期即清 cache并重新ground
```

`MONITOR` 不是“这只手不动”的别名。只有 fixture/support relation 的 fresh native evidence存在时，`SUPPORT_REQUIREMENT -> MONITOR` 才合法；evidence缺失必须 no-packet。`MONITOR_ONLY` 自己 action slice恒为零，但其 token可以条件化 physical OPERATE slice是否推进。

## 7. Finite schema 与 bit 裁决

### 7.1 不增加 core action field

`minimal-r1d` 已经有完成 grounding 所需的全部 action-visible字段：

```text
constraint_function: OPERATE | STABILIZE | MONITOR
assignment: LEFT_RH56 | RIGHT_OMNIPICKER | MONITOR_ONLY
primary_role / primary_part
locator: POINT | AXIS | FRAME | PATH
secondary_role / intended_target_relation
```

因此 Idea M 的推荐版本：

```text
incremental action-visible bits = 0
wire                         = unchanged 32 bytes
hard tokens                  = unchanged 5
G_struct_r1d                 = unchanged
log2 |G_struct_r1d|          = 96.99142134161319 bit
G_profile(g1-rh56-omni, t)   = still UNDEFINED / NO-GO until real profile exists
```

“0 bit”只表示相对 `minimal-r1d` 不新增字段；不能说系统总 allocation 信息是 0 bit。O/S/M、assignment与bundle topology已经被计入 r1d 的 96.991 structural bits。

### 7.2 只新增 annotation/receipt sidecar

建议增加一个 action-invisible `functional_allocation_receipt_v1` 概念对象：

```text
schema_id
shared_semantic_key_hash
static_profile_hash
runtime_receipt_hashes

source_requirements[]:
  canonical requirement rank
  source_actor_group          # annotation QA only

bindings[]:
  requirement rank
  target constraint slot
  binding kind = PHYSICAL | PASSIVE_MONITOR

allocation_status = UNIQUE | AMBIGUOUS | INFEASIBLE | UNKNOWN
finite reason_codes[]
candidate_set_hash
```

禁止加入 action 的字段：

```text
allocation_motif_id
candidate rank/order/count
allocator score/logit/margin
learned role embedding
human actor ID or hand side
profile selector
continuous capability vector
```

receipt用于 audit、label provenance与失败分析；action模型不能读取它。若 action需要它携带的信息，必须把对应语义 promotion 为有限 core field、重新计 bit 并做 equal-capacity control，不能偷接 sidecar。

### 7.3 不把“一手→两constraint”编码成新 topology bit

source actor cardinality是 annotation fact，不是 controller所需命令。最终机器人只需要知道两条 target constraint及其 assignments。比如：

```text
human source_actor_group = H0 for both requirements

q0 SUPPORT_REQUIREMENT(TOOL_TIP)
 -> LEFT_RH56 + STABILIZE

q1 EFFECT_GOAL(ACTUATOR_OR_PLUNGER)
 -> LEFT_RH56 + OPERATE
```

这个信息已经由两个 r1d constraints 完整表达。把 `ONE_HUMAN_TO_TWO_ROBOT` 再编码一次只会增加重复字段和潜在 hidden channel。

## 8. 标注与 human–robot pair 协议

### 8.1 Human 最小标注

每个 event/bundle 两位标注员独立标：

```text
phase / effect / event boundary
functional requirements (1..2, EXHAUSTIVE)
purpose / role / part / relation
source_actor_group for each requirement
active contact evidence and visibility
fixture/support relation if visible
effect/outcome sidecar
unknown vs absent vs not-reviewed
```

不要求标 MANO joints、dense hand trajectory 或 human-to-robot assignment。actor group只回答多个 requirements是否由同一源手承担，不用于配对或控制。

### 8.2 Robot 最小标注/receipt

```text
same shared requirements Q
gold O/S/M + assignment bundle
working-surface and resource-template IDs
same-effector compound status
fixture/monitor evidence receipt
held-tool/public-state receipt
reachability/WBC/collision admission result
task effect/outcome
allocation UNIQUE/AMBIGUOUS/INFEASIBLE/UNKNOWN
```

gold assignment不能由执行成功后反推并只保留成功样本；必须保留失败、ambiguous、no-packet与 capability counterfactual，否则模型只学 dataset shortcut。

### 8.3 Pair gold

pair gold 的单位是：

```text
(human shared requirement bundle,
 target profile/runtime stratum,
 robot allocation bundle or abstain)
```

而不是整段 trajectory。每个任务至少需要：

```text
20 functional matches
20 allocation hard negatives
10 fixture active/passive substitutions
10 ambiguous/infeasible/unknown decisions
```

四任务最低 `240` 条双标并裁决 allocation decisions。每类 hard positive中至少一半必须改变 source-actor/target-physical-effector cardinality；否则数据没有检验 many-to-many insight。

## 9. 四任务 benchmark

四任务不是简单报告 success；每个任务都必须同时含 active-support、verified-fixture、capability-removal 与 misleading 1:1 mapping 四种 matched strata。

### 9.1 Pour：effect/support 与 fixture substitution

Human requirements：

```text
EFFECT_GOAL: source OPENING_OR_RIM, FLOW/TRANSFER
SUPPORT_REQUIREMENT: target BODY or target support relation
```

目标候选：

```text
LEFT_RH56 OPERATE source + RIGHT_OMNIPICKER STABILIZE target
LEFT_RH56 OPERATE source + MONITOR_ONLY verified fixture support
no packet when transparent rim/depth/support evidence is unknown
```

变量：human handedness、source grasp affordance、fixture on/off、target位置；不改变 structured task text。指标：transfer success、spill/overflow、target motion、correct support binding、no-packet precision。

### 9.2 Insert→Pipette compound：一只 human actor group 到同一 robot effector两constraint

Stage A：`TOOL_TIP` 对准并进入 `OPENING_OR_RIM`；Stage B：保持 tip/held pipette frame并操作 `ACTUATOR_OR_PLUNGER`。

关键 allocation：

```text
LEFT_RH56 STABILIZE TOOL_TIP
+ LEFT_RH56 OPERATE ACTUATOR_OR_PLUNGER
```

这两条已经占满 r1d constraint budget；若当前 anchor 还要求独立的 receptacle fixture evidence/active support，第三条 constraint 必须触发 overflow/no-packet，不能静默丢掉。当前 r1d 对 pipette same-left compound仍需 task oracle/calibration，所以真实 executable profile未就绪时只做 offline allocation benchmark，不得声称可部署。指标：compound bundle exact match、overflow detection、dose/insert outcome、invalid LEFT+LEFT/RH56 finger-resource accept rate。

### 9.3 Cap：两人手需求到 active support 或 passive support

```text
EFFECT_GOAL: LID_CAP ROTATE
SUPPORT_REQUIREMENT: bottle BODY remains fixed
```

目标候选：

```text
LEFT_RH56 OPERATE cap + RIGHT_OMNIPICKER STABILIZE body
LEFT_RH56 OPERATE cap + MONITOR_ONLY fixture lock
```

必须加入误导性演示：human literal side 与 target optimal assignment相反；1:1 baseline应失败，FRG应保持同一功能 binding。指标：cap outcome、bottle motion、assignment exact match、fixture switch consistency、unsupported reversed-allocation rejection。

### 9.4 Constrained linear wipe：tool path 与 workpiece support

```text
EFFECT_GOAL: WORKING_FACE_OR_PAD follows admitted straight PATH
SUPPORT_REQUIREMENT: workpiece BODY stable
```

目标候选：左 RH56持工具操作，右 Omni稳定 workpiece，或 fixture fresh 时 MONITOR。只允许 fixed plane/normal/width 的 straight wipe；generic curved wipe、dynamic normal、forbidden regions 明确 out-of-scope。指标：clean/coverage、path deviation、workpiece motion、support-mode accuracy、对 generic wipe 的 abstention。

### 9.5 共同 split 与评测

```text
train: human actors/sessions/tools disjoint from test
test: unseen tool/container instance + unseen session
allocation OOD: held-out fixture/capability regime
independent-promotion OOD: held-out morphology profile
```

离线指标：

- exact canonical bundle accuracy；
- requirement→binding edge macro F1；
- infeasible false accept rate；
- ambiguous/unknown abstention F1 与 coverage；
- same-effector compound accuracy；
- fixture substitution consistency；
- capability-removal counterfactual consistency；
- source actor cardinality→target physical effector cardinality confusion matrix。

未来下游指标：task success、event completion、safety violation、no-packet rate、pre-safety/executed action差异；按 task×morphology×session cluster bootstrap 95% CI，而不是只报 trial pooled mean。

## 10. Baselines 与消融

### 10.1 必须 baselines

1. **Literal 1:1 side retarget**：human left/right→robot left/right，含 ImMimic/TopoRetarget 风格轨迹或 contact retarget。
2. **ScrewMimic-style bimanual geometry**：保持两腕相对 screw/action，测试它在 fixture/same-effector时的局限。
3. **BUDS fixed roles**：固定 left act/right stabilize，及最优手工固定方向。
4. **PA-BiCoop-style dynamic primary/auxiliary**：同等视觉/参数预算但无 capability registry/fixture sentinel。
5. **XSkill/UniSkill latent skill**：shared latent/prototype→action；报告其高容量但强性能上界。
6. **Affordance Equivalence / Human2Any-style functional planning**：effect/object-motion + feasibility planning；是最强功能迁移基线。
7. **Heterogeneous TAMP/ILP**：人工 task requirements + exact capability/resource constraints，完全不学习；这是 M0 的最强非学习对照。
8. **Unstructured learned allocator**：raw observation/profile embedding→assignment logits；容量和参数 matched，但允许 latent。
9. **Robot-only minimal-r1d WAM, fixed assignment**：无 human supervision、无 allocator。
10. **Oracle allocation**：人工 gold packet，只给上界，不与 predicted route混报。

### 10.2 关键消融

```text
- source_actor_group supervision
- SUPPORT_REQUIREMENT
- fixture MONITOR option
- resource conflict factors
- same-effector compound allowlist
- capability counterfactual loss
- calibrated ABSTAIN
- learned scorer (M+) vs deterministic compiler (M0)
- WAM gradient stop (strict) vs end-to-end (diagnostic only)
- finite inputs only vs raw visual allocator
- profile fixed vs illegal per-sample profile choice
```

若 M+ 不优于 M0/ILP，删除 learned allocator；compiler仍可保留。若 raw visual allocator胜出而 finite M+失败，结论应是 finite allocation schema不足，不是扩大 hidden channel保护结果。

## 11. 防 allocator hidden channel 的硬合同

### 11.1 进程/张量边界

最强实现应把 allocator 与 WAM 之间的接口物理序列化：

```text
allocator output -> canonical 32-byte r1d wire
decoder round-trip -> 5 hard tokens
only these tokens enter action model
```

禁止共享 allocator hidden/KV、candidate embedding、logit、attention map、binding receipt或 memory cache。即使在同一训练程序，也必须 stop-gradient并通过 explicit serialization boundary。

### 11.2 Matched-input invariants

对 fixed：

```text
r1d packet bytes
structured text hash
canonical25 proprio
public execution state hash
profile/codebook/manifest/calibration hashes
noise/timestep/anchor
```

以下干预不得改变 pre-safety action：

- randomize candidate order；
- randomize non-selected candidate fields；
- randomize allocator logits/margins但保持 selected packet；
- randomize allocation receipt/reason code/pair ID；
- replace source human actor grouping；
- replace human video或完全移除 human branch。

若动作变化，存在 hidden path，所有 allocator/WAM结果 No-Go。

### 11.3 Packet-level interventions

只有合法、可解释的 hard packet变化才允许改变动作：

- O/S function swap改变对应物理 role；
- active STABILIZE↔fresh fixture MONITOR改变 physical slice ownership；
- assignment change只打开对应 split action slice；
- same-effector compound只影响 allowlisted wrist/finger resources；
- MONITOR own slice恒为零；
- invalid/expired receipt清 packet并进入 HOLD/REOBSERVE。

需要继续运行 `minimal-r1d` equal-bit/permutation control。若相同容量的随机语义 rank与 FRG packet效果持平，不能声称功能 role semantics必要。

### 11.4 Profile 不能成为每样本 codebook

训练/评测 split 前冻结 profile bytes/hash；同一 route不能逐样本选择 profile。profile swap只允许作为显式 OOD实验，并且所有方法共同知道哪个 frozen morphology condition正在评估。否则 `profile_id` 本身可编码 task/scene/trajectory。

## 12. 可证伪预测

### P1：literal handedness 应无关

同一 functional requirements下镜像 human hand assignment，FRG的 target bundle应不变；1:1 retarget应明显退化。若 FRG随 human literal side变化，actor identity没有被正确商掉。

### P2：fixture 必须减少 physical effector count

固定 effect goal，只把 fresh fixture receipt从 absent切到 present，support应从 active STABILIZE变为合法 MONITOR候选；删除证据后必须恢复active support或abstain。若始终用两手，morphology-aware substitution没有发生；若无证据也 MONITOR，安全合同失败。

### P3：same-effector compound 必须在精确任务上成立

pipette gold中，同一 human actor group的tip stabilization+plunger operation应编译为 exact allowlisted `LEFT+LEFT`；任意其它 `LEFT+LEFT`不得被泛化放行。若 learned allocator只学到“pipette总是LEFT+LEFT”而忽略 resource/held receipts，counterfactual false accept会暴露它。

### P4：capability removal 应导致可解释 reallocation 或 abstention

移除 Omni aperture/support capability后，输出只能是经 profile批准的 fixture route、其它唯一合法 assignment或 no-packet。不能保持原packet、用连续score补偿或暗换profile。

### P5：M+ 必须胜过 M0，而不是仅胜过 1:1

若 learned FRG只胜 literal retarget，却不胜 hand-written ILP/capability compiler，独立算法 claim失败；保留 M0 supporting compiler。

### P6：收益必须穿过同一 finite packet

在 zero human runtime、无 allocator hidden、相同 r1d packet容量和相同 WAM下，FRG human supervision才可计入贡献。若只在 action读取 mapping latent/logit或 human video时有效，Idea M 的 strict claim失败。

### P7：粗 role schema不能掩盖动态 manipulability

DexMani/DexMachina 风格的持续旋转、recontact或长 horizon任务，可能有相同 O/S/M bundle但未来可持续性完全不同。若此类 collision >10% 或显著降低后续成功，应把任务判为 schema out-of-scope，不能把 manipulability轨迹藏进 allocator score。

### P8：只有一个 morphology 时不能声称 morphology generalization

在 RH56+Omni 单 profile上成功，只能证明 target-specific compiler；必须在 held-out morphology上仍能从 capability profile正确重分配，才允许 morphology-aware generalization表述。

## 13. 数据门槛、Go/No-Go 与 stop rules

### G0：无训练 contract gate

训练前必须完成：

- shared requirements→candidate bundle唯一编译；
- profile/resource/fixture/held-state分层无循环依赖；
- 1..2 constraints内能表达四任务的预注册 positive；
- actor grouping变化不改变 semantic key；
- all hidden-channel invariants可机械检查；
- every invalid/unknown case deterministic no-packet。

任一失败：停止 M+；只修 schema/profile，不能用 learned allocator补洞。

### G1：最小 annotation feasibility

建议先做每任务 `3 calibration + 12 blind human`，共 `12 calibration + 48 blind human`。最低：

```text
requirement exact-set agreement >= 0.80
purpose/role/part/relation kappa >= 0.70
source_actor_group kappa >= 0.80
critical unknown <= 20%
constraint overflow <= 5%
```

协议最多修订一次；仍失败则停止 many-to-many human claim，M0只能接受人工 task specification。

### G2：target allocation pilot

最低 `48 human + 48 target-robot episodes`，四任务各域12段；robot侧每任务至少覆盖 active support、fixture MONITOR、same-effector/compound or capability-removal、infeasible/ambiguous 四类。另有至少 `240` 条双标 allocation decisions。

Gate：

```text
candidate recall >= 0.95
exact bundle accuracy >= 0.85
infeasible false accept <= 0.05
ambiguous/unknown abstention F1 >= 0.80
fixture switch consistency >= 0.90
capability counterfactual consistency >= 0.90
hidden-channel probes: 100% pass
```

失败：不进入 WAM；保留 failure taxonomy和 capability registry。

### G3：supporting-compiler 下游 gate

在具备真实 calibration/profile后，先用 oracle allocation检验 r1d/WAM sufficiency。predicted FRG必须：

- 相对 literal 1:1/fixed-role strongest baseline：四任务总体 success `>=+10 pp`；
- 相对 robot-only fixed assignment：至少三任务为正，cluster bootstrap 95% CI总体不跨0；
- 恢复 oracle allocation增益至少70%；
- safety、infeasible execution不退化；
- equal-bit control显著更差，且 packet利用率不是只靠assignment token的task shortcut。

失败：FRG不进主方法；M0仍可作部署validator/benchmark compiler。

### G4：独立论文晋级 gate

缺一不可：

1. 至少 **3 个真实 target morphology/profile**，其中必须有一个异构双末端；不能只用同一硬件的mask冒充三种morphology；
2. 每个 morphology至少两任务存在不同但功能等价的真实 allocation；
3. 四任务 leave-one-morphology-out，held-out profile上 exact allocation与下游success均优于强基线；
4. M+ 相对 M0/ILP exact allocation `>=+5 pp`，下游 success `>=+5 pp`，95% cluster CI支持正优势；
5. 相对 PA-BiCoop-style、Human2Any-style 与 unstructured allocator至少两个关键轴有优势，而非只胜 1:1；
6. fixture/monitor 与 actor-cardinality challenge的增益独立成立；
7. 所有 side-channel、profile-swap、equal-bit probes通过。

只有一个 RH56+Omni profile、只有 offline packet accuracy、或只胜 literal retarget时，均不得独立晋级。

### 直接 stop rules

- `minimal-r1d` 两constraint上限导致四任务 positive overflow `>5%`：停止当前schema，不让allocator丢support；
- gold allocation中 `AMBIGUOUS` `>20%`：capability/profile证据不足，先补公开状态而非训练scorer；
- M+与M0差 `<3 pp`：删除 learned allocator；
- raw visual allocator才有效：停止 finite role allocation claim；
- action需要 allocator hidden/logit：停止 strict WAM claim；
- same role bundle内动态可持续性 collision `>10%`：该task出scope或显式升schema；
- held-out morphology不优于 fixed/latent强基线：停止 standalone morphology claim；
- 只有 task name/profile ID决定 assignment：判 shortcut，重新split或停止。

## 14. 最终研究定位

### 14.1 当前应怎样写进 SM-FET

把 Idea M 定位为 `minimal-r1d` 的 **requirement grounding compiler**：

```text
human training supervision
  -> domain-neutral shared requirements
  -> target capability/resource grounding
  -> explicit O/S/M + assignment packet
  -> strict WAM
```

它解决的是现有 r1d 已经提出但尚未闭合的 binding 问题：shared key用 `EFFECT_GOAL/SUPPORT_REQUIREMENT/OBSERVATION_REQUIREMENT`，robot packet用 O/S/M；中间必须有一条可审计、可abstain、不会读取human side的 grounding路径。

### 14.2 当前不应怎样写

不要把 FRG 写成：

- 新的 cross-embodiment latent skill；
- 新的 many-to-many correspondence理论；
- 新的 bimanual role taxonomy；
- 新的 general task-and-motion planner；
- 从 human video直接输出 robot trajectory 的 retargeter；
- 通过 learned allocator score“自动处理”未标定profile。

这些表述分别与 XSkill/Affordance Equivalence、Alissandrakis、BUDS/PA-BiCoop、Human2Any/TAMP、DexMachina/SoftAct、现有 capability contract冲突。

### 14.3 一句话裁决

**Morphology-Aware Functional Role Allocation 的科学价值不在“发现左右手不能一一映射”，而在把这个常识变成一个有限、可拒绝、可做 capability counterfactual、且不会给 WAM 偷送 latent 的 compiler。** 当前单一 RH56+Omni scope下，它应作为 SM-FET `minimal-r1d` 的 supporting compiler与四任务 benchmark；只有跨至少三种真实 morphology、显著胜过 ILP/BUDS/PA-BiCoop/Human2Any-style强基线并通过 hidden-channel审计后，才有资格独立成篇。

## 15. 主要原始来源与本地依赖

### 外部原始来源

- [Correspondence Mapping Induced State and Action Metrics for Robotic Imitation](https://doi.org/10.1109/TSMCB.2006.886947)
- [A Bimanual Manipulation Taxonomy](https://doi.org/10.1109/LRA.2022.3196158)
- [Stabilize to Act: Learning to Coordinate for Bimanual Manipulation](https://arxiv.org/abs/2309.01087)
- [PA-BiCoop: A Primary-Auxiliary Cooperative Framework for General Bimanual Manipulation](https://arxiv.org/abs/2606.28192)
- [XSkill: Cross Embodiment Skill Discovery](https://proceedings.mlr.press/v229/xu23a.html)
- [Cross-Embodied Affordance Transfer through Learning Affordance Equivalences](https://arxiv.org/abs/2404.15648)
- [MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence](https://arxiv.org/abs/2508.13534)
- [ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection](https://robin-lab.cs.utexas.edu/ScrewMimic/)
- [DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation](https://arxiv.org/abs/2505.24853)
- [Functional Force-Aware Retargeting from Virtual Human Demos to Soft Robot Policies](https://arxiv.org/abs/2604.01224)
- [DexMani: Human-Derived Manipulability Guidance for Dexterous Rotation](https://arxiv.org/abs/2608.00554)
- [Human2Any: Human-to-Robot Transfer via Constraint-Aware Compositional Planning](https://arxiv.org/abs/2606.28813)
- [Cooperative Task and Motion Planning for Multi-Arm Assembly Systems](https://arxiv.org/abs/2203.02475)
- [Task-driven Perception and Manipulation for Constrained Placement of Unknown Objects](https://arxiv.org/abs/2006.15503)
- [Learning Symbolic and Subsymbolic Temporal Task Constraints from Bimanual Human Demonstrations](https://arxiv.org/abs/2403.16953)

### 本地只读依赖

- `docs/research/g1_rh56_omnipicker_sm_fet_capability_profile_20260812.md`：O/S/M、working surfaces、resource conflicts、same-effector与fixture/monitor边界；
- `docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`：32-byte/5-token schema、shared purpose key、profile/runtime分层与96.991-bit结构容量；
- `docs/research/sm_fet_wam_v2_final_architecture_decision_20260812.md`：planner/compiler/WAM边界、48预测/8执行、训练期human信息不可直达action；
- `docs/research/idea_k_effect_equivalent_interaction_quotient_review_20260812.md`：Affordance Equivalence、MimicFunc、DexMani与effect grouping边界；
- `docs/research/human_robot_tool_use_pair_dataset_survey_20260812.md`：公开数据不能替代目标36+36/48+48 paired-world pilot的证据。
