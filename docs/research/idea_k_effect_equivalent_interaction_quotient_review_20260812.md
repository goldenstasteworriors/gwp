# Idea K 独立调研：Effect-Equivalent Interaction Quotients

- 日期：2026-08-12
- 审查基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 隔离分支：`codex/idea-k-effect-equivalence-20260812`
- 范围：只做 Zotero-first 文献核查、结构设计与数据协议；未写模型代码、未训练、未使用 GPU
- `idea_id`：`2823cc3b-c066-4e44-ac77-b445f260a683`
- `discussion_id`：`4749b58a-5b7c-4ff9-ab5b-87c30ea6b95d`

## 0. 结论先行

**“Effect-Equivalent”作为独立算法/标题级 novelty 是 No-Go；作为 SM-FET-WAM 的训练期分组监督与严格对照，Conditional Go。**

原因不是 idea 没价值，而是最直白的表述已经被直接占据：Aktas 等人的 **Affordance Equivalence** 已明确把 agent、object、action、effect 放入共享表示，并把“不同 agent 以不同 action 造成相同 effect”形式化为 cross-embodiment equivalence；MIR 已主张忽略 manipulator、关注 environment change；FUNCTO/MimicFunc 已在工具使用中以 functional keypoint/function frame 做功能等价迁移；RHyME 已处理 mismatched execution 与 sequence-level OT；XSkill/UniSkill 已学习 embodiment-agnostic skill/prototype。因此不能声称首次提出“动作不同、效果相同即可配对”。

仍可守住的窄问题是：

> 在不把 human video、dense trajectory、matcher hidden state 或 quotient ID 送入动作侧的前提下，能否把人工可审计的 `pre-relation → 1–3个operation events → post-effect` 投影用于离线构造 cross-embodiment 多正例组，并将其蒸馏为 SM-FET 的有限硬字段；相较完整轨迹 DTW、无配对共训、post-only 分组和 CAPOT-only，这种训练期 quotient supervision 是否能提高对无关准备段、不同操作节奏、不同手形以及“动作相似但效果错误”hard negative 的鲁棒性？

推荐名称不要继续使用宽泛的 **Effect-Equivalent Interaction Quotients** 作为论文主标题。更安全的模块名是 **RE-Q（Relational Event Quotient supervision）**，并明确：

1. quotient 是**标注/编译出来的训练期 projection**，不是新的 OT/DTW；
2. CAPOT 仍是 event pair table builder，RE-Q 只做 candidate gate、multi-positive group 与 counterfactual negative；
3. 部署时 action 只看到现有 `event_tube_v2` hard packet，视觉容量增加 **0 bit**；
4. 22D contact packet仍是 legacy baseline，v2 与 22D 不拼接；
5. 若增益只来自 post-state、更多标注或额外容量，则停止这条路线。

## 1. Zotero-first 结果与检索质量

idea-discussion 已创建并更新 idea card。自动 Zotero 检索返回的前三项是 RoboBrain 2.5、RT-1、RoboBrain 2.0，主题命中较弱；因此不能把自动摘要当 novelty 证据。随后直接核查本地 826 条 Zotero 条目，找到以下真正相关的个人库条目：

| Zotero key | 条目 | 与本 idea 的关系 |
|---|---|---|
| `JEZ26KWY` | RHyME / One-Shot Imitation under Mismatched Execution | sequence-level OT、片段检索组合、弱结构 human video；直接冲突“无需全程一致” |
| `8WMVVJAL` | ImMimic | DTW/GMS-SDTW human–robot trajectory mapping 与 MixUp；直接冲突 pairing 算法 claim |
| `A89APIR3` | EgoBridge | joint latent-action OT；直接冲突 OT + cross-domain co-training claim |
| `EK2CTN7J` | UniSkill | 无标签 embodiment-agnostic skill representation；冲突抽象 skill claim |
| `V7LLF6VJ` | XSkill | skill prototype discovery、cross-embodiment prompt following；冲突 prototype/quotient claim |
| `WQJXZYHX` | Time-Contrastive Networks | 关注手/夹爪与环境关系、跨视角时序差异；早期 representation baseline |
| `VUYF5NYC` | HumanEgo | entity-level hand–object interaction、robot-data-free transfer；冲突 entity abstraction claim |
| `CSMS2VV2` | TopoRetarget | 保留 task-relevant hand–object contact structure；冲突 contact-topology claim |
| `3SVRDJ39` | HARP-VLA | 少量 paired human/robot bridge + 大量 unpaired video；数据协议与 representation alignment baseline |
| `78FMEV92` | D(R,O) Grasp | robot–object interaction unified representation；但主要是静态 grasp |
| `ZJQINYLA` | VIDEOMANIP | 3D hand/object trajectory reconstruction + contact optimization + retargeting |
| `KCNBA8FP` | Do As I Do | 从日常 RGB video 重建并逐轨迹 retarget；强 direct-retarget baseline |
| `95PMFGL4` | LIDEA | latent distillation + explicit geometry alignment；冲突 shared representation claim |

Zotero 未命中但必须补入的最关键原始前例，是 [Cross-Embodied Affordance Transfer through Learning Affordance Equivalences](https://arxiv.org/abs/2404.15648)。该论文不是“邻近相关”，而是名称、形式化和核心直觉都直接相撞。

## 2. 前例边界：真正撞在哪里

### 2.1 最强冲突：Affordance Equivalence

[Aktas et al., 2024](https://arxiv.org/html/2404.15648v2) 将 affordance 写成：

```text
(effect, (agent, (object, action)))
```

并显式定义：同一 object 上不同 agent 的不同 action，只要产生同一 effect，就形成 action/agent equivalence。模型用多通道 Conditional Neural Movement Primitives，把 robot action trajectory、object depth image 与 effect trajectory 编入共享 latent，再从缺失 channel 生成其他 channel；真实机器人实验甚至直接从示范者造成的 final object position 生成机器人 action trajectory。

因此以下表述全部不能 claim：

- 首次以 effect 而非动作外观建立 cross-embodiment correspondence；
- 首次定义 action/object/effect equivalence class；
- 首次由相同 effect 推出不同机器人 action；
- 首次用 causal/environment effect 做跨具身模仿。

可区分之处只有工程与证据契约：它用 robot self-exploration/robot-to-robot data、连续 action/effect trajectory 与自由 latent；RE-Q 面向被动 human video + target robot weak pairs，只在 1–3 个已审计 event bundle 上形成有限 relational code，并用 strict route 保证 action 看不到 latent/pairing。这个差别足以支持一个**模块或评测协议**，不足以单独支持“Effect Equivalence”大标题。

### 2.2 工具功能对应：FUNCTO 与 MimicFunc

[FUNCTO](https://arxiv.org/html/2502.11744v2) 已用 function point、grasp point、tool center 三个 3D functional keypoints，提取 human tool trajectory、建立不同工具间 function-centric correspondence 并规划机器人动作；覆盖 pouring、cutting、hammering、scooping、scrubbing。

[MimicFunc](https://arxiv.org/html/2508.13534v1) 进一步构造 function frame，重点对齐 interaction keyframe，同时允许其余运动按任务上下文优化。尤其值得注意的是，其实现已在 approach 前 30% 放松 trajectory cost；所以“工具使用不应关注全程”本身也不是空白。

RE-Q 与它们的可守差别：

- FUNCTO/MimicFunc 是**同一工具功能的显式 3D 几何迁移/动作生成**；
- RE-Q 是**不同具身片段的有限关系—效果分组监督**，不生成 SE(3) trajectory；
- RE-Q 必须允许 human 与 robot 使用不同 operation motif，只要求 audited pre/effect projection 相同；
- RE-Q 的主要评价应是 pair retrieval、counterfactual rejection 与下游 WAM 的有限 route，而非一镜到底 one-shot planning。

### 2.3 环境变化、对象轨迹与 mismatch execution

- [Manipulator-Independent Representations](https://arxiv.org/abs/2103.09016) 已明确以 environment change 为核心，学习跨 domain、temporal-smooth、actionable representation。
- [RHyME](https://openreview.net/forum?id=mCcyU1l9xP) 已用 sequence-level OT，从 unpaired human clips 检索并组合与 robot execution 语义等价的片段，处理 movement style/capability mismatch。
- [X-Sim](https://openreview.net/forum?id=BO7qo66YJ2) 以 object trajectory 作为跨具身 dense reward。
- [Im2Flow2Act](https://openreview.net/forum?id=cNI0ZkK1yC) 用 object flow 作为 human/sim/robot 的共同 manipulation interface。
- [XSkill](https://proceedings.mlr.press/v229/xu23a.html) 与 [UniSkill](https://openreview.net/forum?id=L79oJ0aemZ) 分别学习 skill prototypes 与无标签 cross-embodiment skill representation。

所以 RE-Q 不能再主张“object-centric”“sequence-level”“skill equivalence”之一是 novelty。它唯一可检验的组合增量是：**finite audited pre/post relation projection + bounded event bundling + strict WAM route + counterfactual effect controls**。

### 2.4 CAPOT、EgoBridge、ImMimic、GeCCo、DexMani 的边界

| 方法 | 已占据空间 | RE-Q 必须避开的重合 | 允许保留的角色 |
|---|---|---|---|
| CAPOT（本项目 Idea G） | event-aware partial/unbalanced OT、many-to-many、NULL、monotonic projection | RE-Q 不能声称自己是新 matcher | CAPOT 在 q-compatible candidate graph 内分配局部 coupling；q 仅作 gate/group |
| EgoBridge | human/robot joint latent-action OT | 不声称首次用 OT 对齐 human/robot | balanced joint OT baseline |
| ImMimic | DTW/GMS-SDTW + mapping/interpolation | 不声称首次忽略多余前后段或 many-to-many | full/subsequence trajectory baseline |
| GeCCo | high-level contact plan 驱动可复用 low-level quadruped policy | 不声称首次 contact-conditioned modular policy | 只在讨论“planner/control 解耦”时引用；任务域并不直接重合 |
| DexMani | 从 human contact transitions 学 contact-conditioned manipulability evolution，跨手形指导 rotation RL | 不声称首次把连续 contact transition 抽象为跨具身先验 | dynamic-hand/contact prior baseline；RE-Q 不编码 manipulability |
| TopoRetarget | interaction graph、contact precision、跨 hand retarget | 不声称首次保留 interaction topology | dense/topological retarget baseline |

特别地，[DexMani](https://arxiv.org/abs/2608.00554) 比 RE-Q 更保留“怎样操作”的连续信息：它显式建模 contact transition 如何改变可持续旋转方向。RE-Q 若只看粗 post-effect，可能把可持续性完全不同的两个过程错误合并；这应成为 quotient-collision stress test，而非回避的案例。

## 3. 精确定义：先观察 operation，再把它商掉

### 3.1 interaction member 与 quotient projection

每个 human 或 robot 候选不是整段 episode，而是一个最多含 3 个相邻 SM-FET macro events 的 bundle：

```text
x = (domain, embodiment, episode_id, [e_s ... e_t], m(x), q(x))
1 <= t-s+1 <= 3
```

其中 `m(x)` 保留该具身实际怎样完成操作；`q(x)` 只保留前置关系与观察到的对象效果：

```text
m(x) = operation motif + contact modes + local durations + actor-side geometry
q(x) = role/part identities + relation_before + relation_after
       + object_effect + effect_axis + effect_magnitude + outcome
```

定义 projection `pi(x)=q(x)`，仅对字段全部经审定、没有关键 `unknown` 的 bundle 定义严格等价：

```text
x ~ y  iff  pi(x) == pi(y)
```

这保证反身、对称、传递。**不能**把“相似度超过阈值”称为 quotient equivalence，因为 learned pairwise threshold 通常不具传递性。unknown/ambiguous bundle 留在候选图中，但不得通过传递闭包被并入等价类。

此定义刻意先记录 operation，再通过 `pi` 商掉 embodiment-specific mechanism。因此：

- human 倾杯的一段连续腕旋转，可与 robot 的 `grasp_stabilize → incremental_tilt → return` 三事件 bundle 同类；
- human 单次掌推，可与 robot 的多次短推 bundle 同类；
- 同样的腕旋转但液体未转移、溢出或目标容器错误，不能同类；
- 同样的最终位置但 pre-relation 不同、工具/目标 role 反转，也不能同类。

### 3.2 `effect_quotient_v1` 有限 key

标准候选完全复用/扩展 `event_tube_v2` 的有限词表，不引入自由文本或 learned code：

| 字段 | 基数 | 枚举/来源 |
|---|---:|---|
| `primary_role` | 4 | `manipulated/tool/receptacle/fixture` |
| `secondary_role` | 4 | 同上 |
| `primary_part` | 8 | `body/handle/lid_cap/opening/rim_edge/trigger_button/tool_tip/other` |
| `secondary_part` | 8 | 同上 |
| `relation_before` | 9 | `separate/near/touching/overlap_2d/inside/contains/aligned_to_rim/supported_by/unknown` |
| `relation_after` | 9 | 同上 |
| `object_effect` | 8 | `none/translate/rotate/articulate/flow/deform/transfer/unknown` |
| `effect_axis` | 8 | object-local `+u/-u/+v/-v/+w/-w/scalar_or_NA/unknown` |
| `effect_magnitude` | 4 | `none_or_below_threshold/low/mid/high`，阈值只在 calibration split 冻结 |
| `outcome` | 4 | `success/partial/failure/unknown` |

按 factorized codec 机械允许全部组合时：

```text
S_q = 4*4*8*8*9*9*8*8*4*4
    = 84,934,656
log2(S_q) = 26.339850002885 bits
```

这只是 **training-side quotient-key alphabet**，不是 action-visible capacity，也不是互信息。严格 positive class 还要求关键字段不为 `unknown`；因此实际可用类远少于该上界。

`effect_axis` 必须在 object-local/function-frame 坐标中离散，不能用相机 left/right/up/down，否则会把视角当功能。对 `flow/deform` 等没有单轴语义的 effect 使用 `scalar_or_NA`。

### 3.3 operation member descriptor

每个 bundle 同时保存最多 3 个操作码，按事件顺序排列：

```text
operation_primitive_v1 =
  acquire, stabilize, press, pull, push_sweep, rotate,
  impact, slide_rub, pour_tilt, scoop, release_place, verify
```

合法 operation motif 是长度 1–3 的 prefix sequence，理论状态数：

```text
S_m = 12 + 12^2 + 12^3 = 1,884
log2(S_m) = 10.879583249613 bits
```

`m(x)` 不参与标准 quotient equality，但用于三件事：

1. 报告一个 q-class 中到底包含多少不同 human/robot mechanisms；
2. 做 safety/feasibility gate，避免“效果相同但目标 embodiment 无法安全执行”的错误迁移；
3. 构造 hard positive：`q相同且m不同`；以及 hard negative：`m相似但q不同`。

若 pilot 显示 q 太粗，可启用 **strict variant**，将 `operation_function` 加入 key；但这会削弱“不同动作只要效果相同即可配对”的主假设，只能作为保守后备。

### 3.4 bundle composition

一个 bundle 的 quotient 字段机械编译如下：

```text
relation_before = first event's audited relation_before
relation_after  = last event's audited relation_after
object_effect   = composed object delta over bundle
effect_axis/magnitude = delta from first stable pre-window to last stable post-window
outcome         = last verified outcome
roles/parts     = canonical functional role and interaction part, not track ID
```

允许组合的内部 event 数固定 `<=3`。跨过另一个已审定 task-relevant effect、角色交换或失败恢复边界时禁止组合。这样 many-to-one 来自有限 bundle，而不是把整段 video 任意压成一个终点。

## 4. 配对：一对多、多对一与 NULL

### 4.1 不是先做 episode 一一配对

流程应是：

```text
episode -> event segmentation -> enumerate length-1..3 bundles
        -> compile q/m -> q-compatible candidate hypergraph
        -> CAPOT local coupling -> reviewed pair table
```

同一 q-class 是一个 bipartite hypergraph：human side 可有多个 bundles，robot side也可有多个 bundles。训练采样以 class 为单位，不固定 human_i 对 robot_i。

### 4.2 多正例与 group supervision

对 q-class `c`，令 `H_c` 和 `R_c` 分别为 human/robot members：

```text
positive:      any reviewed (h in H_c, r in R_c)
hard positive: q(h)=q(r), but m(h)!=m(r) or bundle_len differs
hard negative: m(h)≈m(r), but relation_after/effect/outcome differs
easy negative: task family and q both differ
```

训练时不强制每个 human member 选择唯一 robot partner；同一个 robot bundle可作为多个 human 的 positive，反之亦然。batch 应按 source episode/group split 采样，避免同一源片段的切窗泄漏。

### 4.3 NULL 与 unknown 不能混写

- `NULL-background`：准备、走动、等待、相机调整、归零；没有 task-relevant effect。
- `NULL-domain-only`：只在一侧需要的 regrasp、安全等待、机械回中。
- `NULL-failed-candidate`：发生操作但 q 不满足配对条件，例如倾斜无转移。
- `unknown`：视觉/传感证据不足，不能确定 effect；loss mask，不是 NULL 或 negative。
- `unmatched`：CAPOT 最终未运输的质量；只存在于 pair table，不是 action command。

NULL 不建 quotient class，不参与 transitive closure；否则大量 background 会因相同 `none` effect 被错误聚成巨类。

## 5. 精确模型结构：RE-Q 只做 training-side auxiliary

### 5.1 三条信息流

```text
human video  -> shared/event teacher -> human event bundle -> q label/prototype
robot video  -> SM-FET planner       -> robot event bundle -> q prediction
robot state + hard event_tube_v2 packet -> WAM action expert -> 48-step action chunk
```

训练目标候选：

```text
L_total = L_WAM
        + lambda_fet  * L_event_tube_v2
        + lambda_q    * L_quotient_field_prediction
        + lambda_mp   * L_multi_positive_quotient_contrast
        + lambda_cf   * L_counterfactual_effect_rejection
```

其中：

- `L_quotient_field_prediction` 是 factorized finite-field CE，不预测自由 quotient embedding；
- `L_multi_positive_quotient_contrast` 只在 planner/event representation 上，使同 q、不同 m 的 human/robot bundle接近；
- `L_counterfactual_effect_rejection` 强制“相似 motion、错误 effect/outcome”分开；
- pair edge 由离线 CAPOT 表提供并 stop-gradient；
- action expert不能读取 human encoder hidden、q prototype、coupling matrix、pair ID 或 source domain。

### 5.2 action 可见什么

**标准方案不新增任何 action 字段。** action 继续只读 meta-synthesis 中的：

```text
node_packet  [4,42]
event_packet [3,51]
edge_packet  [3,2,84]
```

也就是 active + next two events 的 hard `event_tube_v2`；按当前 meta-synthesis 候选定义，其 codec-exact保守上界为 `281.932700038062 bits`，最终以独立 schema 审计结果为准。RE-Q 所需的 role/part、relation_before/after、object_effect、outcome 大部分已在 v2 中；`effect_axis/magnitude` 只做训练期监督，第一版不进 action。

因此相对 SM-FET-WAM：

```text
additional action-visible bits = 0
additional runtime human input = 0
additional runtime OT/DTW cost = 0
```

若实现者把 q-ID 或 learned quotient token送到 action，实验问题就变成“额外 latent side channel是否有用”，不再是本设计。

### 5.3 CAPOT 与 RE-Q 的分工

```text
RE-Q:  q-compatible candidate gate + class-balanced sampling
CAPOT: class内 event/bundle cost、partial mass、NULL、monotonic projection
SM-FET:部署时预测有限 event tube
WAM:   只从 hard tube + robot inputs 生成 action chunk
```

CAPOT cost 中 `d_relation_effect_outcome` 可由 q-field Hamming/ordinal distance实现，但 q equality 不能替代 CAPOT：同一 q-class仍可能有重复 effect、不同顺序、遮挡和局部多对多。

## 6. Baseline 与消融

最小评测矩阵按重要性排序：

1. **Robot-only SM-FET-WAM**：没有 human data，测 human supervision是否真的有用。
2. **Unpaired co-training**：human/robot 共享 event encoder，但不建 q 或 pair；对应 EgoMimic/UniSkill 类强基线。
3. **Full/subsequence trajectory alignment**：ImMimic DTW/GMS-SDTW，包含连续 wrist/object trajectory。
4. **CAPOT-only**：相同 event schema、标签量和优化预算，但不使用 q gate/multi-positive class。
5. **Post-only grouping**：只按 final effect/outcome分组，检验 pre-relation 与 role/part 是否必要。
6. **RE-Q standard**：本方案；q gate + CAPOT + multi-positive/counterfactual losses。
7. **RE-Q strict-operation**：把 operation function加入 key，检验“商掉 mechanism”是否真有益。
8. **Oracle q upper bound**：测试时使用人工 q 仅用于诊断，不能与部署结果混报。

公平性要求：1–7 action-visible packet、action model参数、训练步数、human/robot采样比、field labels与 split 完全相同；RE-Q 不得靠更多视觉 bit 或更多 action input获胜。

## 7. 可证伪预测

### P1：无关前后段鲁棒性

在 human clips 前后添加 2–5 秒准备、等待、视角调整，或插入一侧独有 regrasp：RE-Q pair retrieval AUPRC/F1 相对原 clip下降应 `<=5 pp`，并显著优于 full DTW。若 CAPOT-only同样稳定且与 RE-Q 差异 `<3 pp`，删除 quotient 模块。

### P2：相同 effect、不同 mechanism

构造 hard positives：human 单段 tilt vs robot 三段 stabilize/tilt/return，或 human 连续 sweep vs robot多次短推。RE-Q 对这些 pair 的召回应高于 trajectory/pose alignment，同时不要求 wrist/hand相似。若 strict-operation 反而更好，说明“商掉 operation”假设失败。

### P3：相似动作、错误效果

构造 hard negatives：相同倾斜但无转移/溢出；相同推/拉轨迹但目标 role错误；接触正确但 post relation错误。RE-Q false-positive rate应比 post-blind/CAPOT contact-only低至少 `10 pp`。若不能，q schema没有捕获因果效果。

### P4：pre-relation 必须有用

删除 `relation_before` 后，兼容性错误应上升；若 retrieval与下游结果不变，应采用更简单的 post-effect方法，不保留“pre→operation→post”claim。

### P5：收益必须穿过有限 route

human supervision带来的下游增益必须在部署时零 human input、零 q token、同 `event_tube_v2` hard packet下保留。若只有 learned hidden或 continuous effect trajectory action-visible 时有效，RE-Q/finite route主张失败。

### P6：稀疏 quotient 不能掩盖动态可持续性

在 DexMani 风格连续旋转/重复接触任务上，同一 coarse q 内可能存在“当前效果相同但后续 manipulability不同”的成员。若 quotient collision rate超过阈值或执行明显退化，应把该任务标为 schema out-of-scope，而不是继续合并。

## 8. 最小 `36 human + 36 robot` 数据协议

### 8.1 三个 task family

每个域每类 12 段，共 36+36；任务需覆盖不同 effect，而不能只有同一抓取：

1. **pour/transfer**：容器→容器，主要 effect 为 `flow/transfer`；改变抓法、倾倒节奏和容器形态。
2. **press/lever/articulate**：工具或末端驱动开关/旋钮/杠杆，主要 effect 为 `articulate/rotate`；human 与 robot允许不同接触部位和 operation motif。
3. **sweep/push-to-receptacle**：刷/刮/推使多个目标进入 receptacle，主要 effect 为 `translate/transfer`；允许一次连续 sweep 对多次 robot micro-push。

每类每域 12 段的组成固定：

```text
8  successful effect members
2  hard-negative executions（动作/接触相似但effect错误或partial）
2  NULL-heavy clips（准备、等待、一侧独有regrasp/recovery）
```

机器人执行必须记录真实 target embodiment/end-effector、operator、session/day、tool/target instance、相机、frame rate与 outcome；不能把旧 dual-Inspire grab 或现有 human pour直接冒充 36+36 positive overlap。

### 8.2 split

先按 operator/session/tool instance 切分，再建 bundle/pair edge：

```text
calibration: 3 human + 3 robot（每类各1，专门冻结阈值，不进最终指标）
train:      21 human + 21 robot（每类各7）
validation: 6 human + 6 robot（每类各2）
blind test: 6 human + 6 robot（每类各2）
```

同一 source clip 的裁剪、bundle、增强、pair edge必须在同一 split。test 的 tool/target instance和采集 session不得出现在 train。

### 8.3 标注与 pair gold

两位 annotator 独立完成：event boundary、role/part、relation before/after、effect kind/axis/magnitude、outcome、operation motif、visibility/confidence；第三人只裁决冲突。

最少人工 pair gold：

```text
36 hard positives：每类12，q相同但m/长度/节奏不同
36 hard negatives：每类12，m相似但q/effect/outcome不同
24 NULL decisions：准备/等待/domain-only/recovery各类均覆盖
```

其余候选可由 RE-Q + CAPOT 自动生成，但不能称为人工 GT。报告 exact q agreement、field-wise kappa、boundary tolerance、bundle-length distribution、q-class human/robot coverage与 quotient collision。

## 9. Gate 与停止条件

### G0：可观测性与标注 gate

以下任一成立立即停止，不实现模型：

- `relation_after/object_effect/outcome` 的关键字段 unknown 比例 `>15%`；
- q exact agreement `<75%`，或关键字段 Cohen's kappa `<0.70`；
- 正 pair 中 `>10%` 需要长度大于 3 的 bundle才能表达；
- 角色/关系/effect词表 overflow `>5%`；
- 少于 6 个 q-class 同时拥有 `>=3 human` 与 `>=3 robot` successful members。

### G1：quotient validity gate

- 同一 q-class 中，被盲审判为功能不等价/不安全的 collision rate必须 `<10%`；
- q-compatible candidate graph不得由一个 giant class占据 `>25%` 全部 successful bundles；
- `m不同的hard-positive` 必须至少占 positive gold 的一半，否则数据没有检验用户核心假设。

### G2：离线 pairing gate

- RE-Q 相对 CAPOT-only 的 blind hard-positive AUPRC/F1 增益至少 `5 pp`；
- 相似动作/错误效果 hard-negative FPR 至少下降 `10 pp`；
- prepend/append/NULL perturbation 后性能下降 `<=5 pp`；
- 若 full/subsequence DTW或post-only baseline持平，则停止 RE-Q，不进入 WAM。

### G3：未来下游 gate

在之后真正具备数据并训练时才启用：

- 相同 action-visible packet 下，RE-Q相对 strongest non-quotient baseline 的 held-out task success增益至少 `5 pp`，bootstrap 95% CI 不跨 0；
- 增益必须在三类任务至少两类成立，不能只来自 pour；
- q-ID/learned latent/continuous trajectory不进入 action；
- 若 equal-bit/unstructured control 或增加相同数量普通 event labels即可复现增益，删除 quotient claim。

## 10. 最终研究判断

| 判断项 | 结论 |
|---|---|
| 用户直觉是否合理 | 是：工具学习不应要求准备阶段、手形、节奏和完整轨迹同构 |
| “相同效果的不同动作可等价”是否新 | 否；Affordance Equivalence 已直接形式化并验证 |
| “只关注关键功能状态”是否新 | 否；MimicFunc、FUNCTO、RHyME、object-flow/trajectory方法已覆盖大量空间 |
| 是否值得保留 | 值得，作为 finite audited training supervision 与反事实评测 |
| 是否新增部署输入/容量 | 不新增；0 bit，action仍只读 `event_tube_v2` |
| 推荐状态 | **Conditional Go as auxiliary / No-Go as standalone headline** |

最值得做的不是再发明一个“等价 latent”，而是证明一个更严格、也更容易被推翻的事实：**在 action 侧看不到 human video 与 learned pair code时，按有限 pre/post effect quotient组织的多正例监督，是否仍比 CAPOT-only、unpaired co-training和完整轨迹对齐更好。** 如果不能，就应保留 SM-FET/CAPOT，删除 RE-Q。

## 11. 主要原始来源

- [Cross-Embodied Affordance Transfer through Learning Affordance Equivalences](https://arxiv.org/abs/2404.15648)
- [FUNCTO: Function-Centric One-Shot Imitation Learning for Tool Manipulation](https://arxiv.org/abs/2502.11744)
- [MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence](https://arxiv.org/abs/2508.13534)
- [Manipulator-Independent Representations for Visual Imitation](https://arxiv.org/abs/2103.09016)
- [One-Shot Imitation under Mismatched Execution / RHyME](https://openreview.net/forum?id=mCcyU1l9xP)
- [ImMimic](https://proceedings.mlr.press/v305/liu25b.html)
- [EgoBridge](https://openreview.net/forum?id=Hr6EzaaKQv)
- [XSkill](https://proceedings.mlr.press/v229/xu23a.html)
- [UniSkill](https://openreview.net/forum?id=L79oJ0aemZ)
- [X-Sim](https://openreview.net/forum?id=BO7qo66YJ2)
- [Flow as the Cross-Domain Manipulation Interface](https://openreview.net/forum?id=cNI0ZkK1yC)
- [TopoRetarget](https://arxiv.org/abs/2606.16272)
- [DexMani](https://arxiv.org/abs/2608.00554)
- [GeCCo](https://arxiv.org/abs/2509.17582)
- [HumanEgo](https://arxiv.org/abs/2605.24934)
