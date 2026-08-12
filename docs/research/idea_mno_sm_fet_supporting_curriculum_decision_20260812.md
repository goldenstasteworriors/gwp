# Idea M/N/O 终审：SM-FET `minimal-r1d` 唯一 supporting curriculum

- 日期：2026-08-12
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 隔离分支：`codex/idea-mno-sm-fet-supporting-curriculum-20260812`
- 审查对象：Idea M（asymmetric functional role allocation）、Idea N（ego critical-event predictive pretraining）、Idea O（privileged hand-contact trajectory distillation）
- 审查范围：只做合同、训练课程、数据、因果实验与 novelty 终审；不实现、不训练、不使用 GPU、不连接服务器或真机
- `idea-discussion`：`idea_id=7e10b827-ccc6-4a24-8bfb-88772cc7a3ec`，`discussion_id=f59d98e7-b60d-4a74-8389-80871b56851d`，`memory_updated=true`

## 0. 唯一裁决

最终只保留一条主训练路线：

```text
human causal next finite-event pretraining（Idea N）
  -> 只迁移 planner-owned weights
  -> robot-only event/geometry adaptation
  -> deterministic M0 r1d compiler/validator（Idea M 的必要部分）
  -> canonical active-only 32-byte / 5-token minimal-r1d packet
  -> frozen robot-only action expert
```

裁决分四句：

1. **M 的功能必须存在，但不是 learned FRG。** 必须保留无参数、可拒绝的 M0 compiler/validator，把 domain-neutral shared requirements 与 robot-only geometry、静态 capability/profile 和当前 receipts 编译成唯一 executable packet；M+ learned scorer 当前删除。
2. **N 与 O 不完全相同，但占据同一因果槽位。** N 用 unpaired human past 预测 next finite event，O 用 paired privileged human trajectory 蒸馏同一个 robot planner；二者最终都只是在训练期改变 planner，因此不能作为两个默认模块累加。
3. **主线择 N，不合并 O。** O 降为互斥 challenger/诊断臂，不进入默认 curriculum，也不列为论文贡献；只有预注册的 `N×O` 因子实验显示 O 在 N 之上仍有独立增益，才重新审查。
4. **当前停在 Stage 0。** `minimal-r1d` 书面合同已经是 Doc structural Freeze，但 CPU conformance 仍 PENDING，真实 G1+RH56+OmniPicker executable profile 仍 NO-GO，`|G_profile|` 与 profile bits 仍 UNDEFINED；所以现在不批准模型实现或训练。

这不是 M/N/O 的机械堆叠。主方法只有一个 human training signal（N）、一个确定性边界（M0）和一个 robot-only action learner；O、M+、JEPA、dense pose、CAPOT latent、learned motif 与 action-visible future event都不在默认路线。

## 1. 规范优先级与版本消歧

### 1.1 本轮依赖

三份独立 memo：

- `/home/ykj/project/WM/gwp-agent-idea-m-asym-role-allocation/docs/research/idea_m_asymmetric_functional_role_allocation_review_20260812.md`
- `/home/ykj/project/WM/gwp-agent-idea-n-ego-event-pretrain/docs/research/idea_n_ego_critical_event_predictive_pretraining_review_20260812.md`
- `/home/ykj/project/WM/gwp-agent-idea-o-privileged-distill/docs/research/idea_o_privileged_hand_contact_distillation_review_20260812.md`

上位组合与 novelty 审查：

- `/home/ykj/project/WM/gwp-agent-final-idea-portfolio/docs/research/wam_biochemical_manipulation_idea_portfolio_20260812.md`
- `/home/ykj/project/WM/gwp-agent-wam-novelty-review/docs/research/sm_fet_wam_novelty_adversarial_review_20260812.md`

当前 `minimal-r1d` 规范：

- `/home/ykj/project/WM/gwp-agent-research-docs-integration/docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`
- `/home/ykj/project/WM/gwp-agent-research-docs-integration/docs/research/sm_fet_minimal_r1d_freeze_addendum_20260812.md`
- `/home/ykj/project/WM/gwp-agent-research-docs-integration/docs/research/sm_fet_minimal_r1d_freeze_addendum_review_20260812.md`
- `/home/ykj/project/WM/gwp-agent-research-docs-integration/docs/research/sm_fet_minimal_r1d_final_review_20260812.md`

### 1.2 旧 portfolio 与 r1d 冲突时，以 r1d 为准

旧 final portfolio 的 `active+next2 / 82 uint8 / 13 hard tokens / 207.7497 bit` 属于较早 standard-r1。M/N/O 终审不得再沿用它作为 action wire。当前唯一 action contract 是：

```text
schema                  sm_fet_hard_grammar_v2_minimal_r1d_struct_v1
action clock            active-only; Q_action=1; N_future=0
field wire              32 bytes
hard token slots        5
structural states       157,516,896,299,151,958,972,545,105,921
structural capacity     96.99142134161319 bit
minimum index width     97 bit
incremental M/N/O bits  0
G1 executable profile  NO-GO / UNDEFINED
```

结构容量只描述 hardware-family grammar，不是 target profile 的实际 reachable capacity，更不是任务互信息或 minimal sufficiency 证明。M、N、O 均不得据此声称“动作只含约 97 bit 的全部任务信息”。Packet 外仍有完整 public state、structured text、noise/timestep。

### 1.3 `idea-discussion` 登记结果的边界

本轮按 Zotero-first workflow 登记了综合命题。自动 critic 正确给出“组合 novelty 风险高”的总判断，但其中召回的 `A-MemGuard` 与本题无关，未作为证据。真正约束本裁决的最近邻仍来自三份已核验 memo：

- M：many-to-many correspondence、BUDS/PA-BiCoop、Human2Any、DexMachina 与异构 TAMP 已覆盖宽泛 role/morphology allocation；
- N：EgoWAM、V-JEPA 2、DreamDojo、contact anticipation、XSkill/MOTIF 已覆盖 human video predictive transfer、future latent和离散 skill；
- O：LUPI/generalized distillation、Learning by Cheating、Being-H0.7、LIDEA、EgoBridge 与 DexMani 已覆盖 privileged/future teacher和 human-to-robot distillation。

所以后文只守一个可证伪的**系统因果命题**，不把 M0、next-event CE 或 KD 各自包装成算法首创。

## 2. M：必要的是 M0 编译边界，不是 learned FRG

### 2.1 为什么 M0 不能删除

Human-only shared key合法包含：

```text
phase, effect,
1..2 constraints:
  purpose, primary_role, primary_part,
  secondary_role, target_relation
```

它按定义不包含：

```text
robot OPERATE / STABILIZE / MONITOR binding
LEFT_RH56 / RIGHT_OMNIPICKER / MONITOR_ONLY assignment
metric POINT / AXIS / FRAME / PATH locator
working-surface or controller target
held transform, fixture freshness, reachability, WBC or collision receipt
25-D robot action
```

因此从 shared semantics 到 executable packet 必须有一个 robot-side grounding step。删除它只有两种结果：让 human branch伪造 robot-specific truth，或让 action读取 planner/teacher hidden补齐缺失；二者都违反 r1d。

M0 的精确接口是：

```text
input:
  canonical shared semantic key s
  robot-only metric proposal l from current causal observation
  immutable static capability profile p and task/text stratum t
  immutable metric codebook hash
  complete public_execution_state_v1
  current runtime resource / held / monitor receipts R

pure finite procedure:
  purpose -> candidate function
  enumerate finite assignment/resource bindings
  apply static profile/resource mutex
  apply current runtime admission
  require a unique canonical candidate
  serialize -> decode -> validate -> byte round-trip

output:
  UNIQUE(canonical 32-byte / 5-token r1d packet)
  or AMBIGUOUS / INFEASIBLE / UNKNOWN -> HOLD/REOBSERVE
```

Purpose 的唯一投影是：

```text
EFFECT_GOAL             -> OPERATE
SUPPORT_REQUIREMENT     -> STABILIZE
                        -> MONITOR only with fresh verified passive support
OBSERVATION_REQUIREMENT -> MONITOR
```

如果 active STABILIZE 与 passive MONITOR 都会导致不同动作，且冻结输入不能唯一判别，结果是 `AMBIGUOUS`，不是按 candidate order、score、置信度或 task ID 暗选。

### 2.2 M0 是 r1d 基础设施，不是 headline module

M0 没有可训练参数、自由 latent 或独立 action route：

```text
L_M0 = 0
trainable parameters = 0
incremental action-visible fields/bits = 0
```

它应该并入 `minimal-r1d` compiler/validator 的规范实现，而不是在模型图中画成第二个“morphology network”。“FRG”可以作为开发期简称，但论文最安全的叫法是 **deterministic profile-grounded compiler**。

`source_actor_group`、human literal left/right 与“一只 human 手对应两个 requirements”等事实只保留在 annotation/audit sidecar。最终两条 robot constraints 已完整表达 target cardinality，不新增 `ONE_HUMAN_TO_TWO_ROBOT` bit。

### 2.3 为什么当前删除 M+

M+ learned scorer 只有在公开 finite input 能区分多个候选、存在高质量 adjudicated allocation gold、且 M0/ILP 明确不足时才有研究理由。当前不满足：

1. 真实 G1 profile、resource target identity、compound allowlist和metric artifact尚未就绪；
2. 若完全相同的公开输入对应不同 gold，正确结论是 schema/profile证据不足，而不是让网络记 scene shortcut；
3. 单一 RH56+OmniPicker profile无法支撑 morphology generalization；
4. learned score/logit极易成为 candidate-choice hidden channel。

所以默认 curriculum 中没有 allocation loss、edge BCE、bundle margin或 learned abstain logit。未来若 M+ 相对 M0/ILP exact allocation不足 `+5 pp`、下游不足 `+5 pp`，直接删除；即使超过，也只能作为独立晋级实验，不能回写本轮最小主线。

## 3. N 与 O：不是数学同义，但科研归因高度重复

| 维度 | N：causal next-event pretrain | O：privileged trajectory KD | 终审批注 |
|---|---|---|---|
| human 输入 | anchor前的 causal RGB/event windows | 完整 human event hand/contact trajectory，可能含posterior/future evidence | O信息更强，也更易不可实现 |
| 是否需要人机pair | 否；只需相同冻结ontology | 是；需要split-safe semantic event pair | N的数据合同更简单 |
| 作用位置 | 初始化/预训练robot planner | robot planner adaptation时加KD loss | 都只改变同一planner |
| 合法目标 | next domain-neutral shared event | 最多同一shared semantic fields的soft target | O不得输出full r1d |
| 部署剩余 | planner weights | student planner weights | 两者都应删除human runtime输入 |
| 最近前例风险 | EgoWAM/V-JEPA/procedure/contact anticipation | LUPI/Being-H0.7/LIDEA/EgoBridge | O的通用机制更新 |

### 3.1 为什么不默认合并

同时打开 N 和 O 会增加三类无法归因的变化：

- N使用unpaired sequence prior，O使用paired posterior/KD；数据可用样本和监督强度不同；
- 两者同时优化相同 phase/effect/constraint/transition fields，正结果无法判断来自 prediction、pairing、future privilege还是额外训练步数；
- O的dense trajectory teacher容易把human anatomy、future outcome、pair ID或estimated geometry压入student hidden，再借共享参数或hard-ST绕到action。

因此终审不是“把两者合成一个更完整模型”，而是把它们放进同一个 **finite-event planner training-signal** 因子实验，默认只取 N。

### 3.2 为什么择 N

N 更适合作为唯一主线，因为：

1. 它不需要逐事件human–robot pair或DTW/OT，当前36+36 pilot更容易做到split-safe；
2. human target天然停在domain-neutral shared key，不诱导assignment/locator pseudo truth；
3. past-only causal prediction比full-event posterior更接近部署可实现信息；
4. 它能被current-event、masked-event、JEPA/equal-capacity latent和EgoWAM-style target直接证伪；
5. residual novelty虽仍很窄，但比“再加一个privileged trajectory teacher”更容易守住 strict finite route 的中心命题。

### 3.3 对 O 原设计的必要收缩

原 O memo 中“五个 r1d slot/field logits”若包含assignment、metric locator或resource事实，不可作为human-only teacher truth。合法 O challenger 只能蒸馏：

```text
phase, effect, constraint count,
purpose, role, part, target relation,
field known/unknown/censor distribution
```

禁止蒸馏：

```text
assignment, O/S/M final binding, metric locator,
resource/held receipt, robot action,
free slot state, relational hidden or continuous trajectory target
```

第一版 O challenger只允许 masked semantic KL；不使用slot cosine、relational hidden、human pose regression或action-visible continuous teacher。这样 O 回答“paired privileged semantic target是否比N有效”，而不是靠更宽通道取胜。

## 4. 唯一数据流与信息边界

### 4.1 Human-only N pretraining

在 reviewed exit anchor 后定义目标下一事件 `e_{k+1}`。输入全部严格位于anchor之前：

```text
human inputs:
  past causal ego RGB windows W_1:k
  causal timestamps and deployment-reproducible availability masks
  frozen domain-neutral structured task goal

human targets:
  next phase
  next effect
  canonical roster branch NULL / 1 / 2
  each next constraint's purpose / roles / part / relation
  interval/right-censored time-to-next-event
  per-field KNOWN / REVIEWED_UNKNOWN / MISSING loss support
```

`event_focus_role`由canonical sorted constraints机械派生，不设独立自由head。只有 `EXHAUSTIVE` roster、1..2条requirements且required fields完整known时，才形成完整shared key；partial样本只对known field计loss。`NULL`只来自EXHAUSTIVE no-event，missing/occluded不编NULL。

若clip结束只证明next boundary右删失、却不能证明horizon内不存在event，则只计`L_interval`并mask未知semantic fields；不得把right-censor误标成NULL。

这里作为输入的availability mask只能来自anchor前可重复的sensor/tracker状态；人工review的`KNOWN/REVIEWED_UNKNOWN/MISSING`只决定target与loss support，不得作为部署时不存在的privileged input。

Human-only 禁止目标：

```text
human left/right -> robot left/right mapping
robot O/S/M, assignment or locator
robot profile/resource/held facts
MANO-retarget action, pseudo 25-D action or robot trajectory
transparent liquid depth/volume/contact interface completion
```

主 N 不使用future EMA teacher、Event-JEPA、pixel reconstruction、entry/turn/exit pose loss或dense trajectory auxiliary。它们只可作为互斥强baseline，避免“next-event + JEPA + pose + KD”的机械堆叠。

### 4.2 Robot planner adaptation

Robot decision anchor使用相同事件时钟：上一事件结束后预测即将执行的event；经M0接受后，它成为接下来execute-8窗口唯一action-visible active event。正在持续的event可在下一个anchor重复编译，但不会把next1/next2放入action。

```text
robot planner inputs:
  current/past causal robot RGB
  complete public_execution_state_v1（其中已含canonical25 proprio）
  frozen structured text

robot planner targets:
  same domain-neutral shared key fields
  robot-only metric locator factors and their known masks
  event boundary/censor target for planner calibration

robot-only audit targets, not learned allocation targets:
  adjudicated canonical full packet or reject status
  native/sensor-confirmed contact/effect/outcome
```

只有planner-owned human-pretrained权重迁入该阶段。Robot-specific state encoder、metric head和action expert不从human checkpoint加载。

### 4.3 M0 compiler/validator

M0读取 hard-canonicalized shared prediction、robot metric proposal、固定profile/codebook与当前receipts，输出完整hard packet或拒绝。它不读raw human video、human actor grouping、pair ID、planner hidden、soft logits、candidate score、scene ID或per-sample profile selector。

`MONITOR_ONLY+MONITOR` 是action-visible constraint，但 own actuator ownership mask恒零；fresh evidence可条件化同bundle的physical slice。证据missing/expired时拒绝整个required bundle，不能静默删除MONITOR继续执行。

### 4.4 Robot action expert

Action expert的完整允许输入只有：

```text
canonical minimal-r1d 5 hard tokens
complete public_execution_state_v1
canonical structured text
noisy robot action / diffusion or flow timestep
```

它预测的目标动作布局保持canonical asymmetric25，不因M/N/O改变：

```text
left wrist 9 | right wrist 9 | left RH56 6 | right OmniPicker aperture 1
```

禁止输入：

```text
raw human or robot reference RGB
human prompt/retrieval/pair table
planner/allocator/teacher hidden or logits
confidence, visibility heatmap, trajectory/pose
receipt metadata/reason code/candidate order
future event token or event hazard
```

真实held transform是public state的一部分，不是旁路。正确不变量是固定packet、完整public state、text、noise/timestep后，替换receipt metadata或任何human/planner private value，pre-safety action不变。

Safety plane只能读取allowlisted current validity/contact做clamp、hold或retract；若它生成主要任务轨迹，必须另报direct-controller baseline。始终同时报告pre-safety与executed action。

## 5. 损失与参数/梯度合同

### 5.1 Human N loss：只有显式 next-event 与 censor

对canonical排序后的target：

```text
L_H = L_next_sem + L_interval
```

其中：

```text
L_next_sem
  = active-field-normalized masked CE over
    phase, effect, constraint count,
    purpose, primary_role, primary_part,
    secondary_role, target_relation

L_interval
  = discrete interval/right-censored survival NLL
```

`KNOWN`参与field CE；有明确training ontology UNKNOWN code的`REVIEWED_UNKNOWN`监督UNKNOWN，否则mask=0；`NOT_REVIEWED/MISSING`一律mask=0。两个loss先按有效field/event归一化，再等权相加。不存在human action loss、locator loss、assignment loss、JEPA loss或KD loss。

### 5.2 Robot planner loss

```text
L_P = L_robot_sem + L_robot_locator + L_robot_interval
```

- `L_robot_sem`与human factorized semantic loss同构，但target来自robot current/native review；
- `L_robot_locator`只对POINT/AXIS/FRAME/PATH实际required且有native/calibrated evidence的有限factor做masked CE；
- `L_robot_interval`使用与human阶段相同的interval/right-censored survival NLL，只服务planner calibration和事件anchor，不进入action packet；
- M0 assignment/resource判定是纯程序，不对planner加入allocation score loss；
- full packet只用于exact-match、candidate recall、reject和counterfactual审计，不通过action loss暗教assignment。

### 5.3 Action expert loss

Action expert只用robot data和gold/oracle canonical packet训练：

```text
L_A = E || v_A(a_tau | packet, public_state, text, tau) - v_target ||_2^2
```

若当前实现采用diffusion noise prediction，可用等价的masked action denoising MSE；所有N/O arm必须使用同一形式、相同robot episodes、steps、noise和初始化。Action expert先训练后冻结，其checkpoint hash在所有planner arm完全相同。

### 5.4 不做联合总loss

不存在下面这种联合优化：

```text
No-Go: L_H + L_KD + L_P + L_A -> one shared transformer
```

各loss分阶段、而非联合优化；`L_H`与`L_P`顺序更新同一个planner参数集，action使用与之不重叠的独立参数集，M0无参数：

```text
L_H, L_P -> planner-owned theta_P only
L_A      -> action-expert theta_A only
M0       -> no parameters
```

严格主结果要求：

```text
d L_A / d theta_P              = 0
d L_A / d M0                   = 0
d action / d planner_hidden    = 0 at fixed packet
d action / d human input       = 0
d action / d pair/teacher      = 0
theta_A hash identical across N/O arms
```

Planner与action expert不得共享被human阶段更新的backbone或Transformer blocks。若为了工程复用必须从同一视觉checkpoint初始化，则在human pretraining前复制并冻结action侧副本；参数对象、optimizer和export graph必须分离。否则即使runtime没有human tensor，human更新仍可能通过共享权重直达action，不能称“收益只穿过packet”。

Forward边界使用 `hard-detached`：canonical bytes序列化、decode、五token encode后进入action；action loss不反传planner。`hard-ST`只作为第三项关键ablation的诊断臂。如果只有hard-ST有效，结论必须改成end-to-end quantized bottleneck，而不是strict supporting curriculum。

### 5.5 O challenger 的唯一合法loss与删除时点

O不属于默认路线。若第二项关键ablation运行O：

```text
p_T = stop_gradient(teacher shared-field distribution)
L_O = mean_valid_fields KL(p_T || p_P)
```

Teacher不得读对应robot action，pair solver和pair weight均stop-gradient；gold与teacher冲突的field权重置零。不要加入free slot/relational hidden对齐。

Teacher删除时点是：**robot planner adaptation结束、checkpoint选择完成后。** 此时先冻结student；只允许在隔离的删除等价性审计中，对同一robot input各运行一次pre-delete/post-delete forward，验证planner packet bytes与action bytes相同。随后删除human dataloader、pair sidecar接口、KD projector、teacher cache和teacher checkpoint引用；正式export、评估和closed-loop rollout全部只能在删除后进行。不等价即泄漏。

N主线本身没有teacher。Human预训练结束后只保留planner-owned weights；human clips、human normalization artifact与training-only hazard/auxiliary输出均不进入action graph。

## 6. 唯一最小 training curriculum

### Stage 0：合同与可执行性，当前唯一允许阶段

在训练前必须完成：

1. 实际运行并通过r1d rank/unrank、wire、JSON/text、public-state、metric-codebook CPU golden gates；
2. 冻结真实G1 camera metric codebook与hash；
3. 冻结authoritative RH56/OmniPicker capability/resource profile、working surfaces、same-effector rules和profile hash；
4. 建立held/public state、reachability、WBC/collision、fixture/monitor freshness receipts；
5. 证明M0对每个admitted tuple要么唯一编包、要么deterministic reject；
6. 冻结planner/action独立参数边界、action input allowlist和route/cache receipt。

当前这些条件未完成，所以本memo的后续阶段都是future protocol，不是训练许可。

### Stage 1：标注与split gate

先按operator/session/day、object/tool instance、capture/calibration batch划分split，再做crop、boundary review、normalizer、pairing或teacher cache。先做：

```text
3 human calibration + 12 blind double-annotated human episodes
```

只冻结shared ontology、EXHAUSTIVE/missing/censor规则和past-only anchor。现有66段human pour可用于此gate与透明/missingness audit；旧88段dual-Inspire grab只能作task-mismatch/NULL/hard-negative stress，不能与66 pour组成正pair。

### Stage 2：N human-only causal pretraining

用human train split优化 `L_H`。不接robot action、不生成pseudo action、不运行O teacher。选择checkpoint只看human validation next-event、censor calibration和visual-evidence intervention；test robot rollout不参与选择。

输出只有：

```text
planner-owned encoder/event-transformer initialization
frozen ontology/head mapping receipt
human pretraining data/compute receipt
```

### Stage 3：robot-only oracle与action expert

关闭全部human输入。用target robot native/sensor-confirmed数据完成两件相互隔离的工作：

1. 比较 `gold full packet`、`gold shared key + robot locator + M0`、fixed/literal assignment与direct-visual ceiling，先验证compiler和finite route；
2. 只用gold/oracle canonical packet训练action expert `theta_A`，随后冻结并hash。

若oracle packet本身不充分，不进入planner transfer；human pretraining不能修复一个对action不充分的packet。

### Stage 4：robot planner adaptation

分别从同一architecture的：

```text
robot-only initialization
N human-pretrained initialization
```

开始，用完全相同robot labels、steps、optimizer和parameter count优化 `L_P`。M0把预测hard fields编成packet；action expert冻结，`L_A`不回planner。主比较是达到固定packet/event指标所需robot labels，以及predicted packet接入同一frozen action expert后的闭环结果。

### Stage 5：export、旁路审计与闭环

导出仅包含：

```text
robot planner
M0 compiler/validator
r1d codec/token encoder
frozen robot action expert
public state/text/route validators
```

先通过fixed-input、cache、parameter hash和packet intervention审计，再进行predict-48/execute-8/reobserve。任何required field/receipt过期时清packet/cache并HOLD/REOBSERVE。

### 明确不在curriculum中的内容

```text
M+ learned allocator
N+O 默认联合训练
future-event action token / active+next2 wire
human MANO/pose/flow action condition
Event-JEPA或video generation主loss
CAPOT/OT hidden或pair ID action condition
continuous teacher hidden/slot distillation
action loss反传planner
额外action field或bit
```

## 7. 人机数据需求

### 7.1 Feasibility pilot

最小同任务pilot沿用12-cell设计：

```text
3 task families: POUR / INSERT / CAP
× 2 object conditions
× 2 session/day groups
= 12 cells

per cell: 3 human + 3 target-G1 robot
total:    36 human + 36 target-G1 robot
```

要求：

- 72段全部有shared event/missing/censor review；
- robot段必须有canonical25 action、完整public state、native/sensor-confirmed contact/effect/outcome和packet gold；
- split按cell/group先划分，human与robot无需一一配对；
- N主线只要求task/ontology overlap，不需要pair table；
- 36+36只够feasibility和stop-gate，不足以支撑主论文泛化。

若运行O challenger，额外需要至少 `180` 条双标并裁决的 semantic match/NULL/hard-negative pair decisions；pair graph connected component必须完整落在单一split。O的数据成本不得计入N主方法预算后再称“同数据”。

### 7.2 Paper-scale

只有pilot通过才扩到五个冻结task families：`POUR / INSERT / PIPETTE / CAP / WIPE`。Robot data至少报告每task `n={10,25,50}` 的label/action-demo sweep、3 seeds、operator/day/object/tool group-held-out；最终样本量用pilot的cluster variance做power，不把frame/window数当独立n。

在只有RH56+OmniPicker一个target profile时，论文只能声称 **human-to-one-target-robot transfer**。若要写 morphology generalization或把M升级成独立方法，至少需要三个真实target morphology/profile、leave-one-morphology-out和每个profile的真实不同但功能等价allocation；同硬件mask不算新morphology。

## 8. 三项关键 ablation

除常规日志外，决定主张生死的只有以下三项。

### A1. Compiler mediation：M0是否必要且足够

固定同一frozen action expert、public state、text与robot数据，比较：

```text
A1-a gold executable full packet
A1-b gold shared key + gold robot locator + deterministic M0
A1-c literal human-side / fixed left-act-right-stabilize assignment
A1-d learned M+ scorer（diagnostic only，finite inputs matched）
```

主指标：canonical packet exact match、candidate recall、infeasible false accept、fixture/STABILIZE↔MONITOR consistency、same-effector compound、task success和安全。

解释：A1-b接近A1-a且胜A1-c，证明M0闭合了必要compiler缺口；A1-d不显著胜M0则永久删除M+。若M0 candidate set不含gold，应修profile/schema，不允许用M+补洞。

### A2. N/O 2×2：选择是否正确

固定相同planner architecture、可用human/robot样本、robot steps、总训练compute、action expert hash和action-visible packet：

```text
A2-a no human signal
A2-b N only
A2-c O only
A2-d N + O
```

N必须另对照同标签/同compute的current-event与Event-JEPA/EgoWAM-style generic future target；O必须另对照current/single teacher、Being-H0.7-style future-RGB posterior、zero/random/within-class shuffled teacher。Pair-only样本量和teacher预训练compute单独记账。

解释：默认选择N不是先验保护结果。O有两条互斥晋级路径：若O-only通过common teacher/strict gates且稳定胜N，可用O**替换**N；只有N+O相对`max(N,O)`仍有独立增益，才允许审查**合并**。未过对应gate时，O删除而不是写成默认辅助模块。

### A3. Semantic route与梯度中介

固定canonical packet/public state/text/noise后，比较：

```text
A3-a semantic hard-detached（主结果）
A3-b matched equal-capacity hard code
A3-c semantic hard-ST
A3-d direct planner/visual hidden upper bound（明确非strict）
```

并执行：

- human clip/teacher/pair sidecar swap；
- planner hidden/logit swap但packet bytes固定；
- candidate order/receipt reason swap；
- 合法packet field intervention；
- action expert参数、optimizer state与cache key审计。

固定合法输入时pre-safety action `max_abs <= 1e-6`；packet字段变化应产生预注册的局部动作效应。若equal-bit持平，停止semantic necessity；若只有hard-ST或direct hidden有效，停止strict finite-route claim。

## 9. 最窄 novelty 与明确非主张

### 9.1 唯一可守表述

> 我们检验一种训练期人类知识迁移合同：从egocentric human video学习的past-only causal finite next-event prior只初始化一个参数独立的robot current-observation planner；一个确定性的profile/resource compiler再把domain-neutral semantics与robot-only geometry/current receipts编译为唯一canonical `minimal-r1d` packet。部署action只读取该packet、完整public robot state与structured goal。固定robot data和action expert时，收益必须胜过current-event、generic future posterior/KD、同容量hard code和shuffled teacher，并在action-gradient detached、无human/pair/soft-hidden旁路下成立。

这是关于“human process prior能否仅经有限可审计packet改善target robot action”的实证命题。M0是必要合同，不是novel learned allocator；O是challenger，不是第二贡献。

### 9.2 禁止表述

```text
first next-event prediction from ego video
first human-video world-model transfer
first LUPI / privileged trajectory distillation
first many-to-many or morphology-aware allocation
first dynamic operate/stabilize role assignment
first contact/event/discrete token for robot control
learns robot action from human video alone
minimal-r1d is a proven minimal sufficient statistic
cross-embodiment generalization from one target profile
transparent-liquid physics or 3D understanding
```

最安全标题方向是：

```text
Causal Finite-Event Priors through an Auditable Compiler
for Human-to-Robot World-Action Transfer
```

不要把标题写成“Privileged Hand-Contact Distillation”或“Morphology-Aware Role Allocation”，除非相应独立晋级gate另外通过。

## 10. Stop gates

### G0. Contract gate：当前未过，立即阻止训练

以下任一未完成，N/O/M+和WAM训练均No-Go：

- r1d CPU conformance未实际通过；
- metric codebook/profile/resource target/compound rules未冻结；
- G1 public/runtime receipts或25D目标数据不存在；
- human target包含assignment、locator、resource或action；
- planner/action参数不独立，或action input allowlist无法机械审计。

### G1. Annotation gate

`3 calibration + 12 blind` 最低要求：

```text
shared requirement exact-set agreement >= 0.80
phase/effect/role/part/relation kappa >= 0.67
boundary-status kappa >= 0.70
critical reviewed-unknown/missing <= 40%
1..2 constraint overflow <= 5%
```

协议最多修订两轮；仍失败则删除不可靠字段。若只剩coarse phase，停止N独立headline。

### G2. M0 与 oracle packet gate

```text
M0 candidate recall >= 0.95
gold shared key+locator -> exact packet >= 0.85
infeasible false accept <= 0.05
ambiguous/unknown abstention F1 >= 0.80
ambiguous gold share <= 20%
hidden-channel probes = 100% pass
```

Oracle hard packet的closed-loop success距matched direct-visual ceiling不得超过 `5 pp`，first-contact/stop-release和安全不得退化。失败则停止finite action route；不允许扩大human teacher补救。

### G3. N prediction 与 transfer gate

Human event层：

```text
next-event vs text-only/Markov/current baseline >= +5 pp macro-F1
next-event vs same-label current/masked objective >= +3 pp
window delete/exchange产生局部语义下降
ECE <= 0.10
```

Robot层：相对最佳matched VideoMAE/JEPA/EgoWAM-style target，group-OOD success `>=+5 pp`且cluster 95% CI下界 `>0`，或达到固定success所需robot demos减少至少30%；predicted planner必须恢复 `>=70%` oracle packet gain，安全不退化。只提升offline event F1而不提升action，停止N transfer headline。

### G4. O 重新准入 gate

O默认删除。任何O路线先过共同有效性gate：

```text
full observed trajectory teacher vs current/single >= +10 pp
student-observation不可实现冲突 <= 20%
O胜zero/random/shuffled/future-RGB posterior
teacher删除前后packet/action等价
```

共同gate通过后，两种晋级条件分开判：

```text
替换N：O-only相对N的paper-scale OOD增益 >= +5 pp，95% CI下界 > 0

合并N+O：相对max(N,O)的pilot增益 >= +3 pp，
          且paper-scale OOD增益 >= +5 pp、95% CI下界 > 0
```

共同gate或对应晋级条件任一失败，O只保留为负结果/teacher modality ablation；不能把“可替换N”误写成“应与N叠加”。

### G5. Route与因果 gate

以下任一旁路/梯度违规发生，即strict/auditable route claim无效：

- fixed packet/public state/text/noise时human、teacher、pair、planner hidden swap导致action `max_abs > 1e-6`；
- action cache含planner/teacher/soft/pair值；
- `theta_A`在N/O arms hash不同或被human loss更新；
- action gradient越过hard compiler；
- invalid packet存在visual/soft fallback。

另外两项是科学主张stop，而不是旁路审计失败：legal packet intervention没有局部效应时，停止causal-consumption claim；matched equal-capacity code与semantic route持平时，停止semantic necessity及依赖它的中心novelty，但不据此声称serialization/旁路检查本身失败。

### G6. Scope与paper gate

- 36+36只过feasibility，不得写paper-scale泛化；
- 主结果至少五task families、3 seeds、operator/day/object/tool group-held-out；
- 一个target profile只能写human-to-one-robot transfer；
- morphology claim须至少三个真实profiles并leave-one-morphology-out；
- full/native trajectory若在预注册safe-action endpoint稳定超过finite route且差距无法由planner error解释，停止finite sufficiency；
- required constraints超过2条比例 `>5%`、same semantic bundle内动态可持续性collision `>10%`，相应task/profile停止，不用hidden score压缩第三条constraint。

## 11. 对 canonical 代码的事实边界

本轮基线 `e0ebc692` 仍是legacy v1.1 Q12×22D contact route，不是r1d实现。可复用但不能冒充已完成的代码事实：

- `world_action_model/models/transformer_wa_casual_mot.py:127-194`：已有可关闭action→reference直达的causal mask；
- `world_action_model/models/transformer_wa_casual_mot.py:1253-1285`：planner hidden当前不离开 `_plan_contact_logits()`；
- `world_action_model/models/transformer_wa_casual_mot.py:1405-1445`：当前22D projection是唯一planner→action route，未concat planner hidden；
- `world_action_model/models/transformer_wa_casual_mot.py:2296-2335`：action-only cache已有fail-closed packet receipt检查；
- `world_action_model/datasets/wam_lerobot_dataset.py:741-819`：reviewed sidecar已有SHA-bound fail-closed加载模式；
- `world_action_model/trainer/wa_casual_trainer_mot.py:957-1055`：当前trainer分别处理contact与action loss，但尚未实现本memo要求的参数隔离和r1d课程。

这些位置只是future insertion/audit surface。本轮没有改代码，不能声称r1d codec、M0、N、O或五token action route已实现。

## 12. 最终一句话

**M/N/O 的唯一最小解不是“三个idea一起训练”，而是：用 N 学一个只预测domain-neutral next finite event的human prior；用无参数、可拒绝的 M0把robot侧语义和几何编译成现有r1d hard packet；用完全robot-only、参数隔离且冻结的action expert消费该packet。O不进入默认课程，除非它在N之上通过独立因子、future-posterior、shuffled-teacher和删除teacher后的严格旁路gate。当前由于CPU conformance与真实G1 executable profile未就绪，唯一批准动作仍是完成Stage 0，而不是开始训练。**
