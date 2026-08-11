# Idea E 独立调研：Counterfactual Contact–Action Consistency

- 日期：2026-08-11
- 审查基线：`d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88`
- 隔离分支：`codex/idea-e-contact-action-consistency-20260811`
- `idea_id`：`8f386f3f-50be-45b5-970d-4bca4bd1a062`
- `discussion_id`：`bba6e215-6a61-4203-9c03-696c17a42bf5`
- `idea-discussion`：已执行 Zotero-first，同步 100 条，`memory_updated=true`
- 范围：只做文献调研、架构设计和证据审计；没有训练、没有使用 GPU、没有修改 canonical 代码或硬件配置

## 一句话结论

**Idea E 作为 Idea A 的“语义是否真的被机器人实现”的因果闭环是 Conditional Go；作为当前条件下的独立主论文是 No-Go。**

Ego 视频真正教会机器人灵巧操作，不是因为它让视觉表示更懂“倒水/抓瓶子”这些语义，而是因为它提供了一个机器人能够执行、能够被反事实改变、并能由真实后果检验的中间目标：未来 contact transition 与 object-centric canonical hand/tool pose。机器人侧必须另外学到 `动作 → 实际接触/姿态效果`，再要求动作生成器满足：

```text
ego/robot observation ──> desired contact + canonical pose
                                  │
                                  ▼
robot action generator ──> action chunk ──> frozen forward-effect model
                                  │                    │
                                  └──── consistency ───┘
```

而且在固定 observation、proprio、扩散噪声和机器人数据预算时，对 contact/pose 做可行的 `swap / delay / reorder` 干预，动作的预测效果必须按被干预字段和阶段改变。没有这条 action-mediated、intervention-tested 证据链，Ego 带来的成功率提升仍可能只是场景、对象、语言或动作阶段的语义泛化。

当前 66 段 Ego 倒液数据没有物理 contact truth、没有 robot action/effect、没有可信 OOD group；旧 88 段 dual-Inspire `grab_red_bottle` 与目标 RH56DFTP + OmniPicker 不同，也没有接触/结果真值。因此它们只够做 loader、标签协议、网络接口和弱诊断，不能证明 Idea E。

## 1. 要证明的不是相关性，而是一个可证伪的交换图

令：

- `o_t`：当前 RGB/RGB-D、proprio 和可选触觉；
- `g_t=(c_t,p_t)`：未来 `H=48` 的 contact grammar `c_t` 与 canonical pose `p_t`；
- `π_R(o_t,g_t,ε)`：目标机器人动作生成器；
- `F_R(o_t,s_t,a_{t:t+H},m_R)`：机器人 embodiment-specific forward contact-effect model；
- `y_t^R`：从执行后的观测、仿真 oracle、触觉或人工复核得到的**实际实现效果**，不是计划标签。

Idea E 的核心命题应写成：

```text
F_R(o_t, s_t, π_R(o_t, g_t, ε), m_R) ≈ g_t
```

并对可行反事实 `g_t' = do_k(g_t)` 成立：

```text
F_R(o_t, s_t, π_R(o_t, g_t', ε), m_R) ≈ g_t'
```

其中 `ε` 必须在原样本和反事实样本之间相同。第二条比“shuffle 后性能下降”更强：它要求动作生成器不只依赖 token，而且能让指定 contact/pose 变化通过机器人动作落到预测的物理效果上。

这一定义同时排除了三个常见伪证据：

1. planner 准确但 action 不使用它；
2. action 随 token 变化，但改变的是无关关节或全局速度；
3. forward head 直接读取 token 后复述它，并没有从 action 推断效果。

## 2. 与已有工作的冲突边界

本轮先执行了 `idea-discussion` 的 Zotero-first 流程。自动检索命中的个人笔记主要是 RT-2（`X9TZGLQK`）、π0（`LFI6T4BX`）、RT-1（`EK59GCPY`）、FAST（`CBLGABU2`）和 π0.5（`CIJX7QJ3`）；它们构成 VLA/action token 背景，但没有覆盖本 proposal 的 contact-effect intervention。自动 scout 返回的 failure-recovery 和图像篡改检测论文相关性较弱，没有被当作 novelty 证据。随后核对了仓库已有 Zotero 审计条目和 arXiv/作者论文原文。

### 2.1 Ego-to-robot、affordance 与 retargeting

| 工作 | 已经覆盖什么 | 对 Idea E 的约束 |
|---|---|---|
| [VLAff](https://arxiv.org/html/2608.05215) | 204K Ego episodes；预测 visual heatmap、96D MANO grasp 与量化 6D trajectory；用 3D scene 与 retargeting 变成可执行动作 | 不能声称首次从 Ego 预测 contact/grasp/trajectory 或首次使用 canonical grasp/pose；E 必须证明生成动作的**实际 contact effect**与预测目标一致，并做干预 |
| [BridgeACT](https://arxiv.org/html/2604.23249) | human video → tool/target role、contact region、interaction-centric 3D point flow → grasp + closed-loop SE(3)；无需 robot demo，含真实倒水 | 不能声称首次把 tool-target/contact dynamics 作为 human-to-robot bridge；差异只能是 learned action-effect cycle、反事实可控性和 joint WAM 内的实证 |
| [EgoMimic](https://arxiv.org/html/2410.24221) | 用 Aria 3D hand pose 和机器人 EEF pose 的共享 pose head，共训 human/robot；joint action 只由 robot data 监督 | 直接否定“共享 pose head 让 Ego 教 action”本身的新颖性；E 要证明接触效果而不是共享表示，且不能把 task-matched 共训误写成零 robot grounding |
| [EgoVLA](https://arxiv.org/abs/2507.12440) | 预测 human wrist/hand action，再 IK/retarget 到机器人并用少量 robot demo 微调 | wrist/MANO unified action 与 retargeting 不是 novelty；应作为 pose-only baseline |
| [DexMimicGen](https://arxiv.org/html/2410.24185) | 将双手示范切为 object-centric parallel/coordination/sequential subtasks，用 SE(3) 变换生成约 21K 物理可执行示范，并只保留成功轨迹 | segment/order/synchronization 与物理可执行数据生成已有强前例；E 的 shuffle/delay 不能只做数据增强，必须成为可审计 causal intervention |
| [VideoManip](https://arxiv.org/html/2602.09013) | 从 RGB human video 重建 hand/object 4D trajectory，做 contact optimization、retargeting 与 DemoGen，训练 DP3 dexterous policy | 不能声称首次从 RGB human video 学真实灵巧手或首次用 contact optimization 保证可执行性 |
| [HOWTransfer](https://arxiv.org/html/2606.10743) | 定位 contact interval、taxonomy-aware grasp retargeting、contact-aligned trajectory editing，报告 86% replay success | contact onset + pose retarget + trajectory editing 已覆盖；E 必须是闭环 learned policy 的 effect faithfulness，而不是 replay quality |
| [C2Dex](https://arxiv.org/html/2608.07045) | 在 canonical object space 聚合 stable contact，用于 HOI reconstruction 与跨手 contact-preserving retargeting，再以 residual RL 修正 | “contact-consistent reconstruction/retargeting”这个名称和宽泛 claim 已碰撞；E 需明确是一条 `generated robot action → realized future contact` 的 forward consistency，而不是几何 contact loss |

### 2.2 contact-conditioned policy、forward/inverse consistency 与 world model

| 工作 | 已经覆盖什么 | 对 Idea E 的约束 |
|---|---|---|
| [Hierarchical Diffusion Policy](https://arxiv.org/abs/2411.12982) | 高层预测下一接触，低层 diffusion policy 以接触为条件生成轨迹 | contact-conditioned diffusion policy 不是 novelty |
| [Contact-conditioned locomotion](https://arxiv.org/abs/2408.00776)、[GeCCo](https://arxiv.org/abs/2509.17582) | 用未来 contact location、switch time、duration 条件化低层策略 | `site/start/end` 到 action 的接口本身不新 |
| [OmniContact](https://arxiv.org/abs/2606.26201) | key body trajectories + binary contact flow，支持 humanoid skill chaining 和 recovery | contact time series/flow 不是 novelty |
| [DexMani](https://arxiv.org/html/2608.00554) | 从 human visual-tactile demo 学 contact-conditioned manipulability evolution；用 structured negatives；冻结 energy prior，评分候选 action 诱导的 manipulability change，再指导多种 dex hand RL | **最强直接冲突**：不能声称首次把 human contact transition 变成 action-effect guidance、首次用 structured negatives 或 frozen prior。E 的剩余空间是多任务 event-level grammar、真实 realized contact forward model、WAM action chunk 与显式 do-intervention，而非单步固定接触几何下的旋转 manipulability |
| [Learning Cross-Domain Correspondence with Dynamics Cycle-Consistency](https://arxiv.org/abs/2012.09811) | 以 dynamics cycle 在不配对视频/状态动作域间对齐动态行为 | dynamics cycle/cross-domain unpaired correspondence 不是 novelty |
| [VERA](https://arxiv.org/html/2605.27817) | 视频 planner + embodiment-specific Jacobian IDM；以 joint forward pixel-flow / inverse action reconstruction 训练，并要求 action faithfully reproduce visual transition | “forward/inverse consistency 让视觉计划可执行”已是明确前例；E 必须将一致性对象收缩到 contact/pose，并证明 counterfactual contact controllability |
| [AMPLIFY](https://arxiv.org/html/2506.14198) | human/robot action-free video 学离散 motion forward model；有限 robot action data 学 inverse dynamics；在固定 action budget 下展示 human video 增益 | “human video 学 what、robot data 学 how”的 modular factorization 已被覆盖；E 需要比 motion token 更强的 contact-specific、interventional 证据 |
| [Structured World Models from Human Videos](https://arxiv.org/abs/2308.10901) | human video 学 affordance-space world model，再用少量 robot interaction fine-tune | human video world model + robot grounding 不是 novelty |
| [Deep Visual Foresight](https://arxiv.org/abs/1610.00696) | action-conditioned video prediction + MPC | 用 forward model 预测 action effects 并规划不是 novelty |
| [GigaWorld-Policy](https://arxiv.org/abs/2603.17240) | 当前 backbone 已联合 action prediction 和 action-conditioned future video；论文明确把 visual-dynamics constraint 作为物理合理性信号 | 不能把“WAM 中 action effect consistency”宽泛地当新增贡献；E 必须说明现有 pixel future loss为何不足、contact effect head 新增了什么可验证量 |
| [EVA](https://arxiv.org/html/2603.17808) | 用 frozen IDM 给视频 rollout 产生 executability reward，反向对齐 video world model；论文还报告 reward hacking | frozen evaluator/物理可执行对齐已有前例；同时警告 E 的 learned effect model 也会被 policy 利用，必须用 ensemble、uncertainty gate 和真实 rollout 验证 |
| [Action-conditioned FIGNet](https://arxiv.org/abs/2509.12151) | 学习 action-conditioned contact-rich motion/force dynamics，用于 peg-in-hole MPC | action-conditioned contact dynamics 不是 novelty |

### 2.3 HOI forecasting 的边界

- [Joint Hand Motion and Interaction Hotspots Prediction](https://arxiv.org/abs/2204.01696) 已从 ego video 预测未来 hand trajectory 和 active-object contact hotspot。
- [Forecasting Action through Contact Representations](https://arxiv.org/abs/2102.00649) 已用 making/breaking contact 组织第一视角动作预测。
- C2Dex、VideoManip、HOWTransfer 已覆盖 contact-stable HOI reconstruction、canonical object contact 与跨手 retargeting。

所以“预测未来 hand pose/contact”只能是必要 supervision，不能单独支撑 novelty。

## 3. 可防守的窄 novelty 与不可写的 claim

截至 2026-08-11，本轮没有找到同一工作同时做到：

1. 在 action-centered WAM 中预测显式 contact-transition grammar 与 object-centric functional pose；
2. 用**不读取 contact token/text 的** robot forward-effect model 判断生成 action chunk 实际会造成什么接触效果；
3. 对 contact site/role/timing/order/pose 做 geometry-feasible `do` 干预，并要求 action-mediated effect 随干预等变；
4. 用固定 robot data、相同噪声与 matched controls 证明 Ego 数据的收益被这些干预解释，而不是任务语义或场景识别解释；
5. 在目标异构 `left RH56 6 + right OmniPicker 1` action space 上验证。

可写成的窄 claim 是：

> We test whether egocentric contact plans become executable robot skills only when an embodiment-specific action generator is trained and audited through realized contact effects under feasible contact interventions.

以下说法均不可守：

1. 首次从 human/ego video 学机器人灵巧操作；
2. 首次用 contact、grasp pose、trajectory 或 contact transition 跨 embodiment；
3. 首次把 contact condition 传给 diffusion/VLA/WAM；
4. 首次用 forward/inverse consistency 让视觉计划可执行；
5. 首次用 counterfactual/structured negatives 教 robot contact skill；
6. 首次证明 action-conditioned future prediction改善物理合理性。

独立 novelty 的强度目前是**弱到中等**。只有把方法做成 schema/backbone-agnostic 的 `Contact-Effect Cycle + Feasible Intervention Protocol`，并在不依赖 A 的 finite channel、也不依赖 B 的 topology 时仍稳定优于 pixel-future、pose-only 和 contact-conditioned baselines，才可作为独立方法。否则它应并入 A，成为 A 不可缺少的“realizability/faithfulness”贡献和主因果实验。

## 4. 与 Idea A、Idea B 的区别和合并方式

| Idea | 主要问题 | 核心变量 | 单独成功仍不能回答什么 |
|---|---|---|---|
| A：finite bottleneck | 当前图像到 action 的信息是否只能通过有限、可审计 contact grammar？ | channel capacity、strict route、hard semantic packet | packet 是否被动作真实实现 |
| B：asymmetric topology | 哪个 object/part/contact relation 应分配给 RH56 与 OmniPicker，是否可行？ | relational topology、role allocation、morphology feasibility | 分配后的动作是否产生目标 contact effect |
| E：contact-action consistency | action generator 是否忠实实现 contact/pose，并在干预后产生对应效果？ | action-mediated forward effect、counterfactual equivariance | channel 是否有限、拓扑是否足够表达双手关系 |

最佳合并是 `A + E`：

- A 负责“action 只能读什么”；
- E 负责“action 读到以后有没有真的做到”；
- E 的 `F_R` 是 training-only critic/evaluator，不扩大 A 的 action-visible source channel；
- canonical pose 若是连续 6D/12D，默认只作为 planner/effect 的 training target，不送入 action。若送入 action，必须量化、计入 A 的精确 bit budget，并增加 equal-bit control。

`B + E` 是更强但更贵的版本：`F_R` 不只预测独立 segment，还预测 `sync / precedes / stabilizes / transfers` 是否被动作实现，并对 allocator assignment 做 intervention。它最能说明灵巧双手协调，但需要真实 target hardware 或可信仿真 contact/tactile oracle。

## 5. 建议的 WAM 架构插入点

### 5.1 保留现有 planner，新增“计划”和“实际效果”的严格分离

当前实现已经有：

- `world_action_model/contact_transition_schema.py:64-105`：22D segment schema；
- `world_action_model/models/transformer_wa_casual_mot.py:1096-1169`：由 state/reference/text 预测 contact logits，并做 hard transport；
- `world_action_model/models/transformer_wa_casual_mot.py:1171-1211`：22D semantics 是唯一 planner→action 投影；
- `world_action_model/models/transformer_wa_casual_mot.py:110-177`：future token 可以读取全部 action/state/reference/contact token，而 strict action 不读 reference；
- `world_action_model/trainer/wa_casual_trainer_mot.py:897-981`：现有 loss 只有 visual/action/contact，没有 realized-effect consistency。

建议新增独立 `ContactEffectModel`，逻辑接口为：

```text
F_R(current proprio,
    embodiment id,
    current contact/tactile state,
    scene/object geometry independent of planner packet,
    denoised physical action chunk)
  -> realized contact-transition distribution
  -> realized canonical pose/effect distribution
  -> feasibility + epistemic uncertainty
```

硬约束：

- `F_R` 不得读取 planner logits、hard packet、text、task ID 或 planner hidden；
- geometry 可以读取完整当前场景或独立 object tracks，但不能只读取 packet 选中的 site，否则会形成标签旁路；
- 必须有 `F_R(o,s,a)` 对 `F_0(o,s)` 的 no-action baseline，证明预测增益来自 action；
- human 与 robot 的“计划标签”不能同时充当 robot “实际效果标签”。两者必须用不同字段和 provenance；
- 训练 consistency 时冻结 `F_R`，或使用冻结 EMA target；不能让 policy 与 effect head 联合串通。

建议另建 `world_action_model/contact_effect_model.py` 与 `world_action_model/contact_effect_schema.py`，而不是直接改变 v1.1 22D 文件。只有 oracle gate 通过后，再决定是否把 pose 字段升级进 action-visible grammar。

### 5.2 在 flow matching 中得到可微的 clean action estimate

当前 trainer 在 `world_action_model/trainer/wa_casual_trainer_mot.py:824-827` 定义：

```text
noisy_action = σ * noise + (1-σ) * action
action_target = noise - action
```

因此模型输出 flow velocity `v_hat=action_pred` 后，可以在单次训练 forward 中构造：

```text
a0_hat = noisy_action - σ * action_pred
```

这正是送给 frozen `F_R` 的可微 clean-action estimate。插入位置应在 `action_pred` 得到后、`masked_weighted_action_mse` 之前：

1. 应用 `action_dim_mask`，去掉 25D 后的 padding；
2. 用训练时固定 norm 的逆变换得到 physical canonical25D；
3. `F_R` 前向参数冻结，但允许梯度对 `a0_hat` 回到 action generator；
4. `stop_gradient(g_t)`，防止 planner 把目标移动到“容易实现”的常数 contact；
5. 对 effect-model 高不确定样本将 consistency mask 为 0，不在 OOD 区域强推 policy。

目标 25D action contract 已由 `world_action_model/asymmetric_action_schema.py:18-28,92-163` 定义：左右腕各 9D、左 RH56 6D、右 OmniPicker aperture 1D。forward effect 必须读取这个 25D 的真实物理语义，不能把旧 30D dual-Inspire 或 model padding 当目标 action。

### 5.3 不推荐直接复用 future visual hidden 当 effect head

现有 mask 在 `transformer_wa_casual_mot.py:175` 允许 future token 读取所有 token，其中包括 planned contact；若直接从 future hidden 解码 contact，它可能复述 planner，而不是利用 action。现有 visual loss 也已经是 action-conditioned future dynamics，无法单凭新增 decoder 证明 E。

若必须复用 WAM future branch，至少需要：

1. 新增 effect-only queries；
2. mask 掉 effect query→contact/text/planner hidden，只保留 current geometry/state/action；
3. 或先生成 future video，再用完全冻结、独立训练的 contact detector 从像素解码 actual effect；
4. 与独立轻量 `F_R` 做对照。

主推荐仍是独立、可冻结、可校准的 `F_R`，部署时可丢弃，从而不损害 action-only 推理延迟。

## 6. 训练信号与反事实构造

### 6.1 四类基础损失

总损失建议为：

```text
L = L_visual + L_action + λc L_contact_plan + λp L_pose_plan
    + λF L_forward_observed
    + λcyc L_action_effect_cycle
    + λcf L_counterfactual_equivariance
    + λloc L_intervention_locality
    + λsafe L_feasibility_safety
```

含义：

1. `L_contact_plan`：现有 grouped BCE/CE/Smooth-L1，监督 desired grammar；
2. `L_pose_plan`：object-centric pose 的 geodesic rotation、translation/scale、axis/bin loss，尊重 visibility/missing mask；
3. `L_forward_observed`：在 robot/sim 已执行轨迹上训练 `F_R(s,a_gt) -> y_realized`；
4. `L_action_effect_cycle`：冻结 `F_R`，要求 `F_R(s,a0_hat)` 接近 `stopgrad(g)`；
5. `L_counterfactual_equivariance`：相同 observation/noise 下，`g'` 的 action effect 应接近 `g'`；
6. `L_intervention_locality`：只改 timing 时，变化主要发生在相应 phase；只改一侧/site 时，无关侧和无关阶段保持在预注册容差内；
7. `L_feasibility_safety`：不可达、碰撞、超限或 effect-model 高不确定反事实只训练 infeasible/negative，不强迫生成动作。

### 6.2 三类反事实不是无条件 shuffle

#### Contact swap

- 在相同任务、object role、近似 geometry/reachability 组内交换 `side/site/part`；
- 或同一对象内把 contact cell 移到另一个可达 functional site；
- raw MANO pose、不同物体尺度或不同任务之间的任意交换默认不合法；
- 同一 batch 中完整 packet 交换只能作“错误条件”negative，不能自动当成应被实现的 positive。

#### Timing delay

- 只在 episode 内移动 onset/release，禁止跨 episode wrap；
- delay 必须不越过物理先决条件，例如不能在抓稳前执行倾倒；
- 对可延迟事件要求动作时间局部平移；对违反前置条件的 delay 输出 infeasible，不强迫 `F_R(π(g'))=g'`。

#### Segment shuffle / order intervention

- 只在有已知 non-commutative relation 的任务中使用，例如 `grasp → lift → pour → recover → release`；
- 保留 segment multiset、只改 precedence，以隔离“内容”与“顺序”；
- 对物理不可行顺序训练 feasibility ranking；对有两种都可行的顺序训练双向 effect equivariance。

### 6.3 反事实退化与防护

离线 expert data 只覆盖窄 action manifold。`F_R` 很可能在 counterfactual action 上被 policy exploit；EVA 已公开展示 reward hacking。必须：

- 先在 simulation oracle 或安全 robot perturbation 数据上覆盖 failed/off-policy actions；
- 使用 `F_R` ensemble/epistemic uncertainty，禁止在高不确定区域回传 cycle loss；
- 冻结 effect model并在独立 group-heldout action perturbations 上校准；
- 报告 effect score 与 simulator/真实 rollout outcome 的 Spearman/Pearson 相关，而不只报告训练 loss；
- 定期用真实/仿真 rollout 审计“模型认为实现”是否真的实现；
- 保留 BC/flow likelihood、动作平滑和安全 trust region，防止为了骗 `F_R` 生成离谱动作。

## 7. Canonical hand/tool pose：无需 MANO 的主方案与可选 MANO

### 7.1 主方案：functional object-centric pose，不依赖 MANO

Ego-to-robot 的共享量不应是 15 个人手关节，而应是功能性姿态：

```text
p = {
  active_object/tool/receptacle slot,
  T_object^-1 T_wrist_or_tool,
  approach/retract axis,
  tool axis relative to receptacle/rim,
  grip/closure band,
  selected contact site/normal,
  visibility/confidence/censor mask
}
```

对于倒液，`source vessel axis relative to target rim/opening` 比 human finger joint pose 更接近任务效果。机器人侧可以从 FK、相机标定、object/tool pose 与 gripper state 得到真值；human 侧可用：

- 2D wrist/hand keypoints + object/tool mask；
- camera SLAM/已知内参/metric depth 时提升到 object-centric SE(3)；
- 无可靠 metric depth 时只监督 normalized 2D contact、尺度比、approach/axis bins，并显式 mask 3D 字段；
- 不允许把单目伪深度当精确 3D GT，也不允许把 hand closure proxy 当 contact truth。

为兼容 Idea A，连续 pose 默认是 training-only planner/effect target；动作只读现有 finite contact packet。若 oracle 证明 contact-only 不足，再将 pose 量化为少量 position/orientation bins，并重新计算通道容量。

### 7.2 可选 MANO：只作 teacher，不作跨手 action

可选使用 MANO/WiLoR/Dyn-HaMR 提供：

- wrist SE(3)、finger contact candidates、hand-object relative pose；
- 从 96D/mesh 投影到上述 functional pose 和 contact groups；
- 对遮挡或不稳定帧提供 uncertainty/mask。

raw MANO 96D 不进入 target robot action head，不把 human thumb/index 直接映射到 RH56/OmniPicker，也不把 MANO retargeting包装成 novelty。VLAff、C2Dex、HOWTransfer 和 VideoManip 已经使这一宽泛方向高度拥挤。MANO 只能是更好的 label teacher/pose baseline。

## 8. Ego/robot 配对需求

### 8.1 训练不需要逐帧同步 pair，但需要三层重叠

1. **Ontology overlap**：Ego 与 robot 都能映射到相同 `side/site/object role/mode/timing/pose` 定义；
2. **Skill primitive overlap**：Ego 中出现的接触 primitive 在 robot/sim action-effect 数据中有覆盖，否则 `F_R` 和 action decoder没有 grounding；
3. **Scene/task overlap for causal evaluation**：至少有同任务、同 object geometry/part layout、同初始状态分布的 human/robot 数据，时间上不必同步，但要能构造 matched evaluation cell。

因此可以使用 task-level unpaired Ego/robot 数据训练 planner 与 executor；不能用完全不同任务的数据宣称 zero-shot dexterous skill transfer。对新技能的 zero-shot claim 只能限于已 ground 的 contact primitive 新组合。

### 8.2 机器人数据不可被省略的部分

至少需要：

- target action chunk、current state、执行后 contact/pose/outcome；
- failed/off-policy 或安全扰动 action，用于识别 action effect，而不是只拟合 expert correlation；
- 每个 counterfactual field 的可行正例和不可行负例；
- target morphology/calibration/latency，或可信 simulator oracle；
- 相同 robot demo budget 下的 robot-only 和 ego+robot paired runs。

BridgeACT 可用外部 grasp/controller 做 no-robot-demo execution，VideoManip/C2Dex 可用 reconstruction+retarget+simulation 生成 robot trajectory；但本项目的 learned target action head和 learned `F_R` 若没有任何 robot/sim grounding，不能声称学会了 RH56+OmniPicker 的真实执行。

## 9. Benchmark、baselines、消融和指标

### 9.1 分层 benchmark

#### L1：simulation oracle，先验证机制

- ManiSkill/RoboTwin 2.0 或仓库 `track_b/sim_oracle_benchmark_v0` 扩展；
- 读取真实 contact pair、object pose、penetration、force、success；
- 任务包含 grasp/lift/place、peg/connector insertion、cap/faucet rotation、tool contact、pour-like object-object alignment、bimanual handover；
- 同一 instruction/object 设置不同 contact site/order/hand role，使纯语义模型无法区分。

#### L2：public perception/transfer 辅助

- Ego4D FHO、EPIC-KITCHENS、HOI4D、Ego-Exo4D：planner/forecasting 预训练；
- DexYCB/TACO：canonical pose/contact reconstruction；
- LIBERO：固定 robot budget 的 action learning和 OOD，但其 parallel-jaw/simple physics 不能证明目标异构灵巧性；
- RoboTwin 2.0：bimanual structured tasks和与 EVA/强 VLA baseline 的补充比较。

#### L3：目标硬件主结论

必须在 G1 + 左 RH56DFTP + 右 OmniPicker 上至少覆盖三个任务族：

1. 容器抓取、对准、倾倒、回正、放置；
2. 右侧稳定、左侧旋盖/拨杆/插接/工具尖端操作；
3. handover 或先稳定后释放的双手 contact-order 任务；
4. 可选 role reversal：几何迫使左右职责交换或报告 infeasible。

若普通双夹爪或单手 baseline 已能近似完成任务，任务不能证明 dexterity transfer。

### 9.2 必须有的 baselines

1. 原始 GWP-0.5 direct policy；
2. robot-only GWP，固定 robot demos；
3. Ego visual/semantic pretrain + direct policy，检验纯语义收益；
4. EgoMimic-style shared canonical pose head；
5. VLAff/BridgeACT-style affordance/pose/3D flow + retarget/controller（能适配时才作 executable baseline）；
6. Idea A hard contact bottleneck，无 effect cycle；
7. Idea B topology/allocator，无 effect cycle；
8. contact-conditioned policy + `F_R` 只作 evaluator，不回传；
9. pixel future consistency only，即当前 GWP visual loss；
10. inverse-only canonical pose→action，无 forward effect；
11. `F_R(s,a)` 与 no-action `F_0(s)`；
12. oracle contact/pose、oracle realized-effect 和 simulator MPC upper bound。

### 9.3 关键消融

- 去掉 `L_action_effect_cycle`；
- 去掉 counterfactual，仅保留正样本 cycle；
- swap / delay / order 各自删除；
- random/off-support counterfactual vs feasibility-matched intervention；
- frozen/EMA effect model vs joint train；
- effect model可见 token/text 的 leaky control；
- contact-only vs contact+training-only pose vs quantized action-visible pose；
- no-MANO vs MANO teacher；
- full scene geometry vs selected-site geometry vs no geometry；
- effect ensemble/uncertainty gate 删除；
- action-consistency gradient只进 action、只进 planner、两者都进；主方法应为目标 stop-grad、梯度进 action；
- same-noise paired intervention vs independent noise；独立噪声不能作为 causal 对照；
- Ego 数量 scaling、robot demo budget `0/5/10/25/...` 曲线。

### 9.4 指标

接触/pose 层：

- one-to-one event precision/recall/F1、segment tIoU；
- onset/release MAE、contact-order accuracy；
- site/role/mode/side macro-F1；
- object-frame translation、rotation geodesic error、surface geodesic error；
- calibration ECE/Brier/NLL，effect ensemble uncertainty。

action-effect 层：

- `F_R(s,a)` 相对 `F_0(s)` 的 conditional gain；
- feasible intervention compliance：被改变字段/phase 的实际效果命中率；
- locality：未干预 action dims、手侧、时间段的变化；
- effect score 与 simulator/真实 rollout 的 rank correlation；
- effect-model exploitation rate：高预测分但真实失败的比例。

任务/安全层：

- overall 与 phase success、correct-first-contact；
- spill/volume error、collision、penetration、drop、slip/regrasp；
- unintended contact、force/load overshoot、安全干预；
- latency、闭环频率与 action-only inference overhead；
- 固定 robot demo budget 的 data-efficiency curve。

所有主比较使用 seeds `6666/7777/8888`，按 task/object/operator/session group 做 cluster-aware bootstrap/层级模型；不同窗口不能当独立 trial。

## 10. 成功与停止门槛

### Gate 0：监督可复核

- contact event F1 ≥ 0.80；
- matched segment median tIoU ≥ 0.70；
- onset/release median MAE ≤ 2 帧（20 Hz 为 100 ms）；
- side/site/role/mode 的 κ ≥ 0.60，coverage 加权总体 κ ≥ 0.70；
- canonical pose 的 3D 字段只在可靠可见/标定帧参与损失。

协议修订一次仍不过 Gate 0，停止 human-auditable contact/pose 主张。

### Gate 1：forward effect 可识别且不能忽略 action

- group-heldout 上 `F_R(s,a)` 的 contact macro-F1 比 `F_0(s)` 至少高 10 pp；
- onset/release MAE ≤ 2 帧，pose translation/orientation 达到任务预注册容差；
- observed 与安全 action perturbation 的 intervention ranking AUROC ≥ 0.80；
- effect score 与 simulator/真实 outcome 的 rank correlation ≥ 0.60；
- 高不确定/off-support counterfactual 比例 ≤ 20%。

若 `F_R` 不显著优于 no-action baseline，说明它主要读场景/阶段语义；Idea E 立即 No-Go。

### Gate 2：oracle realizability

相对相同 backbone/data 的 Idea A 无 consistency 版本，使用 oracle contact/pose 时：

- correct realized-contact 至少 +10 pp；
- correct-first-contact 至少 +15 pp；
- overall task success 至少 +5 pp；
- spill/collision/safety 不退化。

oracle packet 加 effect cycle 都无增益，说明损失不能改善 executor，停止扩大 Ego 数据。

### Gate 3：counterfactual causal sensitivity

- 至少 75% 的 geometry-feasible intervention pair，其 realized effect 按指定字段/阶段变化；
- 未干预侧/phase 的动作变化留在预注册 tolerance；
- shuffle/delay/order corruption 至少消除 50% 的 Ego 增益，或显著增加对应 phase failure；
- independent-noise control 不得替代 same-noise matched intervention；
- leaky `F_R`、random counterfactual 与 joint-train `F_R` 不得复制主结果。

如果动作只整体变大/变慢、无字段局部性，或 counterfactual 几乎不影响真实 phase outcome，不能做 causal claim。

### Gate 4：Ego 真正贡献 skill，而不是语义

固定 robot demos、backbone、steps、augmentation、noise 和 hyperparameters，`Ego + A + E` 相对 robot-only、Ego-semantic-direct 和 `A without E`：

- asymmetric OOD aggregate task success 至少 +10 pp，clustered 95% CI 不跨 0；
- realized contact/order 至少 +10 pp；
- 至少三个 task family 分别受益，不能由单个倒水任务驱动；
- predicted contact/pose 至少恢复 oracle gain 的 70%；
- 语义相同、接触结构不同的 test 中仍保持优势。

达不到时，只能说 Ego 改善表示/语义泛化，不能说教会了灵巧操作。

### Gate 5：目标硬件

沿用当前 asymmetric25 gate：OmniPicker driver、canonical-to-device mapper、hardware/camera calibration、latency、limits、watchdog、emergency stop、bench/robot verification 和 receipt 全部完成。当前硬件条件为 No-Go。

## 11. 现有 66ep 与旧 dual-Inspire 数据能证明什么

### 11.1 66 段 Ego 倒液（20,013 帧、20 Hz、单任务）

能证明/可用于：

- 数据读取、H=48 窗口、T5、sidecar、mask、norm 和 trainer plumbing；
- `hand_status=0 open / 1 closed` 的极性与 missingness-aware grip loss；
- 人工 contact/pose 标注协议 pilot；
- 以时间连续块做 engineering regression；
- no-MANO 2D/object-normalized pose extractor 的离线稳定性检查；
- counterfactual packet transform 的 shape、mask、episode boundary 与 same-noise hash 测试。

不能证明：

- `hand_status` 是物理 contact；大量 tracker inactive 帧的 proxy 仍为 closed；
- robot action、robot forward effect 或 action→contact 因果关系；
- metric 3D hand/tool pose、液流/接触/受力真值；
- cross-object/operator/session/liquid OOD；这些 group metadata 缺失；
- Ego 对 target robot success 的增益；
- counterfactual contact 可由 RH56+OmniPicker 实现；
- 66 个 episode 是 66 个独立 contact event；滑窗高度相关。

当前 Track A 仍没有正式双标/裁决 contact GT；episode 0 prescreen 或 proposal 不能变成 scientific label。

### 11.2 旧 88 段 `grab_red_bottle` dual-Inspire robot 数据

已知事实：

- 88 episodes / 40,901 frames / meta 50 Hz；
- `observation.state` 与 `action.wbc` 都是 41D：29 body + left Inspire6 + right Inspire6；
- 任务是 grab red bottle，不是 66ep 的 beaker pouring；
- 没有可直接使用的 contact/tactile/outcome GT；EEF pose 父坐标系未声明；时间戳可能由 frame/fps 生成。

能用于：

- legacy embodiment 下的 action-conditioned branch 与 flow clean-action reconstruction smoke；
- 人工补标后做单任务、弱可见 contact onset/hold/release 的诊断性 `F_R`；
- 检查 forward model 是否完全忽略 action；
- 比较 dual-Inspire 与 future target hardware 的 morphology shift，作为负迁移/不兼容示例。

不能用于：

- 把右侧 Inspire 六通道压缩/重命名为 OmniPicker aperture；
- 训练或验证 target asymmetric25 effect model；
- 证明 pour、tool use、rotation、contact order 或 bimanual role allocation；
- 证明真实力/接触/成功效果；
- 与 66ep 构成 task-matched Ego/robot pair；
- 证明 zero-shot cross-embodiment 或 target-hardware dexterity。

两份数据合起来仍缺：同任务/同对象族的 Ego–robot overlap、真实 realized effect、counterfactual action coverage、可信 OOD group、目标硬件 action/effect。不能因为都有手轨迹就宣称形成了 contact-action cycle。

## 12. 最小推进顺序

1. 不动 5B 模型，先完成 contact/pose 双标 pilot 与 Gate 0；
2. 在 simulation oracle 上用小网络验证 `F_R(s,a)` 是否显著胜过 `F_0(s)`；
3. 先用 ground-truth robot actions训练/冻结 effect model，并加入安全 action perturbation；
4. 在小 policy 或 oracle action-conditioned WAM 上验证 `a0_hat=noisy-σv_hat` 与 effect cycle；
5. 只做 geometry-feasible same-noise swap/delay/order interventions；
6. Gate 1–3 通过后，才将 E 接到 Idea A 的 hard semantic path；
7. 补采 task-matched Ego/target-robot 数据和真实 group metadata；
8. 最后才跑三 seed 5B matched experiment 和目标硬件 Gate 5。

## 13. 最终 Go / No-Go

| 判断对象 | 决定 | 原因 |
|---|---|---|
| 把 E 作为 A 的 realizability/faithfulness 闭环 | **Conditional Go** | A 只证明信息路由，E 补上“动作是否真的实现 packet”的关键缺口 |
| 把 E 作为 B 的 topology executor audit | **Conditional Go，后置** | 价值高，但依赖目标硬件/仿真 contact oracle 和更复杂标签 |
| 当前将 E 写成独立主方法论文 | **No-Go** | DexMani、VERA、AMPLIFY、GWP/EVA 已覆盖大量组件；当前无 effect data、无 causal result |
| 声称 66ep 证明 Ego 教会灵巧操作 | **No-Go** | 只有 human proxy/单任务数据，没有 robot action-effect、GT 和 OOD group |
| 用旧 dual-Inspire 代替 RH56+OmniPicker | **No-Go** | action morphology 和硬件 contract 不同，且无 contact/outcome truth |
| 先做小规模 simulator `F_R`/counterfactual gate | **Go** | 最低成本、最直接地证伪 Idea E 是否值得进入 5B WAM |
| 现在训练 5B / 上真机 | **No-Go** | Gate 0–4 未过，目标硬件 driver/calibration/safety 也未闭环 |

最终建议：论文主叙事仍以 Idea A 为轴，把 Idea E 升级为主方法的第二个必要命题——**finite contact packet 不仅是唯一 action-visible 视觉桥，而且其每个可行干预都必须通过 robot action 产生相应 realized effect**。如果 effect model无法识别 action、oracle cycle 没有增益或 same-noise intervention 不局部地改变真实结果，就应停止“Ego 教会灵巧技能”的说法，而不是继续扩大语义数据规模。

## 14. 原始来源

### Ego-to-robot / contact transfer

- [VLAff](https://arxiv.org/abs/2608.05215)
- [BridgeACT](https://arxiv.org/abs/2604.23249)
- [EgoMimic](https://arxiv.org/abs/2410.24221)
- [EgoVLA](https://arxiv.org/abs/2507.12440)
- [DexMimicGen](https://arxiv.org/abs/2410.24185)
- [VideoManip](https://arxiv.org/abs/2602.09013)
- [HOWTransfer](https://arxiv.org/abs/2606.10743)
- [C2Dex](https://arxiv.org/abs/2608.07045)
- [DexMani](https://arxiv.org/abs/2608.00554)
- [OmniContact](https://arxiv.org/abs/2606.26201)

### Dynamics / inverse / world models

- [GigaWorld-Policy](https://arxiv.org/abs/2603.17240)
- [GigaWorld-Policy-0.5](https://arxiv.org/abs/2607.13960)
- [VERA](https://arxiv.org/abs/2605.27817)
- [EVA](https://arxiv.org/abs/2603.17808)
- [AMPLIFY](https://arxiv.org/abs/2506.14198)
- [Structured World Models from Human Videos](https://arxiv.org/abs/2308.10901)
- [Learning Cross-Domain Correspondence with Dynamics Cycle-Consistency](https://arxiv.org/abs/2012.09811)
- [Deep Visual Foresight](https://arxiv.org/abs/1610.00696)
- [Action-conditioned FIGNet](https://arxiv.org/abs/2509.12151)

### Contact-conditioned policy / HOI

- [Hierarchical Diffusion Policy](https://arxiv.org/abs/2411.12982)
- [Contact-conditioned learning of locomotion policies](https://arxiv.org/abs/2408.00776)
- [GeCCo](https://arxiv.org/abs/2509.17582)
- [Joint Hand Motion and Interaction Hotspots Prediction](https://arxiv.org/abs/2204.01696)
- [Forecasting Action through Contact Representations](https://arxiv.org/abs/2102.00649)
