# Idea A 独立调研：Auditable Finite Contact-Transition WAM

- 日期：2026-08-11
- 审查基线：`d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88`
- 隔离分支：`codex/idea-a-finite-contact-wam-review-20260811`
- `idea_id`：`2fd0a8a6-0efa-4038-9aeb-93a8378e41c0`
- `discussion_id`：`44cf5abe-05c6-477d-93ce-c3dad8a6a3eb`
- 目标硬件：Unitree G1 + 左 RH56DFTP + 右 OmniPicker
- 目标场景：第一视角液体倒取；动作 schema 为 asymmetric25，再 pad 到 32D

## 一句话结论

**研究假设 Conditional Go；当前实验、收益主张和真机部署全部 No-Go。** 最接近的冲突不是某一篇工作完整覆盖本 idea，而是 BridgeACT 已覆盖“人类视频 → 工具/目标/接触区域 → 机器人动作”，VLAff 已覆盖“ego 视频 → 离散 affordance/抓取/轨迹 token”，ST-VLA/CoA-VLA 已覆盖“显式中间 affordance/时空表示驱动动作”，GeCCo/接触条件策略已覆盖“接触模式、位置和时序条件化控制”，VQ-VAE/FSQ/VQ-BeT 已覆盖“有限离散通道与直通梯度”。因此可守的新意只能是这些元素的**受审计交集**：在 joint WAM 内把当前图像到动作的唯一视觉因果路径限制为显式、有限、可人工干预的 contact-segment grammar，并用容量匹配的直接策略、连续语义通道、停止梯度和无语义有限码做配对实验。

当前仓库已经实现了相当完整的 plumbing，但没有真实复核标签、可用的目标硬件数据、真正的 OOD 分组、容量匹配 direct/unstructured controls 或任何训练收益证据。不能把配置可导入、张量可前传、哈希 receipt 或空白标注模板写成科学收益。

## 证据标记

- **[I] 已实现**：当前基线代码中存在、能定位到具体实现的机制或门禁。
- **[S] 仅规范/接口**：文档、模板、配置矩阵或 schema 已有，但没有完成数据或运行证据。
- **[M] 证据缺失**：没有实标、训练、统计、真机或可复现实验证据。
- **[R] 原始来源支持**：由论文原文、作者项目页或当前仓库中的原始报告支持。

## 检索记录：Zotero-first，但不把低质量命中当证据

本轮先按 `idea-discussion` 工作流同步并检索了 Zotero，再核对 arXiv/作者原文。Zotero 中较相关的已有条目包括 FAST（`CBLGABU2`）、π0.5（`CIJX7QJ3`）、RT-1（`EK59GCPY`）、RT-X（`G2VYMJA8`）和 Octo（`XB5R2DVZ`）。它们能界定通用 VLA、跨 embodiment 和 token/action chunk 背景，但没有命中本 proposal 的“有限 contact segment grammar + strict WAM route”精确前例。

自动外部检索还返回了 NVSS 天文和引力透镜等明显误命中；它们被丢弃，没有用于任何结论。以下相似性判断均重新回到论文原文，而不是依据检索摘要。

## 最接近前例与明确冲突

| 工作 | 原文中已经做了什么 | 与本 idea 的冲突 | 尚未覆盖的部分 |
|---|---|---|---|
| [VLAff](https://arxiv.org/html/2608.05215) | 从 egocentric 视频提取视觉 affordance heatmap、96D MANO grasp 和量化 6D trajectory token；规模为约 204K episodes，并把 affordance 映射到可执行动作。[R] | 不能声称首次从 ego 视频学习机器人 affordance、首次离散化交互轨迹、首次把抓取/轨迹中间表示用于动作。 | 不是固定 22D contact segment grammar；没有证明动作视觉路径只能经过可枚举语义；没有 censor/missingness 和本 proposal 的 causal controls。 |
| [ST-VLA](https://arxiv.org/html/2603.13788) | 以平滑空间 mask、3D 轨迹、初末接触点等 3D–4D 表示连接高层 VLM 与低层动作，在约 300K human episodes/14 tasks 上训练。[R] | 不能声称首次使用时空中间表示或接触点连接视觉语言理解与连续控制。 | 表示是几何轨迹/掩码而非有限可枚举 grammar；没有严格的信息路由与同容量因果对照。 |
| [CoA-VLA](https://arxiv.org/html/2412.20451) | 顺序产生 object、grasp、spatial、movement 四类 affordance，并以视觉/文本方式注入 policy，动态选择推理链。[R] | 不能声称首次显式可解释 affordance chain、首次把视觉/文本 affordance 注入 VLA。 | affordance 不是固定有限接触段；动作仍可保留其它感知路径；没有 censor 和 hard-ST 因果审计。 |
| [BridgeACT](https://arxiv.org/html/2604.23249) | 从 human video 解析 tool/target role、contact regions 和 interaction-centric 3D point flow，再接 grasp 与闭环 SE(3) 控制；真实任务包含倒水。[R] | 这是任务和表示层面最强冲突。不能声称首次 human/ego-to-robot 的工具—目标—接触桥、首次从人类视频做倒水、首次用接触表示消除 embodiment 差异。 | 其接口仍包含连续 3D flow 和独立 grasp 模块；没有 22D 固定 grammar、精确有限容量、动作视觉路径唯一性或 matched direct/soft/detached 诊断。论文倒水成功数也较低，反而说明精细旋转/接触仍难。 |
| [GeCCo](https://arxiv.org/html/2509.17582) | 用任意接触点和持续时间条件化四足低层策略；actor 使用接触命令误差/满足状态，critic 可读 privileged timing/duration。[R] | 不能声称首次 contact-conditioned control 或首次把 contact duration 交给策略。 | 接触计划是外部给定/采样而非 WAM 从 ego 图像预测；领域是 locomotion；不是人可审计的操作语法。 |
| [Contact-conditioned multi-gait policy](https://arxiv.org/html/2408.00776) | 用未来 contact switches/modes、末端位置和切换时序条件化 locomotion policy。[R] | `mode/start/end` 作为 policy condition 本身不新。 | 同样是规划器给定的连续几何 locomotion 命令，不是预测的语义瓶颈。 |
| [VQ-VAE](https://arxiv.org/abs/1711.00937) | 学习离散码本，并使用 straight-through 类估计训练离散 latent。[R] | hard forward/直通梯度不是贡献。 | 码无 contact 含义，也没有 causal route 审计。 |
| [FSQ](https://arxiv.org/abs/2309.15505) | 以若干固定标量 level 的笛卡尔积形成隐式有限 codebook，避免传统 VQ codebook 训练问题。[R] | 固定有限 alphabet 和标量量化不是贡献。 | 没有结构化接触语义、区间 censor 或机器人策略。 |
| [VQ-BeT](https://arxiv.org/html/2403.03181) | 用 residual VQ-VAE 将连续 action chunk 离散化，并预测 action code。[R] | “离散 token 驱动机器人动作”不是贡献。 | 它离散的是动作本身，码不要求人可读；没有把感知路径限制到 contact semantics。 |
| [DreamZero/WAM](https://arxiv.org/abs/2602.15922) | 在视频扩散 backbone 上联合预测 future video 与 action，报告跨任务、环境和 embodiment 泛化。[R] | joint future/action WAM、视频预训练带来物理先验和跨 embodiment 学习都不能作为本 idea 的首创点。 | future dynamics 是高带宽隐表示；没有受审计 contact-only 视觉通道。 |
| [GigaWorld Policy](https://arxiv.org/html/2603.17240) / [GWP-0.5](https://arxiv.org/html/2607.13960) | 当前观测和动作 token 联合进入 WAM；future token 可读 action，而 action 不读 future；0.5 用 MoT 分离视觉/动作专家，并支持 action-only 解码。[R] | 本项目的 joint WAM backbone 和 causal future/action 布局来自该线工作，不是新贡献。 | 原始 GWP 没有显式有限 contact grammar，也没有证明当前图像到动作只经过该 grammar。 |

### 不应再写的 “first”

以下说法都不可守：

1. 首次从 ego/human video 学 affordance 或机器人策略；
2. 首次让动作读取 contact/affordance 中间表示；
3. 首次在机器人策略中使用离散 latent、量化轨迹或 ST estimator；
4. 首次用接触模式、接触时刻、持续时间条件化控制；
5. 首次做人到机器人、跨 embodiment 或倒水迁移；
6. 首次把 world model 与 action generation 联合训练。

## 真正可能守住的 novelty

建议把贡献压缩成一个可证伪命题，而不是堆组件：

> 对 joint video-action WAM，在保留 state、language 和 noisy action 条件的前提下，将**当前图像派生的信息到动作头的唯一路径**限制为至多 Q 个显式 contact segment；每段由 presence/side/site/object-role/mode/start/end/censor 组成，hard forward 具有精确可枚举上界；通过容量/计算/数据配对的 direct、soft、hard-ST、hard-detached 和无语义有限码实验，检验该接口是否同时带来可干预性、跨 embodiment 泛化和液体倒取安全收益。

这个表述有四个必要限定：

- “唯一”只指**当前图像派生路径**，不是动作的全部输入；当前 action 仍合法读取 proprioceptive state、language、扩散 noisy action 和 contact token。[I]
- 有限性只适用于 `hard_st`/`hard_detached` 的 forward；`soft` 是连续通道，不能宣称有限。[I]
- 精确有限上界不是低互信息的同义词。Q=12、B=16 时当前实现给出每个 present segment 48,673 种、约 15.57 bit，Q12 上界约 186.85 bit；还需报告真实码使用率、熵和干预敏感度。[I/M]
- 新意在“语义、有限、严格路由、可干预、matched causal evidence”的交集；任一项没有实证，novelty 都会退化成已有组件拼装。

### 可以作为副贡献、不能单独撑论文的点

- missingness-aware、coverage-aware 的 contact segment 标注协议；
- left/right censor 显式进入 grammar；
- asymmetric hand/gripper 的统一 action contract；
- hash-bound 数据、split、噪声和 matched-control receipts。

这些目前主要是工程审计价值，不是策略收益证据。

## 当前架构中的具体插入点

### 1. 语义定义与有限容量

- `world_action_model/contact_transition_schema.py:64-74` 定义每段 22D：presence 1、side 3、site 6、object_role 4、contact_mode 4、start/end 各 1、left/right_censored 各 1。[I]
- `world_action_model/contact_transition_schema.py:114-187` 枚举 canonical hard channel 的精确容量；不存在“把 float tensor 当作有限码”的偷换。[I]
- `world_action_model/contact_transition_schema.py:190-486` 负责 segment 数据结构、coverage 语义、字段 mask 和 sidecar 编码。[I]

### 2. planner 放在动作编码之前

- `world_action_model/models/transformer_wa_casual_mot.py:1096-1169`：planner 用 state、reference visual token 和 text 生成 Q×22 logits；raw `planner_hidden` 在方法内终止，只返回 logits 与 transport 后 semantics。[I]
- planner 看的是当前 reference，不看 future video 或 target action；这避免用训练期 future 泄漏 contact plan。[I]
- field-masked oracle merge 已有接口，可做 schema sufficiency gate；但没有真实 oracle 结果。[I/M]

### 3. soft / hard-ST / hard-detached transport

- `world_action_model/models/transformer_wa_casual_mot.py:995-1094`：binary 阈值、categorical one-hot、时间 bin、`start<=end`、censor canonicalization、absence 清零和 `+0` canonicalization 均已实现。[I]
- `hard_st` forward 为 hard grammar，backward 用 identity surrogate；`hard_detached` forward bit-identical 但切断 action loss 回 planner 的梯度。[I]
- 这两个变体只诊断“动作梯度是否需要穿过离散接口”，不能单独证明语义或瓶颈优于 direct policy。

### 4. 唯一 planner-to-action 路径

- `world_action_model/models/transformer_wa_casual_mot.py:1171-1211`：Q×22 经过 `contact_condition_encoder` 投到 action width；注释和实现均未把 raw planner hidden 拼入 action stream。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:110-177` 与 `1571-1584`：strict mode 的序列布局为 `[state, contact, action] | [ref, future]`，action→ref edge 被 mask；semantic mode 还切断 state/contact 在主干重读 ref，因为 planner 已先读取 ref。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:1999-2051`：action-only prefix cache 在 strict mode 去掉 visual KV，只缓存 state/contact prefix，并保存 transported semantics。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:2105-2240`：推理先规划/缓存 semantics，再迭代 action denoising。[I]

### 5. cross-attention 旁路审计

通用 forward 暴露 `encoder_hidden_states_image`，且 `world_action_model/models/transformer_wa_casual_mot.py:1556-1569` 保留 image/text conditioning 接口。逐路检查后，当前生产路径尚未发现已生效的 action 图像旁路：

- `configs/giga_world_policy_0_5_sonic_g1_finetune.py:119-126` 将 `image_dim=None`；asymmetric contact config 继承此值。[I]
- action condition embedder 在图像 embedding 拼接语句之前已从 text 单独生成 `encoder_action_hidden_states`；动作专家 cross-attention 使用该 text-only 张量。[I]
- action-only prefix cache 没有把返回的 image embedding 加入 action expert KV。[I]
- `scripts/inference_openloop.py:895-898` 设 `guidance_scale=0.0`，当前 action-only conditional 调用也不传 `encoder_hidden_states_image`。[I]

但严格模式没有对非空 `encoder_hidden_states_image` 做 fail-closed 断言，也没有覆盖这一接口的正式不变量测试。[M] 投稿前必须增加：

1. strict mode 传入 image embedding 时明确拒绝，或证明 action output 对其逐 bit/数值容差不变；
2. 固定 oracle semantics 后扰动 reference image，action 输出保持不变；
3. 固定 semantics 后扰动/替换 planner hidden，不影响 action；
4. cached 与 uncached strict inference 输出一致；
5. 对 action output 对 ref token 的梯度做零路径检查，但不能只以梯度为唯一证明。

在这些测试完成前，只能说“代码意图和当前配置形成 strict route”，不能写成经证明的系统级信息流保证。

## 标注协议、missingness 与 censor

### 应标什么

每个 action chunk 最多 Q=12 个有序 segment。每段需标：

- `presence`；
- `side ∈ {left, right, object_object}`；
- `site ∈ {opposition, primary, support, palm_or_jaw, tool_tip, other_surface}`；
- `object_role ∈ {manipulated, tool, receptacle, fixture}`；
- `mode ∈ {stick, slide, roll, impact}`；
- 归一化 `start_tau/end_tau`；
- `left_censored/right_censored`；
- 每字段 `mask`、置信度、annotator、adjudication 与 coverage mode。

对倒取任务，至少覆盖 acquire container、support/stabilize、align receptacle、pour/flow、recover upright、release/place 等阶段；液体—容器接触和流动状态经常不可由单目 ego 帧确定，不能强制伪标签，应允许 field-level unknown。

### missingness 的硬规则

- `coverage=exhaustive` 时，未列出的 query 才能作为 absence negative；[I]
- `coverage=positive_only` 时，未列出只表示 unknown；[I]
- `coverage=unknown` 时，不得补负样本；[I]
- categorical field 必须整组 one-hot 均已知才参与 loss；部分 mask 整组跳过；[I]
- 边界落在 chunk 左/右之外分别用 censor 标记，不能把观察窗边界误当真实接触起止；[S]
- 遮挡、镜面/透明容器、液体不可见、手—物与物—物歧义均应 mask/降置信度，不能由 hand closure proxy 代替物理接触。[M]

### 当前数据与损失的真实状态

- `track_a/contact_labels_v1_1/` 与 `formal_dual_annotation_v1_1/` 是空白任务包/模板，没有 adjudicated GT。[S/M]
- `world_action_model/datasets/wam_lerobot_dataset.py:285-325,741-819` 已要求严格 sidecar、精确 SHA、human-adjudicated identity 和 `[T,12,22]` finite float32。[I]
- `world_action_model/trainer/wa_casual_trainer_mot.py:102-182` 已实现字段平衡的 BCE/CE/Smooth-L1，并尊重 mask/confidence。[I]
- 当前 `start/end` 仍是点回归，censor 仅作为两个独立 BCE 字段；没有 interval-censored/survival likelihood。[I] 若论文把 censor-aware learning 当方法贡献，应改损失并做对照；否则只能说 schema 保存 censor 信息。
- 现有 66 个 ego episodes、20,013 帧的 `hand_status` missingness 已审计，但它不是 contact truth；tracker inactive 时的 hand proxy 不能解除 mask。[I]
- 现有 66 episodes 缺 subject/session/operator、source/target instance、liquid identity/viscosity/fill、batch 等元数据，无法构造可信 production OOD split。[M]

### 标注质量闸门

先双人独立标注，小规模 pilot 只决定协议是否可用，不得写收益。建议闸门：

- coverage 加权 Cohen/Fleiss κ ≥ 0.70；
- one-to-one contact event F1 ≥ 0.80；
- matched segment median tIoU ≥ 0.70；
- onset/release median MAE ≤ 2 frames（20 Hz 时 100 ms）；
- side/site/role/mode 各自 κ ≥ 0.60，且每类有预注册的最小可比事件数；
- 未达标时修订 ontology 后重标一次；第二轮仍失败即停止将“human-auditable grammar”作为主线。

## Benchmark 设计

### 主 benchmark：目标硬件上的 ego 液体倒取

主结论必须落在 G1 + 左 RH56DFTP + 右 OmniPicker，因为 asymmetric embodiment 是论点的一部分。任务按 phase 和可控 shift 分层：

1. 左/右手角色与工具/容器角色互换；
2. source/target 容器实例、几何、材质、透明度变化；
3. 液体颜色、黏度、初始 fill level 变化；
4. 光照、背景、相机姿态/遮挡变化；
5. 同任务新 operator/session/day/calibration batch；
6. 倒取方向、目标量和放置位置组合泛化。

split 单位必须是 group，不是 frame/window；所有属于同一 source episode、operator/session、对象实例或采集 batch 的窗口不得跨 train/test。当前元数据不支持此 split，因此需新采集或补充可验证的 provenance。[M]

### 指标

接触层：

- one-to-one event precision/recall/F1；
- segment tIoU、onset/release MAE；
- side/site/role/mode macro-F1；
- censor 子集指标；
- presence/field ECE 与 Brier score；
- grammar violation rate、实际 code usage、per-query entropy。

任务层：

- overall success 及各 phase success；
- correct-first-contact；
- 实际转移液体质量/体积误差；
- spill mass/area、碰撞、掉落、容器倾覆；
- safety intervention/emergency stop；
- 闭环频率、action latency、cache 内存与 jitter。

所有成功率应按 episode/group 做 paired hierarchical bootstrap 或 cluster-aware binomial model，报告 95% CI；三个 seed 不是三个独立真实世界 trial。

### 公共 benchmark 的位置

LIBERO、RLBench、ManiSkill/RoboTwin 等可做补充的结构化接触/跨对象实验，但它们没有相同的液体物理、ego 视角和 asymmetric hand/gripper。VLAff、ST-VLA、CoA-VLA、BridgeACT 只有在代码、权重、输入和 action space 可匹配时才可作为数值 baseline；否则应作为概念前例，不应伪装成同表公平比较。

## Baselines 与消融

### 最小必要 baseline

1. **Original Q0 direct**：原 GWP-0.5，action 可直接读 reference，无 contact planner。
2. **Q12 direct-capacity**：保留等参数/等 FLOPs 的 Q12 支路但 action 仍直读 reference，planner 输出丢弃；用于排除多 token/算力收益。
3. **Aux-only**：预测 22D 作为辅助任务，但 action 仍直读 reference；区分监督正则化与严格瓶颈。
4. **Soft semantic strict**：action 只读连续 22D 激活；区分语义结构与有限化。
5. **Hard-ST strict**：主方法。
6. **Hard-detached strict**：与 hard-ST forward bit-identical，只切断 action loss→planner 梯度。
7. **Raw hidden strict**：同 Q、同宽度但传 planner hidden；衡量可解释压缩代价。
8. **Equal-bit unstructured code**：与 hard grammar 相同或更严的 alphabet/bit budget，但无字段语义；这是证明“语义而非单纯压缩”的关键。
9. **Oracle hard**：输入人工 contact segments，测试 grammar 是否足以控制任务。
10. **Shuffled / delayed / field-intervention**：同 batch 内打乱，或将 start/end 延迟；逐字段 counterfactual 改 side/site/mode/time。

所有变体必须固定数据和顺序、augmentation、action/visual timestep、visual/action noise、初始化共享部分、训练步数、精度和评估窗口，并保存 hash receipt。

### 现有实现状态

- `configs/contact_transport_ablation_matrix.json` 与 production config 已能运行 `soft_semantic`、`hard_st`、`hard_detached` 三种 transport，并记录多项 matched hashes。[I]
- original direct、Q12 direct、aux-only、raw-hidden、equal-bit unstructured、shuffled/delayed 目前是规范项或未实现。[S/M]
- oracle merge 的 model/trainer plumbing 已有，但受真实标签和科学闸门阻塞。[I/M]
- 当前 asymmetric25 配置明确 `scientific_gate_claimed=False`；这与本审查结论一致。[I]

## 3 seeds × 3 variants：应如何解释

没有一个三变体实验能同时回答“离散化是否有用、语义是否有用、strict route 是否优于 direct、ST 梯度是否有用”四个问题。因此必须先确定论文主张。

### 主因果 3×3（推荐）

固定 seeds `6666/7777/8888`，运行：

| Variant | 回答的问题 | 当前状态 |
|---|---|---|
| Q12 direct-capacity | strict semantic route 是否胜过等容量直接视觉策略？ | [S/M] 未实现 |
| Soft semantic strict | 收益来自语义结构，还是来自 hard finite transport？ | [I] transport 已有 |
| Hard-ST strict | 主方法 | [I] plumbing 已有、无收益证据 |

这是最小的“收益/有限化”主实验。必须报告 ID 与预注册 OOD，不能只挑 best checkpoint。

### 机制 3×3（现有 runner 更接近这个）

同样三 seeds，运行 soft、hard-ST、hard-detached。它只回答 transport 和梯度问题：

- soft vs hard-ST：有限 forward 的代价/收益；
- hard-ST vs hard-detached：action loss 穿过接口是否重要；
- soft vs hard-detached：混合了离散化和梯度两个因素，不宜单独解释。

这一组缺 direct control，**即使 hard-ST 最好也不能说瓶颈提高策略性能**。若算力只允许九次主训练，应先做推荐的主因果 3×3；hard-detached 可先做短程 gate，完整三 seed 作为机制附加实验。若论文要声称 ST 是核心，则总预算必须增加，不能用缺 direct 的 3×3 代替。

## 分阶段成功与停止门槛

### Gate 0：标签可重复

达到前述 κ/F1/tIoU/MAE 阈值后才能启动大训练。低于阈值的 pilot 仅说明协议需改，不是模型失败或收益。

### Gate 1：oracle sufficiency

相对 Q12 direct-capacity，oracle-hard 在预注册 OOD 上至少：

- overall task success +10 个百分点；且
- correct-first-contact +15 个百分点；且
- spill/safety 不退化。

若 oracle 都没有可重复的实质增益，说明 grammar 对任务不充分或 direct policy 已足够；停止大规模 predicted-contact 训练，先改 schema/任务。

### Gate 2：contact 可预测

- event F1 ≥ 0.80；
- onset/release median MAE ≤ 2 frames；
- predicted hard-ST 恢复至少 70% 的 oracle success gain；
- calibration 不劣于预注册阈值。

达不到则停止“从 ego 图像预测有限 grammar 可用于闭环控制”的主张。

### Gate 3：主因果收益

Hard-ST 相对 Q12 direct-capacity 在预注册 OOD overall success 至少 +5 个百分点，并且 group-clustered 95% CI 不跨 0；同时不能以 ID 明显退化、安全退化或 latency 失控换取。Hard-ST 相对 soft 的差异决定能否声称“有限化”有额外价值；若 soft≈hard，应把结论收缩为 semantic interface，不宣称 discrete transport 收益。

若 equal-bit unstructured code 匹配 hard-ST，说明主要收益来自容量正则化而非接触语义；若 shuffled/delayed/intervention 几乎不改变 action 或 phase outcome，说明策略忽略了 channel，停止因果解释。

### Gate 4：部署

- action-only latency 增量 ≤10%，闭环频率满足任务；
- safety intervention、spill、collision 均不劣于 direct；
- OmniPicker driver、canonical-to-device mapper、标定、限位/watchdog、急停和台架验证全部有 receipt。

当前仓库的 `configs/hardware/asymmetric25_hardware_manifest.template.json` 仍有空字段，`docs/asymmetric_action_schema.md` 也明确缺 OmniPicker driver/mapping/watchdog/bench test；因此真机部署为硬 No-Go。[S/M]

### 明确停止条件

出现任一条件即停止或收缩主张：

1. 标注协议修订一次后仍过不了 Gate 0；
2. oracle 过不了 Gate 1；
3. predicted hard 不能恢复 70% oracle gain；
4. direct 或 equal-bit unstructured 与 hard-ST 相当；
5. contact counterfactual 对动作/阶段结果无稳定影响；
6. OOD 元数据和 group split 在投稿截止前无法建立；
7. spill、安全或延迟显著恶化；
8. 目标硬件驱动、标定和 safety chain 不完整。

## 顶会投稿风险

### 1. 新颖性拥挤：高风险

BridgeACT、VLAff、ST-VLA、CoA-VLA、GeCCo 和离散表示工作分别占据了几乎所有组件。审稿人会追问“是不是把已有 contact/affordance token 塞进 WAM”。应以严格因果路由、精确 alphabet、可干预实验和 equal-bit control 回答，而不是堆更多术语。

### 2. 主张与代码语义不一致：高风险

“动作只能读 contact”按字面是假的，因为 action 还读 state、language/noisy action。必须写成“当前图像派生信息的唯一 action path”。通用 image-conditioning 接口还需 fail-closed 测试。

### 3. 数据规模与 OOD：极高风险

66 episodes 且缺 group metadata 无法支撑顶会 generalization。只有随机 window split 会被视为泄漏。需要预先定义跨对象、液体、操作者、session/day 和 embodiment shift。

### 4. 22D/Q12 不一定真是“小瓶颈”：中高风险

约 186.85 bit 的上界并不小，而且固定 query order 可能携带额外位置语义。必须做 Q、B rate–distortion 曲线、真实 code utilization 和等 bit 无语义对照，不能把 finite 直接等同 minimal information。

### 5. ST estimator 与优化解释：中风险

ST gradient 有偏；hard-detached 是必要诊断，但 forward 相同不代表训练轨迹只差一个因素，optimizer state 和 planner supervision也要严格匹配并记录。

### 6. censor/missingness 贡献可能被夸大：中风险

schema 有 censor 字段不等于模型做了 censored likelihood。若不改 loss，只能把它作为监督协议和显式状态，而非新的统计学习方法。

### 7. 液体状态部分不可观测：高风险

透明容器、遮挡、流量和液面无法稳定从单帧/current reference 得到。需要短历史、传感器或明确把不可见字段 mask；否则 contact grammar 会被迫编码猜测，oracle/predicted gap 可能不可闭合。

### 8. 硬件与任务过窄：高风险

单一倒水/倒取任务很难撑 general method。至少需要多个接触拓扑阶段、对象/液体 shift 和 supplemental benchmark；否则更适合作为 focused manipulation paper 或 benchmark/system paper。

### 9. 大 backbone 的混杂：中高风险

GWP-0.5 的参数量和视频先验会掩盖小模块贡献。缺少 Q12 direct、aux-only、raw-hidden、equal-bit code 时，任何收益都不能归因给可审计有限语义。

### 10. 复现与真实比较：中风险

自采液体数据、专有手爪和未公开标注会限制复现。若外部方法不能在同硬件/action space 重训，应诚实分为 conceptual comparison 与 executable baseline。

## 推荐的最小推进顺序

1. 完成标注 pilot 与独立 adjudication，只报告质量，不训练主模型。
2. 新采/补录完整 group metadata，并冻结 OOD split。
3. 实现 Q12 direct-capacity 和 equal-bit unstructured；先做 route invariance/fail-closed 测试。
4. 先跑 oracle sufficiency；未过 Gate 1 立即停止。
5. 再跑主因果 3×3；不要把现有 soft/hard-ST/hard-detached plumbing matrix 当收益实验。
6. 只有 Gate 0–3 均通过，才投入 OmniPicker driver、mapper、watchdog 和真机 Gate 4。

## 最终 Go / No-Go 表

| 项目 | 判定 | 理由 |
|---|---|---|
| 继续做标注与 route 审计 | **Go** | 低算力、能最快证伪 ontology 与旁路。 |
| 将“严格有限语义 WAM”作为研究假设 | **Conditional Go** | 交集 novelty 尚可守，但依赖 matched controls 与 intervention。 |
| 声称现有 plumbing 有性能收益 | **No-Go** | 没有训练、统计或真机结果。 |
| 现有三变体 matrix 支撑 causal benefit | **No-Go** | 缺 direct-capacity 与无语义 control。 |
| 使用 66 episodes 声称 production OOD 泛化 | **No-Go** | group metadata 不足。 |
| 将 hand status/proxy 当 contact GT | **No-Go** | 语义和 missingness 审计已否定。 |
| 立即做大模型三 seed 训练 | **No-Go** | Gate 0/1 均未通过。 |
| 当前 G1+RH56DFTP+OmniPicker 真机部署 | **No-Go** | driver、mapping、标定、watchdog、台架证据不完整。 |
| 顶会主会 ready | **No-Go** | 数据、因果 baseline、oracle、OOD 和硬件证据均缺失。 |

最终建议不是“先把模型跑起来看看”，而是先用标签一致性、oracle sufficiency 和 equal-capacity route tests 砍掉最可能失败的假设。只有 contact grammar 本身足以控制、预测器能逼近 oracle、且 hard semantic route 在严格 matched OOD 下优于 direct/soft/unstructured，才值得把它写成方法贡献。
