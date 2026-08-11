# Idea I 调研：Causal Critical-Window WAM

> 日期：2026-08-12
> 结论：**独立论文主张 No-Go；作为 SM-FET 的因果输入侧研究 Conditional Go**
> 推荐结构：**Causal Event-Triggered Window Memory（CET-WM）**，即在线事件变化点触发的短窗环形记忆
> idea-discussion `idea_id`：`a0da9ceb-7748-4ef5-910c-e57b7cf041dd`
> 主讨论 `discussion_id`：`08578342-f03a-4e9d-a454-0fe1f373b2ef`
> 精确检索追问 `discussion_id`：`a32a9415-7fc6-464d-9bc3-97ee10c0ccf8`

## 0. 一页结论

原问题“工具操作不应关注全程，也不能只看单帧；能否选择少量关键短窗口，并证明它们对机器人动作足够”包含一个正确直觉，但需要改写后才成立：

1. 当前仓库里的 action 分支并没有读取整段稠密视频。默认 48-frame horizon 只取 `[0, 12, 24, 36, 48]` 五个视觉时刻，而训练器把第一帧当 reference、其余帧当未来视觉生成目标；严格路由下，action 只接收由当前 reference、状态和文本产生的有限语义包。因此 Idea I 不是“从当前 action 输入里删掉全视频”，而是**给当前近单帧系统增加一个因果历史短窗记忆**。
2. “关键事件稀疏选择 + embodied world model”已被 **SKIP** 高度占据；接触关键帧、固定短 clip、keypose、时间 token pruning 也分别与 VLAff、BridgeACT、ST-VLA、VLA-IAP/LightVLA 重叠。不能主张 first sparse WAM、first contact keyframe、first short-window affordance 或 first temporal pruning。
3. SM-FET 已拥有事件时钟、事件区间、接触/效果标签、有限 action packet 和人机 partial alignment。Idea I 若再建一套 `pre-contact/contact/regrasp/effect/release` 事件语法，会形成第二套相互竞争的时钟和接口。
4. 尚可保留的研究缝隙是：**短窗口而非单关键帧、严格在线因果历史而非训练期全轨迹选帧、直接针对动作决策的 keep/delete/exchange 干预证据，而不是只验证视频重建质量**。
5. 三个候选模型中，推荐模型 B：在线变化点检测器只看 `x_{≤t}, s_{≤t}`，从环形缓存截取 0.4 s 左右的窗口，最多保留 K 个；原始窗口 token 只进入现有 contact/SM-FET planner，action 仍只看有限 hard packet。模型 A（端到端 budgeted selector）和模型 C（全局 temporal coreset）保留为强基线，不应与 B 混合后再归因。

最终建议是：**No-Go 于“Causal Critical-Window WAM”作为独立架构/首创论文；Conditional Go 于“CET-WM：SM-FET 的因果短窗证据前端 + operational ε-sufficiency benchmark”。** 若 66-episode pilot 无法同时满足保留充分、删除敏感、交换敏感三个条件，就停止该方向并把它降级为工程采样器。

## 1. 先修正问题定义

### 1.1 当前 WAM 不是 action 读取全视频

仓库基线的关键事实如下。

| 位置 | 当前行为 | 对 Idea I 的含义 |
|---|---|---|
| `configs/giga_world_policy_0_5_sonic_g1_finetune.py:17,24-34` | `num_frames=48`，`image_frame_offsets=[0,12,24,36,48]` | 20 Hz 下覆盖约 2.4 s，但仅五个视觉时刻 |
| `world_action_model/transforms/wa_transforms_lerobot_pretrain.py:628-641` | 组装 `input_images` 和 reference mask | 数据层还没有 K 个历史短窗及其有效 mask |
| `world_action_model/trainer/wa_casual_trainer_mot.py:922-942` | 第一帧为 reference，后续视觉帧成为 noisy/future latents | 后四帧主要是生成目标，不是 action 可自由读取的全视频证据 |
| `world_action_model/trainer/wa_casual_trainer_mot.py:989-1000` | 将 reference/future visual states 传入模型 | 新历史窗口必须被显式标成 causal reference，而不能混入 future target |
| `world_action_model/models/transformer_wa_casual_mot.py:1253-1285` | contact planner 读取状态、reference visual、文本 | 最自然的 CET-WM 插入点 |
| `world_action_model/models/transformer_wa_casual_mot.py:1335-1403` | plan/transport 走硬语义路由 | 与 SM-FET 的有限接口一致 |
| `world_action_model/models/transformer_wa_casual_mot.py:1405-1445` | 仅把 22D packet 投影到 action width，planner hidden 不直通 | 可阻止 selector/planner 成为高容量旁路 |
| `world_action_model/models/transformer_wa_casual_mot.py:1809-1822` | 构造 MoT attention mask | 需把历史窗声明为过去/当前 token，并继续屏蔽未来视觉 |
| `world_action_model/models/transformer_wa_casual_mot.py:2296-2335,2359-2499` | action-only cache 的校验与允许 packet | cache 不得保存 selector hidden/logits 或未来相关特征 |
| `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py:460-466` | 明确 `state/ref/text planner -> hard 22D -> action`，无 direct reference/raw hidden 路由 | 主实验应保留这条严格路由；direct-window 只能作为另一条标注清楚的 baseline |

因此，正确的研究问题是：

> 在时刻 t，只使用可因果获得的历史 `H_t=(x_{≤t},s_{≤t})`，能否用固定预算的少量短窗口代替更长历史，向现有 planner 提供足够的操作阶段、接触变化和效果证据，同时不让窗口索引或 selector hidden 偷渡动作标签？

### 1.2 “足够”的可检验定义

这里不能从单一数据集和单一模型证明信息论上的充分统计量或真实因果 Markov blanket。建议只使用明确限定的 **operational ε-sufficiency**：

\[
S_t=g(H_t),\quad |S_t|\le K,\quad \sum_{w\in S_t}|w|\le B,
\]

\[
R(\pi_{S};D_{test})-R(\pi_{H};D_{test})\le \epsilon,
\]

其中 `π_S` 和 `π_H` 在同一模型容量、训练数据、动作损失和 held-out group 分布上比较；`S_t` 还必须通过删除和交换检验。若只有 keep-only 性能接近 full，不能称为关键或因果：模型可能忽略视觉，selector 也可能通过索引编码答案。

推荐把“充分”的经验签名定义为四项同时成立：

- **retain/keep**：只保留选中窗口时，动作风险与 full-history 的差距在 ε 内；
- **delete/comprehensiveness**：从 full 中删除选中窗口的损害显著大于删除等时长随机窗口；
- **exchange/order**：替换窗口内容或打乱窗口顺序会在相应操作阶段造成局部、可预期的动作退化；
- **no side channel**：只用窗口索引、长度、mask 或 selector 分数不能预测任务/动作，或仅高于 chance 不超过门槛。

这比“注意力落在接触附近”强，但仍只能表述为**对指定模型类和评测分布的操作性因果证据**。

## 2. Zotero-first 检索记录

先按 `idea-discussion` 工作流同步并检索了个人 Zotero 库，再追问精确方向，最后才查原始论文页面。

- 同步规模：100 条 Zotero 条目；两轮讨论都完成 memory 更新。
- 初轮主要命中 RT-1（`EK59GCPY`）、RT-2（`X9TZGLQK`）、PaliGemma（`IYNMA8QM`）、FAST（`CBLGABU2`）、π0（`LFI6T4BX`）。这些条目可作为 VLA/action-token 背景，但不直接覆盖“因果短窗选择”。
- 精确追问仍未在已同步索引中找到 SKIP、BridgeACT、VLAff、ST-VLA 等关键冲突。此结果只表示**当前同步索引未命中**，不能当作 Zotero 中绝对不存在的证明。
- 因此本 memo 的具体 novelty 审计以原论文/官方项目页为主，并把个人库条目只作为背景证据。

## 3. 原始工作版图与 novelty 边界

### 3.1 与题目最接近：稀疏关键帧和短 clip

| 工作 | 已经做了什么 | 对本 Idea 的限制 |
|---|---|---|
| [SKIP](https://arxiv.org/html/2606.00664v1) | 面向 embodied world models 的 event-preserving sparse-to-dense pipeline；selector 融合 optical flow、冻结 DINO、可选 proprio，用 KTS 分段并显式覆盖 gripper transitions；训练时看完整稠密轨迹，推理时先由 Wan2.2 生成 K 个 keyframes，再 action-conditioned interpolation | 几乎占据“稀疏事件关键帧 + world model”。本 Idea 只能强调短窗、在线历史和动作充分性干预，不能主张首创稀疏 WAM |
| [SWEET](https://arxiv.org/abs/2605.19319) | 由少量任务相关视觉关键帧构成稀疏计划，再执行 goal-conditioned actions | “少量关键视觉证据驱动动作”已有直接先例 |
| [BridgeACT](https://arxiv.org/html/2604.23249v2) | 将原始人类视频切成固定 1.5 s clip，每 clip 下采样为 4 帧，stride 0.5 s；提取工具/目标 affordance 与 3D point flow 并闭环执行 | 已有“人类视频短窗口到机器人执行”的强基线；不能只用 fixed short clips 作为新意 |
| [VLAff](https://arxiv.org/html/2608.05215v1) | 围绕交互时间戳采样，VLM 提取 contact keyframes，并跟踪 contact region；SfM 表达 pre/post-contact motion | `pre-contact/contact/post-contact` 视觉选择和运动 affordance 都已有覆盖 |
| [ST-VLA](https://arxiv.org/html/2603.13788v1) | 以初末 contact point、对象语义、相对构型和 keypose 组织时空 affordance，支持部分观察下的剩余轨迹预测 | 不能把 contact endpoints/keypose/partial observation 当主 novelty |

SKIP 是最强冲突。它还报告了 K 的 sweep（9/17/33/41/57，主配置 41）、约 4.16× 生成加速，以及 100% synthetic policy data 在仿真/真实上的有限性能下降。因此，“用较少视频证据降低 world-model 成本”本身不足以成稿。真正需要超过它的是：selector 推理时对机器人 history 严格因果、输出窗口而非孤立帧、最终度量是动作干预而非重建质量。

### 3.2 Sparse video、temporal proposal 与 adaptive computation

- [AdaFrame, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html) 已用带全局记忆的序列策略自适应选帧和停止，并以少量帧接近全视频分类。
- [SCSampler, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Korbar_SCSampler_Sampling_Salient_Clips_From_Video_for_Efficient_Action_Recognition_ICCV_2019_paper.html) 的轻量 sampler 扫描长视频、选择 salient clips，再交给重模型；这是模型 A 的直接结构先例。
- [Flexible Frame Selection, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html) 进一步把动态帧预算用于视频推理。
- [FrameExit, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ghodrati_FrameExit_Conditional_Early_Exiting_for_Efficient_Video_Recognition_CVPR_2021_paper.html) 展示了条件早退；[TokenLearner, NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html) 和 [semantic-aware temporal token accumulation, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ding_Prune_Spatio-temporal_Tokens_by_Semantic-aware_Temporal_Accumulation_ICCV_2023_paper.html) 表明自适应计算与 token 压缩已有成熟谱系。
- VLA 内部，[VLA-IAP](https://arxiv.org/abs/2603.22991) 做 training-free interaction-aligned visual token pruning；[LightVLA](https://arxiv.org/abs/2509.12594) 用动态 query/Gumbel 学习 differentiable token pruning。故必须报告 selector 扫描成本，不能只报告送入 WAM 的 token 数。
- Temporal coreset 也不是空白：[Deep Reinforcement Learning for Unsupervised Video Summarization, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/12255) 直接以 diversity 和 representativeness 奖励选摘要；[Unsupervised Video Summarization with Adversarial LSTM, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.html) 用重建/对抗目标学帧重要性。模型 C 的 novelty 不能只是 facility-location、覆盖或多样性，而必须来自机器人动作干预与合法的 deployment visibility。

### 3.3 Event boundary、接触变化与 action anticipation

- [GEBD, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.html) 建立了通用事件边界检测任务。
- [Online GEBD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Online_Generic_Event_Boundary_Detection_ICCV_2025_paper.html) 明确限制只用 `frames ≤ t`，以 anticipation/prediction error 检测边界，是推荐模型 B 的直接方法学基线。
- [Fast and Unsupervised Action Boundary Detection, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Fast_and_Unsupervised_Action_Boundary_Detection_for_Action_Segmentation_CVPR_2022_paper.html) 用平滑后的相似度变化和 NMS 做无监督变化点，也提供在线扩展思路。
- [Ego4D Hands & Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/) 已使用 PRE、CONTACT、PNR、POST 和 state-changing object 标注；这使 `pre-contact/contact/effect` 成为合理标签，却也说明事件词表不是 novelty。
- [Joint Hand Motion and Interaction Hotspots, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Joint_Hand_Motion_and_Interaction_Hotspots_Prediction_From_Egocentric_Videos_CVPR_2022_paper.html) 以及 [Next-Active-Object Guided Anticipation, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Thakur_Leveraging_Next-Active_Objects_for_Context-Aware_Anticipation_in_Egocentric_Videos_WACV_2024_paper.html) 已覆盖 hand/contact hotspot、下一交互对象、未来动作和 time-to-contact 的 anticipation。

因此推荐使用“event-triggered memory”而不是“新事件模型”：边界检测负责何时写入窗口，事件语义仍由 SM-FET 负责。

### 3.4 Minimal sufficient evidence 与 rationale 的警告

- [Visual Representations: Defining Properties and Deep Approximations](https://arxiv.org/abs/1411.7676) 给出视觉表征中 minimal sufficient statistic 的理论背景。
- [L2X](https://proceedings.mlr.press/v80/chen18j.html) 用互信息学习实例级特征子集；[Rationalizing Neural Predictions](https://aclanthology.org/D16-1011/) 引入短且连续的 rationale；[ERASER](https://aclanthology.org/2020.acl-main.408/) 系统化 sufficiency/comprehensiveness。
- 但 [Goodhart's Law Applies to NLP's Explanation Benchmarks](https://aclanthology.org/2024.findings-eacl.88/) 指出 sufficiency/comprehensiveness 可因 OOD 删除输入或 rationale 编码标签而虚高。

对应到视频，简单把未选帧置黑会制造训练分布外输入；窗口索引又可能成为 covert label channel。故主实验必须同时使用：

1. 训练时覆盖 keep/delete mask 分布的共享模型，用于同一参数下的干预；
2. 分别重训 full、keep-only、delete-only 输入模型，用于表示充分性；
3. 对等时长随机/均匀窗口进行同分布控制；
4. 对窗口索引/mask 单独训练 probe。

### 3.5 Video imitation 与跨 embodiment 对齐

- [Temporal Cycle-Consistency, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html) 学习跨视频逐帧阶段对齐。
- [Time-Contrastive Networks](https://arxiv.org/abs/1704.06888) 用多视角自监督表示支持 human-to-robot imitation。
- [XSkill](https://proceedings.mlr.press/v229/xu23a.html) 从无标签人/机器人视频发现技能原型并配合 diffusion policy；[MimicPlay](https://proceedings.mlr.press/v229/wang23a.html) 从人类 play 构建层级 latent plan；[EgoMimic](https://arxiv.org/abs/2410.24221) 和 [UniSkill](https://proceedings.mlr.press/v305/kim25d.html) 继续推进跨 embodiment 学习。

因此 human/robot pairing 应只声称**关键窗口在事件/功能层面的部分对齐**，不能声称首次跨 embodiment 视频模仿，也不能以 raw timestamp 对齐替代功能对应。

## 4. 指定冲突审计

| 冲突对象 | 与 Idea I 的最强重叠 | 必须让出的主张 | 尚可保留的边界 |
|---|---|---|---|
| **SM-FET** | 已定义最多 8 个宏事件、事件区间、接触/效果、有限 active+next packet、人机 partial alignment | 不能再建第二套事件语法、第二个 action packet 或第二套 alignment | Idea I 只做 raw temporal evidence 的因果选择前端，服务现有 planner/SM-FET |
| **SKIP** | embodied WM 的事件保留稀疏关键帧、gripper transition coverage、稀疏到稠密生成 | 不能声称 first sparse/event-preserving WAM 或“少帧足够重建” | 短窗口、在线 robot history、动作 keep/delete/exchange |
| **VLAff** | contact keyframe、contact region、pre/post-contact motion 与 affordance 输出 | 不能声称首次围绕 contact 截取证据或首次把 pre/post motion 用于机器人动作 | 固定预算窗口的动作充分性和反事实交换 |
| **BridgeACT** | 人类视频固定 1.5 s/4 帧 clips，工具/目标角色和 3D flow 到闭环动作 | 不能声称短 clips 或人类视频工具操作迁移本身新颖 | 在线机器人 history 的 causal selector；事件窗口而非均匀 clip |
| **ST-VLA** | contact endpoints、keypose、对象语义、partial observation | 不能声称 contact/keypose/剩余轨迹条件首创 | 变化点触发和有限短窗记忆的因果审计 |
| **VLA-IAP / LightVLA** | 交互相关或可学习视觉 token pruning | 不能把 token 数下降本身当贡献 | 时间窗口层级、selector side-channel audit、动作干预 |

最重要的边界句应写成：

> CET-WM 不提出新的事件 grammar，也不提出新的 sparse world-model reconstruction pipeline；它研究在严格在线部署约束下，短时原始证据窗口对既有有限事件 planner 是否达到可干预验证的动作充分性。

## 5. 三个互斥候选模型

“互斥”指每次主实验只能由一种机制决定 action-visible window set；不能先用变化点切分、再用 coreset 评分、最后再用 learned selector 精排，然后把增益归因给任一个模块。

| 模型 | selector 可见范围 | 输出 | 训练/推理一致性 | 主要风险 | 建议角色 |
|---|---|---|---|---|---|
| A. Budgeted Interval Selector + WAM | robot rollout 只看 `≤t`；若部署时本来就提供完整 source demo，则可看完整 source demo，但绝不能看 target robot future | 固定 K、总时长 B 的硬区间 | 必须以最终部署可见性训练；full-video teacher 只能产伪标签 | 与 AdaFrame/SCSampler/SKIP 重叠，联合训练易成为旁路 | 强 learned baseline |
| B. Event-Change-Point Detector + Ring Memory | 严格 `x_{≤t},s_{≤t}`；允许固定 0–2 帧检测延迟 | 边界触发的 pre/post 短窗，保留最近 K 个 | 最强，训练推理同一在线状态机 | 边界不等于语义关键；需证明动作价值 | **推荐主模型** |
| C. Learned Temporal Coreset | source demo 若部署也完整可见，则可全局看；robot rollout 不能全局看。训练期 posterior 可全局看，但部署必须蒸馏 causal prior | coverage/diversity 最大的 K 个窗口 | posterior 与 prior 容易产生 deployment gap | hindsight、全局扫描成本、facility-location 偏代表性而非因果性 | 非因果 oracle/upper-bound baseline |

### 5.1 模型 A：Budgeted Interval Selector + WAM

轻量 selector 在候选网格上预测 K 个离散起点，每个窗口长度固定为 δ，满足 `Kδ=B`。通过 straight-through/Gumbel 或 policy gradient 学习动作风险，但主审计版先冻结 selector、再训练 action model，并 stop-gradient。输出接口只能是整数起点选择和对应 raw frames；selector hidden、soft score、连续 offset、预测 action 都不得传给 WAM。

该模型有最强拟合能力，也最容易把任务/动作编码进选择模式。因此只有当 index-only probe 失败、交换测试成立且端到端扫描成本仍低时才可信。

### 5.2 模型 B：Causal Event-Triggered Window Memory（推荐）

对每个时刻维护有限环形 buffer。轻量 detector 使用冻结低分辨率视觉特征、光流/预测误差和可选 proprio/gripper proxy 计算变化分数；分数经迟滞阈值和 NMS 触发事件。触发后，从 buffer 截取 pre-boundary 上下文，并允许固定的 0–2 帧 post-boundary 确认延迟，形成约 0.4 s 短窗。只保留最近 K 个有效窗；其中一个窗口固定以当前时刻结束，保证当前观测也计入 K/B，而不是得到未计费的额外帧。

这里必须区分**回溯可见**与**动作前可用**：在 contact 被观察到后从 ring buffer 回取 pre-contact 片段，只能帮助 contact 之后的 acquire/regrasp/effect 决策，不能倒过来宣称它帮助了 contact 前的 approach。若要证明 pre-contact 动作价值，必须用只看 `≤t` 的 TTC/interaction anticipator 在接触发生前触发窗口，并按当时真实可用的信息评测。

推荐把它命名为 memory 而非 selector，原因是其部署语义清晰：它不回看未来寻找最佳片段，而是在事件发生时写入可遗忘的有限证据。其新意不在变化点检测器本身，而在**与现有有限事件 planner 组合后，用动作级干预检验窗口是否充分**。

### 5.3 模型 C：Learned Temporal Coreset

使用 facility-location/representativeness、temporal diversity 与 action-distillation loss，从完整 source demonstration 选择 K 个窗口。对于完整人类示范作为推理输入的场景，全局 coreset 是合法的；对于在线机器人 observation，它只能作为训练期 oracle/posterior，再蒸馏到 causal prior。最终 action model 必须在 prior 真实产生的窗口上重训和测试，不能训练时用 posterior、测试时静默换成 prior。

它适合测“整段示范能被压成什么”，不适合作为在线因果主模型。

## 6. 推荐结构与容量边界

推荐结构为：

```text
causal RGB/state history (<= t)
          |
          v
frozen low-res feature / flow / prediction error
          |
          v
causal change-point trigger ----> fixed-size ring buffer
                                      |
                                      v
                    K raw short windows + valid mask
                                      |
                                      v
                 existing visual encoder (ref-like tokens)
                                      |
                                      v
          existing contact / SM-FET planner
                                      |
                                      v
                  finite hard semantic packet
                                      |
                                      v
                         existing action head
```

### 6.1 固定预算

- selector/change detector：不超过 0.5M 参数，建议 2-layer temporal head、width 128，输入冻结的 128D 低分辨率特征；
- action-visible 视觉：主配置 `K=3`、每窗 `δ=0.4 s`，20 Hz 下原始 8 帧/窗；重视觉编码器固定下采样到 3 帧/窗，共 9 encoded frames；其中一个窗口以当前时刻结束；
- 上限配置 `K=5`、`B=2.0 s`、15 encoded frames；
- 不提供 K/B 之外的免费当前帧；如 ablation 需要额外 current token，必须同时报告“selection budget”和“含 current 的 total encoded-frame budget”，并向所有 baselines 提供相同 token；
- selector 扫描 FLOPs、解码帧数、延迟和缓存占用都计入端到端成本，不能只报告送入主 WAM 的 9/15 帧。

### 6.2 防止 selector 成为高容量旁路

1. 主实验固定 K 和 δ，不允许数量/长度携带标签；
2. WAM 只收 raw selected frames、binary valid mask 和窗口内重置的位置编码；不收 hidden、logits、score、action proposal、pair ID；
3. 默认只暴露窗口的相对先后次序，不暴露绝对 episode time；若必须给时间，固定量化并向 random/uniform baseline 提供完全相同的信息；
4. 注意窗口索引本身具有约 `log2(C(N,K))` bit 的潜在信道。必须报告 index entropy，并用 indices-only probe 预测任务/动作；
5. selector 在主因果审计中先冻结，action 训练 stop-gradient；联合 Gumbel/REINFORCE 仅作为 exploratory ablation；
6. training-only full-video teacher 只能产生边界/选择标签，不能把 teacher hidden 传给 student；最终 action model 必须在 causal student 的真实选择上训练；
7. 保存 selection manifest、raw-frame checksum 与代码 receipt，以审计 future-frame 泄漏；
8. cache 只保存选中窗口 token 或最终 hard packet；窗口集合变化时失效，不缓存 selector score/hidden/未来帧派生量。

## 7. WAM 插入点与兼容路线

### 7.1 数据与 transform

在数据层为每个 action replan time 产生 causal history candidate frames 和不可变 selection manifest。窗口划分必须在 episode/group split 后进行，避免同一 episode 的相邻窗口跨 train/test。

建议 transform 产出：

```text
window_images:     [K, F, C, H, W]
window_valid_mask: [K, F]
window_order:      [K]       # only chronological ordinal, not absolute time
selection_receipt: metadata  # logging only, never input to action model
```

同一窗口内和同一训练样本的各窗口应共享几何 augmentation，以免增广差异本身暴露选择类别。窗口视觉作为 reference-like causal tokens 编码，绝不能拼到 future noisy latents。

### 7.2 模型路径

主兼容路线是在 `world_action_model/models/transformer_wa_casual_mot.py:1253-1285` 的 `_plan_contact_logits` 前，将选中窗口的 ref-like tokens 聚合为 planner evidence；随后沿 `:1335-1445` 的既有 hard semantic packet 到 action。这样 action 仍无法访问 raw planner hidden，保持 SM-FET/有限接口主张。

MoT mask 在 `:1809-1822` 把 window tokens 标为过去或当前，只允许 planner 读取；严格 route 下 action 不直接 attend windows。action-only cache 在 `:2296-2499` 仅允许更新后的 hard packet 或具备 receipt 的窗口 token，并在 observation/window set 变化时失效。

另设一个 **direct-window baseline**：action 对同样的 K/F/token 预算直接 attend selected references。它回答“硬语义 packet 是否丢信息”，但与 strict finite semantic route 不兼容，必须单独命名、单独报告，不能与主模型混成同一结构。

## 8. 标签与无标注训练

### 8.1 最小标签词表

复用而不改写 SM-FET 语义：

```text
pre_contact
contact_acquire
regrasp
effect_onset
release
NULL
unknown_or_censored
```

每个标签记录 `(lower_frame, upper_frame, confidence, provenance)`，以及可观察的 `hand/tool/target` 功能角色和 effect/outcome。当前 `hand_status` 只能作为 gripper proxy，不能直接标成物理 contact/release 真值；遮挡时优先 `unknown_or_censored`，不要强补标签。

### 8.2 有标签训练

- event boundary BCE/focal loss + interval IoU/duration loss；
- 事件类型只作为辅助头，selector 的主输出仍是窗口；
- oracle window 由人工区间产生，用于测可达到的压缩上限；
- 训练 causal detector 时，输入截断到 t，标签允许固定 0–2 帧检测延迟。

### 8.3 无标注/弱标注训练

- frozen feature prediction error、光流峰值、hand-object distance 变化、gripper sign change 构造 proposal；
- temporal contrastive/cycle consistency 学阶段稳定表征；
- facility-location/diversity 只用于模型 C，不混入模型 B；
- 用 full-video teacher 产生软边界时，只蒸馏 boundary probability，不蒸馏 future-dependent hidden；
- action prediction/reconstruction loss 可用于排序，但不能成为唯一监督，否则 selector 可能选择动作可预测却非语义关键的背景 cue。

## 9. 预算实验与 baselines

### 9.1 两组正交 sweep

为了分离“窗口数量”和“总观看时长”，至少做两组：

1. 固定总预算 `B=1.2 s`，`K={1,3,5}`，相应改变单窗长度；
2. 固定 `δ=0.4 s`，`K={1,3,5}`，总预算为 `{0.4,1.2,2.0}s`。

所有 sparse 方法在相同 encoded-frame 数、分辨率、动作模型容量上比较。full dense/history 是高成本上界，不伪装成同成本 baseline。对于 source demonstration 可另比较整段约 15 s；对于在线 robot history，建议统一 full-history cap 为 4.8 s，以匹配 SM-FET lookahead/历史审计尺度，但绝不读取 `>t`。

### 9.2 必须包含的 baselines

- single current frame；
- single oracle contact frame；
- full causal history（或完整 source demo，仅当部署也提供完整 demo）；
- uniform K windows；
- random K windows，至少 20 次选择重复并报告分布；
- contact heuristic：hand-object distance、gripper sign、flow peak；名称明确为 proxy；
- SKIP-style selector：flow + frozen DINO + optional proprio、KTS、gripper-transition coverage；
- unsupervised/online change point（ABD/Online-GEBD style）；
- learned temporal coreset；
- oracle annotated event windows；
- 可选 spatial-token controls：VLA-IAP/LightVLA，用来判断收益来自时间选择还是纯 token 减少。

## 10. 因果检验

### 10.1 Keep / retain

仅输入 selected windows，其他视觉以训练中见过的 mask 机制移除。比较 full-history 相同 action head 的风险差，并另行重训 keep-only policy。主问题是“所选窗口是否足够”，不是“已有 full 模型是否能容忍遮挡”。

### 10.2 Delete / comprehensiveness

从 full-history 中删除 selected windows，并与删除等时长、等位置分布的 random/uniform/contact-heuristic windows 比较。若所有删除都同样损伤，说明退化来自 mask/OOD；若删除 selected 没有额外损伤，则所选窗口不“关键”。

### 10.3 Exchange / content intervention

固定窗口数、长度、mask、相对位置和 token budget，只交换 raw content：

- 同任务/同对象但错误阶段；
- 同阶段/同对象但错误 effect；
- 外观接近但跨任务；
- human/robot 功能角色匹配但 embodiment 不同；
- 同一多重集只交换 chronological order。

预期结果应具有**局部性**：交换 contact 窗主要损伤 first-contact/acquire，交换 effect/release 窗主要损伤 stop/release，而不是所有动作随机崩溃。若只要换视频就全面失效，更可能是 identity/domain shift。

### 10.4 负控与统计

- 对窗口整体时间平移 `±δ`、倒序、label shuffle；
- current/future checksum，禁止 selector 在 t 时使用 `>t`；
- indices-only、mask-only、background-only probe；
- matched intervention distribution：keep/delete 模式在训练中显式出现；
- 以 episode/operator/object/source group cluster bootstrap，报告 95% CI，不以 frame 作为独立样本；
- 多随机种子时报告 selection Jaccard 与性能共同变化，避免只展示最佳 seed。

## 11. 人机 pair 的关键窗 partial alignment

只对同一任务/工具功能/effect 的 pair 建立匹配。使用事件级部分单调对齐，允许 `NULL`，禁止用 raw frame index 或绝对秒数直接做 tIoU，因为 embodiment、速度、相机和手的自由度都不同。

对齐对象应是：

- 事件类别与先后关系；
- hand/tool/target 的功能角色，而非 human-left 对 robot-left 的字面对应；
- contact acquire、effect onset、release/verify 的可观察 anchor；
- 每个事件对应的 selected window 是否覆盖 anchor。

建议指标：matched-event selection recall/F1、`NULL` AUPRC、pair selection agreement、monotonic crossing rate、effect consistency、normalized phase-center error、window retrieval R@K。

当前 66 个 Ego pour episode 与旧 88 个 dual-Inspire grab episode 任务不匹配，不能构成正 pair，也不能验证跨 embodiment transfer。pair 结论必须等未来收集同任务人/机器人片段；不允许为了完整表格强制匹配。

## 12. 指标

| 类别 | 指标 |
|---|---|
| Selection | event coverage/recall、window tIoU、boundary MAE、duration/phase coverage、冗余度、index entropy、seed Jaccard、NULL precision/AUPRC |
| Action | flow MSE/NRMSE、阶段成功率、正确 first contact、regrasp recovery、正确 release/stop、事件顺序违规、spill/drop/collision、group-held-out success |
| Causal | retain gap、selected-vs-random delete harm、exchange harm、order-swap harm、B/K dose-response、阶段局部性、indices/mask/pair-ID probe |
| Efficiency | encoded frames/tokens、selector scan decode/FLOPs/latency、WAM FLOPs/latency/memory、端到端 speedup、cache invalidation rate |
| Calibration | boundary/event ECE、Brier score、unknown/censor coverage |

## 13. 66-episode 低成本 pilot

现有可用规模是 66 个 Ego episode、20,013 帧、20 Hz，平均约 15.2 s/episode。pilot 只做 CPU/已有特征可完成的结构和离线代理验证，不训练 5B/Wan 模型、不运行 GPU。

### P0：标注可行性

1. 先选 12 个 episode 双人独立标注上述事件区间；
2. 其余 54 个做单标/弱标签，并保留 `unknown_or_censored`；
3. 在切任何窗口前，按 operator/session/object/source group 分 train/val/test；若元数据缺失，则不声称 OOD；
4. 统计 anchor 可见性、事件频率、遮挡和 regrasp/NULL 比例。

### P1：冻结特征和 oracle 上限

1. 以 5 Hz 候选网格扫描，每个约 15 s episode 约 75 个候选时刻；特征缓存在项目所在挂载盘；
2. 先比较 full、single-current、single-contact、uniform、random、contact heuristic、oracle-window；
3. 每 8 帧（0.4 s）建立一个 causal replan 样本，预测未来 48 帧动作 proxy/contact/effect；
4. 动作 proxy 只用于结构筛选，不能被表述成机器人闭环成功。

### P2：轻量 causal detector

只有 P1 的 oracle `K=3/B=1.2s` 通过充分性门槛后，才训练 ≤0.5M 的变化点 head 和 ≤2M 的统一 temporal proxy/action aggregator。做 3 个 episode-level seeds 和 cluster bootstrap；各分支使用相同冻结特征、输入帧数和输出头。

### P3：干预和 side-channel audit

在 held-out group 上一次性执行 keep/delete/exchange/order/index-probe。预注册主要终点，避免根据结果临时挑某个阶段或 K。selection manifest 与 checksum 写入项目实验目录；pilot 完成后不产生临时测试脚本。

### Pilot 能回答与不能回答

能回答：事件区间能否可靠标注、少量 oracle 窗是否保留离线动作 proxy、online detector 是否接近 oracle、选择索引是否泄漏、keep/delete/exchange 是否呈现一致签名。

不能回答：真实物理 contact/effect、机器人闭环成功、安全性、跨 embodiment 同任务迁移。现有 Ego `hand_status` 不是物理真值，且 66 Ego 与 88 robot 数据没有任务正配对。

## 14. 停止门槛

这些门槛应在查看最终 test 结果前固定。

### G0：标签门槛

- 12 个双标 episode 上 event-kind `κ ≥ 0.70`；
- contact/release `κ ≥ 0.60`；
- boundary median absolute error `≤2 frames`；
- 至少 80% 样本有一个可观察 anchor；
- unknown/censored critical effects 不超过 40%。

任一主标签明显低于门槛时，停止 supervised critical-window claim，退回纯无监督变化点分析。

### G1：Oracle compression 门槛

主预算 `K=3, B=1.2s`：

- 关键事件 coverage `≥0.85`；
- action NRMSE `≤1.05× full`，或主要 phase/action 指标相对 full 下降 `≤3 pp`；
- 删除 selected 的损害相对 random 至少大 10%，或绝对大 `≥5 pp`，且 episode-cluster 95% CI 下界 `>0`；
- content exchange 或 order swap 造成至少 10% 相对损害，并呈阶段局部性。

若 single-current 与 oracle-window 相差 `≤2 pp`，则没有必要研究关键短窗；若 full 比 `K=5/B=2.0s` 仍高 `>5 pp`，则“少量窗口足够”失败。

### G2：预测 selector 门槛

- 恢复 oracle 相对 baseline 增益的至少 80%；
- event coverage 或 causal specificity 比 uniform/random/contact heuristic 高 `≥5 pp`；
- 主要 action 指标不比 full 低超过 `3 pp`；
- selector 扫描成本 `≤` dense visual encoder 的 10%，总 action-visible token `≤` full 的 25–33%；
- indices-only/task/action probe `≤ chance+2 pp`；
- 三 seed selection Jaccard 建议 `≥0.60`。

任何 side-channel probe 越界都优先判失败，而不是以 action 指标抵消。

### G3：人机 pair 门槛

未来至少收集 12 human + 12 robot 同任务 clips；`NULL AUPRC≥0.80`，event-pair F1 比最佳 TCC/DTW baseline 高 `≥5 pp`，crossing `<1%`，effect consistency `≥0.80`。当前数据不满足输入条件，因此本阶段 pair claim 自动 **No-Go**。

### G4：真正 policy 门槛

只有未来机器人闭环实验中，predicted windows 恢复 oracle 增益的 ≥70%、group-OOD success 提升 ≥5 pp 且 95% CI 下界 >0、spill/drop/collision 不退化，才可把 offline operational sufficiency 扩展为 policy 结论。

## 15. 最终研究判定

### No-Go：作为独立“关键窗口 WAM”论文

原因不是想法无效，而是可主张的新颖空间过窄：SKIP 已把事件保留稀疏关键帧放进 embodied WM；BridgeACT 已使用短 clips；VLAff/ST-VLA 已使用 contact/keypose；SM-FET 已定义项目内的事件时钟、有限 packet 和 pair alignment。如果再同时学习事件语法、选窗和新 packet，系统难以归因，也会与当前严格路由冲突。

### Conditional Go：作为 SM-FET 的因果证据前端

推荐题目可以收窄为：

> **CET-WM: Are Causally Triggered Short Windows Operationally Sufficient for Finite-Event Robot Action Planning?**

其三项明确贡献只能是：

1. 严格在线、无 robot-future 的 event-triggered window memory；
2. 短窗口相对于 single keyframe/full history 的固定预算比较；
3. 带同分布控制和 side-channel audit 的 keep/delete/exchange 动作充分性协议。

若 pilot 只证明“少帧重建得不错”或“token 更少”，应归入 SKIP/token-pruning 工程；若最终只有 SM-FET hard packet 有效而 raw windows 不提供额外证据，则把 CET-WM 合并为 SM-FET 的 evidence preprocessor，不另立 Idea I。

### 禁止使用的宣传句

- “首次为 embodied world model 选择事件关键帧”；
- “首次使用接触前后窗口/关键姿态”；
- “证明这些窗口是物理因果或信息论最小充分统计量”；
- “首次从人类短视频迁移工具操作”；
- “首次进行 VLA 视觉 token pruning”；
- “66 个 Ego episode 证明机器人闭环和跨 embodiment 对齐”。

## 参考文献

1. SKIP: Event-Preserving Sparse Keyframes for Embodied World Models, 2026. <https://arxiv.org/html/2606.00664v1>
2. SWEET: Sparse Visual Planning via Task-Relevant Keyframes, 2026. <https://arxiv.org/abs/2605.19319>
3. VLAff, 2026. <https://arxiv.org/html/2608.05215v1>
4. BridgeACT, 2026. <https://arxiv.org/html/2604.23249v2>
5. ST-VLA, 2026. <https://arxiv.org/html/2603.13788v1>
6. VLA-IAP, 2026. <https://arxiv.org/abs/2603.22991>
7. LightVLA, 2025. <https://arxiv.org/abs/2509.12594>
8. Wu et al., AdaFrame, CVPR 2019. <https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html>
9. Korbar et al., SCSampler, ICCV 2019. <https://openaccess.thecvf.com/content_ICCV_2019/html/Korbar_SCSampler_Sampling_Salient_Clips_From_Video_for_Efficient_Action_Recognition_ICCV_2019_paper.html>
10. Buch et al., Flexible Frame Selection, CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html>
11. Shou et al., Generic Event Boundary Detection, ICCV 2021. <https://openaccess.thecvf.com/content/ICCV2021/html/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.html>
12. Jung et al., Online GEBD, ICCV 2025. <https://openaccess.thecvf.com/content/ICCV2025/html/Jung_Online_Generic_Event_Boundary_Detection_ICCV_2025_paper.html>
13. Du et al., Fast and Unsupervised Action Boundary Detection for Action Segmentation, CVPR 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Du_Fast_and_Unsupervised_Action_Boundary_Detection_for_Action_Segmentation_CVPR_2022_paper.html>
14. Ego4D Hands and Objects Benchmark. <https://ego4d-data.org/docs/benchmarks/hands-and-objects/>
15. Chen et al., Learning to Explain: An Information-Theoretic Perspective on Model Interpretation, ICML 2018. <https://proceedings.mlr.press/v80/chen18j.html>
16. Lei et al., Rationalizing Neural Predictions, EMNLP 2016. <https://aclanthology.org/D16-1011/>
17. DeYoung et al., ERASER, ACL 2020. <https://aclanthology.org/2020.acl-main.408/>
18. Ju et al., Goodhart's Law Applies to NLP's Explanation Benchmarks, EACL Findings 2024. <https://aclanthology.org/2024.findings-eacl.88/>
19. Dwibedi et al., Temporal Cycle-Consistency Learning, CVPR 2019. <https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html>
20. Sermanet et al., Time-Contrastive Networks, 2017. <https://arxiv.org/abs/1704.06888>
21. Xu et al., XSkill, CoRL 2023. <https://proceedings.mlr.press/v229/xu23a.html>
22. Wang et al., MimicPlay, CoRL 2023. <https://proceedings.mlr.press/v229/wang23a.html>
23. Zhou et al., Deep Reinforcement Learning for Unsupervised Video Summarization, AAAI 2018. <https://ojs.aaai.org/index.php/AAAI/article/view/12255>
24. Mahasseni et al., Unsupervised Video Summarization with Adversarial LSTM, CVPR 2017. <https://openaccess.thecvf.com/content_cvpr_2017/html/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.html>
