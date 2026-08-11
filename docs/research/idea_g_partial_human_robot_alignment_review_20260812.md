# Idea G 独立调研：Event-Anchored Partial Human–Robot Pairing

- 日期：2026-08-12
- 审查基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 隔离分支：`codex/idea-g-partial-human-robot-alignment-20260812`
- `idea_id`：`e6561894-fa5a-4e44-912f-24524aac7764`
- `discussion_id`：`4cbab6ef-7c05-412b-a600-3797f393370e`
- 目标硬件：Unitree G1 + 左 RH56DFTP + 右 OmniPicker
- 研究问题：不要求人和机器人从准备动作到结束逐帧复刻，只对齐工具操作的 contact event、subgoal 与 object effect；允许 many-to-many、partial、monotonic、内部 NULL 与两端 unmatched segment。

## 结论先行

**研究假设 Conditional Go；把“partial monotonic alignment”本身当算法贡献是 No-Go；当前数据上的目标硬件收益与真机部署也是 No-Go。**

推荐结构是 **CAPOT-WAM（Contact-Anchored Partial-OT WAM）**：先把人类 clip 和机器人 rollout 压成带置信度的 contact/effect event，再做有时间顺序约束的 partial/unbalanced OT；匹配结果只作为训练监督，或在有 human prompt 的部署模式下压成最多 `K≤16` 个缓存 event token。保留现有 22D contact packet 作为动作可见的硬语义接口，新增 versioned `event_pair_v1` sidecar；不把原始人类帧、dense coupling 或 matcher hidden state直接送进动作头。

最强 novelty 冲突非常明确：

- [EgoBridge, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/434756a76c44e220d5f32d6e0b488c0c-Abstract-Conference.html) 已把 human/robot cross-embodiment co-training 写成 joint feature–action Optimal Transport，并用两域 action chunk 的 DTW cost 构造行为相似 pseudo-pair。
- [Drop-DTW, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html) 已经做了可微、保序、双边任意内部 outlier/drop 的部分序列对齐；“monotonic + partial + null”不能再声称新。
- [ImMimic, CoRL 2025](https://proceedings.mlr.press/v305/liu25b.html) 已用 DTW 构造 human–robot many-to-many frame correspondence；其附录的 GMS-SDTW 还能从含无关动作的长人类视频中找出多个匹配子序列。
- [HARP-VLA](https://arxiv.org/html/2605.31234) 已使用同任务 paired human/robot videos、手腕/末端关键点 DTW、严格 frame-wise temporal alignment 与 cross-prediction。
- [STRAP, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a06e129e01e0d2ef853e9ff67b911360-Abstract-Conference.html) 已用 subsequence DTW 从离线轨迹中检索变长片段。

因此**最强组合冲突是 EgoBridge 的“DTW pseudo-pair + OT joint adaptation”再加 Drop-DTW/GMS-SDTW 的“保序部分匹配”**。可守的新意只能缩到一个任务/表示/因果路由交集：**在强不对称手—夹爪 embodiment 下，以 contact 与 object effect 为锚、显式建模置信度和缺失质量，把 episode-level 弱配对变为稀疏 event coupling，并证明该 coupling 经受严格信息路由对照后改善目标机器人成功率。** 如果不能显著超过 EgoBridge-style OT、Drop-DTW/GMS-SDTW，或 oracle event pairing 对 22D-only WAM 没有下游增益，应停止这条线。

## 证据边界与检索记录

证据标记：

- **[I]**：当前仓库中已实现，可定位到代码/审计。
- **[R]**：论文原文、作者页或正式 proceedings 支持。
- **[D]**：本文提出的设计，尚未实现。
- **[M]**：当前缺少真实标签、训练、统计或目标硬件证据。

本轮严格先运行 `idea-discussion` 的 Zotero 同步与检索，再回到 arXiv、CVF、PMLR、NeurIPS、ICLR 等原始页面核对。idea memory 已更新。Zotero 库中直接相关条目包括：EgoBridge（`A89APIR3`）、UniSkill（`EK2CTN7J`）、ImMimic（`8WMVVJAL`）、XSkill（`V7LLF6VJ`）、Time-Contrastive Networks（`WQJXZYHX`）、HARP-VLA（`3SVRDJ39`）与 Ego2Robot（`UAGPBXR8`）。自动检索也返回 RT-1、π0.5、PaliGemma 以及与本题无关的 recovery/image manipulation 条目；这些噪声命中被丢弃，没有用来支撑结论。

本 memo 只做文献与架构设计；没有训练、没有调用 GPU、没有更改 canonical worktree，也没有把旧硬件数据包装成目标硬件结果。

## 1. “全程一对一”并不是现有方法的统一假设

必须先拆开三个概念：

1. **one-to-one frame pairing**：第 `t` 帧只能对第 `t` 帧；HARP-VLA 的 strict frame-wise 同步最接近。
2. **complete monotone path**：标准 DTW/soft-DTW 可用横/竖步形成 one-to-many 和 many-to-one，但路径通常覆盖两条序列的端点与全部样本；它不是一对一，却仍强迫“全程有归宿”。
3. **partial alignment**：允许序列内部和两端片段落入 NULL；Drop-DTW、subsequence DTW、GMS-SDTW 已覆盖这一算法族。

所以 Idea G 不能以“现有工作都做全程一对一”为前提。正确判断如下。

| 方法 | 人–机器人是否需要显式 paired trajectory | 对齐粒度与覆盖 | 是否已支持 partial/null | 对 Idea G 的实际冲突 |
|---|---|---|---|---|
| [HARP-VLA](https://arxiv.org/html/2605.31234) | paired 与 unpaired 都用；paired 同任务且采集时要求相似动作 | keypoint DTW 后筛/复制人类帧，使人/机器人持续时间和速度严格同步 | 否；paired 分支趋向完整 frame-wise 覆盖 | **最强 paired-data 训练冲突**；Idea G 应直接比较 strict DTW 与 event partial coupling |
| [ImMimic](https://proceedings.mlr.press/v305/liu25b.html) | 主方法同任务 human/robot demos | frame-level action/vision DTW；一个 human timestep 可映射一组 robot timesteps，故不是 one-to-one | 主方法完整；附录 GMS-SDTW 支持长视频内部无关段与多次子序列命中 | **最强方法组合冲突**；不能仅凭 many-to-many、长视频与 preparation drop 声称新 |
| [STRAP](https://arxiv.org/abs/2412.15182) | query demo 与离线同 embodiment 轨迹；不是 human–robot pair | vision feature subsequence DTW，检索变长子轨迹 | 开放起止，但选中 query 内部仍完整匹配 | 对“片段检索/开放边界”直接构成 baseline |
| [Human2Robot](https://arxiv.org/abs/2502.16587) | 2,600 个经 VR teleportation 得到的 perfectly aligned human–robot episode | fine-grained 同步，再生 robot video/action | 否 | 证明高质量强配对是可行路线；本方案必须证明弱配对更便宜且不牺牲效果 |
| [HR-Align, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Mitigating_the_Human-Robot_Domain_Discrepancy_in_Visual_Pre-training_for_Robotic_CVPR_2025_paper.html) | paired human/robot video | 对 paired 表征做对比式语义对齐 | 没有 event-level NULL | 可作 paired contrastive baseline，但不解决不等长细粒度对应 |
| [EgoBridge, NeurIPS 2025](https://arxiv.org/html/2509.19626) | human/robot 数据无需预先一一配对；mini-batch 内按 action-chunk DTW 选 pseudo-pair | balanced Sinkhorn OT 对齐 joint latent–action distribution；DTW 路径固定从两端开始/结束 | 否；其 DTW 是 complete，同长度 action chunk，OT 也守固定 marginals | **最强 OT 架构冲突**；Idea G 必须比较其 pseudo-pair + OT，而差异只能是 event partial/null/effect/strict route |
| [RH20T](https://arxiv.org/abs/2307.00595) | 每个 robot sequence 有对应 human demo video 与语言 | 主要是 episode/task 对应，不等于物理同步 frame pair | 未提出显式 partial temporal matcher | 证明“同任务弱 episode pair”数据协议可规模化 |
| [EgoMimic](https://arxiv.org/abs/2410.24221) | 不需要人/机器人 episode pair | 两域分别采样共训，以相机坐标、标准化与视觉处理统一；human 监督 pose，robot 监督 pose+joint action | 不适用；根本不建立跨域时序 pair | 是强 unpaired baseline；Idea G 必须证明显式 event pairing 比简单 co-training 有增益 |
| [XSkill](https://arxiv.org/abs/2307.09955) | 无显式 pair | Sinkhorn prototypes + TCN 学跨 embodiment skill，Skill Alignment Transformer 在线跟随 human prompt | 学 skill order，不显式输出双边 NULL coupling | 与“共享 event/skill latent”冲突；是 representation baseline |
| [UniSkill](https://proceedings.mlr.press/v305/kim25d.html) | 明确不需 paired/task-aligned data | 固定窗口从人类 prompt 取 skill；机器人按 prompt 时间索引消费 | 无显式跨域 event alignment | 强 unpaired baseline；其固定滑窗会受额外准备段影响 |
| [XIRL](https://arxiv.org/abs/2106.03911) | 跨 embodiment demonstrations，不需 execution-time reference alignment | temporal cycle consistency 学 task progress embedding，再以目标距离给 reward | 不输出显式 NULL pair | 可作共享进度表征 baseline；它规避而非解决 pair construction |
| [Cross-domain Imitation from Observations](https://proceedings.mlr.press/v139/raychaudhuri21a.html) | unpaired、unaligned trajectories | cycle consistency + normalized temporal position | 没有对额外准备/恢复段显式建模 | 其归一化进度先验与不等长、局部任务效应直接冲突 |
| [Semantic Skill Learning](https://proceedings.mlr.press/v205/pertsch23a.html) | human video 到 robot，无低层对应 | 模仿 semantic skill sequence | 不做 frame/event pair | 限定贡献：semantic subgoal transfer 本身不是新意 |
| [DexMimicGen](https://arxiv.org/html/2410.24185) | “human demos”实为目标机器人 teleoperation source demos | 按每臂 subtask 切分，允许异步、同步与顺序约束，再在模拟器重放/retarget | segment 级组合，但不是 passive human–robot alignment | 说明多臂异步/segment dependency 已有成熟先例 |
| [Ego2Robot](https://arxiv.org/html/2608.02580) | 不需 paired target rollout | human hand 逐帧 retarget 到 parallel gripper，平滑并按源速率 subsample | 否 | direct retarget baseline；强制全段变换可能在异形态准备动作上失败 |
| [MimicDreamer](https://arxiv.org/html/2509.22199) | 同一 human sequence 被转成 robot-view video/action | IK 与视频生成保持 64-frame time alignment | 否 | frame-wise transformation baseline，不是 learned partial pair |
| [Ego-Exo4D](https://arxiv.org/abs/2311.18259) | 同时采集 human ego/exo | 同步 human 多视角，1,286 小时 | 无 robot/contact truth | 可作视角预训练，不能当 human–robot pair GT |

配对强度因此至少分四档：Ego-Exo4D 是**同步 human ego–exo view**，但没有 robot；RH20T 是**同任务/episode 对应**，不等于同步；EgoBridge 是**无预配对数据上的 DTW pseudo-pair**；Human2Robot/HARP-VLA 才接近**细粒度或严格 frame-wise human–robot pair**。本文所说的 weak pair 指第二/第三档，不能混写为“paired frame GT”。

### 1.1 contact/effect 表示同样不是空白赛道

- [BridgeACT](https://arxiv.org/html/2604.23249) 明确主张绝对 human/object motion 难以跨 embodiment，转而提取 tool/target role、contact region 与 interaction-centric 3D point flow，并以闭环 SE(3) 控制执行。[R]
- [VLAff](https://arxiv.org/html/2608.05215) 从 egovideo interaction timestamp 提取 contact keyframe、手指—物体接触、接触前后 3D 轨迹，输出 affordance heatmap、96D MANO grasp 与量化 6D trajectory token。[R]
- [VRB, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bahl_Affordances_From_Human_Videos_as_a_Versatile_Representation_for_Robotics_CVPR_2023_paper.html) 已将 contact point 与 post-contact trajectory 用于多种机器人学习范式。[R]
- [GraphIRL](https://proceedings.mlr.press/v205/kumar23a.html) 已用 object-centric graph 与 temporal matching/cycle consistency 从 human video 学 cross-embodiment reward；[VIRL](https://proceedings.mlr.press/v270/huang25d.html) 延伸到视觉图表示的时间对齐。[R]
- [ORION](https://arxiv.org/abs/2405.20321) 从单个 RGB-D human video 构造 open-world object graph；[GraphMimic, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_GraphMimic_Graph-to-Graphs_Generative_Modeling_from_Videos_for_Policy_Learning_CVPR_2025_paper.html) 生成未来 object/action graphs。[R]
- [Human2Sim2Robot](https://proceedings.mlr.press/v305/lum25a.html) 用 object pose trajectory 作为 embodiment-agnostic reward，并保留 pre-manipulation hand pose；[RSRD](https://proceedings.mlr.press/v270/kerr25a.html) 用 part-centric 3D object trajectory 做跨 embodiment 模仿。[R]

结论是：contact、object effect、object graph、subgoal 都有前例。Idea G 唯一可能守住的是**这些语义作为弱 paired-data 的 sparse partial alignment unit，并进入受审计 WAM 路由**，而不是其中任何一个名词。

## 2. 对齐工具的能力边界

| 工具 | 已有能力 | 缺口/误区 |
|---|---|---|
| [soft-DTW](https://arxiv.org/abs/1703.01541) | 对完整 DTW path 的 soft-min，可微；横/竖步天然 one-to-many/many-to-one | 标准端点和覆盖约束仍使每个样本落到路径；不是 partial/null |
| [Drop-DTW](https://arxiv.org/abs/2108.11996) | 双序列任意位置丢弃 outlier，同时保序、可微、动态规划高效 | 已覆盖本题最显眼的算法卖点；只缺 task-specific contact/effect/missingness/WAM evidence |
| [OTAM](https://openaccess.thecvf.com/content_CVPR_2020/html/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.html) | ordered DTW-like video alignment | 没有对双边内部 NULL 做明确建模 |
| [Partial OT](https://proceedings.neurips.cc/paper/2020/hash/1e6e25d952a0d639b676ee20d0519ee2-Abstract.html) | 仅运输固定质量，row/column mass 不超过 marginals，剩余质量可 unmatched | 单独使用不保时间顺序；固定 transported mass 难处理不同 pair 的真实 overlap |
| [Unbalanced OT](https://www.ams.org/mcom/2018-87-314/S0025-5718-2018-03303-8/S0025-5718-2018-03303-8.pdf) | 用边缘 KL 等松弛质量守恒，允许双边质量产生/消失 | 单独使用也不保顺序；过强 marginal relaxation 会 all-null/少量易点坍缩 |
| [EgoBridge](https://arxiv.org/html/2509.19626) 的 joint OT | mini-batch human/robot latent–action balanced Sinkhorn；DTW 最邻近项降低 OT ground cost，直接服务 policy co-training | 已占据“DTW 引导 human–robot OT”表述；但 action chunk 的 DTW 完整覆盖、OT 固定 marginals，不是 contact/effect event partial transport |
| [CTC](https://mlanthology.org/icml/2006/graves2006icml-connectionist/) | blank、repeat-collapse、输入到较短已知标签序列的 monotonic marginalization | 非对称；blank 是输入帧的空标签，不是两条未知序列的双边 NULL pair |
| [TACO](https://proceedings.mlr.press/v80/shiarlis18a.html) | 给定 ordered task sketch，联合学 subpolicy 与 latent alignment | 需要弱 subtask 顺序，不建立 human–robot pair |
| [Segmental RNN](https://arxiv.org/abs/1511.06018) / [segmental models](https://arxiv.org/abs/1603.00223) | semi-Markov 风格显式 segment duration、label 与 latent boundary | 适合 event transducer，但数据和解码复杂度高于 OT pilot |
| [Neural Task Graphs](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Neural_Task_Graphs_Generalizing_to_Unseen_Tasks_From_a_Single_CVPR_2019_paper.html) | 从示范抽取 task graph 作为中间表示 | 不提供跨 embodiment partial pair matcher |

## 3. 三个互斥结构与排序

三者应作为**互斥主模型**比较，不要在第一版把 OT、transducer 和 graph 全堆进一个无法归因的系统。

### Rank 1 — CAPOT-WAM：Contact-Anchor Monotone Partial OT（推荐）

**单位。** 每个 episode 先变成低频 event 序列：`approach → contact_on → hold/slide/rotate/pour → object-object contact → release`；event 带 tool/target role、object part、接触模式、起止区间、前后 object-state delta、subgoal、可见性和字段级置信度。准备、等待、导航、失败后恢复可成为 NULL，而不是硬配给另一个 embodiment。

**代价。** 对 human event `h_i` 与 robot event `r_j`：

```text
C_ij = α d_contact(h_i,r_j)
     + β d_role_part_geometry(h_i,r_j)
     + γ d_object_effect(h_i,r_j)
     + δ d_subgoal_order(h_i,r_j)
```

以 field confidence 形成质量 `a_i,b_j`，求：

```text
min_P <P,C> + ε Σ_ij P_ij(log P_ij - 1)
      + τ_h KL(P1 || a) + τ_r KL(P^T1 || b)
      + λ_mono Σ_{i<k,j>l} P_ij P_kl
```

也可在 pilot 中比较固定 transported mass 的 partial OT。`P` 的 row/column 可分配给多个节点，故 naturally many-to-many；未运输质量解释为 NULL。UOT 本身不保证 monotonic，必须保留 crossing penalty、banded causal support 或 DP 非交叉投影。低置信度质量不等于负例，missing 也不等于 `no contact`。

**优点。** 最适合无同步、少量弱标签、小数据和离线 pair building；coupling 可解释、可审计，先不改在线控制器也能做检索/对齐 benchmark。

**主要风险。** EgoBridge 已把 DTW pseudo-pair 与 OT 结合，Drop-DTW/GMS-SDTW 又已支持 partial；若 gain 只来自 event detector，而不是 effect-aware partial coupling，算法论文不成立。UOT 还可能 all-null 或只匹配最容易的 contact-on/off。

### Rank 2 — MEN-T：Monotonic Event Transducer with NULL

把 human event sequence 编成 memory，robot 每个 semi-Markov segment 输出：`stay`、`advance`、`skip_h`、`skip_r`、`emit(event/subgoal)` 与 `NULL`；用 forward–backward 对 latent boundary/duration 求和。它把 CTC blank 从单边扩展为双边缺失与显式持续时间，并能把 `current_event_pointer` 直接用作在线状态机。

**优点。** 流式执行最自然；能表示“一段 human pour 对多个 robot micro-adjustment”和恢复后重新进入同一 subgoal。

**缺点。** 需要更强 event/subgoal 标注；非对称方向、beam search、exposure bias、长时 credit assignment 和错误 pointer 累积更严重。当前 66+88 数据不足以证明 transducer 优于简单 OT，故不宜作为第一版。

### Rank 3 — OSEG：Object-State-Effect Graph Matching

将每个 episode 建图：object/part 为节点，tool-target/contact/spatial relation 为边，event 是把 `G_before` 变为 `G_after` 的操作；用 partial graph edit、partial Gromov–Wasserstein 或 typed graph matching 对齐相同 effect。人手与 RH56/OmniPicker 仅作为不同 actuator node，核心匹配 `container_A tilted → liquid transferred → container_B filled`。

**优点。** embodiment invariance 最强，理论上最能忽略人/机器人动作形态和长度。

**缺点。** 当前透明杯/液体的 object state 与 effect 最难可靠观测；graph ontology、object tracking、因果 effect 与重复接触的标注成本最高。GraphIRL/ORION/GraphMimic/BridgeACT 已使 novelty 空间拥挤。它适合作为未来结构化上界，不是当前 MVP。

## 4. Paired data：不要求同步，也不强制 episode 一一对应

### 4.1 先 split，再生成 pair edge

先按原始 human clip、robot rollout、操作者、session/day、object instance/shape、采集批次切分；同一源片段的窗口、增强、event 与 pair edge 必须全部落在同一 split。**不能先枚举大量 pair，再随机切 pair**，否则一个 episode 会通过其它边泄漏到测试集。

每条 episode 只需弱标签：

- task family 与语言描述；
- tool/target functional role，而不只按外观类别；
- object category、instance、shape/material，未知可留空；
- outcome signature：成功/失败、目标状态、转移量、spill/drop/collision；
- 粗 event interval 与 subgoal order；
- 数据域、传感器、硬件、操作者/session/day；
- episode/event/field-edge 三层 confidence 与 annotation provenance。

同一个 human clip 可连多个 robot rollout，同一个 robot rollout 也可连多个 human clip。candidate graph 的粗 gate 是 `task family × role × outcome`，再按三种难度分层：同物体实例、同类别异形态、跨类别但同 affordance/effect。episode-level gate 只定义“可以尝试匹配”，不把整条轨迹标成正对。

### 4.2 正例、负例与 NULL

**正 event pair：** functional role、object part、contact mode、effect 与拓扑顺序一致；空间轨迹和持续时间可以不同。一个 human 连续倾倒段可以对应 robot 的抓稳、试探倾角、主倾倒、回正多个 event；反向也允许多个人类 micro-contact 对一个夹爪闭合 event。

**有用的 NULL：** 人类自然准备/调整视角、机器人归零/安全等待、只在一侧出现的 regrasp、失败后的 recovery、传感缺失且无法判定的 interval。NULL 不等于 background negative；不可见、未标与确认无接触必须分开。

**hard negative：**

- 同物体外观但 object effect 错（拿起 vs 推倒、倾斜但无转移）；
- 同 task label 但 subgoal 顺序错；
- 运动方向相似但 tool/target role 反了；
- 接触模式相似但 part 错（杯身 vs 杯口）；
- unsuccessful/no-transfer、spill/drop/collision；
- time shuffle 或制造 crossing 的 pair。

**easy negative：** task 与 functional role 都不同。重复接触不可按 off-diagonal 自动当负例，否则会错误惩罚合法 many-to-many。

### 4.3 标注协议

建议两阶段：自动 proposal 只生成候选，人工双标与 adjudication 才能成为 gold。

1. 标 `contact_on/hold/slide/release/object-object_contact` 区间，显式 left/right censor 与 visibility。
2. 标 tool、target、part、effect-before/after、subgoal、success/failure；每字段可 unknown。
3. 标少量 event edge 为 `match / non-match / ambiguous`；其余由模型弱配，不能伪装成人工 GT。
4. 记录 onset/release 容差和置信度理由；审计 annotator agreement。

## 5. 现有 66 Ego 与旧 dual-Inspire 数据能做什么

### 5.1 事实边界

- Ego 源只有一个 `pour liquid between beakers` task、66 episodes、99,953 个 raw 100 Hz frames 与 20,013 个 exact-stride 20 Hz frames。[I] 仓内审计表明 metadata 缺 subject/session/operator、source/target instance、liquid identity/viscosity/fill 与 batch，因此不能构造可信 production OOD split；见 `docs/ego_discrete_contact_stage1.md:24-35,336-341` 与 `docs/audits/ego_66ep_group_metadata_recovery_20260811.md:40-74`。
- `hand_status` 是 `0=open,1=closed` 的 closure proxy，不是物理 contact truth，也不是 OmniPicker command；见 `docs/asymmetric_action_schema.md:142-156`。[I]
- 旧 `outputs/grab_red_bottle` 有 88 episodes、40,901 frames、50 Hz，来自双 Inspire hand；任务是抓红瓶，不是倒液体，也不是左 RH56 + 右 OmniPicker。仓库明确把它标为 asymmetric25 audit-only；见 `docs/asymmetric_action_schema.md:158-164`。[I]

因此两套数据当前**没有同 task + 同 effect + 目标硬件的正 pair overlap**。不能把 66×88 的笛卡尔积称为 paired dataset，也不能用旧 dual-Inspire 结果声称 G1/RH56/OmniPicker 泛化。

### 5.2 不冒充目标硬件的 CPU/offline 弱预实验

1. 在 Ego 内部人工选少量 clips，标 contact/effect 近似 gold；做 leave-episode-out retrieval，验证 event detector 与 self-alignment。
2. 在旧 robot 内部比较标准 DTW、soft-DTW、Drop-DTW、GMS-SDTW 与 CAPOT 的稳定性；这里只验证算法管线，不验证 cross-embodiment transfer。
3. 将 Ego-pour 与 robot-grab 明确作为 **task/effect mismatch、NULL-heavy negative**，检查 matcher 是否拒绝强行全程对齐；绝不作为 positive pair。
4. 对同一 episode 做可控合成扰动：prepend/append preparation、插入 idle/regrasp/recovery、局部 speed warp、重复 contact。真实 anchor 不变，用来测 NULL 检出、crossing 与 event recovery。
5. 若能找到或新采一小批真正同 task human/robot clips，才开放弱 cross-embodiment pilot；否则 G0 直接失败。

这些实验不需要训练 WAM、不需要 GPU，产物必须标记为 `proxy_alignment_pilot`，结论仅限“对齐器是否比完整路径 baseline 更合理”。

## 6. WAM 融合点与运行时路由

### 6.1 不替换 22D contact packet

当前每个 contact segment 的 22D 定义是 `presence 1 + side 3 + site 6 + object_role 4 + contact_mode 4 + start/end 2 + left/right_censored 2`，位于 `world_action_model/contact_transition_schema.py:65-74`；`encode_ordered_segments` 生成 Q×22 target，位于 `world_action_model/contact_transition_schema.py:624-655`。[I]

它适合做**当前机器人局部接触计划**，但缺少 object instance/part、before/after effect、subgoal、pair edge、source domain、event-level confidence、NULL 与跨 event 顺序。因此建议：

- 保留 `contact_segment_v1_1` 与 22D 动作可见 packet；
- 新增低频、训练/提示级 `event_pair_v1` sidecar，记录 human/robot episode ID、event intervals、22D slot reference、role/part/effect/subgoal、confidence、sparse coupling 与 NULL mass；
- effect/subgoal 先进入单独的低维 event embedding，不直接膨胀 22D；
- 只有 oracle ablation 证明 22D 确实成为瓶颈，才设计 versioned packet v2，并保持旧 checkpoint fail-closed。

### 6.2 数据接入点

现有 dataset 已有严格 contact sidecar 接口与哈希 preflight：构造参数在 `world_action_model/datasets/wam_lerobot_dataset.py:66-89`，reviewed schema/receipt 检查在 `:285-325`，逐 episode 读取和逐 frame 注入在 `:741-819`。[I]

建议平行新增 `event_pair_sidecar_dir`，但按 **chunk/episode** 读取稀疏 event，而不是把 coupling 重复写到每帧；metadata 必须绑定：schema version、source/target dataset identity、split receipt、annotator/adjudication、生成器 commit 和每文件 SHA256。[D]

### 6.3 模型融合点

当前 `plan_contact_route` 在 `world_action_model/models/transformer_wa_casual_mot.py:1358-1403` 生成动作可见 packet；`encode_action_tokens` 在 `:1405-1480` 通过唯一的 `22 → action width` 投影注入 contact semantics，并明确不拼接 `planner_hidden`。[I] MoT attention mask 在 `:127-189`，调用处 `:1809-1822` 可关闭 action→reference 直连。[I]

推荐增加 `AlignedEventMemoryEncoder`，但保持两种模式：

- **无 prompt 部署：** partial alignment 仅是训练监督/预训练正负对；推理仍由当前机器人 reference/state/text 预测 22D，运行时零新增 matcher 开销。
- **human-prompt 部署：** human clip 离线编码为最多 `K≤16` 个 event tokens 并缓存；机器人当前 22D query 只读取 monotonic pointer 邻域，通过 `stay/advance/null` 选择一个 effect/subgoal token。20 Hz 循环内禁止求完整 `O(T_hT_r)` Sinkhorn/DTW。

严格信息流应是：

```text
human prompt -> event encoder -> cached K event/effect tokens
                                      |
robot ref/state/text -> contact planner -> 22D query -> monotone selector
                                                     |
state + language + noisy action + hard 22D + selected event embedding -> action expert
```

raw human frames、dense coupling、pair IDs、future robot frames、matcher hidden state 都不能旁路进入 action expert。无 prompt 或低置信度时，selector 输出 NULL/prototype event，原有路线保持可运行。action-only prefix cache 已在 `world_action_model/models/transformer_wa_casual_mot.py:2159-2173,2520-2593` 实现 packet receipt/复用逻辑，可作为 event-cache 的 fail-closed 模板。[I/D]

### 6.4 Loss

```text
L = L_action/WAM
  + λ_contact L_22D
  + λ_pair L_partial_align
  + λ_mono L_crossing
  + λ_effect L_effect
  + λ_null L_null
  + λ_cycle L_pair_cycle
  + λ_cal L_confidence_calibration
```

- `L_22D` 继续使用现有 field-balanced BCE/CE/Smooth-L1 与 mask/confidence，入口在 `world_action_model/trainer/wa_casual_trainer_mot.py:111-189`。[I]
- `L_partial_align` 是上述 partial/UOT objective；gold/ambiguous/unknown edge 分别监督、软监督、mask。
- `L_mono` 惩罚 crossing；评估时必须另报 crossing rate，不能只靠 loss 值。
- `L_effect` 预测匹配前后 object-state delta；避免只按手势外观对齐。
- `L_null` 用明确 negative/NULL 标注与可校准概率监督；transported-mass prior 只防 all-null，不能把固定 coverage 当 truth。
- 对 episode/effect signature 使用 contrastive/ranking negative；重复 event 不做逐位置 InfoNCE 假负例。
- `L_cycle` 约束 human→robot→human 的 event identity/interval，但不能以 cycle consistency 代替真实 pair 标注。

结构 B 改用 semi-Markov forward–backward/transducer NLL；结构 C 改用 typed partial graph edit/partial GW 加 effect predicate loss。第一阶段不可同时启用三种主损失。

## 7. Benchmark、baselines、ablation 与统计

### 7.1 目标 benchmark

最终 gold benchmark 必须是目标硬件的同 task 数据：至少覆盖多名操作者、多天/session、多 object instance/shape、同实例/同类异形/跨类同 affordance，并保留成功、失败、恢复与部分完成。训练 pair 构造与测试 pair 构造都要在 episode group split 后完成。

**对齐指标：** event-edge precision/recall/F1、interval tIoU、onset/release MAE、NULL AUROC/AUPRC/F1、confidence ECE/Brier、crossing/order violation、transported mass/coverage、object-effect consistency。

**下游指标：** rollout success、correct-first-contact、各 subgoal completion、转移量误差、spill/drop/collision、OOD object/shape/effector；同时报告无 human prompt 与有 prompt 模式。

**系统指标：** 离线 pair construction 耗时、缓存大小、在线 selector latency、20 Hz deadline miss 与峰值显存/内存。第一阶段仅 CPU/offline 时明确报告“无在线系统结果”。

### 7.2 Baselines

1. robot-only GWP/WAM；
2. 22D contact WAM，无 human pairing；
3. normalized-time / strict frame-wise 1:1；
4. standard DTW 与 soft-DTW；
5. Drop-DTW；
6. STRAP subsequence DTW 与 ImMimic GMS-SDTW；
7. HARP-style keypoint DTW + strict synchronization；
8. HR-Align-style paired contrastive；
9. EgoBridge-style action-DTW pseudo-pair + balanced joint OT；
10. EgoMimic-style unpaired co-training；
11. XSkill/UniSkill-style shared skill representation；
12. 只有 task/outcome 的 episode-level positive；
13. balanced OT、partial OT 与 UOT-without-monotonicity；
14. oracle manual event pairing（上界）。

所有 paired 方法使用相同 human/robot episodes、robot demonstration budget、backbone、参数量/FLOPs和动作信息路由。否则“数据更多/路径更宽”会冒充对齐收益。

### 7.3 必做 ablation

- generic visual feature vs contact anchor；
- 去掉 effect/subgoal；
- balanced vs partial vs unbalanced OT；
- 去掉 monotonicity；
- 无 NULL、只允许端点 drop、允许内部 NULL；
- hard 1:1 vs soft many-to-many；
- 22D-only vs 22D + effect embedding vs oracle effect；
- 去掉 confidence/missingness mask；
- 同物体 only vs 同类异形 vs 跨类同 affordance；
- 人工 trim preparation vs 保留 preparation；
- raw-human shortcut vs strict event route；
- `K`、entropy `ε`、mass relaxation `τ` 与 transported-mass prior。

### 7.4 统计

pair edge 不是独立样本。以原始 human clip 与 robot rollout 交叉聚类做 hierarchical bootstrap，报告 95% CI；成功率可用含 operator/session/object random effect 的 mixed-effects logistic regression，或在 rollout cluster 上做 paired permutation。预注册 primary metric、seed 与 stop gate；不以窗口数扩大虚假样本量。至少对 `6666/7777/8888` 三 seed 做完整 matched comparison，但目标硬件 trial 的独立单位仍是 rollout，不是 seed×frame。

## 8. Stop gates

| Gate | 进入条件 | 通过阈值 | 失败动作 |
|---|---|---|---|
| G0：可标注与真实 overlap | 至少 12 human clips + 12 同 task 目标 robot rollouts，双标并 adjudicate | event F1 ≥ 0.80、matched interval tIoU ≥ 0.70、NULL F1 ≥ 0.80、effect predicate κ ≥ 0.65、关键类别 κ ≥ 0.70 | 无同 task overlap 或 agreement 不足：停止 cross-embodiment claim，只修 ontology/采数 |
| G1：对齐算法 | 合成 prep/idle/regrasp/recovery/speed warp + 小 gold | NULL AUPRC ≥ 0.80，anchor F1 比 soft-DTW 高 ≥10 pp，且比 EgoBridge-style OT、Drop-DTW/GMS-SDTW 最优者高 ≥5 pp；crossing <1%，无 all-null | 不能超过最强已有方法：停止“新 matcher”主张，最多把其作为工程 baseline |
| G2：oracle sufficiency | 目标硬件、固定 robot budget、oracle event pair | oracle 相对 22D-only/robot-only success 提升 ≥10 pp，或预注册关键 phase 指标有实质提升且 CI 排除 0 | oracle 无收益：alignment 不是瓶颈，停止模型开发 |
| G3：learned pairing | matched controls、strict no-shortcut | learned model 恢复 ≥70% oracle gain；相对最强 baseline 的 alignment 与 task-success 95% cluster CI 均 >0 | 只改善离线 F1、不改善任务：不得宣称 policy transfer |
| G4：目标硬件 | driver、reverse mapper、calibration、watchdog、bench/robot tests 与目标数据齐全 | deadline/safety 与预注册 rollout 指标全部通过 | 保持部署 No-Go；旧 dual-Inspire 不可解锁 |

当前状态：G0 尚未满足，G1 只能做 proxy sanity，G2–G4 均没有证据。[M]

## 9. 最强反对意见与回应

1. **“EgoBridge 已经是 DTW-guided human–robot OT。”** 正确，也是本题最强架构冲突。CAPOT 必须证明 complete action-chunk DTW 与 balanced OT 在额外 preparation、异形态接触和失败恢复下不足，并由 event/effect partial mass 带来独立增益；否则直接采用 EgoBridge。
2. **“Drop-DTW 已经解决 partial monotone null。”** 正确。CAPOT 只有在 contact/effect anchor、field confidence、many-to-many mass 和 WAM strict route 对目标机器人产生额外收益时才有贡献；否则应直接采用 Drop-DTW。
3. **“ImMimic 已经做人–机器人 DTW，附录还处理长 raw video。”** 正确。论文不能写“首次从 untrimmed human video 自动找对应”；比较必须包含 GMS-SDTW，且差异聚焦异形态 effect correspondence。
4. **“根本不需要 pair，EgoMimic/XSkill/UniSkill 已能共训。”** 这是必要 baseline。显式 pairing 的价值必须体现在相同 robot budget 下的 sample efficiency、失败段拒绝和有 prompt 在线 phase tracking，否则增加标注不划算。
5. **“BridgeACT/GraphIRL 已把 contact/object effect 做成跨 embodiment 表示。”** 正确。object/contact representation 不是 novelty；本题研究的是弱 episode pair 到 sparse event coupling 以及该 coupling 的因果价值。
6. **“66 Ego + 88 robot 足够先跑。”** 不足。它们 task/effect/hardware 不重叠；最多证明 matcher 会拒绝错误配对，不能证明迁移。
7. **“把更多 effect 塞进 22D 就行。”** 不建议。22D 是当前局部 contact contract；pair provenance、双边区间和 dense/sparse coupling 是训练数据关系，不应变成每步动作命令。先 sidecar + 小 event embedding，oracle 后再决定 packet v2。

## 10. 决策与最小下一步

### Go / No-Go

- **Conditional Go：** 只批准 `CAPOT-WAM` 的 CPU/offline annotation + alignment pilot，以及目标同任务 paired-data 的小规模采集设计。
- **No-Go：** 目前启动 WAM/GPU 训练、声称新 partial alignment 算法、声称 G1/RH56/OmniPicker 收益、或真机部署。
- **直接止损条件：** G1 不超过 EgoBridge-style OT、Drop-DTW/GMS-SDTW 最优者，或 G2 oracle pairing 不改善 22D-only WAM。

### 推荐模型

主模型：**CAPOT-WAM**；baseline 必须含 EgoBridge-style DTW-guided OT、Drop-DTW、ImMimic GMS-SDTW、HARP-style strict DTW、EgoMimic-style unpaired co-training。MEN-T 只在 CAPOT 已证明 online event pointer 有价值后作为 streaming follow-up；OSEG 只在 object effect 观测/标注可靠后作为高成本结构上界。

### 最小可执行顺序

1. 先找/采同 task human–目标 robot 小集合，完成 G0 双标；没有 overlap 就停止。
2. 用原始 event intervals 建 gold，跑 soft-DTW、EgoBridge-style OT、Drop-DTW、GMS-SDTW 与 CAPOT 的 CPU benchmark，完成 G1。
3. 只做 oracle event embedding 对 22D-only 的 matched target-hardware comparison，完成 G2。
4. 仅当 G0–G2 全过，才实现 learned pair sidecar、strict WAM route 和 G3；最后单独完成硬件 G4。

## 主要原始来源

- Dvornik et al., [Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers](https://arxiv.org/abs/2108.11996), NeurIPS 2021.
- Punamiya et al., [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](https://arxiv.org/abs/2509.19626), NeurIPS 2025.
- Liu et al., [ImMimic: Cross-Domain Imitation from Human Videos via Mapping and Interpolation](https://proceedings.mlr.press/v305/liu25b.html), CoRL 2025.
- [HARP-VLA: Human-Aligned Robotic Pretraining for Vision-Language-Action Models](https://arxiv.org/html/2605.31234), 2026 preprint.
- Chen et al., [STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning](https://arxiv.org/abs/2412.15182), ICLR 2025.
- Cuturi & Blondel, [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/abs/1703.01541), ICML 2017.
- Chapel et al., [Partial Optimal Transport with Applications on Positive-Unlabeled Learning](https://arxiv.org/abs/2002.08276), NeurIPS 2020.
- Chizat et al., [Scaling Algorithms for Unbalanced Optimal Transport Problems](https://arxiv.org/abs/1607.05816), Mathematics of Computation 2018.
- Graves et al., [Connectionist Temporal Classification](https://mlanthology.org/icml/2006/graves2006icml-connectionist/), ICML 2006.
- Sermanet et al., [Time-Contrastive Networks](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/papers/Sermanet_Time-Contrastive_Networks_Self-Supervised_CVPR_2017_paper.pdf), CVPR Workshops 2017.
- Dwibedi et al., [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846), CVPR 2019.
- Zakka et al., [XIRL: Cross-embodiment Inverse Reinforcement Learning](https://arxiv.org/abs/2106.03911), CoRL 2021.
- Xu et al., [XSkill](https://arxiv.org/abs/2307.09955), CoRL 2023.
- Kim et al., [UniSkill](https://proceedings.mlr.press/v305/kim25d.html), CoRL 2025.
- Bahl et al., [Affordances from Human Videos as a Versatile Representation for Robotics](https://openaccess.thecvf.com/content/CVPR2023/html/Bahl_Affordances_From_Human_Videos_as_a_Versatile_Representation_for_Robotics_CVPR_2023_paper.html), CVPR 2023.
- Kumar et al., [Graph Inverse Reinforcement Learning from Diverse Videos](https://proceedings.mlr.press/v205/kumar23a.html), CoRL 2022.
- Shiarlis et al., [TACO: Learning Task Decomposition via Temporal Alignment for Control](https://proceedings.mlr.press/v80/shiarlis18a.html), ICML 2018.
