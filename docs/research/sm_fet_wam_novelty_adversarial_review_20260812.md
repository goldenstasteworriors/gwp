# SM-FET-WAM v2-min 新颖性对抗审稿备忘录

> 日期：2026-08-12<br>
> 角色：独立新颖性审稿人（adversarial novelty review）<br>
> 审查对象：`SM-FET-WAM v2-min + offline CAPOT partial/many-to-many pairing + training-only relation-effect grouping + entry/turn/exit pose auxiliary`<br>
> 基线提交：`e212afec9db5198a89930c7ff896e3e49a0c8249`<br>
> idea-discussion `idea_id`：`6c12acbf-343e-4b07-9814-96696416a05c`<br>
> idea-discussion `discussion_id`：`b076f98d-d7ce-49da-842a-ca451352a9b0`

## 0. 一句话结论

**以当前“组件拼装”的表述，它还不足以成为主论文的新颖性主张。** 半马尔可夫事件、关键步骤、接触/可供性条件、有限离散 token、部分 OT、NULL 匹配、多段子序列匹配、作用等价分组和多时刻手姿态，均已有直接或强相邻工作。仍可能守住的不是其中任一组件，而是一个很窄、可证伪的交叉命题：

> **在 action-centered WAM 中，一个严格有限、可审计、动作可见的“角色—接触—关系—作用变化”事件管道，能否作为被动人类视频到机器人策略的充分跨具身接口；跨域匹配只在训练前离线完成，允许 NULL 与多对多，部署时既不输入人类视频，也不暴露密集人体轨迹或软跨域 latent。**

这个命题只有在三组决定性实验全部成立后，才有主论文潜力。否则更像一个认真设计的系统/表示消融，而不是新的学习原理。

## 1. 审查口径与证据层级

本轮先检索个人 Zotero，再用 arXiv、OpenReview、PMLR、CVF、NeurIPS 和期刊官网核对年份、venue 与原始链接。Zotero 中直接命中的条目包括：

| 工作 | Zotero key | Zotero 中的状态 |
|---|---:|---|
| EgoBridge | `A89APIR3` | 2025，H2R CoRL workshop / OpenReview |
| XSkill | `V7LLF6VJ` | CoRL 2023 |
| UniSkill | `EK2CTN7J` | CoRL 2025 / arXiv:2505.08787 |
| ImMimic | `8WMVVJAL` | CoRL 2025 / PMLR；本地全文附件 `YNC7LD3C` |
| GigaWorld Policy | `DIETABIS` | 2026 arXiv |
| GigaWorld-0.5 | `726MZ5TZ` | 2026 arXiv |

以下判断区分“正式顶会/顶刊”“workshop”“仅 arXiv/under review”。不能把新近预印本写成已录用工作，也不能因为其尚未正式发表而忽略实质撞车。链接与状态核对截至 2026-08-12。

## 2. 现方案中究竟有哪些可审查主张

根据当前 v2-min 设计，候选贡献可拆为五层：

1. **有限硬事件接口**：四类角色节点、当前事件加未来两个事件、每事件至多两条边、13 个硬 token / 82 个 `uint8` 字段；表示 approach、precontact、acquire、hold、manipulate、regrasp、release、verify-outcome 等事件。
2. **时序手部证据**：不以单帧手姿态代表工具操作，而用 entry/turn/exit 三个操作姿态作为训练辅助证据。
3. **离线 CAPOT 配对**：人类与机器人事件允许部分、NULL、多对多匹配；配对是 stop-gradient 的离线表，不进入部署接口。
4. **关系—作用分组**：训练时把表面动作或对象不同、但关系变化和结果相同的片段组成正对/等价组。
5. **WAM 因果接入**：部署时不需要人类视频或配对信息；人类姿态与密集轨迹也不直接进入 action route。

第 1–4 项分别看都高度拥挤。第 5 项与前四项同时满足，才构成尚可能保留的组合性空隙。

## 3. 最强撞车矩阵

### 3.1 人类视频、跨具身技能与机器人控制

| 工作 | 年份 / venue | 与本方案的直接重叠 | 尚可区分处 | 风险 |
|---|---|---|---|---:|
| [XSkill](https://proceedings.mlr.press/v229/xu23a/xu23a.pdf) | CoRL 2023 | 从未标注人类/机器人视频学习跨具身技能原型，并用人类视频提示组合机器人策略 | 学习的是软技能原型和扩散策略，不是显式有限的角色—接触—作用事件契约 | 高 |
| [UniSkill](https://proceedings.mlr.press/v305/kim25d.html) | CoRL 2025 | 从大规模、无需配对的跨具身视频学习 embodiment-agnostic skill；人类视频可提示只用机器人动作训练的策略 | 共享软表征，不强调有限硬语法、NULL 事件配对或训练/部署信息隔离 | 高 |
| [EgoBridge](https://openreview.net/forum?id=Hr6EzaaKQv) | H2R @ CoRL 2025 workshop；[arXiv](https://arxiv.org/abs/2509.19626) | 以联合策略 latent 和 action OT 缩小第一视角人类数据与机器人数据的域差 | OT 与共训练耦合；本方案可强调“离线语义事件表 + 无人类部署输入”，但不能宣称首次 OT bridge | 高 |
| [ImMimic](https://proceedings.mlr.press/v305/liu25b.html) | CoRL 2025；[arXiv](https://arxiv.org/abs/2509.10952) | 人手到机器人关节的动作映射、DTW 对齐、MixUp 插值、多任务和多具身模仿 | 主要对齐动作轨迹；本方案若成立，是只迁移接触/作用事件而不追随完整运动轨迹 | 极高 |
| [MOTIF](https://arxiv.org/abs/2602.13764) | 2026 arXiv，未核实正式 venue | VQ 统一 action motifs、progress-aware alignment、embodiment-adversarial 表征、在线 motif 预测和 flow policy | MOTIF 是学习型 motif code；SM-FET 是人工约束、可逐字段审计的事件协议 | 极高 |
| [MimicFunc](https://proceedings.mlr.press/v305/tang25a.html) | CoRL 2025 | 从人类 RGB-D 工具操作抽取 function-centric 3D keypoints/function frame，并泛化到新工具与功能等价任务 | 依赖 RGB-D/3D 功能坐标；本方案计划用有限 2D 角色—接触—作用事件 | 极高 |

**审稿结论：**“从人类视频学习跨具身技能”“不要求严格时序配对”“用功能而非外观建立对应”都不是新主张。真正的差别必须落在：不重建人的连续动作、不把软跨域 latent 暴露给动作头、只传递可审计事件关系，并证明这比已有软 skill/motif/trajectory route 更有效。

### 3.2 工具使用、可供性、接触和空间—时间接口

| 工作 | 年份 / venue | 直接撞车点 | 对本方案的约束 | 风险 |
|---|---|---|---|---:|
| [CoA-VLA](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.pdf) | ICCV 2025；[arXiv](https://arxiv.org/abs/2412.20451) | 以对象、抓取部位、放置区域、避碰移动构成 visual-text chain-of-affordance，并注入动作网络 | 不能宣称首次把“操作谁、抓哪里、怎么移动”显式结构化后输入 VLA/WAM | 极高 |
| [VLAff](https://arxiv.org/abs/2608.05215) | IROS 2026 accepted | 从第一视角人类视频估计 3D 几何和手网格，编码位置、抓取姿态与轨迹，映射到机器人动作 | 不能宣称首次从 egocentric video 学 where/grasp/move；区别只能是无需 3D 重建的有限事件接口 | 极高 |
| [ST-VLA](https://arxiv.org/abs/2603.13788) | 2026 arXiv，文稿标注 under review | 统一 3D–4D 表征、时间与几何指导、30 万人类片段、在线重规划 | “空间—时间人类操作表示作为控制接口”已被覆盖 | 高 |
| [BridgeACT](https://arxiv.org/abs/2604.23249) | 2026 arXiv，未核实正式 venue | 人类视频、tool-target affordance、抓取位置与移动方式分解、3D motion affordance、闭环控制和组合泛化 | 工具—目标功能组合及人到机器人 affordance bridge 都不能宣称首次 | 极高 |
| [DexMani](https://arxiv.org/abs/2608.00554) | 2026 arXiv，未核实正式 venue | 把灵巧操作建模为连续接触问题；从人类触觉手套序列获得 contact-conditioned manipulability evolution，指导多种机器人手 | 不能宣称首次利用“支持—释放—再接触”的连续接触演化；SM-FET 必须明确不同于触觉/运动学可操作性先验和逐手 RL | 极高 |
| [Contact-Anchored Policies](https://arxiv.org/abs/2602.09017) | 2026 arXiv，未核实正式 venue | 用接触点替代语言作为模块化策略条件，并讨论跨具身复用 | “contact anchor 作为 policy condition”不是新贡献 | 高 |
| [GeCCo](https://arxiv.org/abs/2509.17582) | 2025 arXiv，未核实正式 venue | 低层通用接触跟踪策略与高层接触规划器 | 领域主要是四足 loco-manipulation，直接性较低；但 contact planner / contact-conditioned utility policy 仍非新概念 | 中 |

说明：文献中另有题名为 “Spatial Traces: Enhancing VLA Models...” 的同名 ST-VLA。这里按任务给出的近期 4D-aware 线索，比较的是 [arXiv:2603.13788](https://arxiv.org/abs/2603.13788)，不能把两篇同名工作混为一篇。

### 3.3 WAM、离散动作表示和事件记忆

| 工作 | 年份 / venue | 直接撞车点 | 对本方案的约束 | 风险 |
|---|---|---|---|---:|
| [GigaWorld Policy](https://arxiv.org/abs/2603.17240) | 2026 arXiv | action-centered WAM；训练期预测未来视觉，推理只解码动作；因果 mask 阻止未来视频 token 泄漏到动作 | “训练期视觉监督、部署只看动作条件”本来就是底座性质，不应包装成本方案贡献 | 极高 |
| [GigaWorld-0.5](https://arxiv.org/abs/2607.13960) | 2026 arXiv | MoT 专家化的 WAM 扩展 | 本方案应作为兼容的事件接口，而非重述 WAM 架构贡献 | 中 |
| [WorldScape Policy 2.0](https://arxiv.org/abs/2607.18840) | 2026 arXiv | 短视觉记忆、长短事件记忆、事件边界和进度检索、近五百万 event segments | “给 WAM 加事件级记忆/进度 grounding”不能宣称首次 | 极高 |
| [OA-WAM](https://arxiv.org/abs/2605.06481) | 2026 arXiv | object-addressable WAM；持久对象地址与变化内容分离，并用可干预 slot 控制动作 | “token 告诉 WAM 当前操作对象”不能宣称首次；必须证明接触/关系/作用变化相对 object address 有新增价值 | 极高 |
| [VQ-BeT](https://proceedings.mlr.press/v235/lee24y.html) | ICML 2024；[arXiv](https://arxiv.org/abs/2403.03181) | 残差向量量化连续动作，支持多模态行为生成与部分观测 | 离散 action/behavior token 或 codebook 本身不是新贡献 | 高 |

另外，`82 uint8` 与 `207.749746919520 bits` 只能说明当前语法的最大编码容量，**不能据此声称获得了信息瓶颈、最小充分统计量或有效语义压缩**。这些需要任务互信息、码利用率、最小描述长度或等容量对照实验支持。

### 3.4 部分、多对多和时序对齐

| 工作 | 年份 / venue | 已解决内容 | 对 CAPOT 的影响 | 风险 |
|---|---|---|---|---:|
| [TCC](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html) | CVPR 2019；[arXiv](https://arxiv.org/abs/1904.07846) | 通过 temporal cycle consistency 学跨视频阶段对齐，并讨论 first-touch/lift 等关键时刻 | 超出近五年但属奠基工作；不能把“跨视频阶段对应”当新问题 | 高 |
| [Drop-DTW](https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html) | NeurIPS 2021；[arXiv](https://arxiv.org/abs/2108.11996) | 可微动态规划，在对齐公共信号时丢弃任意穿插的离群片段 | skip、NULL、无关背景过滤不是 CAPOT 的新意 | 极高 |
| [REMAP / 原题 REALIGN](https://arxiv.org/abs/2509.24382) | 2025–2026 arXiv，未核实正式 venue | partial/unbalanced fused Gromov–Wasserstein OT、virtual sink；处理无关帧、重复动作、非单调程序顺序 | partial OT、unmatched mass、高阶时序结构都已直接覆盖；除非有新目标函数/性质，否则不能称新 OT 算法 | 极高 |
| ImMimic 的 GMS-SDTW | CoRL 2025 supplement | 在长未裁剪人类视频上滑动不同长度窗口，贪心取出多个与单条机器人轨迹匹配的子序列，跳过无关动作 | “不看全程、从长视频找多个相关操作段”已有非常直接先例 | 极高 |
| [Procedure Learning via Regularized GW-OT](https://openaccess.thecvf.com/content/WACV2026/html/Mahmood_Procedure_Learning_via_Regularized_Gromov-Wasserstein_Optimal_Transport_WACV_2026_paper.html) | WACV 2026 | 用正则化 Gromov–Wasserstein OT 学程序结构 | “用 OT 得到 procedure/event structure”也不是空白 | 高 |

REMAP 的 arXiv v2 当前题名是 **REMAP: Regularized Matching and Partial Alignment of Video Embeddings**；搜索索引和 v1 常仍显示 **REALIGN: Regularized Procedure Alignment ... via Partial Gromov-Wasserstein OT**。论文中应说明是同一 arXiv 条目的版本演化，不能误列为两个基线。

CAPOT 目前最安全的定位是：**针对硬语义事件表的、离线的、工程化 partial/many-to-many adjudication protocol**。除非目标函数、约束、求解或统计性质确实不同并在通用 alignment benchmark 上超过 REMAP/Drop-DTW，否则不要称“新的 OT 方法”。

### 3.5 procedure learning、事件分割与关键阶段

| 工作 | 年份 / venue | 对本方案的重叠 |
|---|---|---|
| [StepFormer](https://openaccess.thecvf.com/content/CVPR2023/html/Dvornik_StepFormer_Self-Supervised_Step_Discovery_and_Localization_in_Instructional_Videos_CVPR_2023_paper.html) | CVPR 2023 | 从大量无关帧中用稀疏 slots 发现并定位关键步骤 |
| [STEPs](https://openaccess.thecvf.com/content/ICCV2023/html/Shah_STEPs_Self-Supervised_Key_Step_Extraction_and_Localization_from_Unlabeled_Procedural_ICCV_2023_paper.html) | ICCV 2023 | 无标签程序视频的关键步骤提取，融合 flow/depth/gaze 等线索 |
| [Paprika](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Procedure-Aware_Pretraining_for_Instructional_Video_Understanding_CVPR_2023_paper.html) | CVPR 2023 | procedure-aware 预训练、步骤概念与顺序 |
| [EgoExoLearn](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_EgoExoLearn_A_Dataset_for_Bridging_Asynchronous_Ego-_and_Exo-centric_View_CVPR_2024_paper.html) | CVPR 2024 | 120 小时异步 ego/exo 程序活动、凝视和细粒度标注 |
| [Learning to Predict Activity Progress](https://openaccess.thecvf.com/content/CVPR2024/html/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.html) | CVPR 2024 | 自监督对齐下的进度预测，并处理重复、背景与冗余帧 |
| [Invariant HSMM for Robot Imitation](https://arxiv.org/abs/1811.07489) | 2018 arXiv | 以 invariant hidden semi-Markov model 对示范分段并用于机器人模仿 |

因此，“连续工具操作应由多个阶段而不是一帧姿态表示”是**正确的研究动机，不是新颖性本身**。approach/acquire/manipulate/release 等阶段、事件持续时间、顺序和关键步骤均有长期先例。可保留的区别是：这些阶段被编译成固定容量、动作可见而人姿态不可见的 role/contact/effect packet，并服务于跨具身 WAM，而不是一般视频理解或 event memory。

### 3.6 关系—作用等价

两条文献线直接限制该主张：

- [Affordance Equivalence](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00026/full)（Frontiers in Neurorobotics 2018）已形式化对象—动作—作用间“达到相同作用即可等价”的关系。
- [Cross-Embodied Affordance Transfer through Learning Affordance Equivalences](https://arxiv.org/abs/2404.15648)（2024 arXiv，稿件写明 submitted to IEEE TCDS，未核实正式接收）已把相同 object-action-effect latent 用于不同 agent/object 之间的跨具身 transfer/direct imitation。

所以 training-only relation-effect grouping 应被写成**采用既有功能等价先验构造训练正对或辅助监督**，而不是贡献点。能成为贡献的是对 WAM action route 的因果约束及其带来的数据效率/跨具身增益。

## 4. 哪些 claim 一定不能说

以下句式会被上述文献直接击穿，应从摘要、引言、贡献列表和 rebuttal 预案中禁止：

1. “首次从第一视角/人类视频学习机器人操作。”
2. “首次学习跨具身技能表示/首次无需严格成对的人—机器人学习。”
3. “首次用接触、接触点或接触变化作为策略条件/跨具身先验。”
4. “首次用人类接触的连续演化指导机器人灵巧操作。”
5. “首次把操作对象、抓取位置和移动方式结构化地送入 VLA/WAM。”
6. “首次 object-addressable WAM”或“首次用 token 指示当前操作对象”。
7. “首次给 WAM 建立事件记忆、事件边界或进度表示。”
8. “首次以半马尔可夫事件、阶段、关键步骤或稀疏片段表示操作流程。”
9. “首次用离散 skill/action token、codebook 或 motif 表示操作。”
10. “首次 partial/many-to-many/unbalanced OT 人机对齐。”
11. “首次允许 NULL、跳过准备动作/背景动作，或从长视频找出多个相关子序列。”
12. “首次按结果/作用等价来分组不同对象或动作。”
13. “首次不用单帧而使用多帧/连续手姿态与接触变化。”
14. “人类学习工具操作主要或只看 entry/turn/exit 姿态。”这目前是待检验设计假设，不是已建立的认知事实。
15. “82 字节/207.75 bits 证明是充分、最小或可解释的信息瓶颈。”容量上界不等于统计充分性。
16. “CAPOT 是新的 OT 算法。”除非确有可表述的数学新目标、性质、求解与标准 benchmark 优势。
17. “完全端到端。”只要事件编译、标注或配对是离线/人工裁决，就不能无条件这样写。
18. “无需机器人数据。”如果目标策略仍由机器人 action demo 学习，这一说法就是错误的。
19. “当前数据已经证明 paired transfer。”现有 66 段 human pouring 和 88 段旧 robot grabbing 没有正任务重叠，只能用于本体开发和 hard negative。
20. “证明跨具身泛化。”如果只在一套硬件/末端执行器上验证，最多是 human-to-robot domain transfer，不能外推到多机器人 embodiment。
21. “解决透明物体操作”或“解决双臂灵巧操作。”除非透明实验器皿、双臂角色交换及底层执行均被直接验证。

## 5. 最窄可守的新颖性

### 5.1 建议只守一个中心命题

> **Finite event bottleneck for human-to-robot world-action transfer**：对 action-centered WAM，部署时动作头只看到一个有严格语法和容量上界、可逐字段干预的角色—接触—关系—作用事件包；人类手部轨迹只在训练期帮助提议 entry/turn/exit 事件，跨域对应只在离线事件级执行且允许 NULL/多对多。该接口在相同动作可见容量下，比单帧手姿态、密集轨迹、软 latent skill/motif 与对象地址更能保留对新对象/新工具/新具身有用的因果操作信息。

这里真正可能新的是四个条件的**交集**：

1. **硬且有限**：不是可任意吸收外观/身份/轨迹的软 latent，而是有固定角色、接触、关系变化和结果字段的 canonical contract。
2. **动作可见信息受到审计**：人类 RGB、手姿态轨迹和跨域配对 ID 不可旁路进入动作头；对 packet 的字段交换应产生可预测的动作变化。
3. **匹配是事件级且离线**：只匹配对作用有意义的片段，允许 NULL、多对多和速度差；这是一项训练数据编译协议，不冒充通用新 OT。
4. **部署完全脱离人类参照**：人类视频仅改善训练期知识迁移；部署输入与推理成本不依赖人类 prompt、配对轨迹或 3D 人手重建。

### 5.2 不能拆开单卖

- 单卖“有限 token”：会撞 VQ-BeT、MOTIF 与大量 action tokenizer。
- 单卖“事件”：会撞 procedure learning、WorldScape event memory。
- 单卖“接触”：会撞 Contact-Anchored Policies、DexMani、CoA-VLA。
- 单卖“部分配对”：会撞 Drop-DTW、REMAP、GMS-SDTW、EgoBridge。
- 单卖“作用等价”：会撞 Affordance Equivalence 与 MimicFunc。
- 单卖“三姿态”：最多是合理 inductive bias，不足以构成主贡献。

### 5.3 更安全的论文措辞

可以说：

> Existing methods commonly transfer dense motion, learned skill latents, 3D affordances, object addresses, or progress-aligned motifs. We instead test whether a strictly finite and auditable role–contact–effect event contract is sufficient for training-only human-to-robot transfer in an action-centered world-action model.

不要说：

> We are the first to align human and robot manipulation through contact-aware temporal events.

建议标题方向：

- **What Should a Robot Copy from Human Tool Use? Finite Event Bottlenecks for World-Action Models**
- **Copy the Effect, Not the Motion: Auditable Event Bottlenecks for Human-to-Robot WAMs**

## 6. 是否足以成为主论文

### 当前判定：否，属于“有潜力但尚未过 novelty gate”

作为设计文档，它比“复制整段人体运动”更清楚，也非常适合作为强实验假设；作为主论文，当前仍有四个致命缺口：

1. **组合新颖性风险**：审稿人很容易把它概括为“semi-Markov events + contact affordance + partial OT + auxiliary pose + WAM”。
2. **CAPOT 算法性不足**：如果只是给现成 partial OT 添加手工 event cost 和离线阈值，它是配对协议，不是算法贡献。
3. **充分性没有证据**：固定 schema 是否真的优于 soft latent、dense trace、object slot 或同容量随机 hard code，尚无因果证据。
4. **跨具身结论的数据不足**：36+36 只能做 feasibility；已有 66 human pour + 88 robot grab 没有正重叠，不能证明跨域正迁移。

### 何时可升级为主论文

必须同时满足：

- 相同 backbone、参数、训练预算、机器人数据和动作可见 bit budget 下，SM-FET 显著胜过强软 latent/trajectory/object-address 基线；
- CAPOT 在含 prep/idle、重复、跳步、重排、速度差和遮挡的 gold alignment 上胜过 Drop-DTW、GMS-SDTW、EgoBridge-style OT 与 REMAP，并且 alignment 改善能转化成下游成功率；
- 新人类数据在固定机器人 demo 数量时对 held-out tool/effect/embodiment 有稳定、统计显著的增益；
- 因果 intervention 证明动作确实依赖事件语义，而不是数据量、额外参数、隐式 identity 或视频旁路；
- 至少三个工具作用家族，且有两个末端执行器/机器人形态，或严格把 claim 限制为 human-to-one-robot transfer。

若只能完成其中一两项，建议将论文定位降为：**一个可审计的人类视频监督接口/benchmark + 负结果与设计准则**，而不是“新的跨具身学习范式”。

## 7. 三个决定性实验

以下不是普通 ablation，而是能决定论文是否成立的三项实验。

### 实验 1：等容量、等预算的因果表示对决

**问题：收益来自语义事件结构，还是来自多了一路 token/参数/监督？**

固定同一个 GigaWorld Policy backbone、机器人数据、参数量、优化步数、文本/本体输入和噪声设置，并严格匹配 action-visible capacity。比较：

1. WAM 原始基线，无事件接口；
2. 单帧手姿态/接触；
3. 稠密手/物轨迹或 soft cross-embodiment latent；
4. 旧 22D contact 表示；
5. SM-FET v2-min hard packet；
6. 等 bit 的随机/无结构 hard code；
7. 打乱 contact、relation、effect 字段的 SM-FET；
8. OA-WAM/CoA-VLA 风格的对象地址或目标—抓取—移动接口。

必须增加三类 intervention：

- **字段交换**：只交换 focus/contact/effect，动作应出现语义一致的变化；
- **固定 packet**：packet 相同时更换人类参考视频或配对 ID，动作输出应保持不变；
- **梯度与信息审计**：验证人类视频、pose auxiliary、pair ID 到 action route 不存在旁路或非零梯度。

主指标是实机任务成功率、effect completion、接触错误/掉落/洒漏/碰撞；辅以 calibration、码利用率和干预效应大小。只有 hard semantic packet 胜过同容量控制，才能支撑“有限事件瓶颈”而不是“额外条件通道”。

### 实验 2：时序证据与 partial/many-to-many 对齐基准

**问题：CAPOT 和 entry/turn/exit 是否真正解决了“不应关注全程、也不能只看一帧”的难点？**

建立人工 gold 的 event/NULL/many-to-many 对齐集，系统加入：

- prep/idle 段插入；
- 速度扭曲与长短差；
- 重复 regrasp / 多次倾倒；
- 跳过阶段；
- 非关键子步骤重排；
- 部分遮挡和出视野；
- 一个 human event 对多个 robot event，反之亦然。

比较 full DTW、TCC、Drop-DTW、ImMimic GMS-SDTW、EgoBridge-style OT、REMAP 和 CAPOT。报告 event boundary F1、pair precision/recall、NULL F1/AUROC、many-to-many recovery、transport calibration，以及保留配对训练后的下游增益。

同时进行时序姿态消融：single pose、entry+exit、entry+turn+exit、dense pose trajectory、完全无 pose。手姿态始终只做训练辅助，不进入 action route。预期不是“三帧一定最好”，而是验证最小的阶段证据是否足以识别 contact transition；若 dense trajectory 更好且无跨具身副作用，就不能预设 v2-min 的三姿态设计成立。

### 实验 3：固定机器人数据的人类信息因果增益

**问题：人类数据是否真的提供了机器人数据中没有的可迁移操作知识？**

用完全相同的 robot demos 做因子实验：

1. robot-only；
2. `+human` 但不配对；
3. `+human` full-trajectory/DTW；
4. `+human` CAPOT；
5. CAPOT 去掉 relation-effect grouping；
6. CAPOT 去掉 entry/turn/exit auxiliary；
7. 完整 SM-FET-WAM。

测试 split 必须按 operator/day/object/tool 隔离，并包含至少三个作用家族，例如容器倾倒/液体转移、带工具刮取或舀取、夹持—旋转—释放。理想设置还需两个末端执行器/具身条件，例如 RH56DFTP 与 OmniPicker，或 G1 双臂角色交换；否则把结论限制为单机器人 domain transfer。至少一个测试族应包含透明实验器皿，单独报告视觉失效模式，不把它混在平均分里。

报告成功率、最终关系/作用完成、接触错误、洒漏、掉落、碰撞、时间和 human-data scaling curve。最关键的检验是：在机器人 demo 数量严格相等时，CAPOT+SM-FET 是否对 held-out effect/tool/embodiment 提供稳定增益。如果增益只发生在 seen object，或增加等量 robot demos 后消失，这不是足以支持主论文的跨具身知识迁移。

## 8. 数据与主张边界建议

- `36 human + 36 robot`：足够验证标注协议、编译器、CAPOT 退化情形和小规模因果 sanity check，不足以支撑主论文成功率或泛化结论。
- `66 human pouring + 88 old robot grabbing`：可用于本体/事件定义、负匹配和 domain-gap 分析；因任务不重叠，不能作为正配对收益证据。
- 主论文最低可辩护规模不宜只看 clip 数，而应覆盖**至少三个 effect family、每族多个工具/对象、严格 held-out 的 operator/day/object/tool**。若声称 cross-embodiment，至少两个真实末端执行器或机器人形态。
- 人类—机器人不应从拿取前一直一一对应。可把“完整连续轨迹复刻”作为负基线，但 gold 配对应围绕接触建立、关系转折、接触解除和结果确认，并显式允许 unmatched mass。
- 不要为了对齐漂亮而把 robot event ontology 强行投影成人体运动学。人类证据负责提出功能事件，机器人事件负责满足自己的可执行约束。

## 9. 审稿人可能给出的最终意见

### 如果按当前组件列表投稿

> Weak reject：问题重要，系统设计细致，但每一组件均与既有跨具身 skill/motif、affordance/contact conditioning、procedure segmentation、partial OT 和 WAM event/object interface 高度重叠。论文没有隔离出不可由既有组件直接组合得到的科学主张，也未证明手工有限 schema 比同容量 learned representation 更有效。

### 如果三个决定性实验成立

> Borderline accept / accept：论文没有再提出一个更大的通用 latent，而是以严格信息路由和等容量因果干预，证明 action-centered WAM 只需一小组 role–contact–effect events 就能吸收训练期人类视频知识；部署时不依赖人类 prompt、密集轨迹或 paired identity。贡献是关于“跨具身控制应复制什么”的实证结论，而不是又一个 OT 或事件分割算法。

## 10. 给主线决策者的行动建议

1. 保留 v2-min，但把它明确写成**待检验的有限事件瓶颈**，不要先称为完整理论。
2. CAPOT 降格为 offline pairing protocol；先做标准强基线审计，再决定是否有资格升格为算法贡献。
3. relation-effect grouping 和 entry/turn/exit pose 都放到 training-only auxiliary，贡献列表中只作为实现中心命题所需的约束。
4. 立项优先级应是实验 1 的等容量因果对决，其次是实验 2 的 gold alignment，再投入大规模数据做实验 3。
5. 若资源只够一次大实验，优先证明：**固定 robot data 后，human CAPOT event supervision 对 held-out tool/effect 的增益，并且该增益在 soft latent、dense trajectory 和 object-address 等容量强基线下仍存在。**

最终建议：**继续找 idea，但应从“再加模块”转向“找能击穿该中心命题的反例和等容量基线”。** 当前最值得保留的不是 13 token、CAPOT 或三姿态本身，而是“机器人无需复制整段人的动作，只需要可审计的接触—关系—作用转折；且这个信息必须通过严格受限的 WAM action route 才产生可迁移收益”这一可证伪问题。
