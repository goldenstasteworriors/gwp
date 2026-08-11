# Idea H：Before–During–After Hand–Tool–Object Event-State-Transition WAM

> 日期：2026-08-12
> 工作树：`/home/ykj/project/WM/gwp-agent-idea-h-event-transition`
> 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
> idea-discussion `idea_id`：`f759ccbb-735b-45e4-8efe-158b79a3ce90`
> idea-discussion `discussion_id`：`2ca05e6f-75c5-49df-bbc9-2c8a68d292c9`
> 工作性质：Zotero-first 文献核查、原论文复核、只读代码审计和研究设计；未训练、未调用 GPU、未修改 canonical 工作树。

## 0. 先给决定

**决定：研究方向 Conditional Go；当前直接扩 5B WAM、宣称跨 embodiment 收益或投稿 ready 均 No-Go。**

推荐的主架构是 **Event-Transition Graph WAM（ETG-WAM）**：把一段操作压成最多四类事件窗口
`pre-contact relation → contact acquisition → manipulation contact dynamics → post-effect/release`，
每个事件预测手/工具/目标/容器或 fixture 的可观察 2D 关系图、22D contact segments、边界删失状态和
post-effect/outcome；action expert 每次只接收“当前阶段 + 下一个尚未实现的状态差 packet”，而不是整段
human trajectory 或不可审计的 future hidden。

本方向的合理 claim 不是“首次用 before/after、contact、graph、affordance 或 world model 学工具操作”。
这些单项均有直接前例。可守空间只能收窄为：

> **在 action-centered joint video/action WAM 中，把工具操作的可观察事件状态差、contact dynamics、
> missingness/censor 和 outcome 组成一个有限、角色化、可干预的 action-visible packet；用严格旁路审计、
> 同容量 controls 和人机事件级部分配对，检验这种 packet 是否比单帧 affordance 与 full-trajectory imitation
> 更能跨 embodiment 保留“谁作用于谁、何时接触、产生了什么结果”。**

最大的 novelty 风险不是 VLAff，而是三条合流：

1. [Ego4D](https://arxiv.org/abs/2110.07058) 已把 hand-object state change 标成 PRE、CONTACT、PNR、POST；
2. [Structured World Models from Human Videos（SWIM）](https://arxiv.org/abs/2308.10901) 已从 human video
   学 grasp/post-grasp affordance action，并在 affordance space 训练 world model；
3. [BridgeACT](https://arxiv.org/abs/2604.23249) 已显式建模 tool/target role、contact region 和相对 3D point flow。

因此 ETG-WAM 只有在以下事实同时成立时才值得升为主方法：

- `oracle event graph` 对 action 足够；
- predicted ETG 比 22D-only、single-frame、full-trajectory 和 equal-capacity latent code 稳定提升；
- 提升来自 pre/post state delta，而不只是新 auxiliary supervision、更多标签或 task-progress shortcut；
- 不依赖透明物体/液体的伪 3D 重建；
- 人机配对按事件/角色/outcome 生效，并在 object/operator/session group-heldout 上成立。

## 1. 证据边界与检索记录

本文使用以下标记：

- **[Z]**：本地 Zotero 索引命中，只证明个人库中存在该条目，不等于已充分精读；
- **[R]**：arXiv、CVF、PMLR、NeurIPS、RSS 或项目官方原始页面已复核；
- **[I]**：当前基线代码已经实现；
- **[S]**：本文建议的设计或尚未实现的规范；
- **[M]**：当前数据/证据缺失。

### 1.1 idea-discussion 结果

按 skill 先执行了 `--sync-default`，同步 100 个 Zotero 条目并创建 idea card。自动检索给出的前三项为
RoboBrain 2.0、RoboBrain 2.5 和 FAST，外部结果甚至混入图像篡改检测，相关性不足，不能作为本题的
novelty 证据。随后直接复查同步后的只读 Zotero 索引，并以题名/摘要定向检索。

高相关个人库条目包括：

| Zotero key | 条目 | 对本题的约束 |
|---|---|---|
| `726MZ5TZ` | [GigaWorld-Policy-0.5](https://arxiv.org/abs/2607.13960) | action-centered WAM、future visual training 和 action-only inference 已有；Idea H 只能改变中间预测目标/route。 |
| `XINDFWEJ` / `653NU4JX` | [Being-H0.7](https://arxiv.org/abs/2605.00078) | future-informed latent query 位于 perception 与 action 之间；“先压缩未来再行动”不是新意。 |
| `CV5XA2LA` | [UniVLA](https://arxiv.org/abs/2505.06111) | 从异构/人类视频学 task-centric latent action；纯 latent option 结构已很拥挤。 |
| `RALYIRFV` | [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 大规模 action-free video pretrain + 少量 robot action-conditioned post-train 可做 planning。 |
| `JRGPD6HS` | [OmniContact](https://arxiv.org/abs/2606.26201) | contact flow + meta-skill chaining 已有；continuous contact tube 不能单独作 novelty。 |
| `FJFKKNLF` | [DreamDojo](https://arxiv.org/abs/2602.06949) | 大规模 ego video、continuous latent actions 和 robot post-training 已覆盖 latent world model 路线。 |
| `WVMXS4GN` | [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998) | joint frame/action prediction 和闭环 world-model control 已有；仅称“causal”不足。 |

个人库未直接命中 VLAff、BridgeACT、ST-VLA、CoA-VLA、SWIM、ChangeIt 或 Action Genome。这个缺口只说明
需要外部补检，不能证明没有前例。

### 1.2 arXiv skill 状态

`arxiv-search` skill 的 CLI 入口没有打印查询返回值，且项目外系统 Python 缺少 `arxiv` 包。为避免向
base 安装依赖或为纯调研新建无关环境，本轮没有安装包，改用 arXiv/CVF/RSS/NeurIPS 等官方原始页面。
idea-discussion 自身的 arXiv-first retrieval 已执行，但其宽查询结果仅作检索线索。

## 2. 最接近前例：Idea H 不能再声称什么

### 2.1 状态变化、before/after 与事件图

| 原始工作 | 已经做了什么 | 对 Idea H 的直接限制 | 仍可能保留的差异 |
|---|---|---|---|
| [Ego4D](https://arxiv.org/abs/2110.07058) / [Hands & Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/) | 对 state-changing hand-object clips 标 PRE、CONTACT、PNR、POST，含 hand/tool/state-change object boxes、PNR localization 和 state-change classification。[R] | `before–contact–after` 分段和 state-change keyframe 不是新意；这些标签应直接成为公共 baseline/预训练源。 | Ego4D 是感知 benchmark，不是严格 action-visible WAM route，也没有本题的人机 event-packet 因果对照。 |
| [Look for the Change / ChangeIt](https://openaccess.thecvf.com/content/CVPR2022/papers/Soucek_Look_for_the_Change_Learning_Object_States_and_State-Modifying_Actions_CVPR_2022_paper.pdf) | 用初态、state-modifying action、末态的因果顺序约束，从 2600+ 小时 untrimmed web video 学 object state change。[R] | “从长视频只保留状态前后差”已经有强视觉前例。 | 它不生成可执行 robot action，也不显式建 hand–tool–target contact graph。 |
| [HowToChange](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Learning_Object_State_Changes_in_Videos_An_Open-World_Perspective_CVPR_2024_paper.pdf) | 把 object state change 扩展到开放物体类别，训练集约 36K clips，含 initial/transition/end pseudo labels。[R] | 单纯增加 state-change 类别或开放物体不够新。 | 可作为 object/outcome encoder 预训练源，但 pseudo labels 不能当物理真值。 |
| [Video State-Changing Object Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_Video_State-Changing_Object_Segmentation_ICCV_2023_paper.html) | 只标训练视频首末帧，研究对象跨 state change 的一致分割和 open-vocabulary 泛化。[R] | first/last mask 与 before-after contrast 也已有。 | 适合验证 object identity across change；不含动作/contact。 |
| [Action Genome](https://openaccess.thecvf.com/content_CVPR_2020/html/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.html) | 把 action 分解为随时间变化的 object-relation scene graphs，10K videos、1.7M relations。[R] | “用时空 HOI graph 表示事件变化”不是新意。 | 本题可把 graph 从识别表征变为经过审计的 action interface，并加入工具角色/contact/outcome/censor。 |
| [Egocentric Action Scene Graphs](https://openaccess.thecvf.com/content/CVPR2024/html/Rodin_Action_Scene_Graphs_for_Long-Form_Understanding_of_Egocentric_Videos_CVPR_2024_paper.html) | 为 Ego4D 长视频标 temporally evolving ego action graphs。[R] | ego event graph 和 long-form step representation 已有。 | 仍不是 robot-control sufficiency/causal route 评测。 |
| [Recognizing Manipulation Actions from State-Transformations](https://arxiv.org/abs/1906.05147) | 用 pre-state/post-state transition matrix 识别 EPIC-KITCHENS manipulation actions。[R] | 从对象初末态反推动作是明确前例。 | 本题方向相反：由当前观测预测未来 transition，再约束可执行 action。 |
| [What Changed and What Could Have Changed?](https://arxiv.org/abs/2503.21055) | 用 before/after state-change 描述和 LLM 生成的 failure counterfactual 做 procedure-aware contrastive representation learning。[R] | before/after contrast、state-change text 和 counterfactual negative 均不是新意。 | LLM counterfactual 只能作 representation negative，不能当物理因果或 robot outcome truth。 |

### 2.2 Affordance、action effect、subgoal 和 world model

| 原始工作 | 已经做了什么 | 冲突强度与剩余空间 |
|---|---|---|
| [SWIM](https://arxiv.org/abs/2308.10901)（RSS 2023） | 从 human video 建 human-centric structured action space（grasp/post-grasp affordance），在人类视频预训练 world model，再用少量 robot interaction fine-tune。[R] | **极强冲突**。不能声称首个 human-video affordance-state world model。Idea H 必须证明显式 pre/during/post state delta、contact/censor/outcome 和严格 route 比 grasp/post-grasp pixel action 多提供了什么。 |
| [VRB](https://openaccess.thecvf.com/content/CVPR2023/html/Bahl_Affordances_From_Human_Videos_as_a_Versatile_Representation_for_Robotics_CVPR_2023_paper.html) | 预测从 human video 学到的 contact point 与 post-contact trajectory，并用于四类 robot learning setting。[R] | 单帧 interaction hotspot + post-contact path 是必要 baseline；Idea H 的优势必须来自状态差和阶段，而非又一个 contact heatmap。 |
| [HRP](https://www.roboticsproceedings.org/rss20/p068.html)（RSS 2024） | 自动从 human video 提取 hand/object/contact affordance labels 做 robot representation pretraining，3000+ robot trials。[R] | human affordance pretraining 与跨 morphology 收益已有；需同 human-video budget 对比。 |
| [Action-Effect Memory](https://arxiv.org/abs/2606.12499) | 用 vision-action history masked modeling 学 compact action-conditioned state evolution，报告优于 single-frame/frame stacking。[R] | “时间历史优于单帧”已被直接研究。Idea H 要靠可解释事件状态差和跨 embodiment，而不是泛 history encoder。 |
| [SuSIE](https://arxiv.org/abs/2310.10639) | image-editing model 预测 visual subgoal，再由低层 goal-conditioned policy 执行。[R] | visual subgoal/keyframe hierarchy 已有。 | ETG 应预测可评测关系差，而不是 hallucinated goal pixels。 |
| [UniPi](https://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html) | 把 policy 表成 text-conditioned future video plan，再抽取动作。[R] | video-as-policy 与跨 action space 已有。 | ETG 的理由是压掉无关视觉细节，并让每个 predicted change 可监督/干预。 |
| [GHOST](https://arxiv.org/abs/2606.10025)（RSS 2026） | high-level 预测 3D end-effector subgoals，low-level 执行；high-level 可混合 human/robot data，避免动作 retargeting。[R] | human high-level / robot low-level 与 subgoal 部分配对已有。 | ETG 不以 3D EE pose 为共享语义，而以角色化 state transition/outcome 为共享语义。 |
| [Time-Contrastive Networks](https://arxiv.org/abs/1704.06888) / [TCC](https://arxiv.org/abs/1904.07846) | 学 viewpoint-invariant phase embedding、跨视频时间对齐；TCC 在 pouring 阶段标注上评测。[R] | 仅做跨 embodiment/跨速度 phase alignment 不是新贡献。 | 可作 partial pairing baseline/辅助 loss，不能替代 outcome-grounded event labels。 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 用 web video 自监督表征，再以少量 robot video 训练 latent action-conditioned world model做 image-goal planning。[Z/R] | latent future prediction/planning 已覆盖。 | ETG 必须证明显式图状态差的可预测、可校准、可干预收益。 |

### 2.3 工具使用推理

- [ToolTango](https://arxiv.org/abs/2206.09141) 已用 object/relation graph 和 action history 预测工具序列与
  symbolic actions；它约束“tool role + scene graph + sequential plan”的宽泛 claim。[R]
- [RoboTool](https://arxiv.org/abs/2310.13065) 已让 LLM 分析物理约束并生成 tool-use plan/code；它解决高层
  creative tool selection，不解决视频事件对齐或低层 state-transition sufficiency。[R]
- [Few-shot transfer of tool-use skills](https://sony.github.io/tool-use-few-shot-transfer/) 指出工具操作含 robot–tool
  与 tool–environment 两类 contact，并用 proximity/tactile 做少样本迁移；这支持双接触关系标签，但限制
  “首次建模 tool 双接触”。[R]

## 3. 四个指定冲突的逐项审查

### 3.1 VLAff

[VLAff](https://arxiv.org/html/2608.05215) 从 egocentric video 自动提取 visual affordance heatmap、96D MANO
grasp 和接触后的 6D trajectory；EgoAffordance 约 204K episodes、5.8M heatmaps、11.6M trajectory
sequences，并把离散 trajectory tokens 转回可执行动作。[R]

冲突：

- 已覆盖 `where to interact / how to grasp / how to move`；
- 已从 contact keyframe 提取 hand-object interaction；
- 已重建 pre-/post-contact motion，不能再称 Idea H 首次压缩接触前后手姿态；
- 已通过 object-centric anchor 做 embodiment transfer。

差异：VLAff 的输入仍是单图+instruction，输出是热图、MANO grasp 和整段相对 6D trajectory；它没有显式
监督“关系/对象角色/可观察效果的 pre→post delta”，没有窗口删失/unknown 语义，也没有证明 action 的
视觉路径只能经过该 affordance。因此 VLAff 应是 **single-frame+trajectory 强 baseline**，而不是 Idea H
需要复刻的结构。

### 3.2 BridgeACT

[BridgeACT](https://arxiv.org/html/2604.23249) 明确提出绝对人手/物体轨迹不是可迁移信号；可迁移的是谁作
tool、谁作 target、哪里接触、tool 相对 target 如何运动。其部署时预测 role-conditioned 3D point flow，
再经 grasp selection 和 closed-loop SE(3) controller 执行；实验包含 pour/cut/open/close。[R]

这是本题最强任务层冲突：tool/target role、contact、relative motion、human-video-to-robot 都已直接覆盖。
ETG-WAM 只能主张：

- `pre/post relational state + outcome`，而非仅“where/how motion”；
- 2D observable packet + censor，不依赖完整 point cloud/3D reconstruction；
- action-centered WAM 内的 train-time future supervision 和严格 action route；
- 人机 paired event benchmark 和 same-noise causal controls。

若 ETG 的收益最终依赖 action-visible continuous point/pose tube，方法就会退化成 BridgeACT/ST-VLA 的近邻，
应停止独立主架构 claim。

### 3.3 ST-VLA

[ST-VLA](https://arxiv.org/html/2603.13788) 以 3D trajectories 和 smooth spatial masks 连接 high-level VLM
与 low-level control；ST-Human 约 60K videos、300K episodes、14 tasks、4.3M samples。其 core subset 已
标 initial/final 2D contact points、object semantics 和 relative spatial configuration，并把连续 episode 链接
成 task progress/previous-next-step 监督。[R]

冲突：initial/final contact、relative configuration、4D progress、human data hierarchy 都已有。Idea H 不能将
“before/after contact + progress”包装为首次。仍可区分之处是：ETG 不追求统一 3D–4D scene reconstruction，
而只预测任务可观察的关系/效果差，并对不可见字段显式 unknown/censor；action-visible interface 可枚举、
可 swap/clamp，而 smooth mask/3D trajectory 是连续高带宽接口。

### 3.4 CoA-VLA

[CoA-VLA](https://arxiv.org/html/2412.20451) 将 object、grasp、spatial、movement 四类 affordance 串成 chain，
根据 proprioception 动态选择当前需要的 affordance，并以 visual/text co-injection 和 FiLM 影响 DiffusionVLA；
真实任务包含 PourTea/NailHammer/WipeWater，使用 39K DROID trajectories 和 692 条本地 trajectories。[R]

冲突：阶段化 affordance、动态选择和工具任务都已有。区别是 CoA 的 chain 主要回答“当前下一步需要什么
affordance”，visual prompt 可直接包含 bounding boxes/points/trajectories；它不把动作前后世界关系的真实差值
作为预测目标，也未定义 event boundary censor、outcome verification 或 strict action-visible capacity。

必须做的直接对照是：相同 policy/backbone/data 下，`CoA-like current affordance` 对比 `ETG predicted delta`。
如果 ETG 只在增加更多 pseudo labels 后胜出，不能归因于 state-transition structure。

## 4. 为什么事件状态差可能比两种常见路线更适合

### 4.1 相比单帧 affordance

单帧 affordance 能回答“哪里可能抓/接触”，但通常不能区分：

- 接触尚未建立还是已经稳定持有；
- tool 在 target 上产生了 slide/impact/flow，还是只从前景经过；
- open/closed、inside/outside、source→receptacle transfer 等结果是否已经发生；
- 是否应该 release/stop，而不是继续执行同一 action chunk；
- 手或液体不可见时是真 negative 还是 unknown。

事件状态差以 `G^- → G^+` 为监督，要求模型同时解释“关系变了什么”和“哪个 contact dynamics 与结果共现”。
它尤其适合可恢复闭环：当前 graph 未达到 predicted post-state 时继续，达到且 outcome verified 时停止/释放。

### 4.2 相比 full-trajectory imitation

整轨迹包含大量 task-irrelevant variation：伸手前准备、人体搬运路径、视角/head motion、速度风格、冗余腕姿和
robot-specific collision avoidance。它们在人/机器人间通常不应逐帧对齐。事件 packet 保留的是更接近任务不变量
的内容：角色、接触顺序、对象关系变化和可观察 outcome；低层 executor 仍可基于自己的 morphology、state 和
闭环反馈生成轨迹。

但 event packet 不是免费抽象。如果精细动作（切割角、倒液倾角、沿边擦拭）在给定 packet 下仍多解，丢掉
continuous dynamics 会让 action 不可辨识。因此 **oracle sufficiency 必须先于任何大训练**。若 oracle ETG 明显
差于 full trajectory，ETG 只能做 auxiliary/progress verifier，不能作为严格 action route。

## 5. 三个互斥主结构及排序

“互斥”指主实验中 action-visible 的共享表示只能选一种；不得把 graph、continuous tube 和 latent option 全部
拼上后再把收益归给其中任一项。

| 排名 | 结构 | action-visible 共享表示 | 优势 | 主要风险 | 决定 |
|---|---|---|---|---|---|
| **1** | **ETG-WAM** | 有限 event-transition graph packet + 22D contact segments | 最贴合用户 insight；可观测、可干预、可 mask/censor；与现有 strict contact route 相容 | SWIM/Ego4D/Action Genome/BridgeACT 强碰撞；标签成本高；可能欠缺连续几何 | **推荐，Conditional Go** |
| **2** | **Pre/Post Delta Token + Contact/Pose Tube** | 一个 pre/post delta token + during continuous 2D/6D tube | 最容易保留精细 tool motion，执行上可能最强；和现有 WAM latent/action head自然衔接 | 极接近 VLAff、BridgeACT、ST-VLA、OmniContact；连续 route 高带宽且难审计 | **强 baseline/备选，不与 #1 混成主方法** |
| **3** | **Latent Option Boundary Predictor** | 离散 latent option ID + termination hazard | 可从弱标/无标视频扩展，标注成本最低；适合长程 option switching | 与 UniVLA、Being-H0.7、LAPA/DreamDojo 类路线拥挤；语义/因果可解释性最弱 | **主方法 No-Go；仅作 equal-capacity baseline** |

### 5.1 架构 1：ETG-WAM（推荐）

对每段 interaction 预测最多 `K=4` 个事件：

```text
E_k = (phase_k, [b^-_k,b^+_k], G^-_k, C_k, G^+_k, ΔG_k, outcome_k, censor_k)
```

- `phase ∈ {pre_contact, acquire, manipulate, post_effect_release}`；
- `G^- / G^+` 只含 effector/hand、tool、target、receptacle/fixture 等任务实体和可见 2D relations；
- `C_k` 复用当前最多 `Q=12` 个 22D contact segment；
- `ΔG_k` 是关系、角色绑定和 object-effect 的显式变化；
- boundary 用区间或 16-bin hazard，而不是把不精确边界伪造成一个 float；
- continuous rim polyline/mask 仅作 training-time auxiliary；若需 action-visible，只允许量化的 rim relation/cell，
  并重新计算整个 packet 的有限容量。

训练时 posterior 可以看 pre/during/post windows，prior 只能看 current reference/state/text；prior 对齐 posterior 的
event packet。推理时每个闭环 step 重新预测当前 phase 和 next unresolved delta，action expert 只读 canonical hard
packet、state、text 和 noisy action。视觉 future expert 可继续保留现有 flow loss，但其 raw hidden 不进入 action。

推荐不要让 graph 变成任意节点/任意谓词的大 symbolic system。第一版固定四个 role slots、受控 relation alphabet
和四个 event slots；否则 capacity、assignment、标注一致性和 causal controls 都不可控。

### 5.2 架构 2：Pre/Post Delta Token + Contact/Pose Tube

posterior 从两个稳定窗口编码 `z_Δ = f(W_pre,W_post)`，另以 flow/diffusion 回归 `during` 的 contact/pose tube；
prior 由 current context 预测 `(z_Δ, tube)`，action expert 用它们执行。它不显式预测完整 graph，关系和 outcome 只作
probe/auxiliary。

该结构对需要连续倾角、切割路径或 rim-following 的任务更可能有执行优势，但主论文必须承认它本质上是
VLAff/BridgeACT/ST-VLA/OmniContact 邻域。若采用它，应删除有限 semantic transport claim，转而把贡献定位成
`event-localized delta + tube` 的效率/配对协议。

### 5.3 架构 3：Latent Option Boundary Predictor

posterior 从 future windows 学 FSQ/VQ option code，prior 预测 option ID 和 `terminate_t` hazard；option-conditioned action
decoder 产生动作，语义只通过 probes 事后评测。它最适合大量无标签视频，但很难证明 latent 真的表示 contact/outcome
而非背景、速度或 camera motion。

要作为 control，codebook cardinality、option 数、posterior receptive field、planner 参数、action-visible bits 和
训练数据必须与 ETG 尽量匹配。若 latent option 与 ETG 同样好，结论应是 human semantics 非必要；若 latent code
collapse，不能反过来证明 graph 优越。

## 6. 不做透明 3D 重建，如何预测可观察状态差

推荐 observation contract 是 **object-local 2D + window consensus + honest unknown**：

1. **事件窗口而非单帧。** 在 boundary 两侧取 0.25–1.0 s 稳定窗口，使用 track median/attention pooling，降低 blur、
   单帧遮挡和错误 hand detector 的影响。
2. **object-local 坐标。** 手/tool keypoint、mask centroid、rim point 都相对 target/receptacle box 或可见 rim frame
   归一化；先用背景 homography/2D tracks 做相机运动补偿，不要求恢复透明对象/液体 3D mesh。
3. **定性关系优先。** `left/right/above/below/overlap/inside/contains/near/touching/aligned_to_rim` 比绝对深度更稳；
   可附带 masked `dx,dy,area_ratio,orientation_2d` 作为 auxiliary，但不把它们默认送进有限 action route。
4. **状态效果是可观察 predicate。** 例如 drawer `closed→open`、object `outside→inside receptacle`、tool/target
   `separate→contact→separate`、source/target `no_transfer→transfer_detected`。透明液体体积、流速或接触面不可从 RGB
   可靠获得时标 `unknown`，不让 VLM 猜测替代 GT。
5. **结果可延迟。** `post_effect` 窗口不等于固定 release 后一帧；由稳定性判据或传感器时间戳确定。切割/倒液等
   effect 可先于 release，release 本身是另一 transition。
6. **传感器只作 outcome truth。** 秤、触觉、流量/液位、drawer encoder 可监督 coarse outcome；模型输入仍可是
   RGB+state。这样不需要透明 3D 重建，又能避免 RGB 不可见时造假。

必须保留的限制：2D relation 对深度遮挡、相机旋转和相似物体身份交换仍敏感；某些 3D feasibility 只能由 executor、
depth baseline 或真实闭环解决。本文不声称 2D graph 等价于物理 state。

## 7. Label protocol v0

### 7.1 事件与 coverage

每个物理事件必须有稳定 identity，而不是每个 sliding window 新建事件：

| 字段 | 建议定义 |
|---|---|
| `physical_event_id` | 同一 acquisition/manipulate/release 的全局 identity；所有派生窗口共享 |
| `phase` | `pre_contact/acquire/manipulate/post_effect_release` |
| `boundary_interval` | `[lower_frame, upper_frame]`；exact 时两者相等 |
| `boundary_status` | `observed/left_censored/right_censored/unknown` |
| `coverage_scope` | 按 side、relation、field 声明 exhaustive/positive-only/not-reviewed |
| `evidence_tier` | `human_adjudicated/sensor_confirmed/geometry_proxy/model_pseudo` |
| `confidence` | 只作 loss 权重/筛选，不得替代字段 mask |

未标注不等于 negative。只有对应 `(episode, interval, side, relation, field)` 为 exhaustive coverage 时才允许
`presence=0`。

### 7.2 实体、角色和 2D relation

- entity：`left_hand/right_hand/robot_effector/tool/target/receptacle/fixture/support`，实例 ID 与 track ID 分开；
- object role：复用 22D 的 `manipulated/tool/receptacle/fixture`，另在 graph metadata 中区分 actor/patient/source/
  destination；不要直接扩 22D checkpoint schema；
- relation：受控 multi-label alphabet
  `left_of/right_of/above/below/overlap/inside/contains/near/touching/aligned_to_rim/unknown`；
- 每条 relation 必须记录 `visible_src/visible_dst/localizable` 和 per-field mask；
- `relation_delta` 由 adjudicated `pre/post` 独立标签机械生成，不能让标注员直接写一个没有两端证据的 delta。

### 7.3 Rim / aperture

- `rim_presence ∈ {absent,present,unknown}`；
- `rim_visibility ∈ {not_visible,visible_not_localizable,localizable}`；
- localizable 时标 2D polyline/keypoints、内外方向和 object-local frame；
- action-visible v0 只保留 `aligned/not_aligned/inside_projection/unknown` 和量化 cell；
- transparent boundary、liquid surface、stream/contact 分字段标，不能用容器 rim mask冒充液体真值。

### 7.4 Contact、censor 与 outcome

- contact 继续使用当前 22D segment：presence、side、site、object_role、contact_mode、start/end、left/right
  censor；
- 一个 event slot 可引用多个 contact segment ID，从而表示 robot–tool 与 tool–target 两层 contact；
- outcome 两层：
  - object effect：`none/translate/rotate/articulate/flow/deform/transfer`，可复用旧 v1 taxonomy；
  - task result：`success/partial/failure/unknown`，必须记录传感/人工判据和时间；
- `release` 不自动等于 success；`effect observed` 也不自动等于 intended outcome。

## 8. 与 22D contact grammar、continuous pose tube、partial pairing 的关系

### 8.1 22D grammar：组合，但不静默扩维

ETG 的 `during` edge 直接引用现有 22D contact segments；graph 负责谁作用于谁、pre/post relation 和 outcome，
22D 负责接触侧/部位/模式/时间/censor。这样 Idea H 是 Idea A 的上层 transition semantics，而不是重复 contact
planner。

第一版应新增独立 `event_transition_v0` schema，不修改 `contact_segment_v1_1` 的 22D 宽度。若后续把 event packet
送入 action，必须独立计算其精确有限容量，并增加 equal-bit control；不能继续沿用 22D 的 186.85-bit 上界。

### 8.2 Continuous pose/contact tube：保持互斥的 action route

- ETG 主实验中 tube 可作为 **training-only reconstruction auxiliary** 或 executor supervision；action 不直接读 tube；
- 架构 2 中 tube 是 action-visible 主表示，此时必须明确放弃“finite event route”claim；
- 不允许 ETG+continuous tube 同时作为主模型，再用 ETG ablation 的小幅差值声称 graph 是核心。

### 8.3 Partial pairing：可选数据协议，不是 ETG 必要组件

ETG 可独立用 human video pretrain event predictor、robot data train action decoder；partial pairing 只定义哪些 human/
robot events 可做 cross-embodiment alignment：

- exact pair：相同 task、对象实例、目标 outcome，最好同一场景；
- weak pair：相同 tool/target role + relation delta + outcome class，不要求逐帧或同对象；
- unpaired：只参与 domain-specific event/contact/action loss；
- alignment 使用 event slot/phase，而非 DTW 整条 wrist trajectory。

必须报告 exact/weak/unpaired 三种数据预算。若 ETG 只有 exact pairs 时有效，它不是 scalable human-video transfer；若
unpaired human data 已经足够，partial pairing 只能算 sampler/benchmark 贡献。

## 9. Loss 设计

总目标建议先保持分解，避免一个总 loss 掩盖无效字段：

```text
L = L_action + λv L_visual
  + λb L_boundary_survival
  + λg (L_role + L_relation_pre + L_relation_post + L_delta)
  + λc L_contact_grouped
  + λo L_effect_outcome
  + λp L_prior_posterior
  + λx L_cross_embodiment
```

具体要求：

1. `L_boundary_survival`：离散 hazard/interval-censored NLL；unknown 不回传，left/right censor 进入 likelihood，不能仅
   独立 BCE 一个 censor flag。
2. `L_role/relation`：masked CE/BCE；relation class imbalance 用 class-balanced focal 或 reweighting；entity/slot assignment
   先固定 role slots，必要时才用 Hungarian。
3. `L_delta`：要求 predicted `post` 与 `apply(pre,pred_delta)` 一致，并使用 hard negatives：同 pre/不同 action-effect、
   同 task/失败 outcome、同对象/错误 role；背景/摄像机变化不能作为捷径。
4. `L_contact_grouped`：沿用当前 grouped 22D loss；不能把 unknown/missing 设为零。
5. `L_effect_outcome`：masked CE + Brier/ECE calibration；sensor truth、human review、pseudo label 分层报告。
6. `L_prior_posterior`：posterior 看完整 event windows，prior 只看 current reference/state/text；stop-gradient/EMA target 防止
   两支共同 collapse。
7. `L_cross_embodiment`：只在 exact/weak pair mask 上做 role/phase-aware contrastive 或 partial OT；负样本按 object instance/
   session 分组，防止同 episode 泄漏。
8. `L_action/L_visual`：保持当前 flow matching；ETG auxiliary-only 和 strict route 必须分别跑，避免把 auxiliary 表征收益
   误归因于 action-visible graph。

## 10. 因果控制与防 shortcut 审计

### 10.1 最小 matched controls

| 对比 | 固定什么 | 唯一改变 | 能回答什么 |
|---|---|---|---|
| direct WAM vs event-aux direct | action 仍读 reference、相同 planner/参数/数据 | event losses 0 vs >0 | 纯 auxiliary supervision |
| event-aux direct vs strict ETG | 相同 event planner/loss/Q/训练 | action 读 raw ref vs hard ETG | strict semantic route |
| strict 22D vs strict ETG | 相同 contact packet/action expert | 是否含 pre/post graph+outcome | Idea H 是否超出 Idea A |
| equal-bit event code vs ETG | 相同 action-visible cardinality/Q/监督/参数 | 任意 code vs human semantics | 显式结构价值 |
| ETG vs delta+tube | 相同 event windows/data/backbone | finite graph vs continuous tube | 可审计抽象与执行精度权衡 |
| ETG vs latent option | 相同 posterior horizon/bit budget | supervised semantics vs latent code | 标签语义是否必要 |

### 10.2 必做 intervention

- pre/post swap：只改顺序，phase/outcome 应变化；
- role swap：tool↔target、source↔receptacle，动作或 predicted result 应显著变化；
- event delay：contact onset/release 平移，只有对应阶段 action 受影响；
- outcome shuffle within task：防模型只读 task name；
- same pre + different feasible action：在 simulator/真实受控 setup 验证不同 effect，不用 LLM counterfactual 代替；
- same intended outcome + distractor/camera motion：ETG 应保持，full-pixel latent 不应靠背景；
- no-event/failed-contact windows：检验 acquisition 与 mere proximity；
- target bytes mutation：非 oracle forward 时改变 label 文件不能改变当前 action；
- cache/uncached、gradient、reference clamp：延续现有 strict route 审计，任何 raw event hidden、continuous rim mask 或
  posterior future feature 都不得进入 action/cache。

如果没有 physical intervention，只能写“matched association/control”，不能写 causal state transition learning。

## 11. Human–Robot Paired Event Benchmark

### 11.1 任务族

第一版应选 6 个共享 transition family，而不是追求 task 数：

1. pick/place：`separate→held→inside/on→released`；
2. open/close：`closed↔open`，含 handle/tool role；
3. pour/transfer：`source-held→aligned→flow/transfer→released`；
4. cut/scrape/spread：tool–target 双接触与 irreversible/deform/transfer effect；
5. wipe：distributed contact 与 dirty/wet region outcome；
6. press/hammer/insert：impact/fixture relation 与 task completion。

这些任务同时覆盖 BridgeACT、CoA-VLA、VLAff/VRB、普通 Diffusion Policy/ACT 和当前 contact-WAM baseline。

### 11.2 Pair 单位与 split

`pair_id` 连接一个 human event sequence 与一个或多个 robot rollouts，匹配条件至少包括 task family、tool role、target
role、intended relation delta 和 outcome class。pair 不是逐帧同步，也不要求 wrist pose 同构。

必须保存并 group split：

- human operator/subject/session/day/camera；
- robot operator/collector/session/day/calibration/embodiment；
- tool/target/receptacle physical instance；
- source human clip、robot episode 和 pair family。

任何同 source episode 派生的窗口、同 object instance 或 exact pair 不得跨 train/test。主测试至少含：

- novel object instance；
- novel human operator/view；
- novel robot session/calibration；
- human→robot embodiment shift；
- weakly paired 与 completely unpaired stress split。

### 11.3 指标

**表示/事件：** boundary interval AP/tIoU、phase macro-F1、role F1、pre/post relation macro-F1、delta exact-match、
contact field F1/MAE、outcome macro-F1/Brier/ECE、censor calibration。

**跨 embodiment：** same-event retrieval R@1/R@5、phase alignment error、pair-vs-nonpair margin、event confusion by
task/object；不要只报 t-SNE。

**动作：** task success、correct-first-contact、tool-target role error、event-order violation、release timing、post-effect
verification、collision/spill/drop、action latency。

**压缩 insight 专属指标：**

- `preparation-motion ratio`：到首次 task-relevant contact 前的多余路径/时间；
- `post-success overrun`：outcome 已满足后继续动作的时间/距离；
- `event completion latency`：predicted transition 与真实 outcome 的闭环延迟；
- matched event success：相同 pair group 下人类 event packet 是否比 full human trajectory 更能提升 robot success。

真实成功率按 episode/pair group 做 paired hierarchical bootstrap 或 cluster-aware model，报告 95% CI；三个训练 seed
不是三个真实 trial。

## 12. Baselines 与消融

### 12.1 必要 baselines

1. 当前 GWP-0.5 direct action-centered WAM；
2. ACT/Diffusion Policy/full-trajectory BC，robot data budget 匹配；
3. VRB/HRP 或可复现的 single-frame contact+post-contact affordance；
4. VLAff、BridgeACT、ST-VLA、CoA-VLA：可执行时同数据重训；否则诚实标 conceptual/official-number comparison；
5. SuSIE/GHOST-style visual/3D subgoal；
6. TCN/TCC phase alignment + robot low-level policy；
7. 现有 22D contact-only hard-ST；
8. event auxiliary-only direct；
9. equal-bit unstructured event code；
10. architecture 2 delta+tube 与 architecture 3 latent option。

### 12.2 字段/窗口消融

- single frame、uniform 4-frame、event windows；
- pre only、during only、post only、pre+post、完整四阶段；
- 去 role、2D relation、rim、contact、censor、outcome；
- GT boundary、predicted boundary、固定时间切片；
- 关系只作 auxiliary vs action-visible；
- continuous 2D auxiliary vs quantized action-visible relation；
- sensor-confirmed outcome vs visible-only outcome vs pseudo label；
- exact pair、weak pair、unpaired、no-human pretraining；
- same human video hours、same robot trajectories、same total optimizer steps。

## 13. 分阶段成功/停止门槛

### Gate 0：标签可重复（CPU/人工，先做）

- 至少 12 episodes、每个 transition family ≥20 events 的双标+裁决 pilot；
- role/relation/outcome Cohen/Fleiss κ ≥0.75；contact categorical κ ≥0.70；
- boundary median disagreement ≤5 frames或≤250 ms，interval overlap ≥0.70；
- rim `localizable` 判定 κ ≥0.70，localizable 样本 PCK@0.1 ≥0.80；
- 任何字段 coverage/censor/unknown 无法可靠区分则从 v0 action packet 删除。

**停止：** 两轮指南修订后仍不达门槛；或 >40% 样本的关键 pre/post outcome 在 RGB+现有传感中均不可判。

### Gate 1：oracle sufficiency（小模型/目标 robot data，训练主线前）

- oracle ETG + low-level executor 相对 full-trajectory/direct RGB success 下降不超过 3pp；
- oracle ETG 相对 oracle 22D-only 在 outcome-sensitive tasks 至少 +8pp success 或 +10pp correct event completion；
- oracle ETG 能在 ≥90% 成功 trial 上正确判断 stop/release，且不增加 collision/spill/drop。

**停止 strict ETG：** oracle packet 仍比 direct/full trajectory 低 >5pp；或必须让 action 直读 continuous tube/raw image 才能
恢复。可降级为 auxiliary/progress verifier。

### Gate 2：event 可预测

- held-out phase/role/relation/outcome macro-F1 均 ≥0.70；
- boundary 75% 在 250 ms 或 5 frames 内；
- outcome ECE ≤0.10，unknown/censor 单独报告；
- predicted ETG 达 oracle ETG action success 的 ≥80%。

**停止：** 预测误差主要来自不可观测字段且传感/标签无法补齐；或模型只靠 task text，在 outcome-shuffle/distractor control
上崩溃。

### Gate 3：主因果格

- `strict ETG` 在 3 seeds 上同时优于 `event-aux direct`、`strict 22D` 和 `equal-bit event`；
- group-heldout robot success 绝对提升目标 ≥8pp，95% paired CI 下界 >0；
- correct-first-contact 或 correct-stop 至少一项 ≥10pp，另一项不退化；
- human pretraining 在固定 robot data budget 下相对 robot-only ≥5pp，且 weak/unpaired stress 不反转；
- route swap/delay/clamp 只影响对应 event phase，无全局 shortcut。

**停止主 claim：** ETG≈equal-bit/latent option；ETG 只提升 label metric 不提升 action；或只有 random-window split 有收益。

### Gate 4：系统/投稿门槛

- 至少两个 robot embodiments 或一个极不对称双手平台 + 一个公共 benchmark；
- 每个主 task 足够独立真实 trials，CI 不由少数 object/session 主导；
- action-only latency 增幅 ≤15%，无新 safety regression；
- 发布 schema、annotation guide、group split、sidecar hash 和 matched-control receipts。

当前数据不满足 Gate 0，更不满足 Gate 1–4。

## 14. 当前数据可行性：能做什么、不能做什么

### 14.1 可用线索

- [Ego4D FHO](https://ego4d-data.org/docs/benchmarks/hands-and-objects/) 已有 PRE/CONTACT/PNR/POST、hand/tool/
  state-change object boxes，可做公共 event-boundary/object-effect pretraining。[R]
- ChangeIt/HowToChange 可训练 initial/transition/end 和 open-world state-change encoder；VSCOS 可训练 state-change 后仍
  保持 object identity。[R]
- Action Genome/EASG 可预训练关系图，但其 household action relations 不是物理 contact truth。[R]
- HOT3D 有多目 ego、手/对象/相机 3D pose 和 mesh，可导出 geometry baseline；距离阈值不能冒充传感 contact。[R]
- 当前仓库已有 22D sidecar contract、strict loader、planner/transport/action route 和 causal-control plumbing。[I]

### 14.2 当前项目硬缺口

- 现有 66 个 ego episodes、20,013 frames 的 `hand_status/hand_active` 不是 contact、event relation 或 outcome truth；
- formal reviewed contact packet 仍没有足够真实 adjudicated labels；
- 66 episodes 缺 subject/session/operator、object instance、liquid identity/fill/outcome/batch 等可信 group provenance；
- HOT3D 主任务没有本题的 task outcome 与 continuous physical contact；
- 当前没有可验证的 human–robot exact/weak pair table；
- 目标 asymmetric robot 的 event/outcome 数据和 sensor truth 缺失；旧/异构 robot data 只能作预训练或负迁移研究；
- 透明液体 stream/contact 当前不可从普通 RGB 稳定标注。

因此当前 **可做**：annotation schema、12-episode pilot、公共 Ego4D/ChangeIt perception baseline、CPU route audit、
oracle packet 小模型设计。当前 **不可做**：5B main run、production OOD、液体物理 state claim、human-to-robot causal
transfer、顶会 ready claim。

## 15. WAM 代码插入位置（设计，不是已实现）

### 15.1 Schema 与 sidecar

1. 新建 `world_action_model/event_transition_schema.py`：[S]
   - `EVENT_SCHEMA_VERSION="event_transition_v0"`；
   - event/role/relation/effect/outcome/censor enums；
   - hard canonicalization、finite-capacity receipt、target/mask validator；
   - 不修改 `contact_segment_v1_1`，只引用 contact event IDs。
2. 新建 `scripts/build_event_transition_sidecars.py`：[S]
   - 从 adjudicated physical events 机械派生 per-frame `[T,K,D]` targets；
   - 输出 event、relation、rim、outcome arrays、coverage 和 SHA-bound metadata；
   - model pseudo labels 单独 tier，不能混入 reviewed GT。
3. `world_action_model/datasets/wam_lerobot_dataset.py:741-819` 当前负责 verified 22D sidecar 读取；按同一 fail-closed
   模式增加 `event_target/event_loss_mask/event_confidence`、relation/rim/outcome masks 和 meta/episode SHA。[S]
4. `world_action_model/datasets/hot3d_wam_dataset.py` 只作为 HOT3D geometry/event pretraining adapter；不得把近邻
   几何写成 sensor-confirmed contact。[S]

### 15.2 Model

1. `world_action_model/models/transformer_wa_casual_mot.py:703-863`：现有 contact query/planner/decoder 初始化附近
   增加独立 event queries、factorized heads 和 packet encoder；不要把 variable graph hidden 直接送 action。[S]
2. `...:1253-1285`：仿照 `_plan_contact_logits` 新增 `_plan_event_transition_logits`；planner 可看 state/current ref/text，
   posterior 训练分支可额外看 future windows，但 posterior hidden 必须在 route 前 stop-gradient/quantize。[S]
3. `...:1405-1450`：`encode_action_tokens` 只接 canonical hard event packet；strict ETG 时拒绝 raw event hidden、rim mask、
   future features 和 oracle fields。[S]
4. `...:1531-1589`：`encode_joint_streams` 生成 event packet 并记录 aux logits；需要显式 `event_action_uses_packet` 和
   `event_action_attends_reference` route contract。[S]
5. `...:127-190` 与 `...:1809-1822`：扩 attention mask；strict route 中 action 不读 current ref/future/posterior/raw graph，
   event planner 读 current ref，visual expert 仍可训练。[S]
6. `...:1836-1914`：返回 factorized event predictions；不能用一个 dense vector loss掩盖字段 mask。[S]
7. `...:2240-2710` action-only cache：只缓存 hard packet及其 schema/hash，不缓存 raw logits/planner/posterior/reference KV；
   增加 cached/uncached bit-equality audit。[S]

### 15.3 Trainer / config / eval

1. `world_action_model/trainer/wa_casual_trainer_mot.py:957-1000`：读取 targets/masks，只在 oracle variant 显式传
   model condition；普通 predicted route 的 GT 只能进入 forward 后 loss。[S]
2. `...:1007-1055`：加入 boundary survival、role/relation/delta/outcome、prior-posterior 和 pairing loss；每项单独 receipt。
3. `configs/giga_world_policy_0_5_sonic_g1_contact_causal_controls.py:22-74`：新增独立 event control config，至少包含
   direct、event-aux、strict-22D、strict-ETG、equal-bit-event；现配置明确是 spec-only、禁止 5B/server training。[S]
4. `scripts/evaluate_contact_asym25_heldout.py` / `world_action_model/contact_asym25_heldout_eval.py`：另建 event evaluator，
   报 group/pair-level event/action metrics，不把 frame windows 当独立样本。[S]

## 16. 最小推进顺序

1. 冻结架构争论，先写 `event_transition_v0` annotation guide 和 12-episode 双标 pilot；
2. 用 Ego4D FHO + 现有 ego clips 做 event boundary/role/relation/outcome 可标性审计；
3. 建 exact/weak/unpaired pair manifest 与 group provenance；没有 pair table 不讨论 cross-embodiment benefit；
4. 用小 action decoder 做 oracle ETG vs oracle 22D vs direct/full trajectory sufficiency；
5. Gate 1 通过后才实现 ETG planner/route 和 equal-bit event control；
6. 先跑 action-free/perception 与 CPU route invariants，再考虑小模型训练；
7. 只有 Gate 2/3 通过且目标 robot/sensor truth 到位，才申请 5B/服务器/真实硬件实验。

## 17. 最终 Go / No-Go 表

| 项目 | 决定 | 理由 |
|---|---|---|
| 把 event transition 作为 annotation/perception pilot | **Go** | Ego4D/ChangeIt 前例和现有视频足以验证可标性，成本低。 |
| ETG-WAM 作为下一主架构候选 | **Conditional Go** | 用户 insight 强、与现有 strict route 相容，但 novelty/数据风险高，必须先过 oracle sufficiency。 |
| Delta token + continuous tube | **Conditional Go as baseline/backup** | 执行可能更强，但与 VLAff/BridgeACT/ST-VLA 高度重叠，不能和 ETG 混作单一贡献。 |
| Latent option boundary 作为主方法 | **No-Go** | UniVLA/Being-H0.7/DreamDojo 邻域拥挤，且失去本题可观察状态差的核心价值。 |
| 用现有 66 ego episodes 直接声称人机 transfer/OOD | **No-Go** | 无 event/outcome truth、pair table 和 group provenance。 |
| 透明液体 3D reconstruction | **No-Go** | 不需要且当前无真值；只允许 coarse sensor-confirmed outcome。 |
| 当前启动 5B/GPU/main training | **No-Go** | Gate 0/1 未过，先做标签与 oracle 证伪。 |
| 当前包装为顶会 ready | **No-Go** | SWIM/Ego4D/BridgeACT/ST-VLA 等强冲突，paired causal benchmark 与真实收益均缺失。 |

## 18. 一句话推荐

**推进“有限 Event-Transition Graph packet + 22D contact edge”的 oracle/标注 pilot；把 pre/post delta+tube 作为强连续
baseline，把 latent option 作为同容量对照。若 oracle graph 不能在不读 raw image/tube 的情况下完成工具任务，就立即将
Idea H 降级为 training-only state-change/outcome auxiliary，而不要再扩主 WAM。**

## 19. 原始来源索引

### State change / event / graph

- [Ego4D](https://arxiv.org/abs/2110.07058)
- [Ego4D Hands & Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)
- [Look for the Change / ChangeIt](https://openaccess.thecvf.com/content/CVPR2022/papers/Soucek_Look_for_the_Change_Learning_Object_States_and_State-Modifying_Actions_CVPR_2022_paper.pdf)
- [Learning Object State Changes in Videos: An Open-World Perspective / HowToChange](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Learning_Object_State_Changes_in_Videos_An_Open-World_Perspective_CVPR_2024_paper.pdf)
- [Video State-Changing Object Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_Video_State-Changing_Object_Segmentation_ICCV_2023_paper.html)
- [Action Genome](https://openaccess.thecvf.com/content_CVPR_2020/html/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.html)
- [Egocentric Action Scene Graphs](https://openaccess.thecvf.com/content/CVPR2024/html/Rodin_Action_Scene_Graphs_for_Long-Form_Understanding_of_Egocentric_Videos_CVPR_2024_paper.html)
- [Recognizing Manipulation Actions from State-Transformations](https://arxiv.org/abs/1906.05147)
- [What Changed and What Could Have Changed?](https://arxiv.org/abs/2503.21055)
- [HanDyVQA](https://openaccess.thecvf.com/content/CVPR2026/html/Tateno_HanDyVQA_A_Video_QA_Benchmark_for_Fine-Grained_Hand-Object_Interaction_Dynamics_CVPR_2026_paper.html)

### Affordance / tool use / subgoal

- [Affordances from Human Videos as a Versatile Representation for Robotics / VRB](https://openaccess.thecvf.com/content/CVPR2023/html/Bahl_Affordances_From_Human_Videos_as_a_Versatile_Representation_for_Robotics_CVPR_2023_paper.html)
- [Structured World Models from Human Videos / SWIM](https://arxiv.org/abs/2308.10901)
- [HRP: Human affordances for Robotic Pre-training](https://www.roboticsproceedings.org/rss20/p068.html)
- [VLAff](https://arxiv.org/html/2608.05215)
- [BridgeACT](https://arxiv.org/html/2604.23249)
- [ST-VLA](https://arxiv.org/html/2603.13788)
- [CoA-VLA](https://arxiv.org/html/2412.20451)
- [GHOST](https://arxiv.org/abs/2606.10025)
- [SuSIE](https://arxiv.org/abs/2310.10639)
- [UniPi](https://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html)
- [ToolTango](https://arxiv.org/abs/2206.09141)
- [Creative Robot Tool Use with Large Language Models / RoboTool](https://arxiv.org/abs/2310.13065)
- [Time-Contrastive Networks](https://arxiv.org/abs/1704.06888)
- [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846)

### World model / VLA / latent action

- [GigaWorld-Policy-0.5](https://arxiv.org/abs/2607.13960)
- [Being-H0.7](https://arxiv.org/abs/2605.00078)
- [UniVLA](https://arxiv.org/abs/2505.06111)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [Action-Effect Memory](https://arxiv.org/abs/2606.12499)
- [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
- [DreamDojo](https://arxiv.org/abs/2602.06949)
- [OmniContact](https://arxiv.org/abs/2606.26201)
