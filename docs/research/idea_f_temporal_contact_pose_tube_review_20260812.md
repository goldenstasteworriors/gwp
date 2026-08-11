# Idea F 独立调研：Temporal Contact-Pose Tube / 多阶段手部姿态—接触过程

- 日期：2026-08-12
- 审查基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 隔离分支：`codex/idea-f-temporal-contact-pose-tube-20260812`
- `idea_id`：`a78673af-f56e-4ad1-9974-beee1edb765b`
- `discussion_id`：`8562848c-0b71-485e-abff-064f0900159c`
- 范围：只做文献调研、标签与架构设计；未训练、未使用 GPU、未修改 canonical 代码或数据

## 1. 一句话结论

**作为独立的“首次时序 hand-contact trajectory”主张是 No-Go；作为现有 finite-contact WAM 的多阶段、显式持续时间、可删失扩展是 Conditional Go。**

完整手—物运动和接触序列已有 TOCH、ARCTIC、Text2HOI、LatentAct / *How Do I Do That?*、CAMS、VLAff、Being-H0、EgoVLA、ImMimic、VideoManip 等大量直接前例；“关键姿态而不是逐帧动作”也已有 PPI、Keypose Exploration、Ego4D hand trajectory + time-to-contact 等前例。因此不可声称首次预测手姿态序列、首次联合手姿态与接触、首次用 keypose、首次从 ego/human video 学 trajectory，或首次把这些信号用于机器人。

仍可能守住的命题是：

> 对 action-centered WAM，不把单帧 grasp 当作整项工具操作，也不逐帧重建/模仿 human hand；预测一个带显式 phase、duration、contact transition、visibility 和 temporal censor 的稀疏 contact-pose event tube，并把送入 action 的视觉信息严格编译为可枚举有限 packet。比较单帧 grasp、逐帧 hand trajectory、普通 keypose、无语义同 bit code 与该中间粒度，检验它是否以更少监督保留接触顺序、重抓和释放信息，并改善 OOD 接触执行。

推荐模型不是自由连续 spline，而是**semi-Markov finite event tube**：episode 尺度的 phase/duration 状态机负责全局过程，当前动作窗只读取 active phase 与接下来少量 hard event。连续 2D/3D/MANO 只作训练期 target/probe；若必须把连续姿态、spline coefficient 或 decoder hidden 直接喂给动作才有收益，则 finite/auditable 主张停止。

当前 66 个 ego episodes 只足以做标签可行性、事件压缩、episode-held-out perception probe 和 plumbing；不能证明机器人动作收益、跨对象/操作者 OOD、metric 3D、物理接触或 target hardware 成功。

## 2. 调研方法与证据边界

### 2.1 Zotero-first 结果

本轮先按 `idea-discussion` skill 同步并检索用户 Zotero，再核对 arXiv、CVF、PMLR、官方数据集/项目页原文。idea card 已创建并更新，ID 见文首。

自动讨论器第一次返回的库内条目主要是 RoboBrain、FAST、RT-1，外部候选甚至混入图像篡改检测，query expansion 明显跑偏；这些结果没有被当作 novelty 证据。随后直接只读检索本地 Zotero 标题、摘要和 URL，找到真正相关的个人库条目：

| Zotero key | 条目 | 对本 idea 的意义 |
|---|---|---|
| `8WMVVJAL` | ImMimic | retargeted human hand **full trajectories** 经 DTW 映射到 robot joints，再与 robot trajectory 插值；属于全序列 imitation 端点 |
| `FVS59CQW` | EgoVLA | 从 ego human video 预测 wrist/hand actions，再 IK/retarget；属于连续 human action 预训练 |
| `YBJD8QP4` | Being-H0 | part-level motion tokenization 建模精细 hand trajectories；不是稀疏 event grammar |
| `ZJQINYLA` | VideoManip | 从 RGB human video 重建显式 3D hand/object/robot trajectories，再学 dexterous policy |
| `6EYVGBVZ` | Human Universal Grasping | 单张 RGB-D 生成 wrist + MANO **单 grasp**；属于静态端点 |
| `R2LFERRG` | Uni-Hand | ego hand waypoint / motion forecasting，并强调 contact/separation boundary |
| `JRGPD6HS` | OmniContact | key body trajectories + time-series binary contact 的 contact flow；说明“关键轨迹 + 接触时序”本身也非空白 |
| `LH5DGVVD` | WHOLE | 从 ego video 联合恢复 hand/object world-space motion，并处理 out-of-sight；属于全序列 reconstruction |
| `KCNBA8FP` | Do as I Do | 从 everyday monocular video 重建并 retarget 多指 robot action sequence |
| `7BRCUNE4` / `WM8H622U` | EgoEngine / EgoInfinity | 生成高保真 robot video/action trajectory 或 4D HOI 数据引擎；是完整 video-to-action 管线前例 |

### 2.2 外部检索 fallback

`arxiv-search` skill 的 CLI 存在未打印返回值的问题，直接调用其函数后又发现当前环境没有安装 `arxiv` 包。本任务明确不改环境，因此没有安装依赖，改由 arXiv HTML、CVF Open Access、PMLR 和数据集官方页逐篇核对。下文只使用相关原始来源；不采用自动讨论器的弱命中。

### 2.3 证据标记

- **[R]**：论文原文、作者项目页、官方 benchmark 文档支持。
- **[I]**：当前基线代码已经存在，可定位到具体实现。
- **[S]**：建议的 schema/架构/benchmark，尚未实现。
- **[M]**：缺真实标注、训练、统计、机器人或硬件证据。

## 3. 必须区分的三种粒度

| 粒度 | 表示 | 优点 | 丢失/风险 | 代表前例 |
|---|---|---|---|---|
| **单帧 grasp / 单目标 token** | 一个 grasp pose、contact point/heatmap、末端 target keypose | 标注/预测便宜，空间目标明确 | 不知道接近、首次接触、重抓、施力中姿态和释放顺序 | ContactPose、HUG、OakInk steady pose、PPI 的单个/少量 target keypose、VLAff 的 96D grasp branch |
| **中间粒度 event tube（本提案）** | 4–8 个有序事件；每个事件有时间/持续时间、phase、粗姿态、接触、可见性/删失 | 保留执行结构而避免逐帧 human imitation；可审计、可干预 | 事件定义和压缩充分性必须实证；连续几何容易形成旁路 | Ego4D TTC/hand waypoint、OCT trajectory+hotspot、Keypose Exploration、PPI 最接近，但都不等于这里的 finite/censored WAM route |
| **全序列 reconstruction / imitation** | 每帧 MANO/hand joints/contact/object pose 或 retargeted robot action | 信息最全，可生成平滑动作 | 估计噪声、遮挡、embodiment gap、标注/计算重；容易学习逐帧模仿而非功能过程 | TOCH、ARCTIC、HOI4D、HOT3D、LatentAct、Text2HOI、CAMS、WHOLE、Being-H0、EgoVLA、ImMimic、VideoManip |

Idea F 的科学问题不是“时序有没有用”，而是：**事件对齐的稀疏中间粒度，是否在相同信息预算/标注预算下，比单帧足够、比全序列更可迁移，并且真的被 action 因果消费。**

## 4. 最接近前例与 novelty 冲突

### 4.1 直接的 hand pose + contact sequence 前例

| 工作 | 原文已经做了什么 | 对 Idea F 的约束 |
|---|---|---|
| [TOCH, ECCV 2022](https://arxiv.org/abs/2205.07982) | 学习 spatio-temporal object-to-hand correspondence field，修正完整 3D HOI sequence，并显式覆盖 contact 前后 | 不能声称首次 temporal contact/hand relation 或首次建模 contact 前后平滑性；差异只能是稀疏 event、控制接口和 causal route |
| [ARCTIC, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fan_ARCTIC_A_Dataset_for_Dexterous_Bimanual_Hand-Object_Manipulation_CVPR_2023_paper.html) | 2.1M frames、双手与 articulated object 准确 3D meshes、动态 contact；研究时空一致 reconstruction/interaction field | 是动态手姿态—接触 GT 与 public benchmark 的首选，不可把数据/任务本身包装成新意 |
| [Text2HOI, CVPR 2024](https://arxiv.org/abs/2404.00562) | 将 text-to-3D HOI sequence 分成 contact generation 与 motion generation，并用 contact prior 约束完整 hand-object motion | “先接触、后动作”或联合生成 contact/motion 都已有 |
| [CAMS, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CAMS_CAnonicalized_Manipulation_Spaces_for_Category-Level_Functional_Hand-Object_Manipulation_Synthesis_CVPR_2023_paper.pdf) | 预测 object/contact-centric hand-pose embedding sequence，再生成兼容 object motion 的 hand sequence | object-centric canonical pose sequence 不是新意 |
| [How Do I Do That? / LatentAct, CVPR 2025](https://arxiv.org/abs/2504.12284) | 从单 RGB、文本和 3D contact point 预测 future **3D hand pose + contact-map sequence**；VQ-VAE tokenizes interaction trajectories；HoloAssist 数据引擎抽取 3D hand/contact | 最直接冲突。不能声称首次 hand-contact trajectory、首次把 interaction sequence tokenization，或“无需 object mesh”本身；Idea F 必须证明 event sparsification、duration/censor 和 action route 的不同 |
| [GEARS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_GEARS_Local_Geometry-aware_Hand-object_Interaction_Synthesis_CVPR_2024_paper.pdf) | 由 hand/object trajectories 逐帧生成与 interaction 一致的 hand pose sequence | 完整 hand pose synthesis baseline |
| [WHOLE, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Ye_WHOLE_World-Grounded_Hand-Object_Lifted_from_Egocentric_Videos_CVPRF_2026_paper.html) | 从 ego video 联合生成/恢复 hand 和 object world-space trajectories，并处理遮挡、出视野 | 不能声称时序 prior 解决遮挡/out-of-sight 是 Idea F 的首创 |

### 4.2 稀疏 hand trajectory、keypose 与 phase 前例

| 工作 | 原文已经做了什么 | 与 Idea F 的精确区别 |
|---|---|---|
| [Joint Hand Motion and Interaction Hotspots, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Joint_Hand_Motion_and_Interaction_Hotspots_Prediction_From_Egocentric_Videos_CVPR_2022_paper.html) | 从 ego video 联合预测 future 2D hand motion trajectory 与下一 active object contact hotspot，明确定位为 action label 与 pixel prediction 之间的低维表示 | 已覆盖“中间粒度比 label/pixel 更合适”的核心直觉；但它不预测多阶段 articulated pose/contact transitions，也不进入有限 WAM action route |
| [Ego4D Forecasting](https://ego4d-data.org/docs/benchmarks/forecasting/) | 分别提供 future hand positions、short-term object interaction 的 verb + time-to-contact、long-term action sequence | TTC、future hand 和多步 anticipation 都不是新意；Idea F 的差异是把它们绑定成同一 event grammar |
| [PPI, Gripper Keypose and Object Pointflow, 2025](https://arxiv.org/abs/2504.17784) | 用后续 keyframe 的 target gripper poses 与 object pointflow 条件化 continuous action；keyframe 由 gripper openness/arm-state turning points定义 | 最接近“keypose 与 continuous control 的中间接口”；但它是 robot demonstration、无 human hand phase/contact/censor，也不限制 action 的视觉通道 |
| [Keypose Exploration, 2026](https://arxiv.org/abs/2606.29028) | 用 VLM 识别 grasp/release/object-object interaction 语义事件，再以 trajectory analysis 精确对齐；keypose-guided DP 并做 reachability filtering | 已覆盖事件语义自动标 keypose与 cross-embodiment conditioning；Idea F 不能把“VLM 找 grasp/release keypose”当贡献 |
| [Keyframe-Focused Visual Imitation, ICML 2021](https://proceedings.mlr.press/v139/wen21d.html) | 视觉 imitation 强调 demonstration 的关键帧 | keyframe-focused learning 是通用前例 |
| [Semi-Markov action segmentation, CVPR 2008](https://doi.org/10.1109/CVPR.2008.4587557) / [HSMM review](https://doi.org/10.1016/j.artint.2009.11.011) | segment 级特征、相邻转移和显式 state duration；HSMM 本身是成熟工具 | semi-Markov/显式 duration 只能是建模选择，不能单独作为 novelty |
| [FACT, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html) | 用 frame-action cross-attention 做连续视频的非重叠 action segmentation，并在 ego/third-person 数据评估 | 现代 discriminative segmenter 是 HSMM 的必要 baseline；显式 phase head 本身不是贡献 |

工具使用的 phase vocabulary 也不是空白：HoloAssist 已有 temporally localized atomic actions（grab/open/screw/mix/rotate/align/slide 等），Ego-Exo4D 有 fine-grained keystep segment，ARCTIC区分 use/grasp，human handover 数据常用 reach/transfer/retreat 三阶段。它们能提供 proposal 或 phase-recognition baseline，但 atomic action/keystep 仍不是物理 contact transition truth。

### 4.3 Ego/human-to-robot：单 grasp、trajectory 与完整 imitation

| 工作 | 粒度判断 | 关键事实 |
|---|---|---|
| [Human Universal Grasping](https://arxiv.org/abs/2606.17054) | **单 grasp** | 单 RGB-D 生成 wrist translation/rotation + MANO pose；适合 static baseline |
| [ContactPose](https://contactpose.cc.gatech.edu/) | **静态/稳定 grasp contact** | 重点是 object contact 与 hand pose 的 grasp 数据，不是多阶段工具过程 |
| [OakInk, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.html) | **每段 1–2 个 steady interaction pose** | 原文从 sequence 选 steady pose 作 interaction transfer；是单/少量姿态 baseline |
| [VLAff, IROS 2026](https://arxiv.org/html/2608.05215) | **单 96D MANO grasp + variable trajectory** | 从 contact keyframe 抽取 grasp，另外用量化 6D trajectory tokens 表示 pre/post-contact motion；不是 articulated hand pose event sequence，但已经覆盖 ego video→contact keyframe/grasp/trajectory→robot |
| [BridgeACT, 2026](https://arxiv.org/html/2604.23249) | **grasp region + dense 3D point flow** | “where to grasp / how to move”；1.5 s clip 下采样为 4 帧，预测 role-conditioned tool-target 3D point flow，用 grasp module + closed-loop SE(3) controller 执行；无多阶段 finger pose tube |
| [Being-H0](https://arxiv.org/abs/2507.15597) | **连续 hand trajectory tokenization** | part-level motion tokenization，目标是精细 hand trajectory/action learning |
| [EgoVLA](https://arxiv.org/abs/2507.12440) | **连续 wrist/hand actions** | VLA 从 human video 输出 wrist/hand actions，IK/retarget 到 robot，再少量 robot data finetune |
| [ImMimic, CoRL 2025](https://proceedings.mlr.press/v305/liu25b.html) | **全 trajectory imitation** | DTW 映射 retargeted human hand pose trajectory 与 robot trajectory，再 MixUp/co-train |
| [VideoManip, 2026](https://arxiv.org/abs/2602.09013) | **全 3D reconstruction/retarget** | RGB human video→hand/object trajectory reconstruction→contact optimization→robot policy |
| [HOST, 2026](https://arxiv.org/abs/2607.20033) | **task progress manifold** | 从单 human video 估计 robot 当前 task progress、预测 own future observation、再导出 action；说明 progress/phase-conditioned human-video following 也已有 |

### 4.4 数据集到底提供什么

| 数据集 | 时序 hand pose | contact / event | phase / action | 对本 proposal 的用途与限制 |
|---|---|---|---|---|
| [Ego4D](https://ego4d-data.org/docs/benchmarks/forecasting/) | hand future positions，不是逐帧 articulated 3D hand | time-to-contact 与 future interacted object | long-term action sequence | 做 2D anticipation / TTC；不能当 3D hand-contact tube GT |
| [Ego-Exo4D](https://docs.ego-exo4d-data.org/benchmarks/ego_pose/ego_hand_pose/) | EgoHandPose 有 3D hand joints；[官方 changelog](https://docs.ego-exo4d-data.org/changelog/) 报告 458 takes、68K 3D poses / 340K 2D poses | 没有统一的 dense physical contact truth | [fine-grained keystep](https://docs.ego-exo4d-data.org/benchmarks/keystep/keystep_recoginition/) 是 trimmed clip label | 可把 hand pose 与 keystep弱对齐；不能静默推导 contact truth |
| [HOT3D, CVPR 2025](https://arxiv.org/abs/2411.19167) | 833+ min、3.7M multi-view images；3D UmeTrack/MANO hands、objects、camera poses | 可由准确 meshes 研究 proximity/contact，但官方主任务是 tracking/pose/lifting | kitchen/office/living-room scenarios | 适合 pose tracking、遮挡和 compression，不应把 mesh proximity自动称物理 contact |
| [HOI4D, CVPR 2022](https://hoi4d.github.io/) | 2.4M RGB-D frames、4000 sequences；frame-wise 3D hand pose与 object pose | 官方页面没有 ARCTIC 级动态 contact GT | frame-wise hand action + action segmentation | 适合 phase/event discovery与 dense-vs-sparse pose比较；contact 需额外规则/标注 |
| [ARCTIC, CVPR 2023](https://arctic.is.tue.mpg.de/) | 339 sequences、双手/物体 3D meshes | detailed dynamic contact，且 hand/object joint evolution | use/grasp modes | 最适合验证事件 tube 是否保留 first contact、changing pose/contact 与 release |

### 4.5 明确不可写的 “first”

以下表述全部不可守：

1. 首次预测 temporal hand-object contact；
2. 首次联合预测 3D hand pose 与 contact sequence；
3. 首次把 interaction trajectory 离散化/tokenize；
4. 首次用关键姿态指导机器人 continuous action；
5. 首次从 ego/human video 提取 grasp、contact keyframe、hand trajectory 或 robot action；
6. 首次使用 phase/state machine/显式 duration；
7. 首次用 event keypose 解决 cross-embodiment；
8. 首次把动作分为 where-to-contact 与 how-to-move。

## 5. 精确研究问题与可证伪假设

### 5.1 研究问题

给定当前 ego observation、state、language 和过去的 tube progress，是否可以只预测接下来少量关键 hand-pose/contact 事件，而不是：

- 把整段工具操作压成一个 grasp/contact target；或
- 逐帧恢复 human MANO/hand trajectory 并直接 retarget；

同时让该 event tube 在 action chunk 上成为可审计、有限、不可由任意 visual latent 绕过的视觉桥？

### 5.2 假设

- **H1：event alignment 胜过 uniform sparsification。** 同样 K 个姿态，按 precontact/contact/regrasp/load/release 事件采样优于等间隔采样。
- **H2：contact + pose + duration 必须联合。** contact-only 不足以区分 approach/hold/load/release；pose-only 不知道何时建立/改变接触；无 duration 的 framewise phase 分类容易闪烁。
- **H3：中间粒度胜过两端。** 在等标注/bit/compute 下，event tube 比 single grasp 更好地预测 first contact、regrasp、release；比 dense human trajectory 在 object/embodiment OOD 上更稳健。
- **H4：收益必须经 finite packet 到 action。** 固定 hard packet 后替换 reference visual，pre-safety action 不变；改变某个 event kind/time/contact/pose bin，动作只在相应 phase/hand/时间局部改变。
- **H5：MANO 不是必要条件。** 2D functional landmarks + object-relative coarse pose/contact 已能通过 oracle gate；MANO 只能提高 teacher coverage，不能成为论文成立的前提。

## 6. 三个互斥模型候选与排序

这里的“互斥”指**主时序 latent/decoder**只能选一种；它们都可共享视觉 backbone 和同一标签。

### 排名 1：Semi-Markov Finite Event Tube（推荐）

主 latent 是有向 phase graph与显式 duration：

```text
approach -> precontact -> first_contact -> grasp
                                |          |
                                v          v
                             regrasp <-> load_motion -> release_outcome
```

- 每个 state 产生一个进入/退出 event pose 和 contact emission；state duration 用离散 hazard/HSMM likelihood。[S]
- 允许合法 skip、self-loop 和 `grasp↔regrasp`，禁止任意 phase permutation。[S]
- 全局状态在 episode 尺度追踪，动作只消费 active state + next 1–3 events。[S]
- 可在 66ep 的小数据下用规则/weak proposal 初始化，显式 duration 抑制 phase flicker。[S]
- 缺点：对任务族扩展需要 versioned graph；HSMM/phase-conditioned policy 本身不新，贡献必须来自 finite WAM route 与对照证据。

**为何排第一：** 当前数据单任务、66 段且标签稀少；显式 state/duration 最可解释、最容易做删失 likelihood、最匹配已有 22D contact segment 和 causal audit。

### 排名 2：Autoregressive Variable-Length Contact-Pose Tube

- planner 依次输出 event tokens，直到 EOS；token 含 `kind/time/side/object/contact/pose/visibility-censor`。[S]
- 逻辑长度可变，天然表示重复 regrasp、漏失 event 和不同任务结构。[S]
- 使用 monotonic time mask、canonical tie-break 和 EOS 后全零，防 token 顺序/空位偷带信息。[S]
- 缺点：66ep 容易过拟合 event language；exposure bias、重复/漏 event、非法 transition 需要约束；VLAff 已有 autoregressive variable trajectory token，LatentAct 已有 interaction trajectory codebook，novelty 风险更高。

**适用条件：** ARCTIC/HOI4D/HoloAssist 等公共数据预训练后，event vocabulary 已证明跨任务覆盖；否则不应直接上 5B WAM。

### 排名 3：Fixed-K Event Queries + Spline / Neural ODE

- K 个 query 预测 event time、pose/contact anchor；B-spline/Bezier/ODE 在 anchors 间插值，动作或 auxiliary读取连续曲线。[S]
- 优点是平滑、并行、能与 dense sequence 做 reconstruction loss；PPI 的 keypose + pointflow 是强概念 baseline。
- 缺点是固定 K 处理重复/缺失 event 不自然；spline coefficient/ODE hidden 是高带宽连续通道，最容易变成任意 visual covert channel；插值会跨越真实 contact discontinuity；完整曲线又退回 dense imitation。

**仅作上限/对照：** 连续 curve 默认 training-only。若把曲线送入 action，必须公开为 non-finite upper bound，不得与推荐方法共用“有限可审计”主张。

### 决策

优先做排名 1；排名 2 是数据扩张后的替代；排名 3 只作连续 oracle/upper bound。不要把三者堆成一个系统，否则无法知道增益来自 phase grammar、variable length 还是 spline smoothness。

## 7. 推荐结构：双时钟 Semi-Markov Contact-Pose Tube

### 7.1 为什么必须双时钟

当前 66ep 共 20,013 个 processed 20 Hz frames，平均每段约 303.2 帧，即约 15.2 s；当前 action horizon 是 48 帧，即 2.4 s。[I]

因此完整 `approach→release` 几乎不可能落在一个 action chunk。若把 event time 都归一化到 48 帧，模型只看到局部 contact segment，不是用户提出的完整过程；若把 action chunk扩到整段，又破坏实时闭环和当前 WAM contract。

推荐两个时间尺度：

1. **episode-scale phase belief**：`z_global, elapsed_duration, remaining_duration`，跨 replan 持久化；只读历史预测/观测，不读 future GT。[S]
2. **local event packet**：从全局 tube 编译出当前 active event 与未来 2.4 s 内下一至三个 event，随每次 observation 重规划。[S]

动作 horizon 仍为 48；不为表示完整过程而输出 15 s open-loop robot action。

### 7.2 训练期视觉预测目标

现有 future-video loss 保留。新增 planner target 由 future human/robot clip 派生，但 planner 输入只允许：

```text
current reference visual + current state + text + previous phase belief
                         -> future event tube
```

严禁 planner 在训练时读取 future visual latent、GT action、未来 hand pose teacher 或完整 target tube hidden。标签可以来自未来视频，输入不可以。

推荐把监督拆为两层：

- **action-visible hard tube**：量化 event kind/time/pose/contact；部署保留。[S]
- **training-only continuous probes**：event frame 的 2D landmarks、可选 metric 3D/MANO、contact heatmap、dense interpolation reconstruction；只改善视觉 backbone/评估压缩，不拼入 action。[S]

这使“视觉预测目标”从仅 future RGB 扩展为低维 future interaction events，但不依赖透明物体/液体的伪 3D 重建。

### 7.3 finite packet 是否扩展：必须新建 v2，不静默改 v1.1

当前 `contact_segment_v1_1` 每段 22D，仅含 presence、side、functional site、object role、contact mode、start/end、left/right censor；它有 contact interval，但没有多阶段 event kind 或 pose。[I]

建议新建 `contact_pose_event_v2`，而不是在同名 22D tensor 尾部加 float。一个保守 action-visible slot 可取：

| 字段 | 基数 | 说明 |
|---|---:|---|
| `presence/EOS` | 2 | EOS 后全部 canonical +0 |
| `event_kind` | 7 | approach / precontact / first_contact / grasp / regrasp / load_motion / release_outcome |
| `time_state` | 18 | 16 bins + left-censored + right-censored；区间不确定由 loss mask/interval likelihood 表示 |
| `side` | 3 | left / right / object-object |
| `object_slot` | 4 | 由确定性 role/track 规则排序，不能由 query permutation 决定 |
| `surface_cell` | 16 | 选中 object/part 的固定 `4×4` atlas cell |
| `pose_family` | 8 | open / power / pinch / lateral / hook / support / tool-precision / other |
| `approach_cone` | 12 | object-relative coarse direction |
| `wrist_roll` | 8 | object-relative coarse roll |
| `contact_mode` | 5 | none / stick / slide / roll / impact |

每个 present slot 有：

```text
7 * 18 * 3 * 4 * 16 * 8 * 12 * 8 * 5 = 92,897,280 states
```

若 `Qmax=8`，逐 slot 独立上界为：

```text
8 * log2(1 + 92,897,280) = 211.753064 bits
```

canonical monotonic time、EOS 与 phase graph 会令真实有效状态更少；正式实现应生成精确 valid-state receipt。这个上界高于当前 v1.1 Q12 的约 186.85 bits，所以必须增加相同 bit budget 的无语义 code baseline，不能把容量增加当语义收益。[S]

以上基数只是**预注册候选**，必须先用标签分布验证；不要为了好看的 taxonomy 强制低一致性字段。`surface_cell/approach_cone/roll` 若无法达到标注门槛，应 mask 或从 action-visible packet 删除，而不是用连续 teacher 值代替。

### 7.4 action 如何消费

推荐数据流：

```text
reference/state/text/history
        |
        v
semi-Markov tube planner
        |
hard v2 event packet + phase belief
        |
deterministic local compiler (active + next events only)
        |
finite event encoder -> action expert -> 48-step action chunk
```

- action 读取 hard event semantics、proprio、noisy action 和已有合法 text route；不读 raw event logits、planner hidden、continuous keypoints、spline/ODE hidden 或 GT。[S]
- 若需要 scene localization，resolver 只能从 hard `object_slot+surface_cell` 选固定几何；不得返回 learned ROI feature。若 cell center/normal/scale以连续值送 action，则论文只能声称**task-semantic route finite**，不能声称完整视觉 route finite；若要完整 finite claim，几何也必须预量化并计入 bit receipt。[S]
- 每次 replan 更新 phase belief并重新预测 local packet，实际观测到 release/regrasp 后允许闭环修正，不一次锁死全 episode。[S]
- action-only cache 只保存 hard packet、phase state、state/text 和 action KV，不缓存 reference/future visual KV。[S]

### 7.5 如何避免任意 visual covert channel

必须同时满足：

1. hard forward：所有 action-visible 字段均有限、canonical、可反序列化；soft 只作训练对照；
2. raw planner hidden/logits、query identity、attention map、continuous pose、MANO、heatmap、spline coefficient 不进入 action；
3. event 按 time→kind→side→object slot canonical order；相同时使用固定字段 tie-break；EOS 后和 absent slot 为 literal `+0`；
4. object track/part binding由固定规则生成并写 receipt，不允许任意 learned embedding暗藏图像；
5. fixed-packet clamp：固定 packet/state/text/noise，替换 reference image，pre-safety action最大差异 ≤`1e-6`；
6. permutation test：等价 track permutation 连同 binding 更新后输出等变；event slot permutation经 canonicalize 后 bit-identical；
7. intervention locality：只改 first-contact time，不能让 release 前所有动作无差别整体缩放；只改 right-hand event，left-hand 非协调 phase 留在预注册 tolerance；
8. bypass probe：固定 packet 后，action hidden 不能解码被替换的 image identity；同时 equal-bit code与 semantic packet比较；
9. 记录理论 bit 上界、经验 entropy、code usage、invalid/canonicalized state rate和实际送入 action 的 packet hash。

注意当前 action expert 仍直接读取 text：`transformer_wa_casual_mot.py:1791-1793` 的 action conditioner 与 action-only cache 均保留 text。[I] 所以在不切断 text 的设计中，只能声称“唯一 action-visible **visual** route”，不能声称完整 task-semantic bottleneck。

## 8. 标签 schema：2D/3D/contact/visibility/censor

### 8.1 episode / event 主表

```text
episode_id, clip_sha256, fps, coordinate_contract
event_id, event_kind, hand_side, object_slot, object_role
phase_start_lo, phase_start_hi, phase_end_lo, phase_end_hi
keypose_time_lo, keypose_time_hi
time_support: exact | interval | left_censored | right_censored | unknown
pre_state_observed, post_state_observed
annotator_id, proposal_source, confidence, adjudication_status
```

`event_kind` 可以重复；phase transition validity另存，不用 slot index隐式表达。episode 进入画面时已抓住物体必须标 left-censored，结束仍持有标 right-censored；边界不清用 interval，不能硬点一个 frame。

### 8.2 2D hand pose（主方案，不依赖 MANO）

仅在 event keyframe 标 functional landmarks：

```text
wrist, palm_center, thumb_tip, index_tip, middle_tip,
ring_tip, little_tip
```

每点保存：

```text
x_norm, y_norm
visibility: visible | self_occluded | object_occluded |
            out_of_frame | motion_blur | unknown
support: direct_frame | temporal_context
loss_mask, confidence
```

同时保存 object-relative descriptors：approach direction、palm normal proxy、grasp family、aperture band、selected surface cell。直接可见点才参与 coordinate loss；遮挡点不由标注者想象补全。

### 8.3 3D / MANO（可选 teacher，不是必要 GT）

```text
metric_valid, coordinate_frame: camera | world | object
wrist_translation, wrist_rotation
joint_xyz or mano_pose/mano_shape
intrinsics_id, extrinsics_id, depth_validity
teacher_name/version, reprojection_error, temporal_consistency
human_verified_projection, loss_mask
```

- 没有标定/可靠 depth 时 `metric_valid=false`；不把 monocular reconstruction 当 metric GT。
- MANO 只作 teacher、dense baseline或辅助 loss，不进 action packet，不直接 retarget 到 RH56/OmniPicker。
- 对透明器皿/液体不要求透明重建；hand pose/contact event 可以由 RGB temporal evidence与 object rim/boundary标签独立建立。

### 8.4 contact

```text
contact_state: separated | imminent | first_touch | sustained | released | unknown
contact_mode: none | stick | slide | roll | impact | unknown
hand_site_visible, object_slot, object_surface_cell
contact_evidence: rgb_direct | rgb_temporal | mocap_mesh |
                  native_sim | tactile_force | unknown
contact_time_lo, contact_time_hi, censor
field_mask, confidence
```

`hand_status=closed`、mask overlap、mesh proximity、teacher输出只能作 proposal；除非 source contract 明确，不等于 physical contact truth。

### 8.5 missingness 的硬规则

- existence、observability、censor 分开；`unknown` 不转 negative；
- categorical field 只有完整已知才参与 CE；部分 one-hot 不补零；
- contact 已知但 pose 遮挡时，监督 contact、mask pose；反之亦然；
- left/right censor 用 survival/interval likelihood，不把 observation window 边界当真实 event；
- proposal、human direct、temporal inference、native GT 分库存储并带 source hash；
- phase grammar 不得强迫所有 episode 出现所有七类事件；没有 regrasp 是合法 absence，不是漏标。

## 9. 最小人工成本

不做“每帧 21 joints + MANO + dense contact”。最小协议分三步：

1. **phase/event click pass**：每段点击 4–8 个 event interval，选 kind/side/object/contact state；目标 60–90 s/episode，66 段约 1.1–1.7 人时。
2. **event-frame 2D pass**：只在 keyframes 校正 7 个 functional landmarks、surface cell、visibility；自动 tracker仅作 proposal；目标 2–4 min/episode，约 2.2–4.4 人时。
3. **质量控制**：随机且 phase-stratified 的 20% episode 双标，所有 boundary/contact分歧裁决；估计总计约 4–7 人时，不含 UI 搭建和协议迭代。

先只标 12 个现有 formal packet episodes，若 Gate 0 失败，不扩大到 66。若 event-frame 7 点仍成本/一致性差，可降为 wrist+palm+五指 tip visibility中的可靠子集，而不是转为 dense MANO人工标注。

## 10. 当前 WAM 插入位置（只设计，未实施）

### 10.1 现有可复用 plumbing

- `world_action_model/contact_transition_schema.py:65-106`：当前 22D segment field与连续 start/end定义。[I]
- 同文件 `:114-205`：equal-bit codec 与容量 receipt；v2 应复用审计思想，不复用 v1 state 数。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:1335-1403`：当前 planner 从 state/reference/text 预测并 transport contact route。[I]
- 同文件 `:1405-1445`：`contact_semantic -> contact_condition_encoder -> action width`，raw planner hidden 不进入 action。[I]
- 同文件 `:1809-1829`：strict attention mask 控制 action/state/contact 对 reference 的读取。[I]
- 同文件 `:2359-2485`：action-only prefix cache拒绝 strict mode 的 image conditioning并缓存 packet receipt。[I]
- `world_action_model/datasets/wam_lerobot_dataset.py:741-819`：现有 contact sidecar 的 strict load/hash/shape入口。[I]
- `world_action_model/trainer/wa_casual_trainer_mot.py:897-981`：visual/action/contact loss入口。[I]

### 10.2 建议新增/修改位置

| 位置 | 建议 |
|---|---|
| 新增 `world_action_model/contact_pose_event_schema.py` | v2 enum、HSMM transition graph、duration/censor encoding、canonical order、finite capacity receipt；不修改 v1.1 checkpoint语义 |
| `wam_lerobot_dataset.py:711,741-819` | 加独立 event-tube sidecar loader，校验 episode/frame clock、source hash、annotation identity、2D/3D/contact/visibility/censor masks |
| `wa_transforms_lerobot_pretrain.py` | 所有 2D landmarks/surface cell应用与 RGB 相同 crop/resize/flip；3D/world fields不做错误 image transform；保存 augmentation trace hash |
| `transformer_wa_casual_mot.py:685-856` | opt-in phase/tube planner、finite v2 encoder与 dual-clock phase state；legacy contact route 保持可复现 |
| `:1335-1403` | planner只从 current reference/state/text/history产生 tube；新增 HSMM forward-backward/Viterbi或 differentiable duration loss，不读 future GT input |
| `:1405-1445` | action只接收 compiled hard v2 packet；禁止 raw tube hidden、soft distributions、continuous teacher/spline |
| `:1809-1829` | v2 strict route延续 action→ref cut；增加 fail-closed image conditioner检查；明确 text route的 claim边界 |
| `:2359-2485` | cache只存 active/next hard packet、phase state、receipt与允许的 text/state KV；换 observation 时 invalidation |
| `wa_casual_trainer_mot.py:897-981` | grouped event CE、interval-censored hazard/duration loss、masked 2D/optional 3D/contact loss、order/transition penalty、training-only dense reconstruction probe |
| 新增 opt-in config | 单帧、v1 contact-only、HSMM v2、AR tube、K-spline upper bound、equal-bit、aux-only分别配置，绑定 schema/data/split/receipt hashes |

### 10.3 不能复用/偷换的接口

- 现有 22D 的 `start_tau/end_tau` 是 chunk-local contact segment，不等于 episode-scale phase tube；
- 现有 `Q=12` query 不是 variable-length event sequence，presence slots也没有 event kind/pose；
- contact `site=primary/support/...` 不是 human articulated hand pose；
- future visual hidden 可读 action/contact，不能直接拿它解码 event 再声称 event由 action引起；
- action text cross-attention仍存在，不能写“完整 task semantics 只经过 tube”。

## 11. Benchmark 设计

### 11.1 L0：public perception/compression benchmark

1. **ARCTIC 主 benchmark**：从动态 contact与 dense 3D hands自动提候选 event，再人工/规则核验；按 subject/object分组。比较 single grasp、uniform K、event K、dense sequence。[R/S]
2. **HOI4D**：用 frame-wise 3D hand pose + action segmentation评估 phase boundary、event pose；contact指标只在另有可验证标签时报告。[R/S]
3. **HOT3D**：评估遮挡、tracking和 event pose compression；mesh proximity单列为 geometric proxy。[R/S]
4. **Ego4D**：评估 future hand trajectory/TTC，与 OCT/HandsOnVLM类 trajectory predictor比较；不报告 3D articulated contact。[R/S]
5. **Ego-Exo4D**：多视角训练、ego-only测试，评估 key step + 3D hand pose；不把 keystep当 physical contact。[R/S]

### 11.2 L1：policy benchmark

- RLBench2/PPI tasks：curved motion、handover、tray carry等，检验 keypose不足与连续轨迹上限；
- ManiSkill/robomimic：可得到 native contact/phase/action，做 oracle tube与 predicted tube；
- 至少包含：grasp-lift-place、cap/hinge rotation、tool-target contact、regrasp、bimanual stabilization、release timing；
- 每个 task必须有“相同语义但不同 contact order/pose process”的变体，否则语言/对象语义可替代 tube。

### 11.3 L2：目标硬件与 66ep 的位置

目标硬件主结论必须等待 asymmetric25 driver/calibration/safety全部 ready；当前为 No-Go。66ep 的 ego pouring不能替代 robot policy benchmark，也不能与 BridgeACT/VLAff真实 robot结果直接做公平数值比较。

## 12. Baselines 与消融

### 12.1 必须 baseline

1. Original GWP-0.5 direct action；
2. 当前 v1.1 finite contact-only packet；
3. single final grasp / single contact keypose；
4. single next keypose；
5. uniform-K 2D/3D poses（同 K/标注量）；
6. OCT-style future wrist trajectory + final hotspot；
7. PPI-style target keypose + object pointflow；
8. dense per-frame hand pose/contact predictor；
9. LatentAct-style VQ interaction trajectory；
10. HSMM hard event tube（主方法）；
11. AR variable-length tube；
12. K-query spline/ODE continuous upper bound；
13. action-visible soft/continuous event tensor；
14. aux-only event prediction，action仍 direct；
15. 与 v2 相同理论 bit、参数/FLOPs 的无语义 hard code；
16. oracle tube与 predicted tube；
17. 可适配时再比较 VLAff/BridgeACT-style grasp+trajectory/flow controller，不可适配时只作概念前例。

### 12.2 关键消融

- event-aligned vs uniform/random K；
- contact-only / pose-only / contact+pose；
- 去掉 approach、precontact、regrasp、load-motion、release各 event；
- 无 phase transition graph / 无 duration / framewise phase classifier；
- point time vs interval/censor likelihood；
- 2D functional pose / optional 3D / MANO teacher；
- continuous training-only pose vs continuous action-visible pose；
- Qmax `4/6/8/12` 和 time bins；
- active-only vs active+next1/next3 events；
- full episode tube一次预测 vs receding-horizon更新；
- correct order vs event shuffle / timing delay / contact-state corruption；
- semantic packet vs equal-bit code；
- hard-ST vs hard-detached；
- deterministic quantized geometry vs learned ROI feature（后者只能作 leaky control）。

## 13. 指标

### 13.1 event / phase

- one-to-one event precision/recall/F1；
- onset/offset/keypose boundary MAE与 interval coverage；
- segment tIoU；
- phase sequence edit score、transition violation、duration NLL；
- left/right-censored subset calibration。

### 13.2 pose / contact

- 2D PCK/NME、event ADE/FDE；
- 有 metric GT 时报告 root-relative / camera-relative MPJPE、PA-MPJPE；
- object-relative approach/roll/pose-family macro-F1；
- contact surface cell accuracy/geodesic distance、contact mode macro-F1；
- visibility status κ/F1与 field-level ECE/Brier。

### 13.3 compression 与 action

- K/bit/标注分钟 vs dense sequence reconstruction/contact preservation curve；
- task/phase success、correct-first-contact、regrasp recovery、correct release；
- collision/drop/spill/safety intervention；
- action MSE仅作辅助，不能代替 task/contact outcome；
- action-only latency、replan频率、cache内存/jitter。

### 13.4 causal / covert-channel

- theoretical bits、empirical entropy、code usage、invalid state rate；
- fixed-packet reference-swap action difference；
- field intervention locality与phase-specific action effect；
- image identity/text identity probe（分别报告，因为当前 text route合法存在）；
- semantic vs equal-bit matched control；
- packet clamp、permutation、cache/uncached一致性。

## 14. 成功与停止门槛

### Gate 0：12-episode 标注协议

- event one-to-one F1 ≥0.80；
- phase/event kind κ ≥0.70，contact state/mode κ ≥0.60；
- observed boundary median MAE ≤2 processed frames（20 Hz 为100 ms）；
- matched phase median tIoU ≥0.70；
- 2D visible landmark A/B NME ≤图像对角线3%；
- visibility status κ ≥0.70；
- 至少80% reviewed episode有≥4个可观察 event，且至少3种 event不由单个 episode贡献；
- proposal与human/adjudicated source严格分离。

协议最多修订一次；仍不过则停止“human-auditable event tube”，不扩大到66。

### Gate 1：66ep perception / compression

episode-held-out（不能 frame随机 split）：

- event tube 相对 single-grasp event F1至少 +10 pp，boundary MAE降低≥20%；
- event-aligned K 相对 uniform K，在相同 K/2D标注量下 contact-state F1至少 +5 pp；
- Q≤8保留 dense teacher至少90%的可观察 contact transitions；
- predicted tube恢复 oracle tube perception/probe gain的≥60%；
- phase predictor不能仅靠时间：time-only baseline必须显著更差；
- 95% episode bootstrap CI不跨0。

若只在滑窗/frame级显著、episode级不显著，停止扩大模型。

### Gate 2：public/native oracle sufficiency

- ARCTIC/ManiSkill native GT上，oracle event tube相对 v1 contact-only：correct-first-contact ≥+10 pp、regrasp/release accuracy ≥+10 pp；
- 相对 single grasp，task/phase success ≥+5 pp；
- 相对 dense sequence，性能差≤3 pp，同时监督字段/存储/推理 token减少≥5×；
- uniform K、random K、equal-bit code都不能复制主结果。

若 dense sequence明显更强而 event tube无标注/泛化优势，idea退化为压缩工程，No-Go。

### Gate 3：predicted tube进入 action

- predicted tube恢复 oracle action gain的≥70%；
- hard finite版本距soft continuous上限≤3 pp；
- fixed-packet reference swap action差≤`1e-6`，bypass probe不过随机/预注册阈值；
- 至少75%的可行 field interventions产生正确 phase-local effect，未干预手/phase不越 tolerance；
- latency不破坏当前action-only部署预算。

若增益只在连续 pose/spline/learned ROI进 action时出现，停止 finite/covert-channel主张；可降级为普通 trajectory-conditioned policy。

### Gate 4：真实 OOD / target hardware

- 恢复 operator/session/object/liquid/batch group或重采集；
- 至少三个 task family，语义相同但 contact order/pose不同；
- 相对 matched contact-only / direct policy，OOD task success ≥+5 pp且clustered 95% CI下界>0；
- correct-first-contact/release至少一项 +10 pp，安全指标不退化；
- hardware driver、mapper、calibration、watchdog、E-stop与receipts全部通过。

当前 Gate 0–4 均未通过；现在训练 5B 或上真机都是 No-Go。

## 15. 66ep 现在能做什么、不能做什么

### 能做

- 检查完整 episode 与 48-frame chunk 的双时钟设计；
- 12 episode event/pose/contact A/B+adjudication pilot；
- 66 episode 的 event count、duration、censor、visibility分布；
- episode-held-out phase/event predictor和time-only baseline；
- single grasp、uniform K、event K、dense teacher之间的离线compression/probe；
- 2D functional landmark稳定性、可选MANO teacher coverage/重投影审计；
- v2 sidecar、mask、canonical order、EOS、capacity receipt和counterfactual transform plumbing；
- fixed-packet/reference-swap等CPU级因果路由测试设计。

### 不能做

- 把 `hand_status` 当 physical contact；当前 proxy missingness严重且closed不等于接触；
- 证明metric 3D hand/tool pose、液体/透明接触、force/load；
- 证明robot action或action→contact因果效果；
- 证明target RH56+OmniPicker可执行性；
- 证明跨operator/object/liquid/session/batch OOD，因为group metadata缺失；
- 把20,013滑窗当20,013独立样本；
- 与VLAff/BridgeACT/VideoManip真实robot success直接公平比较；
- 证明MANO reconstruction是真值或event tube已经优于全序列robot imitation。

## 16. 最小推进顺序

1. 不改模型，先完成12 episode的 phase/event A/B+adjudication；
2. 报告 event数、duration、censor、visibility、重复regrasp与Q overflow；据此冻结v2 vocabulary/Q/time bins；
3. 在ARCTIC dense GT做single/uniform-event/dense compression sufficiency；
4. 在66ep做小型frozen-backbone/time-only/HSMM probe，不上5B；
5. 只有Gate 0/1通过才实现独立v2 sidecar与training-only continuous heads；
6. native sim先做oracle tube→action，证明中间表示有控制价值；
7. 通过Gate 2后再接hard packet causal route，并做equal-bit/soft/leaky controls；
8. 最后才考虑更大human video pretraining和target hardware。

## 17. 最终 Go / No-Go

| 决策项 | 判断 | 原因 |
|---|---|---|
| “首次 temporal hand-contact trajectory” | **No-Go** | LatentAct、TOCH、Text2HOI、ARCTIC等直接覆盖 |
| “首次 keypose + contact / phase” | **No-Go** | OCT、PPI、Keypose Exploration、Ego4D/HSMM等覆盖组件 |
| 作为普通 dense hand-sequence reconstruction | **No-Go** | 拥挤且依赖不透明3D/MANO teacher，偏离项目finite route |
| 作为 finite-contact WAM 的 explicit-duration event扩展 | **Conditional Go** | 仍有清晰、可证伪差异，但必须过annotation/compression/oracle/causal gates |
| 推荐 Semi-Markov finite event tube | **Go，仅限设计与低成本Gate 0/1** | 最适合66ep、删失标签与现有22D plumbing |
| AR variable-length tube | **后备 Conditional Go** | 需公共数据预训练和非法序列控制 |
| K-query spline/ODE | **只作upper bound** | 连续高带宽、容易成为visual covert channel |
| 现在训练5B / 使用GPU / 上真机 | **No-Go** | 无reviewed GT、无oracle action证据、无OOD groups、硬件门未过 |

最终推荐把 Idea F 放在 Idea A 后面，而不是与其竞争：Idea A 回答“动作视觉信息是否经过有限可审计接口”，Idea F 回答“这个接口的正确时间粒度是否应该是带持续时间和删失的多阶段 event process，而不是单一 contact/grasp”。只有当 hard event tube 在相同 bit/标注预算下同时打败 single grasp、uniform K 和 contact-only，并接近 dense sequence/soft upper bound，Idea F 才值得成为主方法组成部分。

## 18. 原始来源

### Temporal HOI / hand pose / contact

- [TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement](https://arxiv.org/abs/2205.07982)
- [ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation](https://openaccess.thecvf.com/content/CVPR2023/html/Fan_ARCTIC_A_Dataset_for_Dexterous_Bimanual_Hand-Object_Manipulation_CVPR_2023_paper.html)
- [Text2HOI](https://arxiv.org/abs/2404.00562)
- [How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions](https://arxiv.org/abs/2504.12284)
- [CAMS](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CAMS_CAnonicalized_Manipulation_Spaces_for_Category-Level_Functional_Hand-Object_Manipulation_Synthesis_CVPR_2023_paper.pdf)
- [GEARS](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_GEARS_Local_Geometry-aware_Hand-object_Interaction_Synthesis_CVPR_2024_paper.pdf)
- [WHOLE](https://openaccess.thecvf.com/content/CVPR2026F/html/Ye_WHOLE_World-Grounded_Hand-Object_Lifted_from_Egocentric_Videos_CVPRF_2026_paper.html)
- [Joint Hand Motion and Interaction Hotspots Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Joint_Hand_Motion_and_Interaction_Hotspots_Prediction_From_Egocentric_Videos_CVPR_2022_paper.html)

### Keypose / phase / anticipation

- [PPI: Gripper Keypose and Object Pointflow as Interfaces](https://arxiv.org/abs/2504.17784)
- [Keypose Exploration](https://arxiv.org/abs/2606.29028)
- [Keyframe-Focused Visual Imitation Learning](https://proceedings.mlr.press/v139/wen21d.html)
- [Ego4D Forecasting](https://ego4d-data.org/docs/benchmarks/forecasting/)
- [HandsOnVLM](https://arxiv.org/abs/2412.13187)
- [Discriminative Human Action Segmentation and Recognition using Semi-Markov Model](https://doi.org/10.1109/CVPR.2008.4587557)
- [Hidden Semi-Markov Models review](https://doi.org/10.1016/j.artint.2009.11.011)
- [FACT: Frame-Action Cross-Attention Temporal Modeling](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html)

### Human/ego-to-robot

- [VLAff](https://arxiv.org/html/2608.05215)
- [BridgeACT](https://arxiv.org/html/2604.23249)
- [Being-H0](https://arxiv.org/abs/2507.15597)
- [EgoVLA](https://arxiv.org/abs/2507.12440)
- [ImMimic](https://proceedings.mlr.press/v305/liu25b.html)
- [VideoManip](https://arxiv.org/abs/2602.09013)
- [HOST](https://arxiv.org/abs/2607.20033)
- [Human Universal Grasping](https://arxiv.org/abs/2606.17054)
- [ContactPose](https://contactpose.cc.gatech.edu/)
- [OakInk](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.html)

### Datasets

- [Ego4D](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html)
- [Ego-Exo4D](https://ego-exo4d-data.org/)
- [HOT3D](https://facebookresearch.github.io/hot3d/)
- [HOI4D](https://hoi4d.github.io/)
- [ARCTIC](https://arctic.is.tue.mpg.de/)
