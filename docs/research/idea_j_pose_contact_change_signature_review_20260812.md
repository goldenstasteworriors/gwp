# Idea J：Pose–Contact Change Signature / Hand-Synergy Dynamics 独立调研

- 日期：2026-08-12
- 审查基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 隔离分支：`codex/idea-j-pose-contact-change-20260812`
- `idea_id`：`383b218b-5c66-4503-b977-530c7f7b3094`
- 主 `discussion_id`：`611d78e5-f36f-4332-b2b5-abebb699925a`
- Zotero 定向复查 `discussion_id`：`4781abe9-bac9-40cd-8408-31acd30deff2`
- 目标硬件：Unitree G1 + 左 RH56DFTP（6 个主动通道）+ 右 OmniPicker（1 个 aperture 通道）
- 范围：只做文献审计、表示设计、数据/实验/止损协议；未训练、未调用 GPU、未修改 canonical 代码。

## 结论先行

**作为独立论文主 novelty：当前 No-Go。作为一个便宜、可证伪的表示 pilot：Conditional Go。**

最值得检验的窄命题不是“hand synergy / motion primitive / contact trajectory 是新的”，而是：

> 在给定 instruction text、机器人 proprioception、embodiment ID 与 diffusion noise 的条件下，工具操作所需的跨人/跨形态信息，是否可以由少量、2D 可复核的 **姿态—接触变化原子** 表达；与绝对 MANO pose、完整 wrist trajectory、22D contact-only 和 SM-FET 相比，这些 derivative-only 原子能否以更低、精确可审计的 action-visible 容量保留相同或更好的目标机器人控制效果？

推荐先做三种互斥结构中的 **A：Factorized Pose–Contact Derivative（F-PCD）**，但只把它当作 SM-FET/22D 的 challenger 与 diagnostic，不与 SM-FET 拼接成新“大一统”模型。若 A 不能在 oracle packet 下做到以下任一项，就停止 standalone Idea J：

1. 相对 SM-FET 在动态重抓/旋转任务上提升至少 `5pp` success；或
2. 在 success 差距不超过 `2pp` 时，把 conditional current-visual route 从 SM-FET 的 `281.932700038062 bit` 压到 F-PCD 的 `75.445813886781 bit`，并通过 equal-bit、clamp 与 field intervention；或
3. 作为 training-only auxiliary，在不增加 action-visible 容量时稳定提升 SM-FET 的 group-OOD success 至少 `5pp`。

若只有连续 MANO、完整轨迹、DMP/ProMP 系数或 action-visible 3D geometry 有效，则结论不是“变化签名成立”，而是 finite derivative hypothesis 失败。

## 证据边界与 Zotero-first 记录

证据标记：

- **[Z]**：本轮先同步并检索用户 Zotero/Obsidian 已读库；
- **[R]**：论文原文、作者页、正式 proceedings 或官方数据页；
- **[I]**：当前仓库 `e212afec` 可定位实现/审计；
- **[D]**：本文提出、尚未实现；
- **[M]**：当前缺真实标签、目标机器人数据或统计证据。

按 `idea-discussion` skill 运行 `--sync-default` 后同步 100 条 Zotero 记录，创建并更新上述 idea card。自动语义检索首先返回 RT-1（`EK59GCPY`）、FAST（`CBLGABU2`）、RoboBrain（`RP7VA84J`）、Octo（`XB5R2DVZ`）等宽泛条目；它们只适合做 action token / generalist policy 背景，不能证明本题 novelty。[Z]

随后定向扫描本地已读库，命中更直接的 UniT、LIDEA、DexGrasp-Zero、EgoDex 与 Motion Primitive 笔记。尤其：

- UniT 笔记已明确记录“视觉后果锚定 + 多 embodiment 共用离散 codebook + embodiment-specific action decoder”；
- LIDEA 笔记已记录“2D feature distillation + 3D geometry alignment”；
- DexGrasp-Zero 笔记已记录“morphology-aligned hand graph + tri-axial hand-agnostic motion primitives + hand-specific executable mapping”；
- FAST 笔记已覆盖离散 action compression。

因此不能因为 Zotero 首轮自动命中偏泛，就把这条赛道当成空白。[Z]

## 1. 生物运动学直觉成立，但不能直接推出机器人表示

### 1.1 支持 insight 的证据

- [Postural hand synergies for tool use](https://pubmed.ncbi.nlm.nih.gov/9822764/) 对 57 种熟悉物体的 15 个手关节角做 PCA，前两个主成分解释超过 80% 方差；但高阶成分仍携带对象信息，而且 posture synergy 不等同于 grip taxonomy。这支持“低维协同存在”，也警告不能只保留一两个静态 PC。[R]
- [Kinematic hand synergies differ between reach-and-grasp and functional object manipulation](https://pubmed.ncbi.nlm.nih.gov/39740352/) 直接比较抓取与真实工具/物体操作，结论是功能操作需要更多、不同的高阶 synergies。它支持本题强调 **变化与操作阶段**，同时否定“一个固定 grasp synergy basis 足够所有工具使用”。[R]
- [Dynamical Synergies of Multidigit Hand Prehension](https://pubmed.ncbi.nlm.nih.gov/35684800/) 从 12 个手部区域的接触力时序提取低维 dynamical synergy，说明 force coordination 的确有低维时空结构；但其输入来自 instrumented glove，不是 RGB。[R]
- [Functionally Divided Manipulation Synergy](https://arxiv.org/abs/2003.11699) 指出普通 grasp synergy 在需要固定部分手指、独立操作其它手指时会限制灵活性，并按 manipulation/fixed finger role 分解 synergy。这正是 RH56 与 OmniPicker 映射时必须显式处理可执行性的原因。[R]

### 1.2 不能越过的推论

1. PCA/NMF synergy 是对指定人群、任务、传感器和归一化方式的统计分解，不天然是跨人、跨相机、跨机器人共享语义。
2. 静态 posture covariance 不等于 dynamical control primitive；`Δpose` 也不包含接触力、摩擦锥或稳定性。
3. 仅 RGB 能支持的是 **effort direction proxy**（push/pull/twist/hold），不是牛顿、牛米或 fingertip load 真值。
4. 一通道 OmniPicker 的 finger synergy space 近乎退化。若一个表示的大部分信息只对 RH56 有意义，它不能被包装为统一 `6+1` 动作语言。

## 2. 前例与 novelty 冲突

### 2.1 Motion primitive / trajectory distribution 已是老问题

- [Dynamical Movement Primitives](https://pubmed.ncbi.nlm.nih.gov/23148415/) 用带稳定 attractor 的非线性动力系统和可学习 forcing term表达离散/周期动作，已有时间缩放、目标适配等经典机制。[R]
- [Probabilistic Movement Primitives](https://proceedings.neurips.cc/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) 维护轨迹分布，支持 conditioning、blending 与 co-activation。[R]
- [DexSynRefine](https://arxiv.org/abs/2605.05925) 已把 motion-manifold primitive、task-space residual RL 和 contact/dynamics adaptation用于五类 dexterous task，并报告真实机器人迁移。[R]

所以“把轨迹压成少量动态 basis / primitive”不是 novelty；相位条件 synergy 只能作为一个可解释 baseline，除非证明 **2D derivative label + exact finite route + asymmetric 6+1 transfer** 的组合效果。

### 2.2 Hand pose + contact sequence codebook 已有直接前例

- [LatentAct / How Do I Do That?](https://arxiv.org/html/2504.12284) 用 residual VQ-VAE 对未来 MANO hand pose 与逐帧 778-vertex contact map 的 interaction trajectory 做 codebook tokenization；其数据引擎从 HoloAssist 中用 hand/object mask overlap 和 HaMeR 生成 3D pose/contact pseudo-label。它直接否定“首次对 hand-pose/contact trajectory 建有限 codebook”。[R]
- LatentAct 的结果还暴露出本题应避免的依赖：输入含 3D contact point，训练目标含 MANO、相机系连续平移/旋转和 dense mesh contact；mask overlap 不是物理接触或力真值。[R]
- [ARCTIC](https://openaccess.thecvf.com/content/CVPR2023/html/Fan_ARCTIC_A_Dataset_for_Dexterous_Bimanual_Hand-Object_Manipulation_CVPR_2023_paper.html) 已提供 2.1M 帧、双手/铰接物体 3D mesh 与动态 contact；[HOT3D](https://arxiv.org/abs/2411.19167) 提供 3.7M 多视角 ego 图像、手/物/相机 3D pose 与 MANO/UmeTrack 格式；[H2O](https://arxiv.org/abs/2104.11181) 提供双手、物体、相机 3D 标注。它们适合做 2D proxy 对 3D oracle 的验证，而不是证明无 3D 依赖。[R]

### 2.3 Human-to-robot / cross-embodiment 表示已高度拥挤

| 工作 | 已覆盖内容 | 对 Idea J 的约束 |
|---|---|---|
| [DexCap / DexIL](https://arxiv.org/html/2403.07788) | 抗遮挡 mocap、wrist/fingertip 3D、IK retarget、点云 diffusion policy；论文明确说缺 force sensing，手型尺寸差会令 IK 失败 | F-PCD 必须与 IK/full mocap 比；不能声称首次绕过逐帧姿态 retarget |
| [DexTrack](https://arxiv.org/html/2502.09614) | 从 MANO/object sequence 先做 kinematic retarget，再用 homotopy optimization + RL/IL 数据 flywheel 学通用 tracker | 完整参考轨迹是强 baseline；其高质量轨迹生成很昂贵，是 finite signature 可争的效率点 |
| [Object-Centric Dexterous Manipulation](https://arxiv.org/abs/2411.04005) | human wrist trajectory high-level generator + embodiment-grounded low-level finger RL | “人提供 wrist intent、机器人自行补 finger control”已有强前例 |
| [DexGrasp-Zero](https://arxiv.org/abs/2603.16806) | morphology-aligned graph、tri-axial motion primitives、物理属性注入，跨 Allegro/Shadow/Schunk/Ability 到 LEAP/Inspire | “hand-agnostic primitive + morphology-specific decoder”本身不新；且其主任务仍是 grasp，不是长时工具操作 |
| [TopoRetarget](https://arxiv.org/html/2606.16272) | sparse hand-object interaction graph + distance-weighted Laplacian + direction/kinematic/penetration约束，Pen-Spin 比 baseline 提升 40.6pp | “接触拓扑变化用于 retarget”不能单独声称新；上游缺失/虚假 contact 仍是其局限 |
| [Transferring Contact, Not Just Motion](https://arxiv.org/abs/2606.15516) | shared pose latent + 标定 torque/fingertip force/load descriptor + hybrid force-position control | RGB 的 force direction 只能是 proxy；真实 compliance claim 必须有标定力/触觉 |
| [UniT](https://arxiv.org/abs/2604.19734) | vision/action/fusion 三分支 cross-reconstruction、共享离散 physical-intent token、embodiment-specific action projection | “共享离散跨形态动作语言”已被占据；Idea J 必须证明显式 derivative fields 比 learned latent/equal-bit 更有用 |
| [LIDEA](https://arxiv.org/abs/2604.10677) | 2D feature distillation + explicit 3D geometry alignment | “不用视觉编辑，做 2D/3D 对齐”不新；本题差异只能是诚实地不要求 3D |
| [BridgeACT](https://arxiv.org/abs/2604.23249) | tool/target affordance、contact region、interaction-centric 3D motion、grasp module + closed-loop controller | object-centric motion/contact bridge 与工具操作任务均强冲突 |
| [VLAff](https://arxiv.org/abs/2608.05215) | 204K ego episodes，visual/grasp/trajectory affordance，MANO grasp 与 trajectory 到机器人执行 | “ego video → grasp/trajectory affordance → robot”不新 |
| [ContactFlow](https://arxiv.org/abs/2607.26579) | 3D contact point trajectory 作为跨 human/robot embodiment 的 video action condition | contact-motion change 作为 embodiment-agnostic condition 已有近邻；Idea J 必须坚持 2D/finite/causal route 差异 |

### 2.4 可守 novelty 的最窄边界

只有以下交集目前尚有可能独立成立：[D/M]

1. 标签是人可复核的 **变化类别**，不要求每帧 metric 3D/MANO；
2. 只运输相对变化，不运输绝对 hand pose、完整 spline、latent hidden 或 contact flow；
3. action-visible alphabet 精确有限，并有 same-capacity arbitrary-code control；
4. 在 `left RH56 6 + right OmniPicker 1` 的强非对称映射上，报告哪些 token 可执行、哪些退化/被 mask；
5. 通过 sign flip、order swap、regrasp insertion 与 packet clamp 证明 action 真正使用变化字段；
6. 相对 22D contact-only 与 SM-FET event tube 给出独立增益或显著压缩。

任何弱化为“动态 hand synergy”“cross-embodiment motion primitive”“contact-aware latent”或“MANO sequence codebook”的标题都应 No-Go。

## 3. 无透明 3D / 无准确 MANO 的 2D 标注协议

### 3.1 基本单位：可见事件窗口，不是单帧

对每个物理事件保留 `pre / change / post` 三个短窗口；boundary 记录 `[lower_frame, upper_frame]`，而不是强迫一个精确帧。每个字段独立记录：

- `visibility={visible,partial,occluded,out_of_frame}`；
- `status={observed,left_censored,right_censored,unknown}`；
- `source={manual,detector_prefill,sensor_confirmed}`；
- `confidence` 只用于 loss mask/审计，不进入 action。

首轮 detector 只做预填：2D hand box/keypoints、object/tool mask、hand-object overlap、optical flow、tool axis。最终标签由人工复核；`unknown` 不得转成 `no_change/no_contact`。

### 3.2 五类可见变化字段

1. **相对平移方向**：wrist/palm center 相对 tool/active-object bbox center 的位移，先用 object bbox diagonal 归一化，再量化为 `static + 8 compass directions`。只比较窗口 median，不保存连续坐标到 action。
2. **相对旋转变化**：优先标 tool shaft / rim / rigid part 的 2D 主轴与 palm axis 的相对角变化：`cw_strong/cw_weak/stable/ccw_weak/ccw_strong`。出平面变化仅在可由 foreshortening或已知 part state 稳定判断时进入 training target，否则 `unknown`。
3. **finger synergy proxy**：不用逐关节 MANO，标注 `open/close/thumb_opposition+/thumb_opposition-/ulnar_curl+/ulnar_curl-/stable_or_mixed`。可见 fingertip-palm 距离、thumb-index aperture、index-middle-ring-little ordering 只作 prefill 与连续 auxiliary。
4. **contact topology delta**：`none/make/break/add_site/drop_site/slide/pivot_or_roll/regrasp_or_transfer`。证据来自手/物边界关系、随后共同运动与 object state change；仅 mask overlap 时最多是 pseudo tier，不能叫 sensor contact。
5. **effort-direction proxy**：`none_or_hold/tool_axis_push/tool_axis_pull/lateral_left/lateral_right/torque_cw/torque_ccw`。它表达意图/作用方向，不表达力大小；若目标物体没有可见运动、工具轴不可定位或接触被遮挡，标 `unknown`。

[Human Hands as Probes](https://openaccess.thecvf.com/content/CVPR2022/papers/Goyal_Human_Hands_As_Probes_for_Interactive_Object_Understanding_CVPR_2022_paper.pdf) 已证明可以仅用 hand/object/contact detection 和相对 box motion 构造 object-local hand motion；[Ego4D](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html) 提供 PRE/CONTACT/PNR/POST 与 state-change 基础；但二者都不把这些 2D cue 当物理 force truth。[R]

### 3.3 连续量的唯一合法位置

以下量只可用于 training-only auxiliary、teacher、probe 或 upper bound，不得进入 F-PCD 主 action route/cache：[D]

- 2D keypoint 坐标、mask、flow、连续角速度/尺度变化；
- MANO pose/shape、3D wrist/tool SE(3)、mesh hand-object distance；
- PCA/NMF synergy basis、连续 coefficient 与 coefficient derivative；
- DMP forcing weights、ProMP trajectory mean/covariance、spline coefficients；
- soft contact probability、continuous contact point/normal/flow；
- force/torque/load magnitude、friction、tactile embedding；
- human–robot DTW/OT coupling、pair ID、matcher hidden。

主 route 若需要上述任何连续量才能达到 oracle sufficiency，必须停止“有限变化签名”claim。

## 4. 三种互斥结构

三者是竞争主表示；第一版不得 concat，也不得让一个结构的 hidden state给另一个结构兜底。

### A. F-PCD：Factorized Pose–Contact Derivative Codebook（推荐）

每个 48-step action horizon 最多运输 `M=4` 个有序 change atom；不存在的 slot 是唯一 canonical absence。present slot 字段为：

| 字段 | 类别数 | 语义 |
|---|---:|---|
| `actor_role` | 3 | `manipulator/stabilizer/bimanual`，不保留 human left/right identity |
| `relative_translation` | 9 | `static + 8` 个 object-local 2D 方向 |
| `relative_rotation` | 5 | 强/弱 CW、stable、弱/强 CCW |
| `synergy_delta` | 7 | open/close、thumb opposition ±、ulnar curl ±、stable/mixed |
| `contact_topology_delta` | 8 | none/make/break/add/drop/slide/pivot-roll/regrasp-transfer |
| `effort_direction_proxy` | 7 | hold、轴向 push/pull、横向 ±、torque ± |
| `duration_bin` | 9 | 8 个有限 bin + overflow/right-censored |

精确上界：[D]

```text
S_slot = 1 + 3*9*5*7*8*7*9 = 476,281
bits_slot = log2(476,281)
bits_packet = 4*bits_slot = 75.445813886781 bit
```

优点：直接检验用户假设；字段可人工检查与 intervention；容量远低于 v1 22D 与 SM-FET。缺点：缺绝对 object/part identity 与 outcome，方向量化可能不足以恢复精确 wrist action；它不能代替 SM-FET 的 event kind、order、node/effect/outcome。

### B. PC-SB：Phase-Conditioned Synergy Basis

从 human 与各机器人域分别学习连续 synergy basis，但只用 shared phase/effect pair 约束 basis identity；action 只读量化后的：`phase(8) × basis_id(8) × activation_delta(7) × wrist_twist(7) × effort_proxy(7) × duration(9)`，最多三个 active/next phase slot。

```text
S_slot = 1 + 8*8*7*7*7*9 = 197,569
bits_packet(K=3) = 52.775991209602 bit
```

连续 basis loading、activation curve 和 reconstruction loss均为 training-only。RH56 学 `6×K` loading；OmniPicker 的 hand basis退化为 `1×K`，大部分 finger-specific basis必须 mask。因此这个结构最可能在 RH56-only 看起来漂亮，却在 `6+1` 统一性上失败。它还与 DMP/ProMP、DexSynRefine、UniT 和 DexGrasp-Zero 邻域拥挤，排名第二，只适合作为强 baseline。[D]

### C. TRW：Topological Regrasp Word Sequence

完全丢弃 pose derivative，只运输最多三个有序 word：

```text
approach / make_contact / establish / stabilize / slide / pivot /
roll / regrasp / transfer / release
```

present slot 同时带 `actor_role(3) × word(10) × functional_site(7) × object_role(4) × duration(9) × effect_direction(9)`：

```text
S_slot = 1 + 3*10*7*4*9*9 = 68,041
bits_packet(K=3) = 48.162350172077 bit
```

优点是最易标、最稳健于遮挡。缺点是它基本落入 Idea B topology 与 SM-FET event word 的已有范围；TopoRetarget 又已覆盖 interaction graph retarget。除非 word-only 在极低容量下接近完整模型，否则它应是 baseline，不是 Idea J。[D]

## 5. 与 22D、Idea B 与 SM-FET 的关系

### 5.1 现有 22D：接触段，不是姿态变化

`world_action_model/contact_transition_schema.py:65-106` 定义 `presence/side/site/object_role/contact_mode/start/end/left_censored/right_censored`；Q12/B16 hard grammar 的精确 conditional visual capacity 是 `186.850008943084 bit`。[I]

22D 已包含 contact mode 与 timing，但没有：

- wrist/tool relative rotation or translation derivative；
- finger synergy activation change；
- regrasp/transfer 作为显式拓扑 delta；
- effort direction proxy。

F-PCD 不能与 22D concat 后声称 75.45 bit；拼接后的容量必须重新计算。公平主比较应是 mutually exclusive `22D-only` vs `F-PCD-only`，再给一个 non-main concatenated upper bound。

### 5.2 Idea B topology：大部分 TRW 已被吸收

Idea B 已提出 human finger identity 消除、object/part/relation topology、RH56/OmniPicker feasibility allocator 与 split decoder。TRW 的 make/break/add/drop/regrasp/transfer 基本是 B 的低容量时序化版本。若 J 调用 B 的 target allocator、part graph 与 tactile closure，J 不再是独立 novelty；它最多贡献一个 derivative field ablation。[I/D]

### 5.3 SM-FET：组合主线已覆盖“事件时钟 + finite action route”

同轮 meta-synthesis `8939659` 已推荐 SM-FET-WAM：episode-level Semi-Markov event clock、`active+next2`、node/contact/relation/effect/outcome、精确 `281.932700038062 bit` packet；连续 pose/MANO 明确只作 training-only auxiliary/upper bound。

因此 J 不应与 SM-FET 并列再造一套 event planner。合理关系只有二选一：[D]

1. **主实验 challenger**：SM-FET-only vs F-PCD-only，直接回答 event/effect state 和 micro-derivative 哪个更 sufficient；
2. **training-only auxiliary**：在 SM-FET planner 内增加 derivative prediction loss，但 action仍只读原 SM-FET hard packet；此时 J 没有额外 action capacity，也不声称 standalone architecture。

把 F-PCD fields直接塞进 SM-FET action packet会扩大 schema、容量和标签负担；在 oracle sufficiency 前禁止。

## 6. Human → RH56 / OmniPicker 的映射

### 6.1 当前 action contract

`world_action_model/asymmetric_action_schema.py:18-39,92-128` 已定义 model 32D、physical 25D：

```text
left wrist 9 | right wrist 9 | left RH56 6 | right OmniPicker aperture 1
```

RH56 六通道顺序固定为 `little, ring, middle, index, thumb_bend, thumb_rotate`；两腕是 local relative transform，手/夹爪是 future absolute canonical open-fraction target。`docs/asymmetric_action_schema.md:136-140` 明确当前没有目标 OmniPicker driver、canonical-to-device mapper、watchdog 与 bench/robot test。[I]

### 6.2 角色先 canonicalize，再做 embodiment-specific decode

human source 不运输 `left/right finger ID`，只运输 `manipulator/stabilizer/bimanual`。第一版 target assignment由 task/hardware manifest固定，避免偷偷复用 Idea B 的 learned allocator：[D]

- 左 RH56 默认承担 precision manipulation、thumb opposition、finger add/drop 与 regrasp；
- 右 OmniPicker 默认承担 fixture/stabilize、binary grasp/release；
- 若任务需要 role switch，必须有显式 `transfer` word与可验证 task manifest；
- human handedness可做 augmentation，但不改变 target hardware side。

### 6.3 RH56 六通道

RH56 decoder 输入 hard change packet、当前六通道 measured state、两腕 proprio 与 embodiment manifest，输出 6 个绝对 target。建议以有界 `6×K` synergy loading 或 monotonic small MLP实现，但 loading/hidden 不跨 route：[D]

- `open/close` 可同时调四指 + thumb bend；
- `thumb_opposition±` 主要调 `thumb_rotate` 与 `thumb_bend`；
- `ulnar_curl±` 主要调 little/ring/middle；
- `add/drop_site/regrasp` 允许改变对应 finger group，但必须有 robot-side demonstration或物理 supervisor；
- wrist relative translation/rotation由左 wrist 9D head执行，而不是塞进六个 finger target。

上述只是结构，不是硬编码设备数值；实际幅度、极性、限位必须来自 target hardware calibration。

### 6.4 OmniPicker 一通道

OmniPicker 只有 aperture：[D]

- `make/close` → 减小 canonical open fraction；`break/release/open` → 增大；`hold/slide/pivot` → aperture保持或做小的 sensor-confirmed correction；
- thumb opposition、ulnar curl、finger add/drop 对 aperture 没有独立可执行维度，必须标为 `unexecutable/masked`，不能让 decoder暗中从这些 token产生任意不同动作；
- regrasp主要由 wrist 9D + open/close时序完成；一通道不能执行多指内部 gait；
- 两个 jaw 是两个接触面，但共享一个 actuator，不能把两个 surface写成两个 action channel。

这正是 PC-SB 的风险：一个跨形态 synergy coefficient并不天然映射到一通道 gripper。统一发生在 **change semantics**，不是 joint basis 数值。

### 6.5 严格 action-visible route

主 claim 只允许：

```text
current reference/state/text -> change planner -> canonical hard packet
hard packet + state + allowed text + noisy action -> embodiment decoder -> 25D action
```

禁止 action/cache 读取 raw reference、2D keypoints、MANO、continuous coefficient、planner hidden/logits、human frame、pair matrix或 detector feature。需复用现有 strict mask/ST/canonical-zero/hash 机制，并报告：

- packet固定后替换 reference，pre-safety action `max_abs <= 1e-6`；
- target label bytes变化但不 oracle-forward，当前 action不变；
- hard-ST 与 detached forward bit-equal；
- cache/uncached packet bytes与 action一致；
- semantic packet与 exact equal-bit arbitrary code 同容量比较。

容量是 **conditional current-image-derived route**，不是总输入容量：text、proprio、embodiment 和 diffusion noise 仍是直接条件。

## 7. Human–robot pair 设计

复用 Idea G / SM-FET 的 event-level CAPOT pair table作为数据协议，不把 matcher当 J 的算法贡献。pair 不要求同步，也不要求完整轨迹对应。[D]

### 7.1 先 split，后构造 many-to-many pair

按 `human operator / robot collector / session-day / object instance / calibration / source episode` 冻结 split；同一 episode 的窗口、augmentation、event 与所有 pair edge不得跨 split。再按以下 cell 生成 many-to-many candidates：

```text
task family
+ tool/target functional roles
+ intended object effect/outcome
+ coarse phase/event
+ executable target-role assignment
```

human 额外 preparation、机器人 recovery、遮挡段可落 NULL。变化签名只作为 pair cost 的一项，不能把完整 trajectory DTW 生成的 pair当 GT再宣称 derivative 更好。

### 7.2 Pair tiers

1. `exact`：同 task、同 object instance/geometry、同 intended effect，人工确认 event对应；
2. `weak`：同 functional role + effect + phase，不同实例/速度/路径；
3. `unpaired`：只参与 domain-specific planner/action loss；
4. `hard negative`：同 task/pre-state但不同 contact delta、regrasp或失败 outcome；
5. `NULL`：额外准备、等待、恢复或不可执行 human micro-motion。

### 7.3 最小 paired benchmark

未来应至少采三类目标硬件任务，每类含 standard success、速度/准备变化、regrasp/partial failure：[D]

- cap twist：RH56 precision twist，OmniPicker stabilize；
- pour/tilt：稳定抓持 + wrist rotation + correct stop/release；
- tool reposition/regrasp：RH56 add/drop finger group或 wrist-assisted regrasp，OmniPicker保持 fixture。

每个 task cell 至少 3 human clips、3 robot rollouts；20% pair edge双标+裁决。正式量级与功效分析需在 pilot effect size后冻结，不能现在臆造“足够样本”。

## 8. Benchmark、baselines 与指标

### 8.1 表示/感知 benchmark

- public 3D oracle：ARCTIC、HOT3D、H2O、HOI4D；
- in-the-wild 2D：Ego4D FHO、EPIC-KITCHENS hand/object tracks；
- static contact sanity：ContactPose；
- current project ego pour：只做 annotation feasibility，不做 OOD/robot claim。

用 public 3D truth计算 2D label 与 oracle 的一致性，但训练/部署主输入只保留 RGB；报告 visible、partial、occluded 分层，而不是只报全体平均。

### 8.2 必做 baselines

1. direct RGB/state WAM；
2. single-frame absolute MANO/hand posture；
3. full wrist/finger trajectory + DTW/BC；
4. DMP 与 ProMP trajectory coefficients；
5. LatentAct-style learned VQ interaction code；
6. DexCap-style IK retarget；TopoRetarget contact-preserving retarget；DexTrack reference tracker；
7. robot-only 与 unpaired human+robot co-training；
8. current hard 22D contact-only；
9. SM-FET-only；
10. F-PCD auxiliary-only direct；
11. F-PCD strict semantic；
12. F-PCD exact equal-bit unstructured；
13. PC-SB 与 TRW 两个互斥 challenger；
14. oracle F-PCD、soft F-PCD、action-visible continuous pose/trajectory upper bound。

所有 policy baseline匹配 robot/human episode预算、backbone、optimizer steps、action horizon、参数/FLOPs、augmentation、split与 seed。DMP/ProMP/continuous trajectory若 action-visible，必须如实报告其高带宽，不得和 75.45 bit直接称同容量。

### 8.3 指标

**标注/感知：** field-wise κ、macro-F1、boundary MAE/tIoU、unknown/censor rate、2D-to-3D sign agreement、occlusion degradation、slot overflow、calibration ECE/Brier。

**表示：** same-event cross-embodiment retrieval R@1/R@5、pair-vs-hard-negative margin、code utilization/entropy/collapse、exact packet bits、action latency与cache bytes。

**动作：** task/phase success、correct-first-contact、relative-rotation completion、regrasp recovery、contact-order violation、correct stop/release、spill/drop/collision、安全介入、post-success overrun。

**形态：** RH56 6D reconstruction/action error、OmniPicker aperture error、unexecutable-token rate、per-side success、role-switch failure、`RH56-only / Omni-only / bimanual` 分层。

**因果：** direction sign flip 的 wrist/action响应、open↔close / make↔break / add↔drop swap、regrasp word insertion、slot order shuffle、duration delay、packet clamp invariance。没有 physical intervention只能写 association/control，不能写 causal action semantics。

## 9. 消融矩阵

- derivative-only vs absolute-only vs derivative+absolute continuous upper bound；
- `translation / rotation / synergy / topology / effort / duration` leave-one-field-out；
- event-aligned窗口 vs uniform固定窗 vs single-frame；
- 2D-reviewed vs detector pseudo vs MANO/3D oracle；
- role canonicalization vs human left/right identity；
- F-PCD `M={2,4,8}` 与各字段 quantization bins；
- 训练期 continuous pose/synergy/force auxiliary on/off；
- human pretrain / weak pair / exact pair / robot-only；
- semantic vs exact equal-bit code；
- hard-ST vs hard-detached；
- RH56 decoder共享/独立、Omni feasibility mask on/off；
- F-PCD vs 22D vs SM-FET，禁止三者主 route concat；
- correct packet、field swap、time reversal、task-matched hard negative。

## 10. Stop gates

| Gate | 最小实验 | Go 阈值 | 失败动作 |
|---|---|---|---|
| **G0 标签可复核** | 现有 ego 中 12 episodes，双标+全分歧裁决 | rotation/synergy/contact κ≥0.70；effort proxy κ≥0.60；boundary P50≤5帧或250ms；关键字段 unknown≤40%；slot overflow≤5% | 协议最多修一次；若可靠字段少于3类，停止 Idea J |
| **G1 2D proxy fidelity** | ARCTIC/HOT3D/H2O visible/occluded子集，对比3D oracle | direction/sign macro-F1≥0.75；contact delta F1≥0.70；partial visibility下降≤15pp | 只能 MANO/3D 通过则删除“无准确重建”claim |
| **G2 pair value** | 人工 gold exact/weak/NULL edge | F-PCD cost相对 phase/effect-only R@1≥+10pp；NULL AUPRC≥0.75；不由 task ID shortcut解释 | 不把 signature用于 pairing；回退 CAPOT base |
| **G3 oracle sufficiency** | target robot native action/outcome；oracle packet | F-PCD相对 full trajectory success下降≤3pp；且相对22D动态任务≥+5pp；相对SM-FET满足“+5pp”或“≤2pp且约27%容量”之一；safety不退化 | 不实现 predicted route；降为 auxiliary/probe |
| **G4 predicted + causal** | matched direct/aux/22D/SM-FET/equal-bit，3 seeds + group rollout | predicted恢复≥70% oracle gain；group-OOD≥+5pp且95% CI下界>0；semantic优于equal-bit；≥75%可行field intervention方向/阶段正确；clamp≤1e-6 | 删除 semantic/finite/causal 对应 claim；只 continuous 有效则停 finite |
| **G5 asymmetric hardware** | driver/mapper/calibration/watchdog/E-stop与3 task families齐全 | action latency增量≤15%；RH56与Omni均有非退化可执行子集；success不由单一task/session驱动；无 safety regression | 保持 target hardware deployment No-Go |

当前只具备 G0 的部分视频条件，G1 可借公共数据设计；G2 缺 gold pairs；G3–G5 缺目标机器人数据/硬件闭环。因此现在训练 5B、宣称人机 transfer 或真机部署均 No-Go。[M]

## 11. 现有数据 pilot：能做什么，不能做什么

### 11.1 66 个 ego pour episodes

仓库审计已证明 66 个 episode / 20,013 processed frames 只有单一 pour task，缺 `subject/session/operator/object instance/liquid/batch/outcome` production group metadata；时间连续块只能做 engineering split，不能作 OOD。[I]

`data/contact_transition_pour_v0/meta.json` 当前只标了 episode 0，并使用旧 `contact_transition_v1` 54D schema。它可帮助选择 `pre/change/post` 窗口，但不能冒充 reviewed F-PCD/22D v1.1 truth。[I]

建议低成本 pilot：[D]

1. 从 4 个 diagnostic continuity blocks 各抽 3 段，共 12 episodes；
2. 只新增离线表格/annotation guide，不接模型；
3. 双标五类变化字段、boundary interval、visibility/unknown；
4. 统计 κ、unknown、slot overflow、字段共现和 F-PCD容量；
5. 用 episode 0 旧 contact label只作 disagreement审计，不自动升级为新 GT。

### 11.2 88 个 robot `grab_red_bottle` episodes

它们是 `29 body + left Inspire6 + right Inspire6` 的 legacy30 数据，不是 RH56+OmniPicker；任务又是抓红瓶，不是 human pour。不能压缩右六维为一个 Omni aperture，也不能和 66 个 human pour组成 scientific pair。[I]

当前 `outputs/immimic_right_hand_dtw/human_001_robot_000011` 虽已有一个 human-pour / robot-grab 的 right-hand DTW artifact，但 task/effect不匹配，只能做代码可视化或 hard negative，不能当 same-task pair、alignment GT或 transfer evidence。[I]

### 11.3 当前最便宜的证伪顺序

1. 完成 G0 的 12-episode双标，不改模型；
2. 用 ARCTIC/HOT3D/H2O 做 2D label vs 3D oracle 的 G1；
3. 纸面实现 exact capacity/equal-bit codec specification和枚举测试；
4. 等待同任务目标机器人数据后，只用小 decoder做 oracle F-PCD vs 22D vs SM-FET vs full trajectory；
5. G3 通过才考虑 planner/WAM实现；G4 通过且硬件闭环后才讨论大模型/真机。

## 12. 最终 Go / No-Go

| 决策项 | 判断 | 理由 |
|---|---|---|
| 2D变化签名 annotation pilot | **Go** | 成本低，能直接证伪可标性与遮挡鲁棒性 |
| F-PCD 作为 22D/SM-FET challenger | **Conditional Go** | hypothesis清楚、容量低、可干预；必须先过 oracle sufficiency |
| F-PCD 作为 standalone论文主方法 | **当前 No-Go** | LatentAct/UniT/DexGrasp-Zero/TopoRetarget/SM-FET 冲突强，尚无独立效果证据 |
| PC-SB 主方法 | **No-Go / baseline only** | DMP/ProMP/动态 synergy 前例强，OmniPicker一维退化 |
| TRW 主方法 | **No-Go / baseline only** | 与 Idea B topology和SM-FET event word重复 |
| continuous MANO/DMP/ProMP action route | **upper bound only** | 会失去无3D、finite、可审计的核心命题 |
| 现有66+88直接做人机pair或target deployment | **No-Go** | task不匹配、group provenance缺失、硬件形态错误、driver/calibration为空 |

## 13. 一句话推荐

**先用 12 个 ego episode 检验 F-PCD 的五类 2D变化标签是否可重复；若可重复，再把 `75.4458-bit F-PCD-only` 作为 `186.8500-bit 22D-only` 与 `281.9327-bit SM-FET-only` 的互斥 oracle challenger。它只有在动态重抓/旋转任务上带来独立增益，或以 SM-FET 约 27% 的容量无损逼近它，才值得升级为方法；否则并入 SM-FET 做 training-only auxiliary。**

## 14. 原始来源索引

### Human synergy / movement primitives

- [Postural hand synergies for tool use](https://pubmed.ncbi.nlm.nih.gov/9822764/)
- [Kinematic hand synergies differ between reach-and-grasp and functional object manipulation](https://pubmed.ncbi.nlm.nih.gov/39740352/)
- [Dynamical Synergies of Multidigit Hand Prehension](https://pubmed.ncbi.nlm.nih.gov/35684800/)
- [Dynamical Movement Primitives](https://pubmed.ncbi.nlm.nih.gov/23148415/)
- [Probabilistic Movement Primitives](https://proceedings.neurips.cc/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)
- [Functionally Divided Manipulation Synergy](https://arxiv.org/abs/2003.11699)

### Hand/contact sequence and ego data

- [LatentAct / How Do I Do That?](https://arxiv.org/abs/2504.12284)
- [MANO](https://mano.is.tue.mpg.de/)
- [ARCTIC](https://arxiv.org/abs/2204.13662)
- [HOT3D](https://arxiv.org/abs/2411.19167)
- [H2O](https://arxiv.org/abs/2104.11181)
- [HOI4D](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_HOI4D_A_4D_Egocentric_Dataset_for_Category-Level_Human-Object_Interaction_CVPR_2022_paper.html)
- [ContactPose](https://arxiv.org/abs/2007.09545)
- [EgoDex](https://arxiv.org/abs/2505.11709)
- [Human Hands as Probes](https://openaccess.thecvf.com/content/CVPR2022/html/Goyal_Human_Hands_As_Probes_for_Interactive_Object_Understanding_CVPR_2022_paper.html)
- [Ego4D](https://arxiv.org/abs/2110.07058)

### Human-to-robot / cross-embodiment

- [DexCap](https://arxiv.org/abs/2403.07788)
- [DexTrack](https://arxiv.org/abs/2502.09614)
- [Object-Centric Dexterous Manipulation from Human Motion Data](https://arxiv.org/abs/2411.04005)
- [DexSynRefine](https://arxiv.org/abs/2605.05925)
- [DexGrasp-Zero](https://arxiv.org/abs/2603.16806)
- [TopoRetarget](https://arxiv.org/abs/2606.16272)
- [Transferring Contact, Not Just Motion](https://arxiv.org/abs/2606.15516)
- [UniT](https://arxiv.org/abs/2604.19734)
- [LIDEA](https://arxiv.org/abs/2604.10677)
- [BridgeACT](https://arxiv.org/abs/2604.23249)
- [VLAff](https://arxiv.org/abs/2608.05215)
- [ContactFlow](https://arxiv.org/abs/2607.26579)
