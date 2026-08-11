# Idea B：面向 RH56DFTP + OmniPicker 的不对称双手接触拓扑 WAM

> 日期：2026-08-11
> 研究状态：**有条件 Go；当前硬件部署 No-Go**
> 基线代码：`d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88`
> 隔离分支：`codex/idea-b-asym-contact-topology-20260811`
> idea-discussion `idea_id`：`24a3b591-7839-4593-8cef-169e01acf1d6`
> idea-discussion `discussion_id`：`5e9ca133-76e3-4818-8322-fa6e60185a61`

## 1. 结论先行

这个方向**不能**再以“用 contact latent 缩小 human/robot embodiment gap”或“接触表示再接 embodiment-specific decoder”为主创新。KITE 已经把未分配到具体手指/连杆的接触集合编码成 interaction-intent latent，再用 embodiment-specific decoder 做闭环跨形态迁移；TopoRetarget、Transferring Contact, Not Just Motion、CrossDex、UniDex、D(R,O) Grasp 又分别覆盖了接触图、力/位置接口、共享手部 latent、功能-执行器对齐和跨手型接触几何。AsymDex 甚至已经在一只 Allegro、一只 Ability Hand 的异构双手硬件上研究了 dominant/facilitating role。

仍有机会发表的窄问题是：

> **把第一视角人手—物体交互还原为可枚举、可监督、可干预的时序双手接触拓扑；把 source-human 的左右手与手指身份消去，再显式求解到目标机器人左 RH56DFTP 五指接触集合与右 OmniPicker 双夹面/单 aperture 的可行分配；只允许这个 hard topology packet 把视觉和语言中的 object/part/contact target、手间角色和接触时序送入 action expert，并用左手触觉闭环修正。**

这不是泛称 object/part/contact token。最小可发表单元必须同时包含：

1. 明确的 tracked object、semantic part 和 part-local surface cell；
2. 接触边的开始、保持、滑动/滚动、释放与先后/同步关系；
3. source-human functional contact role 到 target left/right/site-set 的离散分配；
4. 对 `6 DoF dexterous left + 1 DoF coupled-jaw right` 不对称可行性的显式约束；
5. 有限 action-visible 容量、无 raw vision/text/planner-hidden 绕路，以及 topology intervention；
6. 真实目标硬件的标定、触觉/电流反馈和少量 robot data。

最终判断：**方法研究有条件 Go**。若题目退回“contact-aware WAM”“contact latent transfer”或仅给现有 22D grammar 增加几个 object/part 类别，则 **No-Go**。在缺 driver、标定与目标机器人数据的当前仓库状态下，任何真实机成功或部署结论均 **No-Go**。

## 2. Zotero-first 检索记录

先运行了 `idea-discussion` 的默认 Zotero 同步与讨论入口，再直接扫描本地 Zotero 题录。自动讨论给出的初始证据包括 RT-2、RT-1、FAST、RoboBrain 与 π0.5；它们只能作为广义 VLA/跨 embodiment 背景，不能支撑本 idea 的新颖性。随后按 contact retargeting、bimanual、affordance、morphology、HOI 等关键词核对原文。

本地 Zotero 中直接相关的条目包括：

| Zotero key | 条目 | 本 idea 的关系 |
|---|---|---|
| `CSMS2VV2` | TopoRetarget | 接触图与跨手型运动重定向的最近冲突之一 |
| `UAGPBXR8` | Ego2Robot | 第一视角 human-to-robot 的近期背景 |
| `WM8H622U` | EgoInfinity | ego 数据规模与泛化背景 |
| `EY2ZN3YX` | EgoAERO | ego affordance/robot transfer 背景 |
| `ZJQINYLA` | Dexterous Manipulation Policies from RGB Human Videos | RGB 人类视频到灵巧手策略 |
| `YJ3UZAPK` | Being-H0.5 | 人类视频/VLA 迁移背景 |
| `NU5EANJL` | SPIDER | 人类交互先验背景 |
| `9TN8KL2Z`, `Z4H74JXQ` | 2HandedAfforder | 左/右/双手 affordance 最近冲突 |
| `78FMEV92` | D(R,O) Grasp | robot-object interaction-centric 跨手型抓取 |
| `FVS59CQW` | EgoVLA | ego VLA 背景 |
| `F4KBC37A`, `4SBAAFGW` | RDT-1B | 双臂 VLA 背景 |
| `MN4RRCXB` | UMI | 跨 embodiment 数据采集背景 |
| `PF89VDP5` | Mobile ALOHA | 双臂模仿学习背景 |
| `HMS2Z7YG` | DexTrack | 灵巧手跟踪/重定向背景 |
| `6EX2YUFN` | UniDex | 多手型功能执行器对齐的最强冲突之一 |
| `THUD3MXU` | Linker / UniDexGrasp adaptation | 跨手型适配背景 |
| `EHQ3JTCA` | EGOWAM | 当前 WAM 直接背景 |
| `8G2GEP62` | ZeroMimic | 人到机器人模仿背景 |

VLAff、KITE、AsymDex 和 2026 年 6 月后的若干论文尚未形成可依赖的本地 Zotero 条目，因此在 Zotero-first 之后补查了 arXiv/作者项目页和正式会议信息。

## 3. 原文冲突核查

### 3.1 最接近且必须正面对比的工作

| 工作 | 已经做了什么 | 没有覆盖的关键缺口 | 对 novelty claim 的约束 |
|---|---|---|---|
| [KITE: Kinematic Interaction Intent as a General Action Interface](https://arxiv.org/abs/2606.22113) | 将未绑定到具体 link/fingertip 的接触集合编码为连续 interaction-intent latent，策略预测 latent，per-embodiment decoder 只借助运动学模型输出动作；展示了 gripper、dexterous hand、composite effector 迁移 | 任务是单执行器 cube picking、keyboard press、bottle pump；没有异构双手共同操作、手间 role/precedence/transfer graph、有限离散容量和触觉闭环；其局部最近可行策略在远初始位姿会失败，运动学 decoder 不包含软垫、腱绳 slack、线缆耦合 | **彻底否定**“contact latent + decoder”本身的新颖性；本工作必须赢 KITE-style baseline，并把创新放在不对称双手拓扑、可审计通道与触觉动力学 |
| [AsymDex](https://arxiv.org/abs/2411.13020) | dominant/facilitating role、relative observation/action、七类双手任务；实机使用一只 16-DoF Allegro 与一只 6-DoF Ability Hand | 没有 ego human transfer、object-part-surface target、接触拓扑 channel 和触觉时序迁移；主方法是仿真 RL | **否定**“左右手不对称/角色分工”本身的新颖性 |
| [TopoRetarget](https://arxiv.org/abs/2606.16272) | 用 human hand/object keypoint sparse interaction graph、distance-weighted Laplacian、方向/关节/穿透项做接触保持重定向；报告 Pen-Spin 提升 40.6 percentage points | per-frame/trajectory retargeting 与 reference RL 为主；单只 dex hand；没有 ego WAM 的 action bottleneck、双手分配、part target、触觉时序 | **否定**“contact topology graph retargeting”本身的新颖性 |
| [Transferring Contact, Not Just Motion](https://arxiv.org/abs/2606.15516) | MANO pose latent、标定 effort→joint torque→fingertip force、per-finger load descriptor、mask-aware flow policy 与 hybrid force-position control | 多种 dex hand，但不是 `6+1` 异构双手共同操作；没有 object/part temporal topology planner | 说明力/触觉不能只靠视觉伪标签，必须做目标硬件标定与物理量闭环 |
| [UniDex](https://arxiv.org/abs/2603.22264) | 8 种 6–24 DoF 灵巧手、50K trajectories；82-D FAAS 按功能把相似 actuator 放进共享 slot；人类 fingertip trajectory retargeting；包含双手 tool task | 不是一只 dex hand + 一只 1-DoF gripper 的共同拓扑调度；没有 ego bimanual role/contact timing 与触觉 correction；论文消融表明无 robot data 时成功接近零 | **否定**“morphology/actuator-aligned latent”本身的新颖性，也否定轻率的 zero-target-data claim |
| [DexMimicGen](https://arxiv.org/abs/2410.24185) | 从 60 条 source demos 生成约 21K demonstrations；按每臂 subtask 分段，支持 parallel/coordinated/sequential 执行和精确时间同步；覆盖双夹爪、双灵巧手及 GR-1 | 每臂仍假设 EE pose + 1D gripper 或 6D hand action；手指动作主要重放 source；contact 多作分段 heuristic；不同实验 embodiment 不是同一机器人上的左 6 + 右 1 | **否定**“双手分段和同步”本身的新颖性；必须比较其 subtask/timing augmentation 思路 |
| [2HandedAfforder](https://arxiv.org/abs/2503.09320) | 278K 2HANDS 数据，预测 left/right affordance masks，并有 left/right/bimanual taxonomy | 没有建立/保持/释放时序，没有 per-finger/jaw 分配、目标形态可行性和闭环控制 | **否定**“从人类视频预测左右手 affordance”本身的新颖性 |
| [VLAff](https://arxiv.org/abs/2608.05215) | 204K ego episodes、5.6M visual affordances、11.6M grasp/trajectory 标注；输出 heatmap、MANO 96-D grasp 和 `T×6` hand trajectory，并在 Fetch/PR2 上通过 retargeting/GraspNet 验证 | 单目标 anchor 为主；contact 由 fingertip 与 object mask 相交得到；没有双手接触关系、finger/jaw target allocation、力/触觉与有限瓶颈 | **否定**“ego affordance + MANO/trajectory 再 retarget”本身的新颖性；本方法必须显示 topology relational gain |

### 3.2 其他不可忽略的边界

- [A Bimanual Manipulation Taxonomy](https://doi.org/10.1109/LRA.2022.3196158) 已经从 contact graph、空间/时间、手角色和对称性组织双手任务。因此“设计一个双手 contact graph/taxonomy”不是贡献；贡献必须落到可执行、可干预的 WAM interface 与 `6+1` feasibility assignment。
- [D(R,O) Grasp](https://arxiv.org/abs/2410.01702) 用 robot-object point cloud relative distance matrix 做 interaction-centric 跨 embodiment 抓取；[GenDexGrasp](https://arxiv.org/abs/2210.00722) 也用 contact map 匹配。它们主要是静态抓取，不处理双手时序与 transfer topology。
- [CrossDex](https://arxiv.org/abs/2410.02479) 用 eigengrasp/MANO shared latent 与 embodiment randomization 做多手型抓取；[UniDexGrasp](https://arxiv.org/abs/2303.00938) 用 state canonicalization 和 proposal/policy 做 universal dexterous grasp。不能把 shared hand latent 当新意。
- [Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations](https://doi.org/10.1145/3723872) 已覆盖几何、DoF、手指数不同的目标手，并显式保持接触区域。
- [TOCH](https://arxiv.org/abs/2205.07982) 的时空 object-to-hand correspondence field、[BimArt](https://arxiv.org/abs/2412.05066) 的双手 contact map、[Text2HOI](https://arxiv.org/abs/2404.00562) 的 contact-map prior 已覆盖人类 HOI 生成/修正，不能把 HOI contact supervision 本身作为贡献。
- [DexMani](https://arxiv.org/abs/2608.00554) 已研究 sequential contact transition 与 manipulability evolution，但集中在单手 in-hand rotation。动态 contact transition 也不是空白。
- morphology-conditioned policy 近期已有 [Embedding Morphology into Transformers](https://arxiv.org/abs/2603.00182)、[UniMorphGrasp](https://arxiv.org/abs/2602.00915) 等工作；只加入 hardware embedding 或 kinematic graph 不够。
- [Affordances from Human Videos as a Versatile Representation for Robotics](https://arxiv.org/abs/2304.08488) 已把“在哪里、怎样交互”的人类视频先验迁移到多机器人任务；纯 affordance heatmap 不是这里的差异点。

## 4. 精确研究问题与可证伪假设

### 4.1 研究问题

给定第一视角人类双手视频、语言、当前 G1 本体状态和场景几何，是否能先预测一个与 source finger identity 无关但保留 object/part/contact target 与接触时序的离散图，再根据目标硬件能力把图中的 functional roles 分配到：

- 左腕 9D + RH56DFTP 六个执行通道、五指/掌面候选接触集合；
- 右腕 9D + OmniPicker 唯一 aperture 通道、两个物理夹面；

并且比直接 25D policy、现有 22D contact segment、KITE-style continuous contact latent、affordance+retargeting 更少依赖目标 robot demonstrations、更稳健地迁移到未见物体/部件/接触顺序？

### 4.2 主假设

H1. **Relational topology 而不是单边 contact token 承载迁移增益。** 在容量匹配下，显式 `sync / precedes / stabilizes / transfers` 关系在必须双手配合的任务上优于 unordered contact set 或 VQ latent。

H2. **不对称 feasibility allocation 是必要组件。** 将 source human hand identity 直接映射到 robot left/right，或使用对称 decoder，会在需要右夹爪稳定、左五指微操作的任务上显著退化。

H3. **触觉只修正 contact state，不重新传任务语义。** 左手实际接触/滑移/载荷相对视觉计划的 residual 能改善保持与释放时机；若允许一个大容量 tactile encoder 绕过 topology，论文的 bottleneck claim 不成立。

H4. **可干预性是表示有效性的必要证据。** 固定当前几何与 proprio，交换 part/role/timing 字段应产生对应动作变化；固定 hard packet 后替换原始 reference/text，动作应保持不变。

## 5. 提议架构：Topology-Allocated WAM（TA-WAM）

### 5.1 从 ego hand/contact graph 到 canonical temporal topology

对 ego 视频先构造感知图，而不是直接把 MANO joints 当 robot action：

1. 跟踪最多 4 个交互 object instance；对象 slot 用固定规则排序，而不是由网络自由排列。
2. 为每个 object 建立 semantic part 与 part-local surface atlas；人的 fingertip/palm mask 只用来观察接触，不保留 source finger ID。
3. 用 depth、mask distance、相对速度与 hysteresis 检出 contact onset/hold/release；估计 stick/slide/roll/impact。
4. 把人类手指接触归并成 functional actor roles，而非 literal left-thumb/right-index。
5. 跨时间聚合成至多 `Q=8` 条 contact edges，并显式预测 edge 间同步、先后、稳定和 transfer 关系。

每条 canonical edge 的离散字段为：

| 字段 | 基数 | 含义 |
|---|---:|---|
| `presence` | 2 | 空 slot 必须 canonical zero |
| `actor_role` | 6 | `precision_primary / opposition / envelope_support / planar_support / tool_carrier / other` |
| `object_slot` | 4 | 当前场景内确定性排序的 tracked object |
| `part_slot` | 8 | `body / handle / lid_cap / opening / trigger_button / rim_edge / tool_tip / other` |
| `surface_cell` | 64 | part-local `8×8` surface cell；不是连续 hidden vector |
| `contact_mode` | 4 | `stick / slide / roll / impact` |
| `load_band` | 4 | 经目标硬件标定的 touch/low/nominal/high；human 标签缺失时用 field mask，而不是伪造数值 |
| `start/end/censor` | 169 | `B=16` 时沿用现有有效时间/截断状态计数 |

8 条边按 `start_bin → object_slot → part_slot → actor_role → surface_cell` canonical sort。相同键使用稳定 track onset 破平局；不得让 query permutation 偷带语义。

任意两条边的无向 slot pair 使用 9 态关系：

```text
none
sync
i_before_j
j_before_i
i_stabilizes_j
j_stabilizes_i
i_transfers_to_j
j_transfers_to_i
mutually_constrains
```

边指向的几何不是 raw crop。一个无学习的 resolver 根据 `object_slot + part_slot + surface_cell` 从当前场景跟踪结果返回所选 cell 的位姿、法向与局部尺度。decoder 不得访问未选择的 object crop、整图特征或自由连续 part embedding。

### 5.2 canonical topology 到左 6 + 右 1 的目标分配

在 topology planner 与 action decoder 之间加入一个显式 **Target Contact Allocator**。它只接收：

- hard canonical topology；
- 当前 25D proprio；
- 被 topology 选中的局部几何；
- 版本化 target capability manifest；
- 上一时刻低带宽 contact residual。

它**不能**接收 RGB、文本、planner hidden、soft logits 或未选 object features。Allocator 对图中的协作 role group 枚举/求解到目标 side/site-set 的匹配：

- 左候选 site：thumb、index、middle、ring、little、palm；其控制仍严格只有 `little/ring/middle/index/thumb_bend/thumb_rotate` 六通道；
- 右候选 site：jaw-A、jaw-B、both-jaws，但三种接触状态都共享一个 aperture actuator，不能虚构两个独立夹爪自由度；
- 代价包含 wrist reachability、所需接触点数、曲率/法向、load band、stick/slide 能力、手间碰撞与 relation feasibility；
- `stabilizes` role 默认优先分配给 1-DoF gripper，precision/roll/transfer role 优先分配给 RH56，但 reachability 或任务关系可以推翻默认值；
- allocator 输出离散 assignment 与 infeasible flag。部署使用 hard argmin/穷举；若训练时使用 Sinkhorn/Gumbel，只能在边界前使用，送入 action 的仍是 canonical hard assignment。

assignment 是 topology、几何、proprio、manifest 的确定性下游函数，不是额外 planner message，因此不额外增加 source→action 信道容量。每次推理记录 assignment、cost decomposition 和 feasibility margin，便于失败归因。

### 5.3 分裂而耦合的动作解码

现有 generic `EmbodimentSpecificActionDecoder` 只是在输出层按 `embodiment_id` 选线性权重。TA-WAM 应改为三个可审计头：

1. `left_wrist + left_rh56` 头：输出左腕 9D 与 RH56 六个 absolute target；通过 assigned fingertip set、load band 和左手触觉 residual 调制。
2. `right_wrist + aperture` 头：输出右腕 9D 和唯一 aperture；网络结构上不允许产生 per-jaw actuation。
3. `coordination` 图消息头：只通过 edge relation 在两个动作头间传递同步/先后约束，不直接输出额外自由度。

解码后经过确定性 safety/feasibility projection：关节/速度/加速度、腕部 workspace、自碰、aperture、接触载荷和 watchdog 限制均来自已验证 manifest/calibration，不在论文里臆造数值。

### 5.4 触觉与闭环更新

- RH56DFTP 的五指力/触觉必须先标定 raw reading、零偏、滞回、同步延迟与饱和；形成 `no_contact/contact/slip/load_band` residual。
- 右 OmniPicker 在没有经核实的触觉接口前，只能使用 aperture tracking error、电机电流或外部腕力传感作为 **optional masked signal**；不得在方法描述里假定它有独立 jaw tactile。
- residual 只允许更新当前 edge 的 `established/slipping/released/load_error` 小状态，例如每 edge 4 个离散状态；它不访问任务文本，也不重写 object/part target。
- 对视觉遮挡期间的接触保持，触觉决定 action correction；对首次接触目标，仍由 topology packet 决定。

## 6. Action-visible 容量与不可绕过设计

### 6.1 精确容量上界

现有 hard 22D grammar 在 `Q=12, B=16` 下的严格上界是：

```text
valid_time_states = 16*17/2 + 2*16 + 1 = 169
old_edge_states   = 1 + (3*6*4*4)*169 = 48,673
old_total_bits    = 12*log2(48,673) = 186.850009 bits
```

提议 topology packet 在 `Q=8` 时：

```text
new_edge_states = 1 + 6*4*8*64*4*4*169
                = 33,226,753
edge_bits       = 8*log2(33,226,753)
                = 199.886736 bits

relation_bits   = C(8,2)*log2(9)
                = 88.757900 bits

total_upper_bound = 288.644636 bits
```

这是语义通道的严格上界，不把有限精度 tensor 偷换成“无限 bits”。若增加 object 数、part vocabulary、surface bins、edge 或 relation state，必须重新生成 capacity receipt。matched-control baseline 使用不超过 288.644636 bits 的 VQ codes；不能拿更小的 baseline 证明显式图更强。

### 6.2 不可绕过路由

严格模式下，数据流必须是：

```text
RGB/RGB-D + text + proprio
        │
        ▼
topology planner
        │
hard canonical packet（唯一 source-semantic route）
        │
selected-cell deterministic geometry resolver
        │
target allocator(topology, selected geometry, proprio, manifest)
        │
split left6/right1 action expert + low-band tactile residual
        │
25D canonical action → safety projection
```

必须同时满足：

- action self-attention 不读 reference/future；prefix cache 不保留 reference K/V；
- action cross-attention 不读文本或 image embeddings；文本只进入 planner；
- action 不读 planner hidden、soft categorical probabilities、contact logits 或 teacher label；
- object/part geometry 只由 hard field 选择，resolver 返回的 selected geometry 在 packet 固定时也固定；
- absent edge、unused relation、padding 和 query 顺序均 canonical；
- 推理日志保存 packet bytes、capacity receipt、allocator assignment 与实际送入 action 的 input hash。

需要四个自动化干预审计：

1. **Clamp test**：固定 packet、selected geometry、proprio 与 residual，替换 reference/text，pre-safety action 最大差异不超过 `1e-6`。
2. **Swap test**：只换 part、surface、role 或 timing 字段，动作必须出现可预测、局部化变化。
3. **Permutation test**：对 object tracks 做等价 permutation 并同步更新 binding，输出保持等变；对 edge slot permutation 先 canonicalize 后输出完全相同。
4. **Bypass probe**：在 packet 固定后，action hidden 不能解码出被替换的 reference/text identity；同时报告 topology intervention 对动作的因果效应。

注意：这里声称的是 **source task semantics 的有限可审计通道**，不是说 proprio、目标硬件几何和触觉物理量也被压成 288.6 bits。

## 7. 当前仓库基线与真实缺口

### 7.1 已经存在，不能重复包装成贡献

- `world_action_model/asymmetric_action_schema.py:18-28` 已定义 model 32D、physical 25D，以及 `left wrist 9 + right wrist 9 + RH56 6 + OmniPicker aperture 1` 的精确布局；`world_action_model/asymmetric_action_schema.py:92-163` 已有 pack/split/pad/mask。
- `world_action_model/contact_transition_schema.py:61-105` 已定义 v1.1 的 22D segment：presence、side、functional site、object role、contact mode、start/end 与 censor；`world_action_model/contact_transition_schema.py:114-187` 已计算 hard transport 的有限容量。
- `world_action_model/models/transformer_wa_casual_mot.py:110-177` 已有 state/contact/action 与 reference/future 的 mask；`995-1169` 已实现 22D 的激活、hard quantization 与 planner；`1171-1211` 已把 `22→action width` 限定为 planner-hidden 到 action 的唯一视觉 route；`1999-2050` 已处理 action-only cache。
- `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py:202-216` 已固定 12 query、22D、16 time bins、semantic hard mode；`460-470` 已保存 route 与 capacity receipt。
- `world_action_model/datasets/wam_lerobot_dataset.py:741-819` 已有带 SHA preflight 的 contact sidecar loader；`world_action_model/trainer/wa_casual_trainer_mot.py:962-981` 已消费 22D grouped loss 与 field masks。

### 7.2 22D grammar 仍缺什么

22D 是有序 temporal segment list，不是 relational topology。它缺少：

- tracked object instance；
- semantic part 与 part-local surface target；
- edge 间 sync/precedence/stabilize/transfer；
- dominant/support role group；
- source-human finger identity 消去后的 target left/right/site-set allocation；
- load band、tactile establishment/slip/release；
- RH56 与 OmniPicker 的 reachability/contact-cardinality feasibility；
- topology-conditioned split decoder。

因此不能把现有 `site=primary/support/palm_or_jaw` 解释成完整手指/夹爪接触分配。

### 7.3 当前“严格瓶颈”还存在文本旁路

视觉 reference 路径已被切断，但当前 action expert 仍有直接文本 cross-attention：

- `world_action_model/models/transformer_wa_casual_mot.py:634-641` 创建 action-side text conditioner；
- `world_action_model/models/transformer_wa_casual_mot.py:1553-1555` 用原始 `encoder_hidden_states` 生成 `encoder_action_hidden_states`；
- action block 在 `world_action_model/models/transformer_wa_casual_mot.py:1586-1590` 接收它；
- cache 又在 `world_action_model/models/transformer_wa_casual_mot.py:2042-2045` 保存它。

所以当前实现可以声称 hard 22D 是唯一 **planner-to-action visual route**，但不能声称它是完整的 task-semantic bottleneck。TA-WAM 严格模式必须让 action-side text cross-attention 为空，或只接受 hard topology 的嵌入。

### 7.4 建议的代码插入位置（本 memo 不实施）

| 位置 | 建议改动 |
|---|---|
| 新增 `world_action_model/contact_topology_schema.py` | 版本化 `contact_topology_v2`、canonical sort、hard encode/decode、relation validity、capacity receipt |
| `world_action_model/datasets/wam_lerobot_dataset.py:741-819` | 新增独立 topology sidecar，不复用/静默升级 v1.1；加载 edge、relation、field mask、confidence、track/part binding digest |
| `world_action_model/models/transformer_wa_casual_mot.py:685-803` | 配置 topology query/relation heads、hard transport、selected-geometry resolver contract |
| `world_action_model/models/transformer_wa_casual_mot.py:995-1211` | planner 输出 hard edge+relation packet；保留 22D legacy 路径，禁止 raw hidden/logit 合入 action |
| `world_action_model/models/transformer_wa_casual_mot.py:1553-1590, 1976-2045` | topology strict mode 切断 action-side text cross-attention，并对普通/缓存推理做同一审计 |
| 新增 `world_action_model/models/asymmetric_contact_allocator.py` | capability-aware hard target allocation；无 RGB/text/planner hidden 输入 |
| `world_action_model/models/action_projectors.py:75-91` | 不直接改 legacy decoder；新增 split RH56/OmniPicker decoder，并在新 config 显式选择 |
| `world_action_model/trainer/wa_casual_trainer_mot.py:962-981` | edge categorical、surface cell、relation、assignment feasibility、tactile-state 与 action consistency loss |
| 新增 `configs/..._asymmetric25_topology_finetune.py` | 独立 opt-in config、capacity/bypass/manifest receipts；不能覆盖 canonical 22D config |
| preflight / heldout eval 对应模块 | 验证 sidecar SHA、object track binding、canonical order、容量、target hardware/calibration 与 intervention suite |

### 7.5 仍然缺失，不能靠 schema 假装完成

`docs/asymmetric_action_schema.md:136-140` 已明确：仓库只返回 canonical 25D，**没有**目标 G1 OmniPicker driver、calibrated canonical-to-device reverse mapper、endpoint integration、watchdog 和 bench/robot tests。`docs/asymmetric_action_schema.md:158-164` 也明确现有 88 episodes 是 dual-Inspire legacy 数据，不是 RH56DFTP + OmniPicker 数据。

还缺：

- 真实 serial、endpoint、左右腕 frame、WBC ordering 与 latency；
- RH56 六通道 command/state 的独立 affine、方向、死区、速度/力限与五指触觉标定；
- OmniPicker aperture 的 command/state mapping、夹面几何、闭合/夹持阈值和可用反馈接口；
- 相机—双腕—base 外参和时间同步；
- object/part/contact topology sidecar 的真实人工审核版本；
- 任意目标机器人 task demonstration 与 held-out evaluation 数据。

## 8. 标注、弱监督与目标机器人数据

### 8.1 Ego 弱监督流水线

可从 2HANDS/2HandedAfforder、Ego4D/VISOR、HOI4D、HOT3D 或自采 ego clips 获得起点，但实际使用前必须核实数据许可与可发布衍生标注。

建议流水线：

1. hand pose：MANO/WiLoR/HaMeR 类估计器，输出左右手 mesh 与 uncertainty；
2. object/part：视频分割与 track，结合 open-vocabulary part segmentation；
3. 3D contact：mask overlap 只能作候选，必须联合 depth、hand-object distance、相对速度与多帧 hysteresis；
4. temporal edge：从 onset/offset、relative tangential motion 生成 stick/slide/roll/impact 与 censor；
5. relation：根据 overlap interval、object motion 因果与手间 handover 生成 sync/precedence/stabilize/transfer proposal；
6. actor role：由接触面积、物体运动贡献、part 类型和时间关系把 literal human fingers 归并为 functional roles；
7. uncertainty：每个 field 独立 confidence/mask，不能因为 presence 确定就监督错误的 part/load/relation。

建议先做 10k–30k 个高置信双手片段的弱标注，再从中抽取至少 1,000 个 gold contact segments 由两名标注者独立复核。报告 object track、part、surface cell、onset/offset 和 relation 的 inter-annotator agreement；低一致性字段宁可 mask。

### 8.2 触觉/物理监督

- human ego 没有可靠力真值：`load_band` 在人类片段中默认 masked，不能用手部形变伪造物理力。
- 目标 robot 的 load band 来自标定后的 RH56 tactile/force；用同步外部力计做离线校准，训练时保留传感器本身的读数。
- 每个接触事件保存 commanded/actual channel、tactile、aperture/current、wrist pose、object pose、timestamp、contact quality 和安全事件。
- 用“视觉计划 edge + 真实触觉 establishment/slip/release”监督 residual head；不让 tactile encoder预测 object/part 类别。

### 8.3 最小目标机器人数据计划

这里不建议宣称 zero-shot target robot execution。推荐按 gate 收集：

| 阶段 | 数据 | 用途 |
|---|---|---|
| H0 bench calibration | RH56 每个已核实触觉 site 覆盖多个姿态/载荷/材料并重复；OmniPicker 至少 100 个空载开合循环和多尺度标准件夹持；左右腕同步/延迟测试 | command/state reverse mapping、零偏、滞回、阈值、速度/力安全界限 |
| H1 non-task contact probes | 约 500 次受控 touch/press/slide/slip，覆盖五指 site、双夹面、常见材质与法向 | tactile residual、load band 与 selected-cell geometry 对齐；不算 task demo |
| H2 target task adaptation | 4 个训练任务，数据效率曲线 `0/5/10/25 demos per task`，总上限 100 条 task demos | 训练 target allocator/decoder；所有 baseline 使用同一 budget |
| H3 held-out evaluation | 至少 6 个任务或 task-object 组合，每方法/条件至少 30 个独立 trials；新物体、新 part 位置、新材质/摩擦与 ego viewpoint 分层 | 置信区间、失败类型与 OOD 结论 |

若 H0/H1 失败，禁止进入 task collection。若 25 demos/task 不足，可以增加 budget，但必须如实把贡献改成 representation/robustness，而不能继续声称 data-efficient transfer。

## 9. Benchmark 设计

### 9.1 必须真正需要 `left dex + right 1-DoF` 的任务

1. **瓶身稳定 + 左手旋/按 cap**：右夹爪稳定 body，左手完成 cap 的 opposition/roll/slide；检验 `stabilizes` 和同步释放。
2. **容器稳定 + 左手揭膜/标签**：右夹 body，左手在 rim/tab part 建立 precision contact 后 peel；检验小 part 与滑动接触。
3. **右手固定 connector/body + 左手理线/插入**：检验 opening target、顺序和柔性物体接触迁移。
4. **右手固定工件 + 左手操作 trigger/button/拨杆**：检验 part discrimination 和低载荷 contact timing。
5. **双手 handover**：一侧建立接触后另一侧释放，检验 transfer edge 与接触重叠窗口。
6. **工具/容器协作**：右手提供稳定夹持，左手以 tool-tip 与目标 part 接触，检验三对象 slot 和 tool relation。

必须预先做 single-hand feasibility audit。若单手或双普通 1-DoF gripper 在任务上即可达到近似成功率，该任务不支持本 idea。

### 9.2 数据切分

- instance OOD：训练/测试物体实例不重叠；
- part-layout OOD：同类物体的 handle/cap/opening 位置变化；
- material OOD：摩擦和柔顺性变化；
- view OOD：ego camera wearer、背景、遮挡与左右手出现顺序变化；
- role counterfactual：几何迫使默认 `right=stabilize, left=manipulate` 反转或宣告 infeasible；
- temporal perturbation：打乱/延迟接触次序会失败的任务；
- hardware perturbation：在安全范围内改变夹持件厚度、tactile bias 或 latency，检验闭环鲁棒性。

### 9.3 指标

- task success 与 phase success；
- first-contact object/part/site accuracy、surface-cell geodesic/3D error；
- edge F1、relation F1、onset/offset MAE、contact-order accuracy；
- target left/right/site-set assignment accuracy、feasibility violation rate；
- slip/regrasp count、object drop、unintended contact、load overshoot；
- action latency、watchdog stop、安全限制触发；
- 每个 target demo budget 下的成功率曲线与 area under data-efficiency curve；
- topology intervention consistency 与 clamped-packet invariance；
- predicted topology 相对 oracle topology 恢复的 performance gap 比例。

## 10. Baselines 与消融

### 10.1 Baselines

所有可训练 baseline 使用相同 backbone、ego/robot 数据、动作 horizon 和目标硬件安全层；容量 baseline 单独匹配 bits。

1. 当前 direct WAM 25D，无 contact；
2. 当前 hard 22D grammar + generic embodiment decoder；
3. 22D + object/part 独立 auxiliary head，但 action 仍不读 relational topology；
4. KITE-style continuous interaction-intent + per-embodiment kinematic decoder；
5. 2HandedAfforder/VLAff affordance mask + MANO/trajectory retargeting；
6. TopoRetarget + BC/flow action head；
7. UniDex/FAAS 或 CrossDex-style morphology latent + target decoder；
8. AsymDex-style dominant/facilitating role-conditioned policy；
9. matched-capacity `≤288.644636 bits` VQ bottleneck；
10. oracle topology、oracle target allocation 两个上界。

### 10.2 必做消融

- 删除 relation，仅保留 edge set；
- 删除 object instance / part / surface cell 中的各字段；
- 删除 target allocator，使用 source left/right 直映射；
- 对称共享 decoder vs split left6/right1 decoder；
- relation/timing shuffle、固定 simultaneous、去掉 transfer；
- hard discrete vs soft semantic vs hard-detached；
- 移除 tactile residual、延迟 tactile、错误 tactile calibration；
- allocator 去掉 morphology/reachability 或去掉 infeasible flag；
- selected-cell geometry vs full scene feature（后者作为明确的 bypass control，而非主模型）；
- 去掉 weak ego pretraining、仅 robot data；
- 去掉 robot adaptation、仅 human/sim；
- topology capacity `Q={4,8,12}`、surface bins `4×4/8×8`；
- 允许 action 直接读 text/reference 的 upper-bound/bypass 版本。

## 11. Go/No-Go 成功门槛

### 11.1 表示与因果门槛

- gold 集上 object/part/edge/relation 指标显著高于 22D 派生与 affordance baseline；
- predicted topology 至少恢复 oracle-topology 相对 direct baseline 增益的 70%；
- clamp test 在相同 topology/geometry/proprio/residual 下，替换 reference/text 后 pre-safety action 最大差异 `≤1e-6`；
- topology 字段 swap 引发预期动作变化，且不是只改变一个无关全局动作尺度；
- matched VQ 不能在同等容量与数据下复制 relational OOD 增益。

### 11.2 任务门槛

- 在预注册的 asymmetric OOD aggregate 上，相对 **KITE-style 与 hard-22D 两个最强 baseline 中更好者** 至少 `+10 percentage points` task success，95% bootstrap CI 不跨 0；
- first-contact part/site accuracy 至少 `+15 pp`，contact-order accuracy 至少 `+10 pp`；
- 至少三个任务族分别受益，不能由单个瓶盖任务拉高平均数；
- `≤25 target demos/task` 时保持上述优势，并报告 0/5/10/25 全曲线；
- oracle allocator 与 predicted allocator 的差距足够小，否则论文应降级为“拓扑感知但尚不可迁移”。

### 11.3 立即 No-Go 条件

- KITE-style baseline 与本方法差距小于 5 pp，或 relation 消融退化小于 3 pp；
- 单手/双普通夹爪 baseline 即可完成 benchmark；
- source human left/right 直接映射和显式 allocator 无显著差异；
- action 在 topology clamp 后仍随 reference/text 改变；
- 人工审核显示 weak relation/part 标签一致性不足，且无法通过 field mask 控制；
- 无可靠 RH56 tactile calibration，触觉增益只来自未标定 raw channel；
- 目标 robot data 实际远超声明且仍不能超过 direct policy。

## 12. 硬件 No-Go 清单

在以下项目全部满足前，不运行真实机器人论文实验：

1. target hardware manifest 与 calibration capture 达到 `robot_verified`，并由实际测量结果与 SHA receipt 支撑；
2. RH56/OmniPicker/WBC driver、canonical-to-device reverse affine、endpoint、watchdog 与 emergency stop 集成完成；
3. camera/base/left wrist/right wrist 外参和传感器时间同步完成；
4. OmniPicker 是否提供可用 current/force/tactile 反馈得到实机确认；没有就保持 mask；
5. collision、joint/velocity/acceleration、load、aperture 与 workspace limits 来自硬件确认；
6. H0 bench calibration 和 H1 safe contact probes 全部通过；
7. 独立 dry-run、仿真/假设备、低速空载和软物体测试通过；
8. 日志能够复现 packet、assignment、25D canonical command 与 device command 的全链条。

当前仓库不具备 1–6，故**硬件执行 No-Go**。本 memo 只做调研与设计，没有训练、没有 GPU 任务、没有更改 canonical 代码或硬件配置。

## 13. 推荐的最小论文叙事

题目不应是 “Contact Tokens for Cross-Embodiment Manipulation”。更准确的工作标题可以是：

> **Topology-Allocated WAM: Auditable Egocentric Transfer to Asymmetric Bimanual Effectors**

贡献顺序应是：

1. 一个包含 object instance、semantic part、surface target、contact phase 和 hand-to-hand relation 的有限离散时序拓扑，不携带 source finger identity；
2. 一个把 functional topology 显式分配到 `left five-finger/6-actuator + right coupled-jaw/1-actuator` 的可行性 allocator 与 split decoder；
3. 一个真的切断 raw vision、text、planner hidden 和 soft logits 的 action-visible route，并用精确容量与因果干预审计；
4. 一个带触觉 contact-state correction、目标机器人少样本曲线和 asymmetric OOD benchmark 的实证结果。

若最终只能实现第 1 点，或只在离线 ego graph prediction 上有效，适合数据/表征 workshop，不足以支撑完整 robot learning 主会论文。
