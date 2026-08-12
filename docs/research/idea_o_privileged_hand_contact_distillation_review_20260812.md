# Idea O 独立调研：Privileged Hand-Contact Trajectory Distillation

- 日期：2026-08-12
- canonical 代码基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 隔离分支：`codex/idea-o-privileged-hand-contact-distill-20260812`
- 范围：文献、架构、损失、审计、实验与 stop 条件；**未实现代码、未训练、未使用 GPU/服务器**
- `idea-discussion` idea id：`7926f496-4b8b-4282-b517-d8b620cf9b4c`
- `idea-discussion` discussion id：`d9842c1c-c98c-4d85-b760-5d950613db12`

## 0. 裁决

**作为“首次用 privileged teacher / human trajectory / future posterior 蒸馏机器人策略”的独立主张是 No-Go；作为 minimal-r1d finite WAM 的受审计训练机制是 Conditional Go。当前实现和训练均 No-Go。**

原因很直接：LUPI、knowledge distillation、Learning by Cheating、asymmetric actor-critic 已经覆盖“训练时多看、部署时少看”的基本范式；Being-H0.7 已经把训练期 future observation posterior 对齐到部署 prior latent；LIDEA 已经从 human 到 pseudo-robot 再到 real robot 做冻结 teacher 的特征蒸馏；EgoBridge 已经做 human/robot joint latent-action OT；DexMani 已经把人类 contact-conditioned manipulability evolution 作为跨手型指导；BridgeACT、VLAff、VideoManip、EgoVLA、ImMimic 已分别覆盖 human video 的 contact/flow、grasp/trajectory、3D reconstruction、wrist/hand action和 full-trajectory transfer。因而“human hand trajectory 教 robot”本身没有 novelty。

唯一仍可证伪、也可能守住的窄问题是：

> 在 action-visible interface 固定为版本化、可枚举、byte-level 可审计的 `minimal-r1d` packet，部署端只读当前 robot observation 与该 packet，且 teacher、human pose、pair table、连续 teacher hidden 均不进入 action/cache 的条件下，高带宽 human hand-contact trajectory 是否能只通过训练损失改善 student planner；这种改善是否在相同 student 参数、推理 FLOPs、robot data、训练步数和 action-visible capacity 下，胜过 generic video representation、Being-H0.7-style future posterior、EgoBridge/LIDEA-style alignment 与 shuffled/random teacher？

若答案不是稳定的“是”，Idea O 应降级为普通 pretraining trick，不作为方法贡献。若必须把 teacher hidden、MANO、point flow、continuous pose、pair id 或 future frame直接送入 action才能获益，则 **strict Idea O 失败**。

还存在一个先决 blocker：canonical `e0ebc692` 只有 v1.1 Q12×22D contact route；本分支没有 `minimal-r1d` codec、五 token encoder、profile validator或其 cache receipt。因此推荐顺序是：

1. 先冻结并移植 `minimal-r1d` structural codec、profile/public-state合同与 route audit；
2. 再做 no-policy teacher ceiling 与 missingness audit；
3. 最后才允许添加 training-only distillation head；
4. 在前三步之前不启动 WAM 训练。

## 1. 调研方法与证据边界

### 1.1 Zotero-first 结果

本轮按 `idea-discussion` skill 同步 Zotero，保存了上述 idea/discussion。自动讨论器同步了100条最新记录，但首轮召回偏向 RT-2、π0、RoboBrain 与通用 imitation，且外部候选混入与本问题不够接近的 recovery/meta-RL 论文；这些弱命中没有被用于具体 novelty 断言。

随后只读检索本地 Zotero 索引（当前共826条）并核对原始论文。真正相关的个人库条目包括：

| Zotero key | 文献 | 与 Idea O 的关系 |
|---|---|---|
| `95PMFGL4` | LIDEA | human→pseudo→real 的冻结 teacher 双阶段 feature distillation；最直接的 cross-embodiment distillation 冲突 |
| `A89APIR3` | EgoBridge | human/robot joint latent-action OT + DTW pseudo-pair；最强 domain-alignment baseline |
| `8G2GEP62` | ZeroMimic | 从 ego web video distill 可部署机器人 skill；否定宽泛“web human skill distillation”表述 |
| `8WMVVJAL` | ImMimic | retargeted human hand full trajectory 经 DTW 映射与 interpolation；full-trajectory transfer baseline |
| `EK2CTN7J` | UniSkill | 无 aligned pair 的 cross-embodiment skill representation；unpaired baseline |
| `V7LLF6VJ` | XSkill | human/robot 无标签视频的 skill prototype 与 diffusion policy；latent skill baseline |
| `ZJQINYLA` | VideoManip | RGB human video→3D hand/object trajectory reconstruction→contact optimization→robot policy；dense trajectory upper bound |
| `XINDFWEJ` / `653NU4JX` | Being-H0.7 | train-only future posterior 与 deployable prior latent alignment；结构上最接近 |
| `EHQ3JTCA` | EgoWAM | 在固定 policy 条件下比较 Pixel/DINO/3D flow world targets；WAM target baseline |
| `FVS59CQW` | EgoVLA | human wrist/hand action预训练后 retarget/finetune；action imitation baseline |
| `JRGPD6HS` | OmniContact | key body trajectory + time-series binary contact flow；“trajectory + contact sequence”不是空白 |
| `LH5DGVVD` | WHOLE | ego hand/object world-space trajectory joint reconstruction与 out-of-sight prior；teacher input ceiling |

Zotero memory 已更新，但“个人库有记录”不等于论文支持本 proposal；下文方法事实均再核对 arXiv、PMLR、JMLR、CVF、OpenReview或作者原文。

### 1.2 证据标记

- **[R]**：原始论文、作者项目页或正式 proceedings 支持；
- **[I]**：canonical `e0ebc692` 代码可定位；
- **[S]**：本文提出的未来设计，尚未实现；
- **[M]**：缺真实标签、训练、统计、硬件或 deployment 证据。

### 1.3 当前数据与代码事实

- canonical 的 MoT config仍是 `num_contact_tokens/contact_target_dim/contact_semantic_bottleneck`，且只接受 v1.1 schema；见 `world_action_model/models/transformer_wa_casual_mot.py:598-631,703-816`。[I]
- 现有 planner 用 state/ref/text 形成 query hidden并直接输出22D logits；raw `planner_hidden` 不离开 `_plan_contact_logits`；见同文件 `:1253-1285`。[I]
- v1.1 strict route 的唯一 planner→action 投影是 transported `22 -> action width`，未 concat raw hidden；见同文件 `:1405-1445`。[I]
- attention mask可禁止 action直读 reference；见 `build_mot_attention_mask()` `:127-190` 与调用 `:1810-1822`。[I]
- cache会拒绝额外 planner值并校验 canonical packet receipt；见同文件 `:2296-2335`。[I]
- trainer 当前只处理 action/visual/contact grouped losses；见 `world_action_model/trainer/wa_casual_trainer_mot.py:957-1055`。[I]
- dataset已有 fail-closed、SHA-bound contact sidecar加载模式；见 `world_action_model/datasets/wam_lerobot_dataset.py:741-819`。[I]
- `minimal-r1d` 不在本 canonical 分支。已有设计文档给出的 structural contract是32-byte wire、5 hard token slots、`log2 |G_struct| = 96.99142134161319 bit`；但 executable G1/RH56/Omni task profile capacity仍是 `UNDEFINED / NO-GO`。本文把它当 future contract，不当已实现代码或低互信息证明。[S/M]

## 2. 最近前例：哪些主张已被占据

### 2.1 LUPI、teacher–student 与 asymmetric learning

| 工作 | 已覆盖内容 | 对 Idea O 的限制 |
|---|---|---|
| [Vapnik & Izmailov, LUPI](https://www.jmlr.org/papers/v16/vapnik15b.html) | 训练期 intelligent teacher 提供 inference 不可见的 privileged information；讨论 similarity correction 与直接知识迁移。[R] | “训练多模态、部署少模态”不是新意；必须证明 PI 真的迁移而非架构/优化差异。 |
| [Unifying Distillation and Privileged Information](https://arxiv.org/abs/1511.03643) | 将 knowledge distillation 与 LUPI 统一为 generalized distillation。[R] | KL/soft target、teacher logits和训练期额外信息都不是贡献。 |
| [Learning by Cheating](https://arxiv.org/abs/1912.12294) | 先训练可见 ground-truth layout/traffic 的 privileged agent，再教纯视觉 agent；部署没有 privileged input。[R] | 两阶段“cheating teacher→sensor student”已是经典模式。 |
| [Asymmetric Actor Critic](https://arxiv.org/abs/1710.06542) | critic读 simulator full state，actor只读 RGB-D；另比较 full-state auxiliary bottleneck。[R] | train-only full-state critic/auxiliary不是 novelty；本 proposal不是 actor-critic，不能混称。 |
| [Student-Informed Teacher Training](https://arxiv.org/abs/2412.09149) | 指出 privileged teacher可能生成 partial-observation student不可模仿的行为，并让 teacher考虑 student realizability。[R] | teacher“更强”不是充分条件；必须量化 student-observation下的不可实现性。 |
| [Provable POMDP RL with PI](https://arxiv.org/abs/2412.00985) | 给出 expert distillation可能无法得到近优策略的反例，并在强条件下分析有效性。[R] | 不能用 teacher accuracy替代 deployable student最优性；需要 history、belief或 abstain gate。 |
| [Rethinking Knowledge Transfer in LUPI](https://arxiv.org/abs/2408.14319) | 复查 generalized distillation/TRAM，指出很多表面收益来自低数据、欠训练或架构改变，而非 PI；random/zero PI可复现实验收益。[R] | **强制加入 zero/random/shuffled PI、等训练预算和充分收敛 baseline。** |

### 2.2 训练期 future/posterior 与 WAM

| 工作 | 已覆盖内容 | Idea O 还必须多做什么 |
|---|---|---|
| [Being-H0.7](https://arxiv.org/abs/2605.00078) | K=16 prior latent queries仅看 current context；training-only posterior以 frozen ViT + Perceiver读取 future frames；对齐末9层 hidden，部署删除posterior。[R] | 这是最高结构冲突。Idea O 必须证明 **human hand/contact trajectory target + exact finite action route + byte audit** 比同预算 future-RGB posterior更强；否则只是 Being-H0.7 的 teacher modality替换。 |
| [EgoWAM](https://gatech-rl2.github.io/egowam.github.io/EgoWAM.pdf) | 固定 policy backbone/action head/data mixture，比较 Pixel、DINO、3D flow world target。[R] | 必须固定 student/WAM并把 trajectory teacher与 generic world target配对比较。 |
| GigaWorld-Policy-0.5 canonical | joint visual/action MoT、action-only inference、v1.1 strict contact route已存在。[I] | 本工作不能把 action-centered WAM、MoT、action-only inference或 strict 22D route写成新贡献。 |

### 2.3 Human-to-robot、contact trajectory与跨 embodiment

| 工作 | 已覆盖内容 | 对 Idea O 的约束 |
|---|---|---|
| [LIDEA](https://arxiv.org/abs/2604.10677) | human→pseudo-robot与pseudo→real两阶段 frozen-teacher DINO/iBOT distillation，并做3D geometry alignment；目标是人机 shared representation。[R] | “human feature distillation解决 embodiment gap”已被直接覆盖；Idea O只能强调 process-specific teacher、finite compiler与无continuous latent action route。 |
| [EgoBridge](https://arxiv.org/abs/2509.19626) | 在 Diffusion Policy encoder上对齐 human/robot latent-action joint distribution；用DTW与OT形成 pseudo-pairs。[R] | human/robot latent alignment、DTW/OT配对不是 novelty；必须列为 baseline并审计 pair leakage。 |
| [BridgeACT](https://arxiv.org/abs/2604.23249) | human video→tool/target region→role-conditioned 3D point flow→grasp模块+闭环SE(3)控制，无 robot demo。[R] | where-to-grasp/how-to-move、tool-target affordance与 point-flow execution都不是 Idea O首创。 |
| [DexMani](https://arxiv.org/abs/2608.00554) | 从100k+人类视觉-触觉帧学习 contact-conditioned manipulability evolution energy prior；frozen prior在robot control/RL期间跨Shadow/Allegro/XHand/LEAP指导接触转换。[R] | “人类接触过程先验跨dex hand”已被占据。Idea O必须与其区分：teacher部署时完全删除；输出只通过 finite event planner；任务不局限旋转；不使用continuous manipulability guidance。 |
| [VLAff](https://arxiv.org/abs/2608.05215) | 204K ego episodes；预测视觉heatmap、grasp pose与trajectory affordance并转可执行action。[R] | human ego grasp/trajectory affordance不是 novelty；continuous action-visible upper bound必须把它作为强参考。 |
| [VideoManip](https://arxiv.org/abs/2602.09013) | monocular human RGB→hand/object/robot 3D trajectory reconstruction、contact optimization、dexterous policy。[R] | 不能声称首次从RGB hand trajectory教dex policy；dense reconstruction/retarget是对照端点。 |
| [EgoVLA](https://arxiv.org/abs/2507.12440) / [ImMimic](https://arxiv.org/abs/2509.10952) | 前者预测human wrist/hand action并IK/retarget；后者DTW映射full hand pose trajectory并与robot trajectory插值。[R] | full trajectory imitation/retarget已覆盖；Idea O不能偷偷用human action pseudo-label。 |
| [XSkill](https://arxiv.org/abs/2307.09955) / UniSkill | 无标签human/robot视频的cross-embodiment skill representation与prompt-conditioned policy。[R] | 不能把共享 latent skill/prototype本身写成新意。 |

### 2.4 轨迹到 keypose/event 压缩

| 工作 | 已覆盖内容 | Idea O 的公平比较 |
|---|---|---|
| [PPI](https://arxiv.org/abs/2504.17784) | target gripper keypose + object pointflow 与 continuous action联合；显式比较keyframe与continuous控制的优缺点。[R] | keypose作为action interface已有；需要同帧数、同token数比较 single/uniform/motion-peak/semantic keypose。 |
| [Keypose Exploration](https://arxiv.org/abs/2606.29028) | VLM找grasp/release/object interaction语义事件，trajectory analysis精确定位，再用reachability过滤做跨embodiment policy conditioning。[R] | 自动事件keypose与跨embodiment conditioning不是新意。 |
| Being-H0、VLAff、VideoManip | 分别使用motion tokens、quantized trajectory affordance与dense reconstructed trajectory。[R] | full trajectory不能与3-frame/5-token方案伪装成equal capacity；必须单列性能—带宽frontier。 |

### 2.5 R3M、VC-1、Voltron应该放在什么位置

- [R3M](https://arxiv.org/abs/2203.12601)在 Ego4D 上用 time-contrastive、video-language alignment与L1 compactness训练冻结视觉表征；它是 human-video visual backbone baseline，不是 privileged process teacher。[R]
- [VC-1/CortexBench](https://arxiv.org/abs/2303.18240)用4,000+小时 ego video和ImageNet训练MAE视觉表征；原文也强调没有一个PVR普遍占优，task adaptation很重要。[R]
- [Voltron](https://arxiv.org/abs/2302.12766)平衡 language-conditioned reconstruction 与 visually grounded language generation，覆盖五类机器人任务。[R]

公平用法是：在**同一个 student planner**中替换/初始化 frozen visual encoder，或让这些表征产生与 teacher 相同维数的 training-only target；不能让 Idea O 用更大可训练backbone、而 baseline只用小冻结encoder。若 R3M/VC-1/Voltron + `minimal-r1d` 已达到相同收益，则不能说 hand-contact trajectory是必要因素。

## 3. 精确问题定义与信息边界

### 3.1 变量

```text
x_t^R  = deployment-available robot current context
         (causal robot RGB/history, structured text, public_execution_state/proprio)
e_t    = student预测并经validator/compiler得到的canonical minimal-r1d packet
a_t:T  = robot noisy/target action chunk
z^H    = training-only human privileged stream
         (dense/multi-pose hand-object local motion, reviewed contact, visibility masks)
g      = train-split-only human↔robot event pair / shared semantic key
```

必须区分“student planner”和“action expert”：

```text
student planner:  允许读取 x_t^R，预测 e_t 与training-only distill state
action expert:    只读 canonical e_t + robot public state/proprio
                  + noisy action/timestep +预先允许的structured text
                  不读 raw robot RGB/ref token、human stream、teacher hidden、pair id
```

当前robot RGB因此是**planner input**，不是strict action expert的direct visual input。若主实验让action rows直接attend当前RGB，则仍可研究LUPI regularization，但必须改名为“privileged planner regularization”；不能再声称 finite packet 是current visual→action的唯一桥。

`minimal-r1d` 的结构容量也只是条件容量：public execution state、structured text、robot proprio、noisy action与timestep在packet外。任何表述都不能说“全部task information只有96.99 bit”或“总互信息有97-bit上界”。

### 3.2 主假设

- **H1 teacher ceiling：** reviewed full hand-contact trajectory比current-RGB-only或single-pose teacher更准确地预测 contact phase、constraint、locator与next transition。
- **H2 realizable transfer：** student在只看 `x_t^R` 时能恢复teacher增益，而非只在teacher可见、student不可判的状态上被迫平均。
- **H3 process specificity：**完整trajectory teacher胜过R3M/VC-1/Voltron generic feature target、future-RGB posterior与同packet shuffled teacher。
- **H4 finite-route preservation：** teacher只改变student weights和packet prediction；固定canonical packet/public state/text/noise后，human clip、teacher output、pair table或reference image变化不改变pre-safety action。
- **H5 OOD benefit：**增益集中在contact transition、regrasp、twist/pour/verify和held-out object/session/operator，而非仅训练loss、ID action MSE或更快收敛。
- **H6 missingness honesty：**visible/reviewed support上的teacher可用；missing/transparent不可判字段被mask/UNKNOWN/abstain，不由pose estimator或future outcome偷偷补成GT。

## 4. 推荐架构：PHCTD-on-minimal-r1d

本架构名仅作工作代号：**Privileged Hand-Contact Trajectory Distillation (PHCTD)**。[S]

### 4.1 总数据流

```text
训练期 human event clip
  raw RGB / multiview or native pose / reviewed contact / masks
                  |
          frozen trajectory teacher T_phi
                  |
       stop-gradient field logits + 5 slot states
                  |
  train-only shared key / adjudicated event pair g
                  |
                  v
robot current context x_t^R ---> student r1d planner P_theta
                                  |             |
                           KD losses only   factorized logits
                                                |
                                       hard compiler/validator
                                                |
                                    canonical 32-byte / 5-token e_t
                                                |
          robot public state + text + noise ---> strict WAM action expert
                                                |
                                            action chunk

部署：删除 T_phi、human sidecar、pair table、KD projector/output；
      仅保留 robot planner、hard compiler、canonical packet、action expert。
```

### 4.2 Teacher 输入与 object-local 表示

对每个人工或native-sensor确认的 human event，固定 `T_h=32` 个按event-normalized time采样的时刻；另做 `S1/U3/M3/S3/full`表示消融，不把32定为理论最优。[S]

每帧最多8个语义token：left hand、right hand、四个role object、left contact summary、right contact summary。所有连续量先转到**当前event focus object或controlled part的局部坐标**；若reference本身不确定，则整条local coordinate target不可用，不用估计值补齐。

每个 hand token输入：

```text
root translation xyz                         3
root rotation continuous-6D                  6
21 object-local joints xyz                  63
root translation derivative                 3
root rotation 6D first difference            6
21 joint position first differences         63
palm + 5 functional sites:
  reviewed contact state                     6
  object-local contact point xyz             18
parallel per-field observed/reviewed masks  same support, separate embedding
```

每个 object token输入 role、presence、相对SE(3)（3+6）、linear/angular velocity（6）、part/surface category与字段mask。透明液体的volume/flux/interface、不可见mesh、completed depth、future outcome都不在默认teacher输入；只有明确的sensor/native oracle arm可读，并与observed teacher分表。[S]

### 4.3 Teacher 网络

推荐一个小而可审计的 graph-temporal teacher，而不是复用5B WAM：[S]

```text
per-token MLP:     input -> 256 -> 256, GELU + LayerNorm
temporal encoder:  6 Transformer layers, d=256, 8 heads, FFN=1024
dropout:           0.1
query decoder:     5 learned semantic queries
query meaning:     [header, constraint0, locator0, constraint1, locator1]
outputs:
  z_T              [5,256] normalized teacher slot states
  l_T^field         grammar-factorized minimal-r1d field logits
  l_T^transition    next-phase/relation/effect logits (training-only diagnostic)
  u_T               per-field uncertainty / ensemble disagreement
```

五query与wire token位置对齐，不代表teacher hidden可被反序列化为动作。Teacher field head必须经过相同grammar mask与canonical validator；它不能输出自由连续trajectory作为student action condition。

Teacher先在**train split**的人类数据上训练并冻结。建议3个独立seed形成ensemble；只有adjudicated/native字段进入supervised loss。Teacher checkpoint、input normalizer、pose estimator、role binding与source split全部hash绑定。Teacher选择只看validation，不看test robot rollout。[S]

### 4.4 Student planner

student复用GWP当前reference视觉token、state encoder与structured text，但为r1d建立新的五slot planner：[S]

```text
input context: current robot visual tokens + public state/proprio + structured text
planner width: action_expert_dim = 1024（与canonical一致）
queries:       5 fixed semantic positions
planner:       existing MHA + 2-layer FFN topology的r1d版本
outputs:
  h_S          [5,1024] internal planner slots
  l_S^field    grammar-factorized r1d logits
  e_t          compiler/validator产生的canonical hard packet
  q_S          LN(MLP_1024->256->256(h_S)), training-only KD projection
```

`q_S` 与其MLP在所有 no-KD、random-KD、generic-KD和PHCTD arms中都存在，以配平deployable/training参数；导出checkpoint时统一删除。主action arm只接收 `e_t` 经r1d token encoder后的五个hard token，不接收 `h_S/q_S/l_S/l_T/z_T`。

### 4.5 为什么不让 teacher 直接预测 robot action

human hand与G1+RH56/OmniPicker不存在合法逐关节对应；teacher若直接给action会退化为EgoVLA/ImMimic/VideoManip式retargeting，并把teacher错误当动作真值。PHCTD只教：

- 当前处于什么功能事件；
- 哪个constraint/role/part/locator重要；
- 接触与效果如何变化；
- 哪些字段可观察、哪些必须unknown/abstain。

robot的具体腕、RH56与OmniPicker action仍由robot action data、current public state和strict action expert学习。

## 5. Loss、stop-gradient 与优化阶段

### 5.1 Teacher loss

对有证据字段的teacher训练目标：[S]

```text
L_T = L_struct^gold
    + 0.5 L_transition^gold
    + 0.1 L_temporal_contrast
    + 0.05 L_calibration
```

- `L_struct^gold`：factorized categorical CE + binary BCE；locator用其有限factor CE，不回归global rank；只在field mask=1时计入。
- `L_transition^gold`：next phase/relation/effect与boundary/censor目标；只作teacher quality与student auxiliary，不进action packet。
- `L_temporal_contrast`：同event相邻进程为positive，跨effect/outcome或错误顺序为hard negative；不能按同episode背景构造容易negative。
- `L_calibration`：Brier或temperature-calibrated NLL；temperature只在validation fit一次。

如果只有model-estimated pose/contact而没有review/native evidence，该样本可用于self-supervised temporal term，但不能进入`gold`项，也不能解除field mask。

### 5.2 Teacher→student distillation

冻结teacher与teacher projector：

```text
z_T = stop_gradient(T_phi(z^H))
p_T = stop_gradient(softmax(l_T / tau))
tau = 2
```

每个pair/field的权重不是teacher confidence的裸值，而是：

```text
w = reviewed_field_mask
  * adjudicated_pair_or_shared_key_mask
  * pair_confidence
  * teacher_calibration_gate
  * ensemble_agreement_gate
```

初始预注册建议：teacher field entropy `<0.5 nat`、三seed JSD `<0.1` 才通过soft-target gate；阈值可由train/calibration冻结，不得在test后调。teacher与adjudicated gold冲突时 `w=0` 并计teacher error，绝不覆盖gold。

蒸馏项：[S]

```text
L_KL   = mean_w sum_field KL(p_T^field || p_S^field)
L_slot = mean_wslot [1 - cosine(LN(q_S), LN(z_T))]
L_rel  = mean || norm(q_S q_S^T) - stopgrad(norm(z_T z_T^T)) ||_F^2

L_KD = 0.5 L_KL + 0.25 L_slot + 0.05 L_rel
```

`L_rel`只比较五个语义slot之间的关系，减弱human/robot raw coordinate的字面对齐。它不是EgoBridge OT替代；EgoBridge/DTW是pair baseline，pair solver output始终stop-gradient。

### 5.3 Student/WAM 总损失

```text
L_student = L_action_flow
          + 1.0 L_r1d_gold
          + lambda_KD(s) L_KD
          + 0.1 L_transition_student
          + existing_allowed_visual_world_loss
```

`lambda_KD(s)`建议前10% steps从0线性升至1，10%–70%保持1，最后30%线性降至0.25，让最终robot action fit不被错误teacher锁死。所有loss按active fields/token数归一化；teacher sample不能因字段更多而隐式放大batch权重。[S]

主hard-ST arm中，action loss可经ST packet surrogate回student planner；teacher始终stop-gradient。`hard_detached`配对arm保持相同forward packet，但action loss不回planner，用于区分“teacher改善packet监督”与“action end-to-end塑造planner”。

需要机械验证的gradient事实：[S]

```text
d L_KD / d theta_teacher            = 0
d action / d teacher_output          = 0
d action / d pair_table              = 0
d L_KD / d theta_student_planner     != 0 on valid pairs
d L_action / d theta_student_planner != 0 only in hard-ST arm
d L_action / d theta_student_planner  = 0 in hard-detached arm
```

### 5.4 不允许的“蒸馏”实现

- concat teacher hidden到action token；
- teacher/pair prototype保存在action-only cache；
- 用human pose生成robot action pseudo-label；
- 让action读取soft field logits并同时声称finite route；
- 在val/test human clip上运行teacher，再把结果送给student；
- 把teacher proposal、MANO fit或depth completion接受为gold；
- 用teacher confidence决定原本missing字段的loss mask；
- 同时改student backbone大小、数据量或训练步数后把收益归因于PI。

## 6. 五阶段实验顺序

### Stage 0：contract与数据审计，0 GPU·h

1. 先完成r1d codec/validator/public-state/profile/五token route；canonical `e0ebc692` 目前不满足。
2. split先按source episode/operator/session/object/batch/calibration建立；pose keys、crop、teacher cache、pair graph的connected component继承同split。
3. 对human pose/contact来源分级：`HUMAN_REVIEWED_RGB`、`MULTIVIEW_RECON_WITH_GT_CHECK`、`TACTILE/SENSOR_CONFIRMED`、`SIM_NATIVE`、`MODEL_PROPOSAL_ONLY`。
4. 冻结teacher input contract、normalizer、event window和missingness规则。

失败即stop：r1d executable profile仍undefined；pair/source group无法恢复；关键字段无review/native support；或transparent关键状态只能靠completion/pseudo-label。

### Stage 1：teacher ceiling，不接robot policy

在episode-heldout或public human benchmark上比较：

```text
T-RGB-current
T-single-contact-pose
T-U3-uniform
T-M3-motion-peak
T-S3-semantic(entry/turn/exit)
T-local-window
T-full-event-32
T-full-episode
```

所有非full arms用相同encoder/output width/训练步数；full arms单列实际frames/tokens/FLOPs，作为上界而非equal-capacity对照。

Gate：`T-full-event`在预注册核心field macro-F1相对最佳current/single至少`+10 pp`且cluster 95% CI支持正优势；`S3`相对single至少`+5 pp`、相对full差距`<=5 pp`。否则删除“完整trajectory有额外知识”或“semantic keypose近似充分”主张。

### Stage 2：oracle packet sufficiency，关闭teacher

在target robot train数据上，只比较predicted/no-teacher与adjudicated/native causal oracle r1d packet。Oracle不得用human视频、future robot frame或post-hoc outcome。

Gate：oracle r1d对matched direct/equal-bit baseline没有明确contact-phase/OOD收益时，停止Idea O；teacher无法修复一个对action本就不充分的packet。

### Stage 3：小student / frozen backbone蒸馏

先冻结GWP视觉backbone或使用同一个小planner，比较no-KD、generic-KD、future-posterior、shuffled-KD与PHCTD。只评packet、transition、retrieval与action-offline机制指标，不把action MSE写成robot success。

Gate：PHCTD必须恢复至少50%的teacher ceiling packet增益，并相对no-KD、generic-KD和shuffled-KD各有正的group-cluster CI；否则不接完整WAM。

### Stage 4：strict WAM核心矩阵

所有arm固定student checkpoint初始化、robot/human budget、batch、optimizer、augment、action noise/timestep、训练步数、planner/action参数与r1d capacity。3 seeds只在Stage 3 finalist上运行，不对所有失败arm浪费计算。

Gate：PHCTD相对最强matched baseline在预注册OOD success或关键phase success至少`+5 pp`，95% group-cluster CI不跨0；同时恢复`>=70%` oracle-r1d gain，ID/safety不显著下降，action-only latency相对r1d-only增加`<=2%`（理论上teacher已删除）。

### Stage 5：跨task/embodiment与真机

只有Stage 4通过且目标hardware driver/calibration/watchdog/outcome sensor全部admitted才进行。必须包含teacher未见task family、held-out human operator、held-out target object/session，以及至少一个contact-rich rotation/twist任务对照DexMani邻域。

## 7. Baseline与ablation矩阵

### 7.1 必须baseline

| ID | 方法 | action-visible输入 | 目的 |
|---|---|---|---|
| B0 | direct GWP/WAM | direct ref + state/text | 原始高带宽路线；不能与strict claim混写 |
| B1 | r1d-only | predicted hard r1d + public robot state | 主no-PI baseline |
| B2 | r1d + extra train steps | 同B1 | 排除KD只是额外优化时间 |
| B3 | r1d + parameter-matched aux head | 同B1 | 排除额外head/regularization |
| B4 | r1d + random/zero PI | 同B1 | 响应LUPI反例；排除架构收益 |
| B5 | r1d + within-class shuffled teacher | 同B1 | 保持packet marginal，破坏过程对应 |
| B6 | R3M/VC-1/Voltron target或初始化 | 同B1 | generic human-video representation |
| B7 | Being-H0.7-style future-RGB posterior | 同B1 | 最接近train-only posterior baseline；相同5 slots/256D target/steps |
| B8 | LIDEA-style frozen feature distill | 同B1 | cross-embodiment feature KD baseline |
| B9 | EgoBridge-style joint OT/DTW | 同B1 | latent/action domain alignment baseline；同pair universe |
| B10 | PHCTD semantic KL only | 同B1 | teacher只教finite fields |
| B11 | PHCTD slot only | 同B1 | teacher只教internal planner representation |
| B12 | PHCTD full | 同B1 | 主方法 |
| B13 | oracle causal r1d | gold/native hard r1d + state | packet/action upper bound，独立checkpoint |
| B14 | action-visible continuous pose/flow | pose/flow + state | BridgeACT/VLAff/VideoManip-style性能上界；明确高带宽、非strict |
| B15 | DexMani-style guidance（rotation子集） | continuous manipulability/current sensing | 强task-specific challenger；不伪装通用baseline |

若B14赢，结论是高带宽geometry/trajectory更适合该任务，不得把其收益归给finite distillation。若B7与B12相当，结论只支持generic train-only posterior，不支持hand-contact process specificity。

### 7.2 Teacher ablations

1. RGB only；
2. hand pose only；
3. contact only；
4. pose + contact；
5. pose + contact + object local motion；
6. full event vs full episode；
7. observed-only vs native/tactile oracle；
8. time reversed；
9. contact labels shuffled but pose preserved；
10. pose shuffled within samephase/object；
11. teacher width/layers `{128/4, 256/6, 512/8}`；
12. teacher frozen vs student-informed co-training（只作诊断，不能与主stop-gradient arm混表）。

### 7.3 Student/route ablations

1. KL / slot / relational各自与组合；
2. KD weight与late decay；
3. hard-ST vs hard-detached；
4. five semantic slots vs相同数量unstructured slots；
5. r1d semantic vs exact-capacity unstructured code；
6. action strict-no-ref vs direct-ref；
7. current single frame vs deployment-available causal history；
8. pair gold vs DTW/OT vs unpaired prototype；
9. packet clamp后swap teacher/ref；
10. packet field intervention与delay/reorder。

### 7.4 Capacity与compute fairness receipt

每个核心arm必须记录：

```text
student deployable parameter count
student train-only head parameter count
active inference FLOPs / action latency
training steps / tokens / optimizer updates
robot episodes / human clips / valid fields
teacher pretraining and feature-extraction GPU-hours separately
packet schema/hash/wire bytes/token slots/structural capacity
public-state/text/action schema hashes
initial checkpoint and code commit
```

要求B1–B12的deployable student参数、action-visible packet、推理FLOPs与robot data完全相同；train-only projection head也相同。另同时报告equal-step与equal-total-training-compute结果。Teacher预训练成本不能摊掉后不报。

## 8. 信息泄漏与旁路审计

### 8.1 Split leakage

先split、再进行以下全部操作：pose/contact extraction、normalizer、keypose选择、teacher训练、teacher cache、OT/DTW、prototype、hard-negative生成、augmentation。以下实体必须同split：

- raw human clip与全部帧/crop/depth/pose/teacher features；
- raw robot episode与全部window/action chunks；
- 一个human↔多个robot pair graph的connected component；
- 同operator/session/object instance/calibration/capture batch；
- 同一physical execution的multiview streams。

test robot action、test outcome、test human clip、test pair decision绝不用于teacher、threshold、temperature、normalizer或pair solver。只按frame随机split直接判无效。

### 8.2 Action-path leakage

必须生成可机读route receipt并通过：

1. action rows不attend raw reference tokens；current robot RGB只影响planner；
2. `encode_action_tokens`只接受canonical r1d bytes/tokens，不接受planner/teacher hidden；
3. action-only cache只保存hard packet、public-state/text/route receipts和必要K/V；
4. inference API没有human clip、teacher sidecar、pair id、teacher logits参数；
5. 导出checkpoint删除teacher与KD projection，不改变同input action bytes；
6. 固定packet/public state/text/noise后替换reference、teacher、pair table，pre-safety action `max_abs <= 1e-6`；
7. 对teacher operator/clip/pair id做linear probe，在固定packet的action hidden上不得超过chance+2 pp；
8. 同checkpoint修改teacher sidecar只可改变loss，不可改变forward action。

### 8.3 Label leakage

- teacher可看human event的完整轨迹，但不能看对应robot target action；
- human↔robot pairing主cost不得包含robot action target、success-after-window或test outcome；若EgoBridge-style action-aware OT作为baseline，必须明确它是train-only且与主方法分开；
- future human effect可作为PI，但必须标注为posterior；不能在test运行teacher；
- oracle r1d必须来自当前因果robot sensor/native state，不读future frame；
- teacher estimated contact/pose不是gold，也不能决定gold mask。

## 9. Teacher不足、错误与不可实现性

### 9.1 三个teacher gate

1. **准确性：** 在locked human audit上，核心field macro-F1 `>=0.80`、ECE `<=0.08`；任一task/opacity/missingness subgroup相对overall下降超过15 pp需单报且不得默认用于KD。
2. **增量性：** full trajectory相对current/single至少`+10 pp`；否则PI没有被证明有额外信息。
3. **可实现性：** 对近似相同student context却有不同teacher target的组，报告conditional disagreement/entropy。若超过20%的核心样本存在不可由deployment history区分的冲突，先增加**deployment可用**causal history或输出UNKNOWN/abstain；不能加human旁路。

### 9.2 Teacher错时如何处理

- gold优先，teacher冲突field权重为0；
- 三seed ensemble disagreement高则mask；
- 保留失败/partial/unknown，不只从成功片段训练teacher；
- teacher不能以高confidence自动扩大reviewed support；
- 独立报告teacher error transmission：teacher错、student也错的比例；
- 做negative-transfer表：no-KD胜PHCTD的task/phase/subgroup；
- teacher质量不足时仅用作proposal，不再称distillation target。

### 9.3 Student追不上teacher时的解释

这不自动说明student容量不够。可能是：

- PI包含当前observation不可识别的future或遮挡状态；
- human/robot event pair错误；
- teacher表征保留human anatomy/identity而非功能过程；
- finite packet本身不充分；
- no-KD baseline欠训练或PHCTD只是加速早期收敛；
- teacher错误比student视觉噪声更大。

必须通过history、pair gold、packet oracle、充分训练、identity probe和capacity-matched controls逐项区分；不能直接扩大student/teacher后宣称成功。

## 10. Missingness、透明场景与不可见事实

### 10.1 三类teacher必须分表

```text
T_observed:
  只读raw RGB可见且reviewed的pose/contact/local motion；missing field带mask

T_sensor/native oracle:
  可读multiview/tactile/native pose/contact/effect；只作ceiling

T_imputed/proposal:
  MANO fit、tracker、depth completion、mesh/NeRF、VLM/CAM等估计
  只能作proposal或sensitivity baseline，不能作gold/主teacher
```

`T_sensor/native`赢而`T_observed`不赢，只能说明理想PI有价值，不能说明现有ego数据可用。`T_imputed`赢不能升级其输出为真值。

### 10.2 字段规则

- `ABSENT`：EXHAUSTIVE review确认不存在，才监督negative；
- `REVIEWED_UNKNOWN`：存在但字段不可判；有schema UNKNOWN code时监督UNKNOWN；
- `MISSING/NOT_REVIEWED`：mask=0，不进wire，不蒸馏；
- actor/primary reference等无UNKNOWN code而无法解析：整packet不可作oracle/KD pair，部署hold/reobserve；
- left/right censor是coverage边界，不等于occlusion；
- 看不见stream/meniscus且无秤/液位/native truth时，FLOW/TRANSFER/outcome不得从wrist tilt推导；
- 透明杯重叠不推出INSIDE/TOUCHING/ALIGNED_TO_RIM；
- teacher confidence不能把MISSING转UNKNOWN或KNOWN。

### 10.3 缺一只手

确认有hand edge但laterality不可判时，presence与其它known fields可训练teacher/planner，actor mask=0；整个oracle/action packet inadmissible。不能把“未检测到第二只手”编成single-hand complete event，也不能让teacher pose填补后解除completeness gate。

### 10.4 透明任务的可证伪预测

- 若observed teacher在clear-liquid subgroup无增益而sensor/native teacher有显著增益，问题是observability，不是student distillation；应停止透明主claim。
- 若teacher只提升phase、不能提升independent scale/level outcome或spill，则只能称event anticipation，不称液体物理迁移。
- 若UNKNOWN/abstain增多但unsafe action下降，报告coverage–risk frontier；不能只报success并惩罚诚实abstain。

## 11. 指标与可证伪预测

### 11.1 Teacher/planner指标

- factorized r1d field macro-F1、balanced accuracy、NLL、Brier、ECE；
- phase/contact/relation/effect/outcome分项；
- event boundary tolerance F1与censor-aware likelihood；
- human→robot event retrieval Recall@1/mAP、NULL AUPRC；
- teacher–student slot CKA/cosine仅作诊断，不能替代任务指标；
- identity/operator/session probe；
- missingness/opacity/pose-source subgroup。

### 11.2 Action/rollout指标

- correct-first-contact；
- wrong-first-contact、drop/spill/collision、unsafe advance、abstain/reobserve；
- task success与phase-conditional success；
- regrasp/recovery、twist turns、pour transfer/verify；
- action smoothness、wrist/grip error只作离线机制指标；
- group-cluster bootstrap 95% CI、3 seeds、seed×cell散点；
- action-only latency、KV/cache memory与deployable参数。

### 11.3 预注册预测

1. PHCTD收益应在contact-changing windows大于free-motion windows；若全程均匀提升，优先怀疑generic regularization。
2. pose+contact应胜pose-only/contact-only；若contact shuffled不降，teacher没有教contact process。
3. time reversal应显著破坏transition/regrasp/twist表现；若不降，模型只用了static pose。
4. within-packet shuffled teacher应接近no-KD；若仍提升，收益更可能来自loss/architecture。
5. B12应胜B7才支持human hand-contact specificity；与B7相同只支持future posterior。
6. B12应胜B6才支持structured process而非generic PVR。
7. hard packet intervention应只在对应hand/phase/time改变action；无局部效应则停止causal-consumption claim。
8. student最后性能而非早期收敛必须改善；extra-step B2追平时停止sample-efficiency claim。
9. teacher删除后的checkpoint输出必须bitwise/`1e-6`等价；否则存在部署旁路。
10. 若continuous B14显著赢且strict B12无法恢复`>=70%` oracle gain，承认finite packet不足。

## 12. 数据与计算估算

### 12.1 当前数据能回答什么

- 现有66段human ego pour约20,013帧：可做human-only teacher input/missingness/keypose feasibility与episode-heldout diagnostic；缺可靠subject/session/object/liquid/batch provenance，不能做可信OOD主张。[M]
- legacy 88段robot grab、40,901帧、dual-Inspire action：任务与目标hardware都不匹配；只能做loader/ontology/hard-negative，不是human pour的positive pair，也不能验证RH56+OmniPicker action。[M]
- 当前没有目标G1+left RH56DFTP+right OmniPicker的同任务robot policy数据、完整hardware receipts或executable r1d profile；因此本轮真实action benefit为No-Go。[M]

### 12.2 建议的最小D2 feasibility

沿用三task family：`pick_place`、`stabilize_twist_cap`、`pour_transfer`。

```text
3 families * 2 object conditions * 2 session/day cells = 12 cells
每cell: 3 human + 3 target-robot
总计: 36 human + 36 target-robot
split: 每family 2 train / 1 validation / 1 test cell
```

72 clips全部标event/r1d/pose-source/missingness/outcome；至少15 clips双标；至少180条match/NULL/hard-negative pair decisions。预计50–76人时、20–60 CPU·h完成capture metadata、gold、pair audit与经典alignment；不含driver/calibration/watchdog开发。36+36只够feasibility，不够主论文WAM结论。[S]

公开proxy可用于teacher或alignment预检：H2O、HOI4D、HOT3D/ARCTIC做human pose/trajectory；RH20T/MIME/H&R做corresponding/synchronized human-robot proxy；EgoMimic/XSkill做unpaired。同源许可、视角、接触GT与目标embodiment差异必须单列，不能替代目标D2。

### 12.3 Full study规划量级

在D2方差与effect size出来前不伪造精确power。建议量级：[S]

| 阶段 | 规划计算 | 说明 |
|---|---:|---|
| codec/split/hash/classic alignment | 20–60 CPU·h | 0 GPU可完成 |
| 小teacher 72-clip pilot，3 seeds | 3–12 H20 GPU·h | 前提是pose/视觉feature已离线；只作feasibility |
| public human teacher预训练 finalist | 100–400 H20 GPU·h | 取决于数据规模与是否在线视觉encoder |
| frozen-backbone student probe核心arms | 30–120 H20 GPU·h | 先淘汰random/generic/futureposterior失败arm |
| target-robot WAM 3-seed finalist matrix | 600–2,000 H20 GPU·h | 来自同项目roadmap量级；必须先短profile |
| rollout/eval | 由cell-cluster power另计 | 不能用frame数当n |

每100 episodes的数据QA/新标注约15–30人时，另加安全监护与第三人裁决。若要独立object/session/operator OOD，至少增加第三组group；实际可能需要150–300 human clips与300–600 target-robot episodes，但最终数必须由D2 cluster variance/power决定，而不是把frame/window当独立样本。

当前授权与实际消耗：**0 GPU·h、0 server、0 training**。

## 13. canonical WAM插入设计

以下仅是future设计，不是本分支代码修改。[S]

### 13.1 版本边界

不得静默改 `contact_transition_schema.py` 的v1.1 Q12×22D路径。应新增独立、versioned r1d codec/validator/token encoder与route receipt；只有r1d golden vector、rank/unrank、profile/public-state和cache audit通过后，PHCTD才能依赖它。

### 13.2 具体插入面

| canonical位置 | future改动 | 不变量 |
|---|---|---|
| `transformer_wa_casual_mot.py:598-631,703-816` | opt-in r1d schema + PHCTD training flags；teacher配置不写入deploy inference API | legacy v1.1默认行为不变 |
| `:817-853` | r1d五query planner、factor heads、training-only `1024->256` projection | no-KD与KD都有同projection参数 |
| `:1253-1285` | r1d planner可在`return_training_aux=True`时返回`h_S/q_S`给trainer | raw hidden仍不进入action/cache；inference默认不返回 |
| `:1405-1445` | 以r1d五hard token encoder替换22D projection | action只接canonical packet；禁止concat logits/teacher hidden |
| `:1553-1589` | joint encode返回packet logits/bytes与可选training aux | action construction只消费hard packet |
| `:1810-1822` | strict mask禁止action ref attention | current RGB仅供planner |
| `:2296-2335` | cache allowlist加入r1d packet/public-state receipt | teacher/KD/pair字段出现即fail closed |
| `wam_lerobot_dataset.py:741-819` | 参照SHA sidecar模式加载train-only teacher cache与pair receipt | val/test/deploy缺sidecar仍能运行；strict join不跨split |
| `wa_casual_trainer_mot.py:957-1055` | 加`L_KD`、mask与gradient audit输出 | teacher output stop-gradient；action forward不读sidecar |

### 13.3 推荐离线teacher sidecar

为减少在线teacher compute与旁路风险，teacher冻结后生成train-only sidecar：

```text
schema/version
source_human_clip_sha256
teacher_checkpoint_sha256
teacher_input_contract_sha256
split/group ids
paired_robot_episode/window ids
shared_key / adjudicated pair id
field logits [factorized]
slot states [5,256]
field masks / calibration / ensemble disagreement
pair confidence / provenance
```

sidecar不能包含robot action target、test pair、free-form object/session identity embedding或continuous action suggestion。Trainer join必须检查双方split/group/hash；sidecar不存在时B1/B2正常运行，不能把teacher变成部署依赖。

## 14. Novelty、论文位置与stop条件

### 14.1 绝对不能写的first

1. 首次learning using privileged information；
2. 首次teacher–student distillation或training-only full-state/future branch；
3. 首次用human videos预训练机器人视觉表示；
4. 首次human-to-robot feature distillation/domain adaptation；
5. 首次从human hand trajectory学robot action；
6. 首次用contact trajectory/manipulability指导dexterous control；
7. 首次用keypose/trajectory token/point flow连接感知与控制；
8. 首次部署时删除teacher/future branch；
9. 首次用finite/discrete contact token；
10. 首次从ego视频做pour/工具操作。

### 14.2 可守的最窄贡献

若全部gate通过，可写：

> 我们研究了一种受审计的LUPI机制：训练期object-local human hand-contact trajectory teacher只监督robot current-observation planner；部署action只消费版本化、精确容量、可反序列化的minimal-r1d packet与robot public state。通过同student/同packet/同robot data/同compute的generic posterior、cross-embodiment alignment、random/shuffled PI和continuous trajectory controls，以及byte-level route/gradient/cache tests，我们检验过程PI是否改善contact-transition OOD execution，而不建立human/pose部署旁路。

这更适合作为SM-FET/finite-WAM的**辅助方法贡献**。要成为独立主论文，至少还需：

- 明确胜Being-H0.7-style future posterior；
- 明确胜LIDEA/EgoBridge-style feature alignment；
- 在contact-rich多个task family上胜generic PVR；
- 保持strict finite route并通过intervention；
- 对DexMani邻域给出非旋转、无online continuous prior的新增证据。

### 14.3 硬stop条件

任一满足即停止相应claim：

1. r1d executable profile、codec或route audit未冻结：停止实现/训练；
2. teacher full trajectory不比current/single高10 pp：停止“trajectory PI”claim；
3. S3不比single高5 pp或离full超过5 pp：停止“keypose compression充分”claim；
4. oracle r1d不优于matched direct/equal-bit：停止PHCTD WAM；
5. PHCTD不胜no-KD、extra-steps、zero/random/shuffled PI：停止PI transfer claim；
6. PHCTD不胜R3M/VC-1/Voltron：停止process-specific claim；
7. PHCTD不胜Being-H0.7-style posterior：停止新方法claim，最多作为teacher modality ablation；
8. PHCTD不胜LIDEA/EgoBridge-style alignment：停止cross-embodiment distillation claim；
9. student无法恢复50% teacher packet ceiling或70% oracle action gain：停止扩大WAM；
10. 只有teacher/native transparent oracle有益，observed teacher无益：停止真实ego transparent claim；
11. teacher hidden/pose/flow必须action-visible才有效：strict Idea O失败，转高带宽baseline；
12. route/cache/gradient/split任一泄漏：所有causal/deployment结论无效；
13. final improvement只在早期收敛，充分训练B2追平：停止sample-efficiency/final performance claim；
14. OOD gain <5 pp或CI跨0，或以ID/safety下降换取：停止主结果；
15. packet intervention无phase/hand局部效应：停止“action消费过程知识”claim；
16. 目标G1/RH56/OmniPicker数据、driver、calibration、watchdog/outcome truth缺失：停止真机与hardware claim。

## 15. 最终建议

**建议保留Idea O，但把它排在r1d structural/observability/oracle gate之后，并把Being-H0.7、LIDEA、EgoBridge、DexMani设为四个必须击败的最近邻。** 最小的下一步不是训练5B WAM，而是：

1. 证明full hand-contact trajectory在locked human数据上确有current/single看不到的、且可映射到r1d field的增量；
2. 证明r1d oracle对target robot action有用；
3. 用同一个小student做zero/random/shuffled/generic/future-posterior/PHCTD probe；
4. 只有三项都通过才接full WAM。

这个顺序能区分四种完全不同的失败：teacher没知识、student看不出来、packet不够、或蒸馏只是优化正则。若没有这些反事实，Idea O最容易变成“把已有human pose teacher接到已有finite planner”的组合创新，novelty与因果证据都不够。

## 参考原始来源

- Vapnik, V. & Izmailov, R. [Learning Using Privileged Information: Similarity Control and Knowledge Transfer](https://www.jmlr.org/papers/v16/vapnik15b.html), JMLR 2015.
- Lopez-Paz et al. [Unifying Distillation and Privileged Information](https://arxiv.org/abs/1511.03643), 2016.
- Chen et al. [Learning by Cheating](https://arxiv.org/abs/1912.12294), CoRL 2019.
- Pinto et al. [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542), RSS 2018.
- Messikommer et al. [Student-Informed Teacher Training](https://arxiv.org/abs/2412.09149), 2024.
- Cai et al. [Provable Partially Observable Reinforcement Learning with Privileged Information](https://arxiv.org/abs/2412.00985), 2024.
- Provodin et al. [Rethinking Knowledge Transfer in Learning Using Privileged Information](https://arxiv.org/abs/2408.14319), 2024.
- Luo et al. [Being-H0.7: A Latent World-Action Model from Egocentric Videos](https://arxiv.org/abs/2605.00078), 2026.
- Xu et al. [LIDEA](https://arxiv.org/abs/2604.10677), 2026.
- Punamiya et al. [EgoBridge](https://arxiv.org/abs/2509.19626), 2025.
- Han et al. [BridgeACT](https://arxiv.org/abs/2604.23249), 2026.
- Chen et al. [DexMani](https://arxiv.org/abs/2608.00554), 2026.
- Oh et al. [VLAff](https://arxiv.org/abs/2608.05215), IROS 2026.
- Chen et al. [VideoManip](https://arxiv.org/abs/2602.09013), 2026.
- Yang et al. [EgoVLA](https://arxiv.org/abs/2507.12440), 2025.
- Liu et al. [ImMimic](https://arxiv.org/abs/2509.10952), CoRL 2025.
- Xu et al. [XSkill](https://arxiv.org/abs/2307.09955), 2023.
- Yang et al. [Gripper Keypose and Object Pointflow as Interfaces for Bimanual Robotic Manipulation](https://arxiv.org/abs/2504.17784), RSS 2025.
- Lu et al. [Keypose Exploration](https://arxiv.org/abs/2606.29028), IROS 2026.
- Nair et al. [R3M](https://arxiv.org/abs/2203.12601), CoRL 2022.
- Majumdar et al. [VC-1 / CortexBench](https://arxiv.org/abs/2303.18240), 2023.
- Karamcheti et al. [Voltron](https://arxiv.org/abs/2302.12766), RSS 2023.
