# 生化实验 Contact-Transition WAM：投稿级研究 dossier

> 截止日期：2026-08-11
> 实现证据基线：canonical commit `e62d80d`
> 目标系统：Unitree G1，左手 Inspire RH56DFTP（6 个手部控制通道），右手 AgiBot OmniPicker（1 个主动开合通道）
> 目标基础模型：[GigaWorld-Policy-0.5](https://arxiv.org/abs/2607.13960)
> 结论性质：研究设计与证据审计，不是训练结果，也不是部署许可。

## 0. 一句话决定

主线应当是 **Auditable Finite Contact-Transition WAM（有限、可审计的接触转移 WAM）**：让动作分支在严格模式下看不到 reference image 和 planner hidden，只能读取由当前观测预测的、具有精确有限状态数的接触段语法；用 Ego 视频学习“谁与谁、在哪里、以何种模式、何时接触/分离”，再在异构双手机器人数据上学习具体动作。

它的可防守贡献不是“第一个 contact token / affordance / 离散 latent / Ego-to-robot 方法”，而是以下组合：

1. 显式 contact-transition segment grammar；
2. action-visible channel 的精确有限容量上界；
3. 视频侧预测、action-only 推理的 WAM 因果路由；
4. 对缺失双手真值、删失边界和 unknown/negative 的审计语义；
5. matched-control hash 证明不同变体读到了同一批样本、增强、时间步和噪声；
6. 人类 20D 与目标机器人 canonical25D 通过同一语义计划衔接，而不伪装成同一动作空间。

当前结论仍是 **Conditional Go for research，No-Go for scientific training / physical deployment**。原因不是代码接口缺失，而是正式 A/B 标注和裁决尚未完成、OOD 分组元数据缺失、当前真机数据不是目标右手硬件、OmniPicker 设备映射/驱动/安全链未闭环。

## 1. 候选 idea 排名

| 排名 | 候选 | 定位 | 当前决定 |
|---:|---|---|---|
| 1 | A. Auditable Finite Contact-Transition WAM | 主方法与主论文叙事 | **主 idea，Conditional Go** |
| 2 | B. Censored Bimanual Contact Event Benchmark | 主线必需的数据/评测贡献 | **与 A 合并，不建议单独宣称方法首创** |
| 3 | C. Visibility-Gated Spatial Auxiliary | boundary/rim 空间辅助分支 | **Auxiliary，先过可见性与因果增益门槛** |
| 4 | D. Morphology-Aware Asymmetric Executor | 接触语义到 RH56+OmniPicker 的具体化 | **工程上必要，独立 novelty 较弱** |
| 5 | E. Transparent-Liquid Reconstruction / Flow World Model | 透明容器、液面、液流的重建或 dense field | **当前主线 No-Go；只保留风险型 auxiliary** |

推荐论文包不是五篇松散想法，而是：**A 为核心方法，B 为可信数据与 benchmark，C 为可选增强，D 为真实硬件验证边界；E 只进入 limitation/future work，除非补采改变可观测性结论。**

## 2. 当前仓库到底证明了什么

### 2.1 已实现并经过 CPU/静态接口验证的部分

- GWP-0.5 的严格 attention 路由和 contact planner：
  - `world_action_model/models/transformer_wa_casual_mot.py`：`build_mot_attention_mask`（约 110 行）、contact module 初始化（约 605 行）、`transport_contact_semantic`（约 1009 行）、`plan_contact_semantic`（约 1093 行）、forward 规划/动作注入（约 1322 行）、action/reference 可见性开关（约 1578、2006 行）及 action-only cache（约 2069 行）。
- 22D segment grammar 与精确容量：
  - `world_action_model/contact_transition_schema.py`：`contact_transport_capacity_metadata`（约 114 行）。
  - `soft`、`hard_st`、`hard_detached` 已实现；B=16、Q=12 时每段 48,673 个合法状态，上界 15.570834 bit/segment，12 段共 186.850009 bit。
- Ego Stage 1 的 fail-closed 入口：
  - `configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py`；固定 20 Hz、H=48、Q=12、D=22、B=16、hard-ST、predicted condition、strict bottleneck。
  - `world_action_model/contact_ego_stage1_preflight.py`、`world_action_model/ego_dataset_fingerprint.py`、`scripts/compute_ego_wam_norm.py`。
- Robot Stage 2 的 canonical25D 入口：
  - `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py`。
  - `world_action_model/contact_asym25_preflight.py`、`scripts/convert_sonic_wrist_delta.py`、`scripts/compute_wam_task_norm.py`。
  - policy action 是 `left wrist 9 + right wrist 9 + left RH56 6 + right OmniPicker aperture 1 = 25D`，再 pad 到 32D。WBC 的 29 个关节属于硬件/FK/执行契约，**不是再拼到 policy action 后形成 54D**。
- 缺失真值语义和 sidecar 构建：
  - `track_a/contact_labels_v1_1/`；`unknown / positive_only / exhaustive` coverage、字段级 mask、left/right censor、物理 segment ID 与 window query ID 分离。
  - `scripts/contact_labels/build_contact_segment_sidecars_v1_1.py` 与 validator/dual-annotation packet。
- matched controls：
  - `world_action_model/matched_control_hashes.py`、`world_action_model/matched_control_config.py`、`scripts/compare_matched_control_hashes.py`。
  - 已实现 runner 仅为 `soft_semantic`、`hard_st`、`hard_detached`；逐步核对 dataset order、raw batch、augmentation、visual/action timesteps、visual noise、action noise 六类 hash。

### 2.2 还没有形成科学证据的部分

- `track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/` 仍是空白 A/B packet；episode 0 的一个 timing hint 不是 ground truth。
- 66 段 Ego 数据缺少 subject/session/operator/source object/target object/liquid/batch 的真实分组信息；当前 episode-level split 明确 `strategy_claim_allowed=false`。
- `track_b/visibility_audit_v0/` 是单人、100 帧、phase-stratified prescreen，不是 IID benchmark，也不是人际一致性结果。
- `outputs/grab_red_bottle` 是 88 段 `29 body + left Inspire6 + right Inspire6 = 41D` 的旧硬件数据；不能把右侧六指通道压缩或改名为 OmniPicker aperture。
- 仓库没有目标 OmniPicker driver、canonical-to-device 反向标定映射、watchdog 和真实执行器闭环，因此 action output 只能称 canonical25D prediction。
- `configs/contact_transport_ablation_matrix.json` 中 Q12 direct capacity、aux-only、raw planner hidden、shuffle、delay 仍为 `spec_only` 或 blocked。只有三个 transport runner 有 matched-control collector。
- 该 matrix 的内部 `canonical_commit` 字段仍记录其最初集成基线 `913811d`，而本 dossier 审计的是 `e62d80d`；正式实验前必须由集成审阅者更新并重新绑定 matrix/config/collector 的完整 SHA，不能用旧字段替新 commit 背书。
- 所有 server / 5B training gate 仍为 false；`e62d80d` 没有新的 GPU 训练结果。

## 3. 新颖性碰撞地图

### 3.1 直接碰撞，不能再声称的内容

| 工作 | 已覆盖内容 | 对本项目 claim 的限制 |
|---|---|---|
| [VLAff](https://arxiv.org/abs/2608.05215)（IROS 2026） | 从 204K Ego episodes 提取 visual/grasp/trajectory affordance；使用 `<SEG>`、`<GRASP>` 和离散 trajectory tokens；显式回答 where/how/move | 不能声称“首次用 token 告诉机器人抓哪里”“首次从 Ego 学接触区/抓姿/轨迹” |
| [ST-VLA](https://arxiv.org/abs/2603.13788) | 2D guidance → 3D trajectory，smooth spatial masks 表示 4D context；ST-Human 300K episodes；RLBench/real robot | 不能声称“首次用时空 mask/3D-4D 中间表示连接 VLM 与 policy” |
| [CoA-VLA](https://arxiv.org/abs/2412.20451)（ICCV 2025） | object/grasp/spatial/movement 四类 sequential affordance；visual-text co-injection；LIBERO 与真实任务 | 不能声称“首次 chain-of-affordance / 动态选择 affordance / visual token 引导动作” |
| [BridgeACT](https://arxiv.org/abs/2604.23249) | human video → tool-target role + grasp region + 3D point flow → closed-loop SE(3)；真实任务含 pour water | 不能声称“首次用 tool-target 中间表示跨 embodiment”“首次不靠 robot demo 从人视频做倒液” |
| [2HandedAfforder](https://arxiv.org/abs/2503.09320)（ICCV 2025） | 从 human video 学精细、可执行的 bimanual affordance region | 不能声称“首次从视频学习双手可操作区域” |
| [Contact-Anchored Policies](https://arxiv.org/abs/2602.09017) | 用空间接触点取代语言条件，构建跨环境/embodiment utility policies | 不能声称“contact conditioning 本身新颖” |
| [GeCCo](https://arxiv.org/abs/2509.17582)、[contact-conditioned locomotion](https://arxiv.org/abs/2408.00776) | 接触位置、时间、持续长度作为低层 policy 条件 | 不能声称“把接触计划传给控制器首次出现” |
| [OmniContact](https://arxiv.org/abs/2606.26201) | key body trajectories + binary contact time series 的 contact flow，支持 humanoid meta-skill chaining | 不能声称“contact flow / binary contact dynamics 用于 humanoid 是首次” |
| [DexMani](https://arxiv.org/abs/2608.00554) | 用人类 contact transition 导出的 manipulability evolution 跨手形态迁移旋转技能 | 不能宽泛声称“首次靠人类接触变化跨 embodiment 学灵巧操作” |
| [Mask World Model](https://arxiv.org/abs/2604.19683) | 在 world model 中预测 semantic mask 而非 RGB，形成几何 bottleneck | 不能声称“首次在 WAM/world model 中用语义场替代像素” |
| [VQ-VAE](https://arxiv.org/abs/1711.00937)、[FSQ](https://arxiv.org/abs/2309.15505)、[VQ-BeT](https://arxiv.org/abs/2403.03181)、[FAST](https://arxiv.org/abs/2501.09747) | 离散 latent、固定 scalar levels、分层动作量化、动作 token rate/compression | 不能声称“离散 latent / ST estimator / 有限 codebook / 动作 tokenization 首次出现” |
| [EgoVLA](https://arxiv.org/abs/2507.12440)、[EgoScale](https://arxiv.org/abs/2602.16710)、[EgoHumanoid](https://arxiv.org/abs/2602.10106)、[ZeroWBC](https://arxiv.org/abs/2603.09170) | Ego human action pretraining、统一/对齐动作空间、retargeting、humanoid transfer | 不能声称“首次用 Ego 数据教 humanoid 操作” |
| [Being-H0.7](https://arxiv.org/abs/2605.00078)、[EGOWAM](https://gatech-rl2.github.io/egowam.github.io/EgoWAM.pdf) | future-informed latent queries；Ego WAM 的非像素表征比较 | 不能声称“首次用 training-only future branch 或 latent interface 做 action-only 推理” |

### 3.2 仍可防守的窄 claim

截至本次 Zotero-first + arXiv/官方页面检索，尚未找到同一工作同时满足：

- 显式的 contact **transition segment grammar**，不是接触点、mask、轨迹或任意 latent；
- action 分支只能读取该语法，且有精确的 **action-visible finite information bound**；
- 该语法由当前视频观测预测，未来标签只在训练监督/受控 oracle 中出现；
- 在 action-centered WAM 中用 attention mask 切断 reference→action 直接边；
- 缺失双手、unknown coverage、left/right censor 被写进监督语义；
- 用 batch/RNG/noise hashes 做 matched causal controls。

因此可写成：

> “We study whether a finite, auditable contact-transition grammar can be the only action-visible bridge from egocentric visual dynamics to heterogeneous bimanual control in an action-centered WAM.”

不应写成：

> “We are the first to use contact/affordance tokens, discrete latents, or human videos for cross-embodiment manipulation.”

“尚未检索到组合前例”只是截止日期下的检索结论，不是对全体文献不存在的证明。

## 4. Idea A：Auditable Finite Contact-Transition WAM（主 idea）

### 核心假设

人到机器人真正可迁移的不是手指关节轨迹，而是低带宽的交互结构：在未来 H=48 的窗口内，哪一侧/哪条 interaction edge 与哪个 functional site、哪个 object role，以何种 contact mode，在何时建立和释放接触。若动作分支只能通过这个有限语义通道读取视觉计划，那么 Ego 预训练对 OOD 几何/外观/异构手的收益应大于连续 latent、直接视觉 shortcut 或单纯辅助监督。

### 与最近工作的冲突

- VLAff、BridgeACT、CoA-VLA 已经覆盖“where/how/move”“tool-target roles”“affordance chain”。
- CAP、GeCCo、OmniContact、DexMani 已经覆盖 contact-conditioned control、contact flow 或 transition prior。
- VQ-VAE/FSQ/VQ-BeT 已经覆盖离散编码。
- GWP-0.5/Being-H0.7 已经覆盖 action-centered WAM 或 latent interface。

真正区别只能放在 **有限语法是唯一 action-visible route，并对其容量和因果作用做审计**，而不是 token 名称。

### 真正新颖点

1. 22D 字段只是存储形式；科学对象是有限语法：absence 唯一状态，presence/censor 二值，side/site/role/mode one-hot，start/end 量化且满足 censor 和时序约束。
2. 合法 timing/censor 状态数为 `B(B+1)/2 + 2B + 1`，每段状态数为 `1 + (3×6×4×4)×N_time`；B=16、Q=12 时精确上界为 48,673 states/segment、15.570834 bit/segment、186.850009 bit/chunk，而不是笼统说“低维”。
3. action expert 无权读取 reference tokens 或 1024D planner hidden；action-only cache 保存 hard semantic，不保存连续 planner 值。
4. `hard_st` 与 `hard_detached` 前向逐字节一致，只改变 action loss 是否能回传 planner，从而分离“语义通道存在”与“梯度路由”。
5. paired hash collector 证明增益不是数据顺序、增强、timestep/noise 不一致造成的。

### WAM 架构插入点

架构来源是 GigaWorld-Policy-0.5 的 action-centered Mixture-of-Transformers；修改借鉴 VQ-VAE 的 hard-forward/straight-through 优化与 FSQ 的固定有限 levels，但离散化对象从 action/latent 改成有字段语义的 contact segment。它不是从 VLAff 或 BridgeACT 复制一个外部 planner，而是在 GWP 的 causal mask 内增加受限 planner→action channel。

```text
state + reference + text
        │
        ▼
contact query planner ──> segment logits ──> finite grammar transport
                                                │
                                                ▼
                                      22D-to-1024 encoder
                                                │
state + noisy action ───────────────────────────┤
                                                ▼
                                           action expert

future-video expert may read all streams during training;
action expert cannot read reference/future/planner-hidden in strict mode.
```

对应代码：

- 路由/transport/planner：`world_action_model/models/transformer_wa_casual_mot.py`。
- grammar/capacity：`world_action_model/contact_transition_schema.py`。
- Ego Stage 1：`configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py`。
- Robot Stage 2：`configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py`。
- loss 与 oracle merge：`world_action_model/trainer/wa_casual_trainer_mot.py`。
- sidecar loader：`world_action_model/datasets/wam_lerobot_dataset.py`。

### 标注

- segment 字段：presence；side/edge（left/right/object-object）；canonical site；object role；contact mode；start/end tau；left/right censor。
- 只允许 human-adjudicated/reviewed v1.1 sidecar。
- `unknown` 不是 negative；只有 exhaustive coverage 的空 query 才是负例。
- 按物理 segment 计数，不把 359 个滑窗重复项写成 359 个独立接触事件。
- 双人 A/B 独立看原视频，第三人裁决；自动/VLM/EgoLoc 只能生成 proposal。
- 新数据必须采集 subject/session/operator/container instance/material/geometry/liquid/viscosity/opacity/fill/batch/camera calibration/outcome。

### Benchmark

1. **Event benchmark（Ego）**：onset/release MAE、segment tIoU、one-to-one event F1、side/site/role/mode macro-F1、ECE/Brier、censored NLL。
2. **Action prediction（离线）**：masked flow/action loss、grip/aperture误差、wrist trajectory error；但只能作机制指标，不能替代 rollout。
3. **Simulation（受控）**：custom lab spec + RLBench/LIBERO 的容器/工具类任务；测试 first contact、phase order、spill/collision、目标容器命中和体积误差。
4. **真实 G1**：至少覆盖抓取试剂瓶、稳定烧杯、从烧杯到烧杯倒液、瓶盖/塞子操作、容器到架位放置；报告任务成功和分阶段失败，不只报平均值。
5. **OOD 分组**：容器实例/几何/材质，液体黏度/透明度/着色，操作员/采集 session/batch，相机标定/背景。不得用 episode number 代替这些组。

### Baselines

- GWP-0.5 original Q0 direct route。
- Q12 direct λ=0 parameter/FLOP control（待实现 discarded-planner route）。
- aux-only contact supervision（待实现，action 不看 planner）。
- raw 1024D planner hidden route（待实现）。
- continuous 22D `soft_semantic`。
- `hard_st` 与 `hard_detached`。
- oracle hard contact upper bound。
- shuffled whole-condition 与 one-chunk delayed condition（待实现）。
- 同带宽但无字段语义的随机/learned code control；否则不能把有限容量效果归因于 contact grammar。
- 人手 pose/retargeting baseline（EgoVLA-style）与 tool-target 3D flow baseline（BridgeACT-style），条件允许时加入。

### Matched controls

每个可称为 paired 的运行必须使用 seeds 6666/7777/8888，并且六类 hash 全相等：

1. ordered dataset/sample identity；
2. raw batch；
3. sampled augmentation parameters；
4. visual/action timesteps；
5. visual noise；
6. action noise。

此外必须绑定完整 Git commit、control config SHA、variant config SHA、collector source SHA、world size、rank set 和 trace bytes。当前只有 soft/hard-ST/hard-detached 具备 runner；其余消融在 runner 实现前不能进入“paired”表格。

### 关键消融

- B ∈ {8,16,32}：rate-distortion，而非只报告 B=16。
- Q ∈ {0,4,8,12}：segment capacity 与 overflow。
- soft vs hard-ST vs hard-detached。
- semantic grammar vs matched-bit arbitrary code。
- strict route vs direct reference route vs aux-only vs raw hidden。
- predicted vs oracle；oracle 不是 deployable policy。
- shuffled/delayed contact；检查动作变化是否只在对应 phase 出现。
- Ego-only、robot-only、Ego→robot two-stage；固定 robot demo 数量。
- full labels、missing-one-hand、positive-only、censored；对应 Idea B。
- 20 Hz Ego vs robot-native FPS 的 normalized tau；必须显式检查物理时长差异。

### 成功门槛

沿用并细化仓库 gate：

- oracle contact 相对 GWP-0.5：part-sensitive task success ≥ +10 percentage points；correct-first-contact ≥ +15 points。
- predicted contact 至少恢复 oracle 增益的 70%。
- strict bottleneck 相对 matched direct-visual auxiliary baseline：三个预注册 OOD split 上都 ≥ +5 points。
- shuffle/delay 导致预期 phase-specific 行为下降，而不只是全局 loss 抖动。
- cached action-only latency 增加 ≤10%。
- A/B event agreement 与每字段一致性达到预注册门槛；split/group leakage 为零。
- 真实任务结果使用 paired seeds/conditions、置信区间和逐任务 failure taxonomy。

### 停止门槛

- oracle 都达不到 +10/+15：停止扩大弱标注与 Ego 预训练，说明 contact schema 对任务信息不足。
- predicted 恢复不足 70%：先停止 robot finetune，问题在 planner/labels，不应靠更多 robot data 掩盖。
- arbitrary matched-bit code 与 semantic grammar 等价：不能声称语义结构有效，只能称有限 latent。
- strict route 连续三个 OOD split 都不优于 aux/direct：放弃“contact 是充分桥梁”的主 claim。
- 12 段频繁 overflow：先改 grammar/Q，不启动大训练。
- action-only latency >10% 或 cache 泄露连续 hidden：不符合 GWP-0.5 部署故事。

### 资源估计

- 人工：先完成 12 段 minimum proposal 的 A/B+裁决校准；通过后扩到 60–120 段并补齐 group metadata。工时应由首 3 段的实测分钟数外推，不预先伪造固定工时。
- CPU：sidecar rebuild、preflight、matched trace compare；不需要 GPU。
- GPU：先用一个 seed、200–500 steps 测显存/吞吐；预算公式为 `4 × wall_hours × variants × seeds` GPU-hours。只有 oracle/predicted 小样本 gate 通过后才展开 3 seeds；不能先跑完整 5B suite 再解释标签问题。
- 真机：先 bench 验证 action mapping/watchdog，再每个任务做少量 safety rollout；正式统计建议每个 model×condition 至少 20 次，具体总数按预注册功效分析确定。

### 风险

- 186.85 bit 仍可能足以编码非预期信息；需要 matched-bit arbitrary code、field permutation 和 probe 检查。
- hard-ST 是有偏梯度估计；hard-detached 是必要对照。
- 接触语法可能丢失连续力、摩擦和 object state；RH56 tactile 只能在 robot stage 提供残差，不应偷偷塞回 Ego semantics。
- Q=12 ordered queries 对复杂、多接触灵巧操作可能不足。
- 当前数据只有倒液，无法证明“通用生化实验技能”。

## 5. Idea B：Censored Bimanual Contact Event Benchmark

### 核心假设

Ego 视频的双手缺失不是随机缺失：一只手可能遮挡、出画、已经持物，或只有 positive interval 被审阅。若把未标注帧当 no-contact，模型会系统性学到错误的 release/onset 和手侧偏差。显式 coverage + field-level unknown + boundary censor 能让 event predictor 在真实缺失机制下更准、更校准，并改善下游动作。

### 冲突与定位

- [Ego4D FHO 官方指南](https://ego4d-data.org/docs/data/annotation-guidelines/) 已明确“物体已在手中时可以没有 CONTACT frame”，并区分 PRE/CONTACT/PNR/POST。
- [Forecasting Action through Contact Representations](https://arxiv.org/abs/2102.00649) 已按 making/breaking contact 组织 Ego action。
- [Zero-Shot Temporal Interaction Localization / EgoLoc](https://arxiv.org/abs/2506.03662) 已定位 contact/separation timing。
- 2HandedAfforder 已覆盖 bimanual actionable region；[TouchAnything](https://arxiv.org/abs/2605.13083) 已提供同步双手 pose/tactile 的更强监督。

所以不能把“contact/separation 标注”或“双手 affordance”本身当 novelty。它更适合作为 **可信监督协议 + missingness benchmark + 主方法必要的数据贡献**。

### 真正新颖点

- 把 coverage（unknown/positive-only/exhaustive）、字段级 unknown、left/right censor 与物理 segment identity 统一进 WAM contact token 的训练/评测契约。
- 对非随机缺失设计可控 corruption benchmark，并验证错误 negative 是否传导到 robot action。
- 不是只做逐帧 accuracy，而是 one-to-one event matching、boundary MAE、segment tIoU、calibration 和下游 rollout 联合评测。

### WAM 插入点/代码位置

B 不新造 policy backbone；它修改的是 A/GWP-0.5 的 supervision semantics。其事件边界任务来源可追溯到 Ego4D FHO 与 EgoLoc，其 set/segment 输出接到现有 contact queries，而不是另建一个补全网络。

- 数据契约：`track_a/contact_labels_v1_1/schema/csv_schema.json`、`ADJUDICATION_PROTOCOL.md`。
- 构建与复核：`scripts/contact_labels/build_contact_segment_sidecars_v1_1.py`、`validate_contact_segment_sidecars_v1_1.py`、`check_formal_contact_dual_annotation_v1_1.py`。
- loader/loss mask：`world_action_model/datasets/wam_lerobot_dataset.py`、`world_action_model/trainer/wa_casual_trainer_mot.py`。

### 标注与 benchmark

- A/B 隔离 packet，第三人裁决；报告 boundary tolerance 下 event F1、Krippendorff/Cohen agreement、字段一致率。
- 缺失机制分层：left absent、right absent、both partially occluded、pre-held/left-censored、release unseen/right-censored、positive-only coverage。
- 人工构造 mask corruption 仅从完整 adjudicated 段生成，以得到已知 ground truth；不能用现有 unknown 数据当真值。
- 对比 naive-zero、ignore-all-window、field-mask+censor、learned imputation、oracle-full-label。

### Matched controls / 消融

- 相同 adjudicated 完整集派生不同 missingness view，保持 action/frames/RNG 完全一致。
- coverage type、censor、hand side、mask granularity、set ordering 分别消融。
- 检查“标签指标变好但动作无变化”的情况；若如此，B 只能作为 dataset hygiene，不能列主贡献。

### 成功/停止门槛

- 在预注册 missingness severity 上，event F1 和 calibration 显著优于 naive-zero，且 strict WAM 的 correct-first-contact/rollout success 有稳定增益，才作为论文贡献。
- 若只改善离线 label loss、无下游收益，则并入 appendix/protocol。
- 若 A/B 不能达到预设一致性，停止模型训练，先修订标签词典。

### 资源与风险

- 资源主要是独立人工标注和裁决，不是 GPU。
- 最大风险是数据规模太小、缺失模式与任务类别混淆；必须按 subject/session/object/liquid/batch 分组，而非随机 frame split。

## 6. Idea C：Visibility-Gated Spatial Auxiliary（boundary/rim，不重建透明液体）

### 核心假设

完整 3D 重建不是唯一空间监督。对透明倒液，容器 boundary/rim 往往仍可从 RGB 局部化；把可见字段作为 training-only world-model auxiliary，可以强化“抓哪里/对准哪里”，同时让 stream/contact 等不可观测字段保持 unknown，而不制造 dense pseudo-GT。

### 冲突与定位

- VLAff 已预测 visual affordance heatmap、grasp 和 trajectory tokens。
- ST-VLA 已用 3D trajectory + smooth 4D spatial masks。
- CoA-VLA 已用 object/grasp/spatial/movement visual-text affordance。
- BridgeACT 已用 tool-target mask/3D flow，且包含 pour water。
- Mask World Model 已把 semantic masks 作为 world-model bottleneck。

因此“加空间 mask/token”本身没有 novelty。C 只有在 **visibility gating 与有限接触语法的互补因果作用** 被证明时才值得保留。

### 真正新颖点

不是新的 mask decoder，而是把 `不存在 / 存在但不可见 / 可局部化` 明确分开，并检验 visibility-censored spatial supervision 是否在**不绕开有限 contact channel**的前提下产生独立因果收益。若无法同时证明这个 missingness 语义与有限通道的互补作用，C 没有独立新颖性，应从主方法删除。

### 当前数据证据

`track_b/visibility_audit_v0/prescreen_summary.json` 的单人 100 帧预审：

- boundary：70% 可局部化；source rim：70%；target rim：100%；
- liquid surface：20%，且清楚帧全部来自紫色液体 episode 58；
- stream：0%；contact：0%。

这只支持开展 boundary/rim 的正式双标，不支持对 liquid/stream/contact 做像素监督。

### WAM 插入点/架构

- 架构来源是 Mask World Model 的 training-time semantic prediction 和 VLAff/CoA-VLA 的 spatial affordance decoder；修改点是增加 explicit visibility state，并保持该 decoder 对 action 不可见，除非其输出被重新量化、纳入有限 grammar 和容量计数。
- 从 state/reference/text/contact planner 的视觉 hidden 接一个 training-only mask/heatmap decoder；action 严格路由不直接读取该 decoder hidden。
- 可选方案：把被正式标注且可见的 rim/boundary 压成少量 quantized spatial bins，作为 contact grammar 中独立、受容量计数的字段；不能用连续热图绕过信息上界。
- 代码预期插入：`world_action_model/models/transformer_wa_casual_mot.py` 的 planner 输出附近、`wa_casual_trainer_mot.py` 的辅助 loss；当前尚未实现。

### 标注/benchmark

- 先按 `track_b/visibility_audit_v0/annotations_dual_template.csv` 完成 A/B+裁决；0=不存在，1=存在但不可局部化，2/3=可局部化，0 与 1 必须分开。
- boundary/rim：IoU、centerline/rim ellipse distance、temporal jitter；按透明/着色、遮挡、运动模糊分层。
- 下游：first-contact、rim collision、target alignment、spill/volume error。
- Baselines：无 spatial aux、generic segmentation aux、VLAff-style heatmap、MWM-style semantic mask、quantized rim token。

### Matched controls / 消融

- auxiliary-only 与 action-visible spatial token 必须分开；前者检验 representation learning，后者会改变 channel capacity。
- visibility gate oracle、predicted gate、无 gate；0/1 合并是明确负对照。
- rim/boundary、liquid surface、stream/contact 分字段逐步加入，禁止一次性多头后归因。

### 成功/停止门槛

- 正式双标：rim/boundary 二值 kappa ≥0.70；liquid/stream/contact ≥0.60；任何字段 0/1 分歧率 ≤10%。
- frozen 2D teacher 在独立帧上的 precision/recall 与 temporal jitter 先达到预注册阈值，再接 WAM。
- 若 boundary/rim auxiliary 相对 A 没有 ≥3 points 的 OOD task success 或明显减少 rim collision，则删掉 C，保持主故事简洁。
- 当前 stream/contact 为 0% 可局部化：在补充传感/可见数据前直接 **停止** dense stream/contact head。

### 资源与风险

- 正式 100 帧双标和 frozen teacher 是低成本 CPU/单卡离线工作；不应先动 5B WAM。
- 风险是 spatial auxiliary 重新打开视觉 shortcut；必须保证 action 仍只看有限 contact route，或把新增字段纳入容量计算。

## 7. Idea D：Morphology-Aware Asymmetric Executor（RH56 + OmniPicker）

### 核心假设

共享的 contact transition 只规定交互意图；具体执行必须由 morphology-aware adapter 将同一语义计划变成左侧 RH56 的 6 通道操作与右侧 OmniPicker 的 1 通道开合，同时保持两个腕部 9D 局部动作。异构双手可能比强行共享 MANO/对称手 latent 更适合“稳定容器 + 灵巧工具”的生化任务分工。

### 冲突与定位

- EgoVLA/EgoScale/EgoHumanoid、[CrossDex](https://arxiv.org/abs/2410.02479)、[DexUMI](https://arxiv.org/abs/2505.21864)、[DexFormer](https://arxiv.org/abs/2602.08278) 已研究统一动作空间、retargeting 或跨手 policy。
- [Transferring Contact, Not Just Motion](https://arxiv.org/abs/2606.15516) 已提出跨手的 force-position contact interface。
- BridgeACT 已把 embodiment-agnostic affordance 映射为具体闭环控制。

因此“morphology-conditioned adapter”独立 novelty 很弱；D 是 A 的真实系统必要条件。可能的新分析点是 **同一有限 contact grammar 在极不对称手（6DoF tactile hand vs 1DoF gripper）上的 sufficiency/feasibility gap**。

### 真正新颖点

不是 per-hand adapter 本身，而是把同一可审计 contact grammar 固定住，测量它在 6DoF 触觉灵巧手与 1DoF 夹爪之间的可执行性差异，并将失败分解为 planner insufficiency、morphology infeasibility 与 modality missingness。若实验只能证明专用 head 优于共享 head，D 只能算系统工程。

### WAM 插入点/代码位置

架构来源是 GWP-0.5 的 action flow head 与 EgoVLA/EgoHumanoid 的 human/robot action alignment；修改点不是共享 MANO，而是 shared semantic planner + embodiment-specific canonical action heads，并用硬件 manifest 把 learned action 与设备命令隔离。

- canonical25 action schema：`world_action_model/asymmetric_action_schema.py`。
- manifest/calibration：`world_action_model/asymmetric_hardware_manifest.py`、`schemas/asymmetric25_*.schema.json`。
- converter/norm/config：`scripts/convert_sonic_wrist_delta.py`、`scripts/compute_wam_task_norm.py`、`configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py`。
- action head 可进一步拆为 shared wrist trunk + RH56/OmniPicker device heads；当前不是已实现贡献。

### 标注/benchmark

- contact side 增加“执行侧角色”而不是直接编码设备关节；例如 stabilizer/manipulator/tool-holder。
- 机器人数据必须来自目标硬件，记录 measured-state/command 分开标定、latency/skew、触觉同步、watchdog 和 safety limits。
- 任务应有 hand-role swap/role-fixed 条件：右 OmniPicker 稳定、左 RH56 精细操作；可行时反转以测 embodiment feasibility。

### Baselines / matched controls / 消融

- shared single action head、per-hand heads、morphology embedding、contact-conditioned per-hand heads。
- symmetric Inspire old data只能做 representation pretrain/负迁移研究，不能作为目标硬件结果。
- 固定语义计划，比较不同 adapter；固定 robot demonstrations 数量与采样。
- 触觉观测 on/off、left hand channels、right aperture、role swap。

### 成功/停止门槛

- 目标硬件 canonical25 converter、bench calibration、command mapper、watchdog 和 dry-run 全通过后才训练/部署。
- 若 contact plan 对两种手的可行性差异无法预测，或 morphology head 不优于普通 per-device BC，则 D 只保留工程实现。
- 任何旧六指右手→OmniPicker 的压缩映射都是 No-Go。

### 资源与风险

- 需要重新采集目标硬件数据；这是主要成本。
- RH56 tactile 与 OmniPicker 无对称触觉，容易让模型把传感缺失误当接触状态；必须有 modality mask。
- 目前没有物理执行链，不能给出真机成功率或部署承诺。

## 8. Idea E：Transparent-Liquid Reconstruction / Outcome World Model（当前 No-Go）

### 核心假设

透明几何或液体状态估计理论上能为倒液提供更强监督，但当前 RGB/深度数据不足以支撑可靠 dense target。更保守的版本是只预测 coarse outcome（未倒/正在倒/已转移、spill、target fill bin）并使用称重/触觉/染色或仿真真值监督。

### 冲突

- [TranSplat](https://arxiv.org/abs/2502.07840)、[TransDiff](https://arxiv.org/abs/2503.12779)、[Diffusion Knows Transparency](https://arxiv.org/abs/2512.23705)、[DeLTa](https://arxiv.org/abs/2510.05662) 已直接处理透明物体深度、法向、抓取或长时操作。
- [X-WAM](https://arxiv.org/abs/2604.26694)、[WAM4D](https://arxiv.org/abs/2606.14048)、ST-VLA 已将深度/4D geometry 融入 WAM/VLA。

因此“透明物体重建难，所以不用重建”不是足够的方法贡献；“用视频 diffusion 解决透明深度”也已有直接前例。

### 真正新颖点

当前没有可成立的独立方法新颖点。只有在新增同步物理真值后，“对不可见液体状态采用 censored coarse-outcome supervision，并证明它与有限 contact grammar 互补”才可能成为 auxiliary 贡献；在现有数据条件下不得包装成主 idea。

### 可接受的 auxiliary 版本

- 架构来源可以是 GWP-0.5 的 training-only future expert、Mask World Model 的非 RGB target，或透明 depth 方法提供的 frozen perception teacher；修改点是只预测有独立传感真值的 coarse physical outcome，不让连续 depth/flow hidden 绕过 A 的有限通道。
- training-only outcome head：transfer state、spill、fill/weight bin；由秤、触觉或受控仿真提供标签。
- uncertainty/visibility gate：不可见时不回归像素级 stream/contact。
- 只作为 A 的 downstream physical-state probe，动作仍经有限 contact grammar。

预期插入位置是 `world_action_model/models/transformer_wa_casual_mot.py` 的 future/visual expert 输出与 `world_action_model/trainer/wa_casual_trainer_mot.py` 的 training-only loss；数据键应经 `world_action_model/datasets/wam_lerobot_dataset.py` 加载。当前三处都没有该 outcome 分支，因此这是 spec，不是实现证据。

### Benchmark / baselines

- 标注来自同步秤/触觉/流量或受控仿真的 transfer-state、spill、fill/weight bins；RGB 不可见帧保留 unknown，不用 VLM 推断替代传感真值。
- 透明 vs 着色液体、不同黏度/fill、玻璃/塑料、背光/偏振条件。
- depth sensor raw、透明 depth completion、RGB-only、rim-only、contact+outcome。
- 指标：volume transfer error、spill mass、stream onset/offset（传感真值）、task success、inference latency。

### Matched controls / 消融

- 固定视频/action/seed/noise，比较 no-outcome、outcome auxiliary-only、action-visible outcome、oracle outcome；最后一种只能给上界。
- transparent depth teacher on/off、coarse outcome vs dense flow、visibility mask on/off、sim-only vs sensor-real、着色 vs 透明分别消融。
- auxiliary-only 运行仍应通过六类 matched hashes；若让 outcome 对 action 可见，必须新增到 channel capacity 和 comparator 的 config binding，不能沿用 A 的 186.85 bit 声称。

### 成功/停止门槛

- 至少 10 段具有可见液流或同步 force/tactile/weight ground truth 的补充验证集，才允许训练 stream/outcome。
- 若透明 depth/normal teacher 在独立容器/光照上不稳定，停止接入 WAM。
- 若 outcome head 不改善真实 spill/volume error，只保留诊断，不进主模型。

### 资源与风险

- 需要新传感与补采；仿真液体的 sim-to-real gap 很大。
- 完整 reconstruction 会显著扩大论文与工程范围，并削弱 A 的清晰因果故事。因此当前主线明确 No-Go。

## 9. 统一 benchmark 设计

### 9.1 数据层级

| 层级 | 数据 | 允许的结论 |
|---|---|---|
| L0 | 现有 66 段 Ego pour + 未裁决 proposal | loader/shape/标注流程调试；不能训练科学模型 |
| L1 | 双标+裁决 contact segments，真实 group metadata | temporal/contact benchmark；可做小规模 Stage 1 |
| L2 | 目标硬件 canonical25 robot data，bench-verified calibration | robot finetune 与离线 action metric |
| L3 | robot-verified execution + safety chain | 有限真机 rollout |
| L4 | 多任务、多容器/液体/batch 的预注册 OOD 评测 | 投稿级 generalization claim |

### 9.2 任务族

- P1：抓取/放置试剂瓶或烧杯；检验 correct-first-contact。
- P2：一手稳定、一手开盖/塞紧；检验异构手角色与接触 switch。
- P3：烧杯间倒液；检验 source/target/tool roles、rim alignment、flow outcome。
- P4：试管/容器插入架位；检验精确空间和 impact/slide。
- P5：移液器吸取/释放或搅拌；只有在工具/传感/安全数据成熟后加入。

现有数据只覆盖 P3 的一个窄分布；论文标题和摘要不能提前写成“general biochemical manipulation”。

### 9.3 指标

- Contact：event F1、onset/release MAE、segment tIoU、field macro-F1、ECE/Brier。
- Geometry：first-contact site error、rim distance、collision count。
- Action：wrist trajectory、hand/aperture error、smoothness；仅作离线指标。
- Task：阶段成功、总成功、spill、transfer volume error、object drop、safety intervention。
- Generalization：每个真实 OOD axis 单独报告；不能只给 pooled average。
- Efficiency：action-only latency、peak memory、cache size；相对 GWP-0.5。

### 9.4 统计协议

- 数据 split 以 subject/session/operator/object instance/liquid/batch group 为单位。
- seeds 6666/7777/8888；报告 per-seed/per-task 结果和 bootstrap confidence intervals。
- robot rollout 使用相同初始条件清单、随机化 seed、operator protocol；模型顺序交错以避免设备漂移。
- 预注册 primary endpoint：OOD task success 与 correct-first-contact；其余为 secondary。

## 10. 分阶段执行与资源闸门

### Gate 0：标注可用性

- 完成 minimum 12 episodes 的真正 A/B 独立标注和第三人裁决。
- 达到 agreement 门槛；补齐 group metadata；sidecar validator 全过。
- 失败：修词典/重标，不跑 GPU。

### Gate 1：oracle sufficiency

- 在小规模数据上跑 GWP direct 与 oracle hard contact。
- 未达到 +10 task success / +15 first-contact：停止主线扩大。

### Gate 2：prediction and causality

- predicted 恢复 ≥70% oracle gain。
- 完成 Q12 direct、aux-only、raw hidden、shuffle、delay 与 arbitrary matched-bit runners。
- 六 hash 全匹配后才称 paired causal ablation。

### Gate 3：Ego Stage 1

- 先 one-seed 200–500 step throughput/稳定性；再三 seed。
- 固定 robot data budget，比较 robot-only 与 Ego→robot；否则不能把收益归于 Ego。

### Gate 4：target hardware Stage 2

- 新采 canonical25；硬件/标定/时钟/安全 evidence 全通过。
- bench dry-run 后才能真机；旧 41D dual-Inspire 数据不解锁该 gate。

### Gate 5：可选 auxiliary

- C 只有在正式可见性双标和 frozen teacher 通过后接入。
- E 只有新传感/补采改变 stream/contact 可观测性后重开。

## 11. 投稿包装建议

### 建议标题

- **Contact Is a Contract: Auditable Finite Semantic Transport from Egocentric Video to Heterogeneous Bimanual WAMs**
- **Finite Contact-Transition World Action Models for Cross-Embodiment Laboratory Manipulation**

### 三条主贡献写法

1. 一个 contact-transition finite grammar，并证明它是 strict WAM 中唯一 action-visible 的视觉计划通道。
2. 一个 missingness/censor-aware bimanual contact annotation and benchmark protocol，避免把未标注手/边界当负例。
3. 在 Ego→G1 异构双手流程中，用 matched causal controls 验证语义、容量和梯度路由各自的作用。

### 不应出现的宣传句

- “首次将 affordance/contact token 用于 robot manipulation”。
- “首次从 egocentric human video 学机器人操作”。
- “首次跨 embodiment 学接触变化”。
- “透明物体无需重建即可解决”——当前只有 prescreen，没有任务结果。
- “359 个 contact events”——它们是重复窗口，不是独立物理事件。
- “25D 包括 G1 WBC29”——canonical25 是双腕+双手；WBC29 是单独执行契约。

## 12. Zotero-first 证据

本次先同步 Zotero，再用本地 SQLite/Obsidian note 核对个人已读材料，最后才补 arXiv/官方页面。个人库当前检索到 826 条记录。直接相关证据如下：

| Zotero item | note | 个人知识证据 | 对设计的影响 |
|---|---|---|---|
| `726MZ5TZ` GigaWorld-Policy-0.5 | `M29LYDRU` | 两个 MoT expert；action 不看 future image；推理只出 action | 选择 action-centered strict routing |
| `DIETABIS` GigaWorld-Policy | `NL62US9E` | 以 attention mask 控制 observation/state/action/future 信息流 | 因果切边应落在 mask，不只加 auxiliary head |
| `FVS59CQW` EgoVLA | `VKIGKI8I` | 个人笔记强调 wrist pose + MANO unified space 和后续 retargeting | 不能把 pose retargeting 当 novelty；改测接触语义 |
| `7REGKFSA` EgoScale | `NM24CP3V` | 大规模 Ego pretrain、中训练、后训练；个人笔记指出方法细节披露有限 | 要用固定 robot budget 和完整 gate 证明迁移收益 |
| `EK7DCV3Q` EgoHumanoid | `AZW7F2RK` | view/action alignment，双腕 delta + grip 等统一接口 | 20D Ego action 是 baseline，不是目标 hardware action |
| `Z4H74JXQ` 2HandedAfforder | `L24YJLU9` | 个人笔记总结为双手可接触区域及机器人引导 | “双手 affordance region”不再可作主 claim |
| `JRGPD6HS` OmniContact | — | humanoid contact flow 与技能串联 | contact flow 本身已碰撞 |
| `HULYEXXC` FSQ | — | 固定 scalar levels 的有限隐空间 | 支持 fixed time bins 与精确容量，而非 learned codebook |
| `JRKAKXNT` FAST | `CBLGABU2` | 动作 token 压缩/频率影响控制性能 | B/Q 必须做 rate-distortion，而非单设置 |
| `YGQWQ9PE` DreamZero | `MEHDBLK9` | 同时预测 video/action，并关注实时推理 | WAM baseline 与 action-only efficiency |
| `XINDFWEJ` Being-H0.7 | `B3MRC9YW` | future-informed latent query 位于 action 前 | raw latent interface 已有强碰撞 |
| `V2P9FFKM` MotionWAM | `W7TBI2D4` | Ego/video DiT 与 motion token/WBC 结合 | humanoid WAM + motion token 不是空白区 |

`idea-discussion` 记录：

- 主有限接触语义 card：`762fb9fc-616f-4d2a-8164-f68ba95da21c`；本轮 discussion：`ea0cb1d3-700b-4f8f-bb70-d495e7dd4282`。
- visibility-gated auxiliary card：`80276d05-4c57-4cf8-b2fd-167815f2db8d`；discussion：`379736e4-35ff-4692-a612-27d291df94d7`。
- censored bimanual card：`e85ea996-3d04-473d-9e83-202513d11a80`；discussion：`ba3f469a-b89a-4cb1-b4f0-505c9b475d0e`。
- 既有 labels/protocol card：`30e9176d-f97d-49a7-b0dc-10f98e8a57da`；discussion：`64931002-fc27-4945-8070-64bf8cbdf78a`。

workflow 的自动 precedent ranking 对长中文复合 query 返回了若干 failure-recovery 论文，相关性明显偏弱；本 dossier 没有把它们当主证据，而是进一步逐篇核对上述原始论文。这一点也应保留在审计记录里，避免把检索器输出误当文献结论。

## 13. 原始来源清单

### WAM / Ego-to-robot

- [GigaWorld-Policy-0.5](https://arxiv.org/abs/2607.13960)
- [GigaWorld-Policy](https://arxiv.org/abs/2603.17240)
- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [Being-H0.7](https://arxiv.org/abs/2605.00078)
- [EgoVLA](https://arxiv.org/abs/2507.12440)
- [EgoScale](https://arxiv.org/abs/2602.16710)
- [EgoHumanoid](https://arxiv.org/abs/2602.10106)
- [ZeroWBC](https://arxiv.org/abs/2603.09170)
- [X-WAM](https://arxiv.org/abs/2604.26694)
- [WAM4D](https://arxiv.org/abs/2606.14048)

### Affordance / contact / bimanual

- [VLAff](https://arxiv.org/abs/2608.05215)
- [ST-VLA](https://arxiv.org/abs/2603.13788)
- [CoA-VLA](https://arxiv.org/abs/2412.20451)
- [BridgeACT](https://arxiv.org/abs/2604.23249)
- [2HandedAfforder](https://arxiv.org/abs/2503.09320)
- [Contact-Anchored Policies](https://arxiv.org/abs/2602.09017)
- [Mask World Model](https://arxiv.org/abs/2604.19683)
- [Contact-conditioned learning of locomotion policies](https://arxiv.org/abs/2408.00776)
- [GeCCo](https://arxiv.org/abs/2509.17582)
- [OmniContact](https://arxiv.org/abs/2606.26201)
- [DexMani](https://arxiv.org/abs/2608.00554)
- [Forecasting Action through Contact Representations](https://arxiv.org/abs/2102.00649)
- [Zero-Shot Temporal Interaction Localization](https://arxiv.org/abs/2506.03662)
- [Ego4D FHO annotation guidelines](https://ego4d-data.org/docs/data/annotation-guidelines/)
- [TouchAnything](https://arxiv.org/abs/2605.13083)
- [CrossDex](https://arxiv.org/abs/2410.02479)
- [DexUMI](https://arxiv.org/abs/2505.21864)
- [DexFormer](https://arxiv.org/abs/2602.08278)
- [Transferring Contact, Not Just Motion](https://arxiv.org/abs/2606.15516)

### Discrete representation

- [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [Straight-Through Estimator](https://arxiv.org/abs/1308.3432)
- [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)
- [VQ-BeT](https://arxiv.org/abs/2403.03181)
- [FAST](https://arxiv.org/abs/2501.09747)

### Transparent object / liquid risk

- [TranSplat](https://arxiv.org/abs/2502.07840)
- [TransDiff](https://arxiv.org/abs/2503.12779)
- [Diffusion Knows Transparency](https://arxiv.org/abs/2512.23705)
- [DeLTa](https://arxiv.org/abs/2510.05662)

## 14. 最终研究决定

1. **保留并推进 A+B**：这是唯一同时与用户 insight、现有 GWP-0.5 实现和可证伪实验对齐的投稿主线。
2. **C 是条件 auxiliary**：先完成 Track B 双标和 frozen teacher；无独立下游增益就删除。
3. **D 是必需系统边界，不单独包装 novelty**：必须等目标硬件新数据和安全链。
4. **E 当前 No-Go**：不以重建、液流 field 或静态 RGB contact mask 扩张主线。
5. **现在不启动 4×H20 训练**：正式 labels、group split、T5/sidecar evidence、target hardware converter/runtime、matched runner 完整性均未解锁。下一项最有信息价值的工作是完成真实 A/B 标注与裁决，而不是增加训练步数。
