# Idea D：Censored Bimanual Contact-Event Benchmark 与 Missingness-Aware WAM 审查

> 日期：2026-08-11
> 代码审计基线：`d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88`
> 工作性质：Zotero-first 文献核查、协议设计和只读代码审计；没有训练、GPU、数据改写或 canonical worktree 修改。
> 本轮 `idea_id`：`15d89a15-c958-4a33-9735-a9547d45fa27`
> 初始 `discussion_id`：`dbf2e3a4-9d4d-436a-9e74-4d727ca8d037`
> 精确 librarian 复查 `discussion_id`：`90187152-d080-45c4-b5ef-b9f12f1a63a0`

## 0. 决定

**当前决定：独立论文 No-Go；作为 Auditable Finite Contact-Transition WAM 主线的数据、监督和评测贡献 Conditional Go。**

这个问题是真问题，而且当前仓库已经做对了最重要的第一步：`unknown` 没有被编码成
negative，物理 segment 与滑窗 query 被区分，left/right censor 进入了 22D 语法。但是，
“mask + censor flags”本身还不足以支持独立方法论文：

1. op17 目前只是 12 段、3,695 个 20 Hz 帧（184.75 秒）的**空白** A/B 正式任务包，
   assignment checker 可通过，但没有一行提交的 physical-contact GT；
2. 当前 66 段只覆盖一个倒液任务，op29 证明 subject/session/operator/object instance/liquid/
   batch 均不能从现有 artifact 恢复，因而不能做 production OOD split；
3. op28 只证明 `hand_status=0=open,1=closed` 的动作代理语义，以及 `hand_active`/tracker
   validity 如何屏蔽该代理的动作损失；它明确不支持 physical-contact truth；
4. 当前 grouped loss 仍是 masked BCE/CE/Smooth-L1，censor 只是要预测的字段，不是
   censor-aware likelihood；MNAR propensity、PU/SAR、敏感性分析都未实现；
5. 现有原始工作已覆盖 contact frame、time-to-contact、contact/separation localization、
   双手 3D/contact、热/触觉 contact maps。尤其 2026 年的 EgoTouch/TouchAnything 已有
   1,891 episodes、208 tasks、头戴+双腕相机、双手 pose 和连续 pressure map，显著提高了
   “双手接触 benchmark”独立成篇的规模与物理真值门槛。

可防守的新颖组合不是“第一次标接触”或“第一次处理缺失标签”，而是：

> 在 action-centered WAM 中，把 per-side/per-relation coverage、物理事件 identity、窗口删失、
> 边界区间不确定性与 informative missingness 统一成一个可审计训练/评测契约，并证明这套契约
> 在严格 finite-contact bottleneck 下同时改善事件校准和机器人接触行为。

若后文的独立论文晋升门槛全部满足，可重新评估为 standalone benchmark/method paper；在此之前，
它应是 finite-contact 主线的必要可信性贡献，而不是另起一篇弱数据论文。

## 1. Zotero-first 结果与检索说明

`idea-discussion` 先同步了个人 Zotero（本次同步 100 条）并更新 idea card。初始宽查询命中
RoboBrain/FAST/RT-1/Octo 等泛化条目，不能作为本题 novelty 证据。使用同一 `idea_id` 做窄化
librarian 复查后，个人库中最直接的三条邻近工作是：

| Zotero key | 工作 | 与本题的关系 |
|---|---|---|
| `WM8H622U` | [EgoInfinity](https://arxiv.org/abs/2606.17385) | 从互联网视频恢复 metric 4D hand/object、6-DoF pose 和 contact-relevant states，并做 robot retargeting；说明“视频到接触相关状态再到机器人”已是直接竞争方向。 |
| `Z9TPXVA8` | [METIS](https://arxiv.org/abs/2511.17366) | 多源 Ego 人/机器人数据统一动作空间并训练 dexterous VLA；限制“Ego pretraining for dexterous control”的宽泛 claim。 |
| `3SVRDJ39` | [HARP-VLA](https://arxiv.org/abs/2605.31234) | 用 paired human-robot data 对齐视觉/latent action；限制“human video bridge”的宽泛 claim。 |

个人库没有直接命中 censored contact-event benchmark、MNAR contact labels 或 censor-aware WAM。
这只能说明个人知识库存在缺口，不能证明外部文献不存在。随后按 arXiv-first/论文或数据集官方页
补查。`arxiv-search` 技能的 CLI 因 `main()` 未打印返回值且系统 Python 缺 `arxiv` 包不可用；本轮
没有向系统/base 安装依赖，改用 arXiv、CVF、PMLR、NeurIPS 和数据集官方页面。

## 2. 原始工作碰撞地图

### 2.1 Ego/video hand-object 数据与任务

| 工作 | 原始监督/任务 | 能否作为本题 physical-contact event GT | 对本题的结论 |
|---|---|---|---|
| [Ego4D](https://arxiv.org/abs/2110.07058) / [FHO 官方说明](https://ego4d-data.org/docs/benchmarks/hands-and-objects/) | PRE、CONTACT、PNR、POST 关键帧，手/工具/状态变化物体框，PNR localization/object state change | **不能直接当连续事件 GT**。CONTACT 是关键帧；官方指南还明确物体已在手中时可以没有 CONTACT frame。 | 支持“pre-held/左删失是真实问题”，但不提供 exhaustive bimanual contact intervals。 |
| [VISOR](https://arxiv.org/abs/2209.13064) / [官方页](https://epic-kitchens.github.io/VISOR/site) | 271K 手工 masks、dense interpolation、67K hand-active-object relations；VOS/HOS/long-term reasoning | 帧级 in-contact/active-object 关系很有价值，但不是物理事件边界或 coverage/MNAR 协议。 | 可做视觉 baseline 和外部 frame benchmark，不能把未标帧当负例。 |
| [Forecasting Action through Contact Representations](https://arxiv.org/abs/2102.00649) | EPIC-KITCHENS 子集的 time-to-contact、手/物体 mask、making/breaking contact state graph | 已直接覆盖 contact-centered action anticipation。 | 不能声称“首次以 contact onset/release 组织 Ego action”。 |
| [100 Days of Hands](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/) | 每只手的 box、side、contact state、in-contact object box，约 100K annotated frames | 帧级 contact state，不是连续事件/删失 benchmark。 | 是必要的 single-frame contact-state baseline。 |
| [EgoHOS](https://arxiv.org/abs/2208.03826) | 11,243 图像，手和交互物体的像素标注，强调 contact boundary | 可作为视觉接触边界/segmentation baseline；无全时轴物理事件 coverage。 | 不能把像素 overlap 直接提升为 physical truth。 |
| [Ego-Exo4D](https://arxiv.org/abs/2311.18259) / [EgoHandPose 官方任务](https://docs.ego-exo4d-data.org/benchmarks/ego_pose/ego_hand_pose/) | 同步 ego/exo、多模态、3D hand/body pose、技能理解/跨视角任务 | 当前公开主 benchmark 不提供物理接触事件 GT；EgoHandPose 评测甚至明确只给 monocular ego 输入。 | 可用于多视角 annotation evidence/pose covariate，不应宣称已有 contact truth。 |
| [HOT3D](https://arxiv.org/abs/2411.19167) | 多目 ego、MoCap 手/物体/camera 3D pose、mesh | 可从几何近邻派生 contact proxy，但官方主任务是 hand/object tracking。 | 适合 geometry-derived baseline 与跨视角研究；阈值 contact 需传感器校准。 |
| [HOI4D](https://arxiv.org/abs/2203.01577) | RGB-D、3D hand pose、object pose、panoptic/motion/action labels | 没有原生连续 physical-contact event 协议。 | 适合 pose/action 外测，不是直接 GT。 |
| [H2O](https://arxiv.org/abs/2104.11181) | 双手/物体 3D pose、interaction label、多视角 RGB-D、第一视角 action recognition | interaction/pose 不等于物理接触边界。 | 是 bimanual pose/action baseline，不解决 censor/MNAR。 |
| [ARCTIC](https://arxiv.org/abs/2204.13662) | 8 exo + 1 ego、双手/铰接物体 mesh、每帧 dynamic contact/interaction field | **强邻近 GT**，但 contact 由重建 mesh 的几何 proximity 计算；9 subjects、10 objects、242 sequences。 | 可派生完整 event tracks，必须把 proximity GT 与真实触觉 GT 分层。 |
| [ContactDB](https://arxiv.org/abs/1904.06830) / [ContactPose](https://arxiv.org/abs/2007.09545) | 热成像 contact maps；ContactPose 再配 hand/object pose 和 RGB-D | 物理证据强，但主要是静态或准静态 grasp，不是长时双手事件。 | 适合作为 sensor-truth 校准和接触区域 baseline。 |
| [EgoLoc](https://arxiv.org/abs/2506.03662) | 零样本定位 grasp contact/separation timestamp | 已直接做 temporal contact/separation localization。 | 独立性必须来自 coverage/censor/MNAR + WAM downstream，而非 boundary localization 本身。 |
| [TouchAnything / EgoTouch](https://arxiv.org/abs/2605.13083) | 1,891 episodes、208 tasks、头戴+双腕 RGB、双手 3D pose、连续 pressure maps，Contact/Volumetric IoU | **当前最强直接竞争者**；连续 tactile 可导出物理 contact event。 | 若无 sensor-confirmed subset、真实缺失机制和 WAM 下游，独立 benchmark 不够强。 |

结论：公开工作分别覆盖了 contact frame、frame contact、contact/separation timing、双手 mesh
contact 和触觉 contact，但没有在已核工作中看到一个 benchmark 同时发布 per-side coverage、
global event identity、window-derived censor flags、MNAR corruption protocol、group-safe split 以及
WAM downstream causal comparison。这个“组合空位”仍然存在；单一字段或单一 loss 没有首创空间。

### 2.2 Censoring、partial label 与 MNAR

- [Nnet-survival](https://arxiv.org/abs/1805.00917) 给出可用 minibatch SGD 训练的离散时间
  hazard likelihood，直接说明右删失样本不应被当作“事件未发生”的普通负例。
- [Survival-CRPS](https://proceedings.mlr.press/v115/avati20a.html) 给出 right- 和
  interval-censored 的 proper scoring rule，可用于校准/锐度评测。
- [nnPU](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html)
  处理 positive + unlabeled 并避免无偏 PU risk 在灵活模型上变成负值；但其常见 SCAR/类先验
  假设不能直接套到“遮挡时更容易漏掉接触”的 Ego 场景。
- [SAR-PU](https://arxiv.org/abs/1808.08755) 允许标注 propensity 依赖已观测属性，更接近
  visibility/tracker-dependent missingness。
- [Are labels informative in semi-supervised learning?](https://proceedings.mlr.press/v202/sportisse23a.html)
  显式检验 informative labels，并用估计 propensity + inverse propensity weighting 修正 SSL。

必须把两种机制分开：

1. **administrative/window censoring**：固定 H=48 窗口切断事件，边界由采样设计决定，可使用
   censor-aware likelihood；
2. **label missingness**：手出画、遮挡、透明物体、blur、tracker invalid、标注预算等造成
   coverage 缺失，往往依赖场景甚至真实接触，可能是 MAR/SAR 或 MNAR。

窗口 censor flag 不能替代 missingness model；IPW/PU 也不能自动解决边界被窗口截断。若
propensity 依赖未观测真实 contact，单靠现有 observed data 不可识别，必须有随机抽检、额外
view/tactile validation subset，或只报告 pattern-mixture/selection-model 敏感性区间，不能写成
“已消除 MNAR bias”。

### 2.3 Robot imitation benchmark

- [RoboMimic](https://proceedings.mlr.press/v164/mandlekar22a.html) 强调离线 human demo 的
  数据质量、算法选择和 stopping criterion 会显著影响结果；支持固定 protocol、公开 split 和
  rollout，而不是只比较 offline loss。
- [LIBERO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)
  有 130 tasks 和 human teleoperation，强调 declarative/procedural transfer 与 task order。
- [BiGym](https://proceedings.mlr.press/v270/chernyadev25a.html) 是 40-task、3-camera 的 demo-driven
  mobile bimanual benchmark，限制“首个双手机器人 imitation benchmark”的 claim。
- [DROID](https://arxiv.org/abs/2403.12945) 有 76K trajectories、564 scenes、84 tasks、50
  collectors，说明独立 benchmark 的 group/diversity 门槛远高于当前单任务 66 episodes。

这些 benchmark 没有替代本题的 contact-event 评测，但要求本题至少证明 event benchmark 的
指标会传导到 rollout success/correct-first-contact/wrong-contact，而非成为只改善辅助 loss 的
annotation hygiene appendix。

## 3. op17 / op28 / op29 与当前 WAM 的真实能力

相关提交在当前 HEAD 中以 cherry-pick 后的 hash 存在：op17 为 `214da29`、`923209c`；op28
为 `a7be311`、`771b642`、`66c9819`、`0c5a49a`；op29 为 `e342fa6`。原 feature branch hash
不在 HEAD 祖先链上，但内容已以这些提交集成。

### 3.1 op17：formal labels 是什么、还不是什么

只读运行：

```text
python scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py --phase assignment
```

结果为 `status=valid`、66 episodes audited、12 tasks、3,695 timeline rows、A/B 文件 hash 独立。
这只证明 packet、源 hash、身份槽和 evidence mapping 合法。

**真实能力：**

- `coverage_mode ∈ {unknown, positive_only, exhaustive}`；
- global physical `[start_frame,end_frame)` segment；
- `start_boundary ∈ {observed,left_censored,unknown}`、
  `end_boundary ∈ {observed,right_censored,unknown}`；
- observed onset/release 才允许 event ID；unknown/censored 不伪造 event ID；
- A/B 隔离、第三人裁决、只接受 `human_adjudicated/reviewed` 进入 sidecar builder；
- 一致性代码包含 coverage weighted κ、tIoU≥0.5 的一对一最大匹配、event F1、median tIoU、
  observed boundary MAE、类别 κ、frame contact agreement。

**当前不能做的事：**

- A/B `coverage.csv` 仍是 `in_progress` 且 coverage_mode 为空，`segments.csv` 只有表头；
  没有 physical-contact GT、agreement report 或 adjudication；
- 12 tasks 是单任务 minimum calibration packet，不是 benchmark test set；
- coverage 是整段、全 label-space 的一行，不能表达“左手 exhaustive、右手 unknown”或
  “presence 可判、site 不可判”的 coverage；字段级 mask 只在已有正 segment 上工作；
- segment 没有 object instance/pair ID 或 `bimanual_group_id`，两个同时接触同一物体的 hand
  segments 无法无歧义组成一个 joint bimanual event；
- current agreement gate 对每个类别只要求 2 个 comparable matches，统计上不足；未对 boundary
  status κ、bimanual grouping、adjudication rate 或 cluster-bootstrap CI 设 gate；
- 匹配按 side-compatible+tIoU≥0.5，不检查 object-pair identity；no-event episode 的 per-episode
  F1 为 0，报告时需另列 true-empty accuracy，不能混成 event miss。

代码位置：

- packet/词典：`track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/README.md`、
  `LABEL_DICTIONARY.md`；
- agreement：`scripts/contact_labels/check_formal_contact_dual_annotation_v1_1.py:620`；
- strict CSV 与 window encoding：`scripts/contact_labels/contact_label_io_v1_1.py:163`、`:211`、`:296`、`:309`。

### 3.2 op28：hand_active 与 hand_status

`docs/audits/ego_action_grip_evidence_20260811.md` 的结论是
`semantics_verified_missingness_masked`，并明确“不支持 physical-contact-truth claim”。

- `hand_status` 是 `[left,right]` 的 `0=open,1=closed`，WAM 转成 canonical open fraction；
- `hand_active` 是源 tracker/action proxy 的 validity，不是“该手正在物理接触”；
- 20,013 processed frames 中，left current-active 7,087，right current-active 6,574；左手有
  30 个 episode 在 20 Hz 下完全无 active，右手有 29 个；
- closed proxy 在 tracker inactive 时仍大量出现：left 7,331/11,005、right 7,842/10,621；
- 110/118 个 left/right proxy transitions 中只有 54/58 在相邻两帧都 tracker-valid；
- `ego_hand_active_future` 只把 action dimensions 18/19 的 grip proxy loss 乘 0.25 或 0，
  wrist action loss不受影响；transition-pair mask 目前 record-only，没有辅助头消费。

因此：

> `hand_status` 只能是 actuator/grip-closure proxy；`hand_active` 只能是这个 proxy 的 validity
> covariate。二者都不能填 contact presence、contact coverage exhaustive、onset/release 或
> physical-contact negative。

代码位置：

- sidecar/alignment：`world_action_model/ego_hand_active_sidecar.py:130`、`:294`；
- transform/action weight：`world_action_model/transforms/wa_transforms_lerobot_pretrain.py:645`、`:774`；
- action/grip evidence：`docs/audits/ego_action_grip_evidence_20260811.md`。

### 3.3 op29：group recovery

当前能机械恢复的只有同 task/date/camera backend/calibration、episode clock/length，以及用 30 秒
gap 得到的四个 `diagnostic_continuity_block`：0–8、9–40、41–49、50–65。它们只允许
engineering train/validation/test，明确 `ood_eligible=false`、
`strategy_claim_allowed=false`。

以下 production group fields 全部需要采集者日志或 attestation：`subject_id`、`session_id`、
`operator_id`、`source_object_instance_id`、`target_object_instance_id`、`liquid_id`、
`collection_batch_id`。几何、材质、黏度、fill/outcome 也没有记录。不能从 episode number、时间
gap、视频外观、文件 owner 或 Git author 猜这些字段。

代码位置：

- 审计：`docs/audits/ego_66ep_group_metadata_recovery_20260811.md`；
- 恢复/校验：`world_action_model/ego_group_metadata_recovery.py`、
  `scripts/contact_labels/recover_ego_group_metadata.py`、
  `validate_ego_group_metadata_supplement.py`。

### 3.4 当前 WAM supervision 能力

| 能力 | 当前状态 | 代码位置 |
|---|---|---|
| 22D segment：presence/side/site/role/mode/start/end/left-right censor | 已实现 | `world_action_model/contact_transition_schema.py:191`、`:398` |
| exhaustive coverage 才把空 query 监督为 negative；positive-only/unknown 不填负例 | 已实现，但 coverage 不是 per-side | `contact_transition_schema.py:428`；`contact_label_io_v1_1.py:296` |
| unknown boundary mask | 已实现；只有能证明窗口外的边界才生成 censor target | `contact_label_io_v1_1.py:333`–`:389` |
| strict reviewed sidecar load/hash binding | 已实现 | `world_action_model/datasets/wam_lerobot_dataset.py:741`–`:835` |
| masked field-balanced BCE/CE/Smooth-L1 | 已实现 | `world_action_model/trainer/wa_casual_trainer_mot.py:31`、`:78`、`:101` |
| query assignment | 只有 deterministic chronological order；Hungarian 仅 callable hook，训练调用未传 matcher | `contact_transition_schema.py:408`；`wa_casual_trainer_mot.py:78`、`:962` |
| censor-aware survival/event likelihood | 未实现；当前 start/end 是 sigmoid+Smooth-L1，censor flag 是 BCE | `wa_casual_trainer_mot.py:101`–`:159` |
| propensity/IPW、nnPU/SAR、MNAR sensitivity | 未实现 | 当前仓库无对应路径 |
| finite hard semantic bottleneck 到 action | 已实现 hard-ST/hard-detached/soft route | `world_action_model/models/transformer_wa_casual_mot.py:685`、`:1024`、`:1087`；production config 在 `configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py:51` |

结论：仓库目前应准确描述为 **missingness-masked + censor-encoded**，不能写成
**missingness-debiased** 或 **censor-aware likelihood trained**。

## 4. 推荐 label protocol v2

不要在 v1.1 的 segment 表上再塞一个 `missing=true`。推荐保留原始物理事件表，并把 evidence
coverage、global event 和 derived window 分成三个层次。

### 4.1 `coverage_v2`

一行只声明一个 `(episode, interval, relation_scope, side_scope, field_scope, evidence_tier)` 的
可判定范围：

| 字段 | 允许值/含义 |
|---|---|
| `relation_scope` | `hand_object` / `object_object` / `tool_object` |
| `side_scope` | `left` / `right` / `object_object`；`both` 只作便捷汇总，不取代两行 per-side coverage |
| `field_scope` | 至少拆 `presence_boundary` 与 `site_mode_role`；后者不应阻断已知 presence |
| `coverage_mode` | `exhaustive` / `positive_only` / `unknown`，沿用 v1.1 严格语义 |
| `visibility_state` | `fully_observable` / `partially_observable` / `not_observable` / `sensor_dropout` |
| `missing_reason` | `none` / `out_of_fov` / `self_occlusion` / `object_occlusion` / `motion_blur` / `transparent_interface` / `tracker_invalid` / `sensor_dropout` / `annotation_budget` / `privacy_redaction` / `other` |
| `missingness_class` | `administrative` / `MCAR_design` / `MAR_candidate` / `MNAR_possible`；后两者是分析假设，不是假装已识别的事实 |
| `evidence_tier` | T1 tactile/force；T2 calibrated multiview mesh+validated threshold；T3 multiview human temporal；T4 single-view human temporal；proxy 单列且永不成为 physical GT |
| `authorized_negative` | 仅 `coverage_mode=exhaustive` 且对应 side/relation/field 时为 true |

规范性语义：

- `exhaustive`：列出的 segment 是该 scope 的全部正例，其余帧才可作 observed negative；
- `positive_only`：只声明列出的 positive，空白处全部 unknown；
- `unknown`：不产生正负监督；
- left exhaustive + right unknown 必须可表达；不能因为左手可见就给两个手共同的空 query 负例；
- tracker validity、手部 mask 可作为 missingness covariate，但不能自动把 coverage 改为 exhaustive；
- 所有 coverage 都绑定 annotator/adjudicator、媒体/view、时钟、source hash 和 protocol version。

### 4.2 `physical_contact_segment_v2`

事件以稳定关系边为核心，而不只是 `side`：

- `segment_id`：跨所有滑窗唯一；
- `actor_entity_id`：left hand/right hand/tool/object；
- `patient_object_instance_id`：被接触实例；
- `relation_id=(actor_entity_id,patient_object_instance_id)`；
- `bimanual_group_id`：两手同时作用同一 object/assembly 的 joint event；没有证据时为空；
- `confirmed_positive_start/end`：保守、确知接触的 core interval；
- onset/release 分别记录 `boundary_status`：`observed_exact`、`observed_interval`、
  `left_censored`/`right_censored`、`unknown`；
- interval boundary 使用 `[lower_frame,upper_frame]`，exact 时 lower=upper；censored 时不伪造
  endpoint；
- `source_censor` 与 `window_censor` 分开：前者来自 episode/coverage，后者由 window builder
  确定性派生；
- 保留 side/site/object_role/contact_mode/confidence/evidence tier；object identity 未知时 mask
  identity 字段，不能用类别名假装实例 ID。

“unknown boundary”和“censored boundary”继续不是同义词：censor 表示真实边界确定在观察区间外；
unknown 表示证据不足以判定边界在何处。`interval_censored` 又不同：事件确定发生在两个已观察
时点之间。

### 4.3 `window_contact_targets_v2`

这是由前两表生成的 cache，不是独立标注：

- 固定 `(episode_id, anchor_frame, horizon, fps)`；
- 记录每个 query 对应的 global `segment_id/relation_id/bimanual_group_id`；
- `window_left/right_censored` 只由 global boundary 与窗口相交关系派生；
- 每个 side/relation/field 分别有 `loss_mask` 和 `coverage_fraction`；
- tail padding、source sensor dropout、annotation unknown 与 window censor 使用不同 mask/flag；
- 同一 physical event 的上百个窗口只是 repeated views，统计时 cluster 到 event/episode/group。

### 4.4 建议标注流程

1. 先用 T1/T2 subset 做词典校准，验证“视觉可见接触”相对 sensor truth 的 precision/recall；
2. A/B 在不知道 proposal、`hand_status`、tracker outcome 的条件下独立标 coverage，再标 events；
3. 先裁决 coverage/visibility，再裁决 events，避免用“看见了 event”倒推全段 exhaustive；
4. 自动/VLM/pose proximity 只生成 proposal，保存 proposal provenance，永不直接升 GT；
5. bimanual event 必须同时检查两个手的关系边与共同 object identity；
6. 先完成 3 段 calibration，修词典后重新独立标，不能在正式集边标边改定义；
7. test 的主指标只在 T1/T2 或高一致性 adjudicated exhaustive coverage 上计算；real unknown
   只能用于 missingness stress，不可拿来算“accuracy”。

## 5. Inter-annotator metrics 与 gate

v1.1 指标可以保留为兼容报告，但投稿级报告需要以下增加项：

| 层次 | 指标 | 推荐 gate |
|---|---|---|
| Coverage | per-side/per-field weighted Cohen κ；frame-level coverage IoU；missing-reason macro κ | κ≥0.80，95% cluster-bootstrap lower bound≥0.70 |
| Event detection | relation-compatible Hungarian matching，tIoU 阈值 {0.3,0.5,0.7} 的 F1/AUPR | F1@0.5≥0.85，lower bound≥0.80 |
| Boundary | onset/release status κ；exact boundary MAE P50/P90；interval overlap/coverage | status κ≥0.70；P50≤2 frames、P90≤5 frames |
| Segment | median/P10 tIoU，不只 median | median≥0.75，P10≥0.50 |
| Categories | side/site/role/mode 的 macro κ 和 confusion；unknown rate | 每字段≥30 comparable segments 后 κ≥0.67；不能再用 n=2 过 gate |
| Bimanual | `bimanual_group_id` pair F1、joint-object identity accuracy、overlap-duration IoU | pair F1≥0.80，且至少 50 个 joint events 后才作主 claim |
| Adjudication | unilateral event rate、coverage downgrade rate、adjudication edit distance/time | 只作质量诊断；高于 20% 必须回到词典校准 |
| Physical validity | T3/T4 人工标签相对 T1/T2 的 sensitivity/specificity、event F1、boundary error | event F1≥0.85；否则视觉标签只能降 tier |

所有 CI 以 subject/session 或最粗可得 production group 为 resampling unit；在 group 元数据补齐前，
最多按 episode cluster，且明确不能外推 OOD。frame 数和 overlapping windows 不能当独立样本数。

## 6. Split / group contract

1. 所有 views、windows、augmented copies、两个 hand segments 和同一 global physical event 必须
   进入同一 split；
2. 主 split 至少按 `subject_id + session_id` group，object-OOD 按 source/target object instance，
   strategy-OOD 再按 task family/operator；collection batch、scene/camera calibration 作 blocking；
3. 预注册三类测试：cross-subject/session、cross-object/geometry、cross-task/strategy；另有
   cross-dataset（Ego/ARCTIC/EgoTouch/robot）external test；
4. split 在 annotation/model inspection 前固定，哈希绑定；若需要 3 seeds，只改变 group partition，
   不能改变 test definition；
5. op29 的 0–40/41–49/50–65 只能标 `temporal_diagnostic_only`，不能进入主结果；
6. 若 66 段知情补录后所有 group fields 实际都相同，则该数据仍只能作为 within-group pilot，
   不能靠完整 metadata 把一个 group 变成 OOD benchmark；
7. synthetic missingness masks 只能从 full-label split 内派生；同一原始 event 的多种 mask realization
   不得跨 split，也不增加独立样本量。

## 7. Loss 与 assignment 设计

### 7.1 必须分开的三类 loss

1. **observed field loss**：沿用当前 field mask；presence 只有 exhaustive coverage 能提供 negative；
2. **event-time likelihood**：把 onset 和 release 建成两个 conditional discrete hazards，分别只在
   no-contact/contact risk set 内训练；
3. **missingness model/sensitivity**：估计 annotation/visibility propensity，或在不可识别时报告
   敏感性区间；不与 contact head 偷偷共享未来标签。

离散 hazard `h_k` 的基本项：

- exact event at bin `k`：`-sum_{j<k} log(1-h_j) - log(h_k)`；
- right-censored at `c`：`-sum_{j<=c} log(1-h_j)`；
- interval-censored in `(l,r]`：`-log(S(l)-S(r))`；
- window 开始已经 contact：不回归伪造的 onset=0；用 current-state head + conditional release
  hazard。若有窗口前 context，可对 `T_onset<=0` 的概率作 left-censor likelihood；否则 onset loss mask。

这比当前“start_tau=0 + left_censored BCE + Smooth-L1”更符合统计语义，也避免模型为了降低
回归误差把所有被截断 onset 堆在窗口边界。

### 7.2 Missingness-aware 方案

- **M0 naive-zero**：unknown 当 negative，仅作为预注册错误 baseline；
- **M1 current masked**：现有 field-mask/coverage baseline；
- **M2 nnPU**：仅在可论证 positive-unlabeled + class prior 的 presence 子任务使用；报告 SCAR
  assumption，不套到 boundary/category；
- **M3 SAR/IPW**：用仅在 label decision 时可观测的 view/occlusion/blur/tracker/task covariates
  估计 propensity；cross-fit、stabilized/clipped weight，报告 positivity 和 effective sample size；
- **M4 augmented IPW/doubly robust**：propensity 与 outcome 两模型，任何一方正确时减偏；
- **M5 MNAR sensitivity**：对未观测 positive odds 设 `delta`/selection-bias grid，报告结论何时
  翻转。没有随机 audit/tactile subset 时，M5 是主结果，不能把 M3 写成已解决 MNAR；
- **oracle-full**：只在 fully labeled corruption source 上作上界。

动作 BC/flow loss不能因为 contact label unknown 而被整体屏蔽；只按 action 自身 validity（例如
op28 的 grip proxy tracker validity）处理。反过来，也不能拿 `hand_status` 填 contact label。

### 7.3 Set assignment

当前 chronological ordering 保留为 B0。主 baseline 应实现 mask/censor-compatible Hungarian：

- presence cost 只使用有 coverage 的 query；
- temporal cost 对 exact boundary 用 NLL/MAE，对 interval boundary 用预测落入可行区间的质量，
  对 censor 用 survival mass；
- relation/side/object identity 不兼容时禁止匹配；unknown category 不产生 cost；
- bimanual group 先在 relation-edge 层匹配，再评 joint group，避免两只手被交换后仍得高分；
- 固定 tie-break 和 provenance；训练 assignment 与评测 assignment 分开实现、分别测试。

## 8. Benchmark tasks、metrics 与 baselines

### 8.1 Tasks

1. **Per-relation contact state**：在 exhaustive/sensor GT 上预测 left/right/object-object state；
2. **Onset/release time-to-event**：任意 anchor 的 conditional hazard、censor-aware NLL、
   time-dependent Brier/CRPS、calibration；
3. **Horizon event-set forecast**：event F1@tIoU、boundary error、category macro-F1、empty-window
   specificity；
4. **Bimanual coordination**：两手是否共同作用同一物体、joint onset/overlap/release、role switch；
5. **Missingness robustness**：从 T1/T2/full adjudicated GT 生成 MCAR block、side-dependent、
   visibility-SAR、contact-dependent-MNAR 四族 mask，各 3 severity；
6. **Cross-view inference**：ego only、ego+wrist、ego+exo、view dropout；view 不可用必须走同一模型
   contract；
7. **Cross-dataset transfer**：100DOH/VISOR/Ego4D 只做其原生 label-space 的辅助/外测，
   ARCTIC/EgoTouch/robot sensor subset 才做 physical event 主测；
8. **Downstream WAM**：同 backbone、同 batch/RNG/noise 的 Q0、current masked、censor-aware、
   MNAR-aware；报告 success、correct-first-contact、wrong-contact、spill/collision、phase order、
   action-only latency，而非只报 contact loss。

### 8.2 Baselines

- naive missing-as-negative BCE；
- current deterministic-order masked grouped loss；
- masked loss + Hungarian；
- discrete hazard NLL；Survival-CRPS；
- nnPU、SAR/IPW、augmented IPW、MNAR sensitivity；
- 100DOH contact-state detector、EgoHOS/VISOR hand-object segmentation、EgoLoc boundary
  localization；
- pose/mesh proximity（ARCTIC/HOT3D）与 tactile oracle（EgoTouch/robot）；这些必须标 proxy/oracle；
- WAM：Q0 direct、aux-only、soft finite contact、hard-ST、hard-detached、oracle contact、
  shuffled/delayed condition、matched-bit nonsemantic code；
- robot imitation：BC/RoboMimic-style recurrent policy、ACT/transformer policy，条件允许时在
  BiGym/LIBERO 接触敏感任务上做 simulator external validation。

## 9. Statistical power 与样本量

### 9.1 当前数据为什么不够

op17 的 12 episodes/3,695 frames 只适合协议校准。overlapping windows 不增加独立 event 数；
当前 event 数甚至尚未知。66 episodes 也只有一个任务且 production group 未知。任何用 frame-level
`N=3,695` 做显著性检验的结果都会严重伪重复。

### 9.2 预注册功效基准

- 对 event recall/precision 约 0.80，若希望简单 binomial 95% CI 半宽约 0.05，需要约 246 个
  独立 events；若每 episode 平均 10 events、intra-episode correlation ρ=0.1，design effect
  `1+(10-1)ρ=1.9`，需要约 468 event-equivalents（约 47 此类 episodes），实际再按 group
  imbalance 上调；
- 由于 categories、side、censor strata 需要分别报告，独立论文的最低目标应为 ≥300 个
  adjudicated physical segments、每个核心类别≥30 comparable、≥50 个 bimanual joint events，
  且来自 ≥30 个 production groups；这只是最低统计门槛，不是数据规模卖点；
- synthetic masks 的 5 次重复不能把 300 events 变成 1,500 独立 events；用 hierarchical
  bootstrap：先 group/episode，再 event，再 mask realization；
- 对 rollout success 从 0.40 提升到 0.55，双侧 α=0.05、power=0.80 的独立两比例近似需要
  173 rollouts/model；若只提升 10 percentage points（0.40→0.50），约需 388/model。
  paired/McNemar 设计可因 discordant-pair rate 降低样本量，但必须先用 pilot 估计，不能默认
  “20 rollouts 足够”；
- 所有主比较报告 group-cluster bootstrap 95% CI 和 paired effect；多 missingness severity/
  split 使用预注册 hierarchical model 或 Holm correction；seeds 不是独立任务组。

独立论文建议的设计下限：≥3 task families、≥10 subjects、≥10 object instances、≥30
subject-session groups、≥300 events、≥50 bimanual joint events；至少 20% events 或 100 events
有 T1/T2 sensor/geometry truth。若做 dataset paper，考虑 TouchAnything 的规模，最好再有跨数据集
protocol adapter 和公开 evaluation server，而不是只发布当前单任务 CSV。

## 10. 分阶段成功/停止门槛

### Gate D0：标签可行性（只用现有 12 段）

**通过：** 完成真实 A/B+裁决；coverage κ≥0.80、event F1≥0.85、median tIoU≥0.75、boundary
P50≤2/P90≤5 frames；per-side unknown/positive-only/exhaustive 和 bimanual event 数可报告。

**停止：** 任一主 gate 失败；或 >40% person-time 在两个手上都 unknown；或 T3/T4 相对 sensor
subset event F1<0.85。先修词典/证据，不训练。

### Gate D1：missingness 确实造成偏差

**通过：** 在 fully labeled corruption source 的 visibility-SAR/contact-MNAR regime 中，
naive-zero 相对 current masked 至少差 3 event-F1 points 或 calibration/Brier 恶化≥10% relative，
且 side-specific bias 方向跨两个 task/group strata 一致。

**停止/降级：** 差异 <2 points 且 calibration/downstream 均无差异；把 D 降为数据卫生 protocol，
不作主贡献。

### Gate D2：censor/MNAR 方法增益

**通过：** censor-aware hazard 相对最强 masked+Hungarian baseline 在 censored NLL/IBS 至少改善
10% relative 或 event F1≥+3 points，group-bootstrap 95% CI 排除 0，并跨两个 missingness regimes；
MNAR-aware 相对 censor-only 再提升≥2 points，propensity ESS≥原 observed 样本的 30%，主 strata
没有严重 positivity violation。

**停止/改 claim：** propensity 极端、ESS<30%、结果只在一个人为 mask 上成立，或 sensitivity
analysis 在合理 delta 下立即翻转；删除“MNAR debiasing”，只保留 sensitivity-aware evaluation。

### Gate D3：WAM downstream

**通过：** 继承 finite-contact 主线门槛：oracle contact 相对 Q0 的 part-sensitive success≥+10pp、
correct-first-contact≥+15pp；predicted 至少恢复 oracle gain 的 70%。此外，missingness-aware 相对
current masked 在预注册 group split 的 rollout success≥+5pp 或 wrong-contact rate 降低≥20%
relative，95% CI 排除 0；shuffled/delayed contact 产生预期 phase-specific 退化。

**停止/并 appendix：** 只改善 label NLL/event F1而不改善 contact-sensitive rollout；或
arbitrary matched-bit code 与 contact semantics 等价。

### 独立论文晋升 Gate

只有同时满足以下条件才从 finite-contact 主线拆出：

1. D0–D3 全部通过；
2. production group metadata 完整且 leakage=0；
3. ≥3 task families、≥30 groups、≥300 events、≥50 bimanual events；
4. T1/T2 physical truth subset 达到前述质量门槛；
5. 至少两个数据域（例如 Ego+ARCTIC/EgoTouch 或 Ego+robot tactile）复现；
6. 发布 protocol、adapters、fixed corruptions、evaluation code 和 CI；
7. 相对 TouchAnything/EgoLoc/ARCTIC 的区别由 missingness+censor+downstream 证据支持，而不是
   仅靠新字段名。

若任一不满足，最终定位应是 finite-contact 论文中的“Censored/Missingness-aware supervision
and benchmark protocol”主贡献之一，而非独立投稿。

## 11. 最短执行顺序

1. 不训练；先补 op29 的真实 group attestation，并确认 12 段的授权/隔离标注可执行；
2. 把 v1.1 coverage 扩为 per-side/per-field v2，同时保留向后兼容导出；
3. 做 3 段 A/B calibration，再完成 12 段 D0；
4. 在 T1/T2 或 ARCTIC/EgoTouch 可合法使用 subset 上验证 visual annotation 与 missingness masks；
5. 先做 CPU/offline 的 naive-zero、current masked、Hungarian、hazard NLL 小实验；
6. 只有 D1/D2 通过才进入 WAM pilot；D3 通过后才讨论独立论文扩量。

## 12. 最终 Go / No-Go

- **研究方向：Go。** 缺失不当负例、窗口删失与双手非对称 coverage 是 finite-contact WAM
  必须解决的科学有效性问题。
- **纳入 finite-contact 主线：Conditional Go。** 现有 v1.1 是良好起点，优先完成真实标注、
  per-side coverage 和 censor-aware likelihood。
- **当前 scientific training：No-Go。** op17 无 GT，op29 无 production group，训练不能产生
  可发表结论。
- **当前 standalone paper：No-Go。** 数据规模、物理真值、方法和 downstream 证据都未达到
  独立门槛；TouchAnything/EgoLoc/ARCTIC 已覆盖相邻核心任务。
- **hand_status as physical contact GT：永久 No-Go。** 它是 open/closed 动作代理，任何训练或
  论文表格都不得把它改名为 contact truth。
