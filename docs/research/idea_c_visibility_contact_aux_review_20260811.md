# Idea C：visibility-gated boundary/rim/liquid/stream/contact auxiliary 与 censored contact events

> 日期：2026-08-11  
> 分支：`codex/idea-c-visibility-contact-aux-20260811`  
> 基线提交：`d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88`  
> idea_id：`5757566f-4c34-4cf1-aecf-0dfac76a8ab9`  
> discussion_id（最近一次）：`e1e27a11-cb4b-4ac8-bef8-28dd3e51f634`  
> 初次 discussion_id：`7ef28570-2d28-45c8-8b9b-8e9d74fcb3cf`  
> 决策：**训练 / 论文主张 No-Go；仅“正式标注审计 + oracle ceiling 设计”Conditional Go。**

## 1. 一句话结论

不要从当前 ego RGB-D 为透明杯和透明液体制造伪 3D，也不要把 depth completion、inpainting、composite、diffusion 或弱分割当 GT。当前 66 集最现实的信号是：**杯体边界、source/target rim 在可见时的 2D 几何，以及带“存在性—可观测性—时间删失”三者分离的接触事件区间**。它们只能作为训练期 auxiliary；动作在推理时仍只能读取现有 finite contact grammar，不能读取连续 mask、heatmap、raw planner hidden 或人工标签。

这个方向单独看不够新：透明深度、弱监督 liquid/flow、手物接触、visibility head、TTC/contact-state 和 semantic-mask world model 都已有工作。它与主 finite-contact WAM 形成有说服力贡献的唯一方式是同时做到：

1. 给出不伪造不可见真值的 observation contract；
2. 用可验证的信息流约束证明 spatial auxiliary 没有绕开 22-D finite grammar；
3. 在 matched control 中证明辅助监督对 first contact、rim alignment、spill/collision 有独立增益；
4. 分开报告真实数据可标注性、sim native oracle ceiling 和 sim-to-real，绝不相互替代。

当前数据不支持 dense stream/contact-interface head，也没有 reviewed contact GT；因此现在启动 WAM 训练会先验证标签偏差而不是研究假设。

## 2. 调研方法与证据边界

本次先按 `idea-discussion` 的 Zotero-first 流程同步并检索个人库，再用 arXiv、CVF、PMLR、OpenReview、官方数据集和模拟器文档核对原文。idea memory 已更新两次；Zotero 初检对本题召回偏泛，因此又直接核查本地库的透明物体、world action model、egocentric HOI/contact 条目。与本题最接近的库内记录包括：

- `726MZ5TZ`：GigaWorld-Policy-0.5；
- `QDHRRU56`：FlowWAM；
- `XINDFWEJ`：Being-H0.7；
- `V2P9FFKM`：MotionWAM；
- `JRGPD6HS`：OmniContact；
- `C25N2EGS`：Diffusion Knows Transparency；
- `GPWIUHVW`：egocentric 3D hand-object pose；
- `EY2ZN3YX`：EgoAERO；
- `GRSBFCQ9`：zero-shot temporal interaction localization。

本地 `arxiv-search` 包装器被调用过，但当前环境缺少其 Python `arxiv` 依赖，且未确认应该写入哪个项目环境；本次没有向 base 或任何 conda 环境安装依赖，改用 arXiv 官方摘要页与论文官方页面核对。文献表中的 2026 工作若尚无正式 venue 页面，明确按 preprint 对待。

本 memo 只做只读研究、CPU 元数据/深度统计和设计；没有训练、没有 GPU、没有启动模拟器，也没有改 canonical benchmark。

## 3. 文献地图：哪些已经有，哪里还可能有贡献

### 3.1 透明物体与透明深度：boundary 是老信号，伪 3D 不是新贡献

| 工作 | 已经证明的能力 | 对本 idea 的约束 |
|---|---|---|
| [ClearGrasp, ICRA 2020](https://arxiv.org/abs/1910.02550) | 从单 RGB-D 预测透明 mask、表面法向和 occlusion boundary，再做深度补全；合成训练并有真实抓取评估 | boundary auxiliary 有直接先例；若最终主张仍是 dense depth completion，贡献会落回拥挤赛道 |
| [RGB-D Local Implicit Function, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_RGB-D_Local_Implicit_Function_for_Depth_Completion_of_Transparent_Objects_CVPR_2021_paper.html) | 局部隐式函数补透明物体深度 | 说明“先补深度再控制”是明确 baseline，不应被当成 GT 管线 |
| [TransCG](https://arxiv.org/abs/2202.08471) | 57,715 个真实 RGB-D 样本、130 个场景，并有专门采集的深度/法线/mask 标注 | 可靠透明深度依赖专门真值装置；当前 66 集不具备同等监督条件 |
| [ClearPose, ECCV 2022](https://arxiv.org/abs/2203.03890) / [官方仓库](https://github.com/opipari/ClearPose) | 大规模透明物体 pose/depth/segmentation 数据，含带液体情形 | 其 GT depth 使用人工 3D 模型、pose 与渲染流程；不能把本项目的 observation 修补图冒充同类 GT |
| [DREDS, ECCV 2022](https://pku-epic.github.io/DREDS/) | domain randomized sim-to-real transparent depth | 是 synthetic/depth baseline，不是当前真实 episode 的 ground truth |
| [Consistent Depth Prediction, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Consistent_Depth_Prediction_for_Transparent_Object_Reconstruction_from_RGB-D_Camera_ICCV_2023_paper.html) | 多视角一致透明重建 | 多视角一致性是高成本 baseline；单目 ego 片段不能默认获得可信 3D |
| [Confidence-aware single/multi-view transparent depth, 2024](https://arxiv.org/abs/2405.15299) | 在透明深度里显式融合置信度与多视角 | uncertainty 已有前例；本题的差异必须是“何时不监督/不输出”的 observation contract 和策略作用，而非普通 confidence head |
| [Dex-NeRF](https://arxiv.org/abs/2110.14217) | 多相机与可控照明下重建透明物体并支持抓取 | 说明透明场景能做，但硬件条件与本项目单 ego RGB-D 不同 |

结论：杯体 boundary、rim、visibility/confidence 都不是独立新意。可以贡献的是主动拒绝不可观测 3D，并证明一个低带宽、训练期、visibility-gated 的 2D 辅助信号确实改善 finite-contact policy。

### 3.2 液体、流束与倾倒：弱分割和闭环 pouring 已经存在

| 工作 | 已有内容 | 与本题的关系 |
|---|---|---|
| [Visual Closed-Loop Control for Pouring Liquids, ICRA 2017](https://arxiv.org/abs/1610.02610) | 视觉估计液量并闭环控制倾倒 | “视觉 liquid feedback 改善 pouring”不是新主张 |
| [Adaptable Pouring, CoRL 2017](https://proceedings.mlr.press/v78/lopez-guevara17a.html) | 近似 fluid simulation、校准和迁移 | 是 sim calibration baseline，也提醒 oracle 物理不能直接代表真实透明液体 |
| [SPNets, CoRL 2018](https://proceedings.mlr.press/v87/schenck18a/schenck18a.pdf) | 可微粒子网络结合视觉与控制 | 粒子 latent/流体动力学路线已有先例 |
| [Accurate Pouring with RGB-D](https://arxiv.org/abs/1810.03303) | 追踪液面并评估大量 pours | liquid level 是强 baseline，但依赖其传感条件 |
| [Self-supervised Transparent Liquid Segmentation](https://arxiv.org/abs/2203.01538) | 用着色液体/背景差分获得标签，再以非配对翻译处理透明液体 | 与本题非常接近；着色、translation 和 synthesis 只能做预训练或 baseline，不能变成真实透明 GT |
| [PourIt!](https://arxiv.org/abs/2307.11299) / [项目页](https://hetolin.github.io/PourIt/) | image-level label、CAM flow region、container pose 和 3D point cloud 用于闭环 pouring | 弱监督流束/flow region 直接撞题；若 C 只做 stream CAM，不足以构成贡献 |
| [Transparent-container liquid manipulation, 2024](https://arxiv.org/abs/2404.16529) | 透明容器液量视觉与 simulation pouring | 可作为透明小开口任务 baseline |
| [Multigen, CoRL 2025](https://proceedings.mlr.press/v305/wang25a.html) | audiovisual sim-to-real pouring | 提醒声音/力/触觉是未来可观测性替代，但当前数据没有这些同步真值 |

当前 100 帧审计里 `stream` 与 `contact interface` 的可局部化率都是 0%。因此 C 不应以 dense stream/contact mask 开局。更可守住的目标是：在杯口/边界可见时监督几何；对 liquid/stream 只监督“存在/不可局部化”和时间事件区间，直到正式双标证明像素级标签可靠。

### 3.3 HOI、手姿态与接触变化：contact state/TTC/boundary 也已有前例

- [100DOH, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.html) 已预测手、左右侧、接触状态与接触物体框。
- [Ego-OMG](https://arxiv.org/abs/2006.03201) 用 hand-object contact 的建立/断开定义交互状态。
- [Forecasting through Contact](https://arxiv.org/abs/2102.00649) 已做 time-to-contact、手/物 mask 和 future contact map。
- [VISOR](https://arxiv.org/abs/2209.13064) 提供大规模 egocentric hand/object mask 与手物关系；插值 mask 仍需和 manual label 分开报告。
- [EgoHOS](https://arxiv.org/abs/2208.03826) 有详细 hand-object segmentation/contact boundary。
- [Interaction Hotspots, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Joint_Hand_Motion_and_Interaction_Hotspots_Prediction_From_Egocentric_Videos_CVPR_2022_paper.html) 与 [Human Hands as Probes, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Goyal_Human_Hands_As_Probes_for_Interactive_Object_Understanding_CVPR_2022_paper.html) 已把手运动、接触位置和物体可交互性联系起来。
- [WACV 2024 hand-object contact/object-state auxiliary](https://openaccess.thecvf.com/content/WACV2024/html/Shiota_Egocentric_Action_Recognition_by_Capturing_Hand-Object_Contact_and_Object_State_WACV_2024_paper.html) 表明 contact/object state 可作为 action-recognition 辅助。
- [Ego4D FHO annotation guideline](https://ego4d-data.org/docs/data/annotation-guidelines/) 明确区分 PRE、CONTACT、PNR、POST；片段开始时物体已在手里可能根本没有可见 CONTACT。这直接支持 left censor，而不是补造 onset。
- [WACV 2024 visibility-aware pose](https://openaccess.thecvf.com/content/WACV2024/html/Sun_Rethinking_Visibility_in_Human_Pose_Estimation_Occluded_Pose_Reasoning_via_WACV_2024_paper.html) 说明显式 visibility reasoning 本身也不是新贡献。

因此不能把 hand tracker 的 valid/active、手和杯子的二维重叠或 grasp proxy 叫“真实接触”。本题能留下的空间是：让 finite contact event 标签明确携带 temporal censor 与 observation visibility，并验证这种诚实的缺失机制比强行二值化更有用。

### 3.4 World model / VLA auxiliary：mask、affordance、future latent 都已很拥挤

下列工作说明“加一个空间辅助头”本身不够：

- [CoA-VLA, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.html)：object/grasp/spatial/movement affordance chain；
- [ToolFlowNet, CoRL 2022](https://openreview.net/forum?id=2gfB_kMVFvP)：tool point flow，含 pouring/scooping 类任务；
- [Flow as a Foundation Affordance, CoRL 2024](https://openreview.net/forum?id=nmEt0ci8hi)：用 flow 表达可供性；
- [Being-H0.7, 2026 preprint](https://arxiv.org/abs/2605.00078)：训练期 future posterior 对齐 current prior，且明确讨论 action shortcut；
- [Mask World Model, 2026 preprint](https://arxiv.org/abs/2604.19683)：用 semantic mask 代替像素 world prediction，并建模 contact relations；
- [SelfWAM, 2026 preprint](https://arxiv.org/abs/2608.00725)：action-conditioned future RGB 与 self-mask auxiliary；
- [Tactile-WAM, 2026 preprint](https://arxiv.org/abs/2606.26663)：future tactile state 和 asymmetric mask；
- [ST-VLA, 2026 preprint](https://arxiv.org/abs/2603.13788)、[VLAff, 2026 preprint](https://arxiv.org/abs/2608.05215)、[AffordanceVLA, 2026 preprint](https://arxiv.org/abs/2606.06155)：轨迹、空间 mask、grasp/affordance 都会成为直接 baseline。

### 3.5 可辩护的新颖性边界

| 可能主张 | 判定 | 原因 |
|---|---|---|
| 透明 boundary/rim 或 uncertainty head | 已有 / 弱 | ClearGrasp、confidence-aware transparent depth、visibility-aware pose 已覆盖构件 |
| liquid/stream segmentation | 已有 / 当前数据不支持 | 自监督透明液体分割、PourIt! 已覆盖；当前真实透明流束不可定位 |
| hand-object contact/TTC 辅助 | 已有 | 100DOH、Forecasting through Contact、EgoHOS、WACV 2024 均为直接前例 |
| semantic mask/affordance world model | 已有且近期拥挤 | Mask World Model、SelfWAM、CoA-VLA、VLAff 等直接碰撞 |
| **observation contract + censored event + strict finite grammar 的因果隔离** | **可能成为 supporting contribution** | 需要同时证明不可观测标签不进 loss、连续空间预测不进 action、matched oracle 有独立增益 |
| **真实可标注性审计 + same-state native sim oracle 的双层 benchmark** | **可能成为证据贡献** | 必须保持 real annotatability 与 sim physics ceiling 两种 claim 严格分离 |

## 4. 当前 66 episode 到底能标什么

### 4.1 数据规模与 split 限制

当前规范化证据记录的是 66 episodes、20,013 个 processed 20 Hz frames、99,953 个 raw 100 Hz frames，动作 horizon 为 48。可是 subject、session、operator、object instance、liquid 和 collection batch 等 group metadata 都缺失；已知内容基本只有同一 task/date/camera calibration。位置见：

- `evidence/ego_wam_norm_v3.json`；
- `docs/audits/ego_66ep_group_metadata_recovery_20260811.md`。

这意味着当前可以做 episode-held-out，但不能声称有效的人、器皿、液体或采集批次 OOD。任何“透明到不透明”“着色到透明”“新容器”泛化都必须先恢复 group metadata 或重采集。

### 4.2 Track-B 100 帧 audit 的真实结论

`track_b/visibility_audit_v0` 从 episode `0, 29, 58, 30, 14` 各取 20 帧，做了 phase-stratified 100-frame prescreen。分数定义为：0 absent，1 present/expected but unlocalizable，2 partial，3 clear。当前只是单人 prescreen，正式 A/B ±2-frame 表仍为空，不能当 GT。

| 字段 | 0 | 1 | 2 | 3 | 可局部化（2+3） | 结论 |
|---|---:|---:|---:|---:|---:|---|
| container boundary | 0 | 30 | 70 | 0 | 70% | 值得正式双标，但多为 partial |
| source rim | 0 | 30 | 19 | 51 | 70% | 值得正式双标 |
| target rim | 0 | 0 | 21 | 79 | 100% | 最强现实辅助信号 |
| liquid surface | 0 | 80 | 0 | 20 | 20% | 20 帧全部来自着色 liquid 的 ep58，不能代表透明 liquid |
| stream | 74 | 26 | 0 | 0 | 0% | 禁止 dense real-GT head |
| liquid/contact interface | 51 | 49 | 0 | 0 | 0% | 禁止 dense real-GT head |

审计文件位置：

- `track_b/visibility_audit_v0/README.md`；
- `track_b/visibility_audit_v0/prescreen_summary.json`；
- `track_b/visibility_audit_v0/formal_dual_annotation_v0/`。

可标注性判断：

- **现在可试标**：target rim、source rim、container boundary；用 polyline/ellipse/edge band，不要求补全遮挡部分。
- **只做 presence/visibility 或 interval event**：liquid surface、stream existence、pour onset/offset；geometry loss 必须关闭。
- **不可作为视觉真值**：真实 hand-liquid、tool-liquid 或 liquid-vessel contact point/interface；只能由 native sim、力/触觉或专门传感器提供。
- **手物接触**：可标“视觉上确认的接触区间”与 left/right/interval censor；不能把几何邻近强制变成物理接触。

### 4.3 raw depth 孔洞：全图确实严重，但尚不能归因于透明杯

本次只读、CPU 地读取这 100 个被选帧对应的原始 `depth_raw`（`uint16, 480×640`），统计全图零值比例：

- 总体均值 11.24%，中位数 10.99%，P95 13.38%；
- 最小 9.28%，最大 14.34%；
- episode 0 / 14 / 29 / 30 / 58 的均值分别为 11.73% / 9.99% / 12.22% / 10.93% / 11.32%。

这个统计只证明 raw depth 在整幅图像上有 9.3%–14.3% invalid pixels。它**没有 vessel ROI、透明/不透明配对、投影校准或专门 GT**，所以不能声称这些孔洞由透明器皿造成，更不能用 completion 后的值评估 completion 本身。

下一步合法的 depth audit 必须让 A/B 在 RGB 上独立标 rim、可见杯壁、杯内区域与背景 control ROI，再报告每类 ROI 的 zero/saturation/discontinuity；标签依旧来自 RGB 人工直接证据，depth completion 只作为 predictor/baseline。若无透明/不透明同实例配对，结论只能是 sensor invalidity audit，不是 transparency causality。

### 4.4 contact 真值目前为空

`track_a/contact_labels_v1_1/formal_dual_annotation_v1_1` 覆盖设计上要求的 12 episodes、3,695 processed frames、184.75 s；但 annotator A、annotator B 与 adjudication segment CSV 目前都只有 header。当前 README 的正确状态是 **No-Go / not GT**。

手跟踪 proxy 也不能顶替它：20,013 帧中 left current-active 7,087、right 6,574、both-active 仅 165、neither 6,517；proxy transition 虽分别有 110/118 个，但前后相邻帧都有效的只有 54/58 个。很多 episode 对某只手完全无有效值。它最多是候选窗口采样器，不是 hand-object contact label。

## 5. 标签 contract：把“不可见”与“删失”彻底分开

建议单独建立 `visibility_aux_v0` sidecar；不要把空间字段塞进现有 `contact_segment_v1_1` 的 22-D action-visible schema。

### 5.1 每个 field、frame/horizon 的最小 schema

```text
sample_id, episode_id, frame_index, horizon_index
field_name: container_boundary | source_rim | target_rim |
            liquid_surface | stream | contact_interface
existence: absent | present | unknown | not_applicable
observability: clear | partial | present_unlocalizable |
               out_of_frame | sensor_invalid | unknown
geometry_type: none | polyline | ellipse | mask | point_set
geometry: <仅 clear/partial 时允许非空>
geometry_support: visible_only
evidence_source: human_direct_rgb | human_temporal_rgb | sim_native_state |
                 same_state_native_render
annotator_id, confidence, adjudication_status
source_artifact_sha256, schema_sha256
```

硬规则：

1. `absent` 可以训练 existence/status，geometry loss mask 必须为 0；
2. `present_unlocalizable` 只能训练 observability/status，不能给全零 mask；
3. `unknown` 所有任务 loss mask 为 0，不能映射成 negative；
4. `partial` 只监督看见的曲线/点，不补画遮挡段；
5. `inpaint/composite/diffusion/image_edit/manual_paint/pseudo_label/reconstruction_as_gt` 永远不能出现在 GT `evidence_source`；
6. frozen teacher、PourIt!-style CAM、depth completion 只能生成 proposal 或 baseline prediction，必须与 human/native GT 分库存储。

当前 audit 的 0/1/2/3 可兼容映射为：0→absent，1→present+unlocalizable，2→present+partial，3→present+clear；正式版本必须另有 `unknown/out_of_frame/sensor_invalid`，避免把原因混在一个序数里。

### 5.2 event schema 与 temporal censor

```text
event_type: hand_object_onset | hand_object_release | rim_entry |
            pour_onset | pour_offset | spill_onset
time_support: exact | interval | left_censored | right_censored | unknown
t_lo, t_hi
pre_state_observed, post_state_observed
event_observability: clear | partial | unobservable | unknown
physical_truth_source: human_visual_interval | sim_native_contact |
                       sim_native_rim_flux
```

这里的两个概念不能混：

- **visibility missingness**：事件可能发生，但图像无法定位；对应 `unknown/unobservable`，不应强加时刻 loss；
- **temporal censor**：能确认状态，但真边界在 review window 外。left-censored 表示窗口开始前已进入该状态；right-censored 表示窗口结束时仍未观察到终止。

离散 hazard loss 可对 observed interval 累加区间内事件概率；right censor 只最大化截止窗口末仍未发生的生存概率；left censor 监督初始 active state，不伪造 onset bin。`unknown` 全部 mask 掉。现有 contact schema 的 censor 语义应保持不变；若加入 interval censor，必须升 schema 版本，不能静默改 22-D grammar。

## 6. 不依赖重建的模型修改

### 6.1 推荐结构

以现有 strict finite-contact WAM 为主干，添加一个**训练期可见、推理动作不可见**的小型 decoder：

1. 从 reference visual tokens、state/text context 构造 6 类 field queries × 少量 forecast horizons；
2. status head 预测 existence + observability；
3. 只对 boundary/source rim/target rim 的 clear/partial 样本输出低分辨率 edge heatmap、polyline distance field 或 ellipse parameters；
4. liquid/stream/contact-interface 在 Gate 0 之前只做 status，不做 geometry；
5. contact/pour event head 输出离散 hazard 与 interval/censor likelihood；
6. auxiliary prediction 和 target **都不拼回 action stream**。动作仍只得到 12 segments × 22-D 的 hard finite grammar。

总损失建议为：

```text
L = L_visual + L_action + lambda_contact * L_finite_contact
  + lambda_status * L_existence_observability
  + lambda_geom * I_localizable * L_visible_geometry
  + lambda_event * L_interval_censored_hazard
```

`I_localizable` 由人工/native 标签 contract 决定，不能由模型置信度自行打开；预测 visibility 可用于推理输出校准，但不能回写训练 mask。为防 shortcut，至少加入以下自动不变量：

- 同一 checkpoint 下交换/删除 aux target 不得改变 forward action prediction；
- action attention keys 中不出现 aux logits、heatmap、decoder hidden 或 GT mask；
- strict 运行 receipt 仍显示唯一 action-visible 视觉桥为 transported hard 22-D contact semantics；
- `aux_only` 与 `action_visible_spatial` 分成不同 variant，后者只作非部署 upper bound，参数量/训练步数配平。

### 6.2 与 finite-contact WAM 的互补关系

finite grammar 负责“谁与谁、哪个 functional site、哪种 contact mode、在哪个粗时间 bin”；auxiliary 只让共享视觉表征学会“当前证据是否足够、可见 rim/边界在哪里、事件时刻是否只在一个区间内”。它不承担连续 3D 几何，也不扩大动作通道的信息容量。

如果最终必须把 heatmap 喂给动作才能获得增益，则本 idea 与 strict finite-contact 的核心论点冲突，应停止或降级为普通 spatial-affordance baseline。

## 7. 代码插入位置（仅设计，本分支未实现）

### 7.1 模型

- `world_action_model/models/transformer_wa_casual_mot.py:590`：在 transformer 初始化参数中加入 opt-in `visibility_aux_*` 配置；现有 contact head 初始化集中在约 `:685-803`。
- `world_action_model/models/transformer_wa_casual_mot.py:1096`：`plan_contact_semantic()` 当前从 state/ref/text 得到 `planner_hidden`，并声明 raw hidden 不离开方法。可在此内部调用独立 aux query/decoder，只返回 aux logits；transported 22-D 仍是动作唯一输入。若需要像素拓扑，decoder 可读取 `ref_visual_hidden_states` 与 grid metadata，但其输出禁止进入 `encode_action_tokens()`。
- `world_action_model/models/transformer_wa_casual_mot.py:1297`：`encode_joint_streams()` 目前返回 contact logits/semantic；可增加结构化 `aux_pred`，同时保存 action-visible route 的独立审计字段。
- `world_action_model/models/transformer_wa_casual_mot.py:1571`：strict mask 通过 `action_attends_ref=False` 等约束直接视觉路径；新增分支不得改变这些参数。
- `world_action_model/models/transformer_wa_casual_mot.py:1598`：输出打包处可增加 training-only `aux_pred`；不可把它拼到 `action_hidden_states`。

### 7.2 数据、变换、loss 与配置

- `world_action_model/datasets/wam_lerobot_dataset.py:711`：在 `_get_data()` 增加独立 `_load_visibility_sidecar()`；仿照 `:741-819` 的 contact sidecar 做 fail-closed SHA-256、schema、dtype、frame count 校验，但不要复用 `contact_*` namespace。
- `world_action_model/transforms/wa_transforms_lerobot_pretrain.py:500` 与 `:983`：把 status/event 字段原样透传；所有 geometry 必须应用与 RGB 完全相同的 crop/resize/flip 参数，并在 augmentation trace 中绑定 hash。不能像现有 geometry-free contact tensor 那样直接跳过空间变换。
- `world_action_model/trainer/wa_casual_trainer_mot.py:897`：读取 aux targets/masks；`model_output` 解包在 `:915-931`；在现有 contact loss `:962-978` 后计算 masked status/geometry/censored-event loss。empty/unknown batch 返回可审计的 zero contribution，而不是全零负标签。
- `configs/giga_world_policy_0_5_ego_contact_pretrain.py:39-64` 与 `:131-174`：增加非 production 的 opt-in dataset/model/loss 配置。
- `configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py:151-166`：未来只有通过 preflight 才能加入 aux；`:193-199` 的 action-visible route 文字和实测 contract 必须保持不变，并加入 visibility sidecar hashes。

不建议在验证假设前更改这些文件；本次只新增研究 memo。

## 8. Oracle ceiling 与 benchmark

### 8.1 真实离线 oracle 层级

先在小模型/冻结 backbone 上做下列 ceiling，不启动 5B WAM：

| 层级 | 训练输入/监督 | 测试时 action 可见 | 回答的问题 |
|---|---|---|---|
| R0 finite-contact base | reviewed finite contact | hard 22-D grammar | 主基线 |
| R1 oracle visibility mask | 人工 status 决定哪些 geometry loss 生效 | 仍只有 hard 22-D | “诚实地不监督”是否有价值 |
| R2 oracle visible geometry aux | 人工 visible-only boundary/rim | 仍只有 hard 22-D | boundary/rim 表征的训练期上界 |
| R3 predicted gate + human geometry | 预测 status，human geometry 仅训练 | 仍只有 hard 22-D | 从 oracle gate 到可部署 gate 的恢复率 |
| R4 action-visible oracle spatial | GT spatial field | 连续 spatial oracle | 只作不可部署 ceiling，不能与 strict 方法混为同一 claim |

若 R1/R2 在正确 first-contact、rim alignment 或 spill/collision proxy 上都无增益，则不值得扩大标注。R4 有增益而 R1/R2 无增益，说明收益依赖绕开 finite bottleneck，不支持本 idea。

### 8.2 现有 `track_b/sim_oracle_benchmark_v0`

当前 spec 已经很接近所需 oracle protocol，位置是：

- `track_b/sim_oracle_benchmark_v0/benchmark_spec.json`；
- `track_b/sim_oracle_benchmark_v0/README.md`；
- `track_b/sim_oracle_benchmark_v0/local_readiness_20260811.json`。

它是 `spec-ready / execution No-Go`：32 sanity + 40 ID×4 seeds + 20 OOD×4 seeds，共 272 scenarios；7 个 variants 组成 1,904 个 planned rollout slots。已有 variant 包括 `rgb_only`、`rgb_depth`、`no_contact_aux`、`direct_action`、`oracle_contact`、`predicted_contact` 和 action-visible、不可部署的 `oracle_all_fields`。GT contract 已要求 same-state native physics/render，并递归拒绝 `inpaint/composite/diffusion/image_edit/manual_paint/pseudo_label/reconstruction_as_gt`。

对 C 的未来最小增量是预注册两个**训练期** variant，而不是改写现有 oracle：

- `predicted_contact_plus_visibility_aux`：预测 finite contact + 预测 status/geometry auxiliary，test action route 不变；
- `oracle_visibility_geometry_aux_train_only`：native status/field 只作训练 loss，test action route 不变。

它们必须与 `predicted_contact` 共用 scenario_id、physics/render/policy seeds、初始化、训练预算与 policy stochastic schedule。`oracle_all_fields` 保持 action-visible ceiling，不能替代上述两个 variant。

现有主要指标已经覆盖：OOD task success、correct-first-contact、first-contact/pour/release timing、transfer/spill/overflow/collision/mass conservation、boundary F1、rim chamfer、liquid/stream/contact field、temporal jitter 和 risk calibration。C 还应把 field 指标按 `clear/partial/unlocalizable` 分层，并同时报告 coverage；只在可见子集上的高分不能外推到全部透明场景。

### 8.3 Isaac PBD 与 MuJoCo 的能力边界

- Isaac/Omni PhysX 官方 particle 文档说明 PBD particle/fluid 是 GPU-accelerated，CPU particle simulation 不受支持；可配置 viscosity、surface tension、cohesion、adhesion 等。参见 [Omni Physics particles](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html) 与 [Isaac Sim API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html)。本任务明确不使用 GPU，因此当前不能执行 PBD。
- 本机静态审计虽发现 Isaac Sim 5.1，但缺准确 `G1 + 左 RH56DFTP + 右 OmniPicker` 装配、transparent/opaque 成对 watertight vessels、rim semantic、volume/flux adapter 和相机/质量守恒 receipt。一个 `Beaker_01.usd` 不能证明这些条件。
- [MuJoCo 官方 fluid 文档](https://mujoco.readthedocs.io/en/3.1.1/computation/fluid.html) 把流体限定为对 body 的 stateless phenomenological force；proper fluid dynamics 超出其范围。MuJoCo 可做刚体 contact/timing/control baseline，但不能提供 free-surface、stream、rim flux、spill volume 或容器间液体转移 oracle。
- 即使未来 Isaac PBD 可运行，particle contact、free-surface 和 stream occupancy 也依赖粒子半径/分辨率/阈值；必须以 native state、质量守恒和 repeatability 校准。它只回答 controlled ceiling，不证明真实透明视觉有效。

## 9. Baselines、消融与评价

### 9.1 必须的 baselines

1. strict finite-contact WAM，仅 hard 22-D；
2. strict WAM + 参数量配平的无标签 auxiliary decoder；
3. ungated field head：把 absent/unlocalizable/unknown 都当全零；用于量化错误 missingness 的伤害；
4. visibility status-only，无 geometry；
5. boundary/rim aux-only；
6. PourIt!/CAM 或 frozen segmentation teacher proposal，明确不是 GT；
7. raw depth、transparent depth completion、RGB-only 三个传感 baseline；
8. Mask-World-Model/SelfWAM-style generic semantic/self mask auxiliary；
9. action-visible continuous heatmap/field upper bound；
10. oracle gate、oracle visible geometry、oracle finite contact 与 oracle-all-fields ceiling。

### 9.2 关键消融

- existence 与 observability 合并 vs 分离；
- `unknown` masked vs 当 negative；
- exact onset vs interval/censored hazard；
- predicted gate vs oracle gate；
- current-frame vs temporal clip；
- boundary/source rim/target rim 逐项移除；
- stream/contact status-only vs dense geometry（只有正式审计通过后才允许后者）；
- aux-only vs action-visible；
- 共享视觉 backbone vs stop-gradient aux probe；
- RGB-only / raw depth / completion depth；
- transparent/colored、clear/partial、episode、phase 分层；
- human-direct/native GT 与 teacher/pseudo proposal 分层。

### 9.3 指标

标注层：A/B Cohen/Fleiss κ、0↔1 冲突率、unknown 率、各 episode/phase 的 localizable coverage、boundary F1、visible-polyline Chamfer、rim ellipse 中心/轴长/角度误差、事件 interval IoU、adjudication 改写率。

预测层：status macro-F1/ECE、conditional boundary F1/rim Chamfer、coverage-risk curve、event F1@100 ms、interval NLL、onset/offset MAE、temporal jitter。任何 geometry 指标必须同时报 denominator 和 visibility strata。

策略层：task success、correct first contact、source/target rim collision、transfer/spill/overflow、forbidden collision、动作平滑度；sim 还要 mass residual 与四 seed repeatability。真实数据在 group metadata 恢复前不得报告 object/liquid/operator OOD。

## 10. 分阶段 Go / Stop 门槛

### Gate 0：100 帧正式双标

只有满足下列条件才允许扩大标注：

- status κ：boundary/source rim/target rim ≥0.70，其余字段 ≥0.60；
- absent↔present-unlocalizable（原 0↔1）分歧 ≤10%；
- boundary/source rim/target rim 的 localizable positives 不只来自单 episode，至少跨 3/5 episodes；
- visible geometry A/B：boundary F1 ≥0.80，rim symmetric Chamfer ≤图像对角线 2%；
- 每条标注都有 source frame/clip hash、annotator、版本和 adjudication；
- liquid surface 若仍只在 ep58 着色情形可见，则只能保留为 dataset-specific diagnostic；
- stream/contact-interface 若可局部化率仍低于 20%，停止 dense head，只保留 status/unknown/event interval。

当前只有 prescreen，**Gate 0 未通过**。

### Gate 1：contact/event 真值

- 完成现有 12-episode A/B + adjudication，不允许 header-only CSV；
- 至少 50 个 reviewed event boundaries，且 observed/left/right/interval/unknown 分布全部报告；
- hand tracker proxy 只用于候选采样；对接触时刻不得自动覆盖人工/native truth；
- 若关键 pour/contact event 的可确认区间在至少 3 个 episode 中都不足 50%，停止真实 event head，转为 sim-only oracle question。

当前 **Gate 1 未通过**。

### Gate 2：低成本 oracle/probe

- 在冻结或小 backbone 上，R1/R2 相对 R0 对预注册的 correct-first-contact 或 rim/collision proxy 至少提升 5 个绝对百分点，并且 episode bootstrap 95% CI 下界 >0；
- predicted gate 保留至少 60% oracle-gate 增益；
- ungated/all-zero-label baseline 必须显著更差，否则 observation contract 没有经验价值；
- 若 oracle train-only aux 增益 <3 points，或只有 action-visible spatial oracle 有增益，**停止，不进入 WAM 训练**。

当前没有 reviewed label，**Gate 2 不可执行**。

### Gate 3：sim oracle

先满足 `track_b/sim_oracle_benchmark_v0` execution readiness。沿用其现有门槛：质量守恒超 1% 的 rollout ≤1%，同 condition 四 seed 的 transfer range ≤0.05；`oracle_contact` 对 `direct_action` 的 OOD success / correct-first-contact 至少 +0.10 / +0.15；predicted contact 恢复 ≥70% oracle gain；oracle fields 至少让 spill AUPRC +0.05 或 collision rate 相对降低 20%。

对 C 新增：train-only oracle fields 相对 `predicted_contact` 的 OOD success 或 correct-first-contact ≥5 points，且 predicted visibility auxiliary 恢复 ≥60% 该增益。若 action-visible `oracle_all_fields` 通过而 train-only variant 不通过，不能宣称 C 支持 finite-contact WAM。

当前 readiness 为 No-Go，**不得启动 execution**。

### Gate 4：最终主张

- 恢复 operator/object/liquid/session group metadata 或重采集后，预注册真正 OOD split；
- 相对 strict finite-contact matched control，真实 OOD task success ≥3 absolute points，95% bootstrap CI 下界 >0；
- correct-first-contact 不下降，spill/collision 至少一项改善且另一项不恶化；
- 所有 action-visible input、sidecar、checkpoint shard、config 与 code commit 都进入 receipt；
- 若增益依赖 pseudo/completed/reconstructed label、连续 field 进入动作，或只在 ep58 着色 liquid 成立，停止论文主张。

## 11. 建议决策与最小下一步

### 决策

**No-Go：现在不训练、不实现 dense stream/contact、不主张透明 3D 或真实液体状态。**

**Conditional Go：只允许完成 `track_b/visibility_audit_v0` 的 100-frame A/B + adjudication，并把 12-episode contact packet 从空模板变成 reviewed intervals。** 这一步不需要 GPU，且能用最小成本决定 idea C 是否还有数据基础。

若 Gate 0/1 通过，下一步也不是直接上 5B WAM，而是先做冻结 backbone 的 status/rim probe 与 R0–R4 oracle ceiling。只有 Gate 2 出现稳定增益，才值得实现独立 visibility sidecar 和 training-only decoder。

### 可支持主 finite-contact WAM 的最终贡献表述

可以尝试的表述是：

> 在透明对象和液体不可可靠重建时，我们不估计伪 3D，而以 factorized existence/observability 和 temporally censored events 定义可审计 supervision；训练期 visible-only boundary/rim auxiliary 改善共享视觉表征，但推理动作仍受 hard finite contact grammar 严格限制。真实双标审计与 same-state native sim oracle 分别验证可标注性和物理 ceiling。

不能使用的表述包括：

- “从当前 66 集恢复了透明液体 3D/真实深度”；
- “补全图、composite 或 teacher mask 是 ground truth”；
- “hand tracker active 等于物理 contact”；
- “Isaac/MuJoCo benchmark 已运行或已证明 sim-to-real”；
- “100-frame 单人 prescreen 已经是训练标签”；
- “普通 mask/visibility/contact auxiliary 本身是新颖贡献”。

## 12. 最终判断

| 维度 | 判断 |
|---|---|
| 科学问题价值 | 高：透明/液体不可见时如何让 policy 诚实处理缺失证据 |
| 单独方法新颖性 | 低到中：构件已有；需要 strict grammar + observation contract + causal benchmark 的组合证据 |
| 当前真实标签准备度 | 低：100-frame 仅 prescreen，contact formal packet 为空 |
| 现有 66ep 对 rim/boundary 的潜力 | 中：70%/70%/100% 可局部化，值得正式双标 |
| 现有 66ep 对 liquid/stream/contact pixels 的潜力 | 低：20%/0%/0%，且 liquid positives 集中在单个着色 episode |
| sim oracle 准备度 | No-Go：spec 完整但资产、adapter、GPU PBD、标定 receipt 不齐 |
| 与主 finite-contact WAM 的兼容性 | 有条件兼容：aux 必须 training-only，动作仍只读 hard 22-D grammar |
| 总决策 | **No-Go for training/claim；Conditional Go for annotation + oracle audit only** |

