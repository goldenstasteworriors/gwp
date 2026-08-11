# SM-FET 透明容器/液体非重建式状态—事件—效果契约

- 日期：2026-08-12
- 审阅性质：独立结构审查；只读核对代码、设计文档、Zotero 与原始论文
- canonical 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 目标问题：透明物体/液体重建差，且不能把 reconstruction optimization 当主路线
- 本轮产物：仅本 memo；未实现代码、未训练、未调用 GPU、未启动仿真/服务器
- `idea_id`：`f875e240-6c50-4949-9d1b-ccb7cbcaefed`
- `discussion_id`：`d8e376b4-28e4-4497-8c6d-9510aae900e1`

## 0. 唯一裁决

唯一推荐不是另建 transparent branch，也不是缩小版 reconstruction，而是把现有
`event_tube_v2_standard / sm_fet_hard_grammar_v2_standard_r1` 冻结为一个证据更严格的任务
profile：

```text
SM-FET-O2RE-NR-v1
= Observable 2D Role–Relation–Event–Effect, Non-Reconstruction profile
```

它不是新 schema；它只规定透明场景中哪些直接证据可以编译进 standard-r1，哪些只能作
training-only auxiliary，哪些必须保持 UNKNOWN/MISSING/censor。部署信息流为：

```text
raw current RGB + causal history + robot proprio + text
          |
  Semi-Markov planner
          |
  evidence-aware hard compiler
          |
4 role nodes + active/next1/next2 + <=2 edges/event
82 canonical uint8 codes / 13 hard tokens / exact 207.749746919520 bit
          |
 strict WAM action expert
          |
predict 48 steps -> execute 8 -> reobserve
```

核心决定只有四条：

1. **不恢复不可见世界。** 不估透明 depth、mesh、NeRF、液体体积、流速或接触界面，再把估计值当状态真值。
2. **动作只读可复核的有限投影。** source/tool/target/support 的粗 2D node、actor/part、对象 pre/post relation、事件时序、effect 与 verify-only outcome 沿用 standard-r1；连续手/杯口/边界几何不直达 action。
3. **不可判就显式拒绝。** 有 UNKNOWN code 的字段编 UNKNOWN；未复核为 MISSING、loss mask 为零；没有 UNKNOWN code 的引用字段无法确定时，packet 不可编译，部署应 hold/reobserve，而不是猜一个值。
4. **transparent auxiliary 只能教 planner 看证据。** visible-only hand/tool/rim/target 边界和删失事件可训练共享视觉/planner，但其 logits、heatmap、hidden、proposal 和重建结果不得进入 action、cache 或 oracle GT。

因此，本审查对用户 original issue 3 的答案是：**绕开 reconstruction optimization，用“粗 2D
角色状态 → 接触/对象关系变化 → 事件效果 → 可验证结果”的有限契约控制；把液体本体状态视为部分可观测变量，而不是必须补全的几何。**

当前状态仍是 **No-Go for model training/claim；Conditional Go for annotation、compiler audit 与
oracle design**。现有 66 段 Ego 倒液没有足以解除这个结论的透明液体、contact、outcome 或 group
真值。

## 1. 证据与上位规范

### 1.1 Zotero-first 记录

本轮先按 `idea-discussion` 同步个人 Zotero，再创建上述 idea/discussion。Zotero 自动召回主要是
RoboBrain、RT-1/RT-2 等通用条目，外部自动 scout 还出现 image-forensics 等明显误召回；因此不把
它们用于具体技术断言，随后逐篇核对 arXiv、CVF、PMLR、OpenReview、IEEE/官方项目页中的原始工作。
这次 memory 已更新，但“Zotero 有记录”不等于相关性或新颖性已获支持。

### 1.2 必须同时成立的内部规范

本 memo 读取并以如下版本作裁决：

- Idea C visibility/contact auxiliary，commit `de3cbc9`；
- op33 same-state sim oracle，commit `7d8a6d1`；
- SM-FET schema，commit `190f302`；
- SM-FET final architecture，commit `6bc51b9`；
- 修订后的 annotation/pairing protocol，commit `288ef5d`；
- final portfolio，commit `3c545aa`。

冲突时的优先级为：final architecture + standard-r1 schema/annotation > final portfolio > Idea C/op33。
Idea C 仍以旧 22D contact grammar 为主，因此其中“所有空间量都 training-only”的表述不能机械搬到
v2；standard-r1 已把**粗粒度、有限、可反序列化的 2D node/part/relation**纳入 action。Idea C
仍然正确的部分是：连续 boundary/rim/mask、液体几何、confidence 与重建结果不得形成第二条 action
视觉通道。

### 1.3 原始工作给出的设计约束

| 路线 | 原始工作已覆盖的内容 | 对本契约的约束 |
|---|---|---|
| 透明深度/重建 | ClearGrasp 从 RGB-D 预测 mask、normal、occlusion boundary 后补深度；TransCG 提供专门真实 GT；A4T 用 affordance map 辅助透明深度重建与操控；ClearPose 提供专门 pose/depth 数据 | completion/reconstruction 必须是互斥 baseline，不是当前 Ego GT；boundary 本身不是新贡献 |
| 非重建 affordance | VRB 用 human video 学 contact point 与 post-contact trajectory；Human Hands as Probes 学 interaction region、grasp 与 object state；Interaction Hotspots 预测 future hand trajectory/contact heatmap | 2D hand/contact/affordance 已有直接前例；本工作只能主张有限 evidence contract 和 strict route |
| event/contact/state | Forecasting through Contact 用 TTC/contact map 和 contact-delineated state；Ego4D 明确 PRE/CONTACT/PNR/POST；WACV 2024 联合 hand-contact/object-state；relation-centric work 预测 action 后关系/效果 | contact/event/pre-post/effect 不是新 primitive；UNKNOWN/censor、编译和对 action 的可干预性才是可能残余 |
| liquid pouring | Visual Closed-Loop Pouring 已用视觉体积估计闭环控制；Adaptable Pouring 与 SPNets 走模拟/粒子动力学；透明液体分割用 colored-liquid/background subtraction + 生成式翻译；PourIt! 用 image label/CAM、container pose 与 3D recovery | 不得把生成图/CAM/3D recovery 叫真实透明 GT；物理模拟只可提供受控 ceiling；非重建主线必须与这些路线公平比较 |

所以论文不能写“首次不用 3D 处理透明物体”“首次用 hand/contact/event/effect”或“首次做视觉
pouring”。可辩护的窄主张是：

> 在透明 RGB-D 中不补造不可观测 3D，而将直接可见证据编译成固定容量的
> role–relation–event–effect packet；用字段级 UNKNOWN/MISSING/censor 和 provenance 保证
> auxiliary 不成为 action 旁路，并用 same-state oracle 与 matched controls 证伪其控制充分性。

## 2. O2RE-NR 的唯一字段契约

### 2.1 固定 task ontology

普通 beaker-to-beaker pour 的四个 node slots 只能是：

```text
slot 0 manipulated_or_source         = source vessel
slot 1 tool                          = ABSENT；只有真实独立 pipette/stirrer/funnel 才 present
slot 2 receptacle_or_destination     = target vessel
slot 3 fixture_or_support            = table/rack/holder（被 active/next2 引用时才 present）
```

禁止把 `liquid`、`stream`、source rim、target rim 各自升为自由 node。它们分别由 effect、part、
relation 和 auxiliary 表达，否则四 role 容量与 reference union 会被静默改写。一个 slot 同时对应多个
实体或无法唯一 binding 时记录 overflow/invalid reference，不让标注者挑最像的实例。

pour task 的推荐 edge 用法是：

- effector→source：primary 为 source 的 `body/handle`，contact mode 描述持握；若同一 edge 还承载
  source→target relation，secondary 为 target；
- 第二只 effector→target：仅在真实稳定 target 时记录；
- 无第二只 effector且需要显式 source–target relation 时，可用 `OBJECT_OBJECT` edge，primary part 为
  `opening/rim_edge`、secondary 为 target、contact mode 为 `NONE`；
- 不允许因液流经过杯口而生成 hand-liquid 或 liquid-rim “contact edge”。

每事件最多两条 edge。双手各接触一只杯且还需要第三条对象关系时，优先把对象 relation 放在
effector→source edge 的 secondary/relation 字段中；若仍无法无歧义表达，记 `EDGE_OVERFLOW`，不静默
截断。

### 2.2 哪些信息 action-visible

动作侧严格保持 standard-r1 的 82 codes / 13 tokens：

| 组 | action-visible 字段 | 透明场景解释 |
|---|---|---|
| node `[4,4]` | `observability/presence, center_cell, scale_band, orientation_2d` | source/tool/target/support 的直接可见粗 2D 状态；不是 metric pose/depth |
| event `[3,6]` | `kind/presence, focus_node, onset, duration, object_effect, outcome` | active+next1+next2；effect 可为 FLOW/TRANSFER，但 GT 必须有直接或 sensor 证据；outcome 只在 verify event |
| edge `[3,2,8]` | `actor/presence, primary_node, primary_part, primary_surface_cell, secondary_node, relation_before, relation_after, contact_mode` | actor 是离散参与者；part/cell 是有限 interaction site；relation 是对象间可复核状态，不是 hidden ROI |
| 既有合法输入 | robot proprio/state、instruction text、diffusion timestep/noisy action | 当前机器人手指/末端状态必须来自 proprio，不能由 human hand pose替代 |

具体解释：

1. **hand：** raw hand box、wrist/fingertip 坐标、aperture 和 human trajectory 不可见；action 只看 edge
   actor/part/contact mode，以及机器人自己的 proprio。human/robot literal left/right 不用于跨域 OT cost。
2. **tool：** 独立工具才占 tool node；其 coarse cell/scale/orientation 可见，连续 mask、6D pose和 tool
   flow 不可见。
3. **rim：** exact rim ellipse/polyline 不可见；action 只能通过真实 interaction part `opening/rim_edge`、
   part-local 4×4 cell，以及冻结定义的 `ALIGNED_TO_RIM` relation 感知杯口语义。手抓 source body 时不把
   primary part伪写成 rim。
4. **target：** target node 的 coarse 2D 状态与 source→target relation 可见；目标 exact contour、rim
   center、depth 和容量不可见。
5. **effect/outcome：** `FLOW/TRANSFER` 是有限事件效果，不是液体像素或体积；`SUCCESS/PARTIAL/FAILURE`
   只属于 `VERIFY_OUTCOME`。release 一律 outcome=0，不能把“已经松手”推导为成功。

### 2.3 哪些信息只能 auxiliary/probe/provenance

以下内容 action-visible 增量必须为 **0 bit**：

| 类别 | 允许用途 | 禁止用途 |
|---|---|---|
| visible-only source/target rim ellipse、polyline、boundary edge band | planner/backbone masked auxiliary；annotation QA；field metric | 拼到 packet/action token；作为完整遮挡轮廓或 3D vessel GT |
| hand/tool/target 2D box、7 hand landmarks、entry/turn/exit pose、optical flow | event-multipose auxiliary、compression probe、proposal benchmark | human joint→robot joint retarget；action KV；CAPOT 主 cost中的高带宽 shortcut |
| liquid surface/stream/contact-interface `existence + observability` | status-only auxiliary；coverage-risk audit | 在可标性 gate 前做 dense geometry；把 unlocalizable 当全零 mask |
| exact event boundary support、interval、left/right censor、coverage | survival/interval loss、compiler source、评测 | exact frame/time直接进入 action；把 coverage 边缘当事件边界 |
| confidence、evidence tier/scope、review state、annotator、hash | loss weighting、provenance、fail-closed audit | planner/action 输入；用 confidence 扩大 action alphabet |
| depth hole、raw/completed depth、mesh/NeRF、liquid volume/flux | 独立 sensor/reconstruction baseline，或 native oracle | 当前真实 GT；缺失字段回填；predicted route 的隐藏 side channel |
| proposal logits/heatmap/teacher mask/CAM/CAPOT mass | proposal-only 数据层或对照 | gold 指标、loss mask开关、oracle target、action/cache |

辅助标签的最小 observation record 沿用 Idea C 的四分法，但与 action 编译分开：

```text
existence:      ABSENT | PRESENT | UNKNOWN | NOT_APPLICABLE
observability:  CLEAR | PARTIAL | PRESENT_UNLOCALIZABLE |
                OUT_OF_FRAME | SENSOR_INVALID | UNKNOWN
geometry:       visible-only polyline/ellipse/box/point-set，或 empty
evidence:       HUMAN_DIRECT_RGB | HUMAN_TEMPORAL_RGB |
                SENSOR_CONFIRMED | SIM_NATIVE_STATE | SAME_STATE_NATIVE_RENDER
review_state:   KNOWN | REVIEWED_UNKNOWN | NOT_REVIEWED
```

`PARTIAL` 只画可见段；`PRESENT_UNLOCALIZABLE` 不生成全零 geometry；`UNKNOWN/NOT_REVIEWED` 不产生
geometry loss。模型预测的 visibility 可以输出校准分数，但不能反向决定训练 mask。

## 3. 何时必须 UNKNOWN、MISSING、censor 或拒绝编译

### 3.1 机械决策表

| 证据状态 | annotation/loss | action compiler |
|---|---|---|
| EXHAUSTIVE review 明确无 slot/entity/event/edge | `ABSENT` supervised negative | presence code=0，其 payload literal positive zero |
| 实体/事件存在且已复核，但某字段不可判 | `REVIEWED_UNKNOWN`，该字段参与 UNKNOWN CE | 该字段有 UNKNOWN code时编 UNKNOWN |
| 未看、无证据、未覆盖 | `NOT_REVIEWED/MISSING`，mask=0 | required target不可用于 oracle；不得自动变 UNKNOWN/ABSENT |
| onset 在 coverage 开始前、状态已存在 | `LEFT_CENSORED` survival target | 只能作为 active，onset=`ACTIVE` |
| offset 在 coverage 结束后、状态仍存在 | `RIGHT_CENSORED` survival target | duration=`RIGHT_CENSORED` |
| 遮挡导致边界不明 | boundary `UNKNOWN` | 不是 censor；对应 onset/duration UNKNOWN 或 target missing |
| focus、actor、primary node等没有 UNKNOWN code的 required reference 无法确定 | 保留原始 reviewed state与原因 | packet/sample `INVALID_REFERENCE`；部署 hold/reobserve，不猜 |

`UNKNOWN` 已经是 standard-r1 的正常离散类别，不是“低置信预测值”；`MISSING` 完全不进 wire。对
predicted planner，模型当然必须输出一个有限候选，但 validator 拒绝非法 reference/grammar/overflow；
低证据时应输出合法 UNKNOWN/NULL，或触发 safe reobserve，不允许 soft logits/hidden 旁路到 action。

### 3.2 透明 pouring 的 hard unknown 规则

以下规则不允许由 task name、动作意图或常识覆盖：

1. 只有 2D overlap 时，relation 最多是 `OVERLAP_2D`；它不能推出 `TOUCHING/contact`。
2. `INSIDE/CONTAINS` 必须有直接时序证据或同步 sensor/native truth；透过透明杯看到投影重叠不够。
3. `ALIGNED_TO_RIM` 只有 source/target rim 都可定位，且 A/B 支持区域经过冻结量化后落在同一判定区间
   才 known；否则 relation=UNKNOWN。它是任务 ontology 冻结的 image-plane predicate，不宣称 3D 同轴。
4. node 的可见支持跨多个 4×4 cell、scale band或orientation bin时，相应字段=UNKNOWN；不能取中心点
   强行量化。
5. hand/object mask overlap、`hand_status`、gripper command 或 tracker valid 不能证明 contact mode、
   acquire/release 边界；最多用于 proposal window。
6. 看不到 stream、meniscus、目标液位变化，且无秤/液位/flux 传感器时，`FLOW/TRANSFER` GT 与 pour
   outcome 必须 `REVIEWED_UNKNOWN` 或 MISSING；不能从 wrist tilt 推出液体已转移。
7. `VERIFY_OUTCOME` 只有在结果实际可观察时建立；“按脚本做完”不是成功。若无可见或 sensor 结果，
   outcome=UNKNOWN；release 仍保持 outcome=0。
8. 透明 liquid volume、flow rate、contact interface、metric rim offset、spill fraction 无 native/sensor
   truth时一律不标数值 GT。

## 4. 反 reconstruction / pseudo-GT 防线

### 4.1 GT allowlist 与永久 denylist

可进入 adjudicated/native GT 的来源只有：

```text
human_direct_raw_rgb
human_temporal_raw_rgb
sensor_confirmed
sim_native_state
same_state_native_render
mechanical_derivation_from_the_above
```

如下来源递归出现在 artifact、父记录或生成 receipt 中时，GT admission 必须 fail closed：

```text
inpaint, composite, diffusion, image_edit, manual_paint,
pseudo_label, teacher_fill, depth_completion_as_gt,
mesh_or_nerf_as_gt, reconstruction_as_gt
```

op33 已对其中核心词做递归拒绝；O2RE-NR 进一步要求 proposal 与 gold 使用不同 schema、目录和
`label_origin`，不能靠人工点“接受 proposal”就升级为 direct evidence。A/B 标注者只看原始帧/视频，
不看 completion、CAM、VLM relation、tracker contact 或对方标注。

### 4.2 可以比较但不能偷换的路线

- raw depth 是传感 baseline；全图 depth hole 不能自动归因于透明杯；
- ClearGrasp/TransCG/A4T 类 completion 是 predictor + action-visible reconstruction baseline；
- colored-liquid/background subtraction、生成式透明翻译和 PourIt!-style CAM 是 weak/pseudo baseline；
- native simulator depth、segmentation、flux/volume 是 oracle channel，只证明 controlled ceiling；
- same-state native projection可以产生 boundary/rim oracle，但不能证明真实透明 RGB 可预测它；
- reconstruction baseline 赢，只说明高带宽几何路线可能更有效，不授权把它的输出回填 O2RE GT。

任何主结果若依赖 completed depth、mesh、continuous ROI hidden 或 action-visible heatmap，结论应改成
“reconstruction/spatial-affordance route 有效”，并停止 SM-FET non-reconstruction sufficiency claim。

## 5. WAM 插入与信息流审计（未来设计，本轮不实现）

### 5.1 唯一合法插入面

在 canonical `e212afec` 上，未来先实现 standard-r1 自己的独立 v2 module，再挂 O2RE auxiliary：

1. `world_action_model/models/transformer_wa_casual_mot.py:598-631,703-819`：新增 opt-in v2
   config/module；与 v1 22D route 互斥。transparent auxiliary 是同一 planner/backbone 的训练头，不是
   新 action route。
2. `transformer_wa_casual_mot.py:1253-1285`：现有 planner 从 state/ref/text 取证且 hidden 不导出；
   v2 planner邻近输出 factorized hard fields，visible-only decoder 可读 ref visual tokens，但只返回 aux
   prediction/loss所需张量。
3. `transformer_wa_casual_mot.py:1405-1490`：现有 22→action projection 是唯一 v1 bridge；v2 应改成
   13 packed factor tokens。禁止 concat rim/hand heatmap、aux decoder hidden、reconstruction feature。
4. `transformer_wa_casual_mot.py:1531-1589`：joint encode 可返回 hard packet logits、canonical bytes、
   aux predictions 与 receipt；只有 canonical packet进 action encoder。
5. `transformer_wa_casual_mot.py:1806-1821`：strict v2 必须拒绝 image cross-attention，并切断
   state/packet/action→reference 旁路；否则 non-reconstruction 仍可能被 raw image hidden 绕开。
6. `world_action_model/datasets/wam_lerobot_dataset.py:711-819`：沿 contact sidecar 的 SHA、schema、
   frame-count、dtype fail-closed 模式加载独立 event 与 visibility sibling sidecar；不把 whole-episode 表
   复制成 frame action input。
7. `world_action_model/trainer/wa_casual_trainer_mot.py:957-1055`：在现有 contact target/loss 入口邻近
   增加 v2 categorical + interval/survival + masked status/visible geometry loss；NOT_REVIEWED 梯度为零，
   REVIEWED_UNKNOWN 有对应 CE。

### 5.2 必须证明的隔离不变量

- 固定 packet/state/text/noise，替换 raw reference 或 aux logits/hidden，strict action 输出不变；
- predicted action forward 改 GT aux target/label bytes时，pre-safety action不变；
- action attention/cache allowlist 中没有 ref/future/planner hidden/heatmap/depth/pose/CAPOT；
- `encoder_hidden_states_image` 在 strict route非空时 fail closed；
- auxiliary target swap 只能改变 loss，不能改变同一次 forward 的 action condition；
- oracle、predicted、train-only auxiliary 与 action-visible spatial ceiling 使用不同 route ID/checkpoint
  receipt，不做同 batch field-wise oracle merge；
- sidecar、augmentation trace、source frame、schema、config、checkpoint bundle和 commit 均 hash 绑定。

如果 continuous auxiliary 必须回流 action才有增益，它就不是 supporting auxiliary，而是一个应独立
计容、独立命名的 spatial-affordance baseline。

## 6. Baselines 与消融

### 6.1 互斥 action routes

| ID | action-visible 视觉桥 | 目的 |
|---|---|---|
| B0 `direct_rgb_wam` | raw current RGB/reference | 高带宽直接视觉基线 |
| B1 `direct_raw_rgbd` | raw RGB-D | 原始传感基线；不补深度 |
| B2 `reconstruction_action` | completed depth/mesh/point cloud | ClearGrasp/TransCG/A4T 型互斥 reconstruction baseline |
| B3 `contact_v1_22d` | 旧 finite contact packet | 已实现 finite 基线 |
| B4 `sm_fet_standard_no_trans_aux` | 208-bit hard packet | 判断 standard-r1 本体收益 |
| **M `sm_fet_o2re_nr`** | 与 B4 完全相同的 208-bit packet | 推荐方法；区别只在 evidence contract 与 train-only aux |
| B5 `action_visible_2d_affordance` | contact/rim heatmap 或 2D flow | VRB/Hotspots/PourIt!-style spatial ceiling，不可与 M 混写 |
| B6 `full_continuous_trajectory` | local/full pose/geometry trajectory | 信息充分性上界 |
| B7 `sm_fet_equal_bit` | 全 packet 可逆等容量置换 | 测试 semantic field 是否必要 |

B2/B5/B6 的 action capacity不是 208 bit，不能与 M 合并后仍引用 207.7497-bit claim。所有路线必须
匹配 scenario、seed、backbone、训练步数、action horizon、execute/replan schedule；新增 aux 参数时用
无标签同结构 decoder或参数/FLOPs配平。

### 6.2 O2RE 内部消融

1. no auxiliary / status-only / boundary-only / source-rim / target-rim / all visible-only；
2. existence 与 observability 分离 vs 合并；
3. UNKNOWN masked correctly vs 当 negative vs all-zero geometry；
4. exact-bin 强制量化 vs support-spans-bin→UNKNOWN；
5. current-frame vs causal clip；
6. exact boundary regression vs interval/censored survival；
7. relation-only / effect-only / relation+effect+verify outcome；
8. entry/exit vs entry/turn/exit hand pose auxiliary；
9. RGB-only / raw depth / completion depth；
10. teacher/CAM proposal-only vs human/native supervision；
11. oracle gate vs predicted gate；
12. aux action-invisible vs action-visible spatial ceiling。

关键判读不是“哪个 field F1最高”，而是：M 在 action bits 不变时是否改善策略；若只有 B5/B2
改善，finite non-reconstruction 路线失败。

## 7. Benchmark 与 oracle 设计

### 7.1 三层 benchmark，互不替代

| 层 | 数据 | 回答的问题 | 不能声称 |
|---|---|---|---|
| R-real 可标注性 | 3 calibration + 12 blind Ego pour；A/B+第三人裁决 | raw RGB中哪些 node/rim/relation/effect可重复复核 | 策略收益、真实 OOD、液体物理真值 |
| P-offline prediction | group-safe raw RGB clips + reviewed labels | predicted hard fields/visibility/event 的恢复率和风险校准 | closed-loop control充分性 |
| S-same-state oracle | op33 式 native state/render/contact/liquid truth | hard O2RE packet在受控物理中的 action ceiling | 真实透明重建、真实 depth、sim-to-real |

### 7.2 op33 的最小扩展

op33 当前是 `spec-ready / execution No-Go`，已有 272 scenarios、7 variants、1,904 planned rollout
slots；本轮不改它。未来在 execution readiness 全部通过后，新增四个 matched variants：

```text
oracle_sm_fet_o2re
  native state -> deterministic standard-r1 packet -> action

predicted_sm_fet_o2re
  native raw RGB render -> predicted packet -> action

predicted_sm_fet_o2re_plus_train_only_visible_aux
  与上一条 test action route相同；boundary/rim/status只参与训练

oracle_train_only_visible_aux
  native/same-state field只作训练 target；test仍是 predicted hard packet
```

原 `oracle_all_fields` 保持 action-visible spatial ceiling；不能替代 train-only auxiliary variant。另加
`completed_depth_action` 作为 reconstruction baseline。所有 variants 共用 scenario ID、physics/render/
policy seeds、初始化、训练预算、policy stochastic schedule与 native GT timeline。

native O2RE 编译必须从同一个 immutable state ID 机械得到：

- geometry projection→node cell/scale/orientation 与 visible rim metric；
- native contact→actor/part/contact mode 与 acquire/release；
- native source-rim flux→FLOW onset/offset；
- per-vessel/outside volume→TRANSFER/spill 与 verify outcome；
- source/target geometry→relation before/after；
- RGB/depth/instance→同状态 render channel，camera-state skew ≤5 ms。

透明/折射只影响 native RGB。native state label 不从 render completion、segmentation teacher或 policy
prediction反推。

### 7.3 指标

标注层：event F1/tIoU、boundary error、kind/role/part/contact/relation/effect/outcome κ、node cell exact
agreement、rim visible Chamfer、UNKNOWN/MISSING/ABSENT/censor confusion、compile coverage、overflow、工时。

预测层：field macro-F1/NLL/ECE、conditional rim/boundary metric、coverage-risk、event F1@100 ms、
interval NLL、onset/offset MAE、invalid packet rate。所有 geometry 指标必须同时报告 denominator，并按
`CLEAR/PARTIAL/UNLOCALIZABLE/OUT_OF_FRAME/SENSOR_INVALID` 分层。

策略层：OOD task success、correct first contact、first-contact/pour/release timing、transfer/spill/overflow、
forbidden collision、mass residual、action smoothness和 risk calibration。成功不能只用脚本完成标志，必须
由冻结的 volume/spill/collision/stability contract判定。

### 7.4 Go / Stop 门槛

#### T0：visible 2D 可标注性

- boundary/source rim/target rim status κ `>=0.70`；其它 status `>=0.60`；
- absent↔present-unlocalizable 分歧 `<=10%`；
- source/target rim localizable positives 至少跨 3/5 prescreen episodes；
- boundary F1 `>=0.80`，rim symmetric Chamfer `<= image diagonal 的 2%`；
- stream/contact-interface localizable coverage `<20%` 时，永久停止 dense real-GT head，只留 status/event；
- 当前 100-frame 结果只是单人 prescreen，故 **T0 未通过**。

#### T1：standard-r1 局部编译可行性

沿用修订 annotation D1：event F1 `>=0.80`、matched tIoU `>=0.70`、kind/role/relation/effect κ
`>=0.70`、contact mode κ `>=0.60`、boundary P50 `<=2 processed frames/100 ms`、关键 effect/outcome
双侧 unknown `<=40%`、Q/N/R overflow各 `<=5%`。任何未出现两个类别的字段为 NOT_EVALUABLE，不用
100% agreement过门。

若 source/target node、focus、actor或 relation 的联合可编译 coverage 低到不能形成足量 reviewed
positive packet，则现有 66 Ego 只保留 annotation benchmark，不训练 O2RE policy。当前 D0 checker 和
真实 reviewed labels 均不存在，故 **T1 未通过**。

#### T2：低成本 train-only auxiliary value

在小模型/冻结 backbone、matched budget上：

- `oracle_train_only_visible_aux` 相对 `predicted_sm_fet_o2re` 在预注册 OOD task success 或
  correct-first-contact 至少 `+5 pp`，cluster/bootstrap 95% CI 下界 `>0`；
- predicted visibility auxiliary 恢复至少 `60%` 的 oracle-train-only gain；
- 若 oracle train-only gain `<3 pp`，或只有 action-visible spatial/reconstruction ceiling有增益，
  **停止 O2RE auxiliary，不进入大 WAM**。

#### T3：sim/native oracle sufficiency

先满足 op33 execution readiness：准确 G1+RH56+OmniPicker资产、watertight透明/不透明 vessels、native
liquid/contact/render adapter、same-state/skew receipt、双 reviewer、所有 artifact/checkpoint hash。当前
readiness=`no_go`，故不得执行。

执行后同时满足：

- mass residual >1%的 rollout `<=1%`；同 condition 四 seed transfer range `<=0.05`；
- op33 `oracle_contact` 相对 direct action：OOD success `>=+10 pp`、correct-first-contact `>=+15 pp`；
- `oracle_sm_fet_o2re` 相对 oracle 22D：outcome-sensitive success `>=+8 pp` 或 event completion
  `>=+10 pp`；
- 相对 full continuous trajectory 下降不超过 `3 pp`，安全不退化且 cluster CI 支持正优势；
- `predicted_sm_fet_o2re` 恢复至少 `70%` 的正 oracle gain；
- regrasp、透明/不透明、colored/clear、rim occlusion、role swap与 viscosity strata 单报。

未通过时 SM-FET action route No-Go；event/annotation 数据仍可作为 benchmark。`oracle_all_fields` 通过
而 train-only/predicted hard route失败，不能支持 non-reconstruction finite claim。

#### T4：最终真实主张

只有 group metadata恢复或新采集真正 group-safe target-robot data 后才做。相对 B4
`sm_fet_standard_no_trans_aux`，真实 group-OOD task success 至少 `+3 pp`、cluster bootstrap 95% CI
下界 `>0`；correct-first-contact不下降，spill/collision至少一项改善且另一项不恶化。若收益依赖单个
着色 episode、pseudo/completed label、连续 field 直达 action或 test-time reconstruction，停止论文主张。

## 8. Capacity 与模型表面积影响

### 8.1 action capacity：严格不变

O2RE-NR 不新增/删除 action 字段、不改 r1 grammar、N/K/Q/R或 code dictionary，因此：

```text
exact reachable action packet states
= 345864043392188289642033658185203030253953267411599196753571073

exact conditional current-visual route capacity
= 207.74974691951957 bit

fixed binary container width
= 208 bit
```

UNKNOWN/ABSENT 已被该枚举计入；MISSING、exact interval/censor bounds、confidence、rim/hand geometry和
auxiliary predictions不进 action，因此增加 `0 bit`。pour profile 中 tool 常为 ABSENT，会让经验使用熵
下降，但在没有另做 task-conditioned exact enumeration 前，不能把全局 207.7497-bit 数字任意减小。

### 8.2 何时必须重算并停止原容量 claim

以下任一变化都会使 207.7497 bit 失效：

- 新增 hand center、rim ellipse/cell、liquid status、depth、confidence 或额外 visibility token；
- 把 aux decoder hidden/heatmap、raw ROI/track embedding或 completed depth拼入 action；
- 增加 liquid/rim node、第三条 edge或 secondary part/cell；
- 与 v1 22D、F-PCD、ETG、continuous pose concat；
- 让 action/state/packet通过 attention/cache读取 reference/future/image cross-attention。

这些路线可以作为 baseline，但必须重新给出 action-visible alphabet/continuous bandwidth、参数/FLOPs和
route receipt，不能继续使用 208-bit headline。

### 8.3 参数与数据成本

train-only status/rim/boundary/pose decoder会增加参数、显存和标注成本，但不增加 action bits。实验必须
有同结构无标签 decoder、matched steps/FLOPs 与 stop-gradient probe，否则正结果可能只是更多参数或
更强 regularization。当前 100-frame prescreen 表明 target rim最可标，source rim/container boundary
次之，transparent stream/contact interface最差；首个允许的 auxiliary 顺序只能是：

```text
node observability/status
-> target rim
-> source rim/container visible boundary
-> event interval/censor
-> liquid/stream status only
```

dense liquid/stream/contact geometry不在默认路线中。

## 9. 失败模式与不可协商的停止条件

1. **projection ambiguity：** 2D aligned/inside 被误写成 3D物理关系；解决方式是冻结 predicate 与
   UNKNOWN，不是加 VLM 猜测。
2. **semantic shortcut：** task text直接让 planner输出 FLOW/SUCCESS，即使视频无证据；GT必须由
   observed/sensor criterion产生，并做 wrong-effect/text swap。
3. **aux leakage：** heatmap或 planner hidden进入 action/cache；任一 swap/attention审计失败即停止 strict
   claim。
4. **censor confusion：** 遮挡被标 left/right censor，或 coverage边界被当 exact event；协议/版本必须
   fail closed。
5. **contact hallucination：** 2D overlap、gripper closed或 hand tracker valid被当物理接触；相关 row
   不得进入 contact GT。
6. **oracle laundering：** native volume/segmentation或 completed depth被叫 predicted-real evidence；
   sim只报告 ceiling，reconstruction只报告 baseline。
7. **capacity laundering：** action-visible spatial field有增益后仍声称208-bit finite route；必须改名、
   重计容并判原 headline失败。
8. **visibility selection bias：** 只在 visible subset报好成绩；所有 field metric必须带 coverage和 risk
   curve，策略结果按透明度/遮挡分层。
9. **task-conditioned overfit：** 从“正在pour”反推 transfer/success；做 failure/no-flow/spill/empty-source
   hard negatives和 effect/outcome swap。
10. **数据不足：** T0/T1未过即扩到66或训练5B；停止，不用更大模型掩盖标签不可辨识性。

## 10. 最终建议

保留 SM-FET standard-r1，不为透明物体另开 reconstruction branch。新增的只是一份
`SM-FET-O2RE-NR-v1` evidence/annotation profile：

- action-visible：有限 node/event/edge 的 coarse 2D role、part、relation、effect、verify outcome；
- auxiliary：raw visible-only hand/tool/rim/target 几何、event boundary/censor与 status；
- unknown：任何无直接/temporal/sensor/native证据的 contact、3D relation、flow/transfer、volume与结果；
- fail closed：focus/actor/reference不可确定、overflow、schema/receipt不符；
- benchmark：real annotatability、offline prediction、same-state oracle三层分开；
- stop：train-only auxiliary无独立收益，或只有 reconstruction/action-visible spatial route有效时，停止
  non-reconstruction finite claim。

最小下一步仍只应是：冻结 pour ontology 与字段 predicate，完成 D0 annotation→82-code 映射设计，做
3 calibration + 12 blind raw-video双标，并把 T0/T1 结果按 PASS/FAILED/NOT_EVALUABLE 报告。未通过前
不实现模型、不训练、不启动仿真，也不把当前 66 段数据包装成透明液体技能真值。

## 11. 原始工作与官方来源

- Sajjan et al., [ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation](https://doi.org/10.1109/ICRA40945.2020.9197518), ICRA 2020.
- Fang et al., [TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline](https://arxiv.org/abs/2202.08471), 2022.
- Chen et al., [ClearPose: Large-scale Transparent Object Dataset and Benchmark](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680372.pdf), ECCV 2022.
- Jiang et al., [A4T: Hierarchical Affordance Detection for Transparent Objects Depth Reconstruction and Manipulation](https://arxiv.org/abs/2207.04907), RA-L 2022.
- Bahl et al., [Affordances From Human Videos as a Versatile Representation for Robotics](https://openaccess.thecvf.com/content/CVPR2023/html/Bahl_Affordances_From_Human_Videos_as_a_Versatile_Representation_for_Robotics_CVPR_2023_paper.html), CVPR 2023.
- Goyal et al., [Human Hands As Probes for Interactive Object Understanding](https://openaccess.thecvf.com/content/CVPR2022/html/Goyal_Human_Hands_As_Probes_for_Interactive_Object_Understanding_CVPR_2022_paper.html), CVPR 2022.
- Liu et al., [Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos](https://arxiv.org/abs/2204.01696), CVPR 2022.
- Dessalene et al., [Forecasting Action through Contact Representations from First Person Video](https://arxiv.org/abs/2102.00649), 2021.
- Ego4D, [Forecasting + Hands & Objects annotation guidelines](https://ego4d-data.org/docs/data/annotation-guidelines/).
- Shiota et al., [Egocentric Action Recognition by Capturing Hand-Object Contact and Object State](https://openaccess.thecvf.com/content/WACV2024/html/Shiota_Egocentric_Action_Recognition_by_Capturing_Hand-Object_Contact_and_Object_State_WACV_2024_paper.html), WACV 2024.
- Tekden et al., [Object and Relation Centric Representations for Push Effect Prediction](https://arxiv.org/abs/2102.02100), 2021.
- Schenck and Fox, [Visual Closed-Loop Control for Pouring Liquids](https://arxiv.org/abs/1610.02610), ICRA 2017.
- Lopez-Guevara et al., [Adaptable Pouring: Teaching Robots Not to Spill using Fast but Approximate Fluid Simulation](https://proceedings.mlr.press/v78/lopez-guevara17a.html), CoRL 2017.
- Schenck and Fox, [SPNets: Differentiable Fluid Dynamics for Deep Neural Networks](https://proceedings.mlr.press/v87/schenck18a.html), CoRL 2018.
- Narasimhan et al., [Self-supervised Transparent Liquid Segmentation for Robotic Pouring](https://arxiv.org/abs/2203.01538), ICRA 2022.
- Lin et al., [PourIt!: Weakly-Supervised Liquid Perception from a Single Image for Visual Closed-Loop Robotic Pouring](https://hetolin.github.io/PourIt/), ICCV 2023.
- Seita et al., [ToolFlowNet: Robotic Manipulation with Tools via Predicting Tool Flow from Point Clouds](https://openreview.net/forum?id=2gfB_kMVFvP), CoRL 2022.
- Yuan et al., [General Flow as Foundation Affordance for Scalable Robot Learning](https://openreview.net/forum?id=nmEt0ci8hi), CoRL 2024.
