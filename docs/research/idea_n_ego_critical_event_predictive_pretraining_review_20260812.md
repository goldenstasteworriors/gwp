# Idea N 独立调研：Ego Critical-Event Predictive Pretraining

- 日期：2026-08-12
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 工作树：`codex/idea-n-ego-event-pretrain-20260812`
- idea-discussion card：`98b1be4c-ae33-460d-a237-a0483cae1384`
- discussion runs：`7d63caeb-85c6-4a80-8622-88f2d560bf03`、`3cf8a9c5-b32e-4fd1-94b6-8d5569e44e71`
- 本轮范围：文献、目标函数、数据契约、接口、benchmark 与停止条件；**不实现、不训练、不使用 GPU、不连接服务器**

## 0. 结论先行

### 0.1 裁决

**Conditional Go，且只能作为 SM-FET / `minimal-r1d` 的 observation-pretraining 论文主线候选；以“ego 视频预测下一动作/过程”作为独立 novelty 则 No-Go。**

最接近、也最危险的冲突是 [EgoWAM](https://arxiv.org/abs/2607.08436)：它已经明确研究 human ego 数据如何通过 world-prediction target 改善 robot policy，并受控比较 Pixel、DINO 与 3D flow。再加上 [DreamDojo](https://arxiv.org/abs/2602.06949)、[V-JEPA 2](https://arxiv.org/abs/2506.09985)、[XSkill](https://arxiv.org/abs/2307.09955)、[MOTIF](https://arxiv.org/abs/2602.13764) 和 contact-centered anticipation，以下说法均不可再用：

- 首次从人类第一视角视频学习操作动力学；
- 首次用 future prediction 促进 human-to-robot transfer；
- 首次用离散 skill/event latent 跨 embodiment；
- 首次以 contact onset/release 组织 action anticipation；
- 首次让视频 world model 经过少量 robot data 适配到控制。

唯一尚可守的窄主张是：

> **从已经结束的 past critical windows 中，以严格 causal 的方式预测下一项有限 interaction event：`phase/kind + effect + focus role + 1..2 个 purpose–role–part–relation constraints + censored time-to-event`；human 阶段只学习 `minimal-r1d` 的 domain-neutral shared semantic key，不预测 robot assignment、metric locator、resource receipt 或 action。robot 阶段仅用 robot 数据完成 event perception、binding 与 action fine-tuning；action expert 对任何 human frame、human hidden、retrieval embedding、soft event logits均不可见，只能读取 canonical hard packet、完整 public robot state、structured text 与 diffusion noise/timestep。**

这个主张要成立，必须同时证明四件事：

1. **prediction，而非标签 auxiliary**：next-event objective 胜过使用同样标签的 current-event classification / masked-event modeling；
2. **finite semantics，而非任意 latent capacity**：胜过同容量 equal-bit / VQ latent；
3. **human pretraining，而非更多 robot compute**：在匹配 robot data、steps、parameters、FLOPs 后提高低数据 robot policy；
4. **strict route，而非 hidden bypass**：固定 packet、public state、text、noise 后，替换 human reference/hidden 不改变 pre-safety action，`max_abs <= 1e-6`。

任一项失败，论文标题不能再是 Ego Critical-Event Predictive Pretraining；最多保留为 SM-FET planner 的辅助预训练。

### 0.2 推荐结构

三种互斥结构中，推荐 **A：Factorized Causal Next-Event Model（FCNE）**：

| 结构 | 预测目标 | 优点 | 核心风险 | 裁决 |
|---|---|---|---|---|
| A. FCNE | 显式 next `phase/effect/focus/constraint-set/hazard` | 和 `minimal-r1d` shared key 一致；可审计、可干预、可处理 unknown/censor | 标签成本；容易被 task text/事件频率 shortcut | **主方案** |
| B. Event-JEPA | 预测 next event window 的 teacher latent，再线性读出 event | 可用大量无标签 human video；是强 representation baseline | 与 V-JEPA / EgoWAM 过近，latent 可能带 human appearance | **强 baseline，不进入 action** |
| C. Event-conditioned video generation | 生成 future pixels/video，再从生成结果提事件 | 可见地建模未来 | 与 DreamDojo/DreamZero/LOME 重叠；昂贵；透明液体易幻觉 | **No-Go 主线** |

### 0.3 一句话推荐

**先做 36 human + 36 robot 的同任务、many-to-many semantic-key pilot；先过 annotation、causal next-event、oracle sufficiency 和 strict isolation 四道门，再谈大规模 ego pretraining。现有 66 个 human pour 与旧 88 个 robot grab 任务不匹配，只能做标注可行性，不能证明 transfer。**

---

## 1. 问题定义：不是 ego 语义泛化，而是有限操作过程动力学

### 1.1 事件时钟

把一段操作写为离散事件序列：

```text
e_1 -> e_2 -> ... -> e_k -> e_{k+1}
```

每个已完成事件 `e_i` 有一个 past-only critical window：

```text
W_i = {entry_i, turn_i, exit_i, observability_i}
```

- `entry`：关系/接触变化前或刚进入事件；
- `turn`：最大姿态、方向、接触模式或 object-state 变化附近；
- `exit`：事件结束、关系改变或 outcome 可判附近；
- 三者都必须在 prediction anchor 之前；不可用下一事件的 turn/exit 作为输入；
- 某姿态不存在或不可见时保留 field-level mask/reason，不能以最近帧复制。

推荐的预测 anchor 是 **事件 `e_k` 的 reviewed exit 之后**。这样在离线与在线中都可定义同一个问题：

```text
p(e_{k+1}, Delta t_{k+1} | W_{1:k}, e_{1:k}, task_text)
```

online 变体允许当前事件未结束，但只能看到 anchor 前的 prefix，并把未出现的 `turn/exit` 标为 unavailable；不能用 future-aligned crop 构造输入。

### 1.2 目标不是 verb–noun next action

`e_{k+1}` 不是 Ego4D LTA 的自由 verb–noun，也不是机器人 action chunk。它是有限、任务本体冻结的 interaction event：

- `phase/kind`：`APPROACH_ALIGN / ACQUIRE / CONTROLLED_CONTACT / REGRASP / RELEASE / VERIFY`；
- `effect`：`NONE / TRANSLATE / ROTATE / ARTICULATE / FLOW / DEFORM / TRANSFER`；
- `event_focus_role`：source/manipulated、tool、destination/receptacle、support/fixture；
- `constraint roster`：1..2 个 `purpose + primary_role + primary_part + secondary_role + target_relation`；
- `time-to-next-event`：离散 hazard / interval-censored target；
- `NULL`：仅在 EXHAUSTIVE review 确认没有 active/future-in-horizon event 时合法；missing/occluded 不得变 NULL。

预测的是“接下来哪种有限关系/效果变化会发生”，不是“人手下一帧怎么动”。entry/turn/exit 的手姿与 flow 只作 train-only visible auxiliary。

### 1.3 可证伪 hypothesis

> 在匹配 architecture、human/robot data、compute 和 action route 后，显式预测 next finite interaction event，比 masked reconstruction、generic JEPA、current-event auxiliary、verb–noun anticipation、同容量无语义 latent 更能提高 target-robot low-data sample efficiency；增益必须穿过 canonical hard event packet，而非 human appearance 或 planner hidden。

如果只有 representation probing 提升、robot action 不提升，hypothesis 失败。

---

## 2. Zotero-first 与证据边界

### 2.1 本轮实际检索

按 `idea-discussion` workflow 同步个人 Zotero，`synced_count=100`。长 query 的自动 FTS 发生明显主题漂移，错误召回 failure-recovery；因此没有采用自动 agent 的 novelty 结论，而是直接复查本地 Zotero 数据库中题名与摘要。最相关条目为：

| Zotero key | 条目 | 对本题的约束 |
|---|---|---|
| `EHQ3JTCA` | EgoWAM | 最强冲突：human ego world-target 对 robot policy transfer 的 controlled study 已存在 |
| `FJFKKNLF` | DreamDojo | 44k 小时 ego human video + continuous latent action + robot post-training 已存在 |
| `RALYIRFV` | V-JEPA 2 | action-free video JEPA + 小量 robot action-conditioned post-training/planning 已存在 |
| `LP4RBJUQ` / `YGQWQ9PE` | World Action Models are Zero-shot Policies | video/action joint WAM 与 human/other-robot video transfer 已存在 |
| `V7LLF6VJ` | XSkill | 无标签 human+robot video 的 discrete cross-embodiment skill prototypes 已存在 |
| `WVMXS4GN` | Causal World Modeling for Robot Control | video/action autoregressive causal modeling 已存在 |
| `XA2HCL84` | LOME | human body/hand conditioned ego future video 与 physical effect generation 已存在 |

本地库还含 VLA/robot policy 笔记 `RT-1`、`RT-2`、`Octo`、`RoboBrain` 等，但它们不是本题最强 novelty 冲突。

### 2.2 证据等级

- `[R]`：原论文/arXiv/CVF/Ego4D 官方页面直接支持；
- `[Z]`：个人 Zotero 中已有条目，且本轮核对 title/abstract；
- `[I]`：canonical 仓库代码或已冻结内部 contract；
- `[P]`：本调研提出的设计，不是已实现事实。

2026 年工作多为近期 arXiv；本文只用其公开方法/实验声明界定 novelty，不把尚未独立复现的数值当作项目事实。

---

## 3. 前例矩阵：Idea N 剩下什么

### 3.1 通用视频 masked pretraining

| 工作 | 已覆盖 | 不能再声称 | Idea N 的残余 |
|---|---|---|---|
| [VideoMAE](https://arxiv.org/abs/2203.12602) | 高比例 tube masking，重建 pixels；通用 video representation | masked video pretraining 能学 motion | 不重建随机像素，而从 past completed events 预测 future finite event/effect/relation |
| [MaskFeat](https://arxiv.org/abs/2112.09133) | mask 后预测 HOG/feature | masked feature target 比 pixel 更适合作为 pretext | target 是 task-grounded event grammar，并要求 action transfer 与 causal intervention |

VideoMAE/MaskFeat 是必要 baseline，不是最强冲突。若 Idea N 只是在事件窗口上跑 VideoMAE，再接分类头，则没有方法 novelty。

### 3.2 Procedure planning 与 Ego action forecasting

| 工作 | 已覆盖 | 与 Idea N 的关键差异 |
|---|---|---|
| [Procedure Planning in Instructional Videos](https://arxiv.org/abs/1907.01172) / [PDPP](https://arxiv.org/abs/2303.14676) | 从当前 observation 与 goal 预测多步 action plan；学习 latent/plannable state | goal-conditioned procedure steps，不是局部物理 event/effect；没有 finite robot packet 或 human-pixel isolation |
| [AntGPT](https://arxiv.org/abs/2307.16368) | Ego4D/EPIC 的 autoregressive/top-down future verb–noun sequence | 预测语义 action sequence；可强依赖语言 procedure prior；不验证物理关系变化或 robot action route |
| [Ego4D LTA](https://arxiv.org/abs/2207.12080) | 长期 human future action sequence 与 intention conditioning | Idea N 只能声称更低层、agent-invariant event dynamics；必须设 text-only/Markov baselines |
| [Forecasting Action through Contact Representations](https://arxiv.org/abs/2102.00649) | TTC、contact anticipation map、next active object、contact-delineated state graph | 已否定“首次 contact-centered anticipation”；残余是 effect/role-relation/unknown-censor + strict robot packet |

因此“next event prediction”本身不新。必须同时有 **有限 effect/relation target、honest missingness、minimal-r1d alignment、robot transfer 和 route intervention**。

### 3.3 JEPA / latent future prediction

[V-JEPA 2](https://arxiv.org/abs/2506.09985) 已经以 action-free internet video JEPA 学习 latent future，并在少量 DROID robot video 上 post-train action-conditioned world model做 planning。[R][Z]

Idea N 的差异不是“latent prediction 更抽象”，而是：

1. 目标是外显有限事件，不是 dense latent；
2. unknown/censor/missing 是 label semantics，不强迫预测不可观测内容；
3. event token 与 robot action route 共用 schema；
4. equal-capacity latent 是必要对照。

若 Event-JEPA latent 胜过显式 event，结论应是 generic latent predictive learning 更有效，Idea N 的 finite-event 主张失败。

### 3.4 VLA reasoning / affordance / spatiotemporal interface

| 工作 | 已覆盖 | Idea N 不可声称 | 可守差异 |
|---|---|---|---|
| [CoA-VLA](https://arxiv.org/abs/2412.20451) | object/grasp/spatial/movement affordance chain直接指导 action | 首次显式中间 reasoning/affordance chain | past→next physical event prediction；不是 current scene affordance prompting |
| [ST-VLA](https://arxiv.org/abs/2603.13788) | human 300k episodes、3D trajectory、4D smooth mask、policy interface | 首次 human spatiotemporal supervision bridge robot action | non-reconstruction finite event target；metric 3D/4D 不从 human pixels直达 action |
| [VLAff](https://arxiv.org/abs/2608.05215) | 204K ego episodes、heatmap、MANO grasp、hand trajectory，再转 executable action | 首次 ego affordance/grasp/trajectory transfer | 禁止 human MANO/trajectory/actionable heatmap进入 action；只迁移 finite process prior |
| [BridgeACT](https://arxiv.org/abs/2604.23249) | tool/target affordance、3D motion affordance、closed-loop controller、人视频直接部署 | 首次 human-video tool-target bridge 或零 robot demo | Idea N 明确需要 robot action data；强调 auditable finite event，不是 3D flow execution |

这组工作使“entry/turn/exit 多姿态”非常危险：如果 entry/turn/exit pose、flow、heatmap、MANO 或 completed 3D 进入 action route，Idea N 就会退化成 VLAff/ST-VLA/BridgeACT 类路线，失去 residual novelty。

### 3.5 WAM / video world models

| 工作 | 已覆盖 | Idea N 的必要比较 |
|---|---|---|
| [EgoWAM](https://arxiv.org/abs/2607.08436) | 固定 backbone/action head/data mixture，比 Pixel、DINO、3D flow human ego world targets | 复刻其 controlled design，加入 Event target；否则无法归因 |
| [DreamDojo](https://arxiv.org/abs/2602.06949) | 大规模 ego human video、continuous latent action、robot post-training、world generation | event target vs latent action / future video 的低数据效益与成本 |
| [World Action Models are Zero-shot Policies / DreamZero](https://arxiv.org/abs/2602.15922) | video-action joint WAM、实时闭环、human/other-robot video transfer | strict event packet 是否在小模型/小数据下胜过 dense video dynamics |
| [LOME](https://arxiv.org/abs/2603.27449) | human body/hand conditioned ego interaction video与液体效果 | transparent unknown 不能用生成视频“补真值”；video generation 只作 baseline |

最强 reviewer 问题会是：**“这是不是 EgoWAM 的第四种 world target？”** 正确回答是“是一个强相关 target family，但主创新必须落在 finite schema、censor/missing contract、human→robot compilation boundary 与 causal action route；实验也必须遵循 EgoWAM-style matched target comparison”。如果没有这一组对照，不够发表。

### 3.6 Skill discovery：MOTIF / XSkill

| 工作 | 已覆盖 | Idea N 的残余 |
|---|---|---|
| [XSkill](https://arxiv.org/abs/2307.09955) | 从无标签 human+robot videos 发现 cross-embodiment skill prototypes，并用 diffusion policy 映射 action | human-only pretrain 先学显式 event；robot action阶段不需要 human prompt/retrieval；semantic code 可审计 |
| [MOTIF](https://arxiv.org/abs/2602.13764) | VQ action motifs、progress alignment、embodiment adversarial、few-shot transfer | next-event causal prediction不是当前 motif classification；同容量 learned motif 必须作 baseline |

若 VQ/XSkill-style prototype 在相同容量与 robot data 下相同或更好，不能声称显式 semantic event 必要；最多说它可解释。

---

## 4. 推荐目标：human shared semantic event，而不是 robot packet

### 4.1 `minimal-r1d` 的正确边界

内部 `minimal-r1d` contract 的 robot wire 是 `32 bytes / 5 hard tokens`，结构 grammar 包含：

- header：phase、effect；
- 每个 constraint：function、assignment、primary role/part、metric locator kind/payload、secondary role、relation；
- robot public state、metric codebook、capability profile、runtime resource 与 held-state receipts；
- action 条件只允许 hard tokens、public state、structured text、noise/timestep。

**human-only 视频不能监督完整 32-byte packet。** human 阶段只允许预测 shared semantic key：

| 字段 | human-only 是否监督 | 进入 robot hard packet的方式 |
|---|---:|---|
| task family / phase / effect | known 时监督；unknown只留在 sibling field target | 仅 complete-known target可成为 semantic header候选 |
| focus role | complete-known 时是 | 由 constraints机械派生；不是额外 action field |
| purpose：effect/support/observation | known 时是 | robot binding 映射到 OPERATE/STABILIZE/MONITOR |
| primary role/part | 可观察、reviewed且known时是 | robot compiler验证 task ontology/capability |
| secondary role / relation | 可观察、reviewed且known时是 | robot compiler形成 intended relation |
| robot LEFT/RIGHT/MONITOR assignment | **否** | robot capability + runtime state绑定 |
| metric POINT/AXIS/FRAME/PATH locator | **否** | 只由 robot current observation、标定和 codebook产生 |
| resource/held receipts | **否** | robot runtime 每 anchor生成 |
| 25-D robot action | **否** | 只用 robot demonstration fine-tune |

factorized human predictor可以输出每字段的 known/unknown分布；只有 roster `EXHAUSTIVE`、所有 required shared fields known且 canonical validation通过时，其输出才记为 `shared_event_key_hat`。其它样本只是 sibling field predictions，不能生成 shared key，更不能称 `r1d_packet_hat`。只有 robot binder 在 robot evidence 完整时才可编译 canonical packet。

### 4.2 human shared target

推荐 next-event **annotation/training record**：

```text
y_(k+1) = {
  phase,
  effect,
  event_focus_role,
  roster_coverage,
  constraints[1..2] = {
    purpose,
    primary_role,
    primary_part,
    secondary_role,
    target_relation
  },
  boundary_status,
  time_interval,
  field_masks,
  provenance
}
```

`event_focus_role` 由 canonical sorted constraints 中第一个 `EFFECT_GOAL` 的 primary role 派生；VERIFY 无 effect goal 时由第一个 constraint 派生。它可作 consistency target，但不能扩大 action packet。这个 training record 比 shared key 多 boundary/mask/provenance；**只有 complete-known + EXHAUSTIVE 的语义子集才序列化成 `minimal-r1d` shared key**。

`relation_before`、`relation_after` 与 2D entry/turn/exit pose/change signature可保存在 sibling label中训练辅助 head；action-visible只保留当前 event 的 intended target relation与 finite effect，不把连续 pose/trajectory编进去。

### 4.3 为什么不直接预测 full sequence

第一版只做 one-step next event：

- 标注和 calibration 可控；
- 便于区分视觉证据与 procedure-language prior；
- 与在线 receding-horizon robot planning一致；
- 能用 rollout 评估 exposure bias，但不把 20-step LTA 变成主问题。

只有 one-step 过 gate 后才允许 rollout 2–4 events。不得一开始预测完整 procedure，再用 LLM 先验掩盖视觉不敏感。

---

## 5. 模型结构与训练 objective

### 5.1 FCNE 主结构

```text
past RGB windows W_1:k
  -> shared video encoder
  -> per-event entry/turn/exit pooling + observation mask
  -> causal event transformer
  -> next-event factorized decoder
       phase/effect/focus
       constraint-set (1..2)
       discrete hazard / censor

future event window W_(k+1)
  -> EMA target encoder (training-only, stop-gradient)
  -> optional JEPA target/probe; never enters robot action
```

per-event pooling必须只见 anchor 前帧。window proposal可以由 reviewed boundary 或过去 causal detector生成，但禁止从 target boundary反向裁剪 input。

### 5.2 总 loss

推荐：

```text
L = lambda_next * L_next_event
  + lambda_set * L_constraint_set
  + lambda_time * L_survival
  + lambda_delta * L_transition_consistency
  + lambda_pose * L_visible_multipose
  + lambda_jepa * L_event_jepa
  + lambda_obs * L_observability_aux
```

各项定义：

1. **`L_next_event`**：对 phase/effect/focus/roster count 做 masked categorical CE；只有 `KNOWN` 或 training-side field ontology有显式 `REVIEWED_UNKNOWN` code 的字段参与。后者不能因此进入 `minimal-r1d` shared key。
2. **`L_constraint_set`**：1..2 个 constraint。主实现应按 frozen ordinal canonical sort；若标注顺序不稳定，可在 train-only 用 Hungarian set loss，但输出必须 canonicalize 后再评价。
3. **`L_survival`**：next boundary 的 discrete hazard / interval-censored NLL。observed interval累计区间概率；right-censor最大化 horizon 内未发生概率；left-censor只监督初始 active state或 `T<=0`；boundary unknown完全 mask。
4. **`L_transition_consistency`**：effect 与 pre/post relation delta 必须相容，例如 `ACQUIRE` 后 touching/held relation不应仍是 known separate。只用 ontology允许矩阵，不从腕旋转猜 outcome。
5. **`L_visible_multipose`**：entry/turn/exit 的 2D relative axis、rotation sign、overlap/near/touching proxy；仅 visible support上训练；永远 train-only。
6. **`L_event_jepa`**：past context预测 future reviewed event-window的 EMA latent，teacher branch stop-gradient；这是 representation regularizer，不是 action condition。
7. **`L_observability_aux`**：预测 visibility/missingness reason用于 calibration；logits/hidden action-invisible，不能补 GT。

第一轮应把 `lambda_jepa=0`，先验证显式 next-event；第二轮再加 JEPA。否则正结果无法区分 finite target 与 generic latent future prediction。

### 5.3 必须分开的四种训练臂

| Arm | 标签/输入相同 | 唯一变化 | 回答问题 |
|---|---|---|---|
| Current-event aux | same human windows/labels | 预测当前事件，不预测下一事件 | 收益是否只是多标签 |
| Masked-event model | same event sequence | 随机 mask past token再恢复 | 收益是否只是 bidirectional context |
| Next-event FCNE | past-only causal | 显式预测下一事件 | prediction 本身是否有用 |
| Event-JEPA | past-only causal | 预测 future latent，不用 semantic CE | finite semantics 是否必要 |

所有臂在 robot stage必须匹配 parameters、robot steps、robot batches 与 action route。

---

## 6. Teacher forcing 与 causal mask

### 6.1 两层 teacher forcing

**事件历史层：** 训练时可用 reviewed `e_1..e_k` 作为 history token；同时必须报告 free-running rollout，用模型预测替换 history。主结果不能只报 teacher-forced NLL。

**target 字段层：** `e_(k+1)` 的 decoder按以下 shifted-right 顺序：

```text
BOS -> phase -> effect -> focus -> count
    -> constraint_0 fields -> constraint_1 fields -> EOS
```

每一 token 只能 attend past windows、past event tokens 和当前 target 的更早 token。label token本身不可同位输入。

### 6.2 mask 不变量

主 prediction prior必须满足：

```text
past visual/event tokens -> may attend past only
next-event query         -> may attend all past, never target/future pixels
target token j           -> may attend past + target tokens < j
EMA future target        -> loss branch only; no cross-attention back to prior
robot action token       -> may attend robot hard packet/public state/text/action history;
                            never human pixels/human hidden/soft event logits
```

canonical 代码已有可复用的 causal 思路：`world_action_model/models/transformer_wa_casual_mot.py:130-194` 明确 contact query可看 state/reference但不能看 noisy target action/future video，并可关掉 action→reference 直达路径。[I] Idea N 的新 event mask必须额外保证 target event tokens与 future EMA branch不可回流。

### 6.3 scheduled sampling

不推荐把 scheduled sampling作为 main method：它增加机制变量且常掩盖 calibration。顺序应为：

1. one-step teacher-forced causal prediction；
2. free-running 2–4 event rollout；
3. 若 rollout gap显著，再把 event-history dropout或 scheduled sampling作为二级消融；
4. robot action fine-tune默认 hard-detached predicted packet；hard-ST只作优化机制对照。

### 6.4 防未来泄漏

必须审计：

- input crop边界是否由 future event annotation决定；
- clip总长、padding位置、文件名、absolute timestamp是否泄漏 next event；
- normalization是否跨越 prediction anchor；
- target event的 entry frame是否错误进入 input；
- text是否直接包含下一步；
- pair ID、operator、object instance是否变成 sequence lookup key。

负控包括 future-label shuffle、past-window order shuffle、same-task distractor window exchange、time-padding randomization 和 text-only baseline。

---

## 7. human-only：哪些能监督，哪些不能

### 7.1 合法 supervised targets

只在 reviewed evidence支持时：

- event phase/kind 与 observed/interval/censored boundary；
- object/tool/receptacle/support 的功能角色；
- visible 2D relation：separate/near/touching/inside/contains/aligned-to-rim/supported-by；
- visible object effect：translate/rotate/articulate/deform；
- flow/transfer 仅在 stream/level/change 或人工 outcome evidence足够时；
- contact onset/hold/release，仅在 direct/temporal/multiview/sensor evidence足够时；
- entry/turn/exit 的 2D pose/change sign、hand-object overlap和 tool axis；
- task outcome，仅由人工/传感判据，不由动作姿态推断。

### 7.2 合法 self-/weak-supervised targets

- VideoMAE/MaskFeat-style past-window masking；
- JEPA-style past→future-window latent prediction，future target encoder stop-gradient；
- temporal order / event adjacency discrimination；
- object-track consistency与visible optical flow；
- change-point detector只作 proposal；
- VLM/pseudo-label只作预填与 reviewer queue，不升级为 GT。

### 7.3 human-only 不可监督

- target robot 25-D action、joint/wrist command、gripper aperture；
- LEFT_RH56 / RIGHT_OMNIPICKER assignment；
- OPERATE/STABILIZE/MONITOR 的最终 robot binding；
- metric POINT/AXIS/FRAME/PATH locator；
- robot reachability、WBC、collision、held transform与resource receipt；
- 被遮挡 contact 的确定真值、force/load/slip；
- transparent liquid 3D surface、volume、stream depth或 transfer amount；
- 由 completed depth、mesh、generative video、MANO retarget或VLM猜测产生的 oracle truth；
- human pose/flow/heatmap 到 robot action 的直连映射。

自监督目标可以使用未标视频，但不能在论文中把 latent/pseudo label称为 physical ground truth。

---

## 8. Missing hands 与 transparent unknown

### 8.1 missing hands：五件事必须拆开

1. participation：present / confirmed absent / reviewed unknown / not reviewed；
2. anatomical side：known / reviewed unknown / not reviewed；
3. observation：visible / partial / self-occluded / object-occluded / out-of-frame / blur；
4. tracker：active / inactive；
5. boundary：exact / interval / left-censored / right-censored / unknown。

规则：

- detector没框不等于 hand absent；
- `OCCLUDED/OUT_OF_FRAME/TRACKER_INACTIVE` 是 provenance/mask，不直接是 event token；
- 若确认有 hand-object edge但 side不可判，presence与其它 known fields可监督，actor/side mask=0；不得生成完整 oracle shared key；
- single-hand/bimanual是 EXHAUSTIVE roster review 的结论，不是“只看到一只手”的默认；
- model imputation可作 calibration baseline，但不能回写 annotation或解除 compiler fail-closed。

### 8.2 transparent container / liquid

只允许 visible、任务本体冻结的 observation：

- rim/aperture存在性与 image-plane aligned relation；
- visible stream existence；
- pour onset/offset/spill onset 的 observed/interval/censor；
- before/after coarse level或 transfer outcome，前提是人工/传感可判。

以下情况必须 `REVIEWED_UNKNOWN`、MISSING 或 packet reject：

- rim支持跨多个会导致不同 safe action 的 bins；
- 透明材质无 direct depth/normal；
- stream/contact-interface不可定位；
- 只看到 wrist tilt，未看到液体/结果；
- 视频生成或 depth completion给出看似合理结果；
- 关系/效果需要不可观察 3D 才能决定。

LOME/DreamDojo-style future video可以是生成 baseline，不能成为 transparent GT author。

### 8.3 loss contract

```text
KNOWN                                  -> field loss on
REVIEWED_UNKNOWN + schema UNKNOWN code -> UNKNOWN CE on
REVIEWED_UNKNOWN without UNKNOWN code  -> field mask 0; full key inadmissible
NOT_REVIEWED / MISSING                 -> field mask 0
CONFIRMED_ABSENT                       -> only where ontology has explicit ABSENT/NULL semantics
```

unknown不是 negative，missing不是 unknown，censor不是 boundary=window edge。这里的 UNKNOWN 属于 annotation/prediction ontology；`minimal-r1d` required shared field没有可执行 UNKNOWN 时，结果必须 key/packet inadmissible。

---

## 9. Human → robot transfer：不让 human pixels 进 action

### 9.1 三阶段，而非 joint hidden fusion

```text
H0: human video only
    past human windows -> FCNE -> shared semantic event prediction
    output: pretrained encoder/event-transformer weights

R0: robot event adaptation only
    robot RGB/state -> initialized FCNE -> robot-observed shared event
    robot binder/compiler -> assignment + metric locator + receipts -> hard r1d packet

R1: robot action fine-tuning only
    hard packet + robot public state + structured text + noise/timestep
    -> robot action expert -> 25-D action
```

human data不出现在 R1 dataset；human frame、retrieval、prompt video或human encoder activation不出现在 action batch。**预训练参数携带的统计先验不是 runtime input，但必须与 matched initialization baseline比较。**

### 9.2 唯一合法 action route

主方法采用 hard-detached：

```text
robot RGB -> robot event planner -> canonicalize/validate -> 5 hard tokens
                                                      |
robot public state + text + noise --------------------+-> action expert
```

禁止：

- `concat(human_hidden, action_hidden)`；
- soft event logits、confidence、visibility heatmap、missingness embedding进入 action；
- nearest human clip / human prompt video retrieval at inference；
- human entry/turn/exit pose或optical flow进入 action KV；
- planner hidden、EMA target hidden、allocator score或receipt metadata进入 action/cache；
- invalid/incomplete packet时 soft fallback到 visual hidden。

invalid packet只能 HOLD/REOBSERVE，或由 robot allocator在唯一、安全、有 receipt 的条件下重新绑定。

### 9.3 route audit

至少做：

1. **human swap**：固定 robot packet/public state/text/noise，替换任意 human clip/embedding，action `max_abs <= 1e-6`；
2. **planner-hidden swap**：固定 canonical packet，替换 planner hidden/logits，action不变；
3. **packet intervention**：合法交换 effect/relation/phase，action出现预注册局部变化；
4. **equal-bit**：同容量任意 hard code；若与 semantic相同，语义因果主张失败；
5. **action-hidden probe**：fixed packet后不能恢复 human identity/clip ID高于 chance；
6. **cache audit**：只保存 canonical packet/receipt-bound public inputs，不保存 human/planner values。

canonical 当前 contact planner已经有有用的不变量：planner hidden只在 `_plan_contact_logits()` 内存在（`world_action_model/models/transformer_wa_casual_mot.py:1253-1285`），而 `:1439-1445` 明确 22D→action projection 是唯一 planner-to-action route。[I] Idea N 必须保持并收紧到 `minimal-r1d` hard token route。

---

## 10. `minimal-r1d` 接口建议

### 10.1 最小接口

| 模块 | 输入 | 输出 | action-visible |
|---|---|---|---:|
| Human FCNE pretrainer | past human RGB windows、reviewed shared labels | pretrained weights、next shared key logits | 否 |
| Robot event planner | current/past robot RGB、robot state、text | shared key + observed geometry proposal | 否，直到 compile |
| Robot binder/compiler | shared key、profile、metric codebook、public state、runtime receipts | canonical 32-byte / 5-token packet或 reject | 是，hard only |
| Action expert | 5 hard tokens、public execution state、structured text、noise/timestep | 25-D action | — |
| Train-only probes | future target latent、multipose、visibility/missingness | metrics | 否 |

### 10.2 两个禁止混淆的 packet

```text
shared semantic key:
  human/robot 可共同标注和预测；无 assignment/locator/resource

robot executable r1d packet:
  robot evidence + capability/runtime binding后产生；含 assignment/locator；action-visible
```

论文图中必须画出中间 compiler boundary，不能用一个“event token”框把二者混在一起。

### 10.3 当前 canonical 的插入面（仅设计，不实现）

| 位置 | 当前事实 | 未来 Idea N 插入 |
|---|---|---|
| `world_action_model/contact_transition_schema.py:36-46` | 已有 transition/effect有限枚举 | 只借 ontology，不把 legacy v1 当 r1d |
| `world_action_model/contact_transition_schema.py:65-112` | 已有 22D segment字段与 soft/hard-ST/hard-detached route | 建立独立 versioned event schema；不可静默改 22D |
| `world_action_model/datasets/wam_lerobot_dataset.py:741-790` | reviewed sidecar、fingerprint、target/mask/confidence/condition arrays | 新增 versioned event-history sidecar与 past-only anchor manifest |
| `world_action_model/models/transformer_wa_casual_mot.py:130-194` | contact query不看 future/action的 causal mask | 增加 event-history/target-decoder triangular mask |
| `world_action_model/models/transformer_wa_casual_mot.py:1253-1403` | planner logits与 route transport | robot FCNE adapter、hard compile前不得送 action |
| `world_action_model/models/transformer_wa_casual_mot.py:1405-1445` | packet是 planner→action唯一 route | 替换为 validated r1d tokens，继续禁止 hidden bypass |
| `world_action_model/trainer/wa_casual_trainer_mot.py:957-1055` | contact target/oracle/mask及 grouped loss | 增加 next-event、survival、field mask loss；human batch action loss必须关闭 |

`minimal-r1d` 的 full codec/profile在 canonical `e0ebc692` 不是本轮实现对象；本设计只声明未来接口兼容，不宣称当前代码已经支持 32-byte/5-token executable packet。

---

## 11. Benchmark：先证明 event，再证明 transfer，再证明 action

### 11.1 B0：human-only event benchmark

数据来源分两类：

1. **external scale/pretraining**：Ego4D FHO/state change、EPIC-KITCHENS/VISOR、可审计的自有 ego video；不同数据集只对 ontology交集字段监督；
2. **internal clean evaluation**：同一 task ontology 下双标、裁决、group split 的 human clips。

评测任务：

- next phase/effect/focus；
- 1..2 constraint set prediction；
- time-to-next-event / censor；
- 2–4 event free rollout；
- unknown/missing calibration；
- task text / past visual evidence利用率。

不能把 Ego4D verb–noun label自动映射成 contact/effect GT；只能在人工 review 后进入 strict set。

### 11.2 B1：最小同任务 human–robot pilot

**36 human + 36 robot**，先只做 3 个 task families：

| Task | 必须覆盖的 event链 | 关键风险 |
|---|---|---|
| pour | acquire -> align -> rotate/flow -> return -> verify/release | transparent transfer unknown、stop |
| insert | acquire -> align -> controlled contact -> insert -> verify | relation与geometry混淆 |
| cap/open-close | acquire -> touching -> rotate/articulate -> verify/release | repeated turns、outcome |

每个 domain 每 task 12 episodes；其中 3 calibration、9 blind。pair 是 semantic-key many-to-many，不要求 frame/trajectory一一对齐。split先按 object family、operator/session、scene分组，再建 pair；禁止同物体同场景泄漏到 train/test。

这个规模只够 gate/pilot，不够最终 SOTA claim。若 pilot通过，paper-scale至少扩到 5 task families（再加 pipette、wipe），每 task robot data sweep `n={10,25,50}`、3 seeds、group-held-out test。

### 11.3 当前数据的正确用途

- 66 human ego pour：可做 annotation G0、window/event可预测性和 transparent unknown audit；
- 88 robot `grab_red_bottle`：可做 robot action pipeline sanity、oracle packet sufficiency的单任务试验；
- 二者任务不匹配：不能作为 positive human–robot pair，不能验证 transfer，也不能计算 cross-domain retrieval。

---

## 12. Baselines 与消融

### 12.1 representation/pretraining baselines

1. robot from scratch；
2. ImageNet/DINO frozen initialization；
3. VideoMAE；
4. MaskFeat/HOG target；
5. V-JEPA/Event-JEPA latent target；
6. EgoWAM-style Pixel / DINO / 3D-flow world target；
7. Forecasting-through-contact：TTC/contact/next-active-object；
8. Ego4D verb–noun LTA / Markov action sequence；
9. XSkill/MOTIF-style VQ discrete skill，容量匹配；
10. current-event aux、masked-event model、next-event FCNE。

大型方法无法全量重训时，可以使用 released frozen features或小型 matched instantiation，但不得声称全面击败原 14B/44k-hour 系统。核心因果表必须在同 backbone、同 data/steps/FLOPs 下比较 target。

### 12.2 action-route baselines

| Route | action输入 | 用途 |
|---|---|---|
| direct robot visual | robot RGB hidden + state/text | performance ceiling，不允许 human reference |
| aux-only direct | direct视觉 + event auxiliary | 检查是否只是多监督 |
| hard semantic predicted | predicted canonical packet | 主方法 |
| hard semantic oracle | oracle robot packet | sufficiency ceiling |
| hard equal-bit | 同容量任意 finite code | semantic factorization因果对照 |
| VQ motif | learned discrete code，同token/bit预算 | XSkill/MOTIF-style对照 |
| soft semantic | soft logits | 机制对照，不作为 strict claim |
| hard detached vs hard-ST | forward packet相同 | action gradient是否必要 |

### 12.3 必做消融

- entry/exit vs entry/turn/exit；
- random uniform windows vs reviewed critical windows；
- phase only / +effect / +role-relation / full shared key；
- next-event vs current-event；
- one-step vs free rollout；
- text full / delexicalized / text-only / no text；
- unknown正确处理 vs mask all vs误当 negative；
- exact boundary vs interval/censored survival；
- human pretrained encoder frozen / robot fine-tuned / random init；
- human data amount与 robot data amount正交 sweep；
- transparent/missingness strata；
- semantic code vs equal-bit/VQ；
- human future EMA target on/off。

---

## 13. 指标与统计

### 13.1 Event prediction

- phase/effect/focus field macro-F1、balanced accuracy；
- constraint set exact match、field macro-F1、set precision/recall；
- next-event NLL、Brier、ECE；
- top-k recall与 sequence edit distance；
- hazard NLL、boundary MAE P50/P90、interval coverage；
- 2–4 event rollout exact prefix / edit distance；
- unknown/censor coverage-risk curve；
- per-observation-reason degradation；
- text-only gap、window-delete/exchange sensitivity。

### 13.2 Transfer

- robot event label efficiency：达到固定 event F1需要的 robot labels；
- human→robot linear/frozen/fine-tuned transfer gap；
- same semantic key human–robot retrieval recall，只作 probe；
- embodiment classifier accuracy（希望接近 chance，但不能以 adversarial loss替代 action结果）；
- object/session/group-OOD event metrics。

### 13.3 Robot action

- task/phase success；
- correct first contact；
- correct relation transition；
- regrasp/recovery；
- correct stop/release；
- event order violation；
- spill/drop/collision/post-success overrun；
- success-vs-robot-data curve AUC；
- predicted packet恢复 oracle增益比例；
- action latency与 HOLD/REOBSERVE频率。

所有 robot主比较至少 3 seeds；group-cluster bootstrap 95% CI；安全事件不能只报平均 success。多 task / data-regime比较需预注册主 endpoint，避免事后挑 task。

### 13.4 因果指标

定义三个窗口检验：

- **Keep**：只保留预测器选中的 past critical windows，应维持大部分 oracle/full-history性能；
- **Delete**：删除这些窗，next-event/action性能应显著下降；
- **Exchange**：与 same-task distractor交换特定 event window，应产生局部、语义对应的下降。

交换 contact/acquire窗应主要损伤 first contact；交换 effect/release窗应主要损伤 stop/release。若任意换视频都全面崩溃，更像 identity/domain shift，不是 event causality。

---

## 14. Novelty 与可发表性

### 14.1 novelty 评分

| 维度 | 评分 | 判断 |
|---|---:|---|
| “用 ego 视频预测未来/过程” | 1/5 | 已高度覆盖 |
| “next contact/action event” | 1.5/5 | contact forecasting、Ego4D LTA、procedure planning均覆盖 |
| “finite phase/effect/role-relation target” | 2.5/5 | event/state/effect primitive不新，具体冻结组合尚有空间 |
| “human shared key -> robot executable packet” | 3.5/5 | 受 XSkill/MOTIF/VLAff/BridgeACT挤压，但严格 compiler boundary较少见 |
| “UNKNOWN/censor + strict no-human-pixel route + equal-bit causal audit” | 4/5 | 最可守、也是实验最难部分 |
| 当前数据就绪度 | 1/5 | 无同任务人机 action/effect benchmark |

### 14.2 是否够 novel

**只做模型：不够。** 一个 event transformer + next-event CE 很可能被审稿人归为 Ego4D anticipation / procedure forecasting 的 schema change。

**做成完整证据链：可能够。** 至少需要：

1. frozen finite shared-event ontology 与 public annotation protocol；
2. missing/unknown/censor 的可重复标签与 benchmark；
3. EgoWAM-style matched world-target comparison；
4. same-task human–robot low-data transfer；
5. oracle/predicted/equal-bit/current-aux factorial；
6. strict hard route 与 human swap intervention；
7. 多任务闭环 robot improvement与安全不退化。

更现实的投稿形态是：**SM-FET 的核心 representation/pretraining + benchmark paper**，而不是独立“新视频 pretraining loss”。如果只做到 B0+B1 offline，适合作为 workshop/benchmark/technical report；要作主会机器人 learning paper，必须有 robot closed-loop causal result。

### 14.3 最强论文标题边界

可以：

> Causal Finite-Event Predictive Pretraining from Egocentric Human Video for Auditable Robot World-Action Models

不可以：

- first world model from human ego video；
- learns robot actions from human video alone；
- embodiment-invariant by construction；
- understands transparent liquid physics；
- universal manipulation process model；
- finite packet is sufficient before oracle gate。

---

## 15. Stop gates

### G0：标注可重复

先用 3 calibration + 12 blind human episodes双标、全分歧裁决：

- phase/effect/role/relation `kappa >= 0.67`；
- boundary status `kappa >= 0.70`；
- observed boundary P50 `<= 5 frames` 或 `<=250 ms`；
- critical fields reviewed-unknown/missing `<=40%`；
- 每个主字段至少两个非稀有类别，每类至少 20 comparable events后才作主 metric；
- roster/constraint overflow `<=5%`。

**停止：** 指南最多修订两轮仍不达标；或 >40% 的 next effect/role-relation在现有 RGB/传感中不可判。删掉不可靠字段；若剩余只是 coarse phase，停止独立 Idea N。

### G1：next-event 真的利用 past visual evidence

- FCNE相对 text-only/Markov/current-event baseline：next-event macro-F1至少 `+5 pp`；
- window delete造成显著下降，exchange造成局部语义下降；
- ECE `<=0.10`；
- free-running 2-event prefix accuracy至少保留 teacher-forced增益的 70%；
- group-held-out中结果仍成立。

**停止：** task text或 object ID即可达到同等结果；past-window shuffle不降；收益只在 random-window split存在。

### G2：predictive objective 不是同标签 auxiliary

在相同标签、encoder、compute下：

- next-event FCNE至少胜 current-event aux或 masked-event主指标 `+3 pp`，或用同量 robot labels把 event F1提高 `+5 pp`；
- semantic target胜 Event-JEPA/VQ/equal-bit至少一项预注册 robot-transfer endpoint。

**停止 predictive/semantic headline：** current-event或 generic latent相同/更好。可以保留最佳 auxiliary，但不要重命名结果。

### G3：oracle `minimal-r1d` sufficiency

在同任务 robot 数据上，oracle shared event + robot binder/hard packet：

- task success距 direct robot visual ceiling不超过 `5 pp`；
- correct first contact / stop-release不退化；
- spill/drop/collision不增加；
- invalid packet可安全 HOLD/REOBSERVE。

**停止 finite action route：** oracle仍低 direct >5 pp，或必须让 action直读 continuous pose/visual hidden才成功。

### G4：human pretraining带来 robot label efficiency

在 robot data sweep与 3 seeds：

- 相对最佳 matched VideoMAE/MaskFeat/V-JEPA/EgoWAM-target baseline，group-OOD success `>=+5 pp` 且 95% CI下界 `>0`；或达到固定 success所需 robot demos减少至少 30%；
- predicted packet恢复 oracle packet增益 `>=70%`；
- safety endpoint无退化；
- 增益不由单一 task/session驱动。

**停止 transfer headline：** 只提升 offline event F1、不提升 action；或只比 from-scratch强，不胜 video pretraining强 baseline。

### G5：strict isolation

- fixed hard packet + fixed public state + fixed text/noise下，human/reference/planner-hidden swap action `max_abs <=1e-6`；
- action cache不含 human/planner/soft values；
- invalid packet无 visual fallback；
- packet intervention有局部可解释 action effect；
- equal-bit不能匹配 semantic全部结果。

**立即停止 auditable route claim：** 任一 hidden bypass、cache leakage、human retrieval或 soft fallback被发现。

### G6：paper-scale

- 至少 5 task families；
- subject/object/scene group-held-out；
- paper-scale same-task human/robot数据；
- closed-loop robot结果与完整 baselines；
- annotation、split、schema、route receipts可复现。

**停止主会投稿扩量：** pilot G0–G5任一未过；不要用更大 backbone、更多 pseudo-label或生成式 completion掩盖失败。

---

## 16. 最小推进顺序

1. 冻结 3-task event ontology与 shared-key mapping；不扩 `minimal-r1d` core fields。
2. 在 15 human clips上做 G0 双标，包含 missing hands与 transparent困难段。
3. 用现有 66 human pour只做 label coverage、boundary/censor与 text shortcut audit。
4. 构造轻量 frozen-feature baselines：Markov/text-only/current-event/next-event/VideoMAE-or-JEPA probe；不训练 WAM。
5. 收集或整理同任务 36 human + 36 robot pilot；先 split，再建 semantic-key many-to-many pairs。
6. 在 robot 数据上做 oracle shared key / direct visual / equal-bit sufficiency。
7. 只有 G0–G3通过，才实现 human pretrained FCNE→robot event adapter。
8. 只有 offline predicted event恢复 oracle增益，才进入 robot closed-loop action fine-tuning。
9. 只有 G4–G5通过，才扩到 5 tasks和 paper-scale。

本顺序的目标是优先证伪“标签不可判”“event不够充分”“收益只是通用 video pretraining”“human hidden偷渡”四种最常见失败。

---

## 17. 最终 Go / No-Go 表

| 项目 | 当前裁决 | 理由 |
|---|---|---|
| 独立“ego next-event prediction”论文 | **No-Go** | procedure/LTA/contact anticipation前例充分 |
| 通用 video pretraining novelty | **No-Go** | VideoMAE/MaskFeat/V-JEPA已覆盖 |
| human video world-model transfer首创 | **No-Go** | EgoWAM/DreamDojo/DreamZero已直接覆盖 |
| discrete cross-embodiment latent首创 | **No-Go** | XSkill/MOTIF已覆盖 |
| finite event target作为 EgoWAM-style新 world target | **Conditional Go** | 必须 matched target comparison |
| unknown/censor + compiler boundary + strict route | **Go 研究** | residual novelty最强，尚待数据证明 |
| 现有 66 ego + 88 grab直接验证 transfer | **No-Go** | 任务不匹配、无正 pair |
| 36+36同任务 pilot | **Go** | 最小可证伪规模，不等于 paper-scale |
| human-only预测 full r1d/action | **No-Go** | assignment/locator/resource/action不可监督 |
| Event-JEPA / video generation | **baseline only** | 强前例，且有 appearance/幻觉风险 |
| robot action fine-tuning | **Conditional Go** | 只能在 oracle/event/isolation gates后进行 |

最终判断：**Idea N 值得作为 SM-FET 的高风险、高信息量主实验继续，但不是因为“事件预测”新，而是因为它把预测目标、缺失语义、human/robot compilation boundary 和 action causal route锁成一条可证伪链。若这条链不能在小 pilot胜过 EgoWAM-style target与同容量 latent，就应尽早停止。**

---

## 18. 原始来源索引

### Masked / predictive video representation

- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)
- [Masked Feature Prediction for Self-Supervised Visual Pre-Training / MaskFeat](https://arxiv.org/abs/2112.09133)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)

### Procedure / anticipation / contact

- [Procedure Planning in Instructional Videos](https://arxiv.org/abs/1907.01172)
- [PDPP: Projected Diffusion for Procedure Planning in Instructional Videos](https://arxiv.org/abs/2303.14676)
- [AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?](https://arxiv.org/abs/2307.16368)
- [Intention-Conditioned Long-Term Human Egocentric Action Forecasting](https://arxiv.org/abs/2207.12080)
- [Forecasting Action through Contact Representations from First Person Video](https://arxiv.org/abs/2102.00649)
- [Ego4D FHO annotation guidelines](https://ego4d-data.org/docs/data/annotation-guidelines/)

### VLA / affordance / human-to-robot

- [CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](https://arxiv.org/abs/2412.20451)
- [ST-VLA: Enabling 4D-Aware Spatiotemporal Understanding for General Robot Manipulation](https://arxiv.org/abs/2603.13788)
- [VLAff: Vision-Language-Affordance Model for Unified Actionable Affordances](https://arxiv.org/abs/2608.05215)
- [BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances](https://arxiv.org/abs/2604.23249)

### WAM / video world model

- [EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data](https://arxiv.org/abs/2607.08436)
- [DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos](https://arxiv.org/abs/2602.06949)
- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model](https://arxiv.org/abs/2603.27449)

### Cross-embodiment skill

- [XSkill: Cross Embodiment Skill Discovery](https://arxiv.org/abs/2307.09955)
- [MOTIF: Learning Action Motifs for Few-shot Cross-Embodiment Transfer](https://arxiv.org/abs/2602.13764)

### 内部上位 contract（跨工作树只读参考）

- `gwp-agent-research-docs-integration/docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`
- `gwp-agent-sm-fet-missing-hands/docs/research/sm_fet_missing_hands_contract_20260812.md`
- `gwp-agent-sm-fet-transparent/docs/research/sm_fet_transparent_nonreconstruction_contract_20260812.md`
- `gwp-agent-idea-h-event-transition/docs/research/idea_h_event_state_transition_wam_review_20260812.md`
- `gwp-agent-idea-i-causal-critical-windows/docs/research/idea_i_causal_critical_windows_review_20260812.md`
- `gwp-agent-idea-j-pose-contact-change/docs/research/idea_j_pose_contact_change_signature_review_20260812.md`
- `gwp-agent-idea-k-effect-equivalence/docs/research/idea_k_effect_equivalent_interaction_quotient_review_20260812.md`
- `gwp-agent-idea-l-motif-dictionary/docs/research/idea_l_sparse_interaction_motif_dictionary_review_20260812.md`
