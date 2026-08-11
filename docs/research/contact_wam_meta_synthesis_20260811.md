# Contact-WAM 五案 Meta-Synthesis 与 op38 决策书

- 日期：2026-08-11
- 代码审计基线：`d9234a6f5566c6d22f0ca8ca5d39988de1fe0e88`
- 隔离分支：`codex/contact-wam-meta-synthesis-20260811`
- 输入：Idea A/B/C/D/E 五份独立 memo 与上述基线代码
- 本轮边界：只做综合决策与只读代码审计；没有新增外部文献检索、训练、GPU、模拟器或 canonical 修改

## 0. 最终决定

**唯一推荐主线是 Idea A：Auditable Finite Contact-Transition WAM。** 但主张必须从“动作只能读 contact”收缩为：

> **在给定 instruction text、proprioceptive state、embodiment ID、扩散 timestep/noise 与 noisy action 的条件下，action expert 能获得的全部 current-reference-image-derived 信息，只能通过 Q=12、B=16 的 canonical hard contact-segment packet；该 packet 的 action-visible alphabet 上界为 `48,673^12` 个状态，即 186.850009 bit。容量/监督匹配的 direct、无语义有限码与 packet intervention 用于验证它是否被因果使用。**

这是一条 **conditional visual bottleneck** claim，不是 total-information bottleneck，也不是完整 task-semantic bottleneck。当前 action expert 直接读取 text；这不破坏上述有条件视觉主张，但禁止论文声称所有任务语义都经过 22D。

最多保留两个辅助 claim：

1. **D：coverage/censor-aware supervision contract。** 区分 `unknown / positive_only / exhaustive`、field missingness 与 window censor，并在真实双标通过后用 censor-aware event likelihood 验证校准收益。当前只能称 `missingness-masked + censor-encoded`，不能称 MNAR-debiased 或 censor-aware trained。
2. **E：action-mediated realizability audit。** 在存在 native/sensor-confirmed robot action→effect 数据后，用不读取 text/contact/planner 的 frozen `F_R` 和 same-noise、physics-feasible intervention 验证 packet 是否被动作真实实现。当前只允许做接口与语法级 intervention harness，不允许训练或宣称 `F_R`。

Idea B 暂作未来 asymmetric hardware stress test，不列方法 claim；Idea C 暂作标注可行性/训练期辅助候选，不列论文 claim。

当前总状态：

- **Go**：op38 的纯代码路由控制、精确容量 codec、fail-closed receipt、CPU tiny audit、语法级 same-noise harness。
- **Conditional Go**：完成真实 A/B contact 标注、group provenance、oracle sufficiency 后的小规模因果实验。
- **No-Go**：当前 5B 三 seed 训练、frozen effect model 训练、液体 physics claim、production OOD claim、目标真机与顶会 ready claim。

## 1. 综合证据边界

本轮按 `idea-discussion` 的结构做综合，但遵守“不再做泛化资料搜集”的要求，没有重新运行外部 retrieval。所有前例判断只继承五份 memo 已核实的原始来源；本轮新增的是跨 memo 的冲突消解和代码级实验定义。

五份 memo 的共同事实是：

1. contact/affordance intermediate、ego-to-robot、时空 contact representation、contact-conditioned control、离散 latent、action-effect consistency 均已有强前例；
2. 当前可守空间不在任一组件，而在“有限、语义、严格视觉路由、missingness/censor 诚实、capacity/supervision matched、可干预”的组合；
3. 当前仓库的 schema、transport、mask、sidecar gate 和 hash plumbing 远领先于真实科学证据；
4. 真实标签、production group、目标 asymmetric robot 数据、native liquid/contact truth 和硬件执行链均为空或 No-Go；
5. 因而下一步应优先实现最便宜的证伪器，而不是继续增加 topology、dense visibility 或 effect-model 模块。

## 2. 五案决策矩阵

评分只表示当前项目中的相对优先级：5 为最强/最高，依赖与风险列中 5 表示最重/最高。

| Idea | 可守 novelty | 数据/硬件依赖 | 当前实现覆盖 | 最小可证伪实验 | 顶会投稿风险 | 决策 |
|---|---|---|---|---|---|---|
| **A：finite contact WAM** | **4/5**。新意是 conditional visual bottleneck、精确 alphabet、matched causal controls 与 intervention 的交集；contact token/ST 本身不新。 | **4/5**。路由审计不需要标签；收益、oracle 和 OOD 需要真实 contact labels/group；真机需要目标硬件。 | **4/5 plumbing，0/5 benefit evidence**。22D、hard-ST/detached、strict ref mask、planner isolation、cache 与 hashes 已有；direct/equal-bit/aux controls 未实现。 | CPU route invariance + exact capacity；之后 oracle-hard vs matched direct。 | **4/5**：组件拥挤、186.85 bit 未必小、单任务/无 OOD、text scope 易被过度表述。 | **唯一主线，Conditional Go** |
| **B：asymmetric contact topology** | **3/5**。只有 object/part/relation + `left6/right1` feasibility allocator + tactile closure 的组合尚可守。 | **5/5**。需要 topology labels、对象/part binding、RH56/OmniPicker geometry/driver/calibration/tactile、真实多任务。 | **1.5/5**。只有 asymmetric25 action contract 和基础 22D grammar；topology、allocator、split decoder 均未实现。 | 小型枚举 allocator 可测软件正确性；科学证伪必须是 KITE/22D baseline 上的真实 asymmetric tasks。 | **5/5**：范围爆炸、KITE/AsymDex/TopoRetarget/UniDex 强冲突，硬件链为空。 | **后置 benchmark/stress case；op38 不做** |
| **C：visibility/contact auxiliary** | **2/5**。boundary/rim/visibility/stream auxiliary 都拥挤；只剩 observation contract + train-only strict isolation。 | **3/5**。rim/boundary 需正式双标；stream/contact-interface 当前不可局部化；液体 sim oracle 还需 GPU/native assets。 | **1/5**。有 100-frame prescreen 和 sim spec；无正式 A/B、无 aux decoder、无训练。 | 100-frame A/B；冻结小 probe 比较 oracle visibility/visible-rim aux 是否有 ≥5pp proxy gain。 | **4/5**：伪 3D、pseudo-GT、单 episode 着色液体、普通 aux novelty 弱。 | **仅 annotation audit 候选；不列 claim** |
| **D：censor/missingness benchmark** | **3/5 supporting，2/5 standalone**。组合协议有价值，但 mask/censor/survival/MNAR 各自不新。 | **4/5**。需要真实双标、per-side/per-field coverage、group metadata、部分 sensor truth 和足够 events。 | **2.5/5 plumbing，0/5 evidence**。v1.1 coverage/mask/censor 字段与 loader/loss 已有；真实 CSV 空；hazard、IPW/MNAR、v2 group identity 未实现。 | 先完成 12-episode D0；CPU synthetic hazard 只能验公式，不能验收益。 | **4/5**：TouchAnything/EgoLoc/ARCTIC 提高规模与物理真值门槛，当前单任务太小。 | **辅助 claim 1，Conditional Go** |
| **E：contact-action consistency** | **3/5 supporting，2/5 standalone**。A 的逻辑缺口真实，但 forward/inverse consistency、frozen evaluator 与 contact effect 都有强前例。 | **5/5**。必须有 target/sim action→realized contact、失败/扰动 action、native truth、group-heldout 与物理 feasibility。 | **0.5/5**。只有 flow clean-action 公式可接；无 `F_R`、effect schema、effect data 或 counterfactual oracle。 | `F_R(s,a)` 必须显著胜过 `F_0(s)`；same-noise feasible intervention 必须由 native truth 验证。当前不可执行。 | **5/5**：伪 physics、reward hacking、off-support exploit、旧 dual-Inspire/ego 数据不匹配。 | **辅助 claim 2，后置 Conditional Go** |

### 优先级含义

- A 是唯一能用当前代码先做低成本结构证伪、又可能形成主方法的方向。
- D 是 A 的监督可信性，不应独立膨胀成另一条 benchmark 论文，除非达到跨域、sensor-truth 和规模晋升门槛。
- E 是 A 的 realizability 证据，不应在没有 robot/native effect 数据时先造网络。
- B 同时引入新 ontology、resolver、allocator、触觉和硬件，当前会把一个可证伪问题变成五个相互混杂的问题。
- C 可以帮助标签诚实性，但若连续 heatmap 必须进入 action 才有收益，就直接冲突 A 的有限瓶颈主张。

## 3. 当前代码到底证明了什么

### 3.1 已实现

- `world_action_model/contact_transition_schema.py:64-105`：每段 22D grammar。[I]
- `world_action_model/contact_transition_schema.py:114-187`：hard route 的精确有限容量；Q12/B16 为 186.850009 bit。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:685-803`：Q12 planner、22→action-width encoder 与 contact decoder 初始化。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:995-1094`：soft/hard-ST/hard-detached transport、时间/censor canonicalization、absence 清零、正零 canonicalization。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:1096-1169`：planner 读 state/reference/text，raw planner hidden 不离开该方法。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:1171-1211`：semantic route 中 action 只得到 transported 22D 的线性投影。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:110-177,1571-1595`：strict route 禁止 action self-attention 读取 reference tokens。[I]
- `world_action_model/models/transformer_wa_casual_mot.py:1946-2051`：action-only cache 在 bottleneck 模式去除 visual KV 并缓存 transported semantics。[I]
- `world_action_model/trainer/wa_casual_trainer_mot.py:897-981`：contact target/oracle plumbing 与 masked grouped loss。[I]
- `configs/contact_transport_ablation_matrix.json`：soft/hard-ST/hard-detached 的当前三变体 contract；其它关键 controls 明确为 spec-only。[I/S]

### 3.2 未实现或无证据

- Q12 direct-capacity、aux-only、equal-bit unstructured、raw-hidden、shuffled/delayed runner。[S/M]
- strict route 的正式 clamp、gradient、cached/uncached 与 image-conditioning fail-closed 证明。[M]
- 真实 reviewed contact segment；当前 formal A/B/adjudication `segments.csv` 只有 header。[M]
- production OOD group；现有四个 continuity blocks 只允许 temporal diagnostic。[M]
- censor-aware hazard/survival loss；当前 start/end 是 Smooth-L1，censor 是独立 BCE。[M]
- 任何 contact/action/OOD/safety 收益。[M]
- `F_R`、native realized-effect label、target-robot counterfactual、物理 feasibility oracle。[M]
- OmniPicker mapper/driver、完整标定、watchdog 和真机验证。[M]

因此当前准确表述是：**finite route plumbing implemented；causal isolation incompletely audited；scientific benefit absent。**

## 4. 为什么现有 3-variant contract 不足

当前 runner 的三个变体为：

| Variant | action 读 reference | action-visible packet | action loss→planner | contact aux |
|---|---:|---|---:|---:|
| `soft_semantic` | 否 | 连续 22D，非有限 | 是 | 是 |
| `hard_st` | 否 | canonical hard 22D | ST 是 | 是 |
| `hard_detached` | 否 | 与 hard-ST forward bit-identical | 否 | 是 |

这组三臂只回答：

1. soft activation 与 hard finite forward 的训练/性能差异；
2. hard forward 相同时，action loss 穿过 ST 是否影响 planner。

它不能回答：

1. strict semantic route 是否优于/不劣于 direct visual action；三个变体全都 strict；
2. 收益是否只是 contact auxiliary supervision 改善共享 backbone；没有 aux-only direct；
3. 收益是否只是任何有限压缩/regularization；没有 equal-bit unstructured；
4. Q12 额外 planner 参数与计算是否造成收益；没有 Q12 direct-capacity；
5. grammar 是否足以控制任务；没有可运行的 reviewed oracle；
6. action 是否真的使用字段；没有 clamp/swap/delay 的 route 与 phase-locality evidence。

另外，soft 没有相同的 finite alphabet，hard-detached 改的是优化路径而不是 forward information。把这三者称为“matched causal benefit experiment”会过度解读；目前它只是 **matched transport mechanism experiment**。

### 推荐的实验格，而不是强行让三个变体回答所有问题

| 对比 | 需要的两个变体 | 能归因的量 |
|---|---|---|
| Q12 direct λ0 vs aux-only direct | 相同 direct route，只有 contact loss 不同 | contact auxiliary supervision |
| aux-only direct vs hard-ST semantic | 相同 planner/contact loss，direct ref vs strict hard packet | strict semantic routing |
| soft semantic vs hard-ST semantic | 相同 strict route/监督，continuous vs finite | hard finite transport |
| equal-bit unstructured vs hard-ST semantic | 相同 strict route、Q、alphabet、监督，任意码 vs explicit fields | action-visible semantic factorization |
| hard-detached vs hard-ST | forward packet 相同，action gradient route不同 | ST/end-to-end optimization |
| oracle-hard vs predicted hard | 同 strict decoder，计划真值不同 | schema sufficiency 与 prediction gap |

如果算力只允许一个 `3 seeds × 3 variants` 主表，推荐：

1. `aux_only_direct_q12`；
2. `equal_bit_unstructured_hard_st`；
3. `hard_st_semantic`。

这三者应使用相同 contact auxiliary supervision、Q12、planner/action 参数预算、数据、初始共享权重、噪声和训练步数。它能直接比较 direct、任意有限码和显式有限语义，是最贴近主 claim 的三臂。

但它仍不能证明 hard 优于 soft，也不能说明 ST 梯度为何有用。`soft_semantic` 与 `hard_detached` 应各自作为三 seed 的二级机制实验；`q12_direct_lambda0` 用于量化 aux-only 的监督增益；oracle 是训练前 sufficiency gate。若这些附加臂不跑，论文必须删掉对应归因，不得用一个三臂表替代完整 factorial logic。

## 5. Q12 direct-capacity 的精确定义

### 5.1 它控制什么、不控制什么

建议正式命名为 `q12_direct_shadow_lambda0`，避免把它误写为 bit-capacity control。它只匹配：

- Q=12 planner 的可训练参数数量；
- planner forward 计算；
- 可选的 Q12 shadow token tensor shape/每层 dense kernel 计算；
- 原始 direct action→reference route。

它**不匹配 action-visible information capacity**：direct reference 是高带宽，不能和 186.85 bit 等同。它也无法精确匹配 backward FLOPs，因为 λ=0 且 planner output 不影响 loss；receipt 必须报告 active-gradient parameter count 和 backward-FLOP 差异，不能只写 matched FLOPs。

### 5.2 数据流

```text
state + reference + text ──> same Q12 planner ──> hidden/logits/semantic
                                                │
                                                └── DISCARD from every action input/loss

state + noisy action + reference + text ──> direct action expert
```

若为了匹配 joint-stack 序列长度加入 Q12 shadow tokens，它们必须：

- 与 planner outputs 无关，或 planner output先 `detach` 后仅作为被隔离的 shadow value；
- shadow 自身最多 self-attend；action/state/reference/future 均不能读 shadow；
- shadow 输出全部丢弃，不进入 visual/action/contact loss 或 cache；
- attention mask 单独有 `action_attends_shadow=False`，不能复用当前总是把 `contact_idx` 加入 action columns 的逻辑。

更保守的第一版可以不加 shadow tokens，只匹配 planner 参数/forward FLOPs，并在 receipt 明确 joint-stack token FLOPs 未匹配；不能暗称 exact compute matched。

### 5.3 不能偷看的量

该 route 必须 fail closed：

- `contact_semantic`、contact logits、planner hidden 不得传给 `encode_action_tokens()`；
- `contact_condition/contact_condition_mask` 非空时直接拒绝；
- `contact_loss_weight=0`、`require_contact_targets=False`；
- 即使 raw batch 为了数据 hash 一致而含 contact target，model/trainer 也不能读取它；
- prefix cache 不得保存 contact/planner/shadow K/V；
- action loss 对所有 planner 参数梯度必须为 `None` 或精确零；
- 固定 reference/state/text/noise 时，任意替换 planner output 不得改变 pre-safety action，容差 `max_abs<=1e-6`。

### 5.4 aux-only 与它不是同一个 baseline

`aux_only_direct_q12` 使用相同 direct action route，但 contact planner 和 grouped contact loss 正常训练；planner output仍不进 action。两者的差值量化“接触监督作为共享表征辅助”的收益。若把 Q12 direct λ0 与 aux-only 合并，strict route 的收益就会和 auxiliary supervision 混在一起。

## 6. Equal-bit unstructured 的精确定义

### 6.1 精确 alphabet

当前 B16 hard grammar 每 query 的有效状态数为：

```text
K = 1 + (3 * 6 * 4 * 4) * (16*17/2 + 2*16 + 1)
  = 48,673
```

因此 equal-bit control 必须满足：

```text
states_per_query = 48,673
states_Q12 = 48,673^12
bits_Q12 = 12 * log2(48,673) = 186.850009
```

建议使用一个无需 48,673-way embedding table 的 exact codec：

1. 每个 query 的 22 个 raw planner logits 先经过固定、无参数的 `16×22` Rademacher mixing matrix，再经 sigmoid 产生 16 个 soft bits；matrix seed、完整 bytes 和 SHA 写入 receipt；
2. hard threshold 得到 16-bit unsigned `id`；
3. `id ∈ [0, 48,672]` 保留；`id ∈ [48,673, 65,535]` 全部 canonicalize 为 `id=0`；
4. 将合法 id 还原为 16 个 hard binary digits；
5. 后 6 维填 canonical `+0`，形成 Q×22；
6. 使用与 semantic route 同一个 22→action-width encoder；
7. forward 只有 48,673 个唯一状态，action backward 使用与主方法同类的 ST surrogate。

不能直接把整数 id 放进 float scalar，也不能把 16 个未约束 bits 全部送入 action；前者有连续/数值 side channel，后者会得到 65,536 states，超过主方法容量。

### 6.2 参数、监督与路由匹配

推荐的主 control 不是 label-free code，而是 **same-contact-aux unstructured code**：

- Q12 planner、ref/state/text 输入、planner hidden 和 semantic contact decoder 与主方法相同；
- semantic decoder保留与主方法相同权重的 grouped contact auxiliary loss；其 raw logits还可以进入下述固定 codec，但不能以原值进入 action；
- codec只从 raw contact logits 经固定 mixing、hard quantization和 invalid-id canonicalization得到有限码；action从不直接接收 raw logits、soft semantic、GT、field masks或 planner hidden；
- action→reference 关闭；action 只读取 hard unstructured Q×22 code、state、text 与 noisy action；
- contact target 只在 model forward 后进入 trainer loss，不能作为 model input；oracle condition 在此 route 禁止；
- 参数量与 22→width encoder匹配；fixed mixing matrix是 registered buffer而非可训练旁路；
- action loss 可以通过 ST 和固定 mixing 更新 contact decoder/planner，和 hard-ST 主方法保持相同的可达参数集合。

这样 equal-bit code可能在 contact auxiliary 和 action loss作用下学出 contact-correlated bits，这是合理的：它检验的是**显式字段结构**是否优于同监督、同容量的任意有限码，而不是禁止模型学习 contact。若使用完全 label-free unstructured code，hard-ST 差异会同时包含接触监督，不能独立归因语义。

### 6.3 防旁路不变量

- action 不读取 raw planner hidden；只读取 canonical hard 16-bit code + 6 个正零；
- action 不直接接收 semantic decoder logits/activations，只接收 codec 边界后的 hard code；
- action self-attention 不读取 reference/future；
- `encoder_hidden_states_image` 在 strict route 非空时应 fail closed；
- cache只保存 hard code，不保存 hidden/logits/ref K/V；
- 穷举 65,536 个 raw ids 后，输出 unique count 必须精确为 48,673；
- 任意 invalid id 均得到完全相同的 canonical zero code；
- 不能出现 NaN/Inf、`-0.0`、第 17–22 维非零；
- 固定 hard code/state/text/noise 后替换 reference/planner hidden，action 不变 `max_abs<=1e-6`；
- 修改 contact target bytes而不重新训练/不作 oracle forward，当前 action forward 不变。

### 6.4 它仍不是完美控制

semantic one-hot/time-bin 与 mixed binary id 的局部几何不同，固定 mixing 增加少量无参数 FLOPs，ST surrogate 的条件数也不同；即使 cardinality、Q、encoder、监督和可训练参数匹配，优化难度不会完全相同。因此必须报告 code utilization、collapse、entropy、invalid-to-zero rate 与 gradient norm。equal-bit 若失败但大量 collapse，不能直接证明 human semantics 优越；先修优化或降低主张。

## 7. hard-detached 应放在哪里

`hard_detached` 不是 direct baseline，也不是 semantic baseline。它与 hard-ST 的 action-visible forward 完全相同，只改变 action loss 是否能回到 planner，因此应放在：

1. **CPU mechanism gate**：相同 weights/input 下，hard-ST 与 detached forward bytes一致；hard-ST action gradient到 planner非零，detached为零；显式 contact loss对两者都能更新 semantic decoder/planner。
2. **二级三-seed mechanism ablation**：只有 hard-ST 已在 direct/equal-bit 主比较中成立，且论文要声称 end-to-end action gradient重要时才跑完整三 seed。
3. **失败解释**：若 hard-detached≈hard-ST，结论是 action-to-planner ST 不是必要机制；保留 finite semantic route claim，删除 gradient claim。若 detached 更好，需警惕 biased ST 或 planner/action co-adaptation。

它不应占据唯一主 `3×3` 的 direct/equal-bit 席位。现有 runner 把它放进三臂是工程 transport 验证的历史结果，不是最终科学优先级。

## 8. Text cross-attention 与主 claim 边界

### 8.1 当前事实

- `world_action_model/models/transformer_wa_casual_mot.py:634-641` 建立 action-side text conditioner；
- `:1553-1555` 从原始 `encoder_hidden_states` 生成 `encoder_action_hidden_states`；
- `:1586-1595` 将其送入 action expert cross-attention；
- `:1976-1980,2042-2045` 在 action-only cache 中继续生成和保存 text-conditioned action KV。

所以 text 是 action 的直接条件。现有 reference self-attention mask与 22D route只能证明视觉路径，不限制 text。

### 8.2 可写与不可写

可写：

> Conditional on instruction text, proprioception, embodiment and diffusion noise, current-reference visual information reaches the action expert only through the finite contact packet.

不可写：

- “the action reads only contact semantics”；
- “the 22D packet is the only task-semantic channel”；
- “all information available to action is bounded by 186.85 bits”；
- “text/object identity cannot bypass contact”。

### 8.3 实验控制

- 所有 matched variants 的 text embeddings、prompt template 和 prompt hash 必须相同；
- OOD prompt 不得包含 object instance、liquid ID、结果或接触标签，否则视觉 OOD 可被 text泄漏；
- visual clamp test 固定 text；替换 reference时 action应不变；
- text替换导致 action变化在本主张下是允许的，不应把它误报为 route failure；
- 若未来要升级为 total task-semantic bottleneck，必须新增 `action_text_conditioning=off`，让 text只进入 planner，并重新做性能/路由实验。这不是 op38 的推荐范围。

另外，production base 当前 `image_dim=None`，action condition embedder也固定 `image_embed_dim=None`；未发现当前生效的 image cross-attention 旁路。但通用 API 仍接收 `encoder_hidden_states_image`，strict route 应显式拒绝非空输入，而不是依赖配置偶然为 None。

## 9. frozen `F_R` 与 same-noise counterfactual 的当前决策

### 9.1 现在不训练 frozen `F_R`

当前没有任何合格的数据源：

- 66ep 是 human ego，无 target robot action 和 realized contact effect；
- hand_status 是 open/closed proxy，不是物理 contact；
- 88ep dual-Inspire 是错误 embodiment/任务，也无 contact/outcome truth；
- sim oracle readiness 为 No-Go，当前液体 PBD 又需要 GPU；
- target RH56+OmniPicker driver/calibration/native effect 数据为空；
- production groups 缺失，无法 group-heldout 校准。

此时训练小 `F_R` 只能学 `scene/phase -> contact` 相关性，或从 planner/future pseudo label 复述目标。把它冻结不会把伪物理变成物理。因此 op38 对 learned `F_R` 为 **No-Go**。

### 9.2 何时才允许训练

必须先有：

1. native simulator contact/pose outcome，或标定后的目标机器人 tactile/force/object-pose outcome；
2. 同一 state 下的 expert action、safe perturbation、failure/no-action 数据；
3. `F_R` 输入严格为 state/proprio、embodiment、current independently observed geometry/contact 和 physical 25D action；禁止 text、task ID、desired contact、planner hidden/logits；
4. 独立 `F_0(state)` baseline；`F_R` contact macro-F1 至少高 10pp；
5. group-heldout、uncertainty calibration、off-support mask 和冻结 checkpoint SHA；
6. effect score与 native/真实 outcome rank correlation至少 0.60，intervention ranking AUROC至少 0.80。

只有这些通过，才把 `a0_hat = noisy_action - sigma * action_pred` 接入 frozen `F_R`，且目标 packet `stop_gradient`，梯度只回 action generator。

### 9.3 same-noise harness 现在值得实现，但只能做语法/路由审计

现在可实现：

```text
same checkpoint
same reference/state/text/embodiment
same action timestep and initial noisy action
same scheduler/noise tensors
change exactly one canonical packet field
compare pre-safety action and per-side/per-phase locality
```

合法用途：

- identity intervention应逐 bit/容差复现动作；
- fixed packet + changed reference 检查视觉旁路；
- side/site/mode/time field swap 检查 action是否对 packet敏感；
- delay不跨 episode、不 wrap；absence/censor/start≤end保持 canonical；
- 记录 intervention delta、action delta、未干预侧/phase 的变化。

当前不能写：

- `physics-feasible counterfactual`；只有 grammar-valid；
- action变化方向代表正确物理效果；
- 某个 swap 应成功执行；
- locality 等于 realizability。

物理 feasibility 只能由校准 kinematics/collision/force solver、native simulator state或真实安全试验授权。手写 heuristic、MANO overlap、future-video prediction、depth completion 或 learned `F_R` 自评分均不能把 counterfactual 升为物理正例。

## 10. 当前无 labels/group 时可安全实现和 CPU tiny 验证什么

### 10.1 可安全实现

1. route enum 与互斥配置；非法 boolean 组合 fail closed；
2. Q12 direct-shadow、aux-only 的 action-invisible planner route；
3. exact K=48,673 unstructured codec 与 capacity receipt；
4. attention adjacency/route receipt，包括 text 作为允许条件；
5. strict mode 对非空 `encoder_hidden_states_image` 的拒绝；
6. cached/uncached route一致性；
7. same-noise grammar intervention harness；
8. current hard transport的小 B/Q 穷举、正零/NaN/Inf/absence/censor测试；
9. matched-control raw batch/augmentation/timestep/noise hash的 synthetic tiny replay；
10. 独立离散 hazard公式的 synthetic exact/right/interval-censor数值检查，但不要接 production trainer或称有收益。

### 10.2 CPU tiny 必须验证

- B16/Q12 semantic metadata精确返回 48,673 states/query 和 186.850009 bits；
- unstructured raw 65,536 ids映射后 unique count精确 48,673；
- hard-ST/detached 相同 input/weights的 forward bytes一致；
- hard-ST action loss到 planner有梯度，detached没有；
- direct-shadow action loss到 planner无梯度；
- direct route action rows可读 ref、不可读 shadow/contact；strict semantic/unstructured不可读 ref；
- fixed packet/state/text/noise下换 reference，semantic strict action `max_abs<=1e-6`；
- fixed unstructured code/state/text/noise下换 planner hidden/reference，action `max_abs<=1e-6`；
- direct route替换 planner hidden/logits/targets，action `max_abs<=1e-6`；
- identity intervention same-noise action逐 bit或在预注册容差内一致；
- cached/uncached输出在同 dtype 下容差一致；
- legacy soft/hard-ST/hard-detached route保持当前数值行为，不静默改变 production receipt。

### 10.3 当前绝对不能验证

- contact F1、timing MAE、oracle gain；
- object/operator/liquid/session OOD；
- spill、流量、质量守恒或 liquid physics；
- counterfactual physical correctness；
- asymmetric allocator/tactile收益；
- `F_R` action effect、真机 success 或 safety。

## 11. 推荐的 op38 精确实现任务书

### 11.1 任务名称

**op38 — Contact causal-route controls and CPU audit harness**

目标是补齐 A 的两个最关键反事实 control：action-invisible Q12 direct/aux route 与 exact-capacity unstructured route，并生成可机读的信息流证据。op38 不做标签、训练、F_R、topology、visibility head、hazard trainer、hardware 或 production runner admission。

### 11.2 允许修改的代码面

1. 新增 `world_action_model/contact_route_contract.py`
   - 定义唯一 route enum：现有三种 semantic transport保持兼容，并加入 `q12_direct_shadow`、`aux_only_direct_q12`、`equal_bit_unstructured_hard_st`；
   - 给出每种 route 的 action-readable sources、losses、cache policy、finite capacity、allowed conditioning；
   - 拒绝互相矛盾的 legacy booleans；生成 canonical JSON receipt。

2. 更新 `world_action_model/contact_transition_schema.py`
   - 新增纯函数 `unstructured_capacity_metadata(K=48673,Q=12,width=22)`；
   - 新增 16-bit id canonicalizer：invalid→0、末 6 维正零；
   - 不改变 v1.1 schema 或现有 transport输出。

3. 更新 `world_action_model/models/transformer_wa_casual_mot.py`
   - planner逻辑拆为“获得 planner hidden”和“semantic decoder/transport”，避免 direct route为了丢弃输出而意外调用 action encoder；
   - direct/aux route的 planner输出不进入 action；action保留 direct ref；
   - 若实现 shadow tokens，扩展 mask使 action/state/ref/future都不可读 shadow，shadow输出丢弃且不缓存；
   - unstructured route从 raw contact logits经 receipt-bound fixed `16×22` mixing生成 exact hard code；raw logits仍只作为 contact aux prediction，绝不直接进入 action；
   - semantic/unstructured strict route action→ref关闭；
   - strict route收到非空 `encoder_hidden_states_image` 直接报错；
   - cache receipt记录 route、packet/code bytes和允许的 text condition，不保存旁路 K/V；
   - 保证现有三 route默认和 state dict兼容。

4. 更新 `world_action_model/trainer/wa_casual_trainer_mot.py`
   - `q12_direct_shadow` 强制 λcontact=0、targets非必需；
   - `aux_only_direct_q12` 与 unstructured route使用与 hard-ST相同 grouped contact loss；
   - unstructured route禁止 oracle condition；contact target只进 loss，不进 model condition；
   - loss receipt分开记录 action、visual、contact aux，不引入 effect/counterfactual physics loss。

5. 新增非 production config，例如 `configs/giga_world_policy_0_5_sonic_g1_contact_causal_controls.py`
   - 只允许明确 variant ID，不接受多个 env boolean拼装；
   - 默认 fail closed，不放开 `allow_5b_training`；
   - 当前 production asymmetric25 config和外部 acceptance保持不动。

6. 新增可复用审计模块/CLI，而不是一次性测试脚本
   - 建议 `world_action_model/contact_route_audit.py` 与 `scripts/audit_contact_route_contract.py`；
   - 仅 CPU tiny model，输出 strict JSON：adjacency、capacity、unique states、forward equality、gradient reachability、clamp invariance、cache invariance、same-noise hashes；
   - 输出路径必须由调用者显式给到项目盘，不写 `/tmp`。

7. 更新 `configs/contact_transport_ablation_matrix.json` 与文档
   - 将新 route标成 `implemented_cpu_route_unadmitted`；
   - 当前 production runner exact set仍保持 soft/hard-ST/hard-detached，避免使既有 external receipt失效；
   - 明确未来主三臂是 aux-only/equal-bit/hard-ST，hard-detached为mechanism arm；
   - 不把 CPU audit写成收益。

### 11.3 op38 不应修改

- 不改 contact label CSV、group metadata、hardware manifest、sim oracle结果；
- 不改 asymmetric25 physical mapping/driver；
- 不加入 topology/allocator/tactile；
- 不加入 visibility/dense stream/depth completion；
- 不实现或训练 `F_R`；
- 不启动 5B、server、GPU、Isaac PBD 或真机；
- 不更新 production matched-control acceptance set，除非另有外部 reviewer receipt。

### 11.4 op38 验收门槛

全部满足才可称“route controls implemented”：

1. 新 route每个都有唯一、可哈希 receipt；任何非法组合在模型构造前失败；
2. equal-bit unique states精确 48,673/query，Q12 bits与 semantic hard相同到 `1e-9`；
3. direct-shadow/aux-only action对 planner hidden/logits/semantics/target替换 `max_abs<=1e-6`；
4. semantic/unstructured strict action对 clamped packet/code下 reference替换 `max_abs<=1e-6`；
5. direct action→ref edge存在；strict action→ref edge不存在；所有 route的 text edge状态在 receipt中明确；
6. hard-ST/detached forward packet bytes相同；梯度 reachability符合定义；
7. unstructured action不读取 semantic logits/GT/oracle，cache不保存 raw planner/ref；
8. identity same-noise intervention完全复现；所有非 identity输出只标 `route_sensitivity`；
9. 当前 soft/hard-ST/hard-detached CPU golden behavior不变；
10. `git diff --check`、CPU audit和配置静态检查通过；没有 GPU调用、临时测试脚本遗留或 canonical修改。

### 11.5 op38 立即停止条件

任一出现就停止扩展并回到设计：

- 必须让 action直接看到 raw planner hidden/logits才能实现 direct/equal-bit control；
- Q12 direct无法证明 action loss对 planner梯度为零；
- equal-bit实际可见状态超过48,673/query，或 float/id/signed-zero形成额外 side channel；
- strict clamp换 reference后 action差异超过 `1e-6` 且无法由允许的 state/text/noise差异解释；
- 新 route破坏现有三 route的数值兼容或 production receipt；
- 需要真实 contact labels、group补录、sim/native physics或硬件才能通过“CPU route implemented”验收；
- same-noise harness开始输出“physical feasible/success”而没有 native oracle。

## 12. op38 之后的科学停止门槛

op38 通过只说明实验工具可用，不允许训练主模型。后续依次通过：

### Gate S0：真实标签

- coverage weighted κ≥0.70；
- event F1≥0.80；
- median tIoU≥0.70；
- onset/release median MAE≤2 frames；
- side/site/role/mode κ≥0.60，并有足够 comparable events；
- `unknown`、positive-only、exhaustive、censor分布完整报告。

协议修订一次仍失败：停止 human-auditable grammar主张。

### Gate S1：oracle sufficiency

相对 `aux_only_direct_q12` 或最强 matched direct：

- oracle-hard task success≥+10pp；
- correct-first-contact≥+15pp；
- spill/safety不退化。

失败：停止 predicted-contact 大训练，先改 schema；不能用更大 backbone掩盖。

### Gate S2：预测恢复

- predicted contact event F1≥0.80；
- timing MAE≤2 frames；
- hard-ST恢复≥70% oracle gain。

失败：停止“ego predicted grammar可闭环控制”claim。

### Gate S3：主因果

- hard-ST相对 aux-only direct在预注册 group-OOD success至少+5pp且95% cluster CI不跨0；
- hard-ST相对 equal-bit有稳定优势，否则删除“显式 semantics优于压缩”claim；
- soft≈hard时删除“finite quantization带来性能收益”，仍可保留审计性；
- hard-detached≈hard-ST时删除“ST action gradient必要”；
- packet intervention无phase/side局部效应时停止因果使用claim。

### Gate S4：E 的重开

只有 native/sensor effect data与 `F_R>F_0` gate通过才训练 effect cycle。若 `F_R` rank correlation<0.60、intervention AUROC<0.80、off-support>20%或存在明显reward hacking，E停止并不进入主论文。

### Gate S5：硬件与投稿

补齐 production groups、至少多个 task/contact topology、OmniPicker driver/mapping/calibration/watchdog/native outcomes 后，才允许 target-hardware与顶会主结果。现有66ep单任务、空 labels和硬件模板下均为No-Go。

## 13. 推荐论文结构与明确舍弃

### 主 claim

**Finite, auditable current-visual routing in a joint WAM。** 重点是 conditional visual route、精确容量、direct/aux/equal-bit controls和 packet intervention，不是“contact token新颖”。

### 辅助 claim 1

**Coverage- and censor-aware contact supervision。** 必须有真实双标和hazard/calibration evidence；否则只放 protocol appendix。

### 辅助 claim 2

**Action-mediated packet realizability。** 必须有 frozen action-only `F_R`、native truth和same-noise physics-feasible intervention；否则只报告route sensitivity，不写effect claim。

### 不进入主论文 claim

- B 的完整 topology/allocator/tactile 系统；它可作为后续论文或硬件 benchmark。
- C 的 dense visibility/liquid/stream head；rim/boundary最多是训练期附加消融。
- “all task semantics are finite”“zero-shot RH56+OmniPicker”“transparent liquid physics learned”“MNAR solved”“plumbing improves performance”。

## 14. 总结

五案中只有 A 同时满足“已有足够代码可低成本证伪”和“组合 novelty 尚可守”。D 与 E 分别补上监督可信性和动作实现性，但当前都缺最关键的数据证据。B/C 现在加入只会扩大依赖和投稿攻击面。

op38 应严格停留在 route controls：先让 direct、aux、semantic hard、equal-bit arbitrary code 的信息边界成为可执行、可哈希、可 CPU 复核的事实。只有真实标签和 oracle 证明 grammar有用后，才值得跑三 seed；只有 native action-effect数据后，才值得训练 frozen `F_R`。任何把当前代码通路、空模板、synthetic intervention或 future-video prediction写成收益/物理证据的做法，都应在进入训练前停止。
