# Idea L 独立调研：Sparse Interaction Motif Dictionary

- 日期：2026-08-12
- 审查基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 隔离分支：`codex/idea-l-sparse-interaction-motif-20260812`
- idea-discussion `idea_id`：`76f0e0a1-cc36-4e72-88ed-fb063f752297`
- idea-discussion `discussion_id`：`6707f93d-9734-4ca1-8858-fa9b24cd6ec1`
- 工作范围：Zotero-first 文献审计、结构设计、数据协议与证伪条件；没有训练、没有调用 GPU、没有修改模型代码或 canonical worktree

## 0. 结论先行

**“学习 action motif/codebook，再用它指导跨 embodiment 策略”作为独立主 idea 是 No-Go。** 2026 年已有一篇名称与结构都高度重合的 [MOTIF: Learning Action Motifs for Few-shot Cross-Embodiment Transfer](https://arxiv.org/abs/2602.13764)：它已经用 VQ、progress-aware alignment、embodiment adversarial loss 学统一 action motifs，再由实时视觉/语言预测 motif 并条件化 flow-matching policy。更早的 [XSkill](https://arxiv.org/abs/2307.09955)、[CompILE](https://arxiv.org/abs/1812.01483)、[Bottom-Up Skill Discovery](https://arxiv.org/abs/2109.13841)、[LOTUS](https://arxiv.org/abs/2311.02058)、[Learning Options via Compression](https://arxiv.org/abs/2212.04590) 和 [PRISE](https://arxiv.org/abs/2402.10450) 已分别覆盖跨人机 skill prototype、可复用变长片段、skill library 与序列压缩。

**“从长视频跳过准备段，只对齐公共子序列”作为算法贡献同样是 No-Go。** [Drop-DTW](https://arxiv.org/abs/2108.11996)、[StepFormer](https://arxiv.org/abs/2304.13265)、ImMimic 的 GMS-SDTW、CAPOT 方案以及最新的 [REALIGN](https://arxiv.org/abs/2509.24382) 已覆盖 outlier drop、稀疏步骤、多个匹配子序列、partial/unbalanced OT、重复动作与非单调过程。

仍值得保留的窄版本是 **Conditional Go**，但它不应被命名为另一个 “MOTIF” 模型，而应定位成现有 SM-FET-WAM 的一个 episode-level 结构先验：

> **Sparse Contact–Effect Macro Compiler（SCEMC）**：先把人类和机器人视频变成有限、角色化的 contact/relation/effect event atoms，只在任务关键区间挖掘长度 2–4 的可复用 macro；离线 partial pairing 只帮助形成训练期 equivalence classes；部署时由机器人当前观测预测 macro sequence，并确定性展开成既有 `event_tube_v2`，动作侧不读取 human video、raw motif latent 或新的视觉通道。

这个版本的可守点不是 motif、VQ、sequence mining 或 partial alignment 中的任何单项，而是下面的窄交集：

1. motif 的组成单位不是连续机器人 action 或通用视觉 step，而是 **hand/tool/target contact transition + relation delta + observable object effect**；
2. prep、idle、camera motion、robot homing 与恢复段是 alignment 的 `NULL/unmatched`，**不是 motif，也不是动作命令**；
3. 一个 human macro 对应一个 **robot execution equivalence class**，同一 macro 可由不同路径、速度、左右手分工和 embodiment 执行，而不是保存一条要复刻的人手轨迹；
4. macro 只约束 episode planner；action compiler 将它展开为现有有限 event packet，**不增加 action-visible bits，也不新增连续 latent bypass**；
5. human video 只在训练期用于词典发现和弱配对；部署模式不需要 human prompt。

因此本 Idea L 的正确角色是：**SM-FET 的可选压缩/组合先验，CAPOT 的全局共享层，ETG 的多边组合；不是替代 SM-FET 的第二条动作通道。** 若它不能在 matched-output 条件下优于直接预测 event tube，就删除整个 motif 层。

## 1. 用户 insight 的精确定义

用户提出的关键判断是正确的，但需要避免两个过度概括：

1. 工具操作不应只看一帧手部姿态。有效信息通常来自接触建立前后的相对手势、tool–target 接触、持续接触下的运动模式、效果出现和释放/停止；它是一个短过程，至少由多个姿态/事件构成。
2. 人机配对不应从抓取前开始做完整一一同步。人体准备、走动、镜头运动、等待、无关手势和机器人归零/避障不是同一物理技能的必要组成，不能被强制映射。

但“现有方法都做 frame-wise one-to-one”并不准确。标准 DTW 已天然支持 many-to-one/one-to-many，只是通常强制完整路径；Drop-DTW、subsequence DTW、GMS-SDTW 和 partial OT 已允许开放边界或内部 drop。Idea L 的研究问题应改写为：

> 相比 pair-specific frame/event alignment，能否从多个人类和机器人 episode 中发现一个**跨 episode 复用、效果保持、有限且可审计**的 contact-effect macro lexicon；它能否在不增加 action 信息容量、不把 human video带到部署端的条件下，改善机器人 event planning 和小样本执行？

三个可证伪假设：

- **H1 — sparse sufficiency**：关键 contact-effect macro 覆盖的帧远少于全 episode，但能保留绝大多数经裁决的任务事件与结果信息。
- **H2 — execution multiplicity**：同一个 macro ID 下存在多个合法 robot action chunks；macro 内 effect/contact 顺序一致，但具体 joint/EE trajectory 随 embodiment、状态与路径变化。
- **H3 — planner benefit, not channel benefit**：在 action-visible packet、容量、数据和 policy 完全匹配时，macro-constrained episode planner 比独立 event heads 更抗 prep insertion、速度变化与跨 group 分布偏移；若只在新增 latent 通道后有效，Idea L 失败。

## 2. Zotero-first 结果与外部补检

本轮先按 `idea-discussion` 同步并检索 Zotero。自动 librarian 的宽查询先返回 RT-1、RT-2、RoboBrain 与 FAST 等通用 VLA 条目，说明自动召回对 “motif/subsequence” 这个窄问题偏弱；随后直接核查同步后的 Zotero 索引，命中了更关键的个人库条目：

| Zotero key | 文献 | 与 Idea L 的关系 |
|---|---|---|
| `V7LLF6VJ` | XSkill: Cross Embodiment Skill Discovery | 从无标签 human/robot 视频发现 cross-embodiment skill prototypes，以 conditional diffusion policy执行，并可按 human prompt 组合；是“人机共享 motif/skill dictionary”最强库内冲突 |
| `EK2CTN7J` | UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations | 不需 aligned human–robot data，human prompt 中的 skill representation 可指导仅用 robot data 训练的策略；是强 unpaired 与 prompt baseline |
| `A89APIR3` | EgoBridge | 用 joint policy latent/action OT 对齐人类与机器人域；否定“human–robot OT alignment”本身的新颖性 |
| `8WMVVJAL` | ImMimic | 用 action/visual DTW 映射 retargeted human hand pose 到 robot joints，并用 paired interpolation；其 GMS-SDTW 支持长视频内部无关动作与多子序列匹配 |
| `WQJXZYHX` | Time-Contrastive Networks | 从视频学 viewpoint-invariant temporal representation，并用于机器人模仿 object interaction；说明跨视角阶段表征与单人示范模仿已有长期前例 |

外部 arXiv/CVF/NeurIPS/PMLR 原始来源补检又发现三个决定性冲突：

- [MOTIF](https://arxiv.org/html/2602.13764) 已明确把 action motifs 定义为与 task semantics/embodiment 独立的显著轨迹子序列；以固定窗口 EE state trajectory、VQ codebook、progress-aware alignment 和 embodiment adversarial loss学习，再由实时图像/语言预测并送入 flow policy。论文采用 `M=16` motif tokens、`K=128` codebook，且在多机器人仿真和真实 ARX5/Piper 上报告 few-shot transfer。
- [Action Motifs](https://arxiv.org/abs/2604.28173) 已从 3D human pose 自监督学习 action atoms 与其变长 temporal compositions；因此“从连续人体姿态自动发现可复用 motion motif”也不是空白。
- [REALIGN](https://arxiv.org/abs/2509.24382) 已用 regularized fused partial Gromov–Wasserstein OT 处理 procedure video 中的 irrelevant frames、repeated actions、order variation 与 non-monotonic steps；因此不能把“partial structural alignment + filtering noise”当贡献。

`arxiv-search` skill 的 CLI 当前没有打印 `query_arxiv()` 返回值，且系统 Python 缺少 `arxiv` 包。遵守项目环境规则，本轮没有向 base 安装依赖；idea-discussion 的 arXiv-first retrieval 仍已执行，精确核查改用 arXiv/CVF/NeurIPS/PMLR 官方页面。

## 3. 与最强前例的重合审计

| 方法 | 已覆盖的核心能力 | 对 Idea L 的冲突 | SCEMC 尚可保留的差异 |
|---|---|---|---|
| [MOTIF, 2026](https://arxiv.org/abs/2602.13764) | VQ action motif、progress alignment、embodiment invariance、实时 motif predictor、motif-conditioned flow policy、few-shot cross-embodiment | **最高**；“action motif 指导跨机器人动作”不可再 claim | 其 motif 来自 robot EE trajectory fixed windows，不含 passive human video、contact/effect ontology、NULL prep skip；SCEMC 不把 learned motif embedding送 action，而展开成受审计 finite event packet |
| [XSkill](https://arxiv.org/abs/2307.09955) | 无标签 human/robot video 中发现 skill prototypes，跨 embodiment 执行与组合 | **最高**；“共享 human–robot skill prototype/dictionary”已有 | XSkill 以 human prompt 指定任务，未把 contact/effect macro 与 irrelevant segments显式分开；SCEMC 的 human data只用于训练 |
| [Drop-DTW](https://arxiv.org/abs/2108.11996) | 双边任意内部 outlier drop、保序、可微 alignment | **高**；skip/null/partial alignment不是新算法 | 只把它当 pair baseline或离线工具；SCEMC 研究的是跨全部 episode 的共享 dictionary，而非一对序列的 alignment path |
| [StepFormer](https://arxiv.org/abs/2304.13265) | 从 instructional video 和字幕自监督发现稀疏 key-step slots，推理以 Drop-DTW 定位 | **高**；“大部分视频无关，只找关键步骤”已有 | SCEMC 的单位是 contact/relation/effect，不依赖 narration；还需与 robot execution class 和 strict WAM route闭环 |
| [REALIGN](https://arxiv.org/abs/2509.24382) | partial Fused-GW OT、virtual mass、重复动作、顺序变化、背景过滤 | **高**；partial structural procedure matching 已有 | 它不解决 human-to-robot execution multiplicity，也没有有限 action compiler；只能作强视觉 procedure baseline |
| [EgoBridge](https://arxiv.org/abs/2509.19626) | joint latent/action OT 的 human–robot domain adaptation | **高**；跨域 OT 和 policy co-training 已有 | SCEMC 的 coupling在 event/macro table离线 stop-gradient，不作为在线 latent/action distribution route |
| [ImMimic](https://arxiv.org/abs/2509.10952) | human hand→robot action mapping、DTW many-to-many、trajectory interpolation、多种 gripper/hand | **高**；人机轨迹配对与多形态 transfer 已有 | SCEMC 不把人手轨迹 retarget 成目标动作；同一 macro 由 robot state/embodiment 独立编译执行 |
| [TCC](https://arxiv.org/abs/1904.07846) / [XIRL](https://arxiv.org/abs/2106.03911) | 跨视频/embodiment task progress alignment、reward | 中高；阶段对齐和 temporal cycle consistency 已有 | TCC 是 dense progress embedding，无 explicit NULL、finite contact-effect macro 或 action route审计 |
| [CompILE](https://arxiv.org/abs/1812.01483) | 无监督变长片段、可复用 latent codes、hierarchical composition | 高；segment-and-compose 已有 | SCEMC 不以 reconstruction 自动定义技能，而要求 contact/effect 可解释且 action-visible 仍是 v2 packet |
| [Bottom-Up Skill Discovery](https://arxiv.org/abs/2109.13841) / [LOTUS](https://arxiv.org/abs/2311.02058) | 从 unsegmented robot demos 找 recurring skill library，由 meta-controller组合 | 高；recurring motif library 和 composition 已有 | SCEMC 加入 passive human domain、effect-equivalent many-to-many pair与 training-only distillation |
| [Learning Options via Compression](https://arxiv.org/abs/2212.04590) / [PRISE](https://arxiv.org/abs/2402.10450) | MDL/sequence compression 与 BPE式变长 action abstraction | 高；“用压缩发现有用 motif”已有 | SCEMC 只借用 compression 作工程工具；合并受 contact/effect、NULL边界和跨域一致性硬约束 |
| SM-FET-WAM / ETG（内部方案） | active+next events、duration、relation delta、effect；event graph/state transition | **直接内部重合** | SCEMC 只把多个 event atoms组合成共享 macro，不能再增加另一套 event fields或图网络 |
| CAPOT（内部 Idea G） | contact/effect anchored partial human–robot event pairing | **直接内部重合** | CAPOT 产生 pair edges；SCEMC 在多个 pair/episode 上建立全局 equivalence class。CAPOT不是部署模块 |

必须写进未来 related work 的限制句：

> 我们不提出新的 motif learning、partial alignment、skill discovery 或 cross-embodiment latent policy；我们检验的是，是否能把 training-only human/robot sparse event pairing 压缩成一个 contact-effect macro grammar，并在严格有限 WAM route 中证明它比 pair-specific alignment 和独立 event prediction更有用。

## 4. 推荐的最小结构：SCEMC-on-SM-FET

### 4.1 单一数据流

```text
训练期 human video / robot rollout
        |
        v
冻结或 EMA 的 event extractor
  -> role/contact/relation/effect event atoms
        |
        +--> prep/idle/camera/homing/recovery -> NULL/unmatched
        |
        v
离线 event-pair table（Drop-DTW / CAPOT / REALIGN-style baseline）
        |
        v
有限 macro dictionary：K<=16，长度 2--4 个 event atoms
        |
        v
训练期 distillation target

部署期 current robot reference + causal history + state + text
        |
        v
episode planner -> macro_id + binding + progress/termination
        |
        v
确定性 macro compiler -> active/next event_tube_v2
        |
        v
既有 finite event encoder -> action expert -> robot action chunk
```

部署期没有 human video、human embedding、pair table、dense coupling 或 motif prototype hidden。动作侧只读编译后的 canonical hard `event_tube_v2`、proprioception、允许的 text/noisy action/embodiment inputs。

### 4.2 motif 不是“常见视觉片段”，而是 effect-terminated event subsequence

每个 event atom 应来自 SM-FET/ETG 已定义的有限字段子集：

```text
atom = (
  event_kind,
  functional_role_binding,
  relation_before -> relation_after,
  contact_mode,
  object_effect,
  outcome_or_unknown,
  duration_bin,
  boundary_censor
)
```

一个 macro 是 2–4 个 atom 的有序模板：

```text
macro_k = (
  atom_template_1, ..., atom_template_L,
  L in {2,3,4},
  entry_predicate,
  termination_predicate,
  allowed_skip/repeat mask
)
```

推荐第一版要求每个 macro 至少包含以下二者之一：

- 一次可裁决的 contact/relation change；
- 一次可裁决的 object effect/outcome transition。

纯 approach、等待、镜头运动和无 effect 的常见人体动作即使频率很高也不能成为 motif。`NULL`、`UNK`、`PAD` 是 codec/control token，不计入语义词典，也不能触发动作。

### 4.3 词典如何学习

不推荐直接对 RGB、MANO trajectory 或 action chunk 做无约束 VQ：这会退化到 MOTIF、Action Motifs、XSkill 或一般 latent skill。最小发现流程是：

1. **事件化。** 用人工裁决/传感确认的 event interval训练或构造 frozen event atoms；连续 2D hand/tool pose只作为 proposal与 auxiliary，不进入 macro identity。
2. **关键区间裁剪。** 只在接触/效果事件附近建立候选；自然 prep、idle、robot homing、无关 recovery 进入 `NULL` pool。不得把 stable hold 中的每一帧重复成大量 token。
3. **候选枚举。** 在每个 episode 中枚举不跨 `NULL/unmatched` 的 2–4 atom subsequence；允许 event 间存在有限 idle gap，但 gap 不被编码为动作内容。
4. **跨域弱配对。** Drop-DTW/CAPOT/REALIGN-style matcher只建立 stop-gradient match/nonmatch/NULL edge；人类 left/right 不字面等于 robot left/right，按 functional role、contact mode、relation delta 与 effect计 cost。
5. **受约束压缩。** 可用 PRISE/BPE 或 MDL 作为实现基线，但只有 contact/effect compatibility一致的相邻 atom 才允许 merge；禁止跨 task failure/outcome 相反的边合并。
6. **全局 equivalence class。** 把来自多个 human clip 和多个 robot rollout 的效果等价候选聚合成同一 macro；至少跨两个 source groups，不能由一个 session 的重复片段独占。
7. **冻结与版本化。** 在 train split 学完后 hard-freeze dictionary、atom schema、canonical ordering、hash与 compile table；validation/test 不增加新 macro，只能使用 `UNK`。

一个可审计的目标可以写为：

```text
J(D, Z) = DL(D) + DL(event_sequences | D, Z)
        + lambda_effect * effect_impurity(Z)
        + lambda_pair   * paired_macro_disagreement(Z)
        + lambda_null   * false_key_coverage(Z)
        + lambda_group  * single_group_concentration(Z)
```

这里 `DL` 只是选择词典大小和合并顺序的工具，不是 novelty；`effect_impurity` 与 `false_key_coverage` 防止频繁但无任务意义的 prep motif；`single_group_concentration` 防止词典记住 operator/session。若要使用 learned continuous embedding，只能用于离线提案/距离，最终 macro必须反序列化为有限模板。

### 4.4 为什么 `K_max=16, L_max=4`

第一版建议最多 16 个语义 macro，每个 2–4 个 event atoms：

- 16 是 pilot 上可逐项审阅、做 usage/死码统计和 counterfactual swap 的上限，不声称是最优值；
- `K={4,8,16,32}` 必须作为 rate–distortion 消融；
- 少于 2 个 atom 就退化为 ETG/event label；大于 4 个 atom 很容易把完整 task记成单一 macro，失去跨任务复用；
- `NULL/UNK/PAD` 独立保留，不占 16 个语义槽。

词典的有限性来自：固定 `K_max`、固定 `L_max`、有限 atom schema、有限 role binding 与 duration/termination bins。不要把 prototype 存成 256D/1024D 可训练向量再宣称“语义有限”；那只是 `K` 个 learned high-bandwidth code embeddings，并且已落入 MOTIF/XSkill 的直接邻域。

### 4.5 episode planner 与 action compiler 的分工

**Episode planner：**

- 输入：current robot reference、短 causal history、state、text、embodiment manifest、上一轮 hard macro/event state；
- 输出：未来最多若干 macro instances 的 `macro_id`、object/role bindings、coarse onset/duration、当前 macro progress与termination belief；
- 只在 event/macro clock 更新，不把 15 秒 episode压成 2.4 秒 action chunk；
- 可在训练期接受 human-derived macro target，但部署不读 human video。

**Macro compiler：**

- 纯确定性、无 learned hidden、无 RGB/text 输入；
- 按当前 macro pointer 展开 active atom 与 next atoms，填入 `event_tube_v2` 的 active/next slots；
- 结合当前有限 object binding和剩余 duration bins，不产生连续 trajectory、ROI feature或自由文本；
- 同一 macro 在不同 embodiment 下展开为相同 contact/effect target，具体 action由 action expert结合 state/embodiment生成。

**Action expert：**

- 不直接读取 `macro_id`、human frames、macro continuous prototype、pair ID或coupling；
- 只读取编译后的 hard v2 packet、state、允许的 text/noisy action/embodiment inputs；
- 因而“一 human macro 对多 robot executions”不是查表返回多条动作，而是条件分布：

```text
p(a_chunk | current_robot_state, embodiment, compiled_event_packet)
```

这比保存一个 canonical human trajectory 更符合跨形态工具操作：目标相同，执行路径可不同。

### 4.6 与 `event_tube_v2` 的关系

SCEMC 不新增 action-visible channel。它只约束 episode planner 可生成的 event 序列，并确定性编译到已有 v2 表示：

- action-visible theoretical upper bound仍由 v2 codec决定；motif ID 不额外加 bit；
- 词典会缩小实际 reachable set 和经验 entropy，但不能把这种经验压缩写成互信息证明；
- v1 22D仍冻结为 legacy baseline，不能和 v2/macro拼接后把收益归给 motif；
- 若 macro 层需要 action 直接读取 learned motif embedding才有效，SCEMC strict claim失败。

因此 Idea L 不与 SM-FET-WAM 竞争主 action schema。它只回答一个更窄问题：**event tube 的 episode-level prior 是否应该是可复用 macro grammar，而不是独立 event slots。**

## 5. WAM 插入点与 strict route

当前基线中可复用的位置：

- v1 finite schema/capacity：`world_action_model/contact_transition_schema.py:65-121,168-217`；
- 当前 planner 与 raw hidden isolation：`world_action_model/models/transformer_wa_casual_mot.py:1253-1285,1335-1403`；
- 当前唯一 `22 -> action width` route：同文件 `:1405-1445`；
- MoT attention mask：同文件 `:1809-1822`；
- strict cache与 image-conditioning拒绝：同文件 `:2296-2335,2359-2499`；
- sidecar loader：`world_action_model/datasets/wam_lerobot_dataset.py:285-325,741-819`；
- grouped supervision/trainer：`world_action_model/trainer/wa_casual_trainer_mot.py:111-189,957-1055`。

未来只有在 annotation/pair/dictionary gates 通过后，才考虑新增独立 `macro_dictionary_v1` schema、offline dictionary receipt 和 deterministic compiler。不能静默修改 `contact_transition_schema.py` 的 v1.1，也不能在当前 branch实施代码。

### 防 visual bypass 的硬规则

1. human/reference training video进入 action的唯一允许影响是：`video -> hard atom/macro supervision -> robot planner -> hard v2 packet`。
2. action rows不能读取 human/reference raw tokens、macro encoder hidden、VQ code vectors、pair/coupling、continuous hand pose或ROI feature。
3. compiler不能接收 RGB、text、planner hidden或soft logits；它只接收 hard macro ID、finite bindings/progress和versioned dictionary。
4. action-only cache只保存 hard v2 bytes与必要的 finite macro pointer/receipt；不保存 human feature、continuous prototype或dense posterior。
5. 固定 compiled v2 packet、state、text、noise后替换 human/reference video，pre-safety action应保持不变；如果当前 robot reference是允许的 planner输入，则必须固定 planner output再做 action-side clamp。
6. dictionary canonicalization要拒绝 NaN/Inf/`-0.0`、自由 float centroid、query-order语义和未哈希 prototype。
7. action对 raw macro/event logits、human embedding和pair table梯度为零；offline pairing全程 stop-gradient。
8. human visual identity、operator/session/object instance在固定 packet后的 action hidden probe不得超过预注册 chance+2 pp。

## 6. many-to-many 配对不是“把每对轨迹都展开”

推荐三层数据结构：

```text
episode_candidate_v1
  human_clip_id / robot_rollout_id / task_family / effect / outcome
  source groups / split / confidence / provenance

event_pair_v1
  human_event_id / robot_event_id
  match | nonmatch | ambiguous | unreviewed
  null masses / cost terms / coupling mass / solver receipt

macro_instance_v1
  macro_dictionary_hash / macro_id
  source_domain / source_episode / event_subsequence
  role bindings / entry / termination / effect / outcome
  equivalence_class_id / review status
```

一个 human macro instance 可连多个 robot macro instances；反之亦然。训练单位是 equivalence class/edge，不是人为挑出的唯一伴侣。下列配对必须区分：

- **positive：** functional role、contact/relation delta、effect/outcome与局部顺序一致，路径/时长/手型可不同；
- **hard negative：** 对象和语言相同但 effect相反、contact part错误、顺序颠倒、只产生运动未产生效果、spill/drop/collision；
- **NULL：** prep、idle、camera/head motion、robot homing、只在一侧存在的 regrasp/recovery；
- **ambiguous/unknown：** 证据不足，mask或软权重，不当 negative也不自动 drop 成无关。

不要先枚举 human×robot 所有 pair再随机切分。必须先按 source clip/rollout、operator、session/day、object instance、collection batch与calibration group切分，然后仅在同一 split 内生成 edge和macro class。

## 7. 最小配对数据协议

当前 66 个 human pour 与旧 88 个 dual-Inspire grab 没有同 task/effect positive overlap。它们只能用于：

- human侧 atom/macro可标注性与 K/L coverage统计；
- robot侧 ontology复用性检查；
- pour↔grab 作为 task/effect mismatch 与 NULL-heavy hard negative；
- synthetic prep/idle/regrasp插入下的 matcher rejection测试。

不能用它们验证 human-to-target-robot motif transfer、many-to-many positive alignment、RH56+OmniPicker benefit或真实 object effect。

未来最小 paired pilot沿用 36 human + 36 target robot 的 group设计：

```text
3 task families
* 2 physical object-instance conditions
* 2 session/day groups
= 12 cells

每 cell：3 human clips + 3 robot rollouts
总计：36 human + 36 robot
```

每个 cell 不采同步复制，而要有意制造 execution multiplicity：

- human：一个标准成功、一个额外 prep/速度/握法变化、一个 regrasp或 partial/failure；
- robot：至少两个达到相同 effect但路径/速度/左右执行分工不同的成功 rollout，再加一个 partial/failure；
- 同 cell 内按 effect/outcome建立 3×3 candidate graph，不预选唯一一对；
- 每个 family至少人工确认 20 个 match、20 个 NULL、20 个 hard-negative event edges；
- 所有 72 段标 event anchors；至少 20% 双标并对全部分歧裁决。

任务 family应覆盖不同 contact dynamics，而不能全是抓放：例如 persistent slide/scrape、load/impact/press、tilt/flow/transfer；如果目标 RH56+OmniPicker 的硬件任务最终仍采用 pick/place、双手 twist/press/insert、pour/transfer，则至少保证每个 family 内存在两种 effect-equivalent robot execution，不用人手轨迹规定机器人路径。

配对标注只围绕 key interval。对整个 episode标精确 hand pose trajectory既昂贵又与本 hypothesis矛盾；连续 2D/MANO只在少量子集作 proposal/oracle upper bound。

## 8. Baselines 与 ablations

### 8.1 必须 baselines

| 类别 | baseline |
|---|---|
| 单帧 | single current hand pose；single contact keyframe；single next keypose；direct current-frame WAM |
| 全轨迹 | full human trajectory DTW；ImMimic-style mapped DTW/GMS-SDTW；dense pose/contact tube |
| 稀疏对齐 | subsequence DTW；Drop-DTW；REALIGN-style partial GWOT；CAPOT；EgoBridge-style OT |
| 进度/步骤 | TCN/TCC/XIRL phase embedding；StepFormer key-step slots |
| 跨 embodiment skill | XSkill；UniSkill；2026 MOTIF；unpaired EgoMimic-style co-training |
| skill discovery | CompILE；Bottom-Up Skill Discovery；LOTUS；LOVE；PRISE |
| 现有结构 | 22D-only；ETG current-delta；SM-FET independent event heads；SM-FET active+next2 |
| macro control | random fixed macro dictionary；frequency-only BPE；unconstrained VQ same K/bits；task-phase labels；equal-capacity latent option |
| 上界 | oracle event boundaries；oracle macro assignment；continuous trajectory action-visible upper bound；direct human prompt upper bound |

策略比较必须匹配 robot demonstrations、human clips、backbone、optimizer steps、planner/action输出、action horizon、参数/FLOPs、augmentation和split。SCEMC 的 action-visible v2输出与 independent-event baseline必须完全一致，才能把差异归于 episode prior。

### 8.2 最关键消融

- `K={4,8,16,32}`，`L_max={2,3,4,8}`；
- 去 contact mode、去 relation delta、去 effect/outcome；
- 允许/禁止跨 NULL merge；
- human-only、robot-only、joint human+robot词典；
- pair-specific CAPOT vs global macro equivalence class；
- one-to-one selected pair vs same-cell many-to-many graph；
- MDL/BPE vs unconstrained VQ vs人工 event grammar；
- semantic macro vs fixed random permutation/equal-cardinality macro；
- planner只预测 independent events vs motif-constrained expansion；
- motif ID直接送 action的 upper bound vs compiler-only strict route；
- training-only human video vs execution-time human prompt upper bound；
- uniform frame sampling vs event-aligned sparse subsequence。

## 9. 指标与可证伪预测

### 9.1 discovery/compression

- event token compression ratio、held-out description length与reconstruction/edit error；
- effective dictionary size/perplexity、dead macro ratio、macro length分布；
- contact/effect/outcome purity、unknown率、single-group concentration；
- macro对 task ID、embodiment、operator/session的互信息/probe；
- train-only dictionary在held-out object/operator/embodiment上的 `UNK`率；
- 可复用性：每个 macro跨多少 clips、groups、tasks/domains出现。

### 9.2 pairing

- gold event/macro edge precision/recall/F1、NULL AUPRC/F1；
- effect consistency、same-macro retrieval R@1/R@5；
- crossing rate、transported/null mass；
- prep/idle insertion后的 normalized alignment/macro edit distance；
- 一个 macro下的 robot action diversity与 outcome consistency，证明不是单轨迹复刻。

### 9.3 planner/policy（未来才允许）

- event/macro boundary F1、event order edit score、duration NLL/ECE；
- correct-first-contact、contact maintenance、correct stop/release、effect/outcome success；
- preparation-motion ratio、post-success overrun、spill/drop/collision；
- oracle vs predicted v2 gap、human-data scaling曲线、group-OOD success；
- fixed-packet reference swap、macro intervention locality、image/pair-ID probe与empirical packet entropy。

### 9.4 预注册预测

1. **稀疏性：** key intervals占原 episode帧数的中位数不超过 40%，同时覆盖至少 90% 的 adjudicated contact/effect events；否则“只关注关键段”不成立。
2. **可压缩性：** `K<=16, L<=4` 在held-out groups上相对未合并 event atoms降低至少 20% description length，且 effect purity不低于 0.80；否则词典只是重新命名 event labels。
3. **prep不变性：** 对同一操作人工插入/拉长 prep、idle、camera motion后，macro sequence normalized edit distance不超过 0.10，关键 macro边界漂移中位数不超过一个 event interval；否则 skip机制失败。
4. **多执行性：** 每个保留 macro至少由两个 source groups支持，并在paired pilot中至少有两个不同 robot executions达到同一 effect；若 macro几乎一一绑定task/trajectory，它不是复用 abstraction。
5. **跨形态语义：** 在group-balanced split上，macro ID的 effect/contact classifier显著高于 chance，而 embodiment/session probe不超过相应 chance+5 pp；若做不到，应降为 morphology-specific skill code。
6. **全局词典增益：** 在相同 event features下，SCEMC 对held-out cross-domain macro retrieval或NULL rejection至少比最佳 pair-specific Drop-DTW/CAPOT/REALIGN-style baseline高 5 pp；否则删除 “dictionary” 贡献，只保留 matcher。
7. **planner增益：** 在相同 compiled v2输出与policy下，macro-constrained planner的 event-order F1/phase edit score至少比 independent event planner高 5 pp，或在 outcome-sensitive policy上带来至少 5 pp success提升且95% CI下界大于0；否则删除 macro planner。
8. **无运行时 human依赖：** 在human prompt完全移除后保留主要增益；如果只有execution-time prompt有效，应另立 Vid2Robot/XSkill式研究，不混入本 claim。

这些数字是未来 pilot 的预注册建议，不是当前结果。

## 10. Stop gates

| Gate | 最小输入 | 通过条件 | 失败动作 |
|---|---|---|---|
| **L0 annotation/recurrence** | 12个human episode双标+裁决 | event/contact/effect一致性先通过SM-FET G0；`K<=16,L<=4`达到≥20%压缩、effect purity≥0.80、≥60% macro instances跨两个groups复现 | 指南最多修订一次；仍失败则停止 motif dictionary，只保留event tube |
| **L1 cross-domain pair** | 至少12 human+12同task robot与gold NULL/match edges | global macro retrieval/NULL metric比最佳 pair-specific baseline高≥5 pp；无all-null/单task collapse；prep insertion预测通过 | 不胜Drop-DTW/CAPOT/REALIGN：删除dictionary/pair novelty，pairer只作数据工具 |
| **L2 planner value** | 冻结dictionary、matched event outputs、oracle/predicted planner audit | macro-constrained planner比independent event heads高≥5 pp event-order或policy success；action-visible bits不增加 | 只在直接motif latent/action route下有效：strict SCEMC No-Go |
| **L3 human transfer** | 同robot budget、human data on/off、group-OOD | human macro distillation相对robot-only和unpaired co-training有稳定增益；不由一个task/object/session驱动 | 无增益则删除human pairing claim；保留robot-only temporal grammar最多作工程优化 |

任何阶段出现以下情况应立即止损：

- effective macro数少于4或死码超过50%；
- 最常见 macro只是 approach/hold/release通用phase，没有contact/effect区分；
- macro几乎等价于task ID、object appearance或session；
- oracle event tube已经足够，但macro prior不改善预测；
- continuous pose/trajectory必须直接 action-visible 才能达到效果；
- 同task/effect human–robot数据仍不存在，却继续声称cross-embodiment transfer；
- 对齐收益仅来自更多人工标签、更多参数、更多human frames或execution-time human prompt。

## 11. 与现有主方案的最终关系

| 组件 | 最终角色 |
|---|---|
| SM-FET-WAM | **仍是唯一主 action architecture**；active/near-future event tube解决多姿态、多接触阶段问题 |
| SCEMC / Idea L | **条件式 episode-planner prior/challenger**；把2–4个event atoms组合为可复用macro，不新增action route |
| CAPOT | 离线 pair-edge builder；给SCEMC equivalence class提供候选，不进入部署 |
| Drop-DTW / REALIGN | 必须强baseline；它们已经覆盖skip/partial/structural alignment |
| ETG | 单event state-delta baseline；SCEMC是多event组合，不另建graph message passing |
| continuous hand/MANO tube | annotation proposal、training auxiliary与upper bound；主route No-Go |
| 22D v1 | 冻结legacy baseline；不与macro/v2拼接 |

最重要的比较不是“motif图看起来是否漂亮”，而是：

> 在 action expert看到完全相同的有限 v2 packet 时，macro grammar 是否真的让 planner 更能忽略人类准备段，并从多个效果等价的人机执行中学到可复用的事件组合？

如果答案是否定的，Idea L 不应进入主模型。现有 SM-FET + CAPOT 已经足够；继续堆词典只会增加一个难以归因、又被 MOTIF/XSkill/PRISE 强覆盖的模块。

## 12. 当前 Go / No-Go

| 决策项 | 结论 |
|---|---|
| standalone “Interaction Motif Dictionary” paper claim | **No-Go** |
| “首次 action motif + cross-embodiment policy” claim | **No-Go，2026 MOTIF直接占据** |
| “首次 sparse/partial subsequence alignment” claim | **No-Go，Drop-DTW/REALIGN/GMS-SDTW直接占据** |
| SCEMC as finite contact-effect macro prior | **Conditional Go，仅在SM-FET之上作challenger** |
| runtime human video | **No-Go；主部署必须为零human input** |
| direct learned motif embedding进入action | **No-Go；只能确定性编译到hard v2** |
| 当前66 human pour + 88 robot grab做positive transfer | **No-Go；仅ontology/NULL/hard-negative audit** |
| 现在训练/GPU/5B/真机 | **No-Go** |
| 现在允许的工作 | 文献与schema审计、12段双标、recurrence/coverage纸面统计、未来paired protocol设计 |

一句话总结：**用户关于“别对齐全程，要看多个关键手姿—接触变化”的方向是对的；但最有研究价值的形式不是再造一个 action-motif policy，而是把有限 contact-effect event tube压成可复用macro，并严格证明这个episode prior在不增加动作通道、不依赖运行时人类视频时仍有价值。**
