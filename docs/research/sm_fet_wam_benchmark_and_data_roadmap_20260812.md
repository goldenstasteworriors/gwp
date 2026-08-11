# SM-FET-WAM：不训练前提下的 Benchmark 与数据路线图

- 日期：2026-08-12
- canonical 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 范围：独立审查现有 idea、数据与未来实验顺序；本轮没有训练、没有调用 GPU/服务器、没有修改模型代码
- `idea_id`：`b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`
- 本轮 `discussion_id`：`c947c620-3f8a-42b1-a52c-f2a11cee581b`

## 0. 结论先行

当前最合理的工作顺序不是“先用 66+88 跑一个 WAM”，而是把五个不同科学问题拆开，逐层解锁：

```text
D0  schema/annotation contract dry-run
 |
D1  3 calibration + 12 blind dual annotation
 |---- D1-M: missing-hand / censor stress
 |---- D1-T: transparent-object non-reconstruction audit
 |
D2  36 human + 36 target-robot, 3 task families
 |---- D2-P: 180 reviewed pair decisions, partial/many-to-many benchmark
 |---- D2-R: single snapshot / 3 semantic poses / full trajectory probe
 |
D3  target-robot oracle representation sufficiency
 |
D4  future Ego pretraining -> target-robot fine-tuning -> locked heldout
```

唯一主结构仍是 **SM-FET-WAM v2-min / standard-r1**。CAPOT 只是一张离线、stop-gradient 的训练配对表；多手姿是 planner/encoder 的 supervision 与 probe；推理动作只能读 hard SM-FET packet、机器人 proprio、文本和 diffusion state。当前只授权 D0/D1 的协议工作，D2 等待补采，D3/D4 不应启动。

最重要的三条止损规则：

1. 现有 66 段 human pour 与 88 段 legacy robot grab **永不作为 positive human–robot pair**；只能做 ontology、标注可行性和 task-mismatch hard-negative。
2. 如果 `entry/turn/exit` 三语义姿态不能稳定优于 single snapshot，或仍显著落后 full trajectory，就删除“稀疏多姿态足够”的 claim，不把连续轨迹偷偷接入 finite action route。
3. 如果 partial/many-to-many 不胜 Drop-DTW、GMS-SDTW、EgoBridge-style balanced OT 等最强 baseline，CAPOT 降级为内部数据工具；如果 oracle SM-FET 不胜 22D，整个 v2 action route No-Go。

## 1. 权威现状与本路线图能证明什么

### 1.1 当前两套数据

只读核验结果：

| 数据 | 权威位置 | 当前事实 | 本路线图允许用途 |
|---|---|---|---|
| Ego human pour | `/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers` | 66 个 RGB MP4、66 个普通 HDF5，另有 depth HDF5；单任务；关键 subject/session/operator/object/liquid/batch group provenance 未恢复 | D1 的 3+12；human-only event/pose/missingness/visibility feasibility；剩余集在 D1 通过后可作 pour ontology 扩展，但不能作可信 OOD |
| legacy robot grab | `/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/grab_red_bottle` | 88 个 episode、40,901 frames、50 Hz、单任务 `grab the red bottle`；41D 是 body 29 + 左 Inspire6 + 右 Inspire6，不是 RH56DFTP + OmniPicker | loader/schema dry-run；robot-domain event vocabulary sanity；与 human pour 的固定 task/effect mismatch stress；不能替代目标硬件动作或 outcome truth |

当前 prescreen 还显示：66 Ego 中 boundary/source-rim/target-rim 有一定 2D 可标潜力，但 stream 与 liquid-contact interface 的像素可局部化率为 0%；这些只是 100 帧单人 prescreen，不是 formal gold。raw depth 全图 invalid 比例也不能归因于透明器皿。故本路线图主动不建立 mesh、transparent depth completion 或 liquid reconstruction GT。

### 1.2 v2-min 的固定对象

本路线图评测的不是原 825-float 草案，而是：

```text
standard-r1 action packet
  4 fixed role nodes
  active + next_1 + next_2 events
  <=2 canonical edges/event
  82 uint8 codes / 13 hard tokens
  exact reachable states:
  345864043392188289642033658185203030253953267411599196753571073
  exact capacity: 207.749746919520 bit
```

冻结 legacy baseline：

```text
contact_segment_v1_1
  22D/query, Q12, B16
  48,673 states/query
  exact Q12 states:
  176789872234370550440845675149006851640243221810058817921
  exact capacity: 186.850008943084 bit
```

两者相差 `20.899737976436 bit`，约是 alphabet size 的 `1,956,356.66×`。因此 **22D 与 SM-FET 的原生比较不是 equal-capacity comparison**。任何结果必须同时包含各自的 exact-capacity equal-bit control，且不得把容量差异直接解释成语义优势。

### 1.3 对全部现有 ideas 的独立裁决

| idea | 在 roadmap 中的位置 | 当前决定 |
|---|---|---|
| A / finite 22D contact WAM | 冻结 legacy baseline、route audit | 保留，不与 v2 concat |
| B / asymmetric contact topology | `stabilize_twist_cap` 与目标硬件 stress | 工程边界；不单列 novelty |
| C / visibility-gated 2D rim/boundary | D1-T training-only auxiliary probe | Conditional；stream/contact dense head No-Go |
| D / missingness/censor | D1-M 必需数据可信性 gate | 必须做；不单独膨胀为主模型 |
| E / contact-action consistency | D3/D4 realized-effect audit | 目标 robot native truth 到位后再开 |
| F / semi-Markov finite event tube | 唯一主 action architecture | Conditional Go |
| G / partial human–robot pairing | D2-P 离线 matcher benchmark | Conditional；runtime CAPOT No-Go |
| H / event-transition graph | `relation before/after + effect` 的强 baseline | 不再加独立 graph runtime |
| I / causal critical windows | D2-R 的 event-window selector ablation | 仅 challenger，不进当前主结构 |
| J / F-PCD | 75.4458-bit pose/contact derivative challenger | 与 v2 互斥比较 |
| K / effect equivalence | candidate gate / effect grouping | training-only grouping，不作 headline |
| L / motif dictionary | D4 episode-prior challenger | MOTIF/skill discovery 前例过强，非主线 |

## 2. 所有阶段共享的数据与统计纪律

### 2.1 先 split，后建标签窗口和 pair

唯一合法顺序：

```text
source inventory + authoritative group metadata
-> group/cell split receipt
-> clip-level event/node/edge/pose annotation
-> within-split candidate graph
-> pair gold / solver mass
-> windows, cached features, augmentation, training samples
```

同一 source 的裁剪、不同 fps 版本、pose keys、event rows、pair edges、增强和 cached feature 必须同 split。一个 human clip 与多个 robot rollout 形成的 candidate graph connected component 也必须同 split。frame、event、edge、window 都不是独立样本，不能用它们夸大 `n`。

### 2.2 必须冻结的 group 字段

human：

```text
subject_id, session_id, operator_id,
source_object_instance_id, target_object_instance_id,
liquid_id, collection_batch_id,
camera_id, calibration_id, task_ontology_version
```

robot 另加：

```text
collector_id, embodiment_id, safety_config_id,
command_schema_id, measured_state_schema_id,
outcome_sensor_id
```

透明任务另记 `container_material/opacity`、`liquid_color/clarity/viscosity/fill_bin`、光照/背景；不能从视频事后猜。当前 66 缺这些 provenance，只能称 episode-heldout diagnostic。

### 2.3 指标与置信区间

- D1 annotation agreement 以 episode cluster bootstrap 报 95% CI；分类只有至少两个类别且有足够 comparable items 才解释 κ。
- D2 pairing 以 cell 为最低 cluster，并对 human clip 与 robot rollout做交叉 cluster bootstrap；primary 点估计之外必须报告 95% CI。
- D3/D4 rollout 才是独立控制单位；seed、frame、action chunk 不能当独立样本。
- 所有 primary endpoint、threshold、candidate universe、NULL cost、solver budget、early stopping 和排除规则先在 calibration/validation 冻结；test 不调参。
- 多任务报告 per-family 与 macro；不能由容易的 `pick_place` 掩盖 `twist/regrasp` 或 `pour`。

## 3. D0：schema dry-run 与金标可编译性

### 3.1 输入

- standard-r1 schema memo 与 annotation protocol；
- 人工设计的极小合法/非法 fixtures：NULL、UNKNOWN terminal、ABSENT、NOT_REVIEWED、left/right censor、overflow、node reference union、双 edge canonical order、verify-only outcome；
- 从当前 human pour 与 robot grab 各取 1 个只读 sample，用来检查字段是否有源，不进入科学指标；
- legacy 22D exact-capacity receipt。

本阶段不需要把 v2 写入模型。未来 checker 可独立于 WAM，但本轮只定义验收，不实现它。

### 3.2 标注/receipt

每个 action-visible 字段必须映射到唯一 annotation source：

- `events_v1`：kind、focus role/entity、boundary interval；
- `event_node_snapshots_v1`：四 fixed-role node 的 observability/2D cell/scale/orientation；
- `event_edges_v1`：actor、primary/secondary role、primary part/cell、relation before/after、contact mode；
- episode sidecar：coverage、censor、immediate failure；
- `verify_outcome` 才能写 action-visible outcome；
- pose、confidence、pair mass、MANO、depth 不得编译入 standard packet。

每套 fixture 保存 schema/grammar/capacity SHA、source hash、expected canonical bytes 与 round-trip result。

### 3.3 baseline 与主要指标

| 对照 | 目的 |
|---|---|
| legacy 22D validator/receipt | 防止 v2 修改旧 schema 语义 |
| SM-FET standard semantic packet | 主候选 |
| SM-FET whole-packet equal-bit permutation | 同 `S_action`、同 13 token、同参数/FLOPs的无语义编号 control |
| independently permuted slots | 明确的错误 control；应被 cross-slot validator拒绝 |

硬阈值全部为 `100%`：合法 fixture唯一 round-trip；非法 fixture fail-closed；UNKNOWN/ABSENT/MISSING/censor 不互相映射；capacity integer/log2 与 receipt逐字一致；v1 disabled 路径不改变旧行为。

### 3.4 停止条件

任何 action 字段无唯一 annotation source、两份 transition grammar 不一致、82-code packet不能唯一反序列化、或 exact capacity receipt不一致：D0 失败，停止 UI 全字段扩展、D1 blind 与模型实现。

### 3.5 资源估算

| 资源 | 预算 |
|---|---:|
| schema/fixture 双人审阅 | 3–5 人时 |
| coordinator/receipt复核 | 1–2 人时 |
| 计算 | 单机 CPU <2 CPU·h；0 GPU·h |

当前状态：**未通过**。现有代码 `/home/ykj/project/WM/gwp-agent-wam-benchmark-roadmap/world_action_model/contact_transition_schema.py:168,340,417` 只覆盖 legacy finite contact/unstructured capacity；v2 文件尚不存在。

## 4. D1：3 calibration + 12 blind dual annotation

### 4.1 输入与固定样本

使用当前 66 human pour。

- calibration：从 formal 12 之外的 54 段中，按“普通成功 / 长 hold或regrasp / 遮挡或censor/partial”规则冻结 3 段；
- blind：固定 episode `43,50,51,58,34,38,9,54,36,53,12,0`；不能看完后换样本；
- episode 0 的历史 timing hint继续隐藏；
- 3 calibration 不进 D1 数值、validation 或 test。

### 4.2 A/B 标注与裁决

A/B 在隔离 packet 中独立完成：

1. coverage/exhaustiveness 与 missing reason；
2. macro event interval/kind/focus；
3. 四 role node snapshots；
4. contact/object-relation edge、effect 与 verify outcome；
5. 独立 pose probe：entry、optional turn、exit 的 7 functional 2D landmarks、reference bbox、visibility；
6. 第三人在 aggregate gate通过后才裁决；不能以裁决覆盖系统性分歧。

三段 calibration 后只允许共同修订一次 protocol，然后冻结 hash。blind 期间不改字典。

### 4.3 split/group

D1 不是 policy train/val/test。3 段是 protocol calibration，12 段是锁定的 protocol test；所有 frame/event以 source episode聚类。由于真实 group metadata缺失，不能作 subject/object/session/liquid OOD 主张。

### 4.4 baseline

- legacy 22D contact-segment annotation：时间、字段 coverage 和人工成本基线；
- SM-FET core event/node/edge；
- `single contact snapshot` pose probe；
- `entry/turn/exit` 三语义姿态 probe；
- full event clip只作 human-information upper bound，不作为 action packet。

core 与 pose probe分开报告：pose失败不能把 core判过，pose成功也不能补救 core失败。

### 4.5 指标与阈值

| 层级 | primary metric | 门槛 |
|---|---|---:|
| coverage | weighted Cohen κ | `>=0.70` |
| event existence | precision/recall/F1 | `F1>=0.80` |
| interval | matched median tIoU | `>=0.70` |
| boundary | P50 absolute error | `<=2 processed frames` 或 `<=100 ms` |
| kind/focus/role/part | Cohen κ | 各 `>=0.70` |
| node observability/scale/orientation | Cohen κ | 各 `>=0.70` |
| node center cell | exact agreement | `>=0.70` |
| contact mode | Cohen κ | `>=0.60` |
| relation/effect/outcome | Cohen κ | 各 `>=0.70` |
| visibility/censor | Cohen κ | `>=0.70` |
| pose | visible landmark image-diagonal NME P50；visibility κ | `<=3%`；`>=0.70`，仅 probe |
| capacity | `Q>8`、node>4、edge/event>2 | 各 `<=5%` |
| missingness | 关键 effect/outcome 双侧 unknown | `<=40%` |

单类或稀疏字段写 `NOT_EVALUABLE`，不能以 100% agreement过门。D1 全是 pour，只批准实际覆盖的 pour 局部字段。

### 4.6 停止条件与资源

- 任一 core primary gate失败：最多修订一次新 protocol，并用新的 disjoint calibration复核；仍失败则停止扩到 66、停止完整 v2 UI/action packet。
- `NOT_EVALUABLE` 不等于失败，也不等于通过；等待 D2 跨任务/跨域覆盖。

| 工作 | 人时 |
|---|---:|
| 3 calibration、讨论、一次冻结 | 3–4 |
| 12 blind A/B integrated pass | 8–11 |
| 第三人裁决 | 2.5–3.5 |
| coordinator/receipt/抽查 | 1–1.5 |
| **合计** | **14.5–20** |
| 计算 | 视频读取、hash、汇总约 2–6 CPU·h；0 GPU·h |

## 5. D1-M：missing-hand 与 censor benchmark

### 5.1 为什么自然缺失和人工 corruption必须分开

自然 missing-hand 只有“看不到/没标到”，常常没有反事实真值；不能把 unknown 当 negative。可量化 recovery 的 synthetic corruption 只能从双方确认、coverage exhaustive、字段完整的 adjudicated events生成，再按真实 missing pattern遮掉一侧、边界或字段。

### 5.2 输入与标注

第一阶段只用 D1 adjudicated 12：

- natural strata：left/right absent、out-of-view、occluded、motion blur、窗口 left/right censor、positive-only；
- synthetic strata：对完整 gold生成 side-drop、contiguous temporal gap、boundary truncation、field-selective masking；
- severity 在 calibration冻结，例如 known coverage 75/50/25%，不能看 test后改；
- D1 通过后，才可从剩余 51 段扩一个 pour-only missingness pool；仍不能声称 group OOD。

### 5.3 baselines

1. `naive-zero`：unknown当 negative，作为明确错误 baseline；
2. reviewed-field mask；
3. mask + missing-indicator；
4. interval/censor-aware survival likelihood；
5. single-hand only；
6. bilateral available-case；
7. full-label oracle ceiling。

所有方法必须看相同可见字段；不可用 tracker/MANO 自动填补后继续称“missingness-aware”。

split 继承原 source episode：calibration-derived corruption只能用于冻结 severity；blind episode及其所有 corruption descendants属于同一 locked test，绝不按mask实例随机切分。扩到剩余51段时先以episode冻结 development/test清单；因group provenance缺失，只允许episode-heldout diagnostic。

### 5.4 metric、阈值与停止条件

- natural：per-side coverage、missing-reason κ、unknown rate、boundary interval width、左右手不对称；
- corruption：event F1、field macro-F1、censored NLL、integrated Brier score、ECE、boundary interval coverage；
- 证明 missingness 确有偏差：`naive-zero` 相对 reviewed-mask 至少差 `3 F1 pp`，或 Brier恶化 `>=10% relative`；否则 D 可保留为数据卫生，不作方法贡献；
- censor-aware 相对最强 masked baseline：NLL/IBS改善 `>=10% relative` 或 event F1 `>=+3 pp`，且至少两个预注册 regime方向一致；否则不用增加复杂 hazard/MNAR 模型；
- 任一方法通过把 unknown预测成易负例获得 accuracy：主 metric仍按 macro-F1/calibration判失败。

资源：D1 labels上约 3–5 人时设计/复核 corruption，2–8 CPU·h；若扩剩余 human pool，另加约 12–20 人时。当前不训练任何网络，0 GPU·h。

## 6. D1-T：透明物体的 non-reconstruction benchmark

### 6.1 当前能检验的窄命题

当前只能检验：**不构造伪 3D 时，event、role、可见 rim/boundary、effect interval是否仍可被人一致标注。** 不能检验“透明与不透明的因果差异”，也不能证明 downstream manipulation 已不需要重建。

### 6.2 输入与 formal set

- 现有 100-frame phase-stratified audit只作 prescreen，不作 formal gold；其中 episode与 D1有重叠；
- formal 建议从 D1/calibration之外冻结 6 个 episode ×20 frames，共120帧，覆盖 approach/contact/manipulate/release/verify与遮挡；
- RGB 为标注证据；raw depth只报告 invalidity，不作透明真值；
- 若现有 66 无明确 opaque/colored control，则所有 transparency comparison标 `NOT_EVALUABLE`。

### 6.3 标注、baseline 和 metric

标注状态严格分：`ABSENT / PRESENT_UNLOCALIZABLE / PARTIAL / CLEAR`。只画可见 container boundary、source rim、target rim；liquid surface/stream/contact interface只标存在性、可见性与 event interval。

| baseline/probe | 允许用途 |
|---|---|
| RGB event + role/contact cells | 主 non-reconstruction contract |
| RGB + visible 2D rim/boundary | training-only spatial auxiliary候选 |
| RGB + raw depth | sensor input baseline；不把零值补成 GT |
| frozen transparent depth completion | proposal/外部 baseline，永不成为 gold |
| dense stream/contact mask | 当前 No-Go |

门槛：boundary/rim localizable status κ `>=0.70`；ABSENT 与 PRESENT_UNLOCALIZABLE 分歧率 `<=10%`；可见 rim geometry用 normalized center/ellipse/rim distance报告；stream/contact-interface若仍无足够清楚样本，必须标 `NOT_EVALUABLE`，而不是降阈值。

这120帧按source episode聚类且只作locked formal audit；同一episode的相邻帧不得跨split。没有可靠 material/opacity/liquid group时不构造“透明train、非透明test”之类伪OOD split。

### 6.4 下一层真正的 sufficiency test

D1-T通过只说明可标。D3-T 才能比较：

```text
RGB-only SM-FET
vs RGB + raw depth
vs RGB + transparent-depth completion baseline
vs RGB + visible-only 2D rim auxiliary
vs sensor-confirmed coarse outcome oracle
```

要求 non-reconstruction SM-FET 相对重建路线的 task success差距不超过 `3 pp`，同时 spill/rim-collision不退化；若 completion只在独立容器/光照上不稳定，删除它而不是把 pseudo depth当 truth。若只有 dense reconstruction action-visible才成功，停止“无需重建”的 claim。

资源：formal 120帧 A/B+裁决约 6–10 人时，depth/RGB统计 2–6 CPU·h，0 GPU·h。

## 7. D2：36 human + 36 target-robot 三任务族补采

### 7.1 输入与任务族

robot 必须是目标形态：Unitree G1 + 左 RH56DFTP + 右 OmniPicker。固定三个互补任务族：

| family | 功能 transition | 建议硬件角色 | 独立 outcome truth |
|---|---|---|---|
| `pick_place` | separate→held→inside/on→release | 右 OmniPicker主抓放；左手不参与基线 | 稳定 inside/on；人工+俯视/位姿 |
| `stabilize_twist_cap` | stabilize + acquire/rotate/regrasp→open/close | 右 OmniPicker稳瓶；左 RH56扭盖 | cap角度/encoder + 视频 |
| `pour_transfer` | held→rim alignment→transfer→upright→verify | 左 RH56持 source；右 OmniPicker稳 target | 秤/液位差；spill另记 |

若硬件 gate 认为角色分工不安全或不可达，只能在采集前修改 ontology version并重新冻结；不能采完后按成功样本重定义。

### 7.2 12 cells 与样本组成

```text
3 families × 2 physical object conditions × 2 session/day = 12 cells
每 cell：3 human + 3 target-robot
总计：36 human + 36 robot
```

每 cell 的 human：标准成功、prep/速度变化成功、regrasp/partial/failure。robot：两个成功或可恢复 + 一个 partial/failure；失败不能重录删除。

pour 的两个 object condition应尽量形成透明难例与视觉较易 control，同时匹配几何和任务：例如 clear container/clear liquid 与外部 marker或colored-liquid control；记录 material/opacity/clarity，不能称完全因果配对。

### 7.3 split/group

每 family 四个 cells固定为 `2 train / 1 validation / 1 test`，总计 `6/3/3 cells`；每域为 `18/9/9 clips`。一个 cell 的 3 human、3 robot及其全部 rows/edges同 split。

这只是 held-out object-session **cell**；同一个 object或session仍可能在其它 split出现，不能写 novel-object 或 novel-session OOD。投稿级独立 OOD 需增加至少第三个、最好更多 object/session group。

每 family从 train cell选一个 paired calibration unit（1 human +1 robot，共6 videos）校准跨域 annotation/pair定义；不进 test和 D2 point metric，协议冻结后可标记为 train-only。

### 7.4 annotation 与 pair gold

- 72 clips全部做单人 event/node/edge/pose review；
- 至少15 clips做第二人 blind review并裁决，覆盖每 family与human/robot域；
- candidate universe在 solver score前按 family/role/effect/难度冻结；
- 至少180条 A/B pair decisions，每 family各20 match、20 one-sided unmatched/NULL、20 hard negative；
- 建议按 split分配每 family每类 `10 train / 5 val / 5 test`，总计 `90/45/45` decisions；
- 未抽中的 edge是 `unreviewed`，不是 negative；ambiguous不强制二值化。

### 7.5 annotation/data gate

D1 的 agreement门槛在 human/robot域分别复核；另要求：

- pair `match/nonmatch/ambiguous` weighted κ `>=0.70`；
- adjudicated positive edge F1 `>=0.80`；
- human/robot unmatched F1各 `>=0.80`；
- effect-match κ `>=0.70`；
- gold positive crossing `=0`；
- 每个 action-visible字段必须有至少两个有支持类别，否则仍 `NOT_EVALUABLE`。

### 7.6 资源

| 工作 | 人时 |
|---|---:|
| human capture、setup、metadata/outcome | 6–9 |
| robot capture、reset、安全观察、传感检查 | 12–18 |
| 72单人标注 + 15第二标 | 24–36 |
| 180 pair A/B + 裁决 | 5–8 |
| coordinator/group/split/hash/QA | 3–5 |
| **合计** | **50–76** |
| 计算 | 视频/hash/solver sweep约 20–60 CPU·h；0 GPU·h可完成 gold 与经典 alignment benchmark |

不含 driver、mapper、calibration、watchdog 开发。36+36 只够 feasibility，不是 WAM policy 数据量承诺。

## 8. D2-R：single snapshot vs 三语义姿态 vs full trajectory

### 8.1 两个不同问题必须分开

1. **信息来源问题**：单帧、三语义帧、完整 event clip各保留多少 event/contact/effect信息？full trajectory天然高带宽，不能叫 equal-bit。
2. **WAM 归纳偏置问题**：在相同 action packet、token/参数/FLOPs下，三语义姿态 supervision是否比单帧或均匀采样更有用？这只能在 D3/D4回答。

### 8.2 representation arms

对同一个 adjudicated event构造：

| arm | 输入 | 公平性角色 |
|---|---|---|
| `S1-contact-repeat3` | 第一个稳定 contact anchor重复到3个位置 | 与 S3 token数相同、无额外时间信息的 single-snapshot主对照 |
| `S1-next-pose-repeat3` | next keypose重复3次 | 防止只选错单帧造成 strawman |
| `S3-semantic` | entry、optional turn、exit；turn缺失用ABSENT，不插值 | 主假设 |
| `U3-uniform` | event support内均匀3帧 | 同帧数、非语义采样 |
| `M3-motion-peak` |同预算3个motion peaks | CET/F-PCD 类强 challenger |
| `W-local` | 三 anchor周围固定总时长local windows | 局部连续运动 challenger |
| `F-full` | 只含 interaction event的完整轨迹，不含episode prep/idle | 信息 upper bound |
| `F-episode` | 从拿取前到结束的完整episode | 检验“关注全程”是否反而受 prep/idle干扰 |

所有 arm 使用同一 source、event boundaries、role boxes、分辨率、instruction和split。S1/S3/U3/M3 的视觉 encoder、3位置embedding、输出维数、probe参数与训练步数相同。不能给 S3 oracle boundary却给 single/full predicted boundary；必须先做全 oracle，再做全 predicted。

`F-full/F-episode` 单独报告实际 frame/token/byte、FLOPs与 latency；它们是性能—带宽 frontier，不是假装 equal-capacity。未来若 full也被压成13 hard tokens，需另称 `full-to-same-packet`，与 action-visible continuous trajectory upper bound分开。

### 8.3 no-policy probe 与 metric

在 D2 可先做统一小 probe或人类盲判，不训练 WAM：

- event kind macro-F1；
- primary role/part、contact mode accuracy；
- relation delta/effect/outcome macro-F1；
- human→robot positive event retrieval Recall@1/mAP；
- NULL rejection AUPRC；
- regrasp/twist/pour子集单报；
- representation成本：frames、tokens、bytes、标注人时。

预注册 evidence gate：

- `S3-semantic` 相对最佳 single snapshot至少 `+5 pp` primary macro-F1，且 cell-cluster 95% CI支持正优势；否则删除 multi-pose贡献；
- S3相对 `F-full`在 effect/contact macro-F1差距 `<=5 pp`，才可称稀疏姿态近似充分；
- S3必须不低于 U3/M3 最优者；若持平，claim收缩为“少量帧足够”，不能强调语义时刻；
- `F-episode`若弱于`F-full`且错误集中在 prep/idle，才支持“不关注全程”的动机；否则不作该 claim；
- D2 样本小，CI宽时只写 feasibility，不用 point estimate发布结论。

### 8.4 后续 WAM 对照

在 D3/D4，各视觉 arm只改变 training evidence/selector，**action-visible packet始终相同**。另设 continuous trajectory action-visible upper bound，但必须报告其无限/高带宽路径，不能与 strict finite route混写。

## 9. D2-P：full 1:1 vs partial/many-to-many pairing

### 9.1 公平输入

所有 alignment 方法使用完全相同：

- adjudicated human/robot macro events；
- episode candidate universe；
- known-field masks；
- role/effect candidate gate；
- feature normalization；
- validation调参次数与CPU预算；
- train/val/test split receipt。

human/robot literal left/right、raw wrist path、逐关节 pose、全程 duration不进入 CAPOT 主 cost；它们只作预注册 ablation。所有 solver output stop-gradient，不进入 action token。

### 9.2 baseline ladder

| baseline | 定义 |
|---|---|
| normalized-time forced 1:1 | 对完整 episode progress强制全质量配对；不能内部 drop |
| forced 1:1 + episode reject | 可拒绝整对 episode，但仍不能跳过内部 prep/idle |
| DTW / soft-DTW | 完整保序路径 |
| Drop-DTW | 双侧内部 outlier drop |
| GMS-SDTW / subsequence DTW | 多子序列/长视频强 baseline |
| EgoBridge-style balanced OT | DTW引导 joint OT、质量守恒 |
| partial OT | 丢弃部分质量，不强制多对多语义 |
| UOT without monotonicity | 检验是否需要顺序约束 |
| CAPOT | contact/effect anchor + partial/unbalanced + non-crossing projection + NULL |
| oracle manual event edges | 上界 |
| no-pair / robot-only | 后续 policy贡献必要对照 |

forced 1:1 本身无法输出内部 NULL，NULL metric会很差；为避免 strawman，主 superiority 必须相对 Drop-DTW/GMS-SDTW/EgoBridge-style 的最优者，而不是只胜 forced 1:1。

### 9.3 gold 与 metric

pair gold允许 one-to-many、many-to-one 与 one-sided unmatched。primary metrics：

- reviewed positive edge macro-F1；
- human NULL、robot NULL F1与NULL AUPRC；
- hard-negative false accept rate；
- effect consistency；
- accepted ambiguous rate；
- projected crossing rate；
- normalized transported mass与all-null/all-match collapse；
- per-family、per-cell与prep/regrasp/failure strata；
- score calibration/ECE。

最低 gate：

```text
NULL AUPRC >= 0.80
unmatched F1 >= 0.80
effect consistency >= 0.80
projected crossing < 1%
CAPOT positive-edge F1 vs strongest baseline >= +5 pp
cell/clip cluster bootstrap 95% CI supports positive gain
no all-null collapse
```

失败：CAPOT只作内部数据工具，删除 matcher novelty与 paired-transfer claim。

### 9.4 legacy mismatch stress

从 66 human pour与88 legacy robot grab按长度/可见性分层冻结有限 pair，不生成 `66×88` 笛卡尔积：

- 只把 effect-bearing mismatch 标 hard negative，例如 transfer/flow vs bottle lift；
- generic approach/acquire/release可能是共享 primitive，不自动标 nonmatch；
- reviewed positive edge必须为0；
- episode false accept `<=5%`；
- transported mass / `min(total_h,total_r) <=0.10`。

这个 split不能进入同任务 positive retrieval、policy transfer或目标硬件结论。

## 10. D2-C：22D vs SM-FET exact-capacity controls

### 10.1 必须诚实报告的三类变量

22D与v2同时改变了：语义字段、事件时域和理论容量。不能用一个二臂结果把三者分开。推荐矩阵：

| route | exact capacity | token/compute处理 | 回答的问题 |
|---|---:|---|---|
| 22D semantic | 186.850008943084 bit | Q12 + 1 constant PAD，统一为13位置；encoder参数/FLOPs matched | legacy baseline |
| 22D global equal-bit | 同上 | 与22D semantic完全相同 | 22D语义 factorization是否必要 |
| SM-FET semantic | 207.749746919520 bit | 13 tokens | 主候选 |
| SM-FET global equal-bit | 同上 | 同 planner/aux/encoder/参数/FLOPs | v2语义 factorization是否必要 |
| F-PCD-only | 75.445813886781 bit | pad到13位置并匹配encoder预算 | pose/contact derivative challenger |
| ETG current-delta-only | 必须单独审计容量 | pad/matched | next-event是否必要 |
| continuous local/full trajectory | 显式报告实际带宽 | 不称 equal-bit | upper bound |

v2 equal-bit必须对整个合法 packet做 enumerative rank permutation：

```text
u = rank(packet)
u_control = (a * u) mod S_action, gcd(a, S_action)=1, a != 1
packet_control = unrank(u_control)
```

逐 slot permutation 会破坏 cross-slot grammar，禁止使用。

### 10.2 为什么不伪造“22D-v2 exact matched”

两个原生 alphabet 不同。把 v2随意裁到186.85 bit会丢字段并改变语义；给22D追加20.90 bit nuisance又会引入新的视觉信息来源。路线图不把这些称干净主对照。正确解释是：

1. 分别做各自 same-alphabet semantic vs global equal-bit；
2. 报经验 entropy、unique code usage与field intervention；
3. 原生22D-v2差异只称“整体 schema trade-off”；
4. 如审稿必须要跨-schema容量曲线，再预注册多个 grammar版本，画 performance-vs-exact-bits frontier，不靠单个任意裁剪点。

### 10.3 结果解释与停止

- v2 semantic≈v2 equal-bit且利用率相当：删除“人类可读字段语义必要”；
- v2只比22D好、但equal-bit也同比好：收益可能是容量/时域，不是语义；
- 22D不差于v2：v2 action route不成立，但event annotation/pairing仍可作为数据工具；
- only `SM-FET + 22D concat`有效：不能把收益归给v2，主实验判失败；
- continuous route明显更好且v2差距>3 pp：停止 finite sufficiency claim。

## 11. D3：target-robot oracle sufficiency

### 11.1 进入条件

D0–D2全部通过；目标硬件 driver/mapper/calibration/watchdog已独立过 gate；robot数据含 native或sensor-confirmed current state、contact、effect/outcome。36 robot feasibility pilot不自动等于足够 policy数据，正式样本量由 pilot rollout方差做 power analysis。

### 11.2 输入、split 与训练隔离

- 只用 robot train data学 executor；human pairing在这一阶段关闭，先隔离 action representation价值；
- oracle packet来自当前因果 robot state/sensor，不得用future frame、人工后验插值或human video；
- cell/group split在生成windows前冻结；
- test objects/sessions/outcomes不参与quantization阈值、normalization、early stop和solver tuning。

### 11.3 baselines

1. robot-only direct WAM；
2. 22D oracle-only；
3. F-PCD oracle-only；
4. ETG current-delta-only；
5. SM-FET active-only / +next1 / +next2；
6. SM-FET global equal-bit；
7. local/full continuous trajectory upper bound；
8. reconstruction/depth-completion baselines只在 D3-T transparent stratum中报告。

所有有限路线互斥，使用同 robot demonstrations、backbone/init、action horizon、optimizer steps、augmentation、seed和safety filter。

### 11.4 primary metric 与 gate

- task success与event completion；
- correct-first-contact、correct-release、regrasp completion；
- pour：transfer amount error、spill mass、rim collision；
- twist：cap angle/error、slip/regrasp；
- safety：drop/collision/limit violation；
- latency与deadline miss。

通过条件：

```text
SM-FET vs oracle 22D-only:
  outcome-sensitive success >= +8 pp
  OR event completion >= +10 pp
SM-FET vs full trajectory:
  success gap <= 3 pp
safety: no degradation
cluster 95% CI supports positive advantage
regrasp/fine-contact subset reported separately
```

D3-T 另要求 RGB-only/2D-rim non-reconstruction route相对最佳 reconstruction baseline差距 `<=3 pp` 且 spill/rim-collision不退化。

失败：SM-FET action route No-Go；event/pair benchmark仍可保留。若 only continuous action-visible pose/trajectory有效，停止 finite claim。

### 11.5 资源预算

数据与模型尚未确定，不能给虚假的固定小时数。预注册计算公式：

```text
total H20 GPU·h
= number_of_arms × seeds × GPUs_per_run × measured_wall_hours_per_run
```

建议先做1-seed短 screen，只让通过无泄漏/正确性的 finalists进入3-seed完整比较；正式 primary comparison仍必须3 seeds。规划量级：小型 frozen/oracle probe约20–80 H20 GPU·h；finalist oracle WAM矩阵约200–600 H20 GPU·h。只有 gate通过后才允许校准实际墙钟，当前为0 GPU·h。

人工：每100个 robot rollout的 outcome/contact sensor QA约10–20人时，采集/安全监护另按本地 SOP与 power analysis计。

## 12. D4：未来 WAM pretrain、fine-tune 与 locked heldout

### 12.1 唯一允许的数据流

```text
human train videos
 -> human event encoder/planner pretraining
 -> entry/turn/exit or local-window auxiliary
 -> no human action pseudo-label

human train events --offline CAPOT--> robot train events
                                  -> sparse stop-gradient pair table

target-robot train image/state/action
 -> robot planner + strict hard packet + WAM action fine-tuning

deployment/test
 -> robot image/history + proprio + text only
 -> no human video, pair ID, OT, MANO or planner hidden
```

### 12.2 split 与数据扩充

36+36只用于 feasibility。D4前基于 D3 variance做 power analysis并扩充 target robot demonstrations；若要 independent object/session OOD，至少增加第三组并让相应 identity跨split完全不相交。建议锁定四张 test表：

- held-out object instance；
- held-out session/day/calibration；
- held-out human operator/subject（planner）；
- transparent/container/liquid condition stress；
- task-compositional test只能由已ground contact primitives组成。

旧88 legacy robot可作跨硬件 diagnostic，但不能混进目标 hardware success，也不能把 Inspire6 action压成OmniPicker action。

### 12.3 matched training matrix

| arm | human supervision | pairing | action route |
|---|---|---|---|
| robot-only | none | none | direct / 22D / v2分别跑 |
| unpaired auxiliary | human event/pose | none | strict v2 |
| forced-1:1 | human event/pose | normalized-time 1:1 | strict v2 |
| best prior aligner | human event/pose | Drop-DTW/GMS/EgoBridge best | strict v2 |
| CAPOT | human event/pose | partial/many-to-many | strict v2 |
| oracle pair | human event/pose | manual edges | strict v2 upper bound |

再正交加入 D2-R 的 single/S3/full evidence和 D2-C 的22D/v2/equal-bit；不能一次跑所有笛卡尔积。先用预注册 screening选择每一轴的 finalist，再做核心 factorial：

```text
pairing: no-pair vs strongest-prior vs CAPOT vs oracle
temporal evidence: single-repeat3 vs S3 vs full-to-same-packet
action representation: 22D vs v2-semantic vs v2-equal-bit
```

所有 arm使用相同 robot budget、human budget、source hashes、initial checkpoint、optimizer/noise/timestep、action horizon、augmentation和seed。pairing只改变 pair table，不能同时多加 robot data。

### 12.4 primary gate

- predicted planner恢复 `>=70%` oracle SM-FET gain；
- strict v2优于 auxiliary-direct、22D-only与v2 equal-bit，并且group-cluster 95% CI支持正优势；
- CAPOT相对 no-pair与最强prior aligner至少 `+5 pp` success或预注册关键phase metric，且CI支持正优势；否则删除human–robot pairing贡献；
- S3相对single至少 `+5 pp`，相对full-to-same-packet差距 `<=3 pp`；否则按 D2-R止损；
- swap/delay/reorder intervention只在对应phase改变动作/effect；cache/swap audit失败则删除causal claim；
- action-only latency增加 `<=10%`，safety不退化。

### 12.5 计算与人时

任何数字都需先由相同 checkpoint 的短 profile冻结。规划量级而非当前授权：

- human event/planner pretraining finalist：约100–400 H20 GPU·h；
- target-robot finalist fine-tune与3-seed核心矩阵：约600–2,000 H20 GPU·h；
- heldout rollout与评估：按任务/seed/power analysis另计；
- 数据QA/新标注：每100 episodes约15–30人时，另加安全监护与第三人裁决。

当前阶段 GPU·h 必须保持0；不能因为4×H20可用而跳过 D0–D3。

## 13. 现有 66 + 88 的逐阶段可用性矩阵

| 阶段/结论 | 66 Ego pour | 88 legacy robot grab | 两者联合 |
|---|---|---|---|
| D0 schema/loader dry-run | 可作human字段源样本 | 可作LeRobot/action-dim样本 | 可检查cross-domain格式，不能科学配对 |
| D1 3+12 | **可用，主数据** | 不进入human blind；可另做robot ontology sanity | 不联合 |
| single vs S3 vs full human信息probe | 可用；只限pour/episode-heldout | 可做robot-domain描述性probe | 不能当同任务cross-domain正检索 |
| natural missing-hand/censor | 可用，先12后条件扩展 | 可统计视觉缺失，但无physical contact GT | 可比较缺失形态，不能合并成同分布 |
| transparent non-reconstruction | 可做rim/boundary/interval可标性；不能透明vs不透明OOD | 抓瓶不是透明倒液control | 不构成透明因果benchmark |
| event ontology | pour局部可用 | acquire/hold/release局部可用 | 可审词表overflow，但generic events不自动为正或负 |
| positive human–robot pair | **不可用** | **不可用** | **永久No-Go**：task/effect/hardware不重叠 |
| hard-negative stress | 提供transfer/effect侧 | 提供grab/lift侧 | **可用**，只标effect-bearing mismatch |
| target hardware oracle | 不可用 | 不可用，Inspire6≠OmniPicker | 不可用 |
| policy transfer/success | 无robot action | 非目标硬件且无匹配human task | 不可用 |
| future Ego pretraining | D1通过且metadata/provenance恢复后可作within-task pretrain | 可作legacy robot diagnostic | 不产生目标硬件/OOD claim |

## 14. 每阶段一页式验收表

| Stage | 输入 | primary output | pass阈值摘要 | stop | 人时 | compute |
|---|---|---|---|---|---:|---:|
| D0 | schema+fixtures+2 source samples | unique compiler/receipt contract | 100% round-trip/fail-closed；207.7497/186.8500 exact | 任一歧义即停UI/model | 4–7 | <2 CPU·h |
| D1 | 3 calibration+12 blind human pour | adjudicated event/node/edge+pose probe | F1≥.80、κ多数字段≥.70、tIoU≥.70、boundary P50≤2 | 最多一版修订；再失败停 | 14.5–20 | 2–6 CPU·h |
| D1-M | D1 gold+natural/synthetic missingness | missing/censor stress | naive-zero bias≥3pp或Brier+10%；censor方法NLL/IBS+10%或F1+3pp | 无差异则只留data hygiene | 3–5起 | 2–8 CPU·h |
| D1-T | fresh 120 frames | non-reconstruction observation gold | rim/boundary κ≥.70；0/1分歧≤10% | stream/contact不可标则保持No-Go | 6–10 | 2–6 CPU·h |
| D2 | 36+36 / 12 cells /3 families | full manifests+events+180 pair gold | 跨域agreement、pair gold门槛 | 无同task overlap/字段覆盖即停 | 50–76 | 20–60 CPU·h |
| D2-R | same D2 events | S1/S3/full frontier | S3 vs single +5pp；距full≤5pp | 不过即删multi-pose sufficiency | 包含于D2，另2–5复核 | CPU probe或未来20–80 GPU·h |
| D2-P | 180 decisions+legacy stress | alignment benchmark | NULL≥.80；CAPOT vs best +5pp+CI；FAR≤5% | 不过则CAPOT降data tool | 包含于D2 | 5–20 CPU·h |
| D3 | powered target robot native truth | oracle representation sufficiency | v2 vs22D +8 success或+10 completion；距full≤3 | 不过则v2 action No-Go | 每100 rollout QA 10–20 | 200–600 H20 GPU·h规划 |
| D4 | expanded train/groups+locked test | causal predicted WAM | ≥70% oracle gain；CAPOT/S3/semantic matched wins | 对应claim逐项删除 | 每100 ep 15–30+capture | 700–2,400 H20 GPU·h规划 |

## 15. 论文叙事边界与最终建议

当前可以写成待验证假设：

> 接触、释放与效果变化定义任务相关事件锚点；事件内 entry/turn/exit 或局部运动保留单帧可能缺失的操作过程；跨 embodiment 只对齐共享功能事件，并允许准备、等待、失败与恢复片段保持 NULL。

当前不能写：

- 人类已被证明只看固定三帧或固定八阶段；
- 透明物体无需重建已经被当前策略结果证明；
- 66 human pour + 88 robot grab构成人机 paired transfer；
- CAPOT首次解决 partial alignment；
- 207.75-bit一定比186.85-bit更“语义”或更小；
- 36+36足够production OOD或目标硬件 policy论文。

最终建议是现在只完成 D0契约与 D1标注，顺带锁定 D1-M/D1-T formal set。不要写 WAM v2代码，不要训练。D1通过后再采36+36；D2的两个关键结果——S3是否胜单帧且接近full、CAPOT是否胜最强partial baseline——决定是否值得进入昂贵的D3，而不是反过来用WAM训练结果替设计辩护。

## 16. 项目代码位置与未来落点（本轮未修改）

- legacy schema/capacity：`world_action_model/contact_transition_schema.py:168,340,417`
- current model contact transport：`world_action_model/models/transformer_wa_casual_mot.py:806`
- current dataset sidecar/receipt：`world_action_model/datasets/wam_lerobot_dataset.py:76-96,741-818`
- current trainer contact route：`world_action_model/trainer/wa_casual_trainer_mot.py:275-303,555-575`

future v2应新建独立 schema/annotation tables，不修改 legacy常量，也不能在同一 checkpoint把22D与v2拼接后归因。本路线图本身不授权这些代码修改。

## 17. 参考工作与证据边界

- GigaWorld Policy / GWP-0.5：当前 joint WAM backbone与action-centered route背景。
- EgoBridge（NeurIPS 2025）、Drop-DTW（NeurIPS 2021）、ImMimic/GMS-SDTW（CoRL 2025）、REALIGN：partial/subsequence/human–robot alignment强前例；CAPOT必须相对它们证伪。
- EgoVLA、XSkill、UniSkill、MOTIF：ego-to-robot、共享skill/motif强前例；人手pose retarget与motif dictionary不是本路线图的独立 novelty。
- Pomp et al. 2021、Bunday et al. 2016、Roberts et al. 2015：仅支持contact边界与动态信息作为候选动机，不证明固定三姿态机制。
- ClearGrasp、ClearPose、DREDS、PourIt!：透明depth、pose、flow已有强路线；本项目选择non-reconstruction必须以传感真值和下游对照证明，而非以“重建难”本身作为贡献。

本轮 Zotero-first 检索同步了100条记录；自动检索返回的通用 benchmark 条目与本题相关性较弱，没有被用来抬高 novelty。有效决策主要来自已存在的SM-FET、CAPOT、missingness、visibility与认知证据 memo，以及对两套本地数据的只读核验。idea memory已更新。
