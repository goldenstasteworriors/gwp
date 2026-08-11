# op41：主因果矩阵生产证据链接纳设计

- 日期：2026-08-11
- 审计基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 隔离分支：`codex/op41-production-matrix-design-20260811`
- 审计输入：Idea A/B/C/D/E 五份 memo、`contact_wam_meta_synthesis_20260811.md`、op38 route-control 设计与当前 gate/config/collector/checkpoint/evaluator
- 本轮范围：只读代码审计与版本化设计；未改模型、production config、gate、数据、GPU、server 或训练状态

## 0. 决策摘要

### 0.1 Go / No-Go

| 项目 | 决定 | 原因 |
|---|---|---|
| 为新主矩阵实现独立、版本化的 production evidence chain | **设计 Go** | op38 已有三条 route 的 CPU 机械实现，缺的是独立 admission、匹配契约和 evaluator 语义。 |
| 直接把新 route 名加入现有 gate v6/v2 或 collector v2 的允许集合 | **No-Go** | 会改变旧 `soft_semantic / hard_st / hard_detached` receipt 的集合语义，并可能让 spec route 被旧 gate 意外承认。 |
| 以现有 hard-ST Ego Stage1 checkpoint 作为三个 Stage2 route 的共同起点 | **No-Go** | semantic 臂在 Stage1 已获得专属 action-route 训练，equal-bit 与 direct 从不等价初始化开始。 |
| 新主矩阵作用于 Robot Stage2 | **Conditional Go** | 必须由一个全九臂共用、route-neutral 的 Ego contact precursor 提供完全相同的初始权重。 |
| 在 Ego Stage1 也跑同一个三臂矩阵，并与 Stage2 结果混为主矩阵 | **No-Go** | 两阶段 treatment 会使 Stage2 初始 checkpoint 按 route 分叉，无法同时声称 Stage2 初始权重 matched。 |
| 当前生成真实 gate、训练 5B、跑真机或声称收益 | **No-Go** | Ego/robot reviewed contact labels、target robot 数据、oracle sufficiency、production groups 与硬件执行链尚未满足。 |

### 0.2 主估计量

新主矩阵只回答：

> 在同一个 route-neutral Ego contact precursor、同一个 Robot Stage2 数据/标签/初始化/优化/噪声/窗口契约下，`aux_only_direct_q12`、`equal_bit_unstructured_hard_st` 与 `hard_st_semantic` 三种 **Robot Stage2 action-visible route** 对接触敏感机器人行为有何差异？

它不回答：

- 三种 route 在 Ego Stage1 端到端预训练时谁更好；
- hard transport 是否优于 soft；
- ST action gradient 是否必要；
- direct route 是否与 finite route 等 bit、等 backward FLOPs；
- packet 是否物理可实现；
- 当前离线 action/contact 指标是否等价于真机成功率。

如果以后要研究“Stage1 与 Stage2 始终使用同一 route”的端到端效应，必须另立
`contact-causal-end-to-end-two-stage-matrix/v1`。它只能要求共同的 **root** 初始化；其 Stage2
parent checkpoint 按 route 分叉，因此不得复用本文主矩阵的 matched-Stage2 claim、gate、receipt
或 aggregate。

## 1. 为什么主矩阵只放在 Robot Stage2

### 1.1 三种可选方案

| 方案 | Stage2 初始权重 | 能回答的问题 | 判定 |
|---|---|---|---|
| 只在 Ego Stage1 跑三臂，Stage2 固定一种 route | Stage2 parent 按 Stage1 route 分叉 | 上游 route 的总迁移效应，但与固定 Stage2 route 交互混杂 | 不作为主矩阵 |
| Stage1、Stage2 都用各自对应 route | Stage2 parent 按 route 分叉 | 整条两阶段 pipeline 的总效应 | 可另立端到端矩阵，不是本文主矩阵 |
| 一个 route-neutral Stage1 precursor，Stage2 才分三臂 | **九臂完全同一 parent bytes** | Robot action-visible route 的条件因果效应 | **本文推荐** |

不能用“同一个 hard-ST Stage1 checkpoint”代替 route-neutral precursor。虽然 checkpoint bytes
对九臂相同，但其中的 action expert、contact encoder 与共享表征已针对 semantic packet 联合优化，
这会把 semantic-specific warm start 偷带进主比较。

### 1.2 route-neutral Ego precursor 的精确定义

建议新增 precursor ID：

```text
ego-contact-planner-precursor-q12/v1
```

它不是第四个 causal route，也不进入 3×3 主表。它的目的仅是从 Ego 数据学习 reference visual
表征与 contact planner，生成一个所有 Stage2 run 共用的冻结起点：

```text
Ego reference/state/text
        ├── visual WAM loss
        └── Q12 contact planner + grouped contact auxiliary loss

action expert / action encoder / action decoder / contact_condition_encoder
        └── 全部冻结；action loss 权重固定为 0
```

必须直接调用 planner/contact-head 训练接口，不能把 `aux_only_direct_q12` 的 direct action forward
跑一遍后再声称 route-neutral。否则 action loss 或 direct reference backward 仍可能改变将要转移的
参数。precursor receipt 必须列出：

- 被训练、冻结和 transfer 的完整 named-parameter 集合及 SHA；
- `action_loss_weight=0`，但 `contact_loss_weight=0.25`；
- 冻结参数从 root checkpoint 到 precursor checkpoint 逐 tensor 不变；
- 预注册的固定保存 step，不允许查看 Stage2 结果后挑 precursor；
- 唯一 checkpoint bundle SHA，九个 Stage2 run 均绑定它。

如果 op42 不愿新增 route-neutral precursor objective，安全的回退是让主矩阵直接从同一个原始
GWP base checkpoint开始；不能退回现有 hard-ST Stage1 checkpoint。

## 2. 各阶段前置条件

### 2.1 Ego precursor（Stage1）

数据与 label 前置：

1. 当前 66-episode Ego source 的 dataset fingerprint、20 Hz frame clock、H=48、T5、HDF5/norm
   与 loader-selected source tree 均通过现有 v6 级别的 byte binding；
2. `contact_segment_v1_1` Q12×22 sidecar 必须来自真实 A/B 独立标注与第三方 adjudication；
3. `unknown / positive_only / exhaustive` 与 field mask/censor 保持现有语义，未标处不得变成 negative；
4. subject/session/operator/object/liquid/batch 等 group provenance 完整，split 无泄漏；
5. 当前空白 A/B/adjudication CSV、proposal、prescreen、tracker proxy 均不能过 gate；
6. `hand_status` 只能表示 `0=open, 1=closed` 的 grip/action proxy；`hand_active` 只能表示该 proxy
   的 tracker validity。二者永久禁止填入 contact presence、contact negative、coverage exhaustive、
   onset/release 或 oracle condition。

硬件前置：

- 不需要 G1、RH56DFTP 或 OmniPicker manifest；Stage1 是 human Ego 数据；
- 不允许把 robot hardware manifest 附到 Ego sidecar，也不允许把 20 Hz Ego label 附到 robot-native
  frame clock；
- precursor 不训练 Ego action，因此 `ego_action_grip_evidence_passed` 不应再作为 v7 的授权条件。
  若为兼容审计而保留 hand-active artifact，只能标为 `record_only_not_contact_truth`。

当前状态：T5 独立 replay、真实 contact labels 与 production groups 均未完成，所以 precursor
training 仍为 **No-Go**。

### 2.2 Robot Stage2 主矩阵

数据与 label 前置：

1. 必须是真实目标 schema `sonic_g1_rh56dftp6_omnipicker1_v1` 的 canonical25D→32D 数据；旧
   dual-Inspire 88 episodes 不能通过重命名或压缩右手 6D 冒充 OmniPicker 1D；
2. 每个 episode 的 dataset/native frame clock、action/state timestamp、H=48 window、group ID、
   train/val/test split 与 train-only norm 完整绑定；
3. Robot 数据自己的 reviewed `contact_segment_v1_1` sidecar、coverage、mask、confidence 与 source
   hashes 完整；不能复用 Ego sidecar；
4. 三条 route 在同一 seed 内使用完全相同的 train episode IDs、ordered windows、sidecar bytes、
   contact targets/masks/confidence、action targets/masks、T5/prompt 与 norm bytes；
5. oracle sufficiency 使用独立 oracle matrix 和真实 reviewed/native label，先于主 5B training；
6. held-out action prediction 前不读取 action target 或 contact target，contact target 只在预测完成后
   进入 metrics。

硬件前置：

- dataset 内嵌 manifest、外部 manifest、calibration capture、nominal rate、action schema 与 norm
  必须一致；
- 数据必须确实来自目标 G1 + 左 RH56DFTP + 右 OmniPicker，且有 command/state mapping、时钟和
  provenance；
- 真实新采集或 rollout 前必须有 driver、canonical-to-device mapper、限位、watchdog、急停与
  bench receipt；离线 CPU route/evaluator 实现本身不需要开设备；
- 当前仓库明确缺 OmniPicker driver/mapper/watchdog/bench 证据，因此 Robot Stage2 production
  training 与物理 evaluation 均为 **No-Go**。

## 3. 旧 mechanism matrix 必须冻结

### 3.1 历史语义

现有 `configs/contact_transport_ablation_matrix.json` 的 schema 是
`contact_transport_ablation_v2`，production runner exact set 是：

```text
soft_semantic / hard_st / hard_detached
```

它只能解释 transport mechanism：continuous vs hard forward，以及 hard-ST vs detached gradient。
不得在该文件中把新三臂状态从 `implemented_cpu_route_unadmitted` 改为 production，也不得扩展
`IMPLEMENTED_RUNNER_VARIANTS` 后继续使用 collector v2 名称。旧 receipt 中的 `hard_st` 保持旧
variant ID；新矩阵中的 `hard_st_semantic` 不作为其 alias 被旧 gate 接受。

### 3.2 推荐 ID 与 schema

| 对象 | 版本/ID | 迁移规则 |
|---|---|---|
| 旧 mechanism matrix | 保留 `contact_transport_ablation_v2` 与旧文件 SHA | 只读冻结；不补写 matrix ID，不重签历史 receipt |
| 新 matrix schema | `contact-causal-production-matrix/v1` | 新文件、新 validator、exact route set |
| 新 matrix ID | `contact-asym25-causal-main-q12-b16-stage2/v1` | 只表示本文 Stage2 条件矩阵 |
| Stage1 precursor | `ego-contact-planner-precursor-q12/v1` | 单一 shared precursor，不是 matrix arm |
| route contract | 复用 op38 `contact_causal_route_contract_v3` | production receipt 绑定完整 canonical JSON 与 SHA |
| equal-bit codec | 复用 `contact_equal_bit_unstructured_v1` | 固定 matrix bytes/rank/SHA，不进入 checkpoint state |
| main route set | exact `aux_only_direct_q12 / equal_bit_unstructured_hard_st / hard_st_semantic` | missing、extra、alias、legacy name 全拒绝 |
| mechanism route set | exact `soft_semantic / hard_st / hard_detached` | 永不由新 gate 放行 |

建议新增：

```text
configs/contact_causal_production_matrix.v1.json
```

不要修改旧 matrix 文件来表达 admission。新 manifest 需绑定上述 ID、route contract receipt、codec
receipt、Stage1 precursor contract、stable/variant-local pointer allowlist、gate/evaluator/checkpoint
schema versions及所有源码 logical paths。

## 4. 三臂的可归因范围

| Route | action 读 reference | action-visible channel | contact aux | action loss→planner | 可声明 |
|---|---:|---|---:|---:|---|
| `aux_only_direct_q12` | 是 | 高带宽 direct visual route | 同权重 | 否 | direct + contact auxiliary baseline |
| `equal_bit_unstructured_hard_st` | 否 | 每 query 精确 48,673 states；Q12 186.850009 bit | 同权重 | ST 是 | 等容量无显式字段 control |
| `hard_st_semantic` | 否 | 每 query 精确 48,673 canonical grammar states；Q12 186.850009 bit | 同权重 | ST 是 | 主 finite semantic route |

### 4.1 direct 的限制必须写进 receipt 与论文

`aux_only_direct_q12` 只匹配：

- Q12 planner/contact decoder 的参数；
- planner/contact decoder forward；
- 相同 grouped contact auxiliary supervision；
- 主干、训练数据与训练预算。

它不匹配：

- action-visible information capacity；direct reference 是高带宽连续视觉输入；
- action sequence/attention FLOPs；direct 无 12 个 action-visible contact token并读取 ref；
- backward FLOPs；direct action loss 不回 planner，却会沿 direct reference path backward；
- active-gradient parameter set或 optimizer state population。

因此名称中不得出现 `equal_capacity`、`equal_bit` 或 `exact_compute_matched`。每个 checkpoint/report
应记录 measured forward FLOPs、backward FLOPs或可复核 proxy、active-gradient parameter count、
peak memory、step time 与 action-only latency；这些是限制报告，不是 gate 可以抹平的差异。

### 4.2 equal-bit 与 semantic 的精确相等项

两条 strict route 必须保持：

- Q=12、width=22、B=16；
- 每 query exactly 48,673 action-visible states；
- Q12 exactly `12*log2(48673)=186.85000894308362` bit；
- 相同 22→action-width `contact_condition_encoder`；
- 相同 trainable parameter names/shapes、strict ref/future adjacency、contact auxiliary loss、
  action/text/state/noisy-action conditions和 token count；
- 相同 ST 可达参数集合。

二者不是 exact kernel/FLOP matched：equal-bit 有固定 `16×22` mixing、threshold/id
canonicalization，semantic 有 field activation/categorical/time canonicalization。必须报告各自 deterministic
transport FLOPs、gradient norm、code utilization、entropy、invalid→zero rate。若 equal-bit collapse，
不能直接把 semantic 优势解释为人类语义优势。

## 5. Stable contract 与 variant-local allowlist

### 5.1 同一 seed 内必须完全相同

新 collector/acceptance 的 stable contract 至少包含：

1. git commit、HEAD-bound training/evaluator/collector/model source bytes；
2. matrix ID/schema、Stage1 precursor checkpoint bundle SHA、root/base weights SHA；
3. 全部 persistent model parameter name/shape/dtype 集合及 step-0 tensor bytes SHA；
4. Robot dataset tree、dataset ID、hardware/capture、split manifest、该 seed 的 train/val/test IDs；
5. sidecar bundle、coverage/segments/inventory/build-config与逐 episode NPZ SHA；
6. contact target/loss mask/confidence、action/state target/mask、prompt/T5 的逐 batch hashes；
7. train-only norm bytes、transform config、每次 sampled augmentation trace；
8. ordered sample/window IDs、visual/action timestep、visual/action noise；
9. dropout/stochastic-depth config与 pre-forward RNG state。若不同 route 因 shape 导致不同 RNG
   消耗，应使用 stateless per-module RNG或统一关闭该 stochastic module，而不是只记录 global seed；
10. optimizer类型/hyperparameters、named parameter-group membership/order、step-0 optimizer state；
11. scheduler类型/config、step-0 state、warmup、max steps、gradient accumulation、clip、precision；
12. action/visual/contact loss定义、reduction、权重、zero-supervision rule；
13. world size、rank set、package/runtime receipt、checkpoint steps 0/100/200；
14. held-out fixed windows、eval noise seeds、dtype、inference steps与center-crop contract。

匹配作用域要精确：同一 seed 的三 route 全部相同；跨 seed 共享原始 dataset/label/source tree、
matrix与初始权重，但 split、train-only norm、window/noise和训练 seed 是显式 seed-derived 字段。当前
`6666/7777/8888` 同时是 split seed 与 training seed，不能宣称三个 seed 之间 train rows/norm相同。

### 5.2 唯一允许变化的字段

新 schema 应使用 exact JSON Pointer allowlist；未来新增字段默认 stable。允许变化仅包括：

```text
/route/causal_route_id
/route/route_contract_sha256
/route/action_attends_reference
/route/action_uses_packet
/route/action_visible_capacity
/route/cache_policy
/route/transport_receipt_sha256
/model_config/contact_causal_route
/model_config/transformer/contact_causal_route
/model_config/transformer/contact_bottleneck
/model_config/transformer/contact_semantic_bottleneck
/run_identity/{variant_id,training_run_id,training_run_nonce,external_acceptance_sha256}
/output_path_record_only
/measured_route_costs/*
/gradient_reachability_receipt/*
/checkpoint_artifact_hashes/*
```

`contact_semantic_transport_mode` 三臂均保持 `hard_st` 作为 planner/codec backward contract，不需要
allowlist。任何 loss weight、dataset、label、norm、optimizer、scheduler、precision、Q/D/B、text
conditioning、initial weight或 training step差异均 fail closed。

### 5.3 route 特有 unused 参数

所有三臂必须实例化相同 persistent parameter key/shape 集，并从同一 precursor bundle加载相同
bytes。equal-bit mixing 是 source-rebuilt、nonpersistent buffer，只能作为 variant-local runtime buffer，
checkpoint 若携带该 key 必须拒绝。

`aux_only_direct_q12` 的 `contact_condition_encoder.*` 是预注册 unused set：

- 参数仍存在、进入同名 optimizer parameter group，但 `grad is None`；
- AdamW 对 `grad is None` 参数不得创建 state、不得 weight decay、不得更新；
- step 0/100/200 的参数 bytes 必须与 step 0 相同；
- optimizer state 缺项只允许精确落在该 named set；
- 任何额外 inactive parameter、非零 grad、state 创建或 byte 漂移均失败。

equal-bit 与 semantic 必须有相同 active parameter set。所有 route 都要求 planner/contact decoder
被 contact auxiliary loss更新；aux-only 中 action loss对 planner为零，strict 两臂则通过 ST 可达。
这些差异进入 gradient reachability receipt，不能放进“隐藏的不匹配”删除列表。

## 6. Label、oracle、aux-only 与 zero-supervision 语义

### 6.1 主矩阵禁止 oracle

三臂都固定：

```text
contact_condition_mode = predicted
contact_condition = absent
contact_condition_mask = absent as model input
contact_target = trainer-loss/metric only
```

`contact_condition_mask` 可作为 sidecar provenance/metric 字段存在于 batch，但不得传给 model。
aux-only 的 planner logits/activated semantics/target也不得进入 action或cache。

### 6.2 oracle 是前置 ceiling，不是第四臂

另立：

```text
oracle-sufficiency-contact-hard-q12-b16/v1
```

它只允许 `hard_st_semantic + reviewed/native oracle merge`，使用独立 gate、run ID、输出目录和
aggregate。oracle result 永远不能出现在主 matrix exact route set或 predicted held-out evaluator。

启动主 5B matrix 前，Robot Stage2 oracle 相对最强 matched direct 至少需要：

- task success +10 percentage points；
- correct-first-contact +15 percentage points；
- spill/collision/safety 不退化。

当前仓库无合格 Robot Stage2 labels/outcomes，oracle gate不可执行。

### 6.3 contact auxiliary loss 完全匹配

三臂使用同一个 grouped loss implementation、`contact_loss_weight=0.25`、target、mask、confidence
和 reduction。`aux_only_direct_q12` 中的 “aux-only” 表示 contact prediction只作为 auxiliary，
不是 `contact_loss=0`。`q12_direct_lambda0` 是另一条 supervision mechanism baseline，不在主三臂。

### 6.4 zero supervision 不能变成 zero label

当前 op38 `contact_grouped_supervision_v1` 会对 batch-level zero eligible item fail closed。production
主矩阵应升级为新语义：

```text
contact_grouped_supervision_v2_zero_safe
```

规则：

1. target/mask/confidence arrays 始终存在且三臂 bytes相同；
2. batch 内 effective eligible count 为 0 时，contact loss返回 canonical `+0`，对 planner零梯度，
   action/visual loss照常训练；
3. 不允许将全零 mask解释为所有 query absent，也不允许把 unknown补成negative；
4. collector记录每 batch各 field eligible count、weighted denominator与 zero-supervision flag；
5. preflight在 dataset/split层要求非零总监督，并预注册最低 event/field/phase coverage；所有三臂看到
   相同的 zero-supervision batch位置；
6. 若决定过滤 zero-supervision window，过滤必须发生在冻结 window manifest生成时，三臂共用同一
   manifest；不能由各 trainer 动态丢 batch。

旧 `contact_grouped_supervision_v1` 与历史 receipt语义不改；新生产 chain显式绑定 v2。

## 7. Gate 与迁移策略

### 7.1 新版本，不原地扩容

| 层 | 旧版本，保持原义 | 新版本 |
|---|---|---|
| Ego gate | `ego-discrete-contact-stage1-training-gate/v6` | `.../v7`，只授权 route-neutral precursor |
| Robot gate | `contact-asym25-training-gate/v2` | `.../v3`，绑定 causal matrix ID与exact route set |
| Gate reviewer acceptance | `contact-asym25-gate-reviewer-acceptance/v1` | `.../v2` |
| Matched collector | `matched-control-hash-collector/v2` | `.../v3` |
| Robot preflight receipt | `contact-asym25-production-preflight-receipt/v1` | `.../v2` |
| Training-data receipt | `contact-asym25-training-data-receipt/v1` | `.../v2` |
| Grouped supervision | `contact_grouped_supervision_v1` | `contact_grouped_supervision_v2_zero_safe` |
| Checkpoint model bundle | `contact-asym25-checkpoint-bundle/v3` | **保持 v3**，纯文件布局未变 |
| Checkpoint training receipt | `.../v4` | `.../v5` |
| Independent checkpoint acceptance | `.../v2` | `.../v3` |
| Held-out checkpoint manifest | `.../v2` | `.../v3` |
| Held-out evaluator/result | evaluator v1 / result v2 | evaluator v2 / result v3 |
| Three-seed aggregate | `.../v4` | `.../v5` |
| Final matrix report | `contact-asym25-production-matrix/v1` | `contact-asym25-causal-production-matrix/v2` |

bundle manifest v3可以保持，因为它只定义 `config.json + one complete weight file` 的字节布局；
route/matrix语义由 training receipt v5与checkpoint manifest v3承载。若 op42改变bundle布局或加入
optimizer文件，才必须升bundle v4。

### 7.2 false/null template 继续 fail closed

- 现有四份 template原样保留，不把 `gate_version`、false或null改成新示例；
- 新增 v7/v3/v2 templates，所有授权 flag为false、所有证据/hash/matrix ID为null；
- new validator要求 exact key set、exact version、exact matrix ID、exact route set；
- old v6/v2 validator继续引用旧 `IMPLEMENTED_RUNNER_VARIANTS`，遇到
  `hard_st_semantic`、equal-bit、aux-only或`contact_causal_route`字段直接失败；
- new production config不得从 old config import后仅改 env；它必须先用新 bootstrap/preflight，
  再构造route-aware config；
- `configs/giga_world_policy_0_5_sonic_g1_contact_causal_controls.py` 保持
  `implemented_cpu_route_unadmitted`，不得被任何 gate path直接 import为production；
- production gate必须绑定 op38 CPU audit report SHA，但 `production_admitted=false` 的 op38报告
  本身不能授权训练；真正授权来自v3 gate与外部 reviewer acceptance。

### 7.3 迁移顺序

1. 冻结 old matrix/gates/collector/checkpoint/evaluator golden tests；
2. 新增 matrix manifest和版本常量，所有新模板仍false/null；
3. 实现新 collector/config canonicalizer与CPU negative tests；
4. 实现v7 precursor gate/config，但保持无真实gate可过；
5. 实现v3 Robot preflight/config/checkpoint chain；
6. 实现route-aware evaluator与27-checkpoint aggregate；
7. 完成真人标签、group/hardware证据和oracle；
8. 外部 reviewer填充新gate；最后才允许production training。

## 8. 精确代码改动面

以下位置基于审计 HEAD；op42应以函数/常量名为主，行号为辅助。

### 8.1 Matrix、route 与 supervision

| 位置 | 当前事实 | op42改动 |
|---|---|---|
| `configs/contact_transport_ablation_matrix.json` | 旧 mechanism exact set，op38 route仍unadmitted | **不改 admission语义**；只做golden SHA测试 |
| 新增 `configs/contact_causal_production_matrix.v1.json` | 不存在 | 新主matrix manifest、stable/variant allowlist、schema/version/source paths |
| `world_action_model/contact_route_contract.py:32-306` | op38 enum、resolve、metadata均已存在；trainer validator明确spec-only | 增加独立production admission metadata/validator；不改变legacy mapping |
| `world_action_model/contact_supervision_contract.py:16,71-101` | v1要求每batch至少一个eligible item | 保留v1；新增v2 zero-safe reduction与dataset-level coverage receipt |
| `world_action_model/contact_route_audit.py` / `scripts/audit_contact_route_contract.py` | CPU route audit，明确non-admission | 加step-0 parameter equality、unused-param/optimizer-state、new registry negative tests；仍不声称收益 |

### 8.2 Matched-control hashes/config

| 位置 | 当前硬编码 | op42改动 |
|---|---|---|
| `world_action_model/matched_control_hashes.py:26-41` | collector v2与旧三variant exact set | 保留旧常量；增加v3 namespace或单独新模块，exact新route set |
| `...:289-643` `MatchedControlCollector` | transport override只认识old三臂 | 新建causal collector class，不放宽旧class |
| `...:839-1259` `compare_variant_traces` | 比较集合必须等于旧三臂 | 新建`compare_causal_matrix_traces`，要求new matrix ID、stable contract与allowlist |
| `world_action_model/matched_control_config.py:23-57` | hash scope与runtime key删除表面向old transport | 新增v3 canonicalizer；仅替换manifest列出的exact pointers |
| `...:112-284` | external expectations强制old variant set | 新增causal evidence validator |
| `...:312-576` | route→transport只允许soft/hard/detached | 新增`configure_causal_matrix_from_environment`，禁止old gate调用 |
| `scripts/compare_matched_control_hashes.py` | old collector CLI | 保持old CLI；新增`compare_contact_causal_matrix_hashes.py`或显式versioned subcommand |
| `configs/matched_control_hash_collector.template.json` | v2 false template | 原样保留；新增v3 causal template |

v3 report必须新增：initial parameter tensor receipt、optimizer group/name/state receipt、scheduler
initial state、explicit contact supervision hash/count、pre-forward RNG hash、Stage1 precursor SHA、matrix
manifest SHA、route receipt SHA、active/inactive parameter set与measured route cost。

### 8.3 Ego precursor gate/config

| 位置 | 当前硬编码 | op42改动 |
|---|---|---|
| `world_action_model/ego_stage1_gate_bootstrap.py:18-58,179-278` | v6，要求hard-ST Stage1与grip evidence | 保留v6入口；新增v7 exact bootstrap，不接受old/new互换 |
| `world_action_model/contact_ego_stage1_preflight.py:962-1127,1719-1924` | v6 matched collector、receipt返回hard_st | 新增precursor validator/receipt，绑定frozen/trainable parameter sets与objective |
| `configs/ego_stage1_training_gate.template.json` | v6 false/null | 原样保留；新增`ego_stage1_causal_precursor_gate.v7.template.json` |
| `configs/giga_world_policy_0_5_ego_discrete_contact_pretrain.py:1-204` | production hard-ST action route | 不改其v6语义；新增`..._ego_contact_causal_precursor.py` |
| `docs/ego_discrete_contact_stage1.md` | 描述v6 hard-ST | 加v7 precursor章节，明确hand-active非contact GT |

precursor config必须在optimizer构造前冻结全部action相关named parameters，并以独立loss contract证明
action loss为0；不能依赖“没有action label”这种偶然状态。

### 8.4 Robot gate、preflight与production config

| 位置 | 当前硬编码 | op42改动 |
|---|---|---|
| `world_action_model/contact_asym25_preflight.py:80-124` | gate v2、preflight v1、acceptance v1、old flags | 旧常量保持；增加v3/v2 dispatch或新模块 |
| `...:785-923` | matched expectation keys/variant set是old三臂 | 新causal expectation schema、matrix/precursor/stable-allowlist hashes |
| `...:932-1536` | gate exact schema、reviewer acceptance | 新v3 exact validator；old v2见新route立即失败 |
| `...:1856-2222` | artifact、split、train-only norm与receipt | 复用验证；v2 receipt新增precursor/matrix/route与label coverage契约 |
| `configs/contact_asym25_training_gate.template.json` | v2 false/null | 原样保留；新增v3 causal template |
| `configs/contact_asym25_gate_reviewer_acceptance.template.json` | acceptance v1 | 原样保留；新增acceptance v2 template |
| `configs/giga_world_policy_0_5_sonic_g1_asymmetric25_contact_finetune.py:50-488` | old variant map、hardcoded action-visible semantic说明 | 原样保留old mechanism production；新增`..._causal_matrix_finetune.py` |
| `scripts/preflight_contact_asym25_finetune.py` | v2入口 | 增加显式`--contract causal-v3`或独立CLI；不自动探测 |
| `world_action_model/production_split_contract.py` | dataset/split/action schema receipt | v2 receipt增加matrix/precursor binding；split数学语义不改 |

新config只接受：

```text
GWP_CONTACT_CAUSAL_MATRIX_ID=contact-asym25-causal-main-q12-b16-stage2/v1
GWP_CONTACT_CAUSAL_ROUTE=<exact one of three>
```

`GWP_MATCHED_CONTROL_VARIANT_ID` 若保留，必须与causal route逐字相同。禁止legacy `hard_st` alias。

### 8.5 Trainer、route resolution 与 checkpoint receipt

| 位置 | 当前硬编码 | op42改动 |
|---|---|---|
| `world_action_model/trainer/wa_casual_trainer_mot.py:280-336` | 已解析spec route，production字段仍叫transport variant | 新增`production_matrix_id`、`production_causal_route_id`与receipt |
| `...:386-407` | production只接受old三transport | 依据明确receipt version分派；old set不扩展 |
| `...:560-589` | spec route要求positive batch supervision | production v2 zero-safe；old spec行为保持 |
| `...:672-790` | 生成training receipt v4与transport ID | 生成v5 causal receipt、parameter/optimizer/scheduler/precursor/route receipts |
| `...:960-1000` | contact/oracle与loss | 三主route都拒绝model-side oracle，zero-mask返回+0并记录计数 |
| `world_action_model/models/transformer_wa_casual_mot.py:715-866` | route resolve/模块构造已实现 | 保持persistent state key一致；暴露可审计named active/inactive set |
| `...:1320-1425` | aux/equal/semantic planner route已实现 | production调用同一explicit enum；不得从legacy booleans推断new admission |
| `...:2164-2590` | action-only cache route-aware | 返回route-specific packet kind与receipt，direct不得伪造semantic packet |

### 8.6 Checkpoint acceptance

| 位置 | 当前硬编码 | op42改动 |
|---|---|---|
| `world_action_model/contact_asym25_checkpoint_acceptance.py:21-112` | receipt v4、acceptance v2、cross-seed allowlist | 新v5/v3 schema与exact pointer lists；old validator保留 |
| `...:346-563` | transport ID只允许old三臂 | causal receipt validator要求matrix ID与new route set |
| `...:565-702` | current-run report replayold collector | 分派v3 causal replay，验证stable/allowlist与step-0 weights |
| `scripts/review_contact_asym25_checkpoint.py` | 只写acceptance v2 | 加显式schema/matrix选择；默认不能猜版本 |

Checkpoint可比性规则：

1. step 0 必须发生在optimizer step前；三route的 **weight tensor digest** 完全相同；
2. `config.json` 因route不同而bundle hash不同是允许的，不能拿bundle hash代替weights-equal证明；
3. 分别记录 model parameter digest、buffer digest、config digest与whole bundle digest；
4. optimizer在step 0验证same named groups/hyperparameters；unused param state缺失按精确allowlist；
5. scheduler step/state完全相同；
6. step100/200只比较stable receipt，不要求trained weight相同；
7. parent chain在每个run内为base→0→100→200；九个step0都指向同一precursor/base tensor digest；
8. step200是预注册primary，step0是初始化sanity，step100是trajectory diagnostic；禁止按held-out挑best step。

### 8.7 Held-out evaluator与3 seed×3 route×3 step聚合

| 位置 | 当前硬编码 | op42改动 |
|---|---|---|
| `world_action_model/contact_asym25_eval_runtime.py:221-271` | checkpoint固定hard_st config | 从v5 receipt读取exact causal route，再验证model config |
| `...:274-372` | 强制输出`contact_hard_semantic` | 分开`semantic_aux_prediction`与`action_visible_packet`；aux packet为none |
| `world_action_model/contact_asym25_heldout_eval.py:68-84` | evaluator/result/aggregate old版本 | 增加v2/v3/v5常量，old读取不变 |
| `...:681-971` | checkpoint manifest v2 | v3绑定matrix、route、precursor与v5 receipt |
| `...:1115-1657` | contact exact指标假设action-visible hard semantics | 对三route统一从semantic aux logits生成field metrics；packet metrics按kind分开 |
| `...:1658-2121` | result bundle固定old contract | result v3绑定route与packet kind，禁止oracle |
| `...:2192-2393` | 3-seed与same-seed old transport checks | 新stable/allowlist比较，验证step0 tensor equality |
| `...:2394-2492` | exact old 3×3 matrix | 新exact 3 seed×3 causal route×3 checkpoint链 |
| `scripts/evaluate_contact_asym25_heldout.py:35-113,156-704` | CLI文案/keys均为transport | 新versioned subcommands或`--matrix-contract causal-v1`，不自动兼容 |
| `docs/contact_asym25_heldout_evaluator.md` | old mechanism命令 | 单独新增causal matrix命令章节，不改旧示例含义 |

Evaluator必须区分：

- 三臂的 contact field F1/Brier/NLL：都从相同semantic auxiliary decoder logits得到；
- semantic action-visible packet：可评grammar validity与field accuracy；
- equal-bit action-visible packet：只评codec validity、utilization、entropy、invalid-zero rate，不能当语义字段；
- aux-only：`packet_kind=none_direct_reference`，不得填一个假的hard semantic packet；
- action metrics：三臂完全相同的action target/mask/denorm规则；
- 主结果：27份独立checkpoint acceptance，即9个training run×step 0/100/200。

## 9. CPU tiny 与 negative tests

### 9.1 无真人标签即可完成

所有测试只用项目内synthetic tiny tensors，不创建一次性脚本；加入正式test/module并在测试后不留
临时文件：

1. old v6/v2/collector-v2/evaluator-v1 golden：旧route exact set与receipt canonical bytes不变；
2. old gate收到任一new route、matrix ID或extra key必须拒绝；new gate收到old route也拒绝；
3. false/null old/new templates全部fail closed，null不能由默认值补齐；
4. op38 capacity：semantic/equal均48,673 states/query与186.85000894308362 bits/Q12；
5. aux direct对planner hidden/logits/semantic/target替换action不变，action loss对planner为零；
6. strict两臂fixed packet/code后替换reference，action `max_abs<=1e-6`；
7. equal/semantic persistent parameter key/shape集合相同；aux只允许精确inactive set；
8. 三route从同一tiny checkpoint加载后step-0 parameter tensor hash相同；
9. optimizer group/name/hash相同；aux unused params无state/无更新，其他额外缺state失败；
10. scheduler初始state相同；
11. zero-mask batch三臂contact loss都是canonical +0且planner零梯度；unknown不变negative；
12. 非zero mask下三臂grouped contact loss逐bit/容差相同；
13. collector v3只允许exact new set，任何stable字段变化失败，allowlist以外新字段默认stable；
14. route-specific evaluator输出schema：aux none、equal code、semantic grammar；禁止用GT作model input；
15. cached/uncached、same-noise identity、step0/100/200 parent chain、27 acceptance唯一性；
16. `git diff --check`、config静态import negative path，不触发GPU/server/training。

### 9.2 必须等真人标签/外部状态

- 填写任一v7/v3 true gate；
- Stage1 precursor真实训练；
- dataset-level contact coverage与agreement gate；
- oracle sufficiency；
- Robot Stage2 matched dry-run与5B主训练；
- contact F1/timing、OOD、task success、spill/collision统计；
- target robot rollout与部署。

Robot Stage2还必须等待真实target hardware dataset、manifest/capture与安全执行链。不能用synthetic
tiny、hand-active、旧dual-Inspire或未来视频prediction替代。

## 10. 停止门槛

### 10.1 实现阶段立即停止

出现任一项，op42不得继续接 production：

1. 必须修改old variant集合或old receipt语义才能支持new matrix；
2. old gate能够接受new spec route或new gate能够fallback到legacy alias；
3. equal-bit与semantic容量不再exact相等；
4. 三route step-0 persistent parameter tensor hash不相同；
5. stable config只能靠删除大段parent object才能匹配，而非exact pointer allowlist；
6. aux direct的planner output或GT影响action；strict route仍读reference/image condition；
7. aux unused set之外出现inactive/optimizer-state缺失参数；
8. zero supervision被编码成absence negative；
9. evaluator必须把equal code解释成contact semantics或给aux伪造packet；
10. new route破坏op38/legacy CPU golden或需要GPU/真人label才能证明代码正确。

### 10.2 科学阶段停止

沿用meta-synthesis主门槛：

- 标签协议修订一次后仍未达到coverage κ≥0.70、event F1≥0.80、median tIoU≥0.70、boundary
  MAE≤2 frames、categorical κ≥0.60：停止human-auditable claim；
- oracle未达到success +10pp、correct-first-contact +15pp且安全不退化：不启动主5B matrix；
- predicted hard未恢复oracle gain的70%：停止predicted-contact闭环claim；
- hard semantic相对aux direct的预注册OOD success优势<5pp或95% cluster CI跨0：停止route benefit claim；
- hard semantic与equal-bit相当：删除“显式contact semantics优于一般有限压缩”claim；
- packet intervention无side/phase局部效应：停止causal-use claim；
- direct/equal/semantic的compute差异未报告：不得称fully compute-matched；
- 硬件、安全或latency退化：停止deployment。

## 11. op42 精确任务书

### 任务名

```text
op42 — Versioned causal-main production admission plumbing and CPU acceptance
```

### 目标

在不训练、不使用GPU/server、不改硬件、不填写真实授权gate的前提下，为
`aux_only_direct_q12 / equal_bit_unstructured_hard_st / hard_st_semantic` 建立独立、fail-closed、
可CPU复核的Stage2 production evidence chain，并保持旧mechanism chain逐字节/逐语义兼容。

### 必做交付

1. 新增 `contact-causal-production-matrix/v1` manifest与false/null templates；
2. 新增Stage1 precursor v7 contract/config的结构，但不运行真实precursor；
3. 新增Robot gate v3、reviewer acceptance v2、preflight/data receipt v2 validator；
4. 新增collector v3与causal config canonicalizer，扩充label/init/optimizer/scheduler/RNG receipts；
5. 新production config显式选择exact causal route，不从spec-only config取得admission；
6. trainer加入v5 checkpoint receipt、v2 zero-safe supervision与inactive-param receipt；
7. checkpoint acceptance v3验证step-0 tensor equality、optimizer state allowlist与current-run v3 traces；
8. evaluator v2支持none/code/grammar三种packet kind，result v3、aggregate v5与exact 27-checkpoint matrix；
9. 所有old schema走原validator；old template、oldroute set、oldCLI语义与golden行为不变；
10. CPU tiny/negative tests覆盖第9节全部项目，输出只能写显式项目路径，不能写`/tmp`；
11. 更新 `docs/contact_causal_route_controls.md`、`docs/matched_control_hash_collector.md`、
    `docs/ego_discrete_contact_stage1.md`、`docs/contact_asym25_integration.md`、
    `docs/contact_asym25_heldout_evaluator.md` 与可复制命令；
12. 结论必须仍为 `production_data_admission=false`、`allow_5b_training=false`、
    `allow_server_training=false`、`benefit_claimed=false`。

### op42 明确不做

- 不填写true gate、不生成真人label、不改annotation CSV；
- 不跑Stage1/Stage2训练、不跑oracle、不加载5B；
- 不碰GPU、server、CUDA/driver、Isaac或真机；
- 不改hardware manifest/calibration；
- 不把hand-active当contact GT；
- 不实现Idea B topology、Idea C dense visibility或Idea E `F_R`；
- 不修改old matrix的production exact set。

### op42 验收

只有以下全部通过才是 **implementation Go / scientific training仍No-Go**：

- old和new schema双向拒绝错误route；false/null模板fail closed；
- op38 route audit及legacygolden全部通过；
- new collector exact三route、stable/allowlist与所有新增hash检查通过；
- step-0 weights、optimizer groups/state、scheduler state按本文规则匹配；
- inactive-param/zero-supervision/oracle/GT leakage negative tests通过；
- route-aware evaluator与27-checkpoint aggregate synthetic tiny通过；
- `git diff --check`与CPU测试通过，无临时测试脚本、无GPU初始化、无production授权。

## 12. 最终结论

**Go：** 可以进入op42，实现一个与旧mechanism chain完全隔离的新版本化production plumbing。

**No-Go：** 当前不能生成真实production admission，更不能训练或声称新主矩阵有效。主矩阵应只在
Robot Stage2分三臂；Ego Stage1只提供一个route-neutral、全九臂共用的contact-planner precursor。
任何把现有hard-ST Stage1当“共同且无偏”的起点、把direct称为equal-bit/equal-backward-FLOPs、
把aux-only称为zero-contact-supervision、把hand-active称为contact GT，或让old gate自动接纳new
route的实现，都应在训练前停止。
