# op38：Contact causal-route controls（spec-only）

## 边界

本变更只实现 CPU 可审计的因果路由控制，不接入 production runner admission，
不改变 production gate v2、三 seed 矩阵、标签、硬件、模拟器或训练许可。现有
production runner 的精确集合仍是 `soft_semantic / hard_st / hard_detached`。
CPU audit 只证明信息流和 codec 机械性质，不证明任务收益、物理可行性或真机安全。

## 唯一路由枚举

`world_action_model/contact_route_contract.py` 定义唯一枚举和 canonical JSON/SHA-256
receipt。显式路由不允许通过多个 boolean 拼装：

- `aux_only_direct_q12`：保留与 semantic route 相同的 Q=12 query、planner 参数、
  planner forward 和 22D contact auxiliary logits/loss；planner hidden、raw logits、
  activated semantics 和 target 均不进入 action。action prefix 中没有 contact token，
  action 使用原始 direct reference attention。它匹配 planner compute，不匹配
  action-visible bit capacity，也不声称精确匹配 backward FLOPs。
- `equal_bit_unstructured_hard_st`：严格视觉路由。action 只能读取 canonical equal-bit
  packet、state、instruction text 和 noisy action，不能读取 raw logits、planner hidden、
  GT/oracle、direct reference 或 future tokens。
- `hard_st_semantic`：显式映射已有 `contact_semantic_transport_mode="hard_st"` 路由。
  未显式设置 route enum 的 legacy `soft / hard_st / hard_detached` 仍走原计算，并由
  CPU audit 检查完整 action/contact 输出逐 bit 相同、state_dict key 集相同。

## Equal-bit v1 codec

codec 常量和纯函数位于 `world_action_model/contact_transition_schema.py`：

- codec version：`contact_equal_bit_unstructured_v1`；
- 固定 mixing：row-major `int8` Rademacher `16×22`，row rank 精确为 16；
- matrix SHA-256：`9de3c743cbd3b5ccfda9df21ea6b3d430560343aad9842d862c0e2f5b00f009f`；
- metadata 包含完整 matrix、row-major bytes hex、bit order、shape、dtype、rank 和 SHA；
- bit order：`little_endian_lsb_first`，第 0 维是 ID 的最低有效位；
- raw ID `0..48672` 合法，`48673..65535` 全部严格 canonicalize 到 ID 0；
- action-visible 22D code 的前 16 维是 canonical hard bits，后 6 维是 IEEE-754 `+0`；
- 每 query forward alphabet 精确为 48,673；Q12 容量为
  `12 * log2(48673) = 186.85000894308362 bit`，与 B16 semantic hard route 完全相同；
- 22 raw planner logits 经固定 mixing 和 sigmoid 形成 soft surrogate；forward 使用
  canonical hard code，backward 使用 hard-ST。固定 matrix 是 equal-bit route 独有的
  nonpersistent registered buffer；每次使用前都从源码常量重建并检查 shape、有限值、
  原始 int8 bytes SHA 与 full row rank。checkpoint 不能携带/替换该 key，strict load 会拒绝；
  legacy routes 的 state_dict key 集不变。

模型仍用原有 `contact_condition_encoder` 完成相同的 22→action-width 投影。raw contact
logits只作为 grouped contact auxiliary prediction和 ST 源，不直接进入 action。

## Fail-closed oracle、mask 和 cache

- `aux_only_direct_q12` 和 `equal_bit_unstructured_hard_st` 默认拒绝任何
  `contact_condition`/`contact_condition_mask`；trainer 只允许 `contact_target` 进入
  grouped auxiliary loss，不允许它成为 model condition。
- semantic masked oracle 保持 legacy field merge；mask 或 oracle 出现 NaN/Inf 会拒绝，
  单独提供 mask 也会拒绝。
- strict semantic/equal-bit route 收到非空 `encoder_hidden_states_image` 会拒绝。
- direct cache保留 direct reference KV，但不保存 planner hidden、logits、semantics 或
  code；equal-bit cache只保存 canonical hard code、packet receipt 和由 state+code 生成的
  prefix KV，不保存 raw logits、semantic 或 reference KV。
- equal-bit cache 检查精确 `[B,Q,22]` shape、与 encoder 相同的 floating dtype/device、
  finite、前 16 维严格 `{0,1}`、后六维严格 IEEE-754 `+0`、little-endian raw ID `<48673`。
- route receipt 声明 finite semantic packet 的所有 resolved `hard_st/hard_detached` route
  都直接验证同一完整 B16 hard grammar，包括 implicit/explicit hard-ST 和 hard-detached：
  canonical absent、presence、side/site/role/mode one-hot、tau lattice、`start<=end`、
  censor/tau 一致和所有零为 `+0`。不把 packet 当 logits 重新 transport；每 query 接受域
  精确为 48,673 个状态。非 B16 finite hard cache fail closed，但普通 legacy forward 不变。
- soft semantic receipt 明确不声明 finite grammar，连续 `0.25` 等有限 packet 仍可缓存；
  direct 无 packet，equal-bit 继续使用独立 codec validator。
- packet receipt 从已验证的 canonical packet bytes、dtype/shape 和重新生成的 route contract
  计算，并与 model-held receipt 比较；同时篡改 packet 和 cache 内 receipt 也会拒绝。
- trainer 要求 top-level route 与 transformer resolved route 均显式且一致；三条 spec route
  的 grouped contact loss weight 必须 finite 且 `>0`，target、显式 mask 和真实正的有效
  supervision count 都必须存在。loss 与 validator 共享 `contact_grouped_supervision_v1`：
  presence、两个 censor 和两个 tau 分别计项，side/site/role/mode 仅在整组 mask 精确全 1
  时各计一项，并跨全部 B/T/Q 前导维求和。全零或类别组仅开一维、NaN weight 均
  fail closed；legacy missingness 不变。

## CPU audit 与 same-noise harness

可复用实现位于 `world_action_model/contact_route_audit.py`，CLI 位于
`scripts/audit_contact_route_contract.py`。调用者必须提供项目盘上的显式输出路径；CLI
不会创建父目录、不会覆盖文件，也不写 `/tmp`。命令需要按下面显示的位置换行：

```bash
CUDA_VISIBLE_DEVICES='' conda run -n dit4dit \
  python scripts/audit_contact_route_contract.py \
  --output /ABSOLUTE/PROJECT/PATH/contact_route_audit.json
```

audit 覆盖：

1. fixed matrix rank/bytes SHA；65,536 raw IDs 的模型 codec 与纯 ID codec 完整一致；
2. 48,673 个合法 ID 全量 roundtrip、所有非法 ID→0、unique forward byte states 精确
   48,673、末六维 `+0`、NaN/Inf 拒绝和有限非零 ST 梯度；
3. direct action 在 planner 参数扰动下逐 bit 不变，action loss 对 planner 无非零梯度；
4. fixed packet/state/text/noisy action/timestep 下，strict route 更换 reference 或 future
   的 action 差异不超过 `1e-6`；只更换 packet 时记录 action route sensitivity；
5. direct/strict attention adjacency、strict image condition、oracle/mask、cache key/KV
   长度和 cached/uncached `<=1e-6`；
6. `encode_action_tokens` 的 AST/参数没有 raw planner hidden/logits/reference/future route；
7. legacy soft/hard-ST/hard-detached transport、action/contact forward bytes和 state_dict
   key 集兼容；
8. matrix checkpoint/runtime 替换、implicit/explicit hard/detached cache grammar/receipt、
   soft 连续 cache、trainer route mismatch、全零/partial-category mask、NaN weight 等负例；
9. config receipt、每个 route receipt和完整 report使用 canonical JSON SHA-256，未使用
   Python `hash()`；顶层 `passed` 由全部命名 invariant 归约，不是固定常量；
10. production runner exact set 和 gate protected paths 相对 baseline 为零 diff。报告明确
    `production_admitted=false`，audit 通过不构成 admission。

same-noise intervention 结果只标为 `route_sensitivity_only_not_physical_feasibility`；没有
native simulator/真实传感器 oracle 时，不得把它解释为物理正确、可执行或成功。

## Spec-only 配置与机器可读矩阵

`configs/giga_world_policy_0_5_sonic_g1_contact_causal_controls.py` 只接受显式
`GWP_CONTACT_CAUSAL_ROUTE`，且仅允许主三臂：
`aux_only_direct_q12 / equal_bit_unstructured_hard_st / hard_st_semantic`。它不调用
matched-control collector，不开放 5B/server training。

`configs/contact_transport_ablation_matrix.json` 记录 route/codec/hash 契约，并明确新路由
状态为 `implemented_cpu_route_unadmitted`；production runner exact set保持不变。

## op42 production namespace

op38 remains spec-only and byte/behavior compatible. op42 does not change this
route registry or its CLI. It adds a separate
`contact-causal-production-matrix/v1` admission namespace with exact routes
`aux_only_direct_q12 / equal_bit_unstructured_hard_st / hard_st_semantic`.
The v1 matrix, v2 zero-safe supervision, v3 Robot gate/collector, v5 training
receipt, and CPU acceptance are documented in
`docs/contact_causal_production_matrix.md`. A passing op38 audit remains
non-admission evidence and cannot substitute for a Robot v3 gate.
