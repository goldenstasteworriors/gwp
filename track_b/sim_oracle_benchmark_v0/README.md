# Track B：仿真 / Oracle benchmark readiness v0

## 结论

截至 2026-08-11，本 benchmark 是 **spec-ready、execution No-Go**。本目录可以确定性生成并校验场景设计，也可以在未来对真实 simulator 输出做成对指标核验；它没有启动过仿真、没有使用 GPU、没有生成 rollout，也没有提供任何透明液体重建结果。

当前最接近的候选是 Isaac Sim 5.1：官方 API 有 PBD particle material、viscosity、原生 contact report 和 RGB/depth/segmentation 通道。但官方粒子文档明确说明 particle simulation 不支持 CPU，必须启用 GPU；本次审计明确禁止 GPU 和启动仿真。更重要的是，本机没有准确的 `G1 + 左 RH56DFTP + 右 OmniPicker` 仿真装配，也没有能导出 rim flux、contained volume、outside volume 与 mass residual 的项目适配器。因此不能把“引擎有 fluid feature”偷换成“本任务已有可信 oracle”。

## 本机静态审计

审计证据保存在 `local_readiness_20260811.json`，只使用 filesystem、git revision 和 Python distribution metadata；没有 import simulator runtime。

| 对象 | 发现 | 判定 |
|---|---|---|
| Isaac Sim | `/home/ykj/miniconda3/envs/isaacsim5.1`，distribution `5.1.0.0` | 有条件候选；液体要求 GPU，本轮不运行 |
| MuJoCo | base distribution `3.5.0` | 可做刚体/contact oracle；没有已审计的黏度自由液面/flux/volume oracle，不能承担本 benchmark |
| Genesis World | 本次 package/conda inventory 未发现本地安装 | 官方有 SPH/MPM/PBD，但本机不可评估；不得因此安装 |
| G1+Inspire USD | `g1_29dof_with_inspire_rev_1_0.usd` | 右手仍是 Inspire，embodiment 不符，拒绝冒充 OmniPicker |
| 左 Inspire URDF | `inspire_hand_left.urdf` | 只有几何/关节候选；未发现 tactile cell 定义，也没有异构整机标定 |
| OmniPicker | 找到 `omnipicker.cpp` 驱动 | 驱动不是几何资产；缺 mesh/collision/inertia/joint/wrist transform |
| Beaker USD | 一个 `Beaker_01.usd` | 单一刚体候选；未证明内壁 watertight、容量、rim semantic 或透明/不透明成对资产 |
| LabUtopia PourController | 腕关节倾倒 state machine | 没有 particle、flux、viscosity、volume、spill 或 mass truth；不能作为液体 oracle |

涉及的外部文件只记录绝对路径、大小和 SHA-256，没有复制大资产，也没有修改其它项目。

## Benchmark 设计

`benchmark_spec.json` 将旧的 `../visibility_audit_v0/sim_scenario_spec.json` 收紧为可执行契约：

- 32 个 sanity records、40 个 ID conditions × 4 physics seeds、20 个 OOD conditions × 4 physics seeds，共 272 个 scenario records；
- 透明/不透明 source 和 target vessel、液体透明度/着色、黏度、fill、rim offset、倾角、双手延迟、光照、背景、相机与 role swap；
- 左 RH56DFTP 6 通道、右 OmniPicker aperture 1 通道，统一到已有 25-D action schema；
- native rim/boundary/liquid surface/stream/contact/transition/volume/impulse/RGB/depth/instance GT；
- `rgb_only`、`rgb_depth`、`no_contact_aux`、`direct_action`、`oracle_contact`、`predicted_contact`、`oracle_all_fields` 七个严格对照，总完整矩阵为 1,904 个 planned rollout slots；这些只是待执行配对，不是已有 rollout；
- primary gate 是 OOD task success 与 correct-first-contact；另有 mass conservation、相同 condition 的四 seed repeatability、predicted recovery 和 oracle-field value gate。

场景分配是 deterministic balanced cyclic assignment。它会报告 marginal coverage，但不声称自己是最优 covering array。

## 真值边界

所有真值必须来自同一个 immutable native simulator state/timeline：

- contact points、normal、impulse 来自 native physics contact；
- liquid surface、stream occupancy、rim flux、container volume 和 outside volume 来自 native liquid state；
- boundary/rim 来自 native geometry 的同相机投影；
- RGB、depth 和 instance segmentation 来自同状态的 native render pass；
- camera/state 最大偏差为 5 ms。

verifier 会递归拒绝任何含以下来源词的 record：`inpaint`、`composite`、`diffusion`、`image_edit`、`manual_paint`、`pseudo_label`、`reconstruction_as_gt`。透明/折射只允许影响 native RGB；标签不能从修补过的 observation 产生。Simulator native depth 只能叫 oracle sensor baseline，不能声称真实透明深度有效。

## 执行门槛

`validate_sim_oracle_manifest.py --mode execution` 是 fail-closed 的。除了 `status=execution_ready`，还必须同时提供：

1. 精确 engine/version/adapter 路径与 SHA-256；
2. G1、左 RH56DFTP、右 OmniPicker、透明/不透明 source/target vessels 七类资产的真实文件、SHA-256 和 calibration receipt；
3. native liquid、viscosity、rim flux、per-vessel volume、outside volume、mass residual、contact、RGB、depth、instance、same-state capture 十一项 capability receipt；
4. camera-state 实测 skew ≤5 ms；
5. 两名不同 reviewer、完整 evidence bundle hash 和全部 capability tests 为 true 的 `overall_status=go` readiness receipt；
6. runtime variant receipt，证明 action-visible inputs、route、variant config、包含所有 checkpoint shard hashes 的 bundle manifest 和代码 commit 与预注册对照一致；verifier 哈希 bundle manifest，不反复读取巨型 checkpoint。
7. 非空 field metric 必须绑定 prediction artifact、metric evaluator 文件与各自 SHA-256；没有 field prediction 的 variant 必须显式写全字段 `null`，不能省略或填零。
8. matched-control report 必须证明所有 variant 共用 scenario factors、physics/render/policy seeds 与 policy stochastic schedule；variant-specific input/route 不进入共同 fingerprint。

缺一项就不能评分。当前 `local_readiness_20260811.json` 是 `overall_status=no_go`，所以 execution validation 必然失败，这是预期安全行为。

## 使用方式

以下命令只构建/校验 spec，不启动 simulator。每行是一条独立命令：

```bash
cd /home/ykj/project/WM/giga-world-policy
/home/ykj/miniconda3/envs/gigaworld-policy/bin/python -B track_b/sim_oracle_benchmark_v0/generate_scenarios.py --summary-only
/home/ykj/miniconda3/envs/gigaworld-policy/bin/python -B track_b/sim_oracle_benchmark_v0/generate_scenarios.py --output track_b/sim_oracle_benchmark_v0/scenario_manifest_spec_only.json
/home/ykj/miniconda3/envs/gigaworld-policy/bin/python -B track_b/sim_oracle_benchmark_v0/validate_sim_oracle_manifest.py track_b/sim_oracle_benchmark_v0/scenario_manifest_spec_only.json --mode spec
```

`verify_sim_oracle_results.py` 只接受 execution-ready manifest；它会重新哈希 manifest、adapter、assets、calibration receipts、variant config/checkpoint bundle manifest、field prediction/evaluator 和每个 native GT artifact（相同 immutable path/size/mtime 在单次核验中只读一次），检查全矩阵 paired coverage 与相同 scenario 的 control/GT receipts，再重算 volume/mass/success/contact timing/risk 指标、汇总带 provenance 的 field metrics并计算 gate。当前不要运行它来制造结果。

审计 verifier 逻辑时，可以在 CPU 内存中构造完整的 1,904 条 synthetic paired records，并显式关闭 artifact 读取来做正反例测试；这种测试只证明 pairing、schema 和 gate 代码行为，不是 simulator rollout、物理证据或 benchmark 结果，不能写入 execution report。

## 代码与数据位置

- `benchmark_spec.json`：factor、异构 embodiment、contact grammar、GT、对照、metrics 和 gate 的唯一设计源；
- `local_readiness_20260811.json`：本地引擎/资产证据与 reopen gates；
- `schemas/scenario_manifest.schema.json`：scenario manifest wire schema；
- `schemas/rollout_record.schema.json`：rollout record wire schema；
- `generate_scenarios.py`：纯 stdlib、确定性、默认不写输出的场景生成器；
- `validate_sim_oracle_manifest.py`：spec / execution 双模式 validator；
- `verify_sim_oracle_results.py`：paired metric verifier，不接受当前 No-Go receipt；
- `scenario_manifest_spec_only.json`：由当前 generator 产生的 272 条 canonical design；它明确不是 rollout。

## 原始工作与官方能力边界

- MuJoCo 的定位和接触动力学来自原始论文 [MuJoCo: A physics engine for model-based control](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)；官方 [MJCF reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) 支持 contact 与 flex/deformable，但本审计没有找到可直接满足本任务的 viscosity-controlled free-surface liquid、rim flux 和 contained-volume contract。
- Isaac Sim 5.1 官方 [Core API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html) 暴露 PBD ParticleMaterial 的 viscosity 等参数；官方 [Contact Sensor](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physics_contact.html) 可返回 pair、position、normal 和 impulse；官方 [Replicator writers](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.replicator.writers/docs/index.html) 有 semantic segmentation 和 distance-to-image-plane。
- Omni PhysX 官方 [Particles documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html) 明确 PBD fluids 是 GPU-accelerated，且 CPU particle simulation 不受支持；底层方法来自 Macklin 与 Müller 的 [Position Based Fluids](https://matthias-research.github.io/pages/publications/pbf_sig_preprint.pdf)。这同时说明“有 API”不等于已经通过本任务质量守恒和 sim-to-real 校准。
- Genesis World 官方 [physics engine API](https://genesis-world.readthedocs.io/en/v0.3.14/api_reference/engine/index.html) 与 [particle emitter docs](https://genesis-world.readthedocs.io/en/latest/user_guide/physics/emitters.html) 支持 SPH/MPM/PBD liquid；[官方概览](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/what_is_genesis.html) 还列出 contact/tactile/camera sensors。但本机没有可验证安装或本任务资产，所以只列为未来候选，不构成选型结论。

## 何时重开

只有在 `local_readiness_20260811.json` 的八项 minimum reopen gates 全部获得实物/代码/标定证据后，才值得另开一次经授权的 simulator bring-up。即使 simulator oracle gate 通过，也只说明 contact grammar 在受控物理中的 ceiling；真实透明重建、真实 depth、真实液体状态与 sim-to-real 仍然是独立 No-Go/验证问题。
