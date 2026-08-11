# SM-FET `minimal-r1d` codec fix1 独立复审

- 日期：2026-08-12
- 复审对象：`aa13981e5959611ad39574dca381fdede48db343`
- fix1 基线：`dec1988465cf444f47859f5acb97a390d1f56d5a`
- 仓库 canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 复审方式：目标 worktree 只读；独立 detached worktree；Python 3.13.9 标准库；CPU only
- 规范优先级：freeze addendum 高于主合同；最终只读 review 确定验收边界
- 严重度：**P0 = 0；P1 = 2 类**
- 最终裁决：**No-Go（仍不得标记为 CPU-conformant Freeze）**

## 0. 一句话结论

fix1 对上一轮 8 类 P1 做了实质修复：32-byte wire、dense rank/unrank、97-bit state integer、13-byte little-endian container、冻结 topology、VERIFY、MONITOR zero slice、strict JSON、public-state/profile/codebook receipts、missing/transparent、strict numeric与unknown-key边界均通过本轮独立复测；核心 schema ID、alphabet整数和 profile capacity 公式未漂移，也未修改 model、trainer、config 或 GPU/CUDA 路径。

但仍有两个 exact-contract release blocker：资源兼容性可以通过未进入 immutable profile/context hash 的逐调用 wrist allowlist 改写；axis/roll quantizer 会把严格位于内部 edge 下方的非 tie binary64 输入吸附到 edge并错误分入高格。后者也说明新增 golden surface 只覆盖 center round-trip/hash，没有覆盖每个 edge 的相邻 binary64 语义。因此本提交不能把最终 review 中的 CPU implementation/conformance 从 PENDING 升级为 Freeze Go。

```text
written contract + normative addendum: Doc structural Freeze
fix1 implementation:                    No-Go for CPU-conformant Freeze
G1 executable profile:                  No-Go（独立于 codec 修复状态）
G1 profile states/bits:                 UNDEFINED
```

## 1. 复审边界与无漂移检查

本轮沿用上一轮规范集合：主合同 `sm_fet_minimal_r1d_schema_contract_20260812.md`、高优先级 `sm_fet_minimal_r1d_freeze_addendum_20260812.md`、addendum 最终只读 review以及此前 final review。fix1 相对 `dec1988` 只改动：

```text
M docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md
M world_action_model/sm_fet_minimal_r1d_schema.py
```

变更量为 1,052 insertions、91 deletions。没有 model、trainer、训练/推理 config、dataset、checkpoint、CUDA/GPU、driver 或 rollout 接线变更。目标 worktree 在复审前后均 clean；临时脚本只存在于 detached worktree，运行后已删除并移除整个隔离 worktree。

fix1 文件 SHA-256：

```text
c8e94c2cb6cfc5385be620ac8a829ac0e6d4dfdeb7567f1611c6bda8ab6c0d52  world_action_model/sm_fet_minimal_r1d_schema.py
1f536983e81a064e5730468dbe13cba91df4d51c8e7878acdc1a64d8c6ea76f1  docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md
```

核心结构常量逐项复算保持不变：

```text
schema_id         = sm_fet_hard_grammar_v2_minimal_r1d_struct_v1
wire bytes        = 32
compact bytes     = 13, unsigned little-endian
P                 = 131,072
L_nonverify       = 70,934,200,320
L_verify          = 2,214,723,584
N                 = 43,695,467,397,120
V                 = 194,895,675,392
B_nonverify       = 14,319,704,032,895,892,051,583,303,680
B_verify          = 151,937,297,146,405,128,765,440
|G_struct_r1d|    = 157,516,896,299,151,958,972,545,105,921
bit_length(G - 1) = 97
```

static profile/public-policy/resource-context/metric receipt新增的 hash字段属于 auxiliary validation envelope，用于关闭上一轮绑定缺口；没有改变32-byte core wire、topology顺序、rank区间或容量公式。profile capacity helper现在拒绝 `C(A,B)>|A||B|`、`U(A)>choose(|A|,2)` 等不可能计数，未发现 capacity drift。

## 2. 独立 CPU 复测

单轮共执行 **222,170** 个检查/断言。所有 nominal case与指定负例均按预期通过；规范应拒绝但实现接受的三条 probe单独记录为 finding，不计作“通过”。

| 项目 | 覆盖 | 结果 |
|---|---|---|
| toy topology/global | `n=3,v=2` 全部 813 states，左右 inverse、无洞、无重、冻结区间 | Pass |
| production wire/rank | 32B exact、13B LE、global/block边界、7个冻结 compact vectors、noncanonical pack | Pass |
| production random | seed `20260812`；25,000 global ranks；packet/wire/compact/rank全链路 | Pass |
| rank mutation | 2,000次 13-byte单 bit mutation；reserved/越界拒绝或精确变为 mutated integer | Pass |
| enum/constraint/topology negatives | 每类 header/field/function/assignment/role/part/kind/relation、VERIFY PATH、排序/重复 | Pass |
| strict JSON | raw bytes入口、duplicate property、unknown key、bool/numeric enum、NaN/Inf、排序/重复/focus | Pass |
| public state/profile | policy hash绑定、quaternion canonical/unit、freshness、held/camera PAD、NaN/Inf/-0 | Pass |
| receipt clocks | `now_ns`、anchor严格 int；bool/float/NaN/Inf；artifact/profile/state/hash/freshness | Pass |
| codebook | exact manifest/tables、half-open centers、512 axes、receipt/authorized hash、fake/all-zero hash | Pass |
| resources | selector exact bytes、single-template、generic same-resource mutex、actuator、MONITOR READ_ONLY | **Mixed，见 P1-01** |
| metric/control surface | axis/frame/path固定 hash；512 axis、16,384 frame/roll center round-trip；PATH/VERIFY oracle | **Mixed，见 P1-02** |
| missing/transparent | unknown/expired/reconstruction/non-unique、required fields、direct-depth、relation约束 | Pass |
| ownership | MONITOR 25维全零 slice；physical ownership形状/类型 | Pass |

新增 geometry golden hashes复算与实现常量一致：

```text
axis  d9d9319054b954fceb759dd6c9c348f588813176b50466e54950691f0d14a5de
frame 0e0e05bad6118cd3b597f895a6ebf997842208f99a1f3580e834f1edd80073f9
path  191a9bdbdb35e030de8b915a441a1bdfce8d0bf691132d95c1e359c5c38eb41a
```

## 3. 上一轮 8 类 P1 的逐项处置

| 原 finding | fix1 独立结论 |
|---|---|
| P1-01 resource identity/conflict | **部分关闭**：selector exact解析、单模板、通用same-resource mutex已修；逐调用 wrist compatibility allowlist未绑定，仍为 P1-01 |
| P1-02 public quaternion tolerance可放宽 | **关闭**：由canonical policy artifact给出并绑定static profile receipt |
| P1-03 receipt clocks非strict | **关闭**：`now_ns`/anchor使用strict integer，bool/float/NaN负例拒绝 |
| P1-04 metric center/tolerance旁路 | **原旁路关闭**：center执行内部half-open、axis policy不再由调用者传 tolerance；但新增 quantizer edge snap产生独立 exactness P1-02 |
| P1-05 arbitrary/all-zero codebook hash | **关闭**：artifact、receipt、profile与authorized hash形成完整验证链 |
| P1-06 duplicate JSON/type boundary | **关闭**：raw bytes parser拒绝duplicate property；bool/enums/primitive边界严格化 |
| P1-07 impossible profile counts | **关闭**：pair count受set-product/combination上界约束 |
| P1-08 metric/control golden surface缺失 | **实现面关闭但覆盖不充分**：axis/frame/path/VERIFY实现与hash已新增；edge邻点缺口由 P1-02 证明 |

## 4. 剩余 P1

### P1-01：wrist compatibility table仍是未绑定的逐调用输入

实现位置：`aa13981:world_action_model/sm_fet_minimal_r1d_schema.py:1919-1967`。

`validate_resource_compatibility()` 接收 `compatible_wrist_target_pairs` 参数，并在第1964–1965行直接用它把不同 wrist target从默认 mutex改为 compatible。但该 collection既不属于已校验的 `ResourceValidationContext`，也没有 canonical bytes/hash，更没有进入 `static_capability_profile_receipt_v1`。独立 probe保持两条 constraint、claims、context及 profile receipt字节完全不变：

1. `compatible_wrist_target_pairs=()`：返回 `PROFILE_RESOURCE_MUTEX`；
2. 仅在调用参数加入该 pair：验证通过。

这违反主合同“compatibility只能读 `(p,t,wire payloads)`”以及 addendum“只有profile证明channel disjoint且列入有限compatibility table时才兼容”。“调用者传入一个有限 collection”不等于 immutable/profile-bound table；同一 profile因进程调用参数不同而得到不同 `G_profile`，也会使报告的 profile count与实际 admission graph脱钩。

**修复判定条件（本审计不实施修复）：** compatibility table必须具有canonical artifact、内容校验与profile hash绑定，validator不得接受可独立改变判定的未绑定 allowlist。

### P1-02：8-ULP edge snap把非 tie输入错误量化到高格

实现位置：

- `aa13981:world_action_model/sm_fet_minimal_r1d_schema.py:2166-2173`
- `aa13981:world_action_model/sm_fet_minimal_r1d_schema.py:2176-2198`
- `aa13981:world_action_model/sm_fet_minimal_r1d_schema.py:2258-2276`

`_snap_binary64_edge()` 将距离任一 materialized edge不超过 `8*max(ulp(value),ulp(edge))` 的值无条件改成 edge。主合同/addendum只允许“恰在 internal edge”取高 index；严格低于 edge的输入必须仍在低格。独立、确定性反例：

```text
axis azimuth: computed atan2 strictly 2.220e-16 below pi/4
reference half-open cell: 276
implementation result:   277

roll theta: computed atan2 strictly 2.220e-16 below pi/4
reference half-open cell: 4
implementation result:   5
```

两例均不是数学 tie，也不是非finite/UNKNOWN；它们只位于内部 edge下方约2 ULP，却被8-ULP吸附带改写。该带具有非零宽度，因而不是只影响构造性测试点。当前 axis/frame golden surface以decoded center round-trip及固定hash为主，不能检出每个 edge的 `nextafter(edge,-inf)`/相邻多ULP语义。

**修复判定条件（本审计不实施修复）：** 量化必须对所有严格低于/高于 materialized edge的representable输入维持half-open归属；若平台 trig不能满足书面 correctly-rounded合同，应以规范授权且逐edge覆盖的materialized golden decision机制闭合，而不能使用对称容差带把邻点改写为 tie。

## 5. P0、通过项与最终 gate

**P0 = 0。** 未发现 wire长度/endian、state count、rank collision/hole、locator/payload dense ordering、topology offset、VERIFY block或13-byte reserved-bit错误。两个剩余finding均是 exact-contract的 admission/metric语义错误，按release gate定为P1；任一项都足以阻止 CPU-conformant Freeze。

通过项包括：

- wire exact 32B，唯一NULL、canonical PAD与noncanonical pack拒绝；
- global rank/unrank/state integer和13B unsigned LE全链路；
- nonVERIFY 10 blocks、VERIFY 8 blocks、VERIFY PATH拒绝、same-type combinadic；
- public execution state、validation policy、profile/codebook/resource receipts的canonical bytes/hash/strict clocks；
- MONITOR保留完整constraint语义但ownership slice全零；
- strict JSON duplicate/unknown key/numeric/bool/NaN/Inf拒绝；
- codebook缺失、placeholder、未授权/不匹配hash均 fail closed；
- missing/transparent主路径及 VERIFY `HOLD_EXISTING` public-anchor oracle。

最终 gate：

```text
CPU-conformant Freeze: No-Go
理由：P1-01 profile-unbound resource compatibility；P1-02 non-tie edge snapping
G1 executable profile: No-Go（真实 calibration/profile仍不存在，与上述CPU结论独立）
G1 count/bits: UNDEFINED
```

fix1 的修复方向总体正确且关闭了上一轮大多数 fail-open boundary；本裁决只表示仍不满足 exact-contract Freeze，不否定已通过的 structural codec与receipt修复。
