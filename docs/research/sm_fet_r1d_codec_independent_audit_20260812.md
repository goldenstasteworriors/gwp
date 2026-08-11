# SM-FET `minimal-r1d` codec 独立实现审计

- 日期：2026-08-12
- 审计对象：`dec1988465cf444f47859f5acb97a390d1f56d5a`
- 审计基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 审计方式：目标 worktree 只读；独立 detached worktree；Python 3.13.9 标准库；CPU only
- 规范优先级：freeze addendum 高于主合同；最终 addendum review 用于确定验收边界
- 最终实现裁决：**No-Go（不得标记为 CPU-conformant Freeze）**
- 严重度：**P0 = 0；P1 = 8 类**

## 0. 一句话结论

`dec1988` 的 32-byte wire、NULL/PAD、dense locator/payload/bundle/global rank、13-byte little-endian container、冻结 topology、VERIFY PATH 拒绝、same-type combinadic、MONITOR zero ownership mask及 human JSON 主路径通过了独立 CPU 复测；没有发现结构整数或 canonical rank 映射的 P0 错误。

但它尚不能把最终复审中的 `CPU implementation/conformance: PENDING` 升级为 `CPU-conformant Freeze`。资源冲突判定不完整、resource target 只验长度、public-state/codebook tolerance 可由调用方放宽、receipt 时钟可用 NaN/bool 绕过、metric center 接受错误边界、假 codebook hash 可被视为 defined、profile count 可报告数学上不可能的容量、duplicate JSON property 无 fail-closed 入口，以及 FRAME/PATH/quantizer golden surface 未实现。任一项都足以阻塞 exact-contract conformance；合并后将这些 API 用作 profile/admission/receipt gate 会产生错误放行。

书面规范本身的裁决不变：

```text
written contract + normative addendum: Doc structural Freeze
this implementation:                    No-Go for CPU-conformant Freeze
G1 executable profile:                  No-Go
G1 profile states/bits:                 UNDEFINED
```

## 1. 审计依据与变更边界

### 1.1 规范依据

本审计逐条对照：

1. 主合同 `5019c1828cf17e169c51ecf6772ad16b23d51c2d`：
   `docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`；
2. 具有更高规范优先级的 addendum `c9ce174f8f89f73d11d1def8865c04c66588ef7d`：
   `docs/research/sm_fet_minimal_r1d_freeze_addendum_20260812.md`；
3. addendum 最终只读复审 `d1ac9e6f609dccc2bdddd3feb825725a00ec6568`：
   `docs/research/sm_fet_minimal_r1d_freeze_addendum_review_20260812.md`；
4. 上一轮六项 conditional gate 的来源 `fe24e2fa00dcd4f66f6b7f7415faec8042a7b855`：
   `docs/research/sm_fet_minimal_r1d_final_review_20260812.md`。

最终 review 明确区分三层结论：书面合同已 Doc structural Freeze；CPU implementation/conformance 仍 PENDING；真实 G1 executable profile 仍 No-Go。因此本审计只判断目标提交能否完成第二层，不要求它伪造真实 G1 calibration/profile。

### 1.2 提交边界

相对 canonical `e0ebc692`，目标提交只新增：

```text
A docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md
A world_action_model/sm_fet_minimal_r1d_schema.py
```

未修改或接线任何 model、trainer、训练/推理 config、dataset、checkpoint、CUDA/GPU、driver 或 rollout 路径。目标 worktree 在审计前后均为 clean。目标文件 SHA-256：

```text
46d416151d4fe9a2730f3a304c760fb9bb4f583e96bcaf123a29cc6e9dd5621d  world_action_model/sm_fet_minimal_r1d_schema.py
69f88109acb91a11e82e344563c67ebc2093de65c57838f5b293c23b8ca9192a  docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md
```

## 2. 独立 CPU 复测

### 2.1 隔离与可复核边界

- 在 `dec1988` 上创建 detached worktree；未在目标 worktree 执行写操作。
- 使用系统 Python 3.13.9 与标准库；未进入 conda，未安装依赖，未调用 GPU。
- 临时测试脚本仅存在于 detached worktree；运行后删除，随后移除整个测试 worktree。
- 最终两轮共执行 165,247 个检查/断言；所有“应通过”与“应以指定 error code 拒绝”的断言通过；另将“规范应拒绝但实现接受”的调用记录为 finding probe，而不是把它们误算成测试通过。

### 2.2 覆盖与结果

| 项目 | 覆盖 | 结果 |
|---|---|---|
| toy topology/global | `n=3,v=2`，全部 813 states，左右 inverse、无洞、无重、所有冻结区间 | Pass |
| production topology boundary | nonVERIFY 10 blocks、VERIFY 8 blocks 的 first/last、相邻 block 边界 | Pass |
| global golden vectors | NULL、h0 first/last、h1 first、h10 last、VERIFY first、global last及冻结 13-byte hex | Pass |
| locator/payload | POINT/AXIS/FRAME/PATH 首尾，`p=0/P-1`，secondary 0..21，part/role 首尾，VERIFY dense rank | Pass |
| 随机 property | seed `20260812`；20,000 个 production global ranks；packet/wire/compact/rank/equal-bit全链路 | Pass |
| rank mutation | 2,000 次 13-byte 单 bit mutation；reserved、越界或 exact mutated integer，无 alias/fallback | Pass |
| wire/constraint negatives | length、NULL、PAD、field range、kind coherence、VERIFY PATH、function/assignment、非法 topology、secondary/relation、sort/duplicate、header blocks | Pass |
| enum/type negatives | phase/effect全组合；function×assignment全组合；role/part/kind/relation；claim mode/namespace/selector；bool/numeric/string type probes | Mixed，见 P1-06及非阻塞偏差 |
| human/text JSON | 412-byte frozen example/hash、key reorder、unknown key、numeric/bool/pipe enum、sort、duplicate tuple、focus、NULL、registry alias | 主路径 Pass；duplicate property入口 Fail |
| public state | exact 226B、hash、stale、NaN/Inf/negative-zero、camera PAD、quaternion sign/unit、held PAD/binding/part | 主路径 Pass；tolerance binding Fail |
| receipts | static profile、metric codebook、runtime resource、held public state；self-hash、unknown key、artifact/profile/state binding、freshness | 主路径 Pass；strict clock Fail |
| resource conflicts | physical actuator claim、MONITOR READ_ONLY、aperture mutex、wrist explicit compatibility | 部分 Pass；通用 conflict/target identity Fail |
| metric codebook | exact manifest/table size、W/H/intrinsics、edges、512 axes、RASTER vertices、NaN/Inf/hash | 部分 Pass；cell/tolerance/full golden surface Fail |
| missing/transparent | unknown、expired、reconstruction、non-unique support、kind-required fields、transparent no-depth/normal/TOUCHING | 主路径 Pass；见 §5.3 规范措辞风险 |

关键冻结值全部复算一致：

```text
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

## 3. P0

**未发现 P0。**

没有复现会破坏整个 structural alphabet 的状态数错误、global rank collision、unrank hole、32/13-byte 长度错误、production block offset错误或 endian 错误。这里的“无 P0”不等价于 Go；以下 P1 均是 exact-contract release blocker。

## 4. P1 findings

### P1-01：resource target identity 只验长度，resource conflict 只覆盖 aperture/wrist

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1481-1506`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1509-1555`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1558-1589`

证据：

1. `validate_resource_claim()` 只根据 selector 检查 `target_key` 字节长度，不解析：
   - `LOCATOR_POINT` 的 translation rank 是否 `<P`；
   - `LOCATOR_AXIS` 的 `p/axis` 是否 canonical；
   - `LOCATOR_FULL` 是否属于对应 dense locator alphabet；
   - `ENTITY_LOCATOR` 内 role/part/secondary/relation/kind/rank 是否 canonical。
2. `LOCATOR_FULL + b"\xff"*8` 被接受。
3. `validate_resource_compatibility()` 只对两个 aperture claim或两个 wrist claim做 distinct-target mutex；同一 `resource_id` 上两个不同 `EXCLUSIVE_TARGET`，若 namespace 为 `STABILIZED_ENTITY` 等其它值，会被接受。
4. 单 constraint claims 只检查“至少一个 actuator claim”，不验证同一 resource 的唯一模板、重复/冲突模板或多模板 ambiguity。

这违反 addendum §3.1 的 selector exact bytes、纯静态唯一映射 `F(p,t,wire,resource)` 与 `UNKNOWN_NOT_ADMITTED`，也未覆盖主合同 §2.3 的完整 finite-resource conflict规则。影响是 `RO-RS`、`RO-RO`、`RS-RS` 等 profile pair可被错误 admit；这不是单纯错误码差异。

### P1-02：public-state quaternion tolerance 未绑定 immutable manifest/profile，可由调用方放宽

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1366-1373`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1376-1417`

`validate_public_execution_state()` 接受调用方传入任意正 finite `quaternion_unit_tolerance`，没有 receipt/hash 参数证明该 tolerance 来自冻结 manifest。反例：moving-camera quaternion `(w,x,y,z)=(0.5,0,0,0)` 在正常小 tolerance 下正确拒绝，但传 `quaternion_unit_tolerance=1.0` 会通过。

主合同 §4.2 要求 unit norm在 manifest tolerance内；最终 conformance gate要求 noncanonical quaternion fail closed。当前 API 校验的是“调用方选择的规则”，不是被冻结/receipt-bound的规则，因此 public-state canonical bytes可被同一实现以不同标准解释。

### P1-03：held/runtime receipt 时钟参数不做 strict integer校验，NaN 可绕过 freshness

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1891-1924`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1941-1974`

receipt 内的 timestamp 会经 `_strict_int`，但调用参数没有相同校验：

- `validate_held_public_state_receipt(..., now_ns=float("nan"))` 接受；
- 同函数接受 `now_ns=True`；
- `validate_runtime_resource_receipt(..., now_ns=float("nan"))` 接受；
- runtime validator 接受 `anchor_timestamp_ns=100.0` 与 receipt integer `100` 的宽松相等。

Python 中 NaN 的 `>` 比较为 false，因此 `now_ns > deadline` 无法触发 stale。该行为直接违反 addendum §6.6 的 stale 必须拒绝，也违反 exact int64 clock boundary。它使过期 receipt 可在 public gate中被视为有效。

### P1-04：metric codebook 的 cell canonicality 与 axis generator校验可被绕过

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1681-1697`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1723-1766`

有两个独立反例：

1. 所有 cell 都用 `edge[i] <= center <= edge[i+1]`。非最后 cell 的 center若恰等于右边界仍被接受；但 addendum §5.2 冻结的是内部 cell `[edge_i,edge_{i+1})`，内部 edge属于高 index。
2. `axis_unit_tolerance` 是任意调用参数，并同时用于 unit norm和“等于冻结 generator center”的比较。把一个 axis center 明显改坏后，传 `axis_unit_tolerance=2.0`，重新计算 artifact hash，validator仍接受。

因此可以生成 checksum自洽、却不符合冻结 bin语义或 axis table的 `metric_codebook_receipt_v1`。hash完整性不能弥补被 hash内容本身的 canonical validator失效。

### P1-05：`require_defined_codebook()` 把任意 64-hex（包括全零 placeholder）视为真实 artifact

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1778-1781`
- 目标 conformance memo第25行

函数只做 `[0-9a-f]{64}` 正则检查；`"0"*64` 直接返回。它没有绑定 manifest/tables、codebook receipt、authorized expected hash或当前 profile。目标 conformance memo却声称“缺失/占位 hash返回 `CODEBOOK_UNDEFINED`”。

addendum §6.7 明确要求真实 metric tables不存在时结果必须是 `CODEBOOK_UNDEFINED` 并阻止任何 G1 executable packet。当前函数只能证明“像一个 SHA-256 字符串”，不能证明“defined artifact存在”。由于它是公开导出的 gate，这一差异是 fail-open，不是文档措辞问题。

### P1-06：strict JSON/type boundary不完整；duplicate object property与若干非canonical API类型被接受

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1172-1201`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1210-1306`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:743-790`
- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:824-888`

通过项：unknown key、numeric/bool enum、pipe-string、duplicate constraint tuple、错误 sort/focus、nonEXHAUSTIVE NULL均正确拒绝；冻结 412-byte example/hash正确。

缺口：公开 validator只接收已经构造好的 Python `dict`，没有 duplicate-preserving JSON bytes/parser入口。带两个 `task_family` property 的原始 JSON经标准 `json.loads` 后前一个值静默丢失，随后 `validate_shared_semantic_key()` 接受并 hash；实现无法证明输入满足 exact JSON object contract。

此外：

- raw string `"LO"` 会因 `ConstraintType` 继承 `str` 而被 `unrank_payload()` / `rank_topology()` 接受；
- `verify=0/1` 会被多个 rank helper当作 bool接受；
- 导出的 `PathPrimitive.LINE` 反而因 `_strict_int(type(value) is int)` 被 locator拒绝，只接受裸整数 primitive。

后两项本身不改变 wire rank，但说明“strict typed public API”并不统一。duplicate property会改变 exact bytes/hash信任边界，因此本项按 P1 处理。

### P1-07：`profile_bundle_capacity()` 可报告数学上不可能的 profile整数

实现位置：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1021-1061`

函数检查 exact mapping keys与非负整数，却不检查：

- `C(A,B) <= |A|*|B|`；
- `U(A) <= C(|A|,2)`；
- 任一集合为空时相关 pair count必须为零。

独立反例给 `|LO|=|RO|=1`，却给 `C(LO,RO)=999`、`U(RO)=999`，函数仍返回正容量。主合同 §9.2把 `C/U` 定义为实际 finite compatibility graph的枚举计数；§12 gate 8要求 formula与枚举图逐项一致。当前 helper不能安全支撑公开 profile capacity或 receipt/report。

### P1-08：最终 review 要求的完整 metric/control CPU golden surface 未实现

实现位置：整个目标模块，尤其：

- `dec1988:world_action_model/sm_fet_minimal_r1d_schema.py:1592-1775`
- `dec1988:docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md:69-84`

已实现的是 manifest/tables静态验证：W/H、edges、materialized axis centers、RASTER vertices和checksum。未发现以下公开实现或可执行 golden surface：

- UV/depth/extent内部 edge与最后右闭的 quantizer；
- `rho2 <= 0x1p-80` vertical branch及 azimuth/elevation tie/wrap；
- roll seed `min abs(dot)` 与 `ex<ey<ez` tie；
- FRAME zero-roll basis、正交/右手检查；
- LINE/ARC/RASTER arc-length参数、vertex outgoing/final incoming tangent；
- PATH fixed-normal orientation transport；
- POINT/AXIS partial-goal manifold与 VERIFY `HOLD_EXISTING` anchor satisfaction的 generic CPU oracle。

最终 addendum review第89–98行明确仍把完整 CPU conformance列为 PENDING；此前 final review第318–329行要求实际运行 axis/frame/path/checksum golden vectors后才能升级。目标 memo只说“for implemented frozen gates”，这不能替代遗漏的冻结 gates。即使其它 bug全部不存在，当前覆盖仍不足以给 CPU-conformant Freeze。

## 5. 通过项与边界说明

### 5.1 Wire、rank与 topology：通过

- field wire恰 32 bytes；每 constraint 15 bytes；唯一 NULL全零；slot padding canonical。
- nonVERIFY/VERIFY使用不同 dense locator alphabet；VERIFY PATH以 `VERIFY_PATH_FORBIDDEN`拒绝。
- 10个nonVERIFY topology blocks与8个VERIFY blocks的顺序、offset、row-major pair、整数 combinadic inverse正确。
- global rank完整覆盖 `0..|G|-1`；13-byte unsigned LE的 reserved bits、invalid rank、length priority正确。
- same-type duplicate与逆序、different-type slot swap均按冻结规则拒绝。
- equal-bit affine/inverse常量与20,000随机 round-trip一致。

### 5.2 MONITOR：结构与 own slice通过，action effect不可由本提交证明

`actuator_ownership_mask()` 对 ZM 返回25维全零；ZM仍有完整 constraint/locator wire并进入合法 topology。该部分符合“action-visible、zero own slice”的结构合同。

本提交没有 model/action expert，因此无法证明 MONITOR token会条件化其它 physical slice；目标提交也没有声称训练/rollout证据。该实验性要求仍保持 pending，不作为 codec算术缺陷。

### 5.3 Missing/transparent：missing主路径通过，transparent措辞需在后续修订中消歧

实现正确拒绝 unknown、expired、reconstructed、non-unique support、kind-required field缺失、transparent无 direct depth、transparent AXIS/FRAME/PATH，以及 transparent `TOUCHING/INSIDE`。

实现会接受 `transparent_overlap=True + direct_depth=True + POINT + NEAR`。主合同 §8.5 的细则可读作“有独立 direct depth时允许 POINT”，但主合同 §11与上一轮 final review §8.2又使用“transparent 2D overlap不能给 depth”的绝对措辞。由于同一规范集内部存在摘要/细则张力，本审计不把该单点另计第9项 P1；实现若保留该组合，后续应把“direct depth的独立来源与 provenance”写成 exact field/receipt rule，不能让 `direct_depth=True` 成为无绑定的布尔旁路。

### 5.4 Receipts：nominal binding通过，不代表真实 artifact/profile存在

以下 nominal行为通过：

- static profile receipt exact keys/self-hash/hardware family/hash字段；
- metric receipt对 manifest、tables、calibration和codebook digest的绑定；
- held receipt对 public-state hash、source/calibration、anchor/deadline的 nominal绑定；
- runtime receipt对 static profile receipt hash、role/resource/monitor hashes和 admission enum的 nominal绑定；
- unknown keys与错误 self-hash拒绝。

这不消除 P1-03至P1-05，也不构成真实 G1 profile。`capacity_metadata()` 继续报告 `G1 profile states/bits = UNDEFINED / NO-GO` 是正确的。

## 6. Go/No-Go

### 当前裁决：No-Go

不批准以下表述或用途：

```text
CPU-conformant Freeze
exact public profile/codebook/runtime admission gate
G1 executable codec/profile
```

可以保留的窄义表述：

```text
32-byte structural wire and global rank core passed independent CPU property tests,
but the full frozen CPU conformance contract did not pass.
```

### 重新送审前的最小条件

1. 用 canonical selector decoder验证每种 `target_key`，实现同一 resource的完整 claim conflict与唯一模板规则；
2. 将 quaternion/axis tolerance绑定到 immutable manifest/profile receipt，禁止任意调用方参数改变 canonicality；
3. 对所有外部 timestamp/anchor参数使用 exact int、拒绝 bool/float/NaN；
4. 修正非最后 metric cell为半开区间，并固定 generator comparison规则；
5. 让“defined codebook”绑定实际 validated artifact/authorized receipt，而不是只验 64-hex；
6. 提供 duplicate-rejecting JSON bytes入口或强制 duplicate-preserving parser contract，并统一公开 enum/bool类型；
7. 对 profile pair/unordered count实施集合基数上界或直接从 compatibility graph枚举；
8. 实现并保存 addendum §5/§6要求的 axis/roll/frame/path/quantizer golden tests；
9. 重跑本审计的 toy813、全部 production boundary、至少10,000随机、rank mutation、bad wire/JSON/state/receipt/resource/hash反例。

修复不应静默改变字段、topology或状态整数；若必须改变，按冻结合同升 schema并重计。

## 7. 最终审计记录

```text
P0: 0
P1: 8 categories
wire/rank core: Pass
full CPU conformance: Fail
implementation acceptance: No-Go
Doc structural Freeze: unchanged
G1 executable profile: No-Go / UNDEFINED
target worktree modified by auditor: no
temporary tests retained: no
model/trainer/config touched by target commit: no
GPU/CUDA used: no
```
