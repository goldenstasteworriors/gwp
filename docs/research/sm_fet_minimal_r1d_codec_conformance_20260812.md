# SM-FET `minimal-r1d` CPU schema codec conformance

- 日期：2026-08-12
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 实现：`world_action_model/sm_fet_minimal_r1d_schema.py`
- 规范来源：`sm_fet_minimal_r1d_schema_contract_20260812.md`（`5019c182`）及优先级更高的 `sm_fet_minimal_r1d_freeze_addendum_20260812.md`（`c9ce174f`）
- 依赖：Python standard library only；CPU only

## 边界

本实现是 **hardware-family structural codec conformance**，不是 G1 executable profile。它不生成真实 depth/extent ranges、`tables.bin`、metric codebook hash、G1/RH56/Omni static capability entries、working-surface/resource allowlist、runtime reachability/WBC/collision receipt，也不批准训练、仿真或真机执行。

因此模块固定报告：

```text
|G_struct_r1d| = 157,516,896,299,151,958,972,545,105,921
log2           = 96.99142134161319 bit
index width    = 97 bit
wire           = 32 bytes / 5 hard tokens
compact        = 13-byte unsigned little-endian
G1 profile     = UNDEFINED / NO-GO
profile bits   = UNDEFINED / NO-GO
```

`require_defined_codebook()` 会对缺失/占位 hash 返回 `CODEBOOK_UNDEFINED`；profile/codebook validator 只校验调用方显式提供且已绑定 hash 的 immutable artifact，不会构造默认 artifact 或把 structural grammar 当作部署 admission。

## 实现覆盖

模块提供：

- typed phase/effect/function/assignment/role/part/kind/relation/primitive enums 与 immutable packet dataclass；
- canonical 32-byte field wire pack/unpack，NULL/PAD、locator-kind、O/S/M assignment、nonVERIFY/VERIFY topology、secondary/relation、same-type distinct/sort 校验；
- dense nonVERIFY/VERIFY locator/payload rank、整数 combinadic、bundle/global rank/unrank、13-byte LE 与 global equal-bit diagnostic；
- exact capacity metadata 与 finite-profile count 公式；
- MONITOR 25-D actuator ownership all-zero mask，同时保留 constraint/token 的结构存在；
- fail-closed locator evidence boundary：unknown/missing/expired、non-unique support、completion/reconstruction 与 transparent missing-depth/normal/relation 不编码为 hard packet；
- human shared semantic JSON 与 structured text 的 exact-key/string-enum/sort/duplicate/focus/EXHAUSTIVE/NULL/ontology-alias 校验、canonical JSON bytes 和 SHA-256；
- 226-byte `public_execution_state_v1` 的 timestamp、float32 finite/negative-zero、camera mode、quaternion sign/unit、held PAD/binding/part/uncertainty 与 hash 校验；
- frozen resource-claim target identity、physical actuator claim、aperture mutex、explicit wrist compatibility 校验；
- exact metric manifest keys/constants、fixed binary64 table layout、W/H/intrinsics、edge/center、512 axis generator、RASTER vertices 与 checksum 校验；
- static profile、metric codebook、runtime resource、held public-state 四层 self-hashing receipt 校验，runtime receipt 不能替换 static profile hash。

Receipt 本体的业务 artifact 可独立演进；本模块公开的 receipt validator 使用 exact binding envelope，拒绝 unknown key，并将完整 artifact 通过 SHA-256 字段绑定：

```text
static_capability_profile_receipt_v1:
  schema_id, profile_id, hardware_family, task_stratum_id,
  hardware_manifest_sha256, metric_codebook_sha256,
  profile_payload_sha256, receipt_sha256

metric_codebook_receipt_v1:
  schema_id, codebook_sha256, source_calibration_sha256,
  manifest_sha256, tables_sha256, receipt_sha256

runtime_resource_receipt_v1:
  schema_id, runtime_state_id, profile_receipt_sha256,
  anchor_timestamp_ns, freshness_deadline_ns,
  role_instances_sha256, resource_facts_sha256,
  monitor_evidence_sha256, admission_status, receipt_sha256

held_public_state_receipt_v1:
  schema_id, public_state_sha256, source_id,
  source_manifest_sha256, calibration_sha256,
  anchor_timestamp_ns, freshness_deadline_ns, receipt_sha256
```

每个 `receipt_sha256` 都是删除自身字段后，对其余 exact object 做 canonical JSON 再 SHA-256。上述 envelope 只做分层、freshness 和 artifact binding，不允许 receipt metadata 成为 action channel。

## CPU conformance 结果

本次在工作树内用临时 stdlib 测试脚本执行并在结束后删除，未使用 GPU、训练或外部依赖：

- toy `n=3,v=2`：穷举 813 个 symbolic topology/global states，无洞、无重、双向 round-trip；
- production：nonVERIFY 10 个 block、VERIFY 8 个 block 的 first/last 及相邻边界，global 7 个冻结 rank/13-byte hex vectors；
- locator/payload：POINT/AXIS/FRAME/PATH 首尾、`p=0/P-1`、secondary `0..21`、part/role 首尾；
- property：固定 seed `20260812` 的 1,000 个 production global rank，逐项执行 packet/wire/compact/rank 与 affine inverse round-trip；
- bad vectors：length/reserved/global-rank、PAD/range、kind/VERIFY PATH、function/assignment/topology、secondary、sort/duplicate、header、profile resource mutex/no-actuator claim；
- JSON/state：冻结 412-byte semantic example/hash、extra/numeric/duplicate/focus/NULL、text registry alias、NaN/Inf/negative-zero/PAD/stale/hash；
- import-time constants：所有冻结整数、`bit_length(M)=97`、`gcd(65537,M)=1`、冻结 inverse 与 `96.99142134161319`。

结论仅为：

```text
structural schema codec: CPU-conformant for implemented frozen gates
G1 executable profile:   NO-GO
G1 profile states/bits:  UNDEFINED
```
