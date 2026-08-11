# SM-FET `minimal-r1d` freeze addendum 最终只读复审

- 日期：2026-08-12
- 唯一文档裁决：**Doc structural Freeze**
- 被审主合同：`docs/research/sm_fet_minimal_r1d_schema_contract_20260812.md`，commit `5019c1828cf17e169c51ecf6772ad16b23d51c2d`
- 被审 addendum：`docs/research/sm_fet_minimal_r1d_freeze_addendum_20260812.md`，commit `c9ce174f8f89f73d11d1def8865c04c66588ef7d`
- 审阅边界：只复核上一轮六项条件及 addendum 是否引入新矛盾；未扩大到实现、训练或真实 G1 profile
- 审阅方式：只读逐条对照与 Python 标准库即时复算；未创建永久测试脚本，未修改被审 worktree

## 0. 三层结论

```text
written r1d schema contract + normative addendum: DOC STRUCTURAL FREEZE
CPU implementation/conformance:                     PENDING
G1 executable profile:                              NO-GO
G1 profile states/bits:                             UNDEFINED
```

这里的 Freeze 只冻结**主合同与具有规范优先级的 addendum 这一两文档整体**。它不表示 codec/codebook checker 已实现或运行，也不表示真实 metric artifact、G1 capability/resource profile、runtime receipts或task admission已经存在。

上一轮六项中，前五项已经逐字闭合；第六项已经闭合“未来 CPU 测试输入、expected结果和错误边界”的书面合同，但测试执行本身仍是 PENDING。按本轮明确要求区分书面规范与实现 conformance，因此最终文档裁决是 **Doc structural Freeze**，不是把 CPU pending误报为通过，也不是继续把书面合同判为 Conditional。

## 1. 六项逐条复核

| 上轮条件 | addendum 闭合方式 | 裁决 |
|---|---|---|
| bundle-local `b`、block offsets、pair rank与inverse/unrank | §1 给出三层顺序、10个nonVERIFY与8个VERIFY区间、different-type row-major、整数 combinadic inverse、global unrank、invalid rank和13-byte LE规则 | **书面闭合** |
| POINT/AXIS partial frame与VERIFY `N_A+STABILIZE` | §2 把POINT/AXIS定义为目标流形，禁止hidden completion；VERIFY LS/RS直接保持公开effector frame，限定`HOLD_EXISTING`且anchor未满足即拒绝 | **书面闭合** |
| RIGHT finite target identity与空claim旁路 | §3 定义finite bytewise `resource_target_id`和纯静态`F(p,t,wire,resource)`；aperture/entity分域；physical O/S必须有actuator claim | **书面闭合** |
| human/text JSON pipe-string歧义 | §4 给exact key sets、逐项one-of enum、NULL分支、真实单值示例并明确pipe-string非法 | **书面闭合** |
| codebook `W,H` 与axis epsilon | §5 将post-preprocess W/H、preprocess/calibration、`epsilon²=0x1p-80`、float/tie规则、tables bytes与checksum纳入manifest合同 | **书面闭合** |
| CPU codec/codebook/public-state golden gates | §6 冻结toy/production边界、bad wire、JSON/state、checksum与expected errors；§6开头明确本轮未实现/未运行 | **测试合同闭合；执行 PENDING** |

未发现上述修订新增或删除 field、token、phase/effect block、bundle topology、locator kind或code。故结构整数不变：

```text
|G_struct_r1d| = 157,516,896,299,151,958,972,545,105,921
log2            = 96.99142134161319 bit
minimum width   = 97 bit
wire            = 32 bytes / 5 tokens
compact         = 13-byte unsigned little-endian
```

## 2. 独立常量复算

本轮即时复算得到：

```text
B_nonverify = 14,319,704,032,895,892,051,583,303,680
B_verify    = 151,937,297,146,405,128,765,440
G           = 157,516,896,299,151,958,972,545,105,921
bit_length(G-1) = 97
gcd(65537,G-1)  = 1
inverse          = 138,514,980,096,989,891,931,988,754,433
```

addendum中10个nonVERIFY和8个VERIFY production block的首尾 `b` 全部吻合；global边界的13-byte LE hex全部吻合，包括：

```text
NULL         00000000000000000000000000
h0 first     01000000000000000000000000
h0 last      000038771cef3b1747fb442e00
VERIFY first 0100681f394693ff0dccf6fc01
global last  0000c8e0932a25893aecf6fc01
```

toy `n=3,v=2` 的 `B_nonverify=72`、`B_verify=20`、`G=813`及全部区间端点正确。Human shared JSON真实示例也独立复算为：

```text
canonical UTF-8 length = 412 bytes
SHA-256 = dafc4e176b925bb553fded9b1f9fcd1bc96542b3c8c5288df1be641ea0278fab
```

因此未发现会迫使重计、改变rank或修改golden expected的书面错误。

## 3. 跨文档覆盖与新矛盾检查

addendum对主合同声明规范优先级，实际存在两类有意细化：

1. 主合同把 locator概括成完整 `T_C_P*`、把VERIFY概括成role-level fact；addendum明确POINT/AXIS是partial-goal manifold，并把VERIFY LS/RS解释为公开effector `HOLD_EXISTING`、ZM保留role-level observation；
2. 主合同用“identical RR target merge/deduplicate”作简写；addendum区分resource claim identity与完整semantic payload：相同claim合并，只有完整canonical payload相同才是wire `DUPLICATE`。

这两处不是未决双重语义，因为 addendum §0 已明确优先级，且新定义与现有 topology/count一致。后续若形成单一合并版，宜把这些定义回填主合同以降低误读风险；这属于文档维护项，不阻塞当前两文档整体的 Doc structural Freeze。

未发现新的 allocator/profile/text/proprio/held-transform private correction channel：POINT/AXIS自由度不得私下补齐；RIGHT映射只读 immutable profile/task/wire；runtime只能继续裁剪；structured text alias被拒绝；public-state matched invariant与MONITOR zero-slice保持不变。

## 4. Freeze 后仍未通过的事项

### CPU conformance：PENDING

addendum只提供未来checker的规范与expected，并未运行 exhaustive rank/unrank、bad-wire priority、JSON/state canonicalization或完整metric checksum测试。因此当前只能称：

```text
Doc structural Freeze
CPU-conformant Freeze: not yet established
```

实际 checker若与任一冻结expected不符，必须修实现；若证明书面alphabet/topology本身错误，则升schema并重计，不能静默修改冻结常量。

### G1 executable profile：NO-GO

真实 W/H虽已有schema字段，但具体camera calibration、depth/extent edges、materialized tables/hash、static capability/resource claims、RIGHT/LEFT compound admission及runtime receipts仍未标定。因此必须继续报告：

```text
G1 executable profile = NO-GO
|G_profile(p,t)|       = UNDEFINED
profile bits           = UNDEFINED
```

结构数不能冒充部署reachable states。

## 5. 最终裁决

**最终文档裁决：Doc structural Freeze。**

上一轮六项 exact-contract 条件均已在书面层面得到唯一、可复算的规范答案，没有引入需要改字段、grammar或容量的新矛盾。CPU conformance仍待实现和实跑；真实 G1 executable profile继续 No-Go。这三个结论必须并列保留，互不替代。
