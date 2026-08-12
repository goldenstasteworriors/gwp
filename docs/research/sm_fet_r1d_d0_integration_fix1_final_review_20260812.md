# SM-FET minimal-r1d + D0 integration fix1 最终独立复审

- 日期：2026-08-12
- 被审 fix1：`addb1d5c3672a5d6147ee891d5f046245e01952a`
- fix1 parent：`c85173af4971fa3b624fc451726297b534b122ae`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 被审 worktree：`/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration`
- review worktree：`/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration-review`
- 范围：只复核原 provenance P1 的关闭和 codec/packet/protected-path 无回归
- 方法：目标 worktree 只读；严格 JSON、Git commit/tree/blob/SHA-256/ancestry 审计；Python 3.13.9 stdlib、CPU only 精简 smoke
- 严重度：**P0 = 0；P1 = 0**
- 最终裁决：**integration Go**

## 0. 最终裁决

fix1 精确关闭上一轮唯一 provenance P1：新增机器可读 provenance JSON，逐条覆盖 index 的 64 个相对 Markdown link occurrence；62 条保持 source/integrated exact bytes，原先两条 byte mismatch 被明确收窄为 integration-only normalization，并同时绑定 content source、normalization commit、source/integrated blob 与 integrated SHA-256。两条 normalization 的实际 byte diff 与文字说明一致，未发现语义改写。

fix1 只修改统一 index 并新增 provenance JSON；codec、packet、formal、model、trainer、config 均零 diff。精简 CPU codec 与 packet source/assignment/self-test 全部通过，目标 worktree 前后 clean。因此：

```text
original provenance P1: closed
new P0/P1:              0 / 0
integration:            Go
```

这个 integration Go 只表示本次文档/CPU codec/D0 packet 工程 artifact 可按 index 所述边界集成，绝不改变：

```text
minimal-r1d frozen structural codec/checker gates: CPU-conformant Freeze Go
D0 packet engineering:                                Engineering Go
G1 executable profile and profile states/bits:        NO-GO / UNDEFINED
真人 distribution / independent annotation / lock:   NOT RUN
人类 D0:                                               未通过
模型训练、GPU/server、仿真、真机部署:                  NO-GO / 未授权
```

## 1. HEAD 与 fix1 范围

- HEAD 精确为 `addb1d5c3672a5d6147ee891d5f046245e01952a`，tree 为 `3b74169402d1da21d30cc067e193612a11253a87`，唯一 parent 为 `c85173af...`。
- fix1 仅一个 commit、无 merge；被审 worktree 在全部命令前后均无 staged、unstaged 或 untracked 项。
- `c85173a..addb1d5` 只包含：

```text
M docs/research/WAM_BIOCHEMICAL_IDEA_INDEX_20260812.md
A docs/research/WAM_BIOCHEMICAL_IDEA_PROVENANCE_20260812.json
```

- index diff 为 7 行级改动；provenance JSON 为 1,186 行。allowlist 外路径为 0。
- fix1 index blob `abe33bfb5da97a04fe1fff588fec2b0adf739ee6`，SHA-256 `236504213806ba4693f81b9a2b330eb14a3a7e8d3c69fb96fa841ad51097fc60`。
- provenance JSON blob `a93fc127f9dfa9faa62a7b033669f873de25ff55`，54,503 bytes，SHA-256 `9f76eaa923fe887f77c48a3c6a66a896d1a1afcd73e443284365038f658e7381`。

## 2. provenance JSON：64/64

`WAM_BIOCHEMICAL_IDEA_PROVENANCE_20260812.json` 通过 UTF-8 strict JSON duplicate-key 检查，且其 bytes 等于 `ensure_ascii=false / indent=2 / sort_keys=true / terminal LF` 的 canonical pretty JSON。

独立解析当前 index 的全部相对 Markdown links，再逐条对照 JSON，结果：

```text
index link occurrences:       64
JSON artifact records:        64
occurrence id / line / ordinal / path alignment: 64 / 64
unique linked paths:          48
exact source-byte records:    62
normalized records:           2
missing HEAD paths:           0
strict duplicate JSON keys:   0
```

每一条记录都重新验证：

1. `content_source_commit:path == content_source_blob_oid`；
2. `integration_commit:path == integrated_blob_oid`；
3. `HEAD:path == integrated_blob_oid`；
4. HEAD artifact bytes 的 SHA-256 等于 `integrated_sha256`；
5. `integration_commit` 是当前 integration ancestry 中的真实提交；
6. `content_source_commit` 出现在对应 index 行；
7. 62 条 `exact_source_bytes=true` 均满足 source blob = integrated blob、`normalization.kind=none`；
8. 2 条 `exact_source_bytes=false` 恰好等于 normalization allowlist，均有非空 operations 且 `semantic_change=false`。

结果为 **64/64 PASS**。`coverage` 自述的 `64 / 48 / 62 / 2 / 0` 与独立计数完全一致；fix1 前后 index 的 64-link path 顺序不变。

第二份独立 in-memory stdlib validator 共执行 **1,306 checks / 192 source-integration-HEAD Git blob bindings，PASS，0.61 s**。manifest 当前没有随附 repository checker/self-check；本轮任务未要求新增 checker，且两套独立只读 validator 已闭合验收，因此不记缺陷。

`generated_against_commit=c85173a` 记录生成 fix1 时的 parent/baseline；64 个 `index_line_at_generation` 则精确对应同次生成的 fix1 最终 index。二者不矛盾：manifest 与 index 同在尚未提交的 worktree 中生成时不可能预先自引用未来 `addb1d5` commit hash，而且 64 个 linked artifact bytes 在 `c85173a..addb1d5` 间均未改变。仓库没有 schema 声称行号必须取自 `generated_against_commit` 的旧 index blob，因此不额外强加该语义；建议未来若升 schema，可把 baseline 与 generated-index snapshot 字段名写得更显式，但这不是本轮 P1。

## 3. 两条 normalization

### 3.1 novelty hardbreak normalization

```text
path:               docs/research/sm_fet_wam_novelty_adversarial_review_20260812.md
content source:     260ac320565c2a156e0036a638358e790a784110
source blob:        ed2b7397129f2f5ac9fff32320f24f35938b298e
normalization:      264d257e35d95fc5c1d32ced7504fb70abdf5e4b
integrated blob:    500c0cb3e0afbabc563207473552fbd1223e8cac
integrated SHA-256: a8223807b7afb035a8790860d4013b24391684e10b4c4b41440b6c933766dd13
```

实际 changed lines 恰为 3–7：每行仅把末尾两个空格加 LF 替换为 `<br>` 加 LF，共 5 处；去掉 hardbreak 表示后，文本、IDs、结论均相同。index 与 JSON 的 `markdown_hardbreak_to_html_br / count=5 / semantic_change=false` 描述准确。

### 3.2 finite-frame EOF normalization

```text
path:               docs/research/sm_fet_finite_local_frame_design_20260812.md
content source:     39adb119b1e039736e5404150ed9f3fa1e10d358
source blob:        743cb0abc3eb72a724dc6486c64c2e3c5d7c68fa
normalization:      6226748f0cc201ec3db3cc39a8de362336487c16
integrated blob:    05698fe371f38ecb67a708720cd9b03cb068b1ce
integrated SHA-256: 192cecd05ba36f64ca72ece03314641130ce566e07b4a0cdc562f907e74ebd5a
```

原 bytes 恰为 integrated bytes 再追加一个 LF，即文件尾从 `正文\n\n` 规范化为 `正文\n`；正文不变。index 与 JSON 的 `remove_eof_empty_line / count=1 / semantic_change=false` 描述准确。

normalization contract 只允许上述两条 path；其余 62 条强制 exact source bytes。原 P1 已被可机械重放的 content-source + normalization 双层来源闭合。

## 4. index 状态边界

fix1 只修改 `source commit` 定义、增加 provenance manifest 指针，并扩写上述两条 provenance/normalization。`## 当前统一状态` 至文件末尾相对 `c85173a` 逐字节不变，所有 gate 继续准确：

- CPU Go 仅限 minimal-r1d frozen structural codec/checker gates；
- G1 executable profile 仍为 **No-Go / undefined**；不存在真实 metric tables、codebook、resource/capability/runtime receipts；
- D0 packet 仅为 source/assignment/submission/end-to-end **Engineering Go**；
- 真人 distribution、independent annotation、submission/lock、A/B comparison/discussion 仍为 **Not Run**；D0 未通过；
- `43/50/58` 的 `1V/13P/4U` 仍只是 coordinator proposal；
- 实现/训练大 WAM、GPU/server 训练、仿真、真机部署和未经 gate 的收益 claim 均 **No-Go**。

没有因 integration Go 发生状态升级外推。

## 5. codec 与 packet 精简无回归复测

环境：Linux x86_64、Python 3.13.9、stdlib only、CPU only；未运行历史 410,146 项完整套件。

### 5.1 codec

- smoke：**31,943 checks / 2,105 unique ranks，PASS，0.64 s**；覆盖 NULL/MAX、24 header 端点、36 topology block 端点、固定 seed 2,048 ranks、packet/wire/compact/equal-bit inverse、全部 UV/depth/extent internal edges、512 axis centers、16,384 frame/roll exact ties和关键负例。
- edge probe：**489 checks，PASS，0.04 s**；覆盖 azimuth/elevation half-open、vertical `0x1p-80` 等值与两侧，以及 5 axis × 32 roll edges 的 below/exact/above。
- 第二次独立只读回归：smoke **6,454 checks / 537 ranks，PASS，0.26 s**；关键边界 **17,630 checks，PASS，0.15 s**。
- 冻结 geometry digest 均一致：

```text
axis  d9d9319054b954fceb759dd6c9c348f588813176b50466e54950691f0d14a5de
frame 0e0e05bad6118cd3b597f895a6ebf997842208f99a1f3580e834f1edd80073f9
path  191a9bdbdb35e030de8b915a441a1bdfce8d0bf691132d95c1e359c5c38eb41a
```

### 5.2 D0 packet

| phase | 结果 |
|---|---|
| source | valid；3 tasks、2 recipients、6 annotation templates；0.03 s |
| assignment A→B | valid；6 blank files、A/B packet IDs 正确；0.03 s |
| assignment B→A | valid；反向 isolation/binding 通过；0.03 s |
| self-test | valid；2 个正例通过；7 个 attack 与 legacy ontology 正确拒绝；0.03 s |

这些只证明 codec/packet 工程无回归；self-test 不是人类 distribution/submission/D0 证据。

当前 A/B recipient roots 仍各为 `0700`，合计 16 个 immutable files 为 `0400`、18 个 annotation/lifecycle files 为 `0600`，source/assignment checker 通过。Git index 只保存普通文件位，README 关于 checkout 后恢复 ACL 并重跑 checker 的说明继续准确。

## 6. protected zero diff 与卫生

fix1 parent `c85173a` 与 HEAD 的以下对象完全相同：

| artifact / tree | Git object |
|---|---|
| R1D codec | `b81ecc3a1661182d5677033794202f1b74d1face` |
| D0 checker | `8452a20f5be0b90acaf3033ec687da64374c86e5` |
| D0 packet subtree | `c67819909375dcb8977e79b0dc29f58278bc306e` |
| `track_a/contact_labels_v1_1` | `41aaf372455d81e307439168c0b233c8c5f49730` |
| formal subtree | `29ae213a60ae8d0b302fe9878d0bfceb3f962fe6` |
| `world_action_model/models` | `6b9160da9a45c5c43df9c1213dc5a0916950a958` |
| `world_action_model/trainer` | `7666b8499fa374aba967a727a18abf121e07d90e` |
| `configs` | `8294bc979af6da36c25ab55a9739e1f528a88ad2` |
| `accelerate_configs` | `24426230ab0b2dc54ff207f0ab422b3cabef6e4b` |
| `scripts/train.py` | `702e77058f6e28d8217abcb8e2f7b7b9294988fe` |

相关 third-party model/train/config 也零 diff。相对 canonical，既有 tracked bytes 没有修改、删除、重命名、type 或 mode 改写。

测试前后 integration clean；没有 `__pycache__`、`.pyc/.pyo`、test cache、临时/备份文件或测试脚本；没有写 `/tmp`。D0 packet 内没有媒体/HDF5/图片、GT/adjudication/merged-label artifact、symlink 或重复 hardlink。未使用 GPU/CUDA，未改环境、driver、model、trainer 或 config。

## 7. 最终 gate

```text
P0: 0
P1: 0
integration fix1: Go

G1 executable profile: No-Go / UNDEFINED
真人 D0:               Not Run / 未通过
训练与部署:             No-Go / 未授权
```
