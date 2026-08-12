# SM-FET minimal-r1d codec + D0 packet integration 独立复审

- 日期：2026-08-12
- 被审 integration：`c85173af4971fa3b624fc451726297b534b122ae`
- 被审 worktree：`/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration`
- review 基线：canonical `e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- review worktree：`/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration-review`
- 方法：目标 worktree 只读；Git object/tree/blob/patch-id 审计；Python 3.13.9 stdlib、CPU only；精简 codec 与 packet 复测
- 严重度：**P0 = 0；P1 = 1（同一 provenance invariant 下两条记录）**
- 最终裁决：**integration No-Go**

## 0. 一句话裁决

R1D codec、D0 packet、两份 final review 的 source→integration patch 与最终 bytes 均无漂移；允许路径、formal packet、model/trainer/config、ACL、精简 CPU codec 和 packet source/assignment/self-test 也全部通过。阻断项仅在统一研究索引：索引明确把 `source commit` 定义为“产生所收录最终研究 bytes”的提交，但 64 个链接记录中有两条所列 SHA 不能重放 HEAD blob。因此本 integration 在修复 provenance 元数据前为 **No-Go**。

这个 No-Go 只裁定 integration 包装/索引准确性，不撤销已经通过的窄工程结论，也绝不能外推：

```text
minimal-r1d frozen structural codec/checker gates: CPU-conformant Freeze Go
D0 packet source/assignment/submission engineering: Engineering Go（依据已收录 final review）
G1 executable profile and profile states/bits:      NO-GO / UNDEFINED
真人 distribution / independent annotation / lock: NOT RUN
人类 D0 gate:                                       未通过
模型训练、GPU、仿真、真机部署:                       NO-GO / 未授权
```

## 1. 阻断项：index 的 final-byte provenance 不自洽

`c85173a:docs/research/WAM_BIOCHEMICAL_IDEA_INDEX_20260812.md:9` 规定：表中 `source commit` 是产生所收录**最终研究 bytes**的原始来源提交。机械扫描 index 全部 64 个相对 Markdown link occurrence，并对每行所有 40 字符 commit ref 执行 `commit:path` blob 对照，结果为：

```text
OK=62
MISMATCH=2
HEAD_MISSING=0
TOTAL=64
```

两处 mismatch 恰为：

| index 位置 | artifact | 行内所列 source | source blob | HEAD blob | 实际 final-byte commit | 差异 |
|---|---|---|---|---|---|---|
| line 47 | `sm_fet_wam_novelty_adversarial_review_20260812.md` | `260ac320565c2a156e0036a638358e790a784110` | `ed2b7397129f2f5ac9fff32320f24f35938b298e` | `500c0cb3e0afbabc563207473552fbd1223e8cac` | `264d257e35d95fc5c1d32ced7504fb70abdf5e4b` | 文首 5 行 Markdown hard break 从尾随双空格改为 `<br>`；文字、ID、结论不变 |
| line 57 | `sm_fet_finite_local_frame_design_20260812.md` | `39adb119b1e039736e5404150ed9f3fa1e10d358` | `743cb0abc3eb72a724dc6486c64c2e3c5d7c68fa` | `05698fe371f38ecb67a708720cd9b03cb068b1ce` | `6226748f0cc201ec3db3cc39a8de362336487c16` | 删除文末一个孤立 `-` 行 |

这不是研究正文语义错误，也不是 P0；但它使 index 自己声明的 exact-byte provenance 无法复现，故为 P1。关闭条件是让两行同时记录 origin 与实际 final-byte commit（或恢复到所列 source bytes），再复跑 64-link blob 对照。纯 index 修订无需重跑 410,146 项 codec 全套。

除这两条 provenance 记录外，index 的 48 个唯一相对链接均存在，56 个完整 commit ref 均可解析；R1D/D0 新增行的状态边界与 source bytes 均准确。

## 2. HEAD、ancestry 与 source-byte 追溯

### 2.1 HEAD 与 ancestry

- 被审 worktree 前后均 clean；HEAD 精确为 `c85173af4971fa3b624fc451726297b534b122ae`。
- target tree 为 `4c00527b8f8501ef17d7de6ca9c9f549e840f28d`，唯一 parent 为 `6eefc55921a9e63837b948226de73526eb00d233`。
- `e0ebc692...` 是 target 的 merge-base 和祖先；target 单线领先 53 commits，无 merge、无分叉。
- 本轮 R1D+D0 tail 以 `6226748f0cc201ec3db3cc39a8de362336487c16` 为已集成父段，至 target 共 10 commits。
- 原 source branch commits 不是 target ancestors；它们以无冲突 replay/cherry-pick 进入 integration。该拓扑本身正常，下面用稳定 patch-id 和最终 blob 证明来源。

### 2.2 九个 source→integration patch

| source | integration | stable patch-id 结果 |
|---|---|---|
| `dec1988465cf444f47859f5acb97a390d1f56d5a` | `8dff0667e0ade2b04db4d8b20df94390f5b3dfe1` | match |
| `aa13981e5959611ad39574dca381fdede48db343` | `b46448b5bd58f56b992164c4c6c4ff6cb0f8d558` | match |
| `2743cbdaec18e03a4c42f7444806584488837611` | `94a518b898788f782988f9e58a6e0b31c4a8e334` | match |
| `3082f05d30c4a685a9303c95ae0d62fc251be2f2` | `b0fdc2e045a9d838f5d5bbc8ac1ce2dcb8845524` | match |
| `9dba875af67e05d4e72a1011cb8a1ae63c22d09e` | `2f3ce8e42aeb43e05d9281aac8f41c3d6407fe00` | match |
| `c5e63590ec5535068db937ee46d0c064d25e6943` | `d8a40e4cea499dc632bcee4fe736daf109c8f9bb` | match |
| `b082680a74b1f0720c8b5e1023feeb8b39fd6cb9` | `7b8f17f6f99134a5d9edc82884446e0521874bfe` | match |
| `44bdee68c54cd816123ae6017d3e33412fd679d1` | `134971711c21ca6ed45913c38bf8567181a7f962` | match |
| `80ff5ef47bd25fa056c3b243a624e8991e5c33f9` | `6eefc55921a9e63837b948226de73526eb00d233` | match |

### 2.3 最终 source bytes

| artifact | source/target Git object | 结果 |
|---|---|---|
| R1D codec | blob `b81ecc3a1661182d5677033794202f1b74d1face` | `2743cbd` = target |
| codec conformance memo | blob `aa767351589e9af5eec16cd4ccc6e925f43371eb` | `2743cbd` = target |
| D0 checker（mode `100755`） | blob `8452a20f5be0b90acaf3033ec687da64374c86e5` | `b082680` = target |
| D0 packet 50-file subtree | tree `c67819909375dcb8977e79b0dc29f58278bc306e` | `b082680` = target |
| R1D final review | blob `66877b80f52f643b5636215e74970c4b79832a31` | `44bdee6` = target |
| D0 final review | blob `b04459beb5612027d5f7ad889e963bb9ac4e9b5a` | `80ff5ef` = target |

## 3. 允许路径与受保护树

相对 canonical `e0ebc692...`，target 是 `93 A / 0 M / 0 D / 0 R`：

- `docs/research/`：41 files；
- `track_a/sm_fet_d0_calibration_packet_v1/`：50 files；
- `scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py`：1 file；
- `world_action_model/sm_fet_minimal_r1d_schema.py`：1 file。

93 个路径全部落在上述 allowlist。相对本轮 tail 基点 `6226748...`，为 55 additions + 1 index modification，共 56 paths；同样无越界。最后的 `c85173a` 只修改统一 index。

受保护路径/树均零差异：

| 路径 | canonical/target tree | 结果 |
|---|---|---|
| `track_a/contact_labels_v1_1` | `41aaf372455d81e307439168c0b233c8c5f49730` | unchanged |
| `track_a/contact_labels_v1_1/formal_dual_annotation_v1_1` | `29ae213a60ae8d0b302fe9878d0bfceb3f962fe6` | unchanged |
| `world_action_model/models` | `6b9160da9a45c5c43df9c1213dc5a0916950a958` | unchanged |
| `world_action_model/trainer` | `7666b8499fa374aba967a727a18abf121e07d90e` | unchanged |
| `configs` | `8294bc979af6da36c25ab55a9739e1f528a88ad2` | unchanged |

`accelerate_configs`、`scripts/train.py` 及相关 third-party model/config 路径也为零 diff。注意不能笼统说整个 `world_action_model/` 零 diff，因为根目录新增了获准的独立 CPU schema codec。

canonical worktree `/home/ykj/project/WM/giga-world-policy` 的指针仍是 `e0ebc692...`，integration 没有改写 canonical Git tree。首次只读观察时该 canonical worktree 已有其它 modified/untracked 用户状态，因此本 memo 只证明 commit/tree/object 与指针未变，不声称该外部工作目录 clean，也未触碰其中内容。

## 4. 统一研究状态核对

除第 1 节的两条 provenance 元数据外，`c85173a` 对当前 gate 的陈述与最高优先级 review、代码输出和 packet 状态一致：

| index 陈述 | 复核结果 |
|---|---|
| minimal-r1d 主合同 + addendum 为 Doc structural Freeze | accurate |
| frozen structural codec/checker gates 为 CPU-conformant Freeze Go | accurate |
| 结构容量 `157,516,896,299,151,958,972,545,105,921`，`96.99142134161319 bit`，97-bit index，32-byte wire / 13-byte compact | accurate |
| 真实 G1 executable profile、states/bits 为 No-Go / UNDEFINED | accurate；没有 metric tables、真实 codebook/profile/runtime receipts |
| D0 packet source/assignment/submission/end-to-end 为 Engineering Go | 与 `80ff5ef` final review 一致；本轮重新验证 source/assignment/self-test |
| 真人 distribution、independent annotation、submission/lock、A/B comparison/discussion 为 NOT RUN | accurate；仓库只有 blank metadata packet/self-test fixture |
| D0 未通过；`43/50/58` 的 `1V/13P/4U` 只是 coordinator proposal | accurate |
| 训练、GPU/服务器训练、仿真、真机部署仍 No-Go | accurate；本 integration 没有 model/trainer/config 改动 |

因此不能把本 memo 中任何 PASS 写成 G1 profile Go、人类 D0 Go 或训练 Go。

## 5. stdlib CPU codec 精简复测

环境为 Linux x86_64、Python 3.13.9、standard library only、CPU only；未使用 conda、GPU、CUDA 或外部依赖。按要求没有完整重复历史 410,146 项套件。

主精简套件结果：

- **31,946 checks / 2,105 unique global ranks，PASS，0.62 s**；覆盖 NULL/MAX、24 个 header 端点、36 个 topology block 端点、固定 seed 2,048 ranks、packet/wire/compact/equal-bit 全链 inverse、UV/depth/extent 全 internal edges、512 axis centers、16,384 frame/roll exact ties、reserved/global/length/VERIFY-PATH 负例。
- 独立关键 edge probe：**489 checks，PASS**；覆盖 azimuth/elevation half-open 边界、vertical `0x1p-80` 等值与两侧、5 个代表 axis × 32 roll edges 的 below/exact/above。
- 第二次独立只读复测：codec smoke **6,454 checks / 537 ranks，PASS，0.26 s**；关键边界 **17,630 checks，PASS，0.15 s**。

冻结 geometry digest 重算一致：

```text
axis  d9d9319054b954fceb759dd6c9c348f588813176b50466e54950691f0d14a5de
frame 0e0e05bad6118cd3b597f895a6ebf997842208f99a1f3580e834f1edd80073f9
path  191a9bdbdb35e030de8b915a441a1bdfce8d0bf691132d95c1e359c5c38eb41a
```

模块继续显式返回 `g1_profile_states = UNDEFINED / NO-GO` 与 `g1_profile_bits = UNDEFINED / NO-GO`；这些字段未被测试通过状态改写。

## 6. D0 packet source / assignment / self-test

所有命令均设置 `PYTHONDONTWRITEBYTECODE=1`，未写测试脚本或临时副本：

| phase | 结果 |
|---|---|
| `source` | valid；3 tasks、2 recipients、6 annotation templates；约 0.03 s |
| `assignment` A→B | valid；A/B packet IDs 正确、6 blank files；约 0.03 s |
| `assignment` B→A | valid；反向 isolation/binding 同样通过；约 0.03 s |
| `self-test` | valid；no-event 与 reference 正例通过；7 个 lifecycle/FK/role/coverage 攻击和 legacy ontology 正确拒绝；约 0.03 s |

本轮没有真实 locked submission root、逐人外部媒体 root 或真人数据，因此没有把 self-test 写成真人 submission/distribution 证据，也没有宣称 D0 pass。index 中 submission/end-to-end **Engineering Go** 的来源是逐字节收录的 `80ff5ef` final review；本轮约定的机械复测范围是 source、assignment 与 self-test。

## 7. ACL checkout 说明

被审 worktree 当前 ACL 符合 assignment gate：

- A/B recipient root：各 `0700`；
- 每方 8 个 immutable files：`0400`；
- 每方 6 个 annotation CSV + 3 个 lifecycle files：`0600`。

Git tree 只能表达普通文件/可执行位，不能保存 `0700/0600/0400` 的完整权限语义。README 第 165–166、220–221 行准确要求每次 checkout/复制后恢复 ACL 并重新运行 assignment/distribution checker，并明确 Git mode 不是 ACL 证据。当前通过只证明这个被审 worktree 的实际 inode modes；不能把它外推为未来 checkout 自动安全。

## 8. 卫生与零副作用

- 被审 integration worktree 在全部命令前后均 clean；无 staged、unstaged 或 untracked 项。
- `__pycache__`、`.pyc/.pyo`、测试 cache：0。
- temp/test-script/backup pattern artifact：0；未写 `/tmp`，未遗留测试脚本。
- D0 packet 内 MP4/MOV/HDF5/NPZ/image：0。
- D0 packet 内 GT/adjudication/merged-label 命名 artifact：0；六张 annotation CSV 仍只有 header。
- D0 packet 内 symlink、重复 hardlink：0。
- integration diff 未增加 media、GT、cache 或临时路径；formal packet 保持原 tree。
- 未使用 GPU/CUDA，未改 driver、环境、模型、trainer 或 config。

## 9. 最终 gate

```text
P0: 0
P1: 1 provenance invariant（2 index records）
integration: No-Go
```

关闭该 P1 后，只需复核 index link/ref/blob 与 target cleanliness/allowlist；若修订只触及 index provenance，既有 codec 与 packet 工程结果不需要借此升级或降级，也无需完整重跑 410,146 项套件。

无论 integration 后续是否转为 Go，以下状态均不随之改变：

```text
G1 executable profile: No-Go / UNDEFINED
真人 D0:               Not Run / 未通过
训练与部署:             No-Go / 未授权
```
