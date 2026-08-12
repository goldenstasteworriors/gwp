# SM-FET D0 coordinator handoff

- 日期：2026-08-12
- 基线：`addb1d5c3672a5d6147ee891d5f046245e01952a`
- 范围：只交接真人校准的准备流程；**本 handoff 不执行分发、标注、讨论或 GT 生成**
- coordinator source bundle：`/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration/track_a/sm_fet_d0_calibration_packet_v1`
- checker：`/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration/scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py`

## 1. 开始前必须由用户/协调员补齐

必须先给出并记录以下信息；任一项缺失即停止：

| 项目 | A | B |
|---|---|---|
| 与真实自然人一一绑定、且互不相同的 `annotator_id` | 待提供 | 待提供 |
| 互不相同的 `annotation_session_id` | 待提供 | 待提供 |
| 独立 OS principal / 机构隔离身份 | 待提供 | 待提供 |
| owner-only packet root（17 files） | 待提供 | 待提供 |
| owner-only media root（3 MP4） | 待提供 | 待提供 |
| owner-only、初始为空的 state root | 待提供 | 待提供 |
| 数据控制者授权人、用途/伦理依据、开放与撤销时间 | 待提供 | 待提供 |

A/B 的 packet、media、state 六个 root 必须两两不相同、互不嵌套，A 无法访问 B，B 无法访问 A；不得使用云同步目录、symlink 或 hard link。root 精确 `0700`，assignment 阶段冻结输入精确 `0400`、可写文件精确 `0600`，媒体精确 `0400`。

当前 source bundle 中的 `d0cal-annotator-a`、`d0cal-annotator-b`、`d0cal-session-a-20260812`、`d0cal-session-b-20260812` 是冻结的工程 assignment 身份。协调员必须确认它们已经是与两名真人一一绑定的 campaign ID；如果实际 ID 不同，**不得手改** `identity.json`、`session.json` 或 receipt。因为这些值已被 `assignment_receipt.json`、`recipient_manifest.json`、`coordinator/assignment_index.json` 和 checker 常量绑定，必须先由 packet 维护者重新生成、重签并复核两套 blank assignment。本 handoff 不做该重发动作。

## 2. 唯一允许分发的媒体

只允许从数据控制者批准的原始 RGB MP4 复制以下三段，并在 recipient media root 使用中性文件名。coordinator 可在自己的受控日志中记录本地绝对源路径；**recipient packet、recipient receipt、文件名和标注表中不得出现绝对源路径或原 episode 源路径**。

| episode | recipient 文件名 | bytes | SHA-256 | raw frames |
|---:|---|---:|---|---:|
| 46 | `sm-fet-d0-cal-00.mp4` | 3,422,955 | `4e03396690a21bc81ce1a20904692cd5a477b312022a74fe13c62bff0902c692` | 324 |
| 65 | `sm-fet-d0-cal-01.mp4` | 3,823,238 | `061f0b24aee2396bde201cb0fbd884c16cca14ed35236af2b45b25631488cf2a` | 384 |
| 1 | `sm-fet-d0-cal-02.mp4` | 6,595,743 | `5582effa3a634b735a2441cc9087e3a52685192e80bf7d0c6225205885387fd0` | 599 |

两人得到相同媒体 bytes，但必须各自独立复制，inode 不得共享。帧时钟只用 `raw_video_frame_30fps`，coverage 是半开区间 `[0,324)`、`[0,384)`、`[0,599)`。

禁止分发或展示：processed HDF5、HDF5-derived timeline、depth/raw depth/completed depth、contact/event/timeline proposal、contact visualization、mask overlap、tracker、MANO/SMPL、NeRF/mesh、光流、`hand_status`、`hand_active`、grip proxy、prescreen、selection audit、旧标签/GT、另一方结果、裁决结果、模型/VLM/CAM/heatmap、DTW/OT/CAPOT/solver score/embedding。**AI 输出、OT 对齐、MANO、`hand_status` 均不是 GT，也不能用于补标签、消歧或升级为 human evidence。**

## 3. Coordinator 预检

以下代码块均按展示位置换行；反斜杠后必须立即换行。先在仓库根目录运行，成功结果必须同时满足退出码 `0` 和 JSON 中 `"status": "valid"`。

```bash
cd /home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration
test "$(git rev-parse HEAD)" = "addb1d5c3672a5d6147ee891d5f046245e01952a"
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase self-test
```

```bash
cd /home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase source \
  --source-root track_a/sm_fet_d0_calibration_packet_v1
```

预期 source 摘要是 3 tasks、2 recipients、6 blank annotation templates。还必须由获批媒体探测流程核对三份源 MP4 的 bytes、SHA-256、frames、duration/fps/尺寸/codec；stdlib source phase 不解析视频容器。

## 4. 复制与初始化

先把下列路径改成实际获批路径，再在**同一个 shell** 中执行后续代码块。前三个 `D0_MEDIA_SOURCE_*` 只属于 coordinator，不写入 recipient receipt。目标 root 必须尚不存在；命令会拒绝覆盖已有内容。不得把任何输出写到 `/tmp`。

```bash
export D0_REPO='/home/ykj/project/WM/gwp-agent-sm-fet-r1d-d0-integration'
export D0_PACKET_SOURCE="$D0_REPO/track_a/sm_fet_d0_calibration_packet_v1"
export D0_SCHEMA="$D0_PACKET_SOURCE/schema_manifest.json"
export D0_MEDIA_SOURCE_46='/replace/with/coordinator-approved/episode_000046.mp4'
export D0_MEDIA_SOURCE_65='/replace/with/coordinator-approved/episode_000065.mp4'
export D0_MEDIA_SOURCE_1='/replace/with/coordinator-approved/episode_000001.mp4'
export D0_A_BASE='/replace/with/annotator-a-private-root'
export D0_B_BASE='/replace/with/annotator-b-private-root'
export D0_A_PACKET="$D0_A_BASE/packet"
export D0_A_MEDIA="$D0_A_BASE/media"
export D0_A_STATE="$D0_A_BASE/state"
export D0_B_PACKET="$D0_B_BASE/packet"
export D0_B_MEDIA="$D0_B_BASE/media"
export D0_B_STATE="$D0_B_BASE/state"
```

先确认源媒体存在且 hash 正确；失败即停止，不得改 allowlist 迁就错误文件。

```bash
set -euo pipefail
test -f "$D0_MEDIA_SOURCE_46"
test -f "$D0_MEDIA_SOURCE_65"
test -f "$D0_MEDIA_SOURCE_1"
test "$(sha256sum "$D0_MEDIA_SOURCE_46" | cut -d ' ' -f 1)" = '4e03396690a21bc81ce1a20904692cd5a477b312022a74fe13c62bff0902c692'
test "$(sha256sum "$D0_MEDIA_SOURCE_65" | cut -d ' ' -f 1)" = '061f0b24aee2396bde201cb0fbd884c16cca14ed35236af2b45b25631488cf2a'
test "$(sha256sum "$D0_MEDIA_SOURCE_1" | cut -d ' ' -f 1)" = '5582effa3a634b735a2441cc9087e3a52685192e80bf7d0c6225205885387fd0'
```

初始化六个独立 root，并从两个 source assignment 分别复制 17-file packet。`state` 只初始化为空目录；当前仓库没有获批的 SM-FET D0 writer/UI 初始化器，不能拿现有 contact 7-file offline UI 代替。

```bash
set -euo pipefail
test ! -e "$D0_A_BASE"
test ! -e "$D0_B_BASE"
install -d -m 0700 "$D0_A_PACKET" "$D0_A_MEDIA" "$D0_A_STATE"
install -d -m 0700 "$D0_B_PACKET" "$D0_B_MEDIA" "$D0_B_STATE"
cp --reflink=never "$D0_PACKET_SOURCE/assignments/annotator_a/"* "$D0_A_PACKET/"
cp --reflink=never "$D0_PACKET_SOURCE/assignments/annotator_b/"* "$D0_B_PACKET/"
cp --reflink=never "$D0_MEDIA_SOURCE_46" "$D0_A_MEDIA/sm-fet-d0-cal-00.mp4"
cp --reflink=never "$D0_MEDIA_SOURCE_65" "$D0_A_MEDIA/sm-fet-d0-cal-01.mp4"
cp --reflink=never "$D0_MEDIA_SOURCE_1" "$D0_A_MEDIA/sm-fet-d0-cal-02.mp4"
cp --reflink=never "$D0_MEDIA_SOURCE_46" "$D0_B_MEDIA/sm-fet-d0-cal-00.mp4"
cp --reflink=never "$D0_MEDIA_SOURCE_65" "$D0_B_MEDIA/sm-fet-d0-cal-01.mp4"
cp --reflink=never "$D0_MEDIA_SOURCE_1" "$D0_B_MEDIA/sm-fet-d0-cal-02.mp4"
```

## 5. 权限与 distribution validate

下列 chmod 只落实 mode；协调员还必须通过独立 OS owner、机构 ACL 或等价隔离保证 A/B 互不可读。若使用不同 OS principal，应由管理员把 A 的三个 root 归 A、B 的三个 root 归 B，并由有审计权限的协调员运行 checker。不要用会把 group/mask 位改宽的 ACL 破坏 checker 要求的精确 mode。

```bash
set -euo pipefail
chmod 0700 "$D0_A_BASE" "$D0_A_PACKET" "$D0_A_MEDIA" "$D0_A_STATE"
chmod 0700 "$D0_B_BASE" "$D0_B_PACKET" "$D0_B_MEDIA" "$D0_B_STATE"
chmod 0400 "$D0_A_PACKET/ANNOTATION_HANDBOOK.md" "$D0_A_PACKET/TASK_ONTOLOGY.json" "$D0_A_PACKET/assignment_receipt.json" "$D0_A_PACKET/media_allowlist.csv" "$D0_A_PACKET/recipient_exact_allowlist.json" "$D0_A_PACKET/recipient_manifest.json" "$D0_A_PACKET/schema_manifest.json" "$D0_A_PACKET/task_manifest.csv"
chmod 0600 "$D0_A_PACKET/coverage.csv" "$D0_A_PACKET/event_edges_v1.csv" "$D0_A_PACKET/event_node_snapshots_v1.csv" "$D0_A_PACKET/events_v1.csv" "$D0_A_PACKET/identity.json" "$D0_A_PACKET/pose_keys_2d_v1.csv" "$D0_A_PACKET/segments.csv" "$D0_A_PACKET/session.json" "$D0_A_PACKET/submission_receipt.json"
chmod 0400 "$D0_B_PACKET/ANNOTATION_HANDBOOK.md" "$D0_B_PACKET/TASK_ONTOLOGY.json" "$D0_B_PACKET/assignment_receipt.json" "$D0_B_PACKET/media_allowlist.csv" "$D0_B_PACKET/recipient_exact_allowlist.json" "$D0_B_PACKET/recipient_manifest.json" "$D0_B_PACKET/schema_manifest.json" "$D0_B_PACKET/task_manifest.csv"
chmod 0600 "$D0_B_PACKET/coverage.csv" "$D0_B_PACKET/event_edges_v1.csv" "$D0_B_PACKET/event_node_snapshots_v1.csv" "$D0_B_PACKET/events_v1.csv" "$D0_B_PACKET/identity.json" "$D0_B_PACKET/pose_keys_2d_v1.csv" "$D0_B_PACKET/segments.csv" "$D0_B_PACKET/session.json" "$D0_B_PACKET/submission_receipt.json"
chmod 0400 "$D0_A_MEDIA/"*.mp4 "$D0_B_MEDIA/"*.mp4
```

先做 assignment 检查，再做包含媒体 bytes/hash、空 state、六 root 非嵌套与 inode 隔离的 distribution 检查：

```bash
cd "$D0_REPO"
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase assignment \
  --packet-root "$D0_A_PACKET" \
  --peer-packet-root "$D0_B_PACKET" \
  --schema-manifest "$D0_SCHEMA"
```

```bash
cd "$D0_REPO"
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase distribution \
  --packet-root "$D0_A_PACKET" \
  --peer-packet-root "$D0_B_PACKET" \
  --schema-manifest "$D0_SCHEMA" \
  --media-root "$D0_A_MEDIA" \
  --state-root "$D0_A_STATE" \
  --peer-media-root "$D0_B_MEDIA" \
  --peer-state-root "$D0_B_STATE"
```

成功：退出码 `0` 且单行 JSON 为 `status=valid`；distribution 还应含 `distribution_roots=valid`。失败：退出码 `1` 且 JSON 为 `status=invalid`，`error` 是首个 fail-closed 原因。任一失败都停止分发，修复根因后从 preflight 重跑；没有 warning-only、`--force` 或关检查的通道。

## 6. 每人操作与时间预算

A、B 各自执行完全相同、但彼此不可见的流程：

1. 原速完整观看 episode 46、65、1；之后才可慢速或逐 raw-video frame 回看。
2. 独立完成三段的 `coverage.csv`、`segments.csv`、`events_v1.csv`、`event_node_snapshots_v1.csv`、`event_edges_v1.csv`、`pose_keys_2d_v1.csv`。不得边讨论边填，不得看另一方 CSV。
3. 每个 episode 必须恰有一条全段 `EXHAUSTIVE`、`submitted` coverage。确认没有 event 时也要交 coverage，其余五表可为空；不得为了填满而猜值。
4. 所有 row 必须是 `label_origin=human_independent_calibration`，`source_proposal_id` 为空；D0 结果始终 `NON_GT`。
5. 完成三段后更新允许的 lifecycle 字段，由 coordinator 重签 `submission_receipt.json`，然后将全部 17 文件降为 `0400`，撤销写权限，再运行 submission checker。

总预算 **3.25–4.0 人时**：coordinator preflight/分发/收件约 0.5h；A、B 独立首轮各 0.75–1.0h；两方锁定后共同讨论约 0.75–1.0h；最多一次规范修订、freeze receipt 与 ACL 撤销约 0.5h。出现大量 unknown 时按上界，不为 timebox 猜值。

提交锁定命令必须在 lifecycle/annotation hashes 与 coordinator content-lock receipt 已正确写入后执行；过早执行会使未完成 packet 只读。

```bash
chmod 0400 "$D0_A_PACKET/"*
chmod 0400 "$D0_B_PACKET/"*
```

```bash
cd "$D0_REPO"
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase submission \
  --packet-root "$D0_A_PACKET" \
  --schema-manifest "$D0_SCHEMA"
```

```bash
cd "$D0_REPO"
python scripts/contact_labels/check_sm_fet_d0_calibration_packet_v1.py \
  --phase submission \
  --packet-root "$D0_B_PACKET" \
  --schema-manifest "$D0_SCHEMA"
```

submission 成功同样要求退出码 `0` 与 `status=valid`，并返回 `references_closed=true`；失败即保持隔离，不进入比较或讨论。checker 只验证结构、hash、ACL、lifecycle 和引用闭合，不判断语义正确性，也不表示 D0 已通过。

## 7. 锁定后的讨论与唯一一次升级

只有 A/B 三段全部通过 submission、17 文件均 `0400`、写权限撤销后，协调员才可比较结构性分歧并组织共同讨论。讨论只处理定义、反例、boundary、unknown/mask/censor/reference/overflow，不投票制造真值，不覆盖首轮 CSV，不输出 merged label 或 GT。

protocol/dictionary/task ontology 在 calibration 后合计最多一次版本升级。任何字段、topology、role、frame clock、header 或语义变化都必须升版本，保存旧/新 SHA-256 与 change log；不能静默改现有 packet。升级后必须重新生成并重签**全新 blank blind assignments**，再做 blind denylist 检查，确保其中没有 D0 README、selection rationale、讨论记录、calibration CSV/proposal 或绝对源路径。旧 D0 结果封存为 `NON_GT/EXCLUDED_FROM_D1`，不能回填 formal 7-file packet，也不能训练或调阈值。

## 8. 收件与返回物

协调员必须收回两个完整、互相独立、锁定的 17-file packet roots。每个 root 恰好包含：

1. `ANNOTATION_HANDBOOK.md`
2. `TASK_ONTOLOGY.json`
3. `assignment_receipt.json`
4. `coverage.csv`
5. `event_edges_v1.csv`
6. `event_node_snapshots_v1.csv`
7. `events_v1.csv`
8. `identity.json`
9. `media_allowlist.csv`
10. `pose_keys_2d_v1.csv`
11. `recipient_exact_allowlist.json`
12. `recipient_manifest.json`
13. `schema_manifest.json`
14. `segments.csv`
15. `session.json`
16. `submission_receipt.json`
17. `task_manifest.csv`

随 handback 返回并由 coordinator 留存：两份 assignment receipt、两份 terminal submission/content-lock receipt、两份 identity/session、每个 17-file root 的 exact file SHA-256 清单、checker stdout/exit status、授权人/recipient/开放/撤销时间、媒体 hash 核验、访问撤销与 working-copy 销毁确认。讨论后另留 disagreement/change log 与 protocol-freeze receipt；不得生成 adjudicated/GT receipt。

recipient receipt 只记录中性 recipient path、文件名、bytes、SHA-256、授权角色和时间，不记录 coordinator 的绝对媒体源路径、HDF5 路径或 selection rationale。D0 媒体 working copy 撤销并销毁后，最小化保留访问/销毁审计；在真实 A/B 分发、独立标注、锁定提交、讨论与 freeze gates 全部完成前，状态只能是 **D0 NOT RUN / NOT PASSED**。
