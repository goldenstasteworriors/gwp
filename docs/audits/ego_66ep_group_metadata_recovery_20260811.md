# Ego 66ep group metadata 恢复审计（2026-08-11）

## 结论

当前 66 个 episode **不能建立 production OOD split**。可直接证明的只有：同一
dataset/task、同一采集日期、同一相机 backend/标定参数，以及每段的采集时钟和
长度。`subject/session/operator/source object instance/target object instance/liquid/
collection batch` 均未写入源 HDF5、depth HDF5、processed HDF5 或 LeRobot 元数据。

以 30 秒“相邻 episode 录制结束到下一段开始”的阈值，可机械恢复 4 个
`diagnostic_continuity_block`。它只是一种时间连续性诊断，不等于真实 session 或
batch，也不能进入 production GT：

| 诊断块 | Episode | 前一块后的已记录空档 | 可用范围 |
|---|---:|---:|---|
| 0 | 0–8 | — | engineering train only |
| 1 | 9–40 | 71.638 s | engineering train only |
| 2 | 41–49 | 726.577 s | engineering validation only |
| 3 | 50–65 | 46.571 s | engineering test only |

因此可建立的边界是 `train=0–40, val=41–49, test=50–65`，但必须同时标记
`split_purpose=temporal_diagnostic_only`、`ood_eligible=false`、
`strategy_claim_allowed=false`。它只能检查晚期数据漂移或工程回归，不能支持跨人、
跨 session、跨容器或跨液体泛化。

机器可复验结果见
`docs/audits/ego_66ep_group_metadata_recovery_20260811.json`；未填伪值的补录文件见
`track_a/contact_labels_v1_1/templates/ego_group_metadata_supplement_unfilled.json`。
其中绝对路径只作 audit record，`source_evidence_sha256` 排除这些 mount-dependent
路径，因此同一数据完整复制到服务器后仍可得到相同 evidence identity。

## 逐字段 provenance 矩阵

这里的三个状态含义为：`observed` 是源文件直接记录；`recoverable` 是可以从已有
记录机械导出、但不一定能成为 production GT；`user-required` 是本地没有权威证据，
必须由实验记录或知情采集者补充。

| 字段 | 状态 | 当前证据 | 能否直接进 production group GT |
|---|---|---|---|
| task family | observed | 66 个源 HDF5 的 `dataset_name=7_4_pour_liquid_between_beakers`；LeRobot 全部 task 为 `pour liquid from one beaker into another beaker` | 否；只有一个 task |
| episode ID / length | observed | 源 HDF5 `episode_index,total_frames`；final HDF5 `action_eef`；LeRobot `episodes.jsonl` 三方一致 | 否；episode 不是 group |
| capture time/date | observed | `local_timestamps_ns` 与 `creation_time`；全部为 2026-07-04 | 否；日期不能替代 session |
| camera backend | observed | 全部 `camera_backend=orbbec` | 否；只有一个值 |
| camera calibration | observed | 66 份 depth HDF5 的内参/畸变/scale 规范哈希相同：`5d53d35ef2652adca40bdd28d261641183b7b489d36f49bd13a7890b63105580` | 否；没有 per-episode serial，且只有一个标定组 |
| source/target object class | observed | task 文本只证明两端类别均为 beaker | 否；不包含实例身份 |
| diagnostic continuity block | recoverable | 根据 HDF5 记录时钟和显式 30 s 阈值导出 0–8 / 9–40 / 41–49 / 50–65 | 否；`inferred_not_production_gt` |
| `subject_id` | user-required | 所有已审计 artifact 均无此字段 | 必须补录 |
| `session_id` | user-required | 大空档不能证明换 session；也不能证明未换 session | 必须补录 |
| `operator_id` | user-required | 文件 owner、Git author 都不能证明实际操作者 | 必须补录 |
| `source_object_instance_id` | user-required | task 只有 beaker 类别；不得靠视频外观给实例编号 | 必须补录 |
| `target_object_instance_id` | user-required | 同上 | 必须补录 |
| `liquid_id` | user-required | task 只有泛化词 `liquid`；颜色/外观不得当身份真值 | 必须补录 |
| `collection_batch_id` | user-required | episode 顺序、空档、mtime 均不能定义 batch | 必须补录 |
| geometry/material | user-required | `source/target_geometry_id`、`source/target_material` 均未记录；外观不能替代材质/实例 GT | 非当前 7 字段 gate，但投稿级 OOD benchmark 应补录 |
| viscosity/color/fill | user-required | `liquid_viscosity_mpas`、`liquid_coloring`、`initial_fill_fraction` 均未记录；RGB 颜色不等于化学身份或黏度 | 非当前 7 字段 gate，但液体 OOD 应补录 |
| success/spill/collision | user-required | 没有 outcome 字段或传感器确认 | 不用于输入 group；评测时需人工复核/传感器证据 |

## 已审计证据与不能推出的内容

- 66 个 source HDF5 的共同 root attrs 只有 body/hand 格式、相机路径/backend、
  `collection_interval_s`、`creation_time`、`dataset_name`、`episode_index` 和长度等；
  对象级 attrs 为空。
- 66 个 processed HDF5 只增加 navigation 派生量；66 个 downsample/final HDF5 只增加
  action、hand-status、相机匹配与处理参数。没有 group identity。
- LeRobot `info.json` 只有一个 train range、一个 task、20 Hz 与 feature schema；
  `episodes.jsonl` 只有 episode/length/task；`tasks.jsonl` 只有一个 task。
- `merge_report.json` 证明 66 段、20,013 processed frames 均转换成功，但不携带实验组
  信息。
- 采集代码 `data_collection/human_data/scripts/human_data_collection.py` 只写上述基础
  attrs，命令行也没有 subject/operator/object/liquid/batch 参数。转换代码只把统一的
  task instruction 写入 LeRobot。
- 项目 Orbbec 日志只含 USB/相机诊断。日志中的 2026-07-03 设备信息没有与 2026-07-04
  的 episode 建立不可否认的 per-episode 绑定，因此连相机 serial 都不提升为本数据 GT。
- Git 历史能证明采集/转换代码由谁提交，不能证明谁佩戴设备、谁操作实验或何时更换
  容器/液体。
- 文件 owner、mtime/birth time 只反映当前文件系统状态或拷贝/处理时间。本审计没有用
  它们填任何生产字段。
- 没有从视频外观、身体形态、液体颜色或轨迹聚类推断人物/实例/液体身份。

## 最少需要用户补充的信息

只需要两项回答，不必逐 episode 重复填写：

1. 请给出 **任一 group 字段发生变化时的 episode 边界**，并为每个连续范围提供这
   7 个 ID：`subject_id, session_id, operator_id, source_object_instance_id,
   target_object_instance_id, liquid_id, collection_batch_id`。若 0–65 全部相同，也请明确
   写“全部相同”并给出各 ID；不能只写 `same/unknown/default`。
2. 请说明每组映射的依据是 `capture_record`、`experiment_log` 还是
   `collector_attestation`，给出可审计的 `evidence_ref`、证明人 ID 和 UTC 时间。

重点请确认 8→9、40→41、49→50 三处大空档是否对应 subject/session/operator/容器/
液体/batch 的任何变化；“没有变化”也需要知情者明确证明，不能从时间连续性反推。

## 使用方式

重新生成只读审计和空白补录模板：

```bash
python scripts/contact_labels/recover_ego_group_metadata.py \
  --source-root /home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers \
  --processed-hdf5-root /home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/vla_processed/hdf5 \
  --final-hdf5-root /home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/vla_processed/final_hdf5 \
  --lerobot-root /home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers/vla_processed/lerobot/7_4_pour_liquid_between_beakers \
  --audit-output docs/audits/ego_66ep_group_metadata_recovery_20260811.json \
  --supplement-template-output track_a/contact_labels_v1_1/templates/ego_group_metadata_supplement_unfilled.json \
  --diagnostic-gap-seconds 30
```

这里每个反斜杠后都需要换行。只检查空模板结构（不会产生 production inventory）：

```bash
python scripts/contact_labels/validate_ego_group_metadata_supplement.py \
  --supplement track_a/contact_labels_v1_1/templates/ego_group_metadata_supplement_unfilled.json \
  --audit docs/audits/ego_66ep_group_metadata_recovery_20260811.json \
  --allow-incomplete-template
```

填完且提供 provenance/attestation 后，去掉 `--allow-incomplete-template` 并显式指定
`--inventory-output`。校验器会拒绝空值、`unknown/same/default/TBD` 等占位符、缺失
provenance、推断类 provenance、部分 attestation、episode 集不一致和 audit 绑定不一致。
