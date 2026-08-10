# Contact labels v1.1：从人工复核 CSV 到 22D sidecar

本目录只提供 Track A 的数据契约、空白双标模板、诊断 split 与 CPU 工具，不修改 GigaWorld Policy 的 model、trainer、dataset 或 transform。输出严格对齐 canonical commit `f3ae3bf` 的 `contact_segment_v1_1`：`[T,Q,22]`，默认 `Q=12`、horizon=48、20 Hz。

## 三层身份，禁止把窗口当事件

1. `segment_id`：一段真实物理接触区间的全局唯一 ID。
2. `onset_event_id` / `release_event_id`：仅当对应物理边界被观察到时创建；删失或未知边界不得伪造 event ID。
3. `(frame_index, query_index)`：同一 segment 在某个滑窗中的临时槽位，不是新事件。映射写入 `window_segment_provenance.csv`。

每个窗口按 `(global start_frame, global end_frame, side-or-empty, site-or-empty, global segment_id)` 全序排序后依次赋给 query 0..11。若一个窗口重叠超过 12 个物理 segment，builder 立即失败；不截断、不覆盖。metadata 分开报告 `unique_physical_segments` 与 `window_segment_rows`。

## Coverage 决定负样本语义

- `unknown`：没有完成可用复核，四个输出数组保持全零 mask；不能推断正例或负例。
- `positive_only`：只监督 CSV 中明确确认的 segment；空 query 不监督，因此不是负例。
- `exhaustive`：完整检查了覆盖区间，只有此时空 query 才能作为负例。无 contact 的 exhaustive interval 本身就是 exhaustive negative，不得写一行虚构 negative segment。

builder 只接受 `annotation_role=adjudicated`、`review_status=reviewed`、`label_origin=human_adjudicated`。prescreen、自动提示和 minimum proposal 都不能直接生成 GT。

## 字段级 unknown 与 censor

- 类别未知用空字符串，只有该 one-hot 字段 mask=0；不得连带清除已确认字段。
- observed 边界若落在窗口内，监督真实 tau 与 censor=0；若已落在窗口外，监督 tau=0/1 与对应 censor=1。
- coverage-censored 或 unknown onset：仅当 `current_frame > conservative_start` 时，才能证明 onset 已在窗口前，此时监督 `start_tau=0,left_censored=1`；等号无法区分“恰在窗口边界”与“此前”，所以等号及更早窗口的两字段均 mask0。
- coverage-censored 或 unknown release：仅当 `real_window_end < conservative_end` 时，才能证明 release 仍在窗口后，此时监督 `end_tau=1,right_censored=1`；等号无法区分“恰在边界”与“其后”，所以等号及结束更晚的窗口两字段均 mask0。
- CSV 的 `[start_frame,end_frame)` 是保守确认的正区间。任何边界 mask0 都不影响 presence、side、site、role、mode 的独立字段级监督。
- `left_censored/right_censored` 专指真实边界位于复核区间之外，不是“看不清”的同义词。

完整字段顺序和枚举见 `schema/csv_schema.json`。

## 数据划分限制

`episode_split_manifest.csv` 给出 seeds 6666、7777、8888 的 episode-level 70/15/15 划分，仅用于 loader、shape 与泄漏检查。当前 66 段缺少 subject/session/operator/source-object/target-object/liquid/batch 元数据，因此所有行均为 `ood_eligible=false`；禁止据此做 OOD 或 strategy claim。

补采必须记录：稳定 `subject_id`、`session_id`、`operator_id`、source/target 容器实例 ID、容器几何/材质、`liquid_id` 与黏度/着色、fill、采集 batch、相机标定版本、任务成功/泼洒/碰撞结果。未补齐前不允许用 episode 编号猜测这些字段。

`minimum_annotation_set.csv` 只是按 episode 长度四分层、seed 6666 选出的 12 段工程提案。episode 0 的已有 timing 文件只记作 `present_not_ground_truth`；A/B 必须独立观看原视频，不得看到该提示。

## 使用

只有在 A/B 独立标注和裁决完成后，才能运行：

```bash
cd /home/ykj/project/WM/gwp-agent-contact-labels
python scripts/contact_labels/build_contact_segment_sidecars_v1_1.py \
  --coverage-csv /path/to/coverage_adjudicated.csv \
  --segments-csv /path/to/segments_adjudicated.csv \
  --inventory-csv track_a/contact_labels_v1_1/episode_inventory.csv \
  --config track_a/contact_labels_v1_1/config/sidecar_v1_1.json \
  --output-dir /path/to/contact_segment_sidecars
```

换行位置如上；每个反斜杠后立即换行。

严格验证：

```bash
python scripts/contact_labels/validate_contact_segment_sidecars_v1_1.py \
  --sidecar-dir /path/to/contact_segment_sidecars \
  --inventory-csv track_a/contact_labels_v1_1/episode_inventory.csv \
  --coverage-csv /path/to/coverage_adjudicated.csv \
  --segments-csv /path/to/segments_adjudicated.csv \
  --config track_a/contact_labels_v1_1/config/sidecar_v1_1.json
```

validator 检查文件集合、dtype/shape、有限值、字段级 binary mask、one-hot、unknown 必须为零、负 query 不得带属性、tau 顺序及输入 SHA-256。输出 NPZ 只含当前 dataset/transform 消费的四个键：`contact_target`、`contact_loss_mask`、`contact_confidence`、`contact_condition_mask`。
