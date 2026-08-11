# SM-FET D0 三段 Ego pour 只读可标性 / 可观测性审计

- 日期：2026-08-12
- 工作树：`codex/sm-fet-d0-observability-20260812`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 输出性质：**proposal-only observability memo；不是 annotation、不是 GT、不是 D0/D1 gate 结果**
- 视频源：`/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers`
- 正式 packet：`../gwp-agent-op17-formal-label-packet/track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/`
- 审计对象：正式 minimum set 中 episode `43 / 50 / 58`
- 资源边界：CPU-only；未训练、未调用 GPU、未改视频/HDF5/packet、未生成任何 GT

## 0. 结论先行

保留 `43 / 50 / 58` 作为本次三段可观测性校准样本，不调整 episode：

1. **episode 43：镜像布局 / 可见右手样本。** source 在画面右侧、target 在左侧；可见操作手的形态支持 `RIGHT` proposal。source/target 的紫色液体前后状态变化清楚，倾倒时杯口和指尖有中等遮挡。
2. **episode 50：遮挡困难样本。** source 在左、target 在右；可见操作手支持 `LEFT` proposal。约 processed `72–116` 帧内，手、腕部设备和 source 同时遮挡 target rim / source opening；只能从前后时序确认粗粒度 transfer，不能看见可靠 stream、flow rate 或 contact interface。
3. **episode 58：普通长上下文 / 可见液体差异样本。** source 在左、target 在右；可见操作手支持 `LEFT` proposal。片段更长，接近、持杯、回正、放杯和事后静止上下文较完整；紫色内容从 source 转到 target 的前后证据清楚，但倾倒中段仍被手遮挡。

三段共同支持的只是“这些字段是否值得进入正式双标”的**提案级证据**：source/target role、粗 2D node、六类候选 event、一个可见操作手、hand→source 与 source/target relation 两种 edge，以及 RGB-temporal 的粗粒度 `TRANSFER`。它们共同不支持：

- 第二只手 `NOT_INVOLVED` 或完整单手 roster；另一手全程画外，不能从“没看到”推出 absent；
- stream、flow rate、volume、spill fraction、透明 contact interface 或 3D `INSIDE/TOUCHING` 真值；
- source/target 圆柱杯的稳定 `orientation_2d`、未冻结阈值下的 `scale_band`、被手遮挡的 part-local exact cell；
- `REGRASP`、`VERIFY_OUTCOME`、双手/ambiguous-side、正例 censor、overflow 等未出现类别的可标性；
- task success / partial / failure。三段只有“可见内容发生转移”的证据，没有冻结 outcome 判据或传感器确认。

因此：

- 本 memo **不生成 GT，也不使任何 packet 成为 oracle-admissible**；
- D0 仍缺 annotation→82-code→annotation checker/round-trip、grammar/capacity receipt；**D0 未通过**；
- 没有 A/B 独立标注、agreement、第三人裁决和 12 blind 结果；**不得宣称 D1 通过**；
- 43/50/58 属于 formal 12，本次“校准”只表示只读 audit sample。它们若将来用于正式 calibration，必须与 blind set 的 episode 集重新按 authoritative protocol 冻结，不能同时计入 blind agreement。

## 1. 规范与证据边界

### 1.1 本次读取的上位文档

以集成索引的裁决优先级为准，读取：

- `../gwp-agent-research-docs-integration/docs/research/WAM_BIOCHEMICAL_IDEA_INDEX_20260812.md`；
- `sm_fet_wam_v2_schema_architecture_20260812.md`：4 role nodes、active+next2、每 event 2 edges、82 `uint8` / 13 tokens、standard-r1 grammar；
- `sm_fet_annotation_and_pairing_protocol_20260812.md`：八类 event、entry/turn/exit pose probe、UNKNOWN/ABSENT/MISSING/censor、3+12 gate；
- `sm_fet_dynamic_focus_token_spec_20260812.md`：`focus_node + node geometry + primary part/cell + secondary relation` 的确定性 focus handle；
- `sm_fet_missing_hands_contract_20260812.md` 与 `sm_fet_missingness_capacity_reconciliation_20260812.md`：per-side、field-level、r1 actor 四码、unresolved actor fail-closed；
- `sm_fet_transparent_nonreconstruction_contract_20260812.md`：只允许可审计的 2D/关系/效果证据，禁止把重建、伪 depth、不可见液体状态当 GT。

本 memo 采用的唯一 action wire 字段类型是：

```text
node:  observability, center_cell, scale_band, orientation_2d
event: kind, focus_node, onset, duration, object_effect, outcome
edge:  actor, primary_node, primary_part, primary_surface_cell,
       secondary_node, relation_before, relation_after, contact_mode
```

### 1.2 formal packet 身份与时间轴

formal minimum set 固定为：

```text
43, 50, 51, 58, 34, 38, 9, 54, 36, 53, 12, 0
```

packet README 明确：这 12 段是空白正式任务，不是 GT；A/B 不得看到 timing hint、prescreen、hand status、action saliency 或模型 proposal。本审计遵守相同证据边界：

- 只看原始 MP4 与 formal packet 的 coordinator manifest / timeline mapping；
- **没有读取或使用**现有 `hand_status`、`hand_active`、contact visualization、contact annotation、tracker/MANO/VLM/算法输出；
- 没有把任务名、杯子运动或“按脚本完成”直接提升为 physical contact / success；
- 所有下述 event、edge、side 和 frame 区间均加 `proposal` 限定词；不能被复制为 human-independent row。

本文中的 `p[a,b]` 指 processed 20 Hz 的**只读证据查看窗口**，不是 onset/offset label，也不是半开 GT interval。窗口有意重叠，以保留 transition uncertainty；遮挡只产生 partial/unknown，不产生 censor。`v` 表示原视频 30 Hz 时间。

### 1.3 媒体完整性与 CPU 探测

使用现有 `decoupled_vla_collection` 环境中的 `ffprobe` 绝对路径只读探测；没有激活/修改 conda 环境，也没有安装依赖。

| episode | packet task | processed frames | MP4 SHA-256 | video | ffprobe |
|---:|---|---:|---|---|---|
| 43 | `contact-min12-00` | 264 | `36bee62146e5351a096e06bb489cdc941c70af92e2d58bf1be6eee0fb91b94e3` | 400 frames, 13.334 s | MPEG-4, 640×480, 30 fps, yuv420p |
| 50 | `contact-min12-01` | 223 | `ace55b2d1a563141c4afc6765c5c528a31946d4554a6fed715efb280a898a618` | 338 frames, 11.267 s | MPEG-4, 640×480, 30 fps, yuv420p |
| 58 | `contact-min12-03` | 294 | `92e9b10dfc7bb86b67021b44fdd026ea1f6fe4a3f06280c110580bacafa1180a` | 444 frames, 14.800 s | MPEG-4, 640×480, 30 fps, yuv420p |

三个本地 SHA-256 均与 `coordinator/source_episode_manifest.csv` 一致。manifest 报告三段最大 processed↔video sync error 分别为 `20.562 / 23.366 / 25.731 ms`，均小于 packet 50 ms 上限。

审计时在本工作树的 `.d0_audit_scratch/` 中生成过只读 contact sheet / keyframe 临时图，只用于人工查看；它们在 commit 前全部删除，没有媒体、截图或衍生视觉资产进入 git。

## 2. proposal-only task binding

以下 binding 只用于把可观测性问题具体化，**不是冻结 ontology**：

| role slot | 本次 proposal binding | action admission 边界 |
|---|---|---|
| `manipulated_or_source` | 初始装紫色液体、由手拿起并最终变空的透明杯 | 三段都有正例；倾倒 turn 时 PARTIAL |
| `tool` | 没有独立 pipette/stirrer/funnel | 按 O2RE pour profile 候选为 ABSENT；本样本不能评估 tool-present |
| `receptacle_or_destination` | 初始空、最后出现紫色内容的透明杯 | 三段都有正例；倾倒 turn 时 rim/内容局部遮挡 |
| `fixture_or_support` | 桌面物理可见，但没有被本次 event focus/edge 引用 | 物理“看得见”不等于 packet present；reference union 中应为 ABSENT proposal，不能评估 fixture-present |

候选 focus：`approach/precontact/acquire/hold/release -> source`，`manipulate -> destination`。这遵循 dynamic-focus 文档的示例，但仍需正式 task ontology 和 A/B 复核；“画面最显眼的杯子”不能决定 focus。

倾倒 `manipulate` 的两条候选 edge 是：

1. 可见手 → source `BODY`，`STICK` proposal；
2. source/target 对象关系 edge，target `OPENING/RIM_EDGE` 与 source 为 secondary，contact mode `NONE` proposal。

第二条只表达对象关系；液流经过杯口不产生 liquid-rim contact edge。RGB 可支持 `NEAR/OVERLAP_2D`，但在杯口被挡时不能安全提升为 3D `TOUCHING/INSIDE/CONTAINS`。`ALIGNED_TO_RIM` 需要冻结 image-plane predicate 和双方 rim 的可定位支持，本次只能记 partial proposal。

## 3. 统一图例

| 标记 | 含义 |
|---|---|
| **V 可见** | 当前原视频直接或通过相邻帧可检查该字段的局部可标性；仍不是 GT |
| **P 部分可见** | 只支持某些时段/值，边界、cell、物理关系或 roster 不完整；正式标注预计需要 interval / REVIEWED_UNKNOWN / mask |
| **U 不可见/不可评估** | 三段没有正例支持，或当前证据不足以给 action code；不能用 task 常识补齐 |
| **未见** | review 中没有看见该 event；由于本 memo 不是 exhaustive A/B annotation，不能解释为 supervised ABSENT |

## 4. Episode 43 observability matrix

### 4.1 只读概况

候选路径为：

```text
APPROACH -> PRECONTACT -> ACQUIRE -> HOLD -> MANIPULATE -> RELEASE
```

source 初始在画面右侧、target 在左侧。可见手从右上进入；手掌/拇指拓扑在 entry 与 exit 支持 `RIGHT` proposal。`p90–108` 左右开始看到紫色内容出现在 target；直接 stream 仍不可定位。另一只手全程画外。

### 4.2 八类 event + pose/focus/effect 矩阵

| event kind | 可观测性 | proposal evidence window | entry / turn / exit 2D pose | focus proposal | effect / outcome | 主要歧义 |
|---|---|---|---|---|---|---|
| `APPROACH` | V | `p24–42` (`v1.2–2.1s`) | entry 手腕、掌心、五指大部可见；接近 source | source | `NONE`; outcome N/A | “持续任务定向接近”起点可在约 4–8 processed 帧内移动 |
| `PRECONTACT` | P | `p36–47` (`v1.8–2.35s`) | 指尖靠近杯壁，仍有无接触帧 | source | `NONE`; outcome N/A | source 透明、指尖/杯缘投影重叠；不能从 overlap 定物理接触 |
| `ACQUIRE` | P | `p42–55` (`v2.1–2.75s`) | entry→exit 手指闭合；部分指尖被杯壁/杯内容挡住 | source | `NONE` 或短 `TRANSLATE`; outcome N/A | 首次 contact 与“稳定可控 grasp”分界是 interval；不能用 grip proxy补 |
| `HOLD` | V | `p52–72` (`v2.6–3.6s`) | source 已稳定持握并向 target 对准 | source | `TRANSLATE/NONE` 需 ontology 冻结；outcome N/A | move-to-align 是否仍是 HOLD 或主 MANIPULATE 起点存在语义分界 |
| `MANIPULATE` | P | `p72–136` (`v3.6–6.8s`) | turn 在 `p78–112`；腕/掌可见，多个指尖与 source/target rims 被挡 | destination | 粗 `TRANSFER` 有 RGB-temporal 支持；`FLOW`、体积与 outcome U | source rotate 与 transfer 同时发生，但 wire 只有一个主 effect；stream/contact interface不可见 |
| `REGRASP` | U/未见 | 全片复核未见明确接触拓扑重分配 | 无独立 turn key 支持 regrasp | — | — | 腕姿变化不能自动算 regrasp；未见不等于 GT ABSENT |
| `RELEASE` | P | `p136–187` (`v6.8–9.35s`) | exit `p176–187` 手指张开、离杯；腕部靠画面边缘 | source | `NONE/TRANSLATE`; outcome 强制 N/A | r1 不允许 `MANIPULATE->HOLD`，故回正/放回整体更适合作长 release proposal；真实 contact end仍是 interval |
| `VERIFY_OUTCOME` | U/未见 | `p187–263` 仅为事后静止画面 | 无 active-hand pose要求 | destination（若未来建立） | target 中紫色内容可见，但 SUCCESS/PARTIAL/FAILURE U | 静止不自动等于“执行进入检查”；无冻结 volume/spill/success 判据 |

### 4.3 四 role nodes、两条 edge 与 missingness

| 项目 | 可观测性 | 证据窗口 | proposal-only 判断 / 主要歧义 |
|---|---|---|---|
| source node | V→P→V | `p0–72 / 72–136 / 136–263` | center 可粗量化；turn 时部分遮挡；圆杯无可靠方向标记，orientation U；scale code无冻结阈值 |
| tool node | U | 全片 | 无独立工具；只能测试 pour-profile absent proposal，不能测试 tool-present geometry/focus |
| destination node | V→P→V | `p0–78 / 78–124 / 124–263` | center 与可见/partial可审计；rim在关键 turn 被手/source挡；orientation U |
| fixture/support node | U（action-positive） | 全片 | 桌面可见但未被引用；不能把背景可见性变成 present node |
| edge 0：手→source | P | `p42–187` | actor `RIGHT` proposal；primary source/BODY；STICK 有时序支持；精确 contact part-local cell被手覆盖，宜 REVIEWED_UNKNOWN |
| edge 1：source↔target | P | `p68–142` | primary target OPENING/RIM_EDGE proposal、secondary source；NEAR/OVERLAP_2D可见；ALIGNED_TO_RIM与 exact cell 不稳 |
| 双手 side / roster | P→U | visible hand `p24–187`；另一手全片不可见 | 可见手 proposal 为 RIGHT；另一手只能记 OUT_OF_FRAME/NOT_REVIEWED，不能记 NOT_INVOLVED。因此 left/right-only完整 packet inadmissible |
| censor / unknown | V（区分能力） | 全 episode coverage | 片头/片尾有空白上下文，候选事件不需要 left/right censor；关键边界应 interval/unknown。遮挡不是 censor |

## 5. Episode 50 observability matrix

### 5.1 只读概况

候选路径同 episode 43。source 初始在画面左、target 在右；可见手从左上进入，entry/exit 的手掌/拇指拓扑支持 `LEFT` proposal。`p72–116` 是三段中最强的 rim/hand 遮挡：紫色内容在 target 中的出现可由前后帧确认，但 source opening、target rim、多个指尖和潜在 stream 同时不可见。

### 5.2 八类 event + pose/focus/effect 矩阵

| event kind | 可观测性 | proposal evidence window | entry / turn / exit 2D pose | focus proposal | effect / outcome | 主要歧义 |
|---|---|---|---|---|---|---|
| `APPROACH` | V | `p28–40` (`v1.4–2.0s`) | entry 手腕/掌心/五指大部可见 | source | `NONE`; outcome N/A | approach 起点只有时序目的性，单帧不能决定 |
| `PRECONTACT` | P | `p36–45` (`v1.8–2.25s`) | 指尖与 source rim/body逐渐接近 | source | `NONE`; outcome N/A | 透明杯边缘和指尖重叠；首次接触不可单帧精确 |
| `ACQUIRE` | P | `p40–50` (`v2.0–2.5s`) | 手指闭合，杯体开始随手移动 | source | `NONE/TRANSLATE`; outcome N/A | contact onset、stable grasp和 lift onset相互接近 |
| `HOLD` | V | `p48–68` (`v2.4–3.4s`) | 稳定持杯并移动到 target 上方 | source | `TRANSLATE/NONE`; outcome N/A | 与 manipulate 的转折依赖“主 effect何时开始” |
| `MANIPULATE` | P（遮挡困难） | `p68–120` (`v3.4–6.0s`) | turn `p72–112`；腕/掌可见，指尖、两杯口和潜在液流大面积遮挡 | destination | target 紫色内容前后变化支持粗 `TRANSFER`; `FLOW`/volume/outcome U | 本段不能提供 direct stream；`ALIGNED_TO_RIM`和 surface cell最容易变 UNKNOWN |
| `REGRASP` | U/未见 | 全片 | 无独立 regrasp pose支持 | — | — | 腕转动是倾倒动作，不足以证明 contact topology改变 |
| `RELEASE` | P | `p120–168` (`v6.0–8.4s`) | exit `p156–168` 手张开并离开空 source | source | `NONE/TRANSLATE`; outcome N/A | 真正 separation在少数帧内，但透明杯边界使 exact frame仍可能有分歧 |
| `VERIFY_OUTCOME` | U/未见 | `p168–222` 仅静止结果 | 无 active-hand pose | destination（若建立） | visible transfer result；verify outcome U | 没有可观察的主动检查行为或传感器判据 |

### 5.3 四 role nodes、两条 edge 与 missingness

| 项目 | 可观测性 | 证据窗口 | proposal-only 判断 / 主要歧义 |
|---|---|---|---|
| source node | V→P→V | `p0–68 / 68–120 / 120–222` | center可粗量化；turn时 source轮廓/开口被手遮挡；orientation U、scale code未冻结 |
| tool node | U | 全片 | 没有独立工具正例 |
| destination node | V→P→V | `p0–72 / 72–116 / 116–222` | target center仍多时可见，但 rim/内容变化起点被挡；不能给透明3D状态 |
| fixture/support node | U（action-positive） | 全片 | 桌面不在 reference union proposal；未覆盖 fixture-present |
| edge 0：手→source | P | `p40–168` | actor `LEFT` proposal；source/BODY + STICK；exact part-local cell U |
| edge 1：source↔target | P（最难） | `p64–124` | target OPENING/RIM_EDGE与source关系的粗2D证据；critical interval内 exact rim/cell/relation被腕/手遮挡 |
| 双手 side / roster | P→U | visible hand `p28–168`；另一手全片不可见 | 可见手 proposal LEFT；另一手不能确认 not involved，single-left oracle packet inadmissible |
| censor / unknown | V（区分能力） | 全片 | 无 clip censor；遮挡区应使用 field-level unknown / interval，而不是 right/left censor |

## 6. Episode 58 observability matrix

### 6.1 只读概况

候选路径同前两段。片段有较长的 pre/post idle 和放杯上下文，source 左、target 右；可见手支持 `LEFT` proposal。紫色内容约在 `p100–116` 出现在 target，`p136` 后 source 已回正并移回。相比 episode 50，前后状态更容易复核，但 stream 和 contact interface仍不可见。

### 6.2 八类 event + pose/focus/effect 矩阵

| event kind | 可观测性 | proposal evidence window | entry / turn / exit 2D pose | focus proposal | effect / outcome | 主要歧义 |
|---|---|---|---|---|---|---|
| `APPROACH` | V | `p36–50` (`v1.8–2.5s`) | entry 手腕/掌心/五指大部可见 | source | `NONE`; outcome N/A | 首次任务定向靠近的起点为interval |
| `PRECONTACT` | P | `p46–57` (`v2.3–2.85s`) | 指尖接近source body/rim | source | `NONE`; outcome N/A | overlap不能证明 contact |
| `ACQUIRE` | P | `p52–64` (`v2.6–3.2s`) | 手指闭合，source从桌面开始随手运动 | source | `NONE/TRANSLATE`; outcome N/A | 稳定 grasp 时点仍需逐帧双标 |
| `HOLD` | V | `p62–88` (`v3.1–4.4s`) | 持杯移动并对准target | source | `TRANSLATE/NONE`; outcome N/A | 与倾倒turn之间边界可做interval |
| `MANIPULATE` | P | `p88–136` (`v4.4–6.8s`) | turn `p92–124`；腕/掌可见，指尖和rims部分遮挡 | destination | 粗 `TRANSFER` 有时序证据；FLOW/volume/outcome U | source旋转与transfer并发；单effect code需ontology裁决 |
| `REGRASP` | U/未见 | 全片 | 无独立regrasp key | — | — | 没有可确认的接触拓扑解除/迁移/重建 |
| `RELEASE` | P | `p136–191` (`v6.8–9.55s`) | exit `p180–191`，手指张开并离开空source | source | `NONE/TRANSLATE`; outcome N/A | release起点可早于实际separation；长release与MANIPULATE边界需protocol固定 |
| `VERIFY_OUTCOME` | U/未见 | `p191–293` 仅长静止结果 | 无 active-hand pose | destination（若建立） | target 中紫色内容明显；success/partial/failure U | 较长静止仍不等于 active verification；无spill/volume阈值 |

### 6.3 四 role nodes、两条 edge 与 missingness

| 项目 | 可观测性 | 证据窗口 | proposal-only 判断 / 主要歧义 |
|---|---|---|---|
| source node | V→P→V | `p0–88 / 88–136 / 136–293` | center与visibility容易复核；turn时部分遮挡；圆杯orientation U，scale code未冻结 |
| tool node | U | 全片 | 无 tool-present 正例 |
| destination node | V→P→V | `p0–92 / 92–128 / 128–293` | 最终紫色内容清楚；rim在倾倒中部分被挡；不能推液位/体积真值 |
| fixture/support node | U（action-positive） | 全片 | 桌面物理可见但不在当前reference union |
| edge 0：手→source | P | `p52–191` | actor `LEFT` proposal；source/BODY + STICK；contact patch/cell大多被手挡 |
| edge 1：source↔target | P | `p84–142` | 2D NEAR/OVERLAP变化可见；ALIGNED_TO_RIM和exact part-local cell只部分可见 |
| 双手 side / roster | P→U | visible hand `p36–191`；另一手全片不可见 | 可见手 proposal LEFT；另一手 OUT_OF_FRAME/NOT_REVIEWED，不能给 exhaustive left-only |
| censor / unknown | V（区分能力） | 全片 | 首尾均有空白，候选事件无censor；turn遮挡属于unknown/partial，不是censor |

## 7. 三段 action-visible 字段可评估性统计

统计单位是 schema 的 **18 种字段类型**（4 node + 6 event + 8 edge），不是 82 个重复 wire slots。这里的“可评估”仅指三段中是否存在足以开展正式可标性试验的局部证据；不等于字段已标、agreement已过或可编 oracle packet。

| action-visible 字段类型 | ep43 | ep50 | ep58 | 三段综合 | 能评估什么 / 不能评估什么 |
|---|---|---|---|---|---|
| node `observability` | P | P | P | **部分可评估** | source/target 的 VISIBLE↔PARTIAL；不覆盖OUT_OF_VIEW/OCCLUDED_TRACKABLE等完整词表 |
| node `center_cell` | V | V | V | **可局部评估** | 可检查source/target 4×4 image cell；跨cell支持时仍应UNKNOWN |
| node `scale_band` | U | U | U | **不可 code-level 评估** | SMALL/MEDIUM/LARGE阈值尚未由train/calibration冻结 |
| node `orientation_2d` | U | U | U | **不可可靠评估** | 两个圆柱透明杯缺少稳定方向标记，8-bin方向基本不可辨 |
| event `kind` | P | P | P | **部分可评估** | 仅支持approach/precontact/acquire/hold/manipulate/release；regrasp/verify未覆盖 |
| event `focus_node` | P | P | P | **部分可评估** | source/destination角色可讨论；依赖尚未正式冻结的task ontology，不覆盖tool/fixture focus |
| event `onset_code` | P | P | P | **部分可评估** | 所有候选event在clip内部，可检查exact/interval；未来bin与left-censor未覆盖 |
| event `duration_code` | P | P | P | **部分可评估** | 可由边界区间派生；区间跨bin时必须UNKNOWN；right-censor/GT96未覆盖 |
| event `object_effect` | P | P | P | **部分可评估** | 粗TRANSFER/TRANSLATE/ROTATE候选；单code如何裁决并发rotate+transfer仍不清楚；其它effect未覆盖 |
| event `outcome` | U | U | U | **不可评估** | 没有可确认VERIFY_OUTCOME event、冻结criterion或sensor truth；非verify只测试机械N/A |
| edge `actor` | P | P | P | **部分可评估** | visible hand支持43=RIGHT、50/58=LEFT proposal及OBJECT_OBJECT edge；另一手roster不完整 |
| edge `primary_node` | V | V | V | **可局部评估** | source与destination两种role reference可区分 |
| edge `primary_part` | P | P | P | **部分可评估** | source BODY、target OPENING/RIM_EDGE可讨论；关键帧受遮挡，完整part词表未覆盖 |
| edge `primary_surface_cell` | U | U | U | **不可可靠评估** | hand-source contact patch和rim关键区域都被手/source遮挡；part-local frame/canonicalization也未冻结 |
| edge `secondary_node` | V | V | V | **可局部评估** | source↔destination relation reference可区分；其它role组合未覆盖 |
| edge `relation_before` | P | P | P | **部分可评估** | SEPARATE/NEAR/OVERLAP_2D；不能从透明投影推出TOUCHING/INSIDE/CONTAINS |
| edge `relation_after` | P | P | P | **部分可评估** | 同上；ALIGNED_TO_RIM在turn遮挡下通常需UNKNOWN/interval support |
| edge `contact_mode` | P | P | P | **部分可评估** | hand-source STICK与object relation NONE proposal；slide/roll/impact等未覆盖，真实contact边界部分不可见 |

按三段综合的最严格三档统计：

```text
可局部评估（V）     3 / 18：center_cell, primary_node, secondary_node
部分可评估（P）    11 / 18：observability, kind, focus, onset, duration,
                           effect, actor, primary_part,
                           relation_before, relation_after, contact_mode
不可可靠评估（U）   4 / 18：scale_band, orientation_2d,
                           primary_surface_cell, outcome
```

这不是 schema coverage pass。即便 V/P 共 14 类字段都有 proposal evidence，也仍有以下 action-level缺口：

- `edge actor/presence` 的完整性 gate在 `0/3` episode可满足：另一只手均未 exhaustive review；
- 完整 `OK/NULL` oracle packet在 `0/3` episode得到证明；本 memo没有 compiler/checker，也不会手工拼包；
- 8 event kinds只观察到6类候选，`REGRASP/VERIFY_OUTCOME` 为 `0/3`；
- 4 role slots只有source/destination正例，tool/fixture positive coverage为 `0/3`；
- outcome正例、bimanual、side-ambiguous、positive censor、`edge/event>2`、node/episode overflow均为 `0/3` 支持，必须记 `NOT_EVALUABLE`，不能以零错误过门；
- entry/exit的手部稀疏2D pose在 `3/3` 可做局部probe，但turn的完整7 landmarks在 `0/3` 稳定可见；pose仍然action-invisible，不能补core。

## 8. 主要跨 episode 歧义

### 8.1 visible transfer 不等于 visible flow 或 success

三段都能看到紫色内容从初始source转移到最终destination；这是 `rgb_temporal` 的粗 `TRANSFER` proposal。倾倒中间的stream、flow onset/offset、流速、体积与contact interface都被透明介质、手和杯体遮挡。不能从杯子倾斜、任务名或最终出现颜色反推每一帧 `FLOW`，也不能在没有volume/spill criterion时写 `SUCCESS`。

### 8.2 单effect code遇到“rotate + transfer”

倾倒时source明显旋转，同时发生内容转移。standard-r1 event只有一个 `object_effect`。若task ontology把主目标冻结为 `TRANSFER`，rotate只是实现动作；若没有该冻结，标注者可能在 `ROTATE/TRANSFER`间分歧。正式 calibration dictionary必须明确“主任务效果优先”或其它唯一规则，不能在盲标后按一致性选择。

### 8.3 release起点与合法grammar

三段在transfer结束后都把空source回正并放回桌面。r1不允许 `MANIPULATE -> HOLD`，而release定义从“开始有意终止最后一条task-relevant contact”起。长放回阶段因此可能被标成长`RELEASE`，也可能导致标注者把MANIPULATE延到更晚。三段很适合校准这项边界，但本 memo不替标注者决定exact onset。

### 8.4 另一只手缺失是MNAR候选，不是absence

可见操作手在43为RIGHT proposal，在50/58为LEFT proposal；另一只手从未进入画面。它可能确实未参与，也可能只是佩戴者姿态/相机frustum造成不可见。根据missing-hands r1 contract，只能保留`OUT_OF_FRAME/NOT_REVIEWED`，不能授权另一个side的ABSENT negative，也不能把一个可见edge编成完整left-only/right-only oracle。

### 8.5 粗2D关系不等于透明3D关系

`NEAR`和`OVERLAP_2D`可从画面审计；`TOUCHING/INSIDE/CONTAINS`不能。`ALIGNED_TO_RIM`需要source/target rim同时可定位并落入冻结predicate，而三段关键turn恰好都受遮挡。若正式标注仍要求该字段，应允许REVIEWED_UNKNOWN并单报coverage，不得用VLM、depth completion或任务常识补齐。

## 9. 对后续正式标注的建议（不构成执行授权）

1. 将43/50/58只作为dictionary calibration时，先解决三个预期高分歧点：`HOLD->MANIPULATE`、长`RELEASE`、`ROTATE vs TRANSFER`主effect。
2. event轮必须单独填per-side hand roster；“另一手未出现”默认NOT_REVIEWED/OUT_OF_FRAME，不默认NOT_INVOLVED。
3. turn pose允许只保留可见landmark与visibility mask；不要想象补指尖，也不要让pose补actor、part/cell或contact GT。
4. 对source/target圆杯，orientation应允许reviewed UNKNOWN；scale阈值必须在正式calibration/train split冻结后再评估。
5. 对part-local cell，先冻结part框/canonicalization；若43/50/58在turn仍无法达到足量可见支持，该字段应在human-pour D1记NOT_EVALUABLE，而不是强迫点击。
6. transfer effect只使用前后原始RGB时序；stream/flow/volume/outcome需要直接可见或sensor-confirmed evidence。任何tracker、hand status、VLM、reconstruction或proposal只能进入另名proposal层。
7. formal calibration与blind set必须互斥。如果未来坚持43/50/58作正式三段calibration，就必须从其余episode重新冻结12 blind，不能沿用同一formal 12并把三段重复计入agreement。

## 10. Gate 状态

| gate / claim | 本 memo 状态 | 原因 |
|---|---|---|
| 三段raw-video字段可观测性prescreen | **完成** | 有hash/ffprobe/timeline绑定与逐段proposal矩阵 |
| GT生成 | **No-Go / 未做** | 没有A/B、裁决、sensor/native truth；明确禁止proposal升级GT |
| D0 standard-r1一致性 | **未通过** | 没有checker、annotation→82-code→annotation round-trip、grammar/capacity receipt |
| D1 3+12局部标注 | **未执行/不得宣称通过** | 本次只看3段；没有12 blind、agreement、unknown/overflow统计和第三人裁决 |
| 完整oracle packet | **0/3得到证明** | actor roster不完整，part/cell/outcome等required coverage不足，且未运行compiler |
| 扩到66 / UI / 模型训练 | **No-Go** | D0/D1未过；3段不覆盖完整schema |

最终结论：43/50/58是一个有用但明显不充分的只读校准三元组。它们覆盖镜像布局与可见手side差异、普通与重遮挡turn、以及可见紫色内容前后转移；同时暴露了transparent pouring中最关键的不可标部分——stream、part-local cell、完整双手roster与verify outcome。应把这些不可见性诚实编为partial/unknown/missing或packet拒绝，而不是用已有可视化、算法输出或任务常识补成GT。
