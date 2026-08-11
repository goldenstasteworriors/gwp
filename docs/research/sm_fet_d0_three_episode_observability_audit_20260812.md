# SM-FET formal blind-set 三段 Ego pour proposal-only observability probe

- 日期：2026-08-12
- 工作树：`codex/sm-fet-d0-observability-20260812`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 输出性质：**coordinator-only blind-set proposal probe；不是 annotation、GT、calibration、D0/D1 evidence 或 gate 结果**
- 视频源：`/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers`
- 正式 packet：`../gwp-agent-op17-formal-label-packet/track_a/contact_labels_v1_1/formal_dual_annotation_v1_1/`
- probe 对象：正式 12-blind minimum set 中 episode `43 / 50 / 58`
- 资源边界：CPU-only；未训练、未调用 GPU、未改视频/HDF5/packet、未生成任何 GT
- 分发边界：**本 memo 不供未来 blind A/B 标注者查看，不冻结 dictionary、task ontology、event/edge proposal、side、窗口或协议**
- fix1 依据：`../gwp-agent-sm-fet-d0-observability-review/docs/research/sm_fet_d0_observability_independent_review_20260812.md`

## 0. 结论先行

按本任务指定，对 formal blind set 内的 `43 / 50 / 58` 做 coordinator-only 单人 proposal probe。三段**仍属于现行协议冻结的 12 blind episodes，不是 calibration，也不能被本 memo改作 calibration**：

1. **episode 43：镜像布局 / 可见右手样本。** source 在画面右侧、target 在左侧；可见操作手的形态支持 `RIGHT` proposal。source/target 的紫色液体前后状态变化清楚，倾倒时杯口和指尖有中等遮挡。
2. **episode 50：遮挡困难样本。** source 在左、target 在右；可见操作手支持 `LEFT` proposal。约 processed `72–116` 帧内，手、腕部设备和 source 同时遮挡 target rim / source opening；只能从前后时序确认粗粒度 transfer，不能看见可靠 stream、flow rate 或 contact interface。
3. **episode 58：普通长上下文 / 可见液体差异样本。** source 在左、target 在右；可见操作手支持 `LEFT` proposal。片段更长，接近、持杯、回正、放杯和事后静止上下文较完整；紫色内容从 source 转到 target 的前后证据清楚，但倾倒中段仍被手遮挡。

三段共同支持的只是“raw RGB中可提出哪些待 A/B 独立回答的问题”的**提案级证据**：source/target role、粗 2D node、六类候选 event、一个可见操作手、hand→source 与 source/target relation 两种 edge，以及 RGB-temporal 的粗粒度 `TRANSFER`。这些 proposal 不冻结协议，也不允许预填、提示或分发给 blind A/B。三段共同不支持：

- 第二只手 `NOT_INVOLVED` 或完整单手 roster；另一手全程画外，不能从“没看到”推出 absent；
- stream、flow rate、volume、spill fraction、透明 contact interface 或 3D `INSIDE/TOUCHING` 真值；
- source/target 圆柱杯的稳定 `orientation_2d`、未冻结阈值下的 `scale_band`、被手遮挡的 part-local exact cell；
- `REGRASP`、`VERIFY_OUTCOME`、双手/ambiguous-side、正例 censor、overflow 等未出现类别的可标性；
- task success / partial / failure。三段只有“可见内容发生转移”的证据，没有冻结 outcome 判据或传感器确认。

因此：

- 本 memo **不生成 GT、不形成 calibration、不计入 D0/D1，也不使任何 packet 成为 oracle-admissible**；
- D0 是 checker/round-trip/grammar/capacity 的机械 gate；视频 V/P/U 看法不是 D0 evidence。当前缺 annotation→82-code→annotation checker/round-trip 与 grammar/capacity receipt，故 **D0 未通过**；
- 没有合法且与 blind set 互斥的 3 calibration、A/B 独立标注、agreement、第三人裁决和 12 blind 结果，故 **D1 未执行，不得宣称通过**；
- 在现行协议下，calibration 必须从其余 54 段预先冻结。若未来改变 43/50/58 的用途，必须先发布新 protocol/assignment version、替换 blind episodes并重新冻结 hashes；本 memo与独立复审仍不得让新 blind A/B看到。

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

packet README 明确：这 12 段是空白正式任务，不是 GT；A/B 不得看到 timing hint、prescreen、hand status、action saliency 或模型 proposal。本 memo 本身包含 episode选择、side/event/edge/window proposal，因而属于 **coordinator prescreen**，不能进入 A/B evidence allowlist。其访问与语义边界是：

- 语义判断只来自原始 MP4；同时读取 formal packet 的 coordinator manifest / timeline mapping用于媒体身份、processed coverage和时间对照；
- **没有直接读取 HDF5 数组，也没有读取或使用**现有 `hand_status`、`hand_active`、contact visualization、contact annotation、tracker/MANO/VLM/算法输出作语义判断；
- coordinator manifest 暴露 processed HDF5 路径/hash，timeline frame mapping与sync summary来自 HDF5 camera timestamps；因此本审计使用了 **HDF5-derived technical time metadata**，不能表述为“完全没有使用HDF5”。这些 metadata不含手/contact语义GT，也绝不能分发给blind A/B；
- 没有把任务名、杯子运动或“按脚本完成”直接提升为 physical contact / success；
- 所有下述 event、edge、side 和 frame 区间均加 `proposal` 限定词；不能被复制为 human-independent row。

本文中的 `p[a,b]` 指 processed 20 Hz 的**只读证据查看窗口**，不是 onset/offset label，也不是半开 GT interval。窗口由单一审计者按语义提出且有意重叠，尚无 A/B 复现；遮挡只产生 partial/unknown，不产生 censor。表中的 `t_p` 是 `processed_frame / 20` 的 processed-time alias，**不是逐点查 timeline 后的 raw-video time**。

### 1.3 媒体完整性与 CPU 探测

使用现有 `decoupled_vla_collection` 环境中的 `ffprobe` 绝对路径只读探测；没有激活/修改 conda 环境，也没有安装依赖。

| episode | packet task | processed frames | MP4 SHA-256 | video | ffprobe |
|---:|---|---:|---|---|---|
| 43 | `contact-min12-00` | 264 | `36bee62146e5351a096e06bb489cdc941c70af92e2d58bf1be6eee0fb91b94e3` | 400 frames, 13.334 s | MPEG-4, 640×480, 30 fps, yuv420p |
| 50 | `contact-min12-01` | 223 | `ace55b2d1a563141c4afc6765c5c528a31946d4554a6fed715efb280a898a618` | 338 frames, 11.267 s | MPEG-4, 640×480, 30 fps, yuv420p |
| 58 | `contact-min12-03` | 294 | `92e9b10dfc7bb86b67021b44fdd026ea1f6fe4a3f06280c110580bacafa1180a` | 444 frames, 14.800 s | MPEG-4, 640×480, 30 fps, yuv420p |

三个本地 SHA-256 均与 `coordinator/source_episode_manifest.csv` 一致。manifest 报告三段最大 processed↔video sync error 分别为 `20.562 / 23.366 / 25.731 ms`，均小于 packet 50 ms 上限。这些 sync 数值与 processed frame→video frame mapping来自 HDF5 camera timestamp的技术派生；它们只绑定时间轴，不提供动作、手或contact语义。

原稿括号时间由 `p/20` 直接换算。independent review 用 timeline逐点复核后给出的例子是：

| episode / proposal window | `t_p` processed alias | timeline raw-video time |
|---|---|---|
| 43 `MANIPULATE p72–136` | `3.6–6.8 s` | `3.633333–6.866667 s` |
| 43 `RELEASE p136–187` | `6.8–9.35 s` | `6.866667–9.433333 s` |
| 50 `MANIPULATE p68–120` | `3.4–6.0 s` | `3.433333–6.033333 s` |
| 50 `RELEASE p120–168` | `6.0–8.4 s` | `6.033333–8.466667 s` |
| 58 `MANIPULATE p88–136` | `4.4–6.8 s` | `4.400000–6.833333 s` |
| 58 `RELEASE p136–191` | `6.8–9.55 s` | `6.833333–9.600000 s` |

其余括号时间继续仅作 `t_p` alias；几十毫秒差异不改变宽松窗口顺序，但任何时间都不能升级为 boundary support。

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
| `APPROACH` | V | `p24–42` (`t_p=1.2–2.1s`) | entry 手腕、掌心、五指大部可见；接近 source | source | `NONE`; outcome N/A | “持续任务定向接近”起点可在约 4–8 processed 帧内移动 |
| `PRECONTACT` | P | `p36–47` (`t_p=1.8–2.35s`) | 指尖靠近杯壁，仍有无接触帧 | source | `NONE`; outcome N/A | source 透明、指尖/杯缘投影重叠；不能从 overlap 定物理接触 |
| `ACQUIRE` | P | `p42–55` (`t_p=2.1–2.75s`) | entry→exit 手指闭合；部分指尖被杯壁/杯内容挡住 | source | `NONE` 或短 `TRANSLATE`; outcome N/A | 首次 contact 与“稳定可控 grasp”分界是 interval；不能用 grip proxy补 |
| `HOLD` | V | `p52–72` (`t_p=2.6–3.6s`) | source 已稳定持握并向 target 对准 | source | `TRANSLATE/NONE` 需 ontology 冻结；outcome N/A | move-to-align 是否仍是 HOLD 或主 MANIPULATE 起点存在语义分界 |
| `MANIPULATE` | P | `p72–136` (`t_p=3.6–6.8s`) | turn 在 `p78–112`；腕/掌可见，多个指尖与 source/target rims 被挡 | destination | 粗 `TRANSFER` 有 RGB-temporal 支持；`FLOW`、体积与 outcome U | source rotate 与 transfer 同时发生，但 wire 只有一个主 effect；stream/contact interface不可见 |
| `REGRASP` | U/未见 | 全片复核未见明确接触拓扑重分配 | 无独立 turn key 支持 regrasp | — | — | 腕姿变化不能自动算 regrasp；未见不等于 GT ABSENT |
| `RELEASE` | P | `p136–187` (`t_p=6.8–9.35s`) | exit `p176–187` 手指张开、离杯；腕部靠画面边缘 | source | `NONE/TRANSLATE`; outcome 强制 N/A | r1 不允许 `MANIPULATE->HOLD`，故回正/放回整体更适合作长 release proposal；真实 contact end仍是 interval |
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
| 双手 side / roster | P→U | visible hand `p24–187`；另一手全片不可见 | 可见手 proposal 为 RIGHT；本单人probe不能把另一手记为NOT_INVOLVED，也不能证明left/right-only完整packet；未来A/B exhaustive结论未知 |
| censor / unknown | V（区分能力） | 全 episode coverage | 片头/片尾有空白上下文，候选事件不需要 left/right censor；关键边界应 interval/unknown。遮挡不是 censor |

## 5. Episode 50 observability matrix

### 5.1 只读概况

候选路径同 episode 43。source 初始在画面左、target 在右；可见手从左上进入，entry/exit 的手掌/拇指拓扑支持 `LEFT` proposal。`p72–116` 是三段中最强的 rim/hand 遮挡：紫色内容在 target 中的出现可由前后帧确认，但 source opening、target rim、多个指尖和潜在 stream 同时不可见。

### 5.2 八类 event + pose/focus/effect 矩阵

| event kind | 可观测性 | proposal evidence window | entry / turn / exit 2D pose | focus proposal | effect / outcome | 主要歧义 |
|---|---|---|---|---|---|---|
| `APPROACH` | V | `p28–40` (`t_p=1.4–2.0s`) | entry 手腕/掌心/五指大部可见 | source | `NONE`; outcome N/A | approach 起点只有时序目的性，单帧不能决定 |
| `PRECONTACT` | P | `p36–45` (`t_p=1.8–2.25s`) | 指尖与 source rim/body逐渐接近 | source | `NONE`; outcome N/A | 透明杯边缘和指尖重叠；首次接触不可单帧精确 |
| `ACQUIRE` | P | `p40–50` (`t_p=2.0–2.5s`) | 手指闭合，杯体开始随手移动 | source | `NONE/TRANSLATE`; outcome N/A | contact onset、stable grasp和 lift onset相互接近 |
| `HOLD` | V | `p48–68` (`t_p=2.4–3.4s`) | 稳定持杯并移动到 target 上方 | source | `TRANSLATE/NONE`; outcome N/A | 与 manipulate 的转折依赖“主 effect何时开始” |
| `MANIPULATE` | P（遮挡困难） | `p68–120` (`t_p=3.4–6.0s`) | turn `p72–112`；腕/掌可见，指尖、两杯口和潜在液流大面积遮挡 | destination | target 紫色内容前后变化支持粗 `TRANSFER`; `FLOW`/volume/outcome U | 本段不能提供 direct stream；`ALIGNED_TO_RIM`和 surface cell最容易变 UNKNOWN |
| `REGRASP` | U/未见 | 全片 | 无独立 regrasp pose支持 | — | — | 腕转动是倾倒动作，不足以证明 contact topology改变 |
| `RELEASE` | P | `p120–168` (`t_p=6.0–8.4s`) | exit `p156–168` 手张开并离开空 source | source | `NONE/TRANSLATE`; outcome N/A | 真正 separation在少数帧内，但透明杯边界使 exact frame仍可能有分歧 |
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
| 双手 side / roster | P→U | visible hand `p28–168`；另一手全片不可见 | 可见手 proposal LEFT；本单人probe未证明另一手NOT_INVOLVED或single-left admission；未来A/B exhaustive结论未知 |
| censor / unknown | V（区分能力） | 全片 | 无 clip censor；遮挡区应使用 field-level unknown / interval，而不是 right/left censor |

## 6. Episode 58 observability matrix

### 6.1 只读概况

候选路径同前两段。片段有较长的 pre/post idle 和放杯上下文，source 左、target 右；可见手支持 `LEFT` proposal。紫色内容约在 `p100–116` 出现在 target，`p136` 后 source 已回正并移回。相比 episode 50，前后状态更容易复核，但 stream 和 contact interface仍不可见。

### 6.2 八类 event + pose/focus/effect 矩阵

| event kind | 可观测性 | proposal evidence window | entry / turn / exit 2D pose | focus proposal | effect / outcome | 主要歧义 |
|---|---|---|---|---|---|---|
| `APPROACH` | V | `p36–50` (`t_p=1.8–2.5s`) | entry 手腕/掌心/五指大部可见 | source | `NONE`; outcome N/A | 首次任务定向靠近的起点为interval |
| `PRECONTACT` | P | `p46–57` (`t_p=2.3–2.85s`) | 指尖接近source body/rim | source | `NONE`; outcome N/A | overlap不能证明 contact |
| `ACQUIRE` | P | `p52–64` (`t_p=2.6–3.2s`) | 手指闭合，source从桌面开始随手运动 | source | `NONE/TRANSLATE`; outcome N/A | 稳定 grasp 时点仍需逐帧双标 |
| `HOLD` | V | `p62–88` (`t_p=3.1–4.4s`) | 持杯移动并对准target | source | `TRANSLATE/NONE`; outcome N/A | 与倾倒turn之间边界可做interval |
| `MANIPULATE` | P | `p88–136` (`t_p=4.4–6.8s`) | turn `p92–124`；腕/掌可见，指尖和rims部分遮挡 | destination | 粗 `TRANSFER` 有时序证据；FLOW/volume/outcome U | source旋转与transfer并发；单effect code需ontology裁决 |
| `REGRASP` | U/未见 | 全片 | 无独立regrasp key | — | — | 没有可确认的接触拓扑解除/迁移/重建 |
| `RELEASE` | P | `p136–191` (`t_p=6.8–9.55s`) | exit `p180–191`，手指张开并离开空source | source | `NONE/TRANSLATE`; outcome N/A | release起点可早于实际separation；长release与MANIPULATE边界需protocol固定 |
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
| 双手 side / roster | P→U | visible hand `p36–191`；另一手全片不可见 | 可见手 proposal LEFT；本单人probe未证明另一手NOT_INVOLVED或exhaustive left-only；未来A/B结论未知 |
| censor / unknown | V（区分能力） | 全片 | 首尾均有空白，候选事件无censor；turn遮挡属于unknown/partial，不是censor |

## 7. 三段 action-visible 字段可评估性统计

统计单位是 schema 的 **18 种字段类型**（4 node + 6 event + 8 edge），不是 82 个重复 wire slots。fix1 按 independent review 的更严格 **code-level repeatability proposal** 统计，而不是只看 raw distinguishability；它仍不是 agreement、vocabulary coverage、D0/D1输入或 oracle admission结论。

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
| edge `primary_node` | P | P | P | **部分可评估** | source/destination raw role可辨，但选哪条edge、谁是primary依赖尚未冻结的ontology/reference union |
| edge `primary_part` | P | P | P | **部分可评估** | source BODY、target OPENING/RIM_EDGE可讨论；关键帧受遮挡，完整part词表未覆盖 |
| edge `primary_surface_cell` | U | U | U | **不可可靠评估** | hand-source contact patch和rim关键区域都被手/source遮挡；part-local frame/canonicalization也未冻结 |
| edge `secondary_node` | P | P | P | **部分可评估** | source/destination可辨；是否需要独立object-object edge及primary/secondary方向仍是审计者proposal |
| edge `relation_before` | P | P | P | **部分可评估** | SEPARATE/NEAR/OVERLAP_2D；不能从透明投影推出TOUCHING/INSIDE/CONTAINS |
| edge `relation_after` | P | P | P | **部分可评估** | 同上；ALIGNED_TO_RIM在turn遮挡下通常需UNKNOWN/interval support |
| edge `contact_mode` | P | P | P | **部分可评估** | hand-source STICK与object relation NONE proposal；slide/roll/impact等未覆盖，真实contact边界部分不可见 |

按三段综合的最严格三档统计：

```text
局部 V（仅proposal） 1 / 18：center_cell
P（仅proposal）     13 / 18：observability, kind, focus, onset, duration,
                            effect, actor, primary_node, primary_part,
                            secondary_node, relation_before, relation_after,
                            contact_mode
不可可靠评估（U）   4 / 18：scale_band, orientation_2d,
                           primary_surface_cell, outcome
```

`1/13/4` 只是第二位审计者建议纳入后的严格 proposal taxonomy，不能进入 gate，也不代表完整词表覆盖。即便 V/P 共 14 类字段有某种局部视觉依据，也仍有以下 action-level缺口：

- 当前提交中可验证的完整 hand roster为 `0/3 demonstrated`：这里只表示本单人probe没有证明；未来A/B exhaustive review仍可能一致判定另一手NOT_INVOLVED，也可能保持OUT_OF_FRAME/NOT_REVIEWED；
- 当前提交中可验证的 admitted `OK/NULL` oracle packet为 `0/3 demonstrated`，因为没有annotation/compiler/checker；这不是对三段潜在可编译性的判决。有UNKNOWN code的字段并非天然拒绝理由，非verify outcome本来也机械为N/A/0；
- 8 event kinds只观察到6类候选，`REGRASP/VERIFY_OUTCOME` 为 `0/3`；
- 4 role slots只有source/destination正例，tool/fixture positive coverage为 `0/3`；
- outcome正例、bimanual、side-ambiguous、positive censor、`edge/event>2`、node/episode overflow均为 `0/3` 支持，必须记 `NOT_EVALUABLE`，不能以零错误过门；
- entry/exit的手部稀疏2D pose在 `3/3` 可做局部probe，但turn的完整7 landmarks在 `0/3` 稳定可见；pose仍然action-invisible，不能补core。

## 8. 主要跨 episode 歧义

### 8.1 visible transfer 不等于 visible flow 或 success

三段都能看到紫色内容从初始source转移到最终destination；这是 `rgb_temporal` 的粗 `TRANSFER` proposal。倾倒中间的stream、flow onset/offset、流速、体积与contact interface都被透明介质、手和杯体遮挡。不能从杯子倾斜、任务名或最终出现颜色反推每一帧 `FLOW`，也不能在没有volume/spill criterion时写 `SUCCESS`。

### 8.2 单effect code遇到“rotate + transfer”

倾倒时source明显旋转，同时发生内容转移。standard-r1 event只有一个 `object_effect`。若未来在**与blind互斥的正式calibration**中把主目标冻结为 `TRANSFER`，rotate可作为实现动作；当前blind probe不能作此裁决。标注者可能在 `ROTATE/TRANSFER`间分歧，dictionary必须在合法calibration后预先给出唯一规则，不能从43/50/58或blind结果反向选择。

### 8.3 release起点与合法grammar

三段在transfer结束后都把空source回正并放回桌面。r1不允许 `MANIPULATE -> HOLD`，而release定义从“开始有意终止最后一条task-relevant contact”起。长放回阶段可能被标成长`RELEASE`，也可能暴露grammar/ontology粒度问题。43/50/58只能把它登记为blind-set风险，**不能用于校准或冻结该边界**。

### 8.4 另一只手缺失是MNAR候选，不是absence

可见操作手在43为RIGHT proposal，在50/58为LEFT proposal；另一只手从未进入画面。它可能确实未参与，也可能只是佩戴者姿态/相机frustum造成不可见。本单人probe只能记录“未证明完整roster”，不能授权另一个side的ABSENT negative或完整left/right-only oracle；未来A/B exhaustive review仍可能一致给出NOT_INVOLVED，也可能保持OUT_OF_FRAME/NOT_REVIEWED。

### 8.5 粗2D关系不等于透明3D关系

`NEAR`和`OVERLAP_2D`可从画面审计；`TOUCHING/INSIDE/CONTAINS`不能。`ALIGNED_TO_RIM`需要source/target rim同时可定位并落入冻结predicate，而三段关键turn恰好都受遮挡。若正式标注仍要求该字段，应允许REVIEWED_UNKNOWN并单报coverage，不得用VLM、depth completion或任务常识补齐。

## 9. Coordinator 风险登记与后续协议边界（不供blind A/B看）

1. 现行formal assignment不变：43/50/58留在12 blind；真正calibration从其余54段预先冻结并与blind互斥。本memo与独立复审均不得向blind A/B分发。
2. `HOLD->MANIPULATE`、长`RELEASE`、`ROTATE vs TRANSFER`只是coordinator需要在合法calibration中验证的风险问题；本memo不提供答案或预填值。
3. event轮需要单独、独立地复核per-side hand roster；另一手未入画不自动是NOT_INVOLVED，也不预先排除A/B在exhaustive evidence下得出NOT_INVOLVED。
4. turn pose只能保存可见landmark与visibility mask；不能想象补指尖，也不能让pose补actor、part/cell或contact GT。
5. source/target圆杯的orientation应允许reviewed UNKNOWN；scale阈值、part框与cell canonicalization必须在合法calibration/train split冻结后再进入blind评估。
6. transfer候选只来自前后原始RGB时序；stream/flow/volume/outcome需要直接可见或sensor-confirmed evidence。tracker、hand status、VLM、reconstruction或本memo proposal都不能进入blind truth或GT。
7. A/B不能被要求对本memo的`1/13/4`投票。正式D1只使用独立rows、review-state-aware comparable set、agreement与裁决；D0只使用checker/round-trip/grammar/capacity receipt。

## 10. Gate 状态

| gate / claim | 本 memo 状态 | 原因 |
|---|---|---|
| 三段blind-set coordinator-only proposal probe | **完成，但不计gate** | 有hash/ffprobe/timeline绑定与逐段单人proposal矩阵；不供A/B看 |
| GT生成 | **No-Go / 未做** | 没有A/B、裁决、sensor/native truth；明确禁止proposal升级GT |
| D0 standard-r1一致性 | **未通过** | 没有checker、annotation→82-code→annotation round-trip、grammar/capacity receipt |
| D1 3+12局部标注 | **未执行/不得宣称通过** | 没有与12 blind互斥的合法3 calibration，也没有blind A/B、agreement、unknown/overflow统计和第三人裁决 |
| 当前提交可验证的admitted oracle packet | **0/3 demonstrated** | 未执行annotation/compiler/checker；不是对视频潜在可编译性的判决 |
| 扩到66 / UI / 模型训练 | **No-Go** | D0/D1未过；3段不覆盖完整schema |

最终结论：43/50/58只是 formal blind set 内的 coordinator-only、proposal-only observability probe，不是calibration，不是D0/D1 evidence，也不冻结任何side、event、focus、edge、effect或时间窗口。更严格taxonomy为`1V/13P/4U`，同样不得进入gate。三段raw RGB能提出镜像布局、可见手side、turn遮挡和紫色内容前后转移等待复核问题，也暴露stream、part-local cell、hand roster与verify outcome的证据风险；正式结论必须来自blind A/B独立rows与checker，而不能来自本memo、已有可视化、算法输出或任务常识。
