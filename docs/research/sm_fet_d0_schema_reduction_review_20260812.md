# SM-FET D0：standard-r1 可标性与 action schema 最小化独立审查

- 日期：2026-08-12
- 独立 worktree：`/home/ykj/project/WM/gwp-agent-sm-fet-d0-schema-reduction`
- canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 审查对象：`event_tube_v2_standard / sm_fet_hard_grammar_v2_standard_r1`
- 审查性质：只读 schema、annotation、focus、missingness、transparent、cognitive 与 benchmark 证据审查；只读查看现有 Ego corpus 元数据和少量视频；未生成 GT、未写代码、未训练、未使用 GPU/服务器
- 独立性：没有读取 sibling `gwp-agent-sm-fet-d0-handbook` 或任何同题 D0 reduction 结论；本 memo 的逐字段判断独立形成
- `idea_id`：`b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`
- 本轮 `discussion_id`：`126b612c-3da6-4a77-8526-0394e60c9be0`

## 0. Executive decision

**裁决：standard-r1 作为当前默认 action schema 过度复杂；它可以保留为信息上界/挑战者，但不应在 D0、D1 尚未通过时被称为“minimal”或默认主路线。**

82 个 `uint8` code 的存储布局很整齐，但把四种性质不同的信息放进了同一个 action alphabet：

1. 跨具身任务语义：event kind、功能 role、part、effect、目标 relation；
2. 目标域当前定位：node center、scale、orientation、part-local cell；
3. 具身执行分配：human/robot literal actor side、第二条手/物 edge；
4. 预测与认识状态：future onset/duration、outcome、observability、各种 UNKNOWN。

“能被标”“能被 planner 预测”“能跨域对齐”和“低层执行必须知道”不是一回事。现有 Ego pour 只能为 source/target、少量 active event、粗 node center、部分 primary part/effect/relation提供待双标的可观察支持；它不能验证 tool/fixture、目标机器人当前视角 grounding、双 edge、actor side、完整 contact mode、future timing、verify outcome或跨任务词表。

本 memo 推荐先冻结一个 **`minimal-r1a` 候选**，只保留：

```text
四个固定功能 role 的 namespace
+ 被当前 active interaction 引用节点的 target-domain 4x4 center
+ 一个 active event 的精简 phase kind
+ active event 的 object effect
+ 一个 active interaction record：primary role/part,
   optional secondary role, one target relation
```

其余字段移到 planner supervision、target-robot allocator、supervisor/control plane、sensor auxiliary或独立 standard-r1 challenger。`minimal-r1a` 不是已经合法化的新 schema；它必须由未来 schema agent另立版本、grammar、validator、codec 与 equal-bit control。

这里的 `minimal-r1a` 也不等同于 schema memo 已列出的既有 `minimal (N=3,K=2,Q=6,R=1)`。后者主要缩 maxima 和词表；本候选审查的是“字段是否应对 action 可见”，保留四 role namespace，却进一步把 future、timing、actor、site、mode、outcome与epistemic状态分层移出 action。

### 容量红线

本 memo **不计算，也不估算** `minimal-r1a` 的 wire bytes、reachable states、bit capacity或最小索引宽度。删掉 future slots、第二条 edge、UNKNOWN code、timing和 node geometry字段会共同改变 presence、reference union、grammar、edge bundle 与包含—排除计数，不能从 standard-r1 的 `207.749746919520 bit` 做字段位宽相减。

因此：

- `207.749746919520 bit / 208-bit index / 82 bytes / 13 tokens` 只属于未修改的 standard-r1；
- `minimal-r1a` 的 exact integer、`log2`、container width、token layout、rank/unrank与 equal-bit control全部标为 **TBD by schema agent**；
- 在新枚举与独立复核完成前，任何给 `minimal-r1a` 报出的 bit 数都属于伪精确。

## 1. 判定口径

每个字段按四个互不替代的问题审计：

| 轴 | 问题 | 本 memo 的判定含义 |
|---|---|---|
| `H-Ego` | 现有 human Ego 原视频能否提供人工可复核证据？ | `Y`=有直接候选；`C`=条件可见/需时序或传感；`R`=只能事后回看或由任务规则推断；`N`=当前 corpus无来源；全部仍未过A/B双标 |
| `R-view` | 目标机器人在部署时能否从当前/因果观测获得？ | `Y`=当前可测；`C`=需 tracker/sensor/allocator；`P`=只能预测 future；`N`=不能因果获得 |
| `X-domain` | human/robot 值能否机械共享？ | `Y`=功能语义可共用；`L`=编码规则共用但具体值是域/相机局部；`C`=需 canonicalization receipt；`N`=literal 值不应跨域 |
| `Exec` | execute-8 的动作 expert 是否逻辑上必须读取？ | `Y`=缺少即不知道当前目标；`C`=可能增益但需 D3 干预证明；`N`=可由 retained field、proprio、allocator或control plane处理 |

`Exec=Y` 只表示本 memo 的逻辑最小判断，不是策略收益证据。真正的“必要”仍需目标机器人 oracle、field intervention和 matched baseline 证明。

## 2. 只读 Ego corpus 证据账本

### 2.1 文件与元数据

只读 corpus：

`/home/ykj/project/EgoHumanoid/data_collection/human_data/data_collection/7_4_pour_liquid_between_beakers`

直接核验结果：

| 项 | 只读事实 | 对 schema 的含义 |
|---|---|---|
| episodes | 66 个普通 HDF5、66 个 RGB MP4、66 个 depth HDF5 | 单任务数量不能替代字段覆盖 |
| 普通 HDF5 | 所有 episode 都只有 `body_pose`、左右 controller pose、左右 hand pose/active、local timestamps | 没有 object role/track、part/site、event、relation、effect/outcome GT |
| raw 时钟 | `collection_interval_s=0.01`；总计 99,953 raw rows；每 episode 1,069–2,173 rows，中位数 1,471 | 原始 pose/active约100 Hz；standard 文档中的 processed 20 Hz 是下游时钟，不能把 raw row直接当 event frame |
| RGB | 抽查四段均为 640×480、30 fps，可解码 | 需明确 30 Hz RGB→20 Hz processed 的映射 receipt |
| depth | 例如 episode 43 是 `[400,480,640] uint16 depth_raw`，含相机内参/时间戳 | raw depth不是透明物体/object-state GT，也没有 contact/outcome语义 |
| identity/group | 66个 root attrs 都不含 `subject/session/operator/source instance/target instance/liquid/batch` | 不能做 group-OOD，也不能从编号/时间猜 identity |
| capture span | root attr 时间从 `2026-07-04 15:17:34` 到 `15:51:22` | 只能说明记录时间相近，不能据此制造 session/batch GT |

普通 HDF5 的 `camera_synced=True` 和视频/depth路径只证明采集侧声明；它不证明 object geometry、physical contact或 outcome truth。`hand_active` 的格式虽是 0/1，也只是 tracker/proxy validity。

### 2.2 少量视频概况，不作 GT

只读解码 episode `0,12,43,58`，每段在 0/25/50/75/99% 位置查看；没有保留衍生帧、没有写 annotation：

| episode | RGB frames / 时长 | raw HDF5 时长 | 只读概况 |
|---:|---:|---:|---|
| 0 | 629 / 20.97 s | 20.96 s | 固定斜俯视工作台；透明容器和装置占画面小区域；单手间歇进入上缘 |
| 12 | 537 / 17.90 s | 17.90 s | 相同布置；目标与 source位置变化小；手部仅在部分时刻出现 |
| 43 | 400 / 13.33 s | 13.32 s | 相同布置；可见有色内容物与透明容器；手/杯存在遮挡时刻 |
| 58 | 444 / 14.80 s | 14.85 s | 相同布置；另一侧手进入；对象仍处固定工作台局部 |

这些四段只支持以下保守观察：

1. source/target 的 coarse role 与大致 2D center有标注可能；
2. 固定相机、固定布置会让 center/scale/orientation 看似容易，却也可能使它们近似常量，不能由高 agreement推导 action value；
3. 手在部分帧出画或被容器遮挡，literal side、contact mode与双 edge不能由单帧稳定决定；
4. 透明/反光与小目标使 rim/site、flow/transfer、inside/contact等字段必须允许 reviewed unknown或传感确认；
5. 这不是 tool/fixture、跨视角或目标机器人 grounding 的覆盖样本。

四段 raw tracker flags 的 left/right active 比例分别约为 `0.622/0.000`、`0.603/0.007`、`0.662/0.000`、`0.000/0.576`。它们与现有 op28 汇总一致地说明 validity 强烈偏侧，但**绝不等于单手参与或 physical contact truth**。已有审计还报告全部66段 processed 帧中双手同时 active 仅 165 帧（0.824%）；这只能反驳“现有数据已经验证双 edge”，不能反向证明任务本身不是双手。

## 3. 16 个 node codes：四 role 与四 geometry 字段

### 3.1 四个固定 role slots

| slot / role | H-Ego | R-view | X-domain | Exec | D0裁决 |
|---|---:|---:|---:|---:|---|
| 0 `manipulated_or_source` | Y | Y | Y | Y | 保留为 `minimal-r1a` role namespace；pour中有真实覆盖候选 |
| 1 `tool` | N | C | Y | C | 当前 Ego pour不可评估；保留 namespace，但不得称 D0/D1 admitted；只有真实独立工具才激活 |
| 2 `receptacle_or_destination` | Y | Y | Y | Y | 保留为 `minimal-r1a` role namespace；pour中有真实覆盖候选 |
| 3 `fixture_or_support` | C | C | Y | C | 工作台/架可见不等于 task-relevant fixture；保留 namespace、默认不进 action，等待 D2 target-robot任务 |

结论：**四 role 作为固定功能 namespace 本身不算主要过度复杂**。问题在于 standard 为每个 slot 总是预留四个 action-visible geometry code，并把 tool/fixture 尚未覆盖的字段一起纳入 headline。`minimal-r1a` 可以保留四 role 名称，但只让当前 active interaction 引用 union中的 role提供 target-domain locator。

### 3.2 每个 role 重复的四字段

下表适用于 slot 0–3，因此穷尽 `4 roles × 4 fields = 16 codes`：

| node field | H-Ego | R-view | X-domain | Exec | 独立判断 | `minimal-r1a` |
|---|---:|---:|---:|---:|---|---|
| `observability/presence` | C | Y/C | N（epistemic state） | N | ABSENT/OUT_OF_VIEW/UNKNOWN主要是 compiler admission和reobserve条件；引用 union已隐含 presence | **移到 control plane**；required locator无证据则 fail closed |
| `center_cell` | Y/C | Y/C | L | Y/C | 4×4规则可共用，但 human cell值不传给 robot；robot必须在自己的当前视图重定位 | **保留**，仅 target-domain current locator；跨域 pairing默认不比 cell |
| `scale_band` | Y | Y | L/N | N | 强依赖焦距、相机与阈值；固定工作台中可能近似常量；不说明可达深度或抓取宽度 | **降 auxiliary/challenger** |
| `orientation_2d` | C | C | L/N | C | 对圆杯、透明边界与部分遮挡常无定义；image-plane角度不跨相机；有把手工具可能有用 | **降 auxiliary/challenger**，按part/task gate再 promotion |

关键区别：node center 是 **target-domain local grounding**，不是 human→robot 的共享值；保留它不授权把 human cell、bbox或 orientation用作跨域正配对目标。

## 4. 18 个 event codes：active、next1、next2逐项审计

### 4.1 Active event 六字段

| code | H-Ego | R-view | X-domain | Exec | 独立判断 | `minimal-r1a` |
|---|---:|---:|---:|---:|---|---|
| `active.event_kind` | C/R | C | Y | Y | 需要时序回看；contact/release证据强于regrasp；八类并未全部获认知或D1支持 | **保留精简 kind** |
| `active.focus_node` | C | C | Y | N/C | 与 primary/secondary role、task ontology存在可检测冗余；还可能出现 focus与edge冲突 | **与 active primary role 合并/机械派生** |
| `active.onset_code=ACTIVE` | Y | Y | Y | N | 对 slot0 是常量，没有 action 信息 | **删除 action field** |
| `active.duration_code` | R/C | P/C | N/C | C | human/robot速度不同；remaining duration是预测量，execute-8会重规划；可作时钟监督但未证执行必需 | **planner auxiliary/control** |
| `active.object_effect` | C | C/Y sensor | Y | Y | `translate/rotate/articulate/flow/transfer...` 是跨任务功能语义；透明场景必须有证据 | **保留 known effect**；required unknown时fail closed或选择不依赖effect的profile |
| `active.outcome` | C/R | C/Y sensor | Y | N/C | 只在 VERIFY成立；更像 supervisor termination/recovery状态，不能把未来结果送给当前动作 | **移到 supervisor/benchmark** |

### 4.2 Next-1 event 六字段

| code | H-Ego | R-view | X-domain | Exec | 独立判断 | `minimal-r1a` |
|---|---:|---:|---:|---:|---|---|
| `next1.event_kind` | R | P | Y | C | human标签来自事后完整视频；robot端只是 causal forecast | planner-only |
| `next1.focus_node` | R | P | Y | C | future focus不是当前可观察值 | planner-only |
| `next1.onset_code` | R | P | N/C | N/C | 跨具身速度和边界误差不同；不是共享 pair字段 | planner boundary auxiliary |
| `next1.duration_code` | R | P | N/C | N/C | 同上，且误差会复合 | planner duration auxiliary |
| `next1.object_effect` | R | P | Y | C | 可由任务 ontology/text预期，不应冒充当前观测 | planner-only |
| `next1.outcome` | N/R | N/P | Y | N | future VERIFY outcome若以GT进入action构成后见泄漏；非VERIFY本来为0 | 删除 action field |

### 4.3 Next-2 event 六字段

| code | H-Ego | R-view | X-domain | Exec | 独立判断 | `minimal-r1a` |
|---|---:|---:|---:|---:|---|---|
| `next2.event_kind` | R | P | Y | N/C | 两跳预测比next1更不可靠，当前 execute-8无必要性证据 | planner-only |
| `next2.focus_node` | R | P | Y | N/C | 同上 | planner-only |
| `next2.onset_code` | R | P | N/C | N | 跨域 timing不可机械对齐 | planner boundary auxiliary |
| `next2.duration_code` | R | P | N/C | N | 同上 | planner duration auxiliary |
| `next2.object_effect` | R | P | Y | N/C | 属于计划，不是当前 observation | planner-only |
| `next2.outcome` | N/R | N/P | Y | N | 未来结果不可作当前 oracle/action visible truth | 删除 action field |

### 4.4 Kind 本体的最小化

standard 的 `APPROACH / PRECONTACT / ACQUIRE / HOLD / MANIPULATE / REGRASP / RELEASE / VERIFY_OUTCOME` 对 annotation 很清晰，但不等于八个都应是 action primitive。认知证据只强支持 contact/untouching是候选边界，并明确指出多数 contact motif不是人类事件边界，regrasp作为普适观察阶段的直接证据最弱。

`minimal-r1a` 建议的 known phase候选为：

```text
APPROACH_ALIGN      = APPROACH + PRECONTACT
ACQUIRE
CONTROLLED_CONTACT  = HOLD + MANIPULATE；由 object_effect=NONE/非NONE区分
REGRASP
RELEASE
VERIFY
```

这只是待 schema agent验证的 ontology proposal。`UNKNOWN kind` 不建议继续作为 action primitive：如果 active phase本身执行必需却无法解析，strict route应 fail closed；reviewed unknown仍保存在 annotation/planner loss。

## 5. 48 个 edge codes：六个位置与八个字段

standard 有 `3 event positions × 2 edges × 8 fields = 48 codes`。为避免把同一个结论重复48次，下面先穷尽六个 edge位置，再穷尽八个字段；任一具体 code 的裁决等于“位置裁决 ∧ 字段裁决”。

### 5.1 六个 edge位置

| edge position | H-Ego | R-view | X-domain | Exec | D0裁决 |
|---|---:|---:|---:|---:|---|
| active / edge 0 | C | C/Y | C/Y | Y/C | 保留一个重构后的 active interaction record |
| active / edge 1 | N/C | C/Y sensor | C | C | 现有 Ego未验证；移到 standard challenger，专测双手稳定/额外object relation |
| next1 / edge 0 | R | P | C/Y | C | planner-only |
| next1 / edge 1 | N/R | P | C | N/C | planner-only；无D0覆盖 |
| next2 / edge 0 | R | P | C/Y | N/C | planner-only |
| next2 / edge 1 | N/R | P | C | N | planner-only；无D0覆盖 |

现有 Ego 的低 simultaneous-hand-active只说明第二条 hand edge没有现成证据，不证明目标机器人双手任务不需要第二条 edge。因此 `minimal-r1a` 用一条 record做主最小假设，standard-r1 的第二条 edge保留为清晰、互斥的 bimanual challenger。

### 5.2 每条 edge 的八字段

| edge field | H-Ego | R-view | X-domain | Exec | 独立判断 | `minimal-r1a` |
|---|---:|---:|---:|---:|---|---|
| `actor/presence` | C | Y/C | N（literal side） | N/Y-local | human LEFT/RIGHT不能机械映射 RH56/OmniPicker；执行确需唯一末端，但应由 target allocator+proprio解析 | **从 shared action schema移除**；target-specific allocator/control receipt负责 |
| `primary_node` | Y/C | Y/C | Y | Y | 功能 target role是最强跨域字段；可同时承担 event focus | **保留，并与 focus合并** |
| `primary_part` | C | C | Y/C | Y/C | 功能 part比像素稳定；当前9类只在pour覆盖body/opening/rim子集 | **保留、缩为6类候选** |
| `primary_surface_cell` | C | C | C/N | C | view-dependent part bbox的4×4 cell没有跨域 canonicalization；执行位置重要，但当前编码未证明充分 | **降为 target-local auxiliary/challenger** |
| `secondary_node` | Y/C | Y/C | Y | Y/C | 两对象任务的目标/参照 role是跨域核心；无第二对象时NONE | **保留 optional secondary role** |
| `relation_before` | C/R | C | Y/C | N/C | 是 event-entry历史状态，active中可能已过时；与kind/current perception重复 | **移到 annotation/planner** |
| `relation_after` | C/R | P/C | Y/C | Y/C | 作为 observed after会后见；作为 intended terminal relation有执行价值 | **改成单个 `target_relation`**，由 ontology/causal planner给出 |
| `contact_mode` | C/N | Y/C sensor | Y | N/C | RGB难区分STICK/SLIDE/ROLL/IMPACT；目标域force/tactile/proprio更可靠；更像controller mode | **降 sensor/planner auxiliary** |

### 5.3 Part、site、mode、relation 的具体缩减

`primary_part` 的 `TRIGGER_BUTTON` 在当前三任务族和 Ego pour都没有覆盖；`OPENING` 与 `RIM_EDGE` 对当前粗 locator可先合并。`minimal-r1a` part vocab候选：

```text
BODY
HANDLE_OR_GRASP_FEATURE
LID_CAP
OPENING_OR_RIM
TOOL_TIP
OTHER
```

`OTHER` 仍只表示已知但未列入词表。不可判 part不建议作为 action `UNKNOWN`；若 part对当前 event是required，则 packet fail closed；若该 task profile声明part不适用，则用明确N/A而不是未知。

relation建议只保留一个 intended `target_relation`。至少把 `OVERLAP_2D` 排除出跨域 core：它是相机投影关系，不是跨视角物理 relation。`ALIGNED_TO_RIM/INSIDE/CONTAINS/SUPPORTED_BY` 只有任务 ontology、直接时序或 sensor/native判据成立时才可作为 known target。

## 6. 三组关键重复/冲突

### 6.1 `focus_node` 与 edge role

standard 同时发送 `event.focus_node`、`edge.primary_node` 和 `edge.secondary_node`。dynamic-focus memo用确定性 anchor edge解释它们，但仍允许 pour中 focus=destination、hand edge primary=source、secondary=destination。这个自由度对表达有利，却也制造三份“当前操作谁”的可能冲突。

`minimal-r1a` 用一个 active interaction record解决：

- `primary_role` 是当前最关键、最可能因定位错误导致失败的 role，同时就是 focus；
- `primary_part` 指该 role 的功能部位；
- `secondary_role + target_relation` 表示相对谁形成什么；
- 当前手里已抓住什么由 robot proprio/current contact state提供，不在 shared human packet重复。

例如 pour 的 active transfer可把 destination rim作为 primary、source作为secondary、effect=TRANSFER；acquire source时再把source body/handle作为primary。若该重构在双手或source-motion控制中失败，standard双edge应在 D3 赢，而不是预先假定三份引用都必要。

### 6.2 `actor` 与跨具身 allocator

执行当然需要知道哪个目标末端动作，但这不推出 human `LEFT/RIGHT` 应进入共享 action packet。三种情况应分开：

1. target allocator唯一选择 RH56/OmniPicker：actor是目标域control-plane输出；
2. 两末端等价：literal side不是任务语义；
3. 多个末端后果不同且allocator无唯一解：packet应 hold/reobserve，不让WAM暗选。

因此 actor是 **执行必要、共享表示非必要**。standard 把这两层合在一个 code里，是最需要降解耦的字段。

### 6.3 `effect/outcome` 与 `relation_before/after`

四者存在明显重叠：effect表示变化类型，before/after表示状态差，outcome表示任务判定。对当前动作最小输入应是“要对哪个 role/part造成什么 effect/target relation”。历史 before、事后 outcome和精确验证结果都可由 planner/supervisor保存。

透明倒液尤其不能把 wrist tilt、任务文本或 release推导成 FLOW/TRANSFER/SUCCESS。`effect` 若保留在 action core，必须来自 causal planner的意图或当前直接/sensor证据；`outcome` 则只进入 verify supervisor，不作为 future GT action condition。

## 7. `minimal-r1a` 候选契约

### 7.1 Action-visible core

```text
role namespace:
  manipulated_or_source | tool |
  receptacle_or_destination | fixture_or_support

target-domain node locator:
  center_cell for each role referenced by the active interaction
  # no action-visible observability/scale/orientation

active event:
  compact phase_kind
  object_effect

active interaction record (exactly one):
  primary_role          # also event focus
  primary_part | N/A    # coarse functional part
  secondary_role | NONE
  target_relation | N/A
```

必要语义未知、引用无 current locator、role binding多解或 task profile不一致时，compiler不产生 action packet，控制面执行 `HOLD/REOBSERVE`。reviewed unknown、missing reason、coverage和confidence仍保留在 annotation/planner loss，但不进入 action alphabet。

### 7.2 Action-invisible，但仍应保留的数据

| 层 | 字段 |
|---|---|
| annotation/planner supervision | exact/interval onset、duration、censor、full eight-kind labels、relation_before/after、surface cell、node scale/orientation、contact mode、第二条edge、future tube |
| target robot allocator/controller | target effector assignment、current contact/gripper state、force/tactile、continuous pose/path、safe hold/reobserve |
| supervisor/benchmark | verify outcome、success/partial/failure、effect sensor truth、overflow/invalid reason |
| provenance/loss | observability reason、review state、coverage、confidence、annotator、tracker state、proposal teacher/hash |

“移出 action”不等于删除 annotation。相反，它把哪些字段能训练 planner、哪些字段能驱动低层action、哪些字段只能评测分开。

### 7.3 Standard-r1 与 minimal-r1a 对照

| 维度 | standard-r1 | `minimal-r1a` 候选 | 主要可证伪问题 |
|---|---|---|---|
| role | 4 fixed slots | 同一4-role namespace，只定位active引用节点 | tool/fixture是否真的需要action geometry |
| node | observability+center+scale+orientation | target-domain center only | scale/orientation是否超越相机/布置shortcut |
| temporal | active+next1+next2 | active only | execute-8是否真的受益于future hard codes |
| kind | 8 known + UNKNOWN + ABSENT | 6 known phase候选；unknown fail closed | precontact/hold独立类别是否必要 |
| focus | 独立 `focus_node` | 与primary role合并 | 显式focus是否提供不可派生信息 |
| timing | onset+duration action-visible | planner/control only | coarse timing是否改善控制而非预测分数 |
| effect | action-visible | 保留 | 跨任务“做什么”是否是核心 |
| outcome | verify-only action-visible | supervisor/benchmark | outcome给action是否超越终止/恢复控制 |
| edges | 每event最多2条 | active恰一条interaction record | 双手/双关系是否要求第二条edge |
| actor | LEFT/RIGHT/OBJECT_OBJECT | target allocator/control plane | shared literal side是否造成负迁移 |
| part/site | 9 part × 4×4 site | coarse 6 part；site auxiliary | part-only是否足以ground，site是否需canonicalization |
| relation | before+after | one intended target relation | 历史before是否冗余，after是否后见 |
| contact mode | action-visible | sensor/planner auxiliary | mode是否应由target controller处理 |
| epistemic codes | 多字段UNKNOWN进action | required unknown fail closed | uncertainty action码是否有独立安全行为 |
| capacity | 已有 exact standard-r1 receipt | **TBD，禁止本 memo 伪算** | 新grammar必须重新枚举 |

## 8. 是否删除字段：最终分组

### 8.1 `minimal-r1a` action core保留

- 四个固定功能 role 的 namespace；
- active引用 role 的 target-domain `center_cell`；
- 精简的 active `kind`；
- active `object_effect`；
- 一个 active `primary_role/primary_part/secondary_role/target_relation` record。

### 8.2 合并或机械派生

- `focus_node` 合并为 active `primary_role`；
- `relation_before/after` 收敛为一个 intended `target_relation`，原始pre/post仍留 annotation；
- `APPROACH+PRECONTACT`、`HOLD+MANIPULATE` 作为候选合并；
- node presence由 active reference union机械派生；
- target actor由目标机器人 allocator/proprio机械解析。

### 8.3 降为 auxiliary/control/provenance

- node observability、scale、orientation；
- onset、duration、censor细节；
- surface cell与完整part-local geometry；
- contact mode；
- active outcome；
- full eight-kind标签和第二条edge；
- next1/next2 的全部 event/edge fields；
- reviewed unknown、missing reason、coverage、confidence与tracker validity。

### 8.4 从当前 action core删除

- active onset常量；
- future outcome；
- human literal LEFT/RIGHT actor作为共享字段；
- `OVERLAP_2D` 作为跨域 relation；
- 无独立证据的 tool/fixture geometry；
- 任何把 annotation uncertainty 本身当执行命令、却没有安全行为定义的 UNKNOWN action code。

## 9. D0/D1/D2/D3 的最小验证顺序

### D0-r1a：只做机械映射

1. 每个 retained field必须有唯一 annotation/robot source；
2. active primary role就是focus，不能出现第二份冲突引用；
3. required center/part/relation缺失时 fail closed；
4. annotation UNKNOWN/MISSING/ABSENT/censor不被编成 action unknown；
5. target actor allocator与 shared packet分离；
6. schema agent重新枚举 reachable grammar，冻结 exact capacity、codec、equal-bit control；
7. 在第6项完成前不得沿用 standard的207.7497-bit receipt。

### D1：human pour只验证实际覆盖字段

优先双标：active coarse kind、primary role/part、secondary role、target relation、effect、source/target center。tool、fixture、robot center、双edge、actor allocator和完整 outcome继续 `NOT_EVALUABLE`。

固定布置下还必须报告字段 prevalence/entropy：若 source/target center、scale或orientation几乎常量，高 agreement不构成保留理由。

### D2：目标机器人与三任务族

专门检验：

- tool/fixture role是否需要进入 action；
- part-only与part+cell的差异；
- active-only与active+next1/+next2的差异；
- one-interaction与two-edge在bimanual twist/pour中的差异；
- allocator actor与literal actor的差异；
- center-only与center+scale/orientation的差异。

### D3：oracle action sufficiency

至少互斥比较：`minimal-r1a / standard-r1 / 22D / whole-packet equal-bit / full trajectory`。字段 promotion只在 matched data、token/参数/FLOPs和 target-robot native truth下，以 task/phase success、correct contact/release、安全和 cluster CI决定。annotation F1提高本身不能把字段升回 action。

## 10. Promotion 与停止规则

只有同时满足以下条件，字段才能从 auxiliary升回 action：

1. human和target-robot域都有可复核 source；
2. 部署时能因果获得，不依赖future GT或human像素复制；
3. 不能由 retained role/effect/relation、robot proprio、text或control plane机械派生；
4. field intervention在对应phase产生局部且可解释的动作变化；
5. 相对 `minimal-r1a` 有预注册的策略增益和安全不退化；
6. 新 schema/grammar/capacity/equal-bit receipt完整。

具体止损：

- second edge只在bimanual/双关系子集有独立增益时promotion；
- site cell若无跨域canonicalization或只提高标注F1，不promotion；
- scale/orientation若收益只来自固定相机/固定布置，不promotion；
- future slots若active-only在execute-8下不劣，不promotion；
- duration/onset若只改善event prediction而不改善rollout，不promotion；
- outcome若 supervisor route与action-visible route持平，保持control-plane；
- contact mode若robot sensor/controller已经吸收其增益，保持auxiliary；
- literal actor若不优于allocator且出现wrong-side风险，永久移出shared schema。

## 11. 对 standard-r1 的最终评价

standard-r1 的优点是真正 machine-checkable：固定 role、prefix、reference union、canonical edge order、unknown/missing/censor分离和 exact enumerative capacity都比开放连续latent更可审计。问题不在工程严谨性，而在 **把信息上界误当执行最小集**。

当前证据下，它至少在五处超前于数据：

1. 用三 event action tube编码尚未证明需要的future plan；
2. 用双 edge编码现有 Ego几乎没有可靠覆盖的双手/双关系；
3. 用scale/orientation/site表达相机局部几何，却缺跨域canonicalization和控制必要性；
4. 用literal actor混合human laterality与target-robot effector assignment；
5. 用outcome、duration、observability/UNKNOWN把supervision、forecast和epistemic state纳入action。

因此唯一建议是：

> **把 standard-r1 降为预注册的高表达 finite upper bound；先让 `minimal-r1a` 作为D0/D1的最小 action候选接受反证。** 若 D2/D3证明 future、second edge、site、mode或orientation有不可替代的目标机器人收益，再逐项 promotion；不要先把82-byte字段全集冻结为默认主路线，再靠unknown和fail-closed掩盖覆盖不足。

## 12. Idea-discussion 登记

本轮按 Zotero-first workflow复用总体 idea card，并以 critic mode登记独立最小化问题：

```text
idea_id        = b4ec9e69-e9aa-4025-8d79-e36df14e0a5e
discussion_id  = 126b612c-3da6-4a77-8526-0394e60c9be0
memory_updated = true
zotero_synced  = 100 records
```

自动 critic只给出“需把 novelty拆成任务/方法/数据/评测”的泛化风险，并召回 ST-VLA、ST-π 以及一个明显低相关的 Minecraft action-space条目；这些结果没有被用作逐字段保留依据。本 memo 的字段裁决来自上位内部契约、认知证据和对本地 Ego corpus 的直接只读审计。

## 13. 只读来源

统一索引与指定专题均从 integration worktree HEAD `61050a48e5a14b7806a131931997010479ffd661` 读取：

- `docs/research/WAM_BIOCHEMICAL_IDEA_INDEX_20260812.md`
- `docs/research/sm_fet_wam_v2_schema_architecture_20260812.md`
- `docs/research/sm_fet_annotation_and_pairing_protocol_20260812.md`
- `docs/research/sm_fet_dynamic_focus_token_spec_20260812.md`
- `docs/research/sm_fet_missing_hands_contract_20260812.md`
- `docs/research/sm_fet_missingness_capacity_reconciliation_20260812.md`
- `docs/research/sm_fet_transparent_nonreconstruction_contract_20260812.md`
- `docs/research/human_observational_tool_learning_evidence_20260812.md`
- `docs/research/sm_fet_wam_benchmark_and_data_roadmap_20260812.md`

本轮没有读取同题 D0 handbook/reduction memo，没有使用现有 contact annotation当 GT，也没有修改 corpus。
