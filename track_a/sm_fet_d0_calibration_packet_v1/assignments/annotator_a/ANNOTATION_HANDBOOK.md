# SM-FET D0 人工标注 Handbook：A/B 标注员操作手册

- 日期：2026-08-12
- 基线：canonical `e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 适用对象：D0 校准讨论中的 A/B 标注员；D0 通过后、获得协调员正式授权的 D1 blind A/B 标注员
- 文档性质：人工操作与判定 memo；不是 UI、checker、CSV schema、GT、训练或模型实现
- 当前状态：D0 checker/round-trip 尚未实现，D0 与 D1 均未通过

## 0. 先读这一页：现在允许做什么

本手册把 SM-FET standard-r1 的人工判断压成一套可执行顺序：先看完整视频，再标 coverage/contact，随后按时间切 macro event，逐事件判断 focus、四个固定角色、node、edge、relation、effect/outcome，最后才做 entry/turn/exit 的稀疏 2D pose probe。

但请先记住三个不可越过的边界：

1. **本手册不产生 GT。** A/B 独立记录只有在一致性 gate 通过、第三人裁决并完成 receipt/hash 绑定后，裁决结果才可能成为 gold。校准讨论、自动 proposal、模型输出与本手册本身都不是 GT。
2. **现有 offline contact UI 不是 SM-FET UI。** 它只允许改当前 7-file packet 中的 `coverage.csv` 与 `segments.csv`。本手册中的 event/focus/node/edge/pose 决策树是 D0 契约，不得自行加列、加 CSV、写入 notes、修改 receipt 或把临时 worksheet 放进正式 packet。
3. **正式 blind 不能抢跑。** 完整 SM-FET D1 必须等待独立 sibling UI/checker、annotation→82-code round-trip 与 D0 receipt 通过。当前工具若被协调员授权，只能完成既有 contact v1.1 范围，不能声称已完成 SM-FET 标注。

遇到任何无法按本手册唯一表达的情况，选择 `REVIEWED_UNKNOWN`、`NOT_REVIEWED/MISSING`、censor、overflow 或 invalid；不要为了“填满表格”猜值。

## 1. 规范优先级与固定不变量

本手册以统一 research index 的优先级为准：final architecture / standard-r1 schema / 修订 annotation protocol，高于专题 contract；专题 contract 高于旧 idea memo。具体读取版本为：

- 统一索引 `WAM_BIOCHEMICAL_IDEA_INDEX_20260812.md`，source commit `61050a48e5a14b7806a131931997010479ffd661`；
- 修订 annotation/pairing protocol，source commit `288ef5dcf4e6299871dc50d33caa603e46345fcd`；
- dynamic focus contract，source commit `2452e60bbaf6edb4714fed7d1397aea6ba9e4111`；
- missing-hands fix1 与 capacity reconciliation，source commits `ca7d641703749a15ff30c644250f5147b5125e72`、`9f6eeff6761168048b49101cb5839170ae6b84bc`；
- transparent non-reconstruction contract，source commit `f795c4f615595d72219e79a0cb43d5adef12b0a0`；
- offline contact annotation UI/protocol，source commit `be93be7bdbe3c340d389ab4bc196fe086d7da04e`。

标注时不得改变以下不变量：

- 事件语义只有八类：`approach`、`precontact`、`acquire`、`hold`、`manipulate`、`regrasp`、`release`、`verify_outcome`。
- `UNKNOWN` 是已复核但不可判的字段值，不是第九种事件；`ABSENT` 是确认不用该 slot，不是事件；`MISSING/NOT_REVIEWED` 是 mask，不是 wire code。
- 四个 node role 永远按固定顺序：`manipulated_or_source`、`tool`、`receptacle_or_destination`、`fixture_or_support`。
- 一个 event 最多两条 edge；双手并发放在同一个 macro event 的两条 edge 中，不创建两个同 onset event。
- `active_hand` 只用于人工 QA；它不进入 action wire，也不能把 human left/right 字面对齐到 robot left/right。
- `focus_node` 没有 UNKNOWN code；actor 与 primary node 也没有 UNKNOWN code。它们无法确定时必须 fail closed，不能借别的 UNKNOWN 补洞。
- `outcome` 只属于 `verify_outcome`。`release` 的 action outcome 永远为 N/A/0。
- entry/turn/exit hand pose、连续 bbox/UV、depth、MANO、tracker、confidence 和 proposal 全部 action-invisible。
- formal standard-r1 仍是 82 个 canonical `uint8` codes、13 个 hard tokens、exact reachable capacity `207.749746919520 bit`；本手册不增加任何 action-visible bit。

## 2. D0 校准讨论与 D1 blind 独立性

### 2.1 D0：允许讨论，但不计分、不提交 GT

为便于操作隔离，本手册把协议冻结前的三段 calibration 工作称为 **D0 calibration working session**；在上位研究协议的 gate 统计中，它仍是 D1“3 calibration + 12 blind”的前置 calibration，不改变 D0/D1 的正式定义。

协调员须在看结果前从 12 段 blind 之外冻结三段 calibration：

1. 一段边界清楚、无 regrasp 的标准成功；
2. 一段含长 hold、双手或 regrasp 的高复杂度样本；
3. 一段含遮挡、interval/censor、effect 不可判或 partial/failure 的 missingness 样本。

D0 calibration 的操作规则是：

- A、B 先各自完整做完三段，再共同查看分歧；不能一边讨论一边填第一版。
- 可以讨论定义、反例、边界和 unknown/mask 选择；协调员最多修订一次 dictionary/task ontology。
- calibration clip 与 12 段 blind 不重叠，不进入 blind agreement、validation/test 或阈值调参。
- calibration 媒体使用独立、带 receipt 的 D0-only ACL root；不得混入 12-MP4 blind media root，D1 开始前撤销该访问。
- 讨论结果只用于冻结 protocol/hash 与 D0 checker 用例；不进入正式 packet，不升级为 GT。
- D0 还必须由 checker/round-trip 证明 UNKNOWN、ABSENT、MISSING、censor、focus/reference、overflow 和 82-code 映射唯一。本手册写完不等于 D0 通过。

### 2.2 D1：协议冻结后，全程独立

协调员宣布冻结的 protocol/hash 与任务 ontology 后，D1 blind 才可开始。此后：

- A/B 只能看各自获批的 packet、12 个原始 MP4 和冻结指南；不能共享屏幕、聊天讨论、ID、中间 CSV、autosave 或选择结果。
- 不得在 blind 过程中改定义、换 episode、补看 calibration 讨论记录或请求“统一答案”。出现新型歧义时按现行规则标 unknown/missing/overflow，并在轮次结束后由协调员统计。
- 不能看另一标注员、coordinator、adjudication、旧 GT、proposal、prescreen、solver score、timing hint 或模型输出。
- 只有 aggregate agreement gate 通过后，第三人才可同时看 A/B 与原视频做裁决。裁决不能把 `POSITIVE_ONLY` 升为 `EXHAUSTIVE`，不能把 interval/censor 压成 exact，也不能把 reviewed unknown 猜成 known。
- 若 gate 失败，不能靠裁决“洗掉”分歧；最多另立一次新 protocol version，并使用新的、不重叠 calibration。

## 3. 12 个 blind episode、media allowlist 与 ACL

### 3.1 固定 assignment

12 个 blind episode 的集合和顺序固定如下。它们共 3,695 个 processed 20 Hz frames、184.75 秒；看到视频后不得换样本，episode 0 的历史 timing hint 始终隐藏。

| 顺序 | `task_id` | `episode_id` | processed 区间 | 视频时长 | 建议人工 timebox |
|---:|---|---:|---|---:|---:|
| 1 | `contact-min12-00` | 43 | `[0,264)` | 13.20 s | 17–24 min |
| 2 | `contact-min12-01` | 50 | `[0,223)` | 11.15 s | 15–20 min |
| 3 | `contact-min12-02` | 51 | `[0,263)` | 13.15 s | 17–24 min |
| 4 | `contact-min12-03` | 58 | `[0,294)` | 14.70 s | 19–26 min |
| 5 | `contact-min12-04` | 34 | `[0,294)` | 14.70 s | 19–26 min |
| 6 | `contact-min12-05` | 38 | `[0,293)` | 14.65 s | 19–26 min |
| 7 | `contact-min12-06` | 9 | `[0,324)` | 16.20 s | 21–29 min |
| 8 | `contact-min12-07` | 54 | `[0,314)` | 15.70 s | 20–28 min |
| 9 | `contact-min12-08` | 36 | `[0,314)` | 15.70 s | 20–28 min |
| 10 | `contact-min12-09` | 53 | `[0,344)` | 17.20 s | 22–31 min |
| 11 | `contact-min12-10` | 12 | `[0,354)` | 17.70 s | 23–32 min |
| 12 | `contact-min12-11` | 0 | `[0,414)` | 20.70 s | 27–37 min |

timebox 是把修订协议中每位标注员 4–5.5 小时的 integrated pass 预算按视频时长比例分配后的操作参考，不是质量门槛。遮挡、双手或透明场景超时，仍须诚实标 unknown/mask；不要为赶时间猜标签。超时情况在轮次结束后单独报告协调员，不在 packet 内新增日志或 notes。

### 3.2 formal packet 永远恰好七个文件

每位标注员的 packet 根目录只能有：

1. `ANNOTATION_GUIDE.md`
2. `assignment_receipt.json`
3. `identity.json`
4. `tasks.csv`
5. `timeline_evidence_index.csv`
6. `coverage.csv`
7. `segments.csv`

不得把本 handbook、媒体、截图、autosave、恢复 JSON、staged output、submission receipt、临时 worksheet 或未来 SM-FET 表放进 packet 根目录。当前工具只修改最后两份 CSV；前五份是不可变输入。

### 3.3 media/state/packet 的隔离原则

- media root 必须与 packet 分开，且恰好有 receipt 指定的 12 个 `<task_id>.mp4`；文件名、顺序、大小和 SHA-256 都必须匹配。出现第 13 个文件、缺片、hash 不符或重命名时立即停止。
- packet、media root、state 三者必须两两不同，不得互为父子、不得指向同一 inode，不得使用 symlink 或 hard link。
- packet/state 根目录仅 owner 可进入；五个不可变 packet 文件和媒体只读；`coverage.csv`、`segments.csv` 仅 owner 可读写。机构 ACL/专用隔离组可提供等价或更严格隔离。
- A 只能访问 A 的 packet/media/state，B 只能访问 B 的；两人都不能访问 source root、processed HDF5、coordinator、adjudication、另一方目录或整个仓库。
- 只用无同步账号的本地非持久浏览器打开离线页面；不启动 Web server，不上传云盘，不截图、录屏、转发或二次使用媒体。
- state 只能保存获批的锁定 receipt、autosave、恢复状态、staged output 与 precommit CSV 备份；不得保存媒体、缩略图、截图、绝对 source 路径或 HDF5 信息。
- 发现权限过宽、错误分发、额外文件、身份/session 漂移或 hash 不符，立即停止并报告；不得通过改 receipt、关检查或藏文件继续。

## 4. 每个 episode 的固定操作顺序

### Pass 0：先看完整视频，不做标记

先以原速从头到尾看完整 episode 一遍，目的是知道任务角色、是否有多次 subgoal、是否存在 left/right censor、遮挡和结果检查。**不能先从局部片段、proposal window 或“看起来最像接触”的时刻开始。**

完整看完后，才可使用 0.5×、0.25×、逐 processed 帧与 ±10 processed 帧上下文定位证据。所有正式帧号使用 packet 提供的 processed 20 Hz 时钟；原视频时间只用于定位，不能另造一套时钟。

### Pass 1：coverage 与 contact segment

1. 逐帧复核完整 coverage。
2. 只有能够声明“列出的接触是全部正例，未列帧确实无接触”时选 `exhaustive`。
3. 能确认正例但不能排除遮挡/画外遗漏时选 `positive_only`。
4. 整段无法形成可用复核时选 legacy `unknown` coverage；此时不得写 contact segment。
5. 每个真实 task-relevant physical contact 持续区间只写一次，使用半开区间 `[start_frame,end_frame)`：start 是第一帧确认接触，end 是第一帧确认已不接触。不要按窗口切碎。
6. 当前 contact v1.1 分类值证据不足时按现有 UI 保持空白；不要在 legacy CSV 写字面量 `unknown`、`none` 或 `N/A`。

### Pass 2：按时间切 macro event

从 episode 最早可确认的 task-relevant event 开始，沿时间向后切 maximal-duration event。相邻同类先合并，不因每个窗口、每次回看或手腕姿态变化重复编号。

合法 successor 只有：

- `approach` → `precontact` 或 `acquire`
- `precontact` → `acquire`
- `acquire` → `hold`、`manipulate`、`regrasp` 或 `release`
- `hold` → `manipulate`、`regrasp` 或 `release`
- `manipulate` → `regrasp`、`release` 或 `verify_outcome`
- `regrasp` → `hold`、`manipulate` 或 `release`
- `release` → `verify_outcome` 或新的 `approach`
- `verify_outcome` → 新的 `approach`
- 任一 known kind → `UNKNOWN`；`UNKNOWN` 必须终止当前 tube

逐类判断：

- `approach`：执行手开始持续、任务定向地接近 primary role entity。普通走动、抬手、镜头调整不是 approach。
- `precontact`：进入接触前对准/微调区，且确认尚未接触。只有 2D overlap 不算接触。
- `acquire`：从首次 task-relevant contact 到稳定可操控 engagement。`hand_status=closed` 不能定边界；impact 也不能跳过 acquire，可用极短或 interval acquire。
- `hold`：稳定接触，但尚未制造主 relation/effect 变化。等待、稳定和对准属于 hold。
- `manipulate`：接触配置开始驱动预注册任务效果；`actuate` 统一归入此类，效果另填 `object_effect`。
- `regrasp`：原稳定接触拓扑有意解除、迁移或重新分配，直到新拓扑稳定。只改变腕姿、未改接触拓扑，不是 regrasp。
- `release`：开始有意终止最后一条 task-relevant contact，直到确认分离或 coverage 结束右删失。效果发生不等于 release；松手失败且仍接触也不产生 release。
- `verify_outcome`：主效果已可观察，执行进入检查/稳定等待，结果可按冻结判据判断。脚本结束、松手或“看起来做完了”都不足以建立 verify。

特殊情况：

- clip 开始时 event 已存在，可从 `hold/manipulate/regrasp/release/verify_outcome` 开始并标 left censor；不是非法 predecessor。
- `precontact` 后无接触而放弃：关闭当前 path，走 action-invisible failure/NULL；不要伪造 `release`。
- 双手同时参与：同一个 macro event 内放两条 edge；不要切两个同 onset event。
- 交替双手：按真实拓扑切为 left-only → bimanual → right-only 等事件；不能在一个 event 内静默换 actor。
- 同一 `manipulate` 内必须频繁切 focus/cell 才能描述：不要制造同类 self-loop；标记 ontology/granularity No-Go 供 D0 处理。
- 一个 episode 超过 8 个 macro event 时记 `Q_episode>8` overflow；不得把后续事件截断、合并成错误 kind 或删掉失败段。

### Pass 3：边界与 support

事件表的两类时间不要混淆：

- 保守 positive support 用半开区间 `[support_start_frame,support_end_frame)`；这些帧可确认属于该 event，但不声称两端就是真实 transition。
- 真实 start/end boundary 可为 `exact`、`interval`、left/right censor 或 `unknown`。`interval` 保存 transition 可能落入的闭区间 `[lo,hi]`；不要取中点冒充 exact。

选择顺序：

1. 看见真实 transition 且能落到一帧：`exact`。
2. 能把 transition 限定在若干帧之间：`interval`。
3. onset 在 coverage 开始前已发生：只对 start 选 `left_censored`。
4. termination 在 coverage 结束后仍未发生：只对 end 选 `right_censored`。
5. event 确认存在、保守 support 存在，但遮挡/模糊使真实边界无法给 support：`unknown`。
6. 没看或没有证据：`NOT_REVIEWED/MISSING`，对应 mask 为 0。

遮挡不是 censor；coverage 边界也不能冒充真实边界。duration 不由标注员另填浮点数，由 checker 从边界机械推导。

### Pass 4：focus 与四角色 node snapshot

先按 task ontology 固定实体到四 role 的一对一 mapping，再逐 event 选 focus。不能根据画面显眼程度临时换 slot。

每个实际查看的 `entry`、可选 `turn`、`exit` 或 `anchor` snapshot 都必须按固定顺序检查四个 role。role 不在当前 active+next1+next2 reference union 时，packet 中由 compiler 机械变为 `ABSENT`；这不表示物理对象不存在，也不允许标注者用“看不见”手工填 ABSENT。

### Pass 5：edge、relation、effect 与 outcome

对每个 event 最多建立两条有证据的 edge，依次判断 actor、primary role/entity、primary part、part-local 4×4 cell、optional secondary role、contact mode 和 object relation。超过两条时标 `EDGE_OVERFLOW`，绝不静默删掉第三条。

最后判断 event-level effect；只有 event 是 `verify_outcome` 时才判断 outcome。不要从任务意图、文件名、手腕倾斜、松手或脚本结束反推 effect/outcome。

### Pass 6：entry/turn/exit 稀疏 pose probe

pose 只在 core 事件/关系判断完成后标，避免手姿反过来“证明”contact/effect。它是 auxiliary/probe，不能补 core，也不能进入 action wire。

### Pass 7：episode 内自检

按本手册第 10 节逐项检查后再进入下一 episode。不要等 12 段全部做完才发现从第一段开始误解了半开区间或 unknown/censor。

## 5. `focus_node` 决策树

对每个 event 按以下问题从上到下走：

1. **task ontology 是否给出了唯一 role→entity binding？**
   - 否：同 role 多实例记 overflow；实体/role 无法解析记 invalid reference。不要自己挑“最像”的物体。
   - 是：继续。
2. **哪个 role 是当前 event 中必须优先重定位、且误定位最可能导致该 event 失败的对象？**
   - 选该 role 为 `focus_role`；它不是最显眼物体，也不必是手当前抓着的物体。
3. **该 focus 在整个 maximal event 内是否保持同一语义？**
   - 是：固定到 event 结束。
   - 否：只有真实发生合法 event transition 才切新 focus；若只是 attention 在逐帧移动，不切 event、不造 self-loop，标为当前 finite ontology 的失败模式。
4. **focus role/entity 能否解析？**
   - 否：focus 没有 UNKNOWN code，packet `INVALID_REFERENCE`，不能猜。
   - 是但画面看不清：focus binding 仍可 known；node observability/part/cell 各自判断 unknown/mask，不要把空间不可见等同于 focus 不存在。

pour profile 中常见但不是按任务名硬填的参照：acquire source 通常 focus `manipulated_or_source`；对准目标杯口并倾倒时可能 focus `receptacle_or_destination`；验证转移结果通常也是 destination。最终仍以当前 event 的失败敏感角色和原始视频证据为准。

## 6. 四个 role/node 决策树

### 6.1 先绑定角色

1. `manipulated_or_source`：当前被直接拿持、移动、转动或作为物质来源的主对象。
2. `tool`：真实、独立的 pipette、stirrer、funnel、cap-as-tool 等功能中介。液体、stream、rim、杯口不是自由 tool node。
3. `receptacle_or_destination`：接收、盛放或作为目标关系终点的对象。
4. `fixture_or_support`：被 current/next events 引用的桌面、rack、holder 等固定支撑。

同一 role 同时需要两个实例、或无法唯一绑定时，标 overflow/invalid；不得把多余对象挤到另一个语义不对的 slot。

### 6.2 再判断 observability

对 union 内每个已绑定 node：

1. 对象清楚可见且可可靠定位：`VISIBLE`；center cell 必须 known。
2. 对象只部分可见，但可可靠定位：`PARTIAL`；center 可 known 或 reviewed unknown。
3. 对象被遮挡，但前后时序足以持续追踪其位置：`OCCLUDED_TRACKABLE`；center 可 known 或 reviewed unknown。
4. 对象 task-relevant geometry 明确在画外：`OUT_OF_VIEW`；center/scale/orientation 全部为各自 UNKNOWN。
5. 确认该 node 被引用并已复核，但观测状态本身不可判：`UNKNOWN`；后三项也 UNKNOWN。
6. 没看、没覆盖或没有证据：`NOT_REVIEWED/MISSING`；全部 mask，不编 UNKNOWN。
7. `ABSENT` 只由 ontology/reference union/exhaustive negative 的机械规则产生；绝不能因透明、遮挡、出画或 detector 没框而选择。

center/scale/orientation 的规则：

- center 是整张当前域图像的 4×4 cell；support 跨多个 cell 且无法唯一量化时选 reviewed unknown，不取 bbox 中心硬凑。
- scale 只选冻结的 `SMALL/MEDIUM/LARGE`；卡在未冻结或跨 band 的支持范围时 unknown。
- orientation 使用 8 个 45° bin；跨 bin、透明轮廓不清或姿态不可判时 unknown。
- global center/scale/orientation 是当前相机自己的观测，不是 human→robot 坐标 target，也不能作为默认跨域 pairing 距离。

## 7. edge / relation / effect / outcome 决策树

### 7.1 edge 与 actor

1. **是否有 task-relevant hand/object contact 或需要显式表达的 object-object relation？**
   - 否，且 coverage 为 `EXHAUSTIVE`：相应尾部 edge 可机械 `ABSENT`。
   - 否，但 coverage 非 exhaustive：是 MISSING，不是 ABSENT。
   - 是：继续。
2. **actor 是谁？**
   - 确认左手：`LEFT`。
   - 确认右手：`RIGHT`。
   - 确认是对象—对象 relation/contact：`OBJECT_OBJECT`。
   - 确认有手参与，但多帧复核后 side 仍不可判：annotation 保存 `CONFIRMED_PRESENT + REVIEWED_UNKNOWN`，actor mask=0；整个 oracle/action packet inadmissible。不要写 `ABSENT`，也不要借 event `UNKNOWN`。
   - 可能有第二只手但未复核：hand roster incomplete；不能把已知一只手提交为“完整单手”oracle。
3. **同一 event 是否同时有 LEFT 与 RIGHT edge？**
   - 是：这是 bimanual bundle，`active_hand=both` 只作 QA；不是一个 `BOTH` actor code。
4. **是否超过两条 edge？**
   - 是：`EDGE_OVERFLOW`。透明 pour 中可在 effector→source edge 的 secondary/relation 表达 source→target；仍无法无歧义表达时停止，不截断。

`hand_active`、`hand_status=closed`、mask overlap、gripper command、tracker valid 都不能决定 actor、contact 或 acquire/release 边界。

### 7.2 primary role / part / surface cell

1. primary role 必须解析到四 role 之一并出现在同 anchor 的 present node union；否则 invalid reference。
2. primary part 只用 `body`、`handle`、`lid_cap`、`opening`、`rim_edge`、`trigger_button`、`tool_tip`、`other`。
3. `other` 表示部位已知但不在词表；`UNKNOWN` 表示已复核但部位不可判；二者不能互换。
4. surface cell 是当前域该 primary part 局部框内的 4×4 cell。只看得出 part、看不出 cell 时 part known、cell reviewed unknown；不要从 human 像素复制到 robot。
5. 手抓 source body 时 primary part 就是 `body/handle`，不能为了表达倾倒目标而伪写成 source rim。

### 7.3 secondary 与 relation

1. 没有 secondary node：`secondary_node=NONE`，`relation_before/after` 必须 N/A/0，不填 UNKNOWN。
2. 有 secondary node：relation 描述 primary object 与 secondary object 的对象关系，不是手型、手指开合或手接触前后状态。
3. 可用 relation 只有 `separate`、`near`、`touching`、`overlap_2d`、`inside`、`contains`、`aligned_to_rim`、`supported_by`。
4. 只有投影重叠时最多是 `overlap_2d`；不能推出 `touching`、`inside` 或 3D alignment。
5. `inside/contains` 必须有直接时序证据或同步 sensor/native truth。
6. `aligned_to_rim` 只有 source/target rim 都可定位，且冻结的 image-plane predicate 可判时才 known；它不声称 3D 同轴。
7. 已复核但 before/after 不可判：对应 relation `REVIEWED_UNKNOWN`；未复核/无证据：MISSING/mask。

### 7.4 effect

按冻结 task ontology 的 observable criterion 选择 `none`、`translate`、`rotate`、`articulate`、`flow`、`deform`、`transfer`：

1. 原始 RGB/时序或获批同步 sensor 是否直接支持对象变化？支持才选 known effect。
2. 只看到执行意图、手腕倾斜、任务名称、脚本步骤或模型预测：不能据此选 effect。
3. 看不到 stream、meniscus、目标液位变化，又无秤/液位/flux 证据：`flow/transfer` 必须 reviewed unknown 或 MISSING。
4. effect 在 `manipulate` 中发生，不要求另造 event kind；effect observed 也不自动推出 intended outcome。

### 7.5 outcome

1. event 不是 `verify_outcome`：outcome 一律 N/A/0，包括 `release`。
2. event 是 `verify_outcome`，且冻结 criterion 明确满足：`success`。
3. 只完成一部分 criterion：`partial`。
4. criterion 明确失败：`failure`。
5. 确认存在真实检查阶段，但结果已复核仍不可判：`REVIEWED_UNKNOWN`。
6. 没有可见或 sensor 结果，只是“动作做完/松手/脚本结束”：不要把它当 success；若不满足 verify 建立条件，不强建 verify event。

## 8. entry / turn / exit 与手部不确定性

### 8.1 何时选 entry、turn、exit、anchor

- 持续 event 默认在实际查看的 source-clock 帧标 `entry + exit`；两帧都是事件状态 snapshot，不自动等于真实 onset/offset。
- `turn` 每个 event 最多加一个，只在存在方向反转、最大开合、contact part/mode 改变、relation/contact 拓扑改变或 regrasp 关键点时使用。普通抖动、腕姿小变、每一帧都不同不是 turn。
- 持续不足 3 帧，或无法诚实分出 entry 与 exit 时只标一个 `anchor`；不得复制同一坐标充作 entry/exit。
- node snapshot 的每个实际 anchor 都检查四个固定 role；pose probe 则对每只 active 且可定位的 hand 分别处理。
- `active_hand=none/object_only` 不要求 hand pose。

### 8.2 七个 hand landmarks

仅记录 wrist、palm center、thumb/index/middle/ring/little tip。对每个 landmark：

1. 清楚可见：visibility=`visible`，填写实际坐标。
2. 被自身遮挡：`self_occluded`，坐标留空。
3. 被对象遮挡：`object_occluded`，坐标留空。
4. 出画：`out_of_frame`，坐标留空。
5. 运动模糊不可定位：`motion_blur`，坐标留空。

不能插值、镜像另一只手、沿手型想象补点，也不能用 MANO/tracker/depth 填坐标。reference bbox 不可定位时留空，对象局部 pose 量 mask。aperture/grasp family/palm axis 已复核但不可判时保持 unknown；未复核则 mask。

### 8.3 双手与 side ambiguity

- 两手接触有时间重叠：同一 bimanual macro event 中分别建立 LEFT、RIGHT edge；不要建立两个平行 event。
- 左右交替且无重叠：按真实 topology 切事件。若有短 handover 重叠，序列是 single → bimanual → single。
- 一只手已知、另一只手可能参与但未审：`hand_roster_coverage` 不完整，整个 oracle packet不可 admission。
- 确认有手 edge、primary role也可解析，但多帧仍看不出 anatomical side：保存 confirmed presence 与其它 known payload；actor reviewed unknown/mask，packet fail closed。
- occluded、out-of-frame、motion blur 与 tracker inactive 是观测原因，不是 `NOT_INVOLVED`。只有 exhaustive review 才能确认某手未参与。

## 9. UNKNOWN、censor、mask、ABSENT：一张决策表

| 你实际知道什么 | 应选状态 | 后果 |
|---|---|---|
| 完整复核后确认 slot/event/edge 不存在或不用 | `ABSENT`，但只限相应 exhaustive/union 规则 | presence negative；payload 机械为 literal positive zero |
| 确认实体/event/edge/字段存在，已复核但值不可判 | `REVIEWED_UNKNOWN` | 只有该字段有 UNKNOWN code时进入 UNKNOWN CE；其它 known 字段保留 |
| 没看、没有证据、coverage 未覆盖 | `NOT_REVIEWED/MISSING` | `field_known_mask=0`，不产生 target；不得自动变 UNKNOWN/ABSENT |
| onset 在 coverage 开始前已发生 | `LEFT_CENSORED` | 只可作为当前 active；不是模糊/遮挡 |
| offset 在 coverage 结束后仍未发生 | `RIGHT_CENSORED` | duration right-censored；不是 GT96 |
| event 存在，但遮挡使 transition 位置不可判 | boundary `UNKNOWN` | 不是 censor；保留 confirmed-positive support |
| focus、actor、primary node等无 UNKNOWN code的 required reference 不可判 | invalid / actor mask + packet inadmissible | 部署应 hold/reobserve；不能猜值 |
| landmark 不可见 | 对应 visibility reason + 坐标 mask | 不能补点，不能清除其它可见 landmark |

特别注意 legacy UI 与未来 SM-FET 逻辑的表示差异：当前 `segments.csv` 的未知 categorical 仍按 contact v1.1 留空，不能写字面量 `unknown`；未来 sibling schema 必须用 value + review state 明确区分 `REVIEWED_UNKNOWN` 与 `NOT_REVIEWED`。在 D0 未实现该 schema 前，不要自行发明编码。

## 10. 严禁泄漏：你只能用什么证据

### 10.1 blind allowlist

可以使用：

- 你自己的冻结 7-file packet；
- 你自己的、hash 匹配的 12 个原始 MP4；
- `timeline_evidence_index.csv` 提供的 opaque frame/time 映射；
- 原速、慢速、逐帧和 ±10 processed frames 的原始视频上下文；
- 经正式 packet 明确授权、且同轮 A/B 都可获得的冻结 task ontology/guide。

### 10.2 永久 denylist

以下内容即使“有助于判断”也禁止用于 A/B truth：

- processed HDF5 及其绝对路径、数组、timestamp、动作/状态字段；
- HDF5 或 sidecar 中的 `hand_status`、`hand_active`、grip proxy、closed/open proxy；
- raw/completed depth、depth hole、mesh、NeRF、MANO/SMPL、单目 3D、光流或 tracker 关键点；
- contact/event detector、mask overlap、geometry proposal、VLM relation/effect、CAM、heatmap；
- prescreen、episode 0 timing hint、旧 GT、另一标注员 CSV、裁决结果；
- DTW/OT/CAPOT coupling、solver mass、模型分数、embedding 或 learned feature；
- 根据任务名、文件序号、外观、时间空档猜 operator/session/object/liquid/batch。

proposal 可以在另名 proposal-only benchmark 中保留，但不能显示给 blind A/B，不能由一个“接受”按钮升级为 human evidence，也不能改变 mask。

本 12 段 blind assignment 没有向标注员分发任何 sensor/native truth，因此本轮所有“sensor-confirmed”分支均不可选；它们只保留给未来另立 receipt、A/B 同等可见且经正式授权的 campaign。不要把 HDF5/depth/proxy 当作替代 sensor。

## 11. 透明、遮挡与液体的硬规则

1. 不重建不可见世界：不标透明 depth、mesh、液体体积、流速、接触界面、metric rim offset 或 spill fraction，除非有正式 native/sensor truth。
2. 只画/标可见支持：partial rim/boundary 只能保留可见段；`PRESENT_UNLOCALIZABLE` 不能写全零几何。
3. 2D overlap 只支持 `overlap_2d`；不能升级为 touching/inside/contains。
4. node/part 的可见支持跨多个 4×4 cell 或多个 orientation/scale bin时选 reviewed unknown，不取中心值。
5. 看不到 stream、meniscus、液位变化且无正式传感器时，`FLOW/TRANSFER` 与 outcome 不可由 wrist tilt 或任务意图推出。
6. 遮挡只改变 observability、field knowledge 或 boundary unknown；真实 transition 位于 coverage 外时才是 censor。
7. 透明物体/液体困难不会创建新 node、新 event kind、第三条 edge或 secondary part/cell。表达不了就 overflow/invalid，而不是扩 schema。
8. focus/actor/primary reference 不可确定时 fail closed；模型、重建或“常识上应该是”都不能解除。

## 12. 常见反例与正确处理

| 错误做法 | 为什么错 | 正确处理 |
|---|---|---|
| 手框与杯子重叠就标 contact | 投影重叠不等于物理接触 | 最多 `overlap_2d`；contact 需 direct/temporal/sensor 证据 |
| `hand_status=closed` 就标 acquire | proxy 不是 physical truth，且禁止查看 | 只看原始 RGB/时序；边界不可判用 interval/unknown |
| 手腕倾斜就标 `flow/transfer=known` | 意图/姿态不能证明液体变化 | 无 stream/液位/传感器则 reviewed unknown 或 MISSING |
| 松手就填 outcome=success | release 与 outcome 是不同字段 | release outcome=0；只有 verify 才可填结果 |
| 遮挡处把 boundary 标 left/right censor | censor 只表示 transition 在 coverage 外 | 遮挡导致不可判时 boundary unknown |
| 对象出画就填 node ABSENT | OUT_OF_VIEW 是已知观测状态 | 填 `OUT_OF_VIEW`；后三个 node geometry 为 UNKNOWN |
| 不知道类别就留空，但声称已完整复核 | 混淆 reviewed unknown 与 not reviewed | 新契约显式 `REVIEWED_UNKNOWN`；legacy CSV 按旧规则空值但由 receipt 区分 |
| 两手同时抓就建两个同 onset event | macro event 应表达并发 edge bundle | 一个 event，两条 LEFT/RIGHT edge |
| 只因腕姿改变就标 regrasp | regrasp 要求接触拓扑变化 | 无 topology 变化则继续 hold/manipulate |
| 把 liquid、stream、source rim 各建 node | 静默改变四 role schema | liquid/stream 用 effect/aux；rim 用 part/relation |
| 需要第三条 edge时删掉最不显眼一条 | 会制造不完整 GT | `EDGE_OVERFLOW`，停止 packet admission |
| 框跨两个 cell时取 bbox 中心 | 把不确定性伪装成 exact quantization | 对应 cell `REVIEWED_UNKNOWN` |
| `secondary_node=NONE` 时 relation 填 UNKNOWN | 没有 secondary 时 relation 不适用 | before/after 均 N/A/0 |
| 已知左手，右手遮挡未审，就称 left-only | single-hand 是完整性结论 | roster incomplete，oracle packet inadmissible |
| 从 depth/MANO/contact proposal 核实疑难帧 | proposal/重建不是 blind evidence | 不看；仅按 raw RGB/temporal/sensor allowlist判断 |
| 为了通过 gate 在第三人裁决时改成一致 | 裁决不能覆盖系统性低一致性 | gate 失败则修订协议并用新 calibration 重来 |

## 13. 三个纯文字示例

以下均为教学用虚构情形，不来自、也不暗示来自任何真实 episode。

### 示例一：普通可见的单手倾倒成功

完整视频中，一只手持续靠近 source，进入杯身旁的微调区，首次可见接触后形成稳定持握，短暂停稳，再将 source 移到 destination 杯口附近并倾倒。视频中能直接看到液流和 destination 液位增加。随后 source 回正，手开始解除最后接触并确认分离；最后镜头稳定地显示 destination 中确有转移结果。

正确切分是 approach → precontact → acquire → hold → manipulate → release → verify_outcome。acquire 的 focus 是 source；对准杯口和倾倒阶段若误定位 destination 最可能失败，则 manipulate 的 focus 是 destination。持握 edge 的 actor 是可确认的 left 或 right，primary 是 source 的 body/handle；source 与 destination 的 relation 只有在 rim 可定位且冻结 predicate 满足时才写 aligned_to_rim。manipulate 可写 flow/transfer，因为原视频提供直接时序证据；success 只能写在最后的 verify_outcome，release 的 outcome 仍为 0。持续事件标 entry/exit；倾倒方向发生明确反转时可加一个 turn。这个例子里的帧号、手侧和物体均需由实际视频重新判断，不能照抄。

### 示例二：双手参与，但第二只手侧别不可解

完整视频显示一只手稳定 source，另一只手在遮挡后接触 destination 进行支撑。前后状态足以确认第二条 hand edge 存在，primary role 也能解析为 destination，但严重自遮挡与物遮挡使 A/B 多帧回看后仍无法可靠判断它是 anatomical left 还是 right；destination 仍可由时序追踪，但中心跨两个 4×4 cell。

正确处理是一个 bimanual macro event，不创建两个同 onset event。已知手的 edge 保留 known actor；第二条 edge 保存 confirmed present、actor reviewed unknown、actor mask=0，其余可判的 primary role/contact payload继续保留。因为 required actor 未解析，整个 oracle/action packet inadmissible，不能把它改写为 ABSENT，也不能假装是单手事件。destination observability 可写 OCCLUDED_TRACKABLE，center cell 写 reviewed unknown。pose 中被对象遮挡的 fingertips 选 object_occluded并留空坐标。遮挡导致边界不可判时选 boundary unknown，不是 censor。

### 示例三：clip 左删失、透明液体结果不可见

clip 开始时 source 已被稳定持握，因此最早可确认 event 是 hold，其 onset 在 coverage 之前。随后手倾斜 source，source 与 destination 在图像中有投影重叠，但 rim 轮廓不完整；看不到 stream、meniscus 或 destination 液位变化，也没有获批的秤/液位证据。最后能看到手与 source 分离，但没有可观察的结果检查。

正确处理是 hold 的 start 记 left_censored，不补一个虚假的 approach/acquire。投影关系最多写 overlap_2d，不能写 touching、inside 或 aligned_to_rim。倾斜动作可属于 manipulate，但 flow/transfer 是 reviewed unknown 或 MISSING，取决于是否已完整复核该字段；不能从 wrist tilt 推断。若分离过程可见，可以标 release，但 release outcome=0。没有实际可观察的结果检查就不强建 verify_outcome，更不能填 success。这个例子也不授权使用 depth completion 或 liquid reconstruction补标签。

## 14. 提交前 checklist

### 14.1 身份、隔离与输入

- [ ] 我使用的是自己的 `annotator_id`、`annotation_session_id` 与 packet role，未看另一方材料。
- [ ] packet 根目录恰好七个文件；media root 恰好 12 个 receipt 指定 MP4；state 独立且不嵌套。
- [ ] 没有额外文件、symlink、hard link、hash/identity/session 漂移或权限过宽。
- [ ] 我未访问 processed HDF5、hand status、depth、proposal、prescreen、timing hint、旧 GT、模型输出或 coordinator/adjudication。
- [ ] 我未截图、录屏、上传、转发或使用同步浏览器 profile。

### 14.2 每个 episode

- [ ] 我先原速看完完整 episode，再看局部慢放/逐帧；没有从候选片段开始。
- [ ] coverage 是 `exhaustive`、`positive_only` 或 legacy `unknown` 的诚实选择；没有把未标帧默认当 negative。
- [ ] contact segment 是全局持续区间 `[start,end)`，没有按窗口切碎；start/end 帧含义正确。
- [ ] macro events 按时间、maximal duration 与合法 r1 successor 切分；impact 有 acquire，无接触放弃没有假 release。
- [ ] episode 若超过 8 个 macro event 已标 overflow，没有截断、改类或删除后续事件。
- [ ] support、exact/interval、censor、boundary unknown 与 MISSING 没有混用；遮挡没有被写成 censor。
- [ ] focus 来自唯一冻结 role/entity binding，并在 event 内固定；focus 无法解析时没有猜值。
- [ ] 每个实际 entry/turn/exit/anchor 都按固定顺序检查四 role node；OUT_OF_VIEW/OCCLUDED/UNKNOWN/ABSENT 没有混淆。
- [ ] 每 event 不超过两条 edge；双手并发在同一 event；actor/primary 无法解析时 fail closed；overflow 未截断。
- [ ] relation 描述对象—对象关系；secondary NONE 时 relation=N/A；2D overlap 未升级成 3D/contact 事实。
- [ ] effect 有 direct/temporal/sensor evidence；release 未携带 outcome；outcome 只在真实 verify event。
- [ ] entry/turn/exit/anchor 使用实际看过的 source-clock 帧；极短 event 未复制姿态；不可见 landmark 坐标留空。
- [ ] 透明/遮挡/双手疑难项按 reviewed unknown、mask、censor、invalid 或 overflow处理，没有用常识、重建或 proposal补全。

### 14.3 当前正式 7-file packet 的提交边界

- [ ] 我只让现有 offline UI 生成/提交 `coverage.csv` 与 `segments.csv`；没有给 legacy header 加列。
- [ ] event/focus/node/edge/pose 的 D0 练习结果未写进 packet、notes、receipt 或 legacy CSV，也未被称为 GT。
- [ ] staged output 只在外部 state，packet 根目录仍恰好七个文件。
- [ ] 12 个 episode 都满足本轮获批的局部语义后才 staged/validate；我没有关闭 fail-closed 检查。
- [ ] 我理解 UI submission receipt 只是交接记录；只有协调员收回 A/B、撤销访问并运行正式 submission/agreement checker，才可能进入第三人裁决。

## 15. 标注员停止并报告的条件

出现以下任一项，不继续猜标：

- packet/media/state allowlist、hash、权限、身份或 session 检查失败；
- 发现 HDF5、depth、hand status、proposal、旧 GT、另一方输出或 timing hint 被误发；
- task ontology 无法唯一绑定 role/entity，或同 role 多实例；
- 一个 event 需要超过两条 edge、超过四 role node，或 episode 的标准事件容量溢出；
- focus、actor、primary reference 等 required field 无法唯一确定；
- 透明/遮挡使关键 effect/outcome、relation 或 hand side 无法按规则判定；
- blind 阶段发现指南歧义，需要与另一标注员讨论；
- 当前 7-file contact UI 被要求记录 event/focus/node/pose 等其 schema 不支持的字段。

正确动作是保持可判字段、对不可判字段选择相应 unknown/mask/invalid/overflow，停止受影响 packet 的正式 admission，并向协调员报告。不要扩词典、改 CSV、加 notes、换 episode、开 proposal 或私下统一答案。

## 16. 一句话记忆

**先看完整视频，再按证据切 maximal event；只标看得见或可由时序/传感器确认的角色—关系—效果，透明/遮挡/双手歧义用 unknown、mask、censor 或 fail-closed 表达；D0 可讨论不计分，D1 冻结后 A/B 绝对独立，现有 formal packet 永远保持七个文件。**
