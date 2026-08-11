# Human video ↔ robot 同任务公开数据：对 SM-FET 36+36 补采的替代性审计

> 日期：2026-08-12
>
> 结论口径：只把官方明确给出一一对应关系的数据称为 paired；“同一任务池”“并行协作”“算法生成伪配对”均单列。
>
> Zotero-first 记录：`idea_id=81b41f7d-57bd-4f74-bc46-ca16809dcb13`，`discussion_id=aa89be35-c0bd-467b-8dca-6d3a9ea9ac98`。
>
> 基线：本文工作树从研究集成分支实际 `HEAD=4569c1fb3a037f2ee790bc9baa585a27b98fc3e7` 创建，而非较旧的 `origin/main`。

## 一页结论

**没有发现能完整替代锁定 D2“36 human + 36 target robot”补采的公开数据。** 缺口不是公开样本总数，而是公开数据没有同时满足：目标任务单元、目标机器人 embodiment（Unitree G1 + 左 RH56DFTP + 右 OmniPicker）、统一相机/标定、失败与结果判定、物理接触证据、同一 protocol 下的人机配对及可审计 split provenance。

但公开数据足以把补采前的大量风险提前消掉：

1. **RH20T 是最强 D2 proxy。** 它公开“机器人 episode 及其对应的人类 episode”，超过 11 万机器人序列，且确有试管、烧杯、移液枪头、倒液、旋盖、擦拭、搅拌、插入等任务。它能预演同任务 episode 配对、事件标注和跨 embodiment 对齐，但不是同一物理执行的逐帧同步配对。
2. **MIME 是第二强的 episode-paired proxy。** 8,260 个 datapoints、20 个任务；每个 Baxter 示范都有相应的人类做同任务视频，明确覆盖 pouring、stirring、open bottle、rotate、wipe。许可缺失使其暂时只能做技术可行性审计，不能默认进入可发布训练集。
3. **H&R（Human2Robot）是最干净的逐帧同步调试集。** 2,600 组精确同步第三人称 human/robot episode，并有手姿态；但作者明确因 embodiment gap 聚焦 pick-and-place，旋拧等任务未采，因而不能验证 SM-FET 的旋盖、倒液和接触事件主张。
4. **H2O、HOI4D、EPIC-KITCHENS/VISOR 可形成独立的 D1-X 外部验证。** 它们能补充倒液、开合、手姿态、in-hand/contact 关系；不能并入锁定的 12 条 D1，也不能把视觉接触或几何接近当作物理接触真值。
5. **FineBio、PalmDex 最适合扩充 biochemical/tool-use 事件词典。** FineBio 有实验室 protocol、逐手 atomic operation 和稀疏接触框；PalmDex 有双手触觉手套与多阶段研磨/坩埚钳操作。两者均为 human-only。
6. **DROID、BridgeData V2、Open X-Embodiment、MOTIF、LabSafe-1700 均为 robot-only。** 可提供 robot-side hard negative、任务词典或 schema stress test，不能贡献正的人机 pair。
7. **EgoBridge、EgoMimic、XSkill 的 human/robot 是同任务池或算法伪配对，不是采集时真配对；HABIT 是人机同步协作但双方通常不是同一子任务。** 三类语义都不能提升为 D2 gold positive。

因此建议：**保留 36 target-robot；若目标是锁定 protocol 下的正式主结果，也保留 36 target-human。** RH20T/MIME/H&R 另建 `D2-X` 外部 proxy；H2O/HOI4D/VISOR 另建 `D1-X`。若预算必须削减，公开数据最多替代“预训练、标注员校准、事件词典和实现调试”，不替代目标 D2 的主检验样本。

## 1. 判定标准与术语

本文把“paired”拆成五级，防止数据表里的一列 `paired=true` 掩盖关键差别：

| `pair_semantics` | 含义 | 能否作为 SM-FET D2 gold positive |
|---|---|---:|
| `frame_sync` | 同一脚本/运动源下，human 与 robot 帧级或严格时间对应 | 仅可作为外部 proxy；还须任务、embodiment、结果口径匹配 |
| `corresponding_execution` | 每条 robot episode 有一条人类做同任务的对应 episode，但不是同一物理执行 | 外部 proxy，可做 episode/event matching，不可宣称帧同步 |
| `concurrent_shared_goal` | 人与机器人同时协作/共处，但承担相同或不同子任务 | 不能默认作正 pair；适合关系/负例审计 |
| `same_task_pool` | 两域分别采集同任务，只有 task label 一致 | 不能作 gold pair；可在 split 内构造弱监督候选 |
| `pseudo` / `none` | OT、DTW、检索等算法生成配对，或根本没有另一域 | 不能作 gold；只可作模型输入或 hard negative |

SM-FET 的“可直接 D1”在本文中意为 **可直接转换成外部 D1-X 字段**，并不表示可混进已锁定的 12 条 D1 指标。类似地，“D2 proxy”表示能测试某一配对机制，不表示能替代目标 D2 的 12 个 cell。

## 2. 候选总表：最高允许用途

规模均按官方项目页、论文或官方数据卡；“未声明”不等于可自由使用。

| 数据集 | human/robot 关系 | 任务与规模 | embodiment / 相机与模态 | 手、接触、事件 | 许可与下载 | 对 SM-FET 的最高允许用途 |
|---|---|---|---|---|---|---|
| **RH20T** | `corresponding_execution`：机器人目录有对应 `_human/` episode；非同一物理执行、非帧同步 | >110k robot sequences 及相应 human sequences；147 tasks / 42 skills；含试管倒出/摇匀、杯间倒水、旋/反旋瓶盖、擦桌、滴管移液、烧杯搅拌、插移液枪头/插孔 | Flexiv、UR5、Franka、Kuka，AG95/WSG50/Robotiq85/Franka gripper；1–2 in-hand + 8–10 global RGB-D，robot state、FT、audio，部分配置 uSkin tactile；human 多视角/ego RGB-D | 无 gold 手姿态/接触事件；robot FT/tactile 可作物理 proxy | 场景 0001–0005 CC BY-SA 4.0；0006–0010 CC BY-NC 4.0；官方 Drive/Baidu/API，低分辨率分包至完整约 40 TB | **D2 proxy 首选**；也可做 lab/tool ontology |
| **MIME** | `corresponding_execution`：每个 kinesthetic robot demo 有对应 human 同任务视频；非帧同步 | 8,260 datapoints / 20 tasks；Pour 290、Stir 508、Open bottle 473、Rotate 419、Wipe 292 | Baxter 双 7-DoF + parallel grippers；robot head Kinect + 2 wrist RGB-D；human 头戴 Kinect（论文措辞对视角有歧义，使用前复核原文件） | joint/gripper + RGB-D；无手姿态、力/接触、事件边界；有人工作业通过/拒绝质检 | 官方仍列 Dropbox/Drive；**数据许可未在项目页/论文明确声明** | **D2 proxy 次选**；许可澄清前不进入发布训练集 |
| **H&R / Human2Robot** | `frame_sync`：精确同步 human/robot 第三人称 paired episodes | 2,600 pairs，每条 300–600 frames；4 basic + 6 long-horizon；主要 pick/place，论文明确说明 screwing 难以采而未覆盖 | xArm 7-DoF；两台 RealSense D435，同视点，240×424@30 Hz；Quest 3；EEF/gripper/qpos/qvel | v1 有 human hand pose 映射至 robot frame；无显式 contact/event gold | Hugging Face 约 233 GB；**数据许可字段未声明** | **D2 帧对齐单元测试**，不能覆盖旋拧/倒液/擦拭 |
| **HABIT** | `concurrent_shared_goal`：人与机器人同步协作；不是“双方各自完成同一任务”的替代示范 | 10,563 episodes / 164.19 h / 60 tasks；collaborator 3,198、coworker 3,969、supervisor 3,396 | 双 Franka Research 3 + Robotiq 2F-85；robot central、双 wrist、human head ego、external 共 5 个同步 RGB；joint/cartesian/gripper | 双方脚踏标子任务；robot 157 / human 182 subtasks、308 co-occurring pairs；无手姿态/物理接触真值 | CC BY 4.0；HF 约 253 GB，另有约 1 GB sample | **hard negative / cross-agent ontology**；共享目标不等于正 pair |
| **EgoMimic** | `same_task_pool`；human/robot 文件按任务组织但非一一 paired | bowl placement、groceries、small-cloth-fold；object-in-bowl 人类约 1 h/1,400、机器人约 1 h/135 | human Project Aria ego RGB/SLAM/MPS 3D hands；robot 双 ViperX300S、wrist D405 + Aria ego；proprio/action | 有 3D hand/SLAM；无物理 contact/event gold | HF 约 243 GB；代码 MIT，**数据许可未单独声明** | **D2 unpaired proxy**；不得按相同文件名/任务名冒充 pair |
| **EgoBridge** | `pseudo`：在独立 human/robot 池上用 OT/DTW 生成伪配对 | drawer：robot 144、human 1 h；scoop coffee：robot 50、human 2 h；laundry：robot 2 h/300、human 2 h/~700 | 继承 ego human hand tracking/SLAM 与双臂 robot ego/wrist 流 | 算法伪对应，无采集 gold contact/event pair | 项目页/论文未提供独立数据下载或数据许可 | **方法基线，不是可新增的公开数据** |
| **XSkill** | `same_task_pool`：论文明确为 unsegmented、unaligned、无 correspondence | real kitchen 两域各 7 tasks ×25 = 175；开烤箱、抓布、关抽屉、开灯等 | UR5 + WSG50 soft fingers；两台 D415 wrist/side RGB；robot 6D EEF+gripper | 无事件/接触/手姿态 gold | 官方提供 `real_kitchen_data.zip`；代码 MIT，**数据许可未单列** | **D2 unpaired/multistage proxy** |
| **H2O** | human-only | 571,645 RGB-D frames；4 人、36 actions、8 objects、933 action instances；含 pour milk、open/close、put in/take out | 4 external + 1 head ego RGB-D，1280×720@30 Hz | 双手 21 joints/MANO、object 6D pose、action；无显式物理 contact | CC BY-NC 4.0；ETH archive/官方脚本 | **可直接 D1-X**：pour + pose；几何接近不可当物理接触 |
| **HOI4D** | human-only | 4,000 sequences / 2.4M RGB-D frames；9 人、800 objects、54 tasks；take/open/pour/close/put 等 | 头戴 D455 + Kinect v2 同步；约 20 s@15 fps；当前标注主要单手场景 | framewise action、panoptic/motion seg、3D hand pose、object pose/mesh；无接触真值 | CC BY-NC 4.0；官方分包下载 | **可直接 D1-X**：事件/姿态/物体几何 |
| **EPIC-KITCHENS-100 + VISOR** | human-only；VISOR 是 EPIC 派生子集 | EPIC：100 h、700 videos、45 kitchens、90k actions；VISOR：36 h，271k manual masks、数百万 dense masks、约 67k in-hand/contact relations | 第一视角 full-HD；分割/关系标注 | verb/noun action；VISOR 有视觉 in-hand/contact relation，无 3D pose/力 | 均 CC BY-NC 4.0；EPIC 约 1.1 TB，VISOR 可分开下 sparse/dense ann | **可直接 D1-X**：视觉接触、倒液/开合/擦拭 ontology |
| **Ego4D FHO** | human-only | Ego4D 总计 3,670 h；FHO v1 1,672 clips / 116.274 h / 91,002 interactions，v2 规模更大；含约 25 h 生物实验录像但叙述较粗 | 多种 head-mounted ego cameras，规范化 30 fps | PRE/CONTACT/PNR/POST critical frames 与 hand/tool/object boxes；是视觉事件，不是力真值 | 自定义 license agreement，需账号/凭据；完整数据 >7 TB | **hard negative / event ontology**；若抽任务子集可作 D1-X 辅助 |
| **FineBio** | human-only，实验室高度相关 | 226 valid trials / 32 participants / 14.5 h（87 camera-hours）；7 protocols、3,541 steps、50,659 atomic ops；DNA extraction、PCR、spin column 等 | 1 头戴 +5 固定 GoPro，同步/标定，30 fps | 每只手 `verb–manipulated object–affected object`；1,935 frames / 71,548 boxes / 33 equipment；稀疏 contact/作用对象，不是连续力 | 自定义签署许可：非商业研究/开发、不可转让，需审批；代码 MIT 不覆盖数据 | **hard negative / biochemical ontology / D1 schema 扩展** |
| **PalmDex** | human-only | 当前 chemistry release：300+ raw episodes、1,001 annotated segments；研钵研杵 415、坩埚钳 586 | 同步 ego+side RGB 640×480@30；双 256-channel tactile gloves；72D hand/head/control pose | tactile + grasp/action segment；多阶段工具操作 | CC BY-SA 4.0；HF 展示总仓约 74.2 GB，标注子集逻辑体量更小 | **hard negative / contact-tool ontology**；不是倒液/旋盖目标 pair |
| **HOT3D** | human-only | 833 min、1.5M synchronized multiview frames、3,832 个 5 s clips；19 subjects、33 rigid objects | Project Aria 或 Quest 3；RGB/mono、SLAM、gaze | UmeTrack/MANO 手与 object 3D pose；无 action/event/contact gold | 序列/多数标注 CC BY-SA；hand ann CC BY-NC-SA；模型另有附加限制；需注册下载 | **D2-R 手姿态/物体几何 proxy**，非 pair |
| **DROID** | robot-only | 76k trajectories / 350 h；564 scenes、86 tasks、50 collectors、13 institutions | Franka Panda；2 external ZED2 stereo + wrist ZED Mini；Quest 2；RGB、cartesian/joint/gripper action/state | 约 95% successful trajectories 有 3 条 language ann；无手/物理 contact gold | 数据 CC BY 4.0；RLDS 约 1.7 TB、raw 约 8.7 TB、debug 100 eps/约 2 GB | **robot hard negative / ontology**，不贡献 human pair |
| **BridgeData V2** | robot-only | 60,096 trajectories：50,365 teleop +9,731 scripted；24 env、13 skills；pick/place、push、sweep、drawer/door、fold 等 | WidowX250 6-DoF；固定/随机/腕部 RGB(D)，640×480，平均 5 Hz | action/state；无人类、无 hand/contact gold | CC BY 4.0；raw 与 TFDS/RLDS 可下载 | **robot hard negative / D2 schema proxy**；且已被 OXE 收录 |
| **Open X-Embodiment** | robot-only 聚合 | 1M+ trajectories、60 datasets、34 labs、22 embodiments、527 skills | RLDS 标准化层；原始相机/状态/动作仍高度异构 | 无统一 contact/event/结果；无 human | wrapper 软件 Apache 2.0、其他材料 CC BY 4.0；**每个成员数据集仍须审其来源条款** | **robot ontology / schema stress test**，不得与 Bridge 重复计数 |
| **MOTIF** | robot-only cross-embodiment | 方法/小型示例含 ARX5、Piper 等；公开 HF 示例约 220 rows / 390 MB、2 tasks | robot video/action | 学得 motion/action motif；无 human/contact gold | 代码 MIT；示例数据许可未声明 | **robot motif ontology**，不是 pair 数据集 |
| **LabSafe-1700** | robot-only，实验室外观 | 卡片宣称 1,700 episodes / 17 configs；HF viewer 当前仅显示 340 rows，规模有待逐文件审计 | SO-100 6-DoF；3 RGB 480×640@30；joint-position action；试管、烧杯、漏斗等 | task label；无液体操作、手姿态、contact gold | CC BY 4.0；约 13.5 GB | **lab robot hard negative**；不能据卡片标题认定生化技能 |
| **HARP-VLA / HR-Align** | 方法复用已有数据；没有新增公开 paired corpus | HARP 使用 RH20T、H&R 及自采 Human–DexHand；HR-Align 使用 RH20T | 见源数据 | 自采数据未公开 | HARP repo 代码 MIT；HR-Align 项目页无新增数据 | **方法基线；不可把论文训练量当作可下载新数据** |
| **IVOS** | 声称有人/机器人类似任务，但 human 下载仍为 `Coming Soon` | cutting/pouring/stirring/drinking；robot 8,984 frames / 56 configs，human 仅 11 个类似任务说明 | 视觉操作片段 | 无足够公开 metadata | robot 链接可见；human 侧不可得，许可未清 | **当前完全不适用** |

## 3. 关键候选的实证边界

### 3.1 RH20T：最接近“同任务人机 episode”，但仍不是目标 D2

[RH20T 项目页](https://rh20t.github.io/)与[论文附录](https://rh20t.github.io/static/RH20T_paper_compressed.pdf)明确给出机器人序列及对应 `_human/` 目录。值得优先抽 metadata 的任务包括：

- 30：从试管倒出；36：摇试管；39：杯到杯倒水；55：滴管移液；
- 42/43：旋紧/旋开罐盖；41、101、102：海绵/布/刷擦洗；
- 100：玻璃棒搅拌烧杯；109/111/112：插入类；
- 120/121：旋转安装/拆卸灯泡；200–202、204、211 等：移液枪头插入/转移。

它对本项目最有价值的不是“替掉若干条采集”，而是提前完成四件事：验证 `pair_semantics=corresponding_execution` 的 loader；在真实多相机、FT/触觉异构缺失下检验事件 schema；训练 annotator 区分手闭合、视觉遮挡与物理接触；构建跨任务 hard negatives。

限制必须写入结果表：公开页面没有逐任务 episode 数，不能从总量推算上述任务各有多少；其机器人、夹爪、视角都不是目标 G1 双末端；human/robot 并非同一物理执行；只有部分机器人配置有 uSkin，human 侧没有相同的物理接触传感。因此 RH20T 只能是 `D2-X/RH20T`，不能计入目标 12 cell 的 36+36。

### 3.2 MIME：任务吻合度高，法律元数据是硬阻塞

[MIME 官方页](https://sites.google.com/view/mimedataset/dataset)和[论文](https://arxiv.org/pdf/1810.07121)给出了每个 robot demo 对应 human 同任务视频的采集设计。公开任务表对本项目尤其友好：Pour 290、Stir 508、Open bottle 473、Rotate 419、Wipe 292。

不过，它没有事件边界、手姿态、力/触觉或结果传感；Baxter 双臂 parallel gripper 与目标形态相距较大。更关键的是，官方数据页没有清晰数据许可。**“论文可在 arXiv 读取”或“代码有许可证”不能外推成数据可训练/再发布。** 因此 MIME 可用于在本地做 metadata/小样本可行性审核；正式训练前须获得数据权利方的书面条款或仓库明确 license。

### 3.3 H&R：配对最强，任务覆盖最弱

[Human2Robot 论文](https://arxiv.org/html/2502.16587)把 2,600 组数据描述为精确同步、相同第三人称视点的人机 episode，[官方 HF 仓库](https://huggingface.co/datasets/dannyXSC/HumanAndRobot)公开 v0/v1 数据。v1 还提供映射到 robot frame 的 human hand pose。

这使其很适合检验：逐帧 pairing loader 是否正确、相机差异是否泄漏 domain label、完整轨迹相比 entry/turn/exit sparse pose 是否真正有增益。但论文也明确把 `screwing` 举为难以收集的动作，最终聚焦 pick/place。它不能支撑“SM-FET 能识别旋拧、倒液、擦拭中的接触事件”这一主张。数据卡没有明确 license，仍需先澄清再用于正式 checkpoint。

### 3.4 HABIT：是真同步，但不是“同任务替代示范”

[HABIT 项目页](https://habit-dataset.github.io/)和[HF 数据卡](https://huggingface.co/datasets/configinc/HABIT)给出五视角同步人机协作数据及双方脚踏标注的子任务边界。其 `collaborator/coworker/supervisor` 设置很适合测试模型是否会把“同时出现、共享场景、共享目标”错误当成同一 action pair。

因此它应该进入 hard-negative 设计：

- 同时但不同子任务；
- 相邻先后子任务；
- supervisor cue 与 robot execution；
- shared object 但不同作用角色。

只有在逐条核验双方确实承担等价 subgoal 后，某个片段才可升级为人工确认的 external positive；数据集整体不能标 `paired_positive=true`。

### 3.5 实验室与接触补充：FineBio、PalmDex、LabSafe

[FineBio](https://github.com/aistairc/FineBio)是最相关的 human biochemical ontology 来源：其 atomic operation 对每只手编码动词、被操纵物体和受影响物体，覆盖 pipette/microtube 的 press、insert、release、detach 等事件；六机位同一 trial 也适合验证相机/视角泄漏。它使用模拟液体而非真实试剂，且是 human-only、非商业审批制许可，因此不构成人机配对。

[PalmDex](https://huggingface.co/datasets/Rimbot/PalmDex)用触觉手套提供比纯视觉更接近“接触证据”的 human 工具操作数据，适合校准 contact onset/offset 和 regrasp。但其主要 chemistry 任务是研磨与坩埚钳，不是当前三类目标任务。

[LabSafe-1700](https://huggingface.co/datasets/Merlyn-L/LabSafe-1700)虽有试管、烧杯、漏斗等器材，动作仍是抓取/放置和障碍规避，不是液体转移或精细插接。数据卡宣称 1,700 episodes，而 viewer 当前只呈现 340 rows；在下载任何大包前，必须先通过 repository tree/metadata 解释该差异。

## 4. 推荐分层与是否能减少 36+36

### A. 可直接转换为 D1-X

- **H2O**：优先抽 `pour milk`、open/close、put-in/take-out；用于 pose + event boundary 外部验证。
- **HOI4D**：优先抽 pour/open/close/put 和有可用手/物体姿态的序列；用于几何和事件一致性。
- **EPIC-KITCHENS/VISOR**：用于视觉 in-hand/contact、工具/容器词典和严重遮挡；必须按源视频与厨房分组。
- **Ego4D FHO**：PRE/CONTACT/PNR/POST 可做事件阶段映射，但只作为视觉事件外部集。

这些数据**不减少 D1 锁定的 12 条**。合理报告方式是：`D1 locked` 给主指标，`D1-X` 给跨数据集 robustness 与标注一致性；不合并 n，不共同拟合阈值。

### B. D2 proxy

推荐顺序：

1. **RH20T**：同任务对应 episode，任务与 lab/tool coverage 最好；
2. **MIME**：同任务对应 episode，倒/搅/开瓶/旋/擦直接命中，但许可待清；
3. **H&R**：帧同步最强，适合实现和泄漏单测，任务主要为抓放；
4. **XSkill / EgoMimic**：同任务但 unpaired，只能测试 weak alignment 与 NULL mass；
5. **HABIT**：并行协作关系，不作为等价动作 positive。

它们可以替代的，是正式采集前的 loader、pair-head、事件 schema、annotation guide 和 ablation 调试工作；不能替代目标 D2 的主统计。若不得不削预算，优先保住 **36 robot**，因为公开源没有目标 embodiment；human 是否缩减必须被视为修改 protocol，并重新做 power/CI 设计，不能用 RH20T/MIME 的人类片段直接填目标 cell。

### C. hard negative / ontology / schema stress test

- human/tool/lab：FineBio、PalmDex、Ego4D、HOT3D；
- concurrent relation：HABIT；
- robot-side：DROID、BridgeData V2、Open X-Embodiment、MOTIF、LabSafe-1700；
- D2-R sparse-pose proxy：H2O、HOI4D、HOT3D、H&R v1。

这些数据的正当用途包括：相似外观不同动作、同工具不同 effect、共享物体不同 actor、相邻阶段、错误 part/cell、不同 embodiment 的相同语言标签，以及物体接近但无可靠物理 contact 的负/未知例。

### D. 当前完全不适用为“新增公开 paired 数据”

- **EgoBridge**：是从独立池计算 pseudo-pair 的方法，项目未发布独立新 corpus；
- **HARP-VLA、HR-Align**：复用已有公开数据，论文里的自采数据未成为可下载公开集；
- **IVOS human side**：官方仍标 `Coming Soon`；
- 任何把 **DROID/OXE/Bridge/MOTIF** 写成 human↔robot paired 的方案。

## 5. 最小转换字段

外部数据不能只转成 `video, task_label`。建议最小 sidecar 如下；源字段不存在时必须为 `unknown/null`，不得补成否定值。

### 5.1 Provenance 与 pair receipt

```yaml
source_dataset: RH20T
source_version: <release/hash>
source_uri: <official artifact URI>
license_partition: BY-SA | BY-NC | custom | unresolved
raw_episode_id: <immutable source id>
raw_hash: <sha256 or official checksum>
domain: human | robot | concurrent
pair_group_id: <nullable stable id>
counterpart_episode_id: <nullable stable id>
pair_semantics: frame_sync | corresponding_execution | concurrent_shared_goal | same_task_pool | pseudo | none
pair_evidence: <official metadata path or reviewer receipt>
task_source_id: <source task id>
task_family_sm_fet: pick_place | stabilize_twist_cap | pour_transfer | external_other
task_ontology_version: <version>
subject_or_operator_id: <group-safe pseudonymous id>
session_id: <nullable>
scene_id: <nullable>
object_instance_ids: [<ids>]
embodiment_id: <human or robot hardware tuple>
split_group_id: <computed connected component id>
```

### 5.2 时间、相机与可观察性

```yaml
stream_id: <episode/camera/modality>
view_role: ego | wrist_left | wrist_right | external | global | depth | tactile | force
timestamp_source: hardware | software | reconstructed | unknown
fps_nominal: <number>
frame_timestamp_ns: <vector or URI>
resolution: [height, width]
intrinsics_uri: <nullable>
extrinsics_uri: <nullable>
time_sync_error_ms: <nullable>
source_interval: [start_frame, end_frame]
visibility_mask: <per role/part/camera>
missingness_reason: not_recorded | occluded | corrupt | annotation_missing | out_of_scope
censor_flags: [left_censored, right_censored, interrupted]
```

### 5.3 SM-FET 事件与配对字段

- `events_v1`：`kind`、`focus`、起止/关键边界、边界来源与置信度；
- `event_node_snapshots_v1`：四角色的可观察性、2D cell、尺度、方向；
- `event_edges_v1`：actor、role、part、cell、relation before/after、`contact_mode`、证据模态；
- episode 级：coverage、censor、failure、outcome、outcome sensor/source；
- 可选姿态：entry/turn/exit 的 7 个功能性 2D landmarks、reference bbox、visibility；若源是 MANO/3D pose，保留源坐标系和投影 receipt；
- `event_pair_v1`：human/robot episode ID、双方 event interval、22D slot reference、role/part/effect/subgoal、pair confidence、sparse coupling、NULL mass、annotator、adjudication、schema/source/split/hash。

三条强约束：

1. `missing contact annotation != no contact`；
2. `gripper closed / hand closed != physical contact`；
3. `pseudo-pair / same task != gold pair`。

物理传感（FT、tactile）、几何推断、视觉 relation、人工事件判断必须放在不同 provenance 字段，不能压成一个无来源的 boolean。

## 6. 数据泄漏与虚高风险

### 6.1 先分组再做窗口、pair 和特征

split 顺序必须是：

1. 按 source episode/operator/session/object batch/camera calibration 建 group；
2. 把明确 counterpart、同一原始多视角、派生 annotation、raw/annotated 副本连成图；
3. 对连通分量分 train/val/test；
4. 只在各 split 内切窗口、做增强、抽特征、生成 pseudo-pair/hard negative。

不能先从视频切出几千窗口再随机分；那会把几乎相同的背景、轨迹与相邻帧复制进测试集。

### 6.2 各数据源的特定 group key

- **RH20T**：human/robot sibling 必须同 split；同时按 user、scene、task、configuration 和 object instance 做更严格分组；BY-SA 与 BY-NC checkpoint/产物另行隔离。
- **MIME**：一对 human/robot 同 split；正式使用前核查官方 split 是否跨示范者/对象泄漏。
- **H&R**：human 与 robot 共享 latent motion/录制设置，pair 同 split；同 session/object/camera setup 也应分组，避免仅识别背景或同步模板。
- **HABIT**：五相机、双方同步流和所有 subtask window 均随原 episode；不要让同一协作过程的不同角色跨 split。
- **H2O / FineBio**：同一 take/trial 的全部 5/6 机位同 split；优先 participant-held-out。
- **HOI4D**：两种头戴 sensor 的同一序列、同人、同房间/物体实例成组。
- **EPIC/VISOR**：VISOR 是 EPIC 派生，按 source video/participant/kitchen 联合去重；不能把 EPIC train 与 VISOR test 当独立数据。
- **Ego4D**：所有 FHO clip 随 source video/wearer/location；相邻 overlapping clip 不能拆开。
- **HOT3D**：同 sequence 产生的 5 秒/重叠 clips、同 participant/object 组合成组。
- **PalmDex**：raw episode 与其 annotated segments 是同一数据，必须同 split。
- **OXE / Bridge / DROID**：OXE 包含 Bridge 等成员，按原 dataset + trajectory hash 去重；不能把原版与 RLDS 转换版双计数。
- **EgoBridge / EgoMimic、HARP / RH20T/H&R/Bridge/HOI4D**：方法数据来自这些源，不能把论文组合包当独立 corpus。

### 6.3 任务、许可与筛选也会泄漏

- task ID、文件路径、固定背景、机器人外观和相机 watermarks 可能成为 domain/task shortcut；报告 task-held-out、session-held-out、embodiment-held-out probes。
- hard negative 不能只换 task label；应匹配场景、工具、阶段长度和视角，否则模型靠低级差异取巧。
- pseudo-pair 必须只用 train split 拟合 OT/DTW/检索器；用全数据建 embedding 或字典会把 test 信息反灌训练。
- 只保留 successful episodes 会引入选择偏差；failure、unknown outcome、截断应保留独立标签，不能当负 contact。
- CC BY-NC、自定义非商业或许可未声明的数据，不能进入意图商业发布的 checkpoint；license partition 本身也要参与 lineage。
- 尊重每个官方 hidden/test split；公开 test annotation 不参与阈值、词典和事件边界调参。

## 7. 低成本执行建议（不下载大数据）

1. **先做 metadata-only manifest**：RH20T 按任务 ID 30/36/39/41–43/55/100–102/109/111/112/120/121/200–204/211…清点 human/robot sibling、相机与 license scene；MIME 清点五个命中任务；H&R 清点任务/场景/session。
2. **只取官方 sample/debug 小包**：先验证 hash、解码、同步、坐标与 sidecar；不要直接拉 RH20T 5–40 TB、H&R 233 GB、HABIT 253 GB 或 DROID/OXE 全量。
3. **建立三套不混合的 scoreboard**：`D1 locked`、`D1-X human public`、`D2-X public pair/proxy`。每张表展示 pair semantics、license 与传感证据。
4. **预注册 pair 升级规则**：`same_task_pool` 只有经双人/仲裁确认 role–part–effect–subgoal 等价后，才能成为 `reviewed external positive`；仍不得改写为采集 gold。
5. **先用 RH20T/MIME/H&R 做 smoke test，再采 36+36。** 若外部 proxy 已暴露模型依赖背景、完整轨迹无增益、NULL mass 崩溃或事件 schema 不稳，应先修 protocol，再启动不可逆补采。

## 8. 最终决策

推荐决策是：

- **不取消 36+36。** 公开数据不能替代目标 embodiment、目标相机/结果传感与 locked cell 的共同 protocol。
- **把 RH20T 设为 `D2-X primary`，MIME 设为 `D2-X secondary/legal-hold`，H&R 设为 `D2-X sync-debug`。** 三者分别回答 episode 配对、任务吻合和帧同步三个不同问题。
- **把 H2O + HOI4D + EPIC/VISOR 设为 `D1-X`。** 它们验证跨数据集 event/pose/contact observation，而不改变 D1 主 n。
- **把 FineBio + PalmDex 设为实验室/接触 ontology 扩展。** LabSafe 只作为 robot laboratory appearance hard negative。
- **EgoBridge、XSkill、EgoMimic、HABIT 的 pair semantics 必须原样展示。** 任何将 pseudo、same-task pool 或 concurrent collaboration 报成“公开 paired data”的结论都应判为无效。

这套安排不会直接省掉目标采集的 72 条，却能显著降低补采失败、schema 返工、伪配对泄漏和接触标签自欺的概率；这是公开数据在当前 SM-FET 路线中的最高可信价值。

## 参考官方入口

- RH20T：[project](https://rh20t.github.io/) / [paper PDF](https://rh20t.github.io/static/RH20T_paper_compressed.pdf)
- MIME：[dataset page](https://sites.google.com/view/mimedataset/dataset) / [paper](https://arxiv.org/pdf/1810.07121)
- H&R：[paper](https://arxiv.org/html/2502.16587) / [dataset](https://huggingface.co/datasets/dannyXSC/HumanAndRobot)
- HABIT：[project](https://habit-dataset.github.io/) / [dataset](https://huggingface.co/datasets/configinc/HABIT)
- EgoMimic：[project](https://egomimic.github.io/) / [repository](https://github.com/SimarKareer/EgoMimic)
- EgoBridge：[project](https://ego-bridge.github.io/) / [paper](https://arxiv.org/abs/2509.19626)
- XSkill：[project](https://xskill.cs.columbia.edu/) / [paper](https://proceedings.mlr.press/v229/xu23a/xu23a.pdf)
- H2O：[ETH archive](https://www.research-collection.ethz.ch/items/e629726a-b4df-4e4e-8474-9eb44eef98e2) / [repository](https://github.com/taeinkwon/h2odataset)
- HOI4D：[project](https://hoi4d.github.io/)
- EPIC-KITCHENS / VISOR：[EPIC](https://epic-kitchens.github.io/) / [VISOR](https://epic-kitchens.github.io/VISOR/)
- Ego4D：[official docs](https://ego4d-data.org/docs/)
- FineBio：[repository](https://github.com/aistairc/FineBio) / [paper](https://arxiv.org/html/2402.00293)
- PalmDex：[dataset](https://huggingface.co/datasets/Rimbot/PalmDex)
- HOT3D：[project](https://facebookresearch.github.io/hot3d/)
- DROID：[project](https://droid-dataset.github.io/) / [dataset docs](https://droid-dataset.github.io/droid/the-droid-dataset.html)
- BridgeData V2：[project](https://rail-berkeley.github.io/bridgedata/)
- Open X-Embodiment：[repository](https://github.com/google-deepmind/open_x_embodiment) / [project](https://robotics-transformer-x.github.io/)
- MOTIF：[paper](https://arxiv.org/abs/2602.13764) / [repository](https://github.com/buduz/MOTIF)
- LabSafe-1700：[dataset](https://huggingface.co/datasets/Merlyn-L/LabSafe-1700)
- HARP-VLA：[paper](https://arxiv.org/pdf/2605.31234) / [repository](https://github.com/anonymity35/HARP-VLA)
- HR-Align：[project](https://jiaming-zhou.github.io/projects/HumanRobotAlign/)
- IVOS：[dataset page](https://msiam.github.io/ivos/dataset/)
