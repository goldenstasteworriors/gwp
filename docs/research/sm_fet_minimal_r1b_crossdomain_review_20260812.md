# SM-FET `minimal-r1b` 跨域监督、robot grounding 与可标性独立审查

- 日期：2026-08-12
- 审查性质：独立设计审查；不实现、不训练、不调用 GPU、不修改 canonical
- 独立工作树：`gwp-agent-sm-fet-r1b-crossdomain-review`
- 基线：canonical `e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 被审 schema：`sm_fet_hard_grammar_v2_minimal_r1b`，source commit `e31208172d194511e7fc95acd6075635c249c335`
- 审查问题：human 视频如何只提供跨具身语义而不提供 robot 几何；robot-view anchor 的当前因果来源；pairing key；entry/turn/exit auxiliary；透明/遮挡 unknown；human pixel、allocator 与 profile 旁路；唯一训练/部署数据流

## 0. 最终裁决

**`minimal-r1b` 的 26-byte wire、5-token布局与 `81.47790257814123 bit` 条件 packet alphabet 可以保留，但当前跨域/可标性契约不能直接放行。需要发布一个不改 wire 的规范性 `minimal-r1b-crossdomain-v1` addendum，然后才可进入 CPU checker、robot-side annotation 与无训练 oracle。**

当前判定分成三层：

1. **共享语义：Conditional Go。** human 与 robot 可以共享 `compact phase + intended current effect + role/part + intended target relation`。human anatomical side、human pixel、2D/3D pose、track、bbox 和 motion direction不共享。
2. **robot locator：当前 No-Go。** `64×64×32 + 512 axis + 8 extent` 必须由目标机器人当前观测/native state独立生成。现有 human D0 三段既不是 GT，也没有 metric depth、axis或extent证据；现有 annotation protocol也没有 r1b robot-locator source table。
3. **训练/部署 route：当前 No-Go，直到旁路契约补齐。** 原 annotation protocol 的 CAPOT cost允许 global image center/scale/orientation；Idea G 还保留 human-prompt/event-memory部署候选。二者都不能直接用于 strict r1b。strict r1b 的唯一部署模式必须是 **robot-only current grounding；CAPOT、human frame、pair ID与human token全部在部署时删除**。

建议保留：

```text
wire schema ID       = sm_fet_hard_grammar_v2_minimal_r1b
field wire           = 26 bytes
action token slots   = 5
reachable states     = 3,367,384,122,335,042,437,906,433
exact packet bits    = 81.47790257814123
```

建议新增但不进 wire：

```text
CROSSDOMAIN_BINDING_VERSION = sm_fet_minimal_r1b_crossdomain_v1
PAIR_KEY_VERSION            = sm_fet_r1b_semantic_pair_v1
ROBOT_LOCATOR_SOURCE_VERSION= sm_fet_robot_locator_source_v1
PROFILE_ADMISSION_VERSION   = sm_fet_r1b_profile_admission_v1
```

`81.4779 bit` 只能表述为 **固定 text goal、hardware、camera calibration 与 task profile 条件下的 packet alphabet**，不能写成整个控制系统总带宽。text、25D state与运行时可变 profile本来就是 packet 之外的条件；若 profile ID能逐样本变化并改变 decode/controller，它必须被计作额外控制输入或固定为单一评测 strata。

## 1. 审查输入与版本边界

| 输入 | source commit | 本审查使用方式 |
|---|---|---|
| `sm_fet_minimal_r1b_schema_capacity_20260812.md` | `e312081` | wire、locator、assignment、capacity与fail-closed主规范 |
| `sm_fet_dynamic_focus_token_spec_20260812.md` | `2452e60` | human/robot共享语义、target-domain重新ground、CAPOT geometry收窄 |
| `sm_fet_annotation_and_pairing_protocol_20260812.md` | `288ef5d` | standard-r1 annotation、pair table、UNKNOWN/MISSING/censor与旧CAPOT cost |
| `sm_fet_d0_three_episode_observability_audit_20260812.md` | `24131df` | coordinator-only三段proposal，不作GT/gate |
| `sm_fet_d0_observability_independent_review_20260812.md` | `189dfa8` | 独立确认`1 V / 13 P / 4 U`只属proposal，D0未过 |
| `idea_g_partial_human_robot_alignment_review_20260812.md` | `9f60f04` | CAPOT、many-to-many/NULL与训练/提示级融合候选 |
| `wam_biochemical_manipulation_idea_portfolio_20260812.md` | `3c545aa` | CAPOT stop-gradient、唯一主线与finite-channel约束 |
| `sm_fet_minimal_control_sufficiency_review_20260812.md` | `8c5bb24` | locator、双约束和robot assignment的控制必要性反例 |
| `sm_fet_d0_schema_reduction_review_20260812.md` | `fdac965` | human actor删除、intended relation/effect与minimal core来源 |

版本优先关系必须写清：annotation protocol 是为 **standard-r1** 写的；focus spec 已明确修正其跨域 geometry cost；r1b 又替换了 standard 的 node/site geometry与human actor。因此不能把 protocol 的82-code compiler、global-image geometry cost或actor字段原样复用到 r1b。

本审查没有把 idea-discussion 自动 scout 当契约证据；登记结果只用于idea memory。最终裁决来自上述已提交本地规范之间的可追踪冲突。

## 2. D0 对 r1b 可标性的真实含义

D0 proposal与独立复审共同支持的最窄结论是：在 episode 43/50/58 的 raw human RGB 中，只有 `center_cell` 达到局部 V；13类字段只是proposal级 P；`scale_band / orientation_2d / primary_surface_cell / outcome` 为 U。三段还有共同限制：

- transparent rim/contact interface在关键 turn 被手、腕和source遮挡；
- 2D overlap不能证明3D touching/inside；
- before/after可见颜色变化最多支持粗 `TRANSFER` proposal，不支持stream、flow rate、volume或success；
- 单effect会遇到 `ROTATE` 与 `TRANSFER` 并发；
- 另一只手画外不是absence；
- proposal memo不是annotation、A/B、GT、D0或D1；D0 checker和round-trip仍不存在。

这些结果对 r1b 有两个直接后果：

1. human视频可能支持部分 **语义** 标注，但完全没有证明 `64×64×32` robot camera locator、512方向或8档extent可标；这些量本来就不该由human标。
2. r1b把每条constraint的depth设为必填，所以透明场景比standard 2D route更容易abstain。没有目标机器人direct/sensor/native depth时，r1b transparent profile应是 No-Go，而不是用human RGB、单目completion或task常识补齐。

因此不能写：

```text
human D0局部可见
=> r1b locator可标
=> r1b packet可编译
```

正确拆分是：

```text
human semantic D1
+ robot locator R0
+ mechanical semantic projection X0
+ robot assignment/profile A0
=> 才可能形成 r1b oracle packet
```

## 3. 每类信息的唯一所有者

| 信息 | human 可提供 | robot 必须独立提供 | 是否进入 r1b action wire |
|---|---|---|---|
| compact phase | A/B裁决后的eight-kind→compact映射 | 当前robot planner再次因果判断 | 是 |
| intended current effect | task ontology与当前事件语义；不得由事后成功倒推 | 当前goal/sensor evidence确认 | 是 |
| primary/focus role、part | fixed ontology下可共享 | 当前robot实例解析与part grounding | role/part进入；instance ID不进入 |
| secondary role、intended relation | 可共享功能语义 | 当前robot对象关系与goal确认 | 是 |
| human left/right、hand pose | 只作human annotation/auxiliary | 不得映射为robot side | 否 |
| `anchor_uv` | **禁止提供** | 当前target camera中的role/part locator | 是，robot-only code |
| depth | **禁止提供** | calibrated depth/stereo/marker/native state | 是，robot-only code |
| axis/extent | **禁止从human motion复制** | target object/part frame、surface/port/rim/path的当前robot证据 | 是，robot-only code |
| target assignment | human side不能提供 | allocator基于locator、25D proprio和hardware constraints求唯一enum | 是，robot-only code |
| entry/turn/exit pose/track | human事件切分、effect/visibility auxiliary | robot自身训练期locator/visibility auxiliary | 否 |
| outcome、solver mass、pair ID | pairing、评测、supervisor | robot supervisor/benchmark | 否 |

关键边界是：**human只回答“当前功能阶段要对哪个role/part形成什么relation/effect”，robot回答“这个role/part现在在本相机哪里、由哪个末端执行”。**

## 4. human standard annotation 到 r1b shared semantics 的唯一投影

当前 annotation protocol直接服务standard-r1，不能让标注者顺手填写r1b geometry。应先生成一个action-invisible、可round-trip审计的 `shared_semantics_v1` manifest。其唯一投影建议冻结为：

| human event kind | r1b compact phase | r1b effect规则 |
|---|---|---|
| `approach`、`precontact` | `APPROACH_ALIGN` | `NONE` |
| `acquire` | `ACQUIRE` | `NONE` |
| `hold` | `CONTROLLED_CONTACT` | `NONE` |
| `manipulate` | `CONTROLLED_CONTACT` | 预冻结的 intended current effect |
| `regrasp` | `REGRASP` | `NONE` |
| `release` | `RELEASE` | `NONE` |
| `verify_outcome` | `VERIFY` | `NONE`；observed outcome留supervisor |

这张表暴露了现有文档中的一个未决冲突：D0 proposal允许把“持杯移动到target上方”描述为 `HOLD + TRANSLATE/NONE`，而 r1b 已定义 `CONTROLLED_CONTACT+NONE` 为HOLD-like、non-NONE为MANIPULATE-like。r1b addendum必须预先选择上表的机械规则：**HOLD一律映NONE；可执行的TRANSLATE effect必须在ontology中进入MANIPULATE。** 不能在看过pairing或policy结果后再决定。

`ROTATE + TRANSFER` 等并发可观察变化也不能由标注者自由选。shared manifest需要分离：

```text
intended_effect       # 来自冻结task/subgoal ontology，允许跨域监督
observed_effects[]    # annotation/benchmark，可多值，不进r1b wire
observed_outcome      # verify/supervisor，不进r1b wire
```

对pour，若预注册task goal是内容转移，则 shared `intended_effect=TRANSFER`；杯体旋转保留为 observed/auxiliary。失败片段仍保留 intended goal，不能因最终无transfer把action goal改成NONE。

每个human event最多产生两条domain-neutral constraint semantics：

```text
primary_role          # 同时为focus
primary_part
secondary_role | NONE
intended_relation | N_A
```

编译时永久删除：

```text
human anatomical side
human global center/scale/orientation
human part-local pixel cell
human bbox/keypoint/track/direction
observed relation_after as goal
outcome
```

`intended_relation`不能默认复制`relation_after`，因为失败/partial片段会把事后状态变成动作条件。它必须来自冻结task/subgoal ontology，再由当前robot planner确认；`relation_before/after`只作annotation、pair evidence和评测。

required phase/effect/role/part/relation任一 `REVIEWED_UNKNOWN` 或 `NOT_REVIEWED` 时，不生成shared semantic key；该字段只mask或令event unmatched，不能用`OTHER/N_A`吸收。

## 5. 精确 human/robot pairing key

现有 `pair_family_id/candidate_pair_id` 字段没有冻结其字节级生成规则；“task family × roles × intended effect/outcome”还不是可复现key。r1b应新增如下规范，但不把key送入action。

### 5.1 Semantic key

对human与robot event分别构造同一 canonical object：

```text
{
  "pair_key_version": "sm_fet_r1b_semantic_pair_v1",
  "task_ontology_version": <frozen string>,
  "task_family": <frozen enum>,
  "compact_phase": <r1b phase enum>,
  "intended_effect": <r1b effect enum>,
  "constraints": [
    {
      "primary_role": <enum>,
      "primary_part": <enum>,
      "secondary_role": <enum or NONE>,
      "intended_relation": <enum or N_A>
    }
  ]
}
```

`constraints` 先按

```text
(primary_role, primary_part, secondary_role, intended_relation)
```

严格升序，长度只能为1或2；再用RFC 8785 canonical JSON的UTF-8 bytes计算：

```text
semantic_key = SHA256(canonical_json_bytes)
```

key中明确**没有**：dataset/episode ID、object instance、frame/rank、duration、human side、robot assignment、human/robot global pixel、4×4 surface cell、64×64×32 locator、512 axis、extent、pose、outcome、confidence、CAPOT mass或appearance embedding。

`semantic_key` 相等只表示可进入event match候选，不自动生成gold positive；many-to-many、NULL、顺序和失败差异仍由reviewed pair edge与CAPOT处理。

### 5.2 Identity keys

split后才生成：

```text
pair_family_id = SHA256(
  canonical_json({
    "version": "sm_fet_r1b_pair_family_v1",
    "split_id": split_id,
    "task_ontology_version": task_ontology_version,
    "task_family": task_family
  })
)

candidate_pair_id = SHA256(
  canonical_json({
    "version": "sm_fet_r1b_candidate_pair_v1",
    "pair_family_id": pair_family_id,
    "human_dataset_id": human_dataset_id,
    "human_episode_id": human_episode_id,
    "robot_dataset_id": robot_dataset_id,
    "robot_episode_id": robot_episode_id
  })
)

event_pair_id = SHA256(
  canonical_json({
    "version": "sm_fet_r1b_event_pair_v1",
    "candidate_pair_id": candidate_pair_id,
    "human_event_id": human_event_id,
    "robot_event_id": robot_event_id
  })
)
```

以上 `canonical_json` 都使用RFC 8785 canonical JSON的UTF-8 bytes，不可裸字符串无分隔拼接。`split_id`属于identity/provenance，不属于semantic key；role/effect级event候选由第5.1节的`semantic_key`控制；pair graph connected component仍必须同split。

### 5.3 CAPOT 主 cost的r1b收窄

annotation protocol §5.1 曾允许node global `center/scale/orientation`进入 `d_object_local_geometry`；focus spec §5.4已要求跨相机屏蔽这些字段。对r1b，后一条必须成为normative supersession：

- 主 cost只使用compact phase、role/part、intended relation/effect、known-field mask与local order；
- contact mode、observed effect/outcome可作action-invisible matching evidence或ablation，但不进入shared r1b target；
- global center/scale/orientation、human part cell、pixel、bbox、pose、track、duration与appearance embedding全部禁止；
- object-centric part cell只有存在双域同一canonicalization receipt时才能作另名ablation，不能进入strict主cost；
- solver mass永远stop-gradient、不是GT；`nonmatch` mass为0，unknown/ambiguous不变成negative。

否则human图像布局会先影响pair edge，再影响robot语义监督，形成**间接跨域pixel channel**，即使human pixel没有显式写进26-byte packet也仍违反本任务边界。

## 6. 唯一训练期数据流

strict r1b只批准以下一条训练数据流：

```text
HUMAN DATA PLANE
human raw video
  -> isolated A/B event/role/part/relation/effect annotation
  -> adjudicated shared_semantics_v1
  -> no pixel geometry, no side, no direction
                              \
                               -> split-first candidate graph
                              /   -> reviewed event edges + offline CAPOT
ROBOT DATA PLANE             /    -> stop-gradient sparse pair table
robot rollout + robot action
+ current robot RGB/history
+ direct sensor/native geometry
+ 25D proprio/contact
  -> independent robot semantic labels
  -> robot_locator_source_v1
  -> deterministic allocator oracle/receipt
  -> r1b packet + robot action rows

MODEL TRAINING
robot current RGB/history/text/state
  -> robot semantic planner heads
       supervised by robot labels
       + matched human hard semantics / transition prior
         weighted by reviewed edge or stop-gradient CAPOT mass
  -> robot-only locator head
       supervised only by robot-view sensor/native labels
  -> compiler + deterministic allocator
  -> hard 26-byte packet / 5 tokens
  -> action expert
       supervised only by robot actions
```

损失所有权必须满足：

```text
human rows may affect: phase/effect/role/part/relation semantic losses
human rows may not affect: locator/depth/axis/extent/assignment/action targets
robot rows may affect: semantic + locator + action losses
CAPOT may affect: semantic pair weighting only
CAPOT may not affect: action input, cache, locator target, assignment or loss mask truth
```

最强可审计版本不把human frame送入robot模型：human pixel在人工/独立human event encoder处终止，跨域边界只传adjudicated hard semantics与field masks。若未来使用human encoder，它必须是独立离线branch，物化hard semantic sidecar后detach；其embedding、logits、KV、ROI与梯度不能连接robot planner/locator/action。

因此：

- **允许的间接影响**：human视频经人工裁决形成离散语义标签；这些标签监督robot图像上的语义预测或事件transition prior。
- **strict route禁止的间接影响**：共享human/robot视觉backbone预训练、human embedding contrastive alignment、pair-conditioned cross-attention、human geometry决定pair、human prompt memory进入action。它们可以是单独的`unpaired_visual_pretrain`或`human_prompt` baseline，但不是minimal-r1b strict证据。

human clip没有robot action label，绝不生成pseudo action、retarget wrist/grip、robot locator或assignment。human-only batch不能调用 `L_action`。

## 7. entry/turn/exit auxiliary 如何影响 locator 而不形成旁路

需要严格区分human与robot两域：

### 7.1 Human entry/turn/exit

human的7-keypoint pose、bbox、2D delta、aperture和direction只用于：

- 帮助独立human branch学习/检查event boundary、regrasp、phase与effect；
- 检查focus/part在event内是否稳定、何时应转event；
- 生成visibility/missingness统计和continuous challenger。

它们**不能监督robot locator**，不能生成robot camera anchor/axis/extent，也不能进入CAPOT strict cost。跨域后只剩裁决后的compact semantic label。

### 7.2 Robot entry/turn/exit

robot自己的entry/turn/exit/native geometry可以作为训练期 auxiliary，帮助robot perception trunk与categorical locator head学习：

- current role/part anchor；
- visibility和support是否跨多个bin；
- target-domain axis/extent类别；
- event切换时需要重新ground，而不是沿用旧cell。

允许aux loss更新robot perception/planner参数；部署时continuous aux head可删除或忽略。唯一进入action的仍是validator接受的hard locator code。

### 7.3 必须通过的无旁路不变量

1. fixed packet/proprio/text/noise/profile下，替换human视频、human pose、pair table、robot raw reference、aux target/logits/hidden，pre-safety action `max_abs<=1e-6`；
2. action attention/cache allowlist没有raw reference、pose/track、ROI/heatmap、planner/aux hidden或CAPOT；
3. 修改continuous aux label只能改变loss/下一次训练后的权重，不能改变同一次forward的action condition；
4. human aux gradient不能进入robot locator head或shared human/robot visual stem；
5. 若只有把entry/turn/exit track/hidden送入action才有效，该路线必须改名为continuous/event-track challenger，不能继续引用81.4779-bit packet claim。

training-only auxiliary改变训练后的参数不增加per-sample action-visible packet bits，但会改变方法容量和数据预算；实验必须给matched参数/FLOPs/数据control，不能把aux收益解释成有限语义本身的收益。

## 8. robot-view anchor 的唯一观测与模块责任

每个execute-8 replan anchor、每条present constraint都必须独立执行以下链：

```text
current target-robot RGB at anchor timestamp
+ causal robot history
+ calibrated direct depth/stereo/fiducial/native object state
+ current 25D proprio/contact
+ sensor-confirmed held-tool/object transform, if allowed
+ frozen task ontology/profile/calibration
  -> RobotRolePartResolver
       uniquely binds primary/secondary role instances and primary part
  -> RobotExecutionLocator
       estimates current camera-frame point/axis/extent with support
  -> R1BLocatorQuantizerValidator
       64x64 uv, 32 depth, POINT|AXIS|PATH, 512 axis, 8 extent
       checks support is within task tolerance and one admissible code
  -> validated functional constraints
  -> TargetEffectorAllocator
       outputs only LEFT_RH56 | RIGHT_OMNIPICKER | MONITOR_ONLY
  -> canonical compiler
```

模块责任不可交换：

- semantic planner决定role/part/relation/effect，不直接输出continuous target给action；
- `RobotExecutionLocator`决定robot-view anchor，allocator不得选择或修改anchor；
- allocator只能在已验证constraint与locator之后分配末端，不能选择profile、gain、decoder、axis、extent或candidate order；
- compiler量化、canonical sort、overflow与provenance验证，不能用planner confidence挑最近bin；
- action expert只读hard packet，不读上述模块的continuous estimate、support、logit、receipt或hidden。

oracle与predicted来源必须分报：

| route | locator truth/source | 允许结论 |
|---|---|---|
| oracle | synchronized native state、marker、calibrated direct depth/stereo、正式robot-view A/B annotation | 检查finite locator sufficiency |
| predicted | target robot当前因果RGB/sensor输入的locator model，带calibrated support | 检查planner恢复oracle gain |
| proposal | monocular depth completion、VLM、tracker无receipt、human point warp | 只能debug/ablation，不是oracle或GT |

需要新增action-invisible `robot_locator_source_v1`，至少绑定：

```text
dataset/episode/event/constraint/anchor timestamp
camera_id + calibration_id + profile_id
role/part binding receipt
locator kind + uv/depth/axis/extent source values and quantized codes
per-field KNOWN|REVIEWED_UNKNOWN|NOT_REVIEWED
evidence source/tier + temporal skew
support/uncertainty versus task tolerance
source hashes + compiler/validator receipt
```

没有这张表，r1b locator虽然有wire定义，却没有可审计的GT来源。

## 9. transparent / occluded 的 UNKNOWN 与 HOLD/REOBSERVE

r1b required字段没有action-visible UNKNOWN；unknown是planner/control-plane状态。唯一状态机应为：

```text
all required semantic + locator + assignment fields known and valid
  -> emit fresh packet

any required field unknown/missing/out-of-range/multi-bin/ambiguous
  -> emit no packet
  -> invalidate prior packet/action cache
  -> safety supervisor HOLD/REOBSERVE (or profile-defined safe retract)
```

具体规则：

1. transparent RGB只有2D overlap时不能给`TOUCHING/INSIDE`；没有direct/sensor/native depth时不能给depth code；没有可复核part frame时不能给axis。
2. occluded可以是 `OCCLUDED_TRACKABLE` planner evidence，但只有receipt-bound causal tracker在当前anchor输出的support完整落入一个安全bin且误差小于profile tolerance时，才能编known locator；否则 `LOCATOR_UNKNOWN`。
3. last-known locator、上一chunk packet、private controller memory、depth completion或human anchor不能被hold-last-value成新oracle。每8步必须有fresh validity receipt。
4. 两条constraint中任一required field未知，整个packet拒绝；不能静默丢掉stabilization/monitor constraint后执行第一条。
5. `EXHAUSTIVE no active event`才是全零NULL；unknown/missing/occluded不是NULL、PAD或nearest bin。
6. `CONTROLLED_CONTACT+NONE` 是**已知任务语义的hold-like动作**，仍要求合法locator/assignment；它不等于 epistemic `HOLD/REOBSERVE`。
7. `MONITOR_ONLY` 是已知、当前可验证的passive constraint，也不能吸收未知fixture/support。
8. safe hold由独立安全控制器只使用verified current proprio/contact/load执行；若继续保持接触不安全，则按冻结profile retract/stop。它不应调用旧r1b task packet推进任务。

对透明倒液，最可能的诚实结果是compile coverage下降。若必须依赖completed depth、continuous rim pose或action-visible reconstruction才能执行，应让reconstruction/direct visual servo challenger获胜，并停止r1b transparent sufficiency claim。

## 10. allocator、profile、planner 与 text 的隐藏连续几何审计

r1b已经禁止allocator logits/cost/continuous target进入action，但还需要堵住四类等价旁路。

### 10.1 Allocator

allocator的唯一action-visible输出是每constraint的hard enum。禁止：

- 修改locator code、target pose、axis、extent；
- 选择不同task profile、camera calibration、controller profile或gain；
- 用action-dimension reorder、candidate order、mask细节、timing或cache携带信息；
- 按scene输出continuous reachability residual；
- 把receipt/reason/hash送action。

相同公开输入必须byte-identical；ambiguous/infeasible只能abstain。

### 10.2 Task profile / calibration

profile只允许保存hardware/camera/task-family级常量：

- camera intrinsics/extrinsics；
- global `[z_min,z_max]`、`[d_min,d_max]`；
- quantization bin center的解析式；
- hardware limits、reachability/safety predicate；
- phase/effect/part/relation到controller family的有限、公开映射。

禁止profile包含per-episode/object/scene的pose、map、ROI、spline、trajectory、affordance field、per-bin learned residual、instance template或动作查表。禁止planner/allocator逐样本选择profile。registry必须在split前冻结并hash，test期间不更新；semantic/equal-bit route完全共享。

若一个checkpoint支持多个运行时profile，结果必须按profile分层，并把capacity表述为条件容量。若profile ID本身携带指定well、rim坐标或路径，它就是额外action channel，不再是81.4779-bit route。

### 10.3 Planner

planner可以从当前robot连续观测中选择一个有限locator code，这是设计本意；但action端只能看到hard code。禁止planner hidden/logits、continuous target/error、ROI、reference KV或confidence流入action/cache。

`current error`只能由：

```text
decoded locator bin center/axis/extent
+ fixed calibration
+ declared robot proprio/kinematics
+ verified held-object transform that contains no target-scene geometry
```

机械计算。若held-object transform是新增连续action input而非25D/fixed calibration的确定函数，必须单独声明、给所有baseline相同输入并说明packet容量不覆盖该robot state；不得把target pose或target-relative residual伪装成held transform。

### 10.4 Text

r1b允许instruction text直达action，所以“finite packet”从来不是全系统总信息量。strict评测应冻结text模板并禁止numeric pixel/depth/pose/axis/path、human coordinate或instance-specific trajectory。text可以给task family、named goal、dose、tighten/loosen和success criterion；不能给当前scene geometry。

### 10.5 最小审计

- fixed packet/profile/calibration/proprio/text/noise，替换planner raw geometry与hidden，action不变；
- fixed public allocator input，只改内部score/candidate order/receipt，hard assignment与action不变；
- fixed packet，profile registry中未被选中的条目变化，action不变；
- 相同profile ID对应的registry bytes必须hash一致；
- 删除任何per-scene profile/table后性能崩溃，则原结果属于continuous/profile-conditioned control，不属于r1b packet sufficiency。

## 11. 唯一部署期数据流

strict r1b只允许无human-prompt部署：

```text
current robot RGB + causal history
+ direct/native geometry sensors
+ current 25D proprio/contact
+ frozen allowed text/profile/calibration
  -> robot semantic planner
  -> robot role/part resolver + execution locator
  -> hard quantizer/validator
  -> deterministic target-effector allocator
  -> fresh canonical 26-byte r1b packet / 5 hard tokens
  -> action expert + split 25D decoder
  -> safety supervisor
  -> execute <=8 steps
  -> reobserve/recompile
```

部署时不存在：

```text
human video/frame/embedding/token/prompt
CAPOT/DTW/OT solver or coupling
pair_family_id/candidate_pair_id/event_pair_id
human pose/track/pixel/cell/direction
entry/turn/exit continuous memory
planner/locator/allocator hidden or logits
continuous target residual or per-scene profile
```

Idea G 的`human-prompt deployment + K event tokens + monotone selector`是有价值的独立候选，但它增加action-visible memory并允许human pixels经event encoder进入运行时，必须作为另一个schema/route/checkpoint报告，不能称为minimal-r1b strict。

## 12. r1b 必须修改什么、保持什么

| 项 | 裁决 | 是否改变81.4779-bit packet容量 |
|---|---|---:|
| 26-byte wire、5 tokens、grammar、assignment enum | 保持 | 否 |
| `human/robot共享语义、locator/assignment robot-only`一句话 | 保持并升为normative field-ownership表 | 否 |
| eight-kind→compact phase/effect投影 | **新增**，解决HOLD/TRANSLATE与ROTATE/TRANSFER歧义 | 否 |
| intended effect/relation 与 observed effect/outcome分离 | **新增** | 否 |
| exact semantic/pair identity key | **新增** | 否 |
| CAPOT屏蔽所有human global/pixel geometry | **新增并明确取代旧protocol cost** | 否 |
| strict训练期human像素终止点 | **新增**；主route禁shared visual backbone/human embedding | 否，但改变方法边界 |
| `robot_locator_source_v1`与oracle/predicted provenance | **新增** | 否 |
| entry/turn/exit分域auxiliary与swap audit | **新增** | 否 |
| transparent/occluded no-packet状态机与fresh receipt | **补强** | 否 |
| profile/text/held-transform anti-channel规则 | **补强** | 否；但要求capacity只作条件表述 |
| human-prompt部署 | 从strict r1b **删除**，保留独立baseline | strict packet不变 |

因此“是否需要修改r1b”的准确答案是：**需要修改规范，不需要扩字段。** 建议不要改现有wire schema ID与capacity receipt；新增crossdomain/compiler/profile contract IDs，并让checkpoint、dataset sidecar与实验receipt同时绑定这些版本。若实现阶段发现必须让human track、continuous locator residual、runtime human token或per-scene profile进入action，则必须建立新route/schema并重算信息边界，不能继续沿用r1b headline。

## 13. 进入实现前的 gates

本轮不实现。后续顺序必须是：

### X0：机械跨域契约

- eight-kind→compact语义投影唯一；
- intended/observed字段分离；
- semantic key与identity hash可复算；
- human side/pixel/geometry删除后round-trip稳定；
- unknown不生成key或packet；
- standard annotation rows不能被误当r1b locator rows。

### H1：human semantic可标性

只复用合法disjoint calibration与blind A/B；43/50/58继续是blind，不看proposal/review。逐字段报告PASS/FAILED/NOT_EVALUABLE，不用D0 proposal的`1/13/4`代替agreement。

### R0：robot locator/assignment可观测性

在目标机器人同任务数据上，分别验证每constraint的64×64×32、axis、extent、role instance与assignment；transparent/occluded单报compile coverage、unknown/reobserve与量化公差。human数据不能补任何一项。

### P0：pairing无geometry泄漏

先split后pair；CAPOT strict cost只读shared semantics与order；geometry/pixel intervention不改变pair候选/主cost；人工gold与solver mass分离。

### B0：action旁路审计

fixed hard packet/state/text/profile/noise下，human、pair、reference、aux、planner/allocator hidden swap均不改变action；profile无per-scene条目；unknown必然no-packet并清cache。

X0/H1/R0/P0/B0任一失败，不批准WAM训练。它们通过也只解锁无训练paired-world oracle；控制充分性仍需r1b已定义的locator、constraint、assignment oracle与continuous challenger比较。

## 14. idea-discussion 登记

按 Zotero-first `idea-discussion` workflow复用总idea card，以critic模式登记本审查：

```text
idea_id        = b4ec9e69-e9aa-4025-8d79-e36df14e0a5e
discussion_id  = 763d2cd1-bb1e-499c-bd69-3dae2006a6f4
memory_updated = true
zotero_synced  = 100 records
```

自动scout返回ST-VLA、ST-π与一个明显低相关的Minecraft action-space条目，并给出泛化的novelty风险提示。它没有回答本审查的字节级pair key、robot locator来源或profile旁路，所以未被用作规范裁决证据。

## 15. 最终一句话

human视频能教r1b的是**经A/B裁决、去side/去pixel/去geometry的当前功能事件语义**；robot必须在每个当前replan anchor用自己的相机、direct/native geometry与proprio重新生成locator，再由只输出hard enum的allocator分配末端。CAPOT只在训练前的稀疏语义pair table中存在，entry/turn/exit只在各自域内作auxiliary，transparent/occluded required geometry不确定时不编packet而由安全控制面hold/reobserve。按这个唯一数据流，r1b需要一个规范性addendum但不需要扩26-byte wire；在addendum、robot locator source与旁路gates完成前，跨域训练和部署均为No-Go。
