# Temporal Event Pairing WAM Meta-Synthesis

- 日期：2026-08-12
- canonical 基线：`e212afec9db5198a89930c7ff896e3e49a0c8249`
- 输入：Idea F / G / H，以及 canonical 中的 Idea A / B / D / E 与五案 meta-synthesis
- 工作范围：只读调研、结构设计与文档；未训练、未调用 GPU、未修改模型代码或数据
- 本轮 idea-discussion：`idea_id=b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`，`discussion_id=4ccb56ec-e102-4d76-b542-080aa0ea3280`

## 0. 唯一决定

唯一推荐主架构是 **SM-FET-WAM（Semi-Markov Finite Event-Tube WAM）**：视觉 planner 在 episode event clock 上预测最多八个带显式 onset、duration、contact edge 与 object-effect 的有限事件；动作侧每次只消费当前 active event 与未来两个 event 的 canonical hard packet，并继续输出 48-step action chunk。

这不是 F/G/H 的机械相加：

| 候选 | 最终角色 | 决定理由 |
|---|---|---|
| Semi-Markov finite event tube | **唯一主模型** | 明确回答“正确时间粒度是否是 event process”；与现有 finite route、censor contract 和小数据条件最相容 |
| ETG-WAM | **状态差强 baseline，不是第二个主模块** | `pre/post relation + object effect` 被吸收到 event 字段；不再保留任意图消息传递或第二套 action route |
| CAPOT | **训练期、离线、stop-gradient 的 pair-table builder** | partial/monotone/OT 均有直接前例；它不进入部署模型，也不承担算法 novelty |
| 22D `contact_segment_v1_1` | **冻结的 legacy baseline** | 保留旧 checkpoint 与已有 186.850009-bit control；主模型不同时拼接 22D 与 v2 |
| continuous hand/contact pose tube | **训练期监督、probe 和连续 upper bound** | 绝不进入主 action route/cache；若只有它 action-visible 时才有效，finite 主张失败 |
| AR variable-length tube / latent option / graph matcher | **baseline 或后备** | 第一版同时引入会混淆 duration、语义、pairing 与容量的因果归因 |

最终研究状态是：

- **Conditional Go**：冻结下面的 schema，做只读数据审计、12 段双标与 pair-overlap 清点。
- **No-Go**：当前启动 WAM/5B/GPU 训练、把旧 dual-Inspire 当目标 robot、做 RH56+OmniPicker 收益或真机 claim。
- **直接止损**：标签 gate、同任务 pair overlap、oracle v2 相对 22D sufficiency 中任一失败，就不实现主模型。

## 1. 主 claim：只守一个窄交集

一句可守的 claim：

> 在给定 instruction text、proprioception、embodiment 与 diffusion noise 的条件下，我们检验训练期的事件锚点弱配对能否被蒸馏成一个部署时不需要人类视频、容量精确受限的 Semi-Markov event tube；该 tube 只保留 active/near-future contact order、duration 与 object effect，并通过 direct、22D-only、full-trajectory 和 equal-bit controls 证明收益不是来自连续人手轨迹或任意视觉码。

这是一条 **conditional current-visual route + temporal abstraction** claim，不是“首次发明 event、OT、state graph 或 contact policy”。当前 action expert 仍直接读取 text；因此不能声称总信息瓶颈或所有 task semantics 均经过 v2。

三句绝对不能 claim：

1. **绝不能声称首次预测 temporal hand/contact、before/during/after state graph、keypose 或 object effect。** TOCH、ARCTIC、LatentAct、Ego4D FHO、SWIM、PPI、VLAff、BridgeACT、ST-VLA 等已经分别覆盖这些组成。
2. **绝不能声称首次做 human–robot partial、monotonic、many-to-many 或 OT alignment。** Drop-DTW、ImMimic/GMS-SDTW、EgoBridge、HARP-VLA、STRAP 已直接占据算法与数据协议空间。
3. **绝不能声称现有 66 Ego + 88 dual-Inspire 已教会 RH56+OmniPicker、实现 zero-shot transfer、物理 contact/outcome 或 causal action effect。** 两套数据任务、形态和真值均不重叠。

最强组合冲突不是单篇论文，而是：

- [EgoBridge](https://arxiv.org/abs/2509.19626) 的 human/robot DTW pseudo-pair + joint OT；
- [Drop-DTW](https://arxiv.org/abs/2108.11996) 与 ImMimic 的 GMS-SDTW 对内部 NULL、开放边界和长视频的支持；
- [Ego4D FHO](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)、[SWIM](https://arxiv.org/abs/2308.10901) 与 [BridgeACT](https://arxiv.org/abs/2604.23249) 对 PRE/CONTACT/POST、human-video world model、tool/target/contact/effect transfer 的合流；
- [LatentAct](https://arxiv.org/abs/2504.12284)、[VLAff](https://arxiv.org/html/2608.05215) 与 PPI 对 hand-contact trajectory、quantized trajectory 和 keypose-conditioned control 的覆盖。

所以 novelty 只能落在：**离线弱 pairing → 有限时序 packet 的蒸馏、部署零 human input、严格 route/equal-bit/field-intervention 证据**。任何单项名称都不是贡献。

## 2. 双时钟：禁止把 15 秒 episode 塞进 2.4 秒 action chunk

### 2.1 Episode event clock

episode clock 使用原始 episode 的稳定 frame identity，不依赖滑窗 query：

```text
episode_fps = 20 Hz（当前 Ego contract）
Q_episode_max = 8 macro events
R_edge_max = 2 contact/relation edges per macro event
event boundary = [lower_frame, upper_frame] + observed/censored/unknown
event state = event_kind + total/remaining duration + node/edge/effect/outcome
```

八类 Semi-Markov state 固定为：

```text
approach
precontact
acquire
hold
manipulate
regrasp
release
verify_outcome
```

合法主路径为：

```text
approach -> precontact -> acquire -> hold/manipulate
                                  -> regrasp -> manipulate
                                  -> release -> verify_outcome
```

允许合法 skip、自循环、`manipulate↔regrasp`；不要求每段出现所有状态。`event_id` 在全 episode 稳定，不能为每个 48-frame window 重新编号。

### 2.2 Action chunk clock

动作时钟不改变 canonical `H_action=48`、20 Hz，即预测 2.4 s action chunk。设计上的 receding-horizon execution stride 为 `R_action=8` frames（0.4 s）：执行前 8 步后重新观察、更新 event belief 并重编译 packet；这是未来需单独实证的 controller contract，不是当前已实现事实。

每次 action forward 只看到：

```text
K_visible = 3 macro events = active + next_1 + next_2
lookahead = 96 frames = 4.8 s
R_edge_max = 2 per visible event
```

`next_1/next_2` 只在 onset 落入未来 96 帧时出现；更远事件保持在 episode planner 内部，不进 action。active event 的 duration code 表示 remaining duration；future event 的 duration code表示 expected total duration。

### 2.3 只关注关键操作区间

训练时的 key interval 由 adjudicated event 支持机械生成：

```text
I_key = union([onset_lower - 10, end_upper + 10])
        union([verify_outcome_start, verify_outcome_end + 20])
```

即在 20 Hz 下，物理事件前后各留 0.5 s，outcome 最多额外留 1 s。准备、走动、等待、镜头调整、失败后与任务无关的恢复不参与 human–robot OT；它们保留为 NULL calibration 样本，而不是被迫对齐。

推理时没有 GT interval。planner 输出 `p(active)`、16-bin onset hazard、8-bin termination/duration hazard与 `p(NULL)`。建议 gate 冻结为：`p(active or start-within-96)>=0.7` 连续两个 replan 进入 key mode；`p(NULL)>=0.7` 或 `verify_outcome/release>=0.7` 连续两个 replan 退出。阈值只能在 calibration split 固定一次，不能在 test 上调。

### 2.4 start、end、NULL、unmatched 的语义

- **start**：训练用 exact/interval/left-censored onset likelihood；部署 packet 用 `already_active + 16` 个 future onset bins。
- **end**：训练用 conditional termination hazard；packet 由 onset 与 duration/remaining-duration code确定 end interval，不额外开放一个重复字段。
- **NULL event**：当前/未来 96 帧没有可信 task-relevant event；以 event `presence=0` 表示，所有尾部 literal `+0`。
- **unmatched event**：只属于 CAPOT pair table，表示某个 human/robot event 的质量未运输；它不是 robot 动作命令，永不进入 action packet。
- **unknown**：证据不足；loss mask，不等于 NULL、negative 或 no-contact。

## 3. `event_tube_v2`：精确有限 schema

### 3.1 版本决策

保留 `contact_segment_v1_1` 的 22D、Q12、B16 语义和 checkpoint，继续作为 legacy / 22D-only baseline。新建独立 `event_tube_v2`；**主方法不把 v1 与 v2 concat**。v2 可以机械投影成较粗的 v1 baseline target，但反向不能假装 v1 已包含 event/delta。

v2 的 action-visible packet 由三张固定表组成：

```text
node_packet  : [N=4,  D_node=42]
event_packet : [K=3,  D_event=51]
edge_packet  : [K=3, R=2, D_edge=84]
```

总 tensor 宽度是 `4*42 + 3*51 + 3*2*84 = 825` 个 hard factorized float entries；维度高不等于容量高，容量由下面的合法 mixed-radix alphabet 精确计算。

### 3.2 Node 字段

每个 node slot 是当前 active+future events 使用的 object/entity；slot 按 `object_role → first_track_frame → track_digest` canonical sort。track ID 仅写 sidecar/receipt，不直接送 action。

| 字段 | 宽度 | 基数 | 枚举/约束 |
|---|---:|---:|---|
| `presence` | 1 | 2 | absent 时整 slot literal `+0` |
| `object_role` | 4 | 4 | `manipulated / tool / receptacle / fixture` |
| `center_cell_2d` | 16 | 16 | 当前图像 object center 的 `4×4` bin |
| `depth_band` | 4 | 4 | `near / mid / far / unknown`；无可靠 depth 必须 unknown |
| `scale_band` | 4 | 4 | `small / medium / large / unknown`，阈值由训练 split 冻结 |
| `orientation_2d` | 9 | 9 | 8 个 45° bins + `unknown` |
| `visibility` | 4 | 4 | `visible / partial / occluded / out_of_view_or_unknown` |

每个 present node 的合法状态：

```text
4 * 16 * 4 * 4 * 9 * 4 = 36,864
S_node = 1 + 36,864 = 36,865
bits_node = log2(36,865) = 15.169964136519
```

这些是 action 可见的全部 current-image-derived object geometry。主 route 禁止再返回 learned ROI feature、连续 box、连续 depth/pose 或 track embedding。

### 3.3 Event 字段

| 字段 | 宽度 | 基数 | 枚举/约束 |
|---|---:|---:|---|
| `presence` | 1 | 2 | active/next event 不存在时 canonical zero |
| `event_kind` | 8 | 8 | 上述八类 Semi-Markov state |
| `onset_code` | 17 | 17 | `already_active` + 16 个未来 onset bins；每 bin 6 frames/0.3 s，覆盖 0–95 frames |
| `duration_code` | 9 | 9 | 8 个 duration bins + `right_censored_or_gt96` |
| `object_effect` | 8 | 8 | `none/translate/rotate/articulate/flow/deform/transfer/unknown` |
| `outcome` | 4 | 4 | `success/partial/failure/unknown` |
| `confidence_band` | 4 | 4 | calibrated `unknown/low/mid/high`；由字段熵与 coverage 确定性编译，不是自由 message head |

八个 duration bins按 processed 20 Hz frame 数固定：

```text
[1,2], [3,4], [5,8], [9,16],
[17,24], [25,32], [33,48], [49,96]
```

第九类表示 remaining/total duration 超过 96 帧或右删失。present event 合法状态与含 absence 的 alphabet 为：

```text
8 * 17 * 9 * 8 * 4 * 4 = 156,672
S_event = 156,673
bits_event = log2(156,673) = 17.257397051042
```

### 3.4 Edge 字段

每个 macro event 最多两个 edge，足以表示双手同时接触；tool/source 到 target/receptacle 的 object-object relation 放在 secondary bundle 内，不另开任意图消息通道。超过两条 reviewed edge 的 event 标 `edge_overflow` 并在 pilot gate 统计，不静默截断。

| 字段 | 宽度 | 基数 | 说明 |
|---|---:|---:|---|
| `presence` | 1 | 2 | absent edge 全零 |
| `effector_side` | 3 | 3 | `left/right/object_object`；pair cost不把 human side字面映射到 robot side |
| `primary_node` | 4 | 4 | 直接被 effector 操作的 node |
| `primary_part` | 8 | 8 | `body/handle/lid_cap/opening/rim_edge/trigger_button/tool_tip/other` |
| `primary_surface_cell` | 16 | 16 | primary part 的 `4×4` cell |
| `secondary_present` | 1 | 2 | false 时所有 secondary 字段 literal `+0` |
| `secondary_node` | 4 | 4 | 被 primary 影响的 target/receptacle/fixture |
| `secondary_part` | 8 | 8 | 同一 part vocabulary |
| `secondary_surface_cell` | 16 | 16 | secondary part 的 `4×4` cell |
| `relation_before` | 9 | 9 | 见下 |
| `relation_after` | 9 | 9 | 见下 |
| `contact_mode` | 5 | 5 | `none/stick/slide/roll/impact` |

relation alphabet 固定为：

```text
separate, near, touching, overlap_2d,
inside, contains, aligned_to_rim, supported_by, unknown
```

primary `part×cell` 有 `8*16=128` 种。secondary bundle 有一种 absent 状态，或 `4*128*9*9` 种 present 状态：

```text
S_secondary_bundle = 1 + 4*128*9*9 = 41,473
present_edge_states = 3*4*128*41,473*5 = 318,512,640
S_edge = 1 + 318,512,640 = 318,512,641
bits_edge = log2(318,512,641) = 28.246775389810
```

### 3.5 总 action-visible 容量

action 每次看到 4 node、3 macro event 和每 event 2 edge。codec 精确总 alphabet与条件视觉容量上界是：

```text
S_total = 36,865^4 * 156,673^3 * 318,512,641^6
        = 7,416,508,007,997,962,370,228,444,436,492,582,899,941,
          770,394,754,475,004,263,349,546,538,457,395,971,610,625

bits_total = log2(S_total)
           = 281.932700038062 bits
```

这比 v1 Q12 的 `186.850008943084 bits` 多 `95.082691094978 bits`，所以 v2 不能与 22D-only 直接解释为“语义更好”；必须同时报告：

1. 22D-only baseline；
2. 与 v2 **精确同 alphabet** 的 equal-bit unstructured control；
3. `K_visible`、N/R overflow、经验 entropy、code usage 与 rate–distortion 曲线。

canonical event/node order与语法会使实际 reachable packet 少于上述独立 slot alphabet；论文应把 `281.9327` 写成 codec-exact、保守 action-visible upper bound，而不是互信息。

### 3.6 Equal-bit control

equal-bit v2 使用相同 `[4,42] + [3,51] + [3,2,84]` shape、planner参数、辅助监督、action encoder、ST surrogate和 cache policy。对每类 slot 先编码为 mixed-radix integer，再用 receipt-bound、可逆的 affine permutation：

```text
u' = (a*u + b) mod S_slot, gcd(a, S_slot)=1
```

分别在 `S_node=36,865`、`S_event=156,673`、`S_edge=318,512,641` 上置乱，再解码回同 shape hard code。`a/b`、字节序和 SHA 固定；absence仍只有一个 canonical zero。这样 cardinality逐 slot和总 packet完全一致，action看到的是无字段局部性的固定置乱码。

该 control 是表示归纳偏置 control，不是信息删除：固定置换理论上可逆。如果它与语义 v2 持平，必须删除“显式 human semantics 必要”的 claim；如果它 collapse，先修利用率，不能反向证明 v2。

## 4. CAPOT 的位置：pairer，不是模型模块

### 4.1 Pair table，而不是同步轨迹

先按 source clip/rollout、operator、session/day、object instance、collection batch 和 calibration split，再建 many-to-many candidate graph。建议两个 versioned 表：

```text
episode_pair_v1
  pair_family_id
  human_clip_id, human_sha256, human_frame_range, human_fps
  robot_rollout_id, robot_sha256, robot_frame_range, robot_fps, embodiment
  task_family
  tool_role, target_role, receptacle_role
  intended_effect, observed_outcome
  pair_label: candidate_positive | hard_negative | easy_negative | unknown
  human_group_ids, robot_group_ids, split_id
  episode_confidence, provenance, adjudication_status

event_pair_edge_v1
  pair_family_id, human_event_id, robot_event_id
  human_anchor_interval, robot_anchor_interval
  gold_edge_label: match | nonmatch | ambiguous | unreviewed
  cost_kind, cost_contact, cost_role_part, cost_effect, cost_order, cost_geometry
  coupling_mass, human_null_mass, robot_null_mass
  crossing_flag, monotone_rank_h, monotone_rank_r
  effect_before, effect_after, outcome
  field_confidence, evidence_tier, solver_receipt_sha256
```

一个 human clip 可以出现在多个 robot rollout row 中，反之亦然；pair edge 才是训练权重单位。不能先构造 66×88 pair 再随机切 split。

### 4.2 Anchor 与 cost

CAPOT 只对 `I_key` 中的 v2 macro events 建 coupling。连续 MANO/wrist trajectory 不进入主 cost；可做 diagnostic baseline。每个 field distance 归一化到 `[0,1]`，预注册主 cost 为：

```text
C_ij = 0.20*d_event_kind
     + 0.20*d_contact_mode
     + 0.15*d_role_part
     + 0.30*d_relation_effect_outcome
     + 0.10*d_local_order
     + 0.05*d_object_local_geometry
```

- literal human left/right 不与 robot left/right直接计 cost；用 functional role、part、mode 与 effect。
- `d_relation_effect_outcome` 权重最高；相同动作外观但 effect 相反是 hard negative。
- geometry 只用 object-local coarse bins，避免 morphology 与 camera path 主导。
- unknown field被 mask，并对剩余权重重新归一化；unknown不计零距离。

event mass 为 `a_i=q_event*q_coverage*q_effect`、`b_j` 同理。离线目标：

```text
min_P <P,C> + eps*H(P)
      + tau_h*KL(P*1 || a)
      + tau_r*KL(P^T*1 || b)
      + lambda_cross*sum_{i<k,j>l} P_ij*P_kl
```

并显式加 NULL row/column。NULL cost 随证据置信度增加：高置信 task event 比低置信/遮挡 event 更难丢弃。求解后做 deterministic maximum-mass non-crossing projection；最终 `crossing_rate<1%` 才可写入 pair table。partial OT（固定 transported mass）与无 monotonic UOT 都只作 baseline。

### 4.3 正、负、NULL 与 effect

- **正 pair**：task family、functional role、contact mode、relation delta、effect/outcome与顺序一致；时长、路径、手型可不同。
- **hard negative**：同对象/语言但 effect错；同 task但顺序错；role反转；part错；无 transfer、spill/drop/collision；合成 crossing/time shuffle。
- **easy negative**：task与functional role均不同。
- **NULL**：自然准备、等待、机器人归零、只在一侧存在的 regrasp/recovery、真正不可判区间。
- **ambiguous/unknown**：mask或软权重，不当 negative。

### 4.4 stop-gradient 与部署

CAPOT 必须离线运行在冻结或 EMA event encoder上，输出 SHA-bound sparse pair table。Sinkhorn/UOT/投影图全部 stop-gradient；robot action loss不得回传并改变 coupling，pair ID、dense `P`、human frame和matcher hidden不得进入 action model。

主部署模式**不需要人类视频**。human clip 只在训练期形成 event-level distillation/alignment supervision。若未来研究 online human prompt，应是另一条独立实验，不得混入本主 claim。

## 5. WAM 结构与插入位置

### 5.1 单一主数据流

```text
current robot reference + causal history + state + text
                         |
                         v
visual Semi-Markov event planner
  node heads + event onset/duration heads + edge/contact heads + effect/outcome heads
                         |
                         v
episode tube (Qmax=8, training/planner state)
                         |
deterministic compiler: 4 nodes + active/next2 events + 2 edges/event
                         |
canonical hard event_tube_v2 + capacity/hash receipt
                         |
finite node/event/edge encoder
                         |
action expert(state, text, noisy action, hard v2) -> 48-step action chunk
```

没有独立 ETG message-passing module；`relation_before/after + object_effect/outcome` 已是 event heads。没有 CAPOT runtime module；没有 continuous pose input。

### 5.2 当前 canonical 可复用的位置

- v1 22D / equal-bit / capacity receipt：`world_action_model/contact_transition_schema.py:65-121,168-217`。
- 当前 planner 与 raw hidden isolation：`world_action_model/models/transformer_wa_casual_mot.py:1253-1285,1335-1403`。
- 当前唯一 `22 -> action width` 路由：同文件 `:1405-1445`。
- 当前 MoT attention mask调用：同文件 `:1809-1822`。
- 当前 strict cache拒绝 image conditioning、保存 packet receipt：同文件 `:2296-2335,2359-2499`。
- strict reviewed sidecar入口：`world_action_model/datasets/wam_lerobot_dataset.py:285-325,741-819`。
- 当前 grouped contact loss与 trainer入口：`world_action_model/trainer/wa_casual_trainer_mot.py:111-189,957-1055`。

未来若 Gate 0–2 通过，才考虑独立新增 `event_tube_schema.py`、event sidecar loader、planner heads与 v2 route；不能静默改 `contact_transition_schema.py` 的 v1.1 语义。本 memo 不实施这些修改。

### 5.3 Attention mask 与 cache contract

planner query可读：current reference、causal visual history、state、text、上一轮 canonical hard phase state；不可读 future video、GT event、GT action、CAPOT coupling或human frame。

action rows可读：state、允许的 text cross-attention、noisy action、hard v2 node/event/edge tokens；不可读 current reference、future visual、planner logits/hidden、posterior、continuous pose、ROI feature、pair ID或human memory。

future visual expert可维持当前 action-conditioned future prediction，但其 hidden不能回流 action。action-only cache只保存：

```text
hard v2 bytes + schema/capacity/hash receipt
discrete active event clock state
allowed state/text/action KV
```

禁止缓存 raw node/event/edge logits、current-reference KV、future/posterior KV、continuous geometry/MANO、CAPOT matrix。每 `R_action=8`、observation hash变化或 event transition时 invalidation。

### 5.4 Continuous hand pose 的唯一位置

2D functional landmarks、object-relative wrist/tool pose、可选 MANO/metric 3D只允许：

1. Ego event pretraining 的 masked auxiliary target；
2. event-aligned vs uniform/dense 的 compression probe；
3. full-trajectory / continuous upper-bound baseline；
4. annotation proposal，且保留 teacher/version/uncertainty。

它们不进入主 action packet、cache、CAPOT主 cost或 target hardware retargeting。若 hard v2 必须附加 continuous pose、spline coefficient、learned ROI或raw flow才能通过 oracle gate，应把方法降级为普通 trajectory-conditioned WAM，并停止 finite claim。

### 5.5 Covert-channel 防护

必须同时满足：

1. 所有 action-visible视觉字段有限、hard、可反序列化；absence与unused secondary literal `+0`，拒绝 NaN/Inf/`-0.0`。
2. node按 role/track digest排序；event按 onset→kind排序；edge按 effector→primary node/part/cell排序；query identity不带自由语义。
3. packet固定后替换 reference，pre-safety action `max_abs<=1e-6`；改变 target label bytes但不 oracle-forward，当前 action不变。
4. action对 raw planner logits/hidden、posterior/human embedding 梯度为零；cache/uncached同 dtype 输出在预注册容差内一致。
5. track permutation连同 binding同步更新时输出等变；slot permutation经 canonicalization 后 packet byte-identical。
6. 字段 intervention具 phase/side locality；只改 duration不能让全 chunk所有关节等比缩放。
7. packet固定后 action hidden 的 image/pair ID probe不高于预注册 chance+2 pp；同时报告 equal-bit利用率。
8. 所有 run记录 theoretical bits、empirical entropy、invalid rate、overflow、packet bytes/hash与实际 action-readable source receipt。

## 6. 三阶段训练设计（只定义，不建议现在运行）

### Stage E：Ego event pretrain

**不需要 human–robot paired data。** human/ego current observation 与 causal history进入 prior；training-only posterior可看 `I_key` 的未来稳定窗口，输出 stop-gradient target。目标：

```text
L_E = L_kind
    + L_onset_survival + L_end_survival
    + L_node + L_edge_contact
    + L_relation_pre + L_relation_post
    + L_effect + L_outcome
    + L_prior_posterior
    + L_pose_aux(training-only)
```

边界 exact/interval/left/right-censored 用 hazard/survival likelihood；unknown field mask。pseudo proposal与 human-adjudicated/sensor evidence 分层，不在 gold test 混用。

### Stage P：pair alignment / distillation

**唯一需要 human–robot paired data 的阶段。** 这里的 paired 是同 task/effect 的 episode或event candidate，不要求同步。冻结/EMA encoder生成 event features，CAPOT离线建 coupling；训练只读 stop-gradient sparse weights：

```text
L_P = sum_ij stopgrad(P_ij) * d(z_h_i, z_r_j)
    + L_hard_negative_rank
    + L_null_calibration
    + L_order_consistency
```

gold non-match永不因 OT 高分变 positive；ambiguous edge mask/soft label。Stage P 必须有 unpaired co-training、strict DTW、Drop-DTW/GMS-SDTW、EgoBridge-style OT对照。

### Stage R：robot policy

需要 robot current observation/state/action/outcome；**不需要运行时 human clip，也不需要每个 robot batch 配 human clip**。从 Stage P 初始化 robot event planner，robot侧重新预测 v2，action expert只消费 hard compiled packet：

```text
L_R = L_action_flow + L_visual_future
    + L_robot_event
    + L_packet_calibration
```

不在第一版加入 E 的 learned `F_R` effect critic：当前没有合格 action→realized-effect数据，加入它只会再引入一个不可验证模块。object effect 先作为 event监督与 rollout metric；未来有 native/sensor truth后，`F_R`最多是独立 evaluator/后续工作。

### no-GT / missing / censor 硬规则

- `coverage=exhaustive` 才允许 absence negative；positive-only/unknown不补零。
- contact已知而pose不可见：只监督contact；反之亦然。
- interval/censor进入likelihood，不把窗口边界当真实onset/end。
- 没有 pair GT 的 candidate可以由 CAPOT给弱权重，但不能进入gold alignment metric。
- effect/outcome不可观测则 unknown；透明液体不由 VLM/单目3D猜成 truth。
- `hand_status` 永远只作 open/closed actuator/grip proxy或其 action-loss validity，**绝不作 contact presence、onset、release、mode或effect GT**。

## 7. Benchmark、baselines、指标与因果控制

### 7.1 Benchmark 分层

1. **L0 annotation/compression**：现有 12/66 Ego；ARCTIC/Ego4D FHO/HOT3D仅按其 native label能力外测。
2. **L1 alignment**：真正同 task human–robot clips，event-edge gold + NULL；pair split先于edge生成。
3. **L2 oracle policy**：有 native contact/outcome 的 simulator或目标 robot；先证 v2 oracle sufficiency。
4. **L3 predicted policy/OOD**：RH56+OmniPicker硬件、group provenance与safety chain齐全后才开放。

### 7.2 必须 baselines

| 类别 | baseline |
|---|---|
| 单帧 | single current frame direct BC/WAM；single grasp/contact keyframe；single next keypose |
| 全轨迹 | full human trajectory DTW + BC；HARP-style strict keypoint DTW；ImMimic/GMS-SDTW；dense pose/contact tube |
| 无配对 | robot-only；EgoMimic-style unpaired co-training；TCN/TCC phase embedding |
| 对齐 | standard/soft-DTW；Drop-DTW；subsequence DTW；EgoBridge-style DTW-guided balanced OT；partial OT；UOT without monotonicity；CAPOT |
| 现有 finite | direct GWP；aux-only direct；22D-only hard-ST；22D oracle |
| event结构 | ETG current-delta only（无 duration/next events）；framewise phase；SM-FET active-only；SM-FET active+next2 |
| 容量 | v2 equal-bit；latent option同容量；Q/bit matched direct shadow |
| 上界 | oracle v2；soft v2；continuous pose/trajectory action-visible upper bound |

所有 policy comparison必须匹配 human/robot episode预算、backbone、robot demonstrations、optimizer steps、参数/FLOPs、action horizon、augmentation/noise与split。不能把更多标签或更大route冒充 event收益。

### 7.3 必做消融/干预

- single frame / uniform K / event K；
- 22D-only / event metadata-only / node+edge / full v2；
- 去 duration、去 next events、去 relation delta、去 effect/outcome；
- active-only / active+next1 / active+next2；
- full path DTW / partial / unbalanced / monotonic CAPOT；
- paired / weak-paired / unpaired；
- semantic v2 / equal-bit / event shuffle / onset delay / relation-before-after swap / effect shuffle；
- training-only pose / action-visible continuous pose upper bound；
- oracle / predicted packet；
- fixed-packet reference swap、cache/uncached、query/track permutation。

### 7.4 指标

**事件/标注：** event F1、phase edit score、boundary MAE、tIoU、duration NLL、censor coverage、node role/visibility F1、edge/contact macro-F1、relation delta exact match、effect/outcome macro-F1/ECE/Brier、Q/R/node overflow。

**pairing：** gold event-edge precision/recall/F1、NULL AUPRC/F1、crossing rate、transported mass、effect consistency、same-event retrieval R@1/R@5、pair confidence ECE。pair edge和滑窗不是独立样本；按human clip与robot rollout交叉cluster bootstrap。

**动作：** task/phase success、correct-first-contact、event-order violation、regrasp recovery、correct stop/release、post-effect verification、spill/drop/collision、preparation-motion ratio、post-success overrun。

**路由/系统：** theoretical/empirical bits、code usage、fixed-packet swap差、field-local intervention、image/pair-ID probe、action-only latency、replan deadline miss、cache memory/jitter。

## 8. 明确 stop gates

| Gate | 必要输入 | 通过阈值 | 失败动作 |
|---|---|---|---|
| **G0 schema/annotation** | 先做12个episode双标+裁决 | event F1≥0.80；kind/role/relation/effect κ≥0.70；contact mode κ≥0.60；boundary P50≤2 frames；tIoU≥0.70；`Q_episode>8`、node>4、edge/event>2 各≤5%；关键effect/outcome双侧unknown≤40% | 协议最多修订一次；仍失败则停止 human-auditable v2，不扩66 |
| **G1 true pair overlap/alignment** | 至少12 human clips+12同task目标robot rollouts与gold edges | NULL AUPRC≥0.80；crossing<1%；CAPOT anchor F1比 Drop-DTW/GMS-SDTW/EgoBridge-style最优者高≥5pp；effect consistency≥0.80；无all-null collapse | 没有同task/effect overlap或不胜强baseline：CAPOT只作数据工具，删除matcher novelty/paired claim |
| **G2 oracle sufficiency** | native/sensor-confirmed robot contact/outcome；oracle v2 | 相对oracle 22D-only，outcome-sensitive task success≥+8pp或event completion≥+10pp；相对full trajectory下降≤3pp；safety不退化 | 不实现predicted v2；若连续trajectory明显更强，v2降为auxiliary |
| **G3 predicted/causal** | G0–G2全过、matched controls | predicted恢复≥70% oracle gain；strict v2同时优于aux-direct、22D-only、equal-bit；group-OOD success≥+5pp且95% CI下界>0；≥75%可行field intervention具phase locality；reference swap≤1e-6 | 任一失败，删除对应semantic/finite/causal claim；只连续pose有效则停止finite路线 |
| **G4 target hardware** | driver/mapper/calibration/watchdog/E-stop/group data | action latency增量≤15%；至少3 task families；success和contact指标CI不由单一object/session驱动；无safety regression | 保持RH56+OmniPicker部署No-Go |

当前只允许 G0 前的只读/标注工作；G1 的“同任务目标 robot”输入都不存在，G2–G4更无证据。

## 9. 现有 66 Ego 与旧 dual-Inspire：只允许做什么

### 9.1 66 Ego（pour，20,013 processed frames，20 Hz）

可做的低成本工作：

1. 从12个formal task起步，双标 event kind、interval/censor、4-node role/2D coarse bins、最多2条edge、relation before/after、visible effect/outcome与unknown；
2. 统计每episode event数、node数、edge数、duration、regrasp、censor、visibility和Q/N/R overflow，决定 schema是否应缩减；
3. 机械编译 oracle v2 packet，做byte canonicalization、capacity、slot order、NULL与双时钟的只读审计；
4. 比较 event-aligned K 与 uniform K 对 reviewed contact/effect的压缩保真度，但不训练5B、不报robot gain；
5. 建 `pair_overlap_manifest`，明确当前没有同task目标robot positive；
6. 记录哪些 outcome 从RGB不可判，为后续传感器采集定义需求。

不能做：物理contact GT、metric 3D/liquid force、robot action gain、production OOD、RH56/OmniPicker feasibility。`hand_status/hand_active` 只用于grip proxy语义与missingness审计。

### 9.2 旧 88 dual-Inspire（grab_red_bottle，40,901 frames，50 Hz）

可做：

1. 对少量可视 episode 复用同一 v2 指南，检查 ontology能否跨human/legacy robot而不改字段定义；
2. 只读检查 robot event clock、action chunk时间单位、event boundary与可见outcome能否标注；
3. 把 Ego-pour ↔ robot-grab 明确作为 task/effect mismatch、NULL-heavy negative，检查pair builder是否拒绝强制完整对齐；
4. 做 synthetic prep/idle/regrasp/speed-warp 的 pair-table单元审计。

不能做：把 dual-Inspire 六通道改名为OmniPicker aperture、把grab当pour positive pair、训练target effect model、验证target硬件或cross-embodiment success。

## 10. 最小同任务 pair 补采设计（RH56+OmniPicker 后续）

硬件 Gate 未闭环前只设计、不执行。最小数据不是同步frame pair，而是 12 个 group cell：

```text
3 task families
* 2 physical object-instance conditions
* 2 session/day groups
= 12 cells

每cell：3 human clips + 3 robot rollouts
总计：36 human clips + 36 target-robot rollouts
```

三个 family 建议：

1. pick/place：`separate→held→inside/on→released`；
2. right stabilize + left twist/press/insert：真正需要 RH56+OmniPicker asymmetry；
3. pour/transfer：`held→aligned→flow/transfer→upright→release`，outcome需秤/液位或人工+传感裁决。

每cell的human clips至少含：标准成功、额外prep/速度变化、regrasp或部分失败；robot rollouts至少含：两个成功/可恢复样本和一个failure/partial，且保存真实 action、state、时间同步、object/outcome、safety事件。不同human operator、robot operator/collector、session、object instance与calibration必须进入group metadata。

标注预算：所有72段标 event anchors；至少20%双标+全分歧裁决；每个family人工确认不少于20个match、20个NULL、20个hard-negative event edges。先按group冻结split，再从同cell生成many-to-many candidate；不能让同一source clip/rollout经不同pair edge跨split。

这只是 G1/G2 pilot规模，不支撑最终OOD论文。若 target driver、mapper、标定、watchdog或sensor outcome未闭环，补采仍是 No-Go。

## 11. 最小下一步：只读与标注，不训练

按顺序只做以下四件事：

1. 写 `event_tube_v2` annotation dictionary 的纸面/表格版本；用3个Ego episode做指南校准，不改模型代码。
2. 盲法双标12个现有formal episodes，裁决后只报告G0字段一致性、unknown、duration和overflow。
3. 对旧dual-Inspire抽取12个可视episode做ontology-only复标，并产出“Ego-pour/robot-grab无positive overlap”的pair manifest。
4. 只读枚举 v2 codec/canonicalization/equal-bit receipt需求，形成未来实现任务书；不生成训练config、不启动小模型或5B。

只有 G0 通过且未来确有同task target-robot pair，才重新讨论 CAPOT CPU benchmark；只有 G1 与 robot oracle数据就绪，才讨论任何 policy训练。

## 12. 最终 Go / No-Go

| 决策项 | 结论 |
|---|---|
| 唯一主架构 | **SM-FET-WAM** |
| 主 action packet | **独立 `event_tube_v2`；4 nodes + active/next2 events + 2 edges/event；281.932700038062 bits** |
| 22D v1 | **保留、冻结、只作legacy/22D-only baseline；不与v2拼接** |
| CAPOT | **离线训练期pair-table builder；stop-gradient；部署不需要human video** |
| ETG-WAM | **强state-delta baseline；其effect字段被v2吸收，不另建主图模块** |
| continuous pose/MANO | **训练期aux/probe/upper bound；action-visible主route No-Go** |
| 当前66+88数据 | **仅annotation、clock、overflow、NULL rejection与schema pilot** |
| 现在训练/GPU/5B/真机 | **No-Go** |
| 当前研究推进 | **Conditional Go，仅限G0前只读与标注任务** |

最重要的止损判断不是“CAPOT对齐得多漂亮”，而是 **oracle v2 是否在不读raw image/continuous pose的情况下，比22D-only多保留了 duration、order 与 object effect所需的控制信息**。在这件事被目标robot/native outcome证实前，不建议训练任何新增模型。
