# WAM 生化操作研究 Idea 索引（2026-08-12）

## 使用规则

- 本索引只描述研究文档，不代表代码实现、模型训练、GPU/服务器使用或硬件部署许可。
- 总体 `idea_id`：`b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`。
- 最终 portfolio critic `discussion_id`：`240c8a16-b6a2-4119-918f-711590af88ea`。
- 裁决优先级：最终 portfolio > final architecture / standard-r1 schema / 修订 annotation protocol > 专题 contract > 单独 Idea memo > temporal meta / op22 历史总览。
- 表中的 `source commit` 指产生所收录最终研究 bytes 的原始来源提交；canonical 已有且无需重复导入的文档会同时注明其 canonical 最终 bytes 提交。
- `Conditional Go` 只允许进入相应数据、标注或实验 gate；不等于允许直接实现大模型、训练或部署。

## 必须先读的两项纠正

1. `281.932700038062 bit` 已被 `207.749746919520 bit` 替代。前者只能保留为旧 independent-slot syntactic upper bound；现行 `sm_fet_hard_grammar_v2_standard_r1` 的 validator-reachable action alphabet 是后者（最小索引宽度 208 bit）。
2. `HAND_UNRESOLVED` action-visible actor 是候选 r2，不是现行规范。r2 的 `210.24289126401328 bit` 口径及其 action admission 已被 r1 裁决否决；现行做法是 annotation-side `CONFIRMED_PRESENT + REVIEWED_UNKNOWN`、actor loss mask、packet fail-closed 与 `HOLD/REOBSERVE`。

## HEADLINE

| 文档 | idea / discussion IDs | source commit | 状态 | 过时关系 |
|---|---|---|---|---|
| [最终 Idea Portfolio](./wam_biochemical_manipulation_idea_portfolio_20260812.md) | idea `b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`; discussion `240c8a16-b6a2-4119-918f-711590af88ea` | `3c545aa31469272be68a9c94ca51fa7c17aa3331` | **最终总裁决；唯一 Headline 是 SM-FET-WAM v2-min**。当前只允许冻结契约和 3+12 标注 gate | 覆盖 A–L 单独 memo 的角色判断；与 standard-r1 冲突时以本表所列最终规范为准 |
| [SM-FET v2 最终架构裁决](./sm_fet_wam_v2_final_architecture_decision_20260812.md) | 继承总体 idea；discussion 未单列 | `6bc51b90002f7f533d7b1ff523caf562949eab64` | **Authoritative architecture**；固定 standard-r1、active+next2、13 hard tokens、训练期离线 partial pairing | 否决 temporal meta 对 `281.9327` 的 codec-exact 表述；其架构裁决高于旧 meta |
| [SM-FET v2 Schema](./sm_fet_wam_v2_schema_architecture_20260812.md) | 继承总体 idea；discussion 未单列 | `190f3023da6ea6e0ee8251da22e2f21efea17bcc` | **Authoritative schema**；82 canonical `uint8` codes，exact reachable capacity `207.749746919520 bit` | 将旧 `281.932700038062 bit` 降级为 independent-slot syntactic upper bound |
| [修订后的 Annotation / Pairing Protocol](./sm_fet_annotation_and_pairing_protocol_20260812.md) | 继承总体 idea；discussion 未单列 | `288ef5dcf4e6299871dc50d33caa603e46345fcd` | **Authoritative protocol（fix1）**；D0 尚未通过，3+12 只检验实际覆盖字段 | `288ef5d` 修订并取代 `66f2857` 原协议中与 standard-r1 冲突的 skip、early release、release outcome 和 secondary geometry |
| [Idea F：Temporal Contact-Pose Tube](./idea_f_temporal_contact_pose_tube_review_20260812.md) | idea `a78673af-f56e-4ad1-9974-beee1edb765b`; discussion `8562848c-0b71-485e-abff-064f0900159c` | `20421c7502e6fa3febfe2234db9e414594c5f964` | **HEADLINE（经最终裁决收敛为 SM-FET-WAM v2-min）**；独立“首次时序 hand-contact trajectory”主张 No-Go | 其早期独立 memo 由 schema、final architecture 和最终 portfolio 收敛；容量口径以 standard-r1 为准 |

## SUPPORTING

| 文档 | idea / discussion IDs | source commit | 状态 | 过时关系 |
|---|---|---|---|---|
| [SM-FET 新颖性对抗审稿](./sm_fet_wam_novelty_adversarial_review_20260812.md) | idea `6c12acbf-343e-4b07-9814-96696416a05c`; discussion `b076f98d-d7ce-49da-842a-ca451352a9b0` | `260ac320565c2a156e0036a638358e790a784110` | **Supporting critic**；组件拼装叙事不够新，只保留严格有限、可审计、跨具身接口这一窄命题 | 不覆盖最终架构；作为最终 portfolio 的 adversarial novelty gate |
| [Benchmark 与数据路线图](./sm_fet_wam_benchmark_and_data_roadmap_20260812.md) | idea `b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`; discussion `c947c620-3f8a-42b1-a52c-f2a11cee581b` | `05698bb8cafbd520a0700466e8d43e422a12588d` | **Supporting roadmap**；D0→3+12→36+36→oracle→predicted/causal，未过 gate 不训练 | 以 final architecture 和修订 protocol 为上位规范 |
| [Temporal Event Pairing Meta-Synthesis](./temporal_event_pairing_wam_meta_synthesis_20260812.md) | idea `b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`; discussion `4ccb56ec-e102-4d76-b542-080aa0ea3280` | `89396591720035d0427679e07bebfad5fed60023` | **Supporting historical synthesis**；保留 SM-FET 与 offline partial pairing 的来源推理 | 容量 `281.9327` 和与 r1 冲突的细节已被 schema / final architecture / protocol fix1 取代 |
| [人类观察工具学习证据](./human_observational_tool_learning_evidence_20260812.md) | idea `f3dfb6e9-d244-4188-95f7-6aca1c537947`; discussion `5c525c2a-5a71-4ab8-9b22-c17a87667783` | `52e763c62856c350245bb38ba0509a2e9da497fe` | **Supporting rationale**；只支持“事件锚点 + 局部运动”的弱归纳偏置 | 明确否定固定八阶段是人类认知状态机、三姿态已被人类实验验证等强叙事 |
| [Dynamic Focus / 操作位置契约](./sm_fet_dynamic_focus_token_spec_20260812.md) | idea `b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`; discussion `092fc9ae-f268-4aa1-b79e-69cb44e6afe9` | `2452e60bbaf6edb4714fed7d1397aea6ba9e4111` | **Supporting contract**；复用 focus/node/part-local cell，不新增连续位置 token 或 action bit | 从属于 standard-r1；任何连续 ROI/hidden action bypass 均 No-Go |
| [Missing Hands Contract（修订最终 bytes）](./sm_fet_missing_hands_contract_20260812.md) | idea `b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`; discussion `2de8b06f-ab83-4197-8fc2-3c903374ce08` | `ca7d641703749a15ff30c644250f5147b5125e72` | **Supporting contract（fix1）**；reason-coded、field-level、per-side、fail-closed，保持 r1 四码 actor | `ca7d641` 修订初稿 `4e84236`；早期 action-visible `HAND_UNRESOLVED` r2 提议由 reconciliation 否决 |
| [Missingness / Capacity Reconciliation](./sm_fet_missingness_capacity_reconciliation_20260812.md) | 继承总体 idea；discussion 未单列 | `9f6eeff6761168048b49101cb5839170ae6b84bc` | **Authoritative supporting reconciliation**；保留 r1，reviewed side ambiguity 不产生合法 oracle/action packet | 明确否决当前 `HAND_UNRESOLVED` r2 admission；r2 仅可在目标机器人独立 gate 后另立版本重审 |
| [透明容器/液体非重建契约](./sm_fet_transparent_nonreconstruction_contract_20260812.md) | idea `f875e240-6c50-4949-9d1b-ccb7cbcaefed`; discussion `d8e376b4-28e4-4497-8c6d-9510aae900e1` | `f795c4f615595d72219e79a0cb43d5adef12b0a0` | **Supporting contract**；只保留可审计 2D/关系/效果证据与传感器确认 outcome | 透明 3D reconstruction、伪 depth/flow/volume GT 被降为 baseline 上界或 No-Go，不是主方法依赖 |
| [Idea B：Asymmetric Contact Topology](./idea_b_asymmetric_contact_topology_review_20260811.md) | idea `24a3b591-7839-4593-8cef-169e01acf1d6`; discussion `5e9ca133-76e3-4818-8322-fa6e60185a61` | `07821f0d3626f16c22cecb54091a058e6f162c42` | **SUPPORTING**；只保留目标硬件 feasibility allocator、split decoder 与 tactile residual | 大 topology packet 不进入 v2；human finger identity 不直接映射机器人关节 |
| [Idea C：Visibility-Gated 2D Auxiliary](./idea_c_visibility_contact_aux_review_20260811.md) | idea `5757566f-4c34-4cf1-aecf-0dfac76a8ab9`; discussion `e1e27a11-cb4b-4ac8-bef8-28dd3e51f634`（初次 `7ef28570-2d28-45c8-8b9b-8e9d74fcb3cf`） | canonical final bytes `e212afec9db5198a89930c7ff896e3e49a0c8249`；原研究 `de3cbc9ffe38e58cbdcb995cb4684a634c696642` | **SUPPORTING，training-only**；canonical 已有，未重复导入 | 原 memo 的训练/论文主张 No-Go；最终 portfolio 只保留 visibility-gated 2D auxiliary，禁止进入 action route |
| [Idea D：Censored Bimanual Benchmark](./idea_d_censored_contact_benchmark_review_20260811.md) | idea `15d89a15-c958-4a33-9735-a9547d45fa27`; discussions `dbf2e3a4-9d4d-436a-9e74-4d727ca8d037`, `90187152-d080-45c4-b5ef-b9f12f1a63a0` | `cd6629f5c7fab60cfcaacd394207e817f364876a` | **SUPPORTING** 数据/监督/评测契约；独立论文当前 No-Go | missingness 编译细节以后续 standard-r1 protocol 与 reconciliation 为准 |
| [Idea E：Contact–Action Effect Consistency](./idea_e_contact_action_consistency_review_20260811.md) | idea `8f386f3f-50be-45b5-970d-4bca4bd1a062`; discussion `bba6e215-6a61-4203-9c03-696c17a42bf5` | `1534ede3e47b9f26980fc2ddcc5805d88b205d8e` | **SUPPORTING，后置 causal/faithfulness audit**；canonical 已有，未重复导入；D3 oracle 前不实现 | 作为独立 headline 与当前数据收益 claim 为 No-Go |
| [Idea G：Partial Human–Robot Pairing](./idea_g_partial_human_robot_alignment_review_20260812.md) | idea `e6561894-fa5a-4e44-912f-24524aac7764`; discussion `4cbab6ef-7c05-412b-a600-3797f393370e` | `9f60f04cb3d0d6a45ee64ef77f71d04b2d722adb` | **SUPPORTING，offline training data module**；允许 NULL、many-to-many 与 unmatched | partial OT 本身不是 headline；当前 66 pour + 88 grab 不得组成正配对 |
| [Idea I：Causal Critical Windows](./idea_i_causal_critical_windows_review_20260812.md) | idea `a0da9ceb-7748-4ef5-910c-e57b7cf041dd`; discussions `08578342-f03a-4e9d-a454-0fe1f373b2ef`, `a32a9415-7fc6-464d-9bc3-97ee10c0ccf8` | `0b67bee397f92224096d2b37f1a0cc713a1410f6` | **SUPPORTING，条件式 planner evidence front-end**；首版不启用 | 独立 WAM / 独立论文主张 No-Go；不得让 raw window 或 selector hidden 进入 action |
| [Idea K：Effect-Equivalent Quotients](./idea_k_effect_equivalent_interaction_quotient_review_20260812.md) | idea `2823cc3b-c066-4e44-ac77-b445f260a683`; discussion `4749b58a-5b7c-4ff9-ab5b-87c30ea6b95d` | `23e3834afe3abc389b8968ea95b6a6cf59e91d92` | **SUPPORTING，后置 training-only grouping**；首轮不启用 | memo 中旧 `281.9327` 引用已过时；独立 novelty / action-visible quotient ID 均 No-Go |

## BASELINE

| 文档 | idea / discussion IDs | source commit | 状态 | 过时关系 |
|---|---|---|---|---|
| [op22 Contact-Transition WAM Dossier](./contact_transition_lab_wam_dossier_20260811.md) | 未声明 | canonical final bytes `d2bf1903227d8398b2c3a229554d861030a881f8`；op22 origin `ce56584e995207bf24702e8ccdb7b2161a6a16aa` | **历史 dossier / BASELINE 背景**；canonical 已有，保留较新修订，未用旧 worktree 覆盖 | 其 headline 与 A–E 章节是历史总览；SM-FET standard-r1、最终 portfolio 和新 A–L 分类优先 |
| [Contact-WAM A–E Meta-Synthesis](./contact_wam_meta_synthesis_20260811.md) | 汇总 A–E IDs；未另立 ID | `279ddf49faef8dd6f3367d91c44c1c2968839cb7` | **历史 v1 决策 / BASELINE 背景** | 唯一推荐主线已从 v1 Idea A 演进为 SM-FET v2-min；186.850009-bit v1 仍作互斥基线 |
| [Idea A：Finite Contact WAM](./idea_a_finite_contact_wam_review_20260811.md) | idea `2fd0a8a6-0efa-4038-9aeb-93a8378e41c0`; discussion `44cf5abe-05c6-477d-93ce-c3dad8a6a3eb` | `70af53256e1936236c61883c64c291a887addc90` | **BASELINE / inherited contract**；v1 22D/Q12 及严格路由原则保留 | 不再是最终 headline；与 v2 不得 concat |
| [Idea H：Event-State-Transition WAM](./idea_h_event_state_transition_wam_review_20260812.md) | idea `f759ccbb-735b-45e4-8efe-158b79a3ce90`; discussion `2ca05e6f-75c5-49df-bbc9-2c8a68d292c9` | `8f0432aa98fab8920f99495db3fc452a57503abc` | **BASELINE**；ETG current-delta-only 互斥对照 | relation-before/after、effect/outcome 已被 v2 字段吸收，不再增加独立 graph network |
| [Idea J：Pose–Contact Change Signature](./idea_j_pose_contact_change_signature_review_20260812.md) | idea `383b218b-5c66-4503-b977-530c7f7b3094`; discussions `611d78e5-f36f-4332-b2b5-abebb699925a`, `4781abe9-bac9-40cd-8408-31acd30deff2` | `85dc53c02a10609dde5777364fdd26daad453a4a` | **BASELINE / challenger**；F-PCD-only 或 training-only auxiliary | memo 中 `281.9327-bit SM-FET` 已过时；不得与 v2 形成新增 action-visible route |

## NO-GO

| 文档 / 决策 | idea / discussion IDs | source commit | 状态 | 过时关系 |
|---|---|---|---|---|
| [Idea L：Sparse Interaction Motif Dictionary](./idea_l_sparse_interaction_motif_dictionary_review_20260812.md) | idea `76f0e0a1-cc36-4e72-88ed-fb063f752297`; discussion `6707f93d-9734-4ca1-8858-fa9b24cd6ec1` | `f2059a0e276bf4eecd70af010f6c4df8396438c6` | **NO-GO**：当前首轮、独立 headline 与 learned motif latent 直达 action 均拒绝 | 仅在 event recurrence 与 matched-output planner gate 通过后，可作为 episode-level challenger 重审 |
| [旧 281.9327-bit capacity claim](./temporal_event_pairing_wam_meta_synthesis_20260812.md) | idea `b4ec9e69-e9aa-4025-8d79-e36df14e0a5e`; discussion `4ccb56ec-e102-4d76-b542-080aa0ea3280` | `89396591720035d0427679e07bebfad5fed60023` | **NO-GO 作为 codec-exact / reachable capacity** | 已由 [schema](./sm_fet_wam_v2_schema_architecture_20260812.md) 和 [final architecture](./sm_fet_wam_v2_final_architecture_decision_20260812.md) 的 `207.749746919520 bit` 替代；只可称旧 syntactic upper bound |
| [`HAND_UNRESOLVED` r2 action admission](./sm_fet_missingness_capacity_reconciliation_20260812.md) | 继承总体 idea；discussion 未单列 | `9f6eeff6761168048b49101cb5839170ae6b84bc` | **NO-GO（当前规范）**；不扩 actor 字典，不沿用 r1 的 208-bit 口径 | r2 已被 r1 fail-closed 裁决否决；若未来复活必须另立 schema/grammar/codec、211-bit equal control 与目标机器人 gate |
| [透明 dense reconstruction 主路线](./sm_fet_transparent_nonreconstruction_contract_20260812.md) | idea `f875e240-6c50-4949-9d1b-ccb7cbcaefed`; discussion `d8e376b4-28e4-4497-8c6d-9510aae900e1` | `f795c4f615595d72219e79a0cb43d5adef12b0a0` | **NO-GO**：伪 depth、mesh、无传感 flow/volume GT 或 reconstruction-dependent action route | ClearGrasp / liquid reconstruction 等只可作上界或 baseline；现行主线是 non-reconstruction contract |

## 当前统一状态

- **现在 Go**：只冻结 ontology/schema，设计 D0 checker/round-trip，并执行 3 calibration + 12 blind annotation。
- **Conditional Go**：D0 和 3+12 的实际覆盖字段通过后，收集 36 human + 36 目标机器人同任务 pilot 与 180 条 pair gold。
- **现在 No-Go**：实现/训练大 WAM、GPU/服务器训练、目标硬件部署、当前 66+88 正配对、透明 3D 重建，以及任何未经 gate 的论文收益 claim。
