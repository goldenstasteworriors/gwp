# 人类观察工具操作的阶段性证据：从接触事件、目标层级到连续轨迹

- 日期：2026-08-12
- 审查基线：`e212afec`
- 隔离分支：`codex/human-observational-learning-20260812`
- `idea_id`：`f3dfb6e9-d244-4188-95f7-6aca1c537947`
- `discussion_id`：`5c525c2a-5a71-4ab8-9b22-c17a87667783`
- 范围：认知科学、动作观察/运动控制、神经科学、观察学习及机器人表征；只讨论证据，不修改模型、数据、GPU 或训练

## 结论先行

**有条件支持“事件锚点 + 局部运动”的归纳偏置，不支持“人类把工具操作天然压缩成固定五阶段、因而可丢弃阶段间轨迹”的强叙事。**

当前最强、最贴题的证据是：人在分割自然物体操作时，主观事件边界高度聚集于 touching/untouching 变化；脑活动在接触附近出现视觉变化响应，在随后脱离附近出现更广泛的更新相关活动。但同一研究也给出关键反证：多数 touching–untouching motif **没有**被报告为事件边界。因此接触变化是强候选锚点，却既非充分条件，也未被证明是观察后学会执行工具任务所需的最小表征。[Pomp et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118534)

另一组较直接的证据表明，动态信息确实不能普遍被单张终态或若干静态图等价替代：完整抓取视频而非单张接触帧引发抓型特异的皮质脊髓反应；自然/恒速运动轨迹而非仅终态支持新颖运动序列学习；手工结绳、解环等教学中动画通常优于静态图序列。[Bunday et al., 2016](https://doi.org/10.1016/j.cortex.2016.09.002) [Roberts et al., 2015](https://doi.org/10.1016/j.actpsy.2015.05.011) [Ayres et al., 2009](https://doi.org/10.1016/j.chb.2008.12.013)

但这仍不能推出“全程逐帧轨迹最优”。目标、动作结果和对象关系可对较大幅度的视觉/运动变化保持不变；观察者的前瞻注视常跳向目标位置；有些运动观察学习对示范试次顺序不敏感；教学动画优势也可以在不显示手时保留。因此，更可守的 SM-FET 认知动机是：

> 接触建立/解除、对象状态效果和任务目标是候选的高信息事件锚点；精细抓取、避障、插入、重抓等任务仍可能需要锚点邻域或锚点之间的连续运动。是否能以少量事件窗口替代稠密轨迹，是待机器人实验检验的工程假设，而不是既有的人类认知结论。

## 检索方法与证据分级

### Zotero-first 记录

本轮先按 `idea-discussion` 工作流同步并查询个人 Zotero，再查原始论文、PubMed/PMC、出版社页面与 arXiv。Zotero 同步返回 100 条；库中更接近的是机器人侧的人类视频学习、affordance 和 3D trajectory 工作，例如 Open-AoE、2HandedAfforder、Dexterous Manipulation from Human Videos via 3D Trajectories，以及 RT-2、RoboBrain、π0.5 等通用条目。它们可界定机器人前例，但库内没有命中本问题所需的经典 event segmentation、end-state comfort、运动共振与教学动画实验。

自动扩展检索还返回了若干与本问题无关的图像篡改检测条目，已排除。下述人类证据均重新核对原论文或权威索引；不把 Zotero 低相关命中包装成支持证据。

### 证据等级

- **A：直接行为因果证据**。操纵观察输入的时间/形式/可见信息，并测观察后的学习、预测或执行；最接近机器人表征假设。
- **B：直接观察相关证据**。观察时的行为分段、眼动、TMS/MEP、fMRI 或单神经元结果；能说明敏感性或表征，但不能自动推出学习必要性。
- **C：相邻任务证据**。动作执行、动作想象、婴儿发展、体育预测或静态对象 affordance；只能约束机制的可行性。
- **D：机器人/计算前例**。证明表示可实现或已有相近主张，不是人类认知证据。

三个推理禁区：脑区/MEP 活动不等于该信息对识别或学习必要；注视落点不等于内部只编码落点；动作执行的协同结构不等于观察视频时也采用同一离散码。

## 分阶段判定

| 候选阶段 | 当前证据强度 | 支持内容 | 不能推出什么 |
|---|---:|---|---|
| pre-contact / 接近 | **中等**（B/C） | 观察抓取时 MEP 随可见手指张开度和运动阶段变化；观察者在手到达前前瞻注视目标；初始握姿可提示动作终态。[Gangitano et al., 2001](https://doi.org/10.1097/00001756-200105250-00038) [Flanagan & Johansson, 2003](https://doi.org/10.1038/nature01861) [Herbort et al., 2012](https://doi.org/10.1016/j.actpsy.2012.01.001) | 不证明必须保存一张固定 “pre-contact keyframe”，也不证明前瞻注视依赖精确手姿态而非目标/场景概率。 |
| contact / 接触建立 | **分段证据强；观察后工具学习证据弱**（B） | 主观事件边界与 touching 高度对齐；动态手部抓取视频比单张接触图更能产生抓型特异运动共振；手—物交互促进婴儿目标预测。[Pomp et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118534) [Bunday et al., 2016](https://doi.org/10.1016/j.cortex.2016.09.002) [Falck-Ytter et al., 2006](https://doi.org/10.1038/nn1729) | 接触不是充分边界线索；不能以接触帧取代接近运动、力学或接触后的对象运动。 |
| regrasp / 重抓 | **弱**（主要是 C） | end-state comfort 和执行研究表明，人会为未来舒适终态选择初始握姿；连续到达—抓取中手形逐渐适配对象。[Santello & Soechting, 1998](https://doi.org/10.1152/jn.1998.79.3.1307) | 未发现人类观察学习实验直接证明重抓是独立、普适的知觉阶段，或其帧权重大于相邻连续运动。把 regrasp 固定为五阶段之一目前是任务工程选择。 |
| effect / 对象效果 | **中等到较强的目标/联结证据**（A/B） | 观察按键—声音可形成动作—效果联结；额顶叶表征动作结果可对部分运动学变化保持不变；工具训练中概念可预测性和运动经验对 EEG beta 有可分离影响。[Paulus et al., 2011](https://doi.org/10.3758/s13423-011-0136-3) [Hamilton & Grafton, 2008](https://doi.org/10.1093/cercor/bhm150) [Gerson et al., 2017](https://doi.org/10.1038/srep46761) | 这些主要支持结果/语义，不支持为达到结果所需的具体手姿态，更不证明单张 effect frame 足够。 |
| release / 接触解除 | **中等**（B） | untouching 与主观分段相关，并比 touching 伴随更广的额叶、海马旁回和岛叶活动，符合事件模型更新解释。[Pomp et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118534) | 尚无直接证据显示 release 帧对观察后工具执行具有独立因果贡献；“更新”是与脑区模式一致的解释，不是唯一机制。 |

因此，五阶段中 **contact 与 release/untouching 最有直接事件分段依据，pre-contact 有连续运动和预测依据，effect 有目标/结果依据，而 regrasp 的独立观察学习证据最弱**。这不是一个经人类实验验证的固定状态机。

## 各证据家族分别支持了什么

### 1. 事件分段：最直接支持接触变化，但同时否定“接触即边界”

Pomp 等让 31 名参与者观看 48 种物体操作；研究结合 fMRI、重复 unit-marking 和从视频提取 touching/untouching 的计算模型。约 80% 的主观 unit marks 落在 touching–untouching motif 附近，touching 峰值约领先报告 260 ms，untouching 峰值约领先 90 ms。该结果强力支持“接触建立—解除是自然操作的候选边界锚点”。然而，约 69% 的 touching–untouching motifs 没有被标成边界，说明视觉接触变化还要受目标、事件粒度、对象角色和预测误差调制。[Pomp et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118534)

更近的自然视频实验也发现，未训练观察者能可靠识别 action change 与 goal change，而且两者在控制低级视觉变化后仍与事件边界相关；这些关系不能被对象接触的开始/结束完全解释。[Su & Swallow, 2024](https://doi.org/10.3758/s13421-024-01525-8) 婴儿在有意图动作边界处对人为停顿较不惊讶、在动作中间停顿处更惊讶，也支持按意图组织动态输入，但不支持具体姿态码。[Baldwin et al., 2001](https://doi.org/10.1111/1467-8624.00310)

**判定：** 强支持 contact/release 是候选事件锚点；中等支持层级事件模型；不支持接触变化充分，更不支持固定阶段数。

### 2. 动作观察网络 / mirror system：同时编码运动与目标，不能作为离散姿态必要性的证明

观察手动作会调制运动系统。TMS 在抓取视频不同时间点测量 MEP，显示与观察到的手指张开度同步的阶段特异变化，说明连续可见运动学进入运动系统。[Gangitano et al., 2001](https://doi.org/10.1097/00001756-200105250-00038) Bunday 等用 `视频/单张接触帧 × 仅手/整个人 × 精细/全手抓取` 范式发现，抓型—肌肉特异反应出现在仅手的动态视频，而非静态接触帧；显示整个人时这种特异性又消失。这既是“动态 > 单帧”的直接神经代理证据，也表明效应高度依赖显示条件，不能泛化为任何视频都逐帧复现动作。[Bunday et al., 2016](https://doi.org/10.1016/j.cortex.2016.09.002)

猕猴 F5 单神经元给出重要的双向约束。观察手、反向钳和棍刺达成抓取目标时，许多手抓取 mirror neurons 都响应，尽管手部运动学相反或效应器不同；手动作一般更早、更强，工具动作较弱、较晚。这表明目标等价性可以跨越具体手运动学，而运动相似性仍调节响应。[Cattaneo et al., 2010](https://doi.org/10.1007/s00221-010-2329-9) 因而“mirror neuron = 精确手姿态模板”不成立。

必要性证据也不支持强运动模拟论。先天无手者仍可形成典型、效应器无关的动作系统，并能识别通常以手完成的动作；这说明视觉/概念动作表征不以拥有同构手部运动程序为前提。[Vannuscorps et al., 2020](https://doi.org/10.1073/pnas.2017789117)

**判定：** 动作观察网络同时含连续运动学、抓取类别和较高层目标；它支持混合层级表示，而不是 SM-FET 的五阶段离散必要性。

### 3. 目标层级与模仿：最明确支持“保目标、丢动作细节”

fMRI repetition-suppression 系列把动作拆成轨迹、抓取对象/目标和结果。后部视觉/STS 区域更敏感于较低层运动变化，aIPS 对抓取目标更稳定，额顶叶区域可表征动作结果而跨越运动学变化。[Hamilton & Grafton, 2006](https://doi.org/10.1523/JNEUROSCI.4551-05.2006) [Hamilton & Grafton, 2008](https://doi.org/10.1093/cercor/bhm150) 这是层级表征的证据，不是“高层替代低层”的证据。

儿童模仿尤其清楚地区分目标与手段：示范以特定手触碰耳朵/目标时，儿童经常实现正确目标但不复制同侧手选择，即目标比具体运动受到更强保护。[Bekkering et al., 2000](https://doi.org/10.1080/713755872) 这类结果支持 task/goal semantic，却反对“观察者必然保留示范者逐阶段具体手姿态”。

**判定：** 强支持 goal/outcome 层级，明确不支持 exact hand pose 必需；不能据此认为低层轨迹对精细执行无用。

### 4. 前瞻注视与接触预测：注意目标落点，不等于只编码关键帧

成年人观看积木操作时，眼睛在手到达前先移向即将接触的目标，模式类似亲自操作时的前瞻注视。[Flanagan & Johansson, 2003](https://doi.org/10.1038/nature01861) 12 个月而非 6 个月婴儿会前瞻注视动作目标，而且效应依赖手—物交互。[Falck-Ytter et al., 2006](https://doi.org/10.1038/nn1729) 婴儿的前瞻注视还与自身抓取能力相关。[Ambrosini et al., 2013](https://doi.org/10.1371/journal.pone.0067916)

这些结果支持对目标/未来接触位置的稀疏注意，也可能支持事件时刻前的 anticipatory window。可是注视是采样策略，不是完整内部状态的读出；前瞻注视也可出现在可预测的非生物运动中。[Craighero & Mele, 2021](https://doi.org/10.1016/j.cognition.2020.104461) 因此不能从“眼睛跳到目标”推出“手的中间轨迹未被利用”。

### 5. 工具 affordance、经验与手部协同：压缩是可行的，但观察侧外推很弱

在执行/想象对熟悉对象的抓取时，15 个关节角的前两个主成分解释超过 80% 的手形方差；较高主成分保留对象特异信息。[Santello et al., 1998](https://doi.org/10.1523/JNEUROSCI.18-23-10105.1998) 这说明手姿态存在低维协同，却也提醒低方差维度可能对区分对象和精细接触很重要。该实验不是视频观察学习，不能直接证明观察者使用相同 synergy token。

一周筷子或扳手训练的 EEG 研究把 motor experience 与关于异常工具—目标配对的 conceptual knowledge 分开：运动经验增加动作观察时的运动激活，概念可预测性降低激活，两者相互独立。刺激按 still、reach、pick-up、move-to、act 分段，但实验没有比较各段是否必要，也没有证明这五段构成统一编码。[Gerson et al., 2017](https://doi.org/10.1038/srep46761)

工具使用视频中的婴儿注视从计划期的脸转向工具作用期的对象变化，表明注意会随 goal establishment、tool transformation、goal resolution 改变。[Libertus et al., 2016](https://doi.org/10.1016/j.jecp.2016.07.004) 这更支持“不同阶段读取不同信息源”，而非“全程只读取手”。

**判定：** affordance/协同支持低维先验的可行性；对观察侧阶段化编码只有间接支持，尤其不能为 regrasp 独立阶段背书。

### 6. End-state comfort：pre-contact 可预示未来，但观察收益未必来自运动模拟

观察者可从演员尚未完成的抓握预测拨盘转动方向与幅度；随视频片段变长，预测改善，且符合 end-state comfort 的动作更易预测。作者将改善更多解释为知觉学习，而非必须依赖运动模拟。[Herbort et al., 2012](https://doi.org/10.1016/j.actpsy.2012.01.001) 这支持早期握姿蕴含未来效果，也支持增量轨迹带来额外信息，而不是单一 pre-contact 帧天然充分。

反面结果是，观察 end-state-comfort-favorable 动作并未稳定调制后续动作计划回忆。[Seegelke et al., 2015](https://doi.org/10.3389/fpsyg.2015.00045) 因此 end-state comfort 可作为机器人阶段消融的动机，不能当作普适学习机制。

### 7. 观察学习与 temporal occlusion：动态/误差信息有用，但严格时间顺序未总是必要

观察他人在力场中的误差和适应，可以使观察者随后在同方向力场中获益、在相反方向受损，显示动态误差轨迹能建立任务相关内部模型。[Mattar & Gribble, 2005](https://doi.org/10.1016/j.neuron.2005.02.009) 不过后续范式发现，高误差示范更有教学作用，而打乱示范试次顺序未显著消除学习；这反对把全局逐帧/逐试次顺序一概视为必要。[Brown et al., 2010](https://doi.org/10.1152/jn.01047.2009)

在新颖上肢序列计时任务中，自然轨迹和恒速轨迹都产生观察学习，仅显示终态则没有；说明运动轨迹优于 endpoint-only，却不要求精确生物运动学。[Roberts et al., 2015](https://doi.org/10.1016/j.actpsy.2015.05.011) 体育 temporal-occlusion 研究显示专家可在投球/击球结果出现前利用早期运动学，背侧前运动皮层 TMS 也可干扰短暂遮挡期间的动作预测；其范式不是工具学习，最多说明预测可能依赖事件前窗口，而非单帧。[Aglioti et al., 2008](https://doi.org/10.1038/nn.2182) [Springer et al., 2012](https://doi.org/10.3389/fnhum.2012.00020)

### 8. 教学动画与静态图：目前最接近“多关键相位 vs 全轨迹”的行为证据

结绳和解环实验比较连续手部动画/视频与内容相当的静态图序列，动态条件的步骤正确率、速度和迁移通常更好。[Ayres et al., 2009](https://doi.org/10.1016/j.chb.2008.12.013) 后续结绳实验也发现动画优于静态图，但动画是否显示手的影响并不一致。[Marcus et al., 2013](https://doi.org/10.1016/j.chb.2013.04.035)

更严格的 `动画/静态 × 显示手/不显示手` 实验中，动画提高运动与认知测验表现、降低负荷，但显示手反而损害部分认知表现。[de Koning et al., 2019](https://doi.org/10.1016/j.compedu.2019.103636) Lego 装配也报告动画有小幅优势，而只呈现任务相关对象变化、不显示手可能更好。[Castro-Alonso et al., 2015](https://doi.org/10.1016/j.compedu.2014.12.022)

这些研究证明“连续对象/关系变化”可胜过静态序列，却同时反驳“优势必然来自连续手姿态”。静态材料的帧选择、观看时长、瞬时信息密度和学习者控制常与动态格式共同变化，不能直接等同于容量匹配的机器人 token 消融。

## 有没有 single snapshot vs multiple key phases vs full trajectory 的直接实验？

**截至本轮检索，没有发现一个干净的三臂人类工具学习实验，同时比较：**

1. 单张、信息量匹配的 snapshot；
2. 预注册且预算固定的 `pre-contact/contact/regrasp/effect/release` 多关键阶段集；
3. 完整轨迹；

并以观察后的真实工具执行/跨对象迁移为结果，同时匹配总观看时间、重放次数、分辨率和信息容量。

最接近的证据只能拼成若干**两两比较**：

| 实验 | 比较 | 结果 | 与三臂问题的距离 |
|---|---|---|---|
| Bunday et al. 2016 | 动态抓取视频 vs 单张接触帧 | 仅手视频产生抓型特异 MEP，静态帧没有 | 神经代理，不是学习；无多关键阶段臂 |
| Roberts et al. 2015 | 自然/恒速轨迹 vs endpoint-only | 两种动态轨迹有学习，终态没有 | 非工具、无语义 key phase；精确生物运动不是必要 |
| Ayres et al. 2009 | 连续动画/视频 vs 多张静态步骤图 | 动态通常更好 | 静态帧未按 SM-FET 事件预算优化，呈现方式有混淆 |
| [Urgesi et al. 2010](https://doi.org/10.1093/cercor/bhp292) | 单独呈现动作开始/中段/终段静态姿态 | 未完成的开始/中段姿态比终态诱发更强未来动作模拟 | 不是多帧序列，也不测学习；只说明 snapshot 的时间位置很重要 |

所以“人类更依赖多关键阶段而不是完整轨迹”目前是**未直接检验**的命题。文献只允许更弱的组合判断：单一终态通常不够；多张未优化静态图有时不如动画；接触/脱离/效果是显著的事件候选；局部和阶段间运动在一些任务仍有不可替代信息。

## 机器人与计算前例：可借鉴，但不能反向充当人类证据

- [Forecasting Action Through Contact Representations](https://doi.org/10.1109/TPAMI.2021.3055233) 已用第一视角视频中的 hand-object contact、next active object 和状态图预测未来动作。它证明 contact-centric 表示有计算效用，也意味着“首次以接触构造第一视角动作表征”不可作为 novelty。
- enriched Semantic Event Chains 仅凭 touching/untouching 和对象间静态/动态关系即可让人/模型在动作结束前识别许多物体操作；这支持对象关系事件对识别的充分性，但没有测精细机器人执行。[Ziaeetabar et al., 2021](https://doi.org/10.1371/journal.pone.0243829)
- [ORION](https://doi.org/10.1007/s10514-026-10253-8) 从单个人类视频提取对象图、接触/速度关键帧，同时保留关键帧间运动轨迹用于机器人复现。这个混合设计恰好说明“事件”与“轨迹”不应被设成伪二选一。
- 当前机器人侧已有从人类视频估计 3D hand/object trajectory、contact/affordance、对象状态转移的多条路线。它们应作为 SM-FET 的强 baseline/相似性约束，而不是生物学佐证。

## 可证伪的机器人假设与 benchmark/消融

### H1：事件邻域在固定 token 预算下优于均匀采样，但不预设它等价于全轨迹

**假设：** 对至少含三次接触状态变化的操作，在相同视觉 token 数、相同参数/FLOPs 下，以预测/真值 contact、untouch 和 object-effect 为中心的短窗口，比均匀帧、随机帧和纯光流峰值采样有更高的跨对象执行成功率；稠密全轨迹保留为上界。

**Benchmark：** 接触丰富的 pick–insert–release、夹持后旋转、盖合、倾倒后回正、工具碰触后脱离；按对象几何、操作者、速度和遮挡 group-OOD 划分。

**必要消融：** event windows / uniform / random / motion peaks / single contact frame / full dense；使用 oracle event 和预测 event 两套结果。

**指标与否证：** rollout success、correct-first-contact、contact F1、下一事件 NLL、边界时间 ECE、安全碰撞/滑落。事件窗口对容量匹配 uniform 的置信区间无优势，或与 dense 的差距在精细任务持续很大，即否证“事件锚点足以压缩”的版本。

### H2：contact/release/effect 的因果价值应高于相同长度的阶段内片段

**假设：** 删除 contact、untouch 或 effect 邻域造成的性能下降，应大于删除同长度、同运动能量的阶段内部片段；如果五阶段顺序承载因果结构，打乱或时间反转也应有额外损害。

**Benchmark：** 明确标注 `approach/contact/manipulate/regrasp/effect/release` 的可重演任务，至少包含有/无 regrasp 配对和相同对象结果、不同接触序列配对。

**必要消融：** leave-one-phase-out、matched random deletion、运动量匹配删除、phase shuffle、time reversal、重复 effect 但打乱接触次序。

**指标与否证：** 各 phase success、动作时序/力峰误差、对象最终状态、因果顺序识别。如果 matched random deletion 等效，事件权重假设失败；如果 shuffle 不影响需要因果顺序的任务，固定阶段序列的主张失败。观察力场学习中 trial order 的 null 结果要求预先保留这一反证可能。

### H3：目标/效果语义负责“做什么”，局部连续运动负责“怎么安全做到”

**假设：** goal/effect-only 或对象关系事件表示可在简单 pick-place 上接近完整轨迹，并改善语义组合泛化；但在插入、穿线、避障切割、旋拧、动态重抓中，只有“事件语义 + 锚点邻域相对手/物轨迹”的 hybrid 能保持成功率与安全性。

**Benchmark：** 构造相同 goal/outcome、不同工具几何/握法/障碍，以及相同早期轨迹、不同目标/结果的正交任务族。

**必要消融：** goal-only、effect-only、contact graph-only、五张 pose-only、局部 trajectory-only、hybrid、full trajectory；所有中间表示容量匹配。

**指标与否证：** group-OOD success、插入力/横向力、滑移、路径碰撞、任务选择准确率。如果 event-only 在全部精细任务等同 hybrid，则局部连续运动假设被否证；如果 hybrid 不优于同容量无语义轨迹，则语义分解没有贡献。

### H4：低维手姿态变化只应在需要重抓/姿态匹配的子任务增益，而不是普适收益

**假设：** 相对于绝对 3D hand pose，在 pre-contact 和 regrasp 邻域使用低维相对手形/抓型变化，会提高跨手形和相机视角泛化；在不需姿态匹配的推拉任务中不应有稳定收益。

**Benchmark：** 同一对象结果由多种初始握法达成，且包含必须重抓与不需重抓配对；训练手/视角与测试 embodiment 分组隔离。

**必要消融：** absolute pose、PCA/synergy code、pose derivative、contact/effect-only、同 bit 数无语义 code；另做隐藏/显示手的输入干预。

**指标与否证：** 抓取稳定性、重抓成功、接触力、跨 embodiment success。若低维 pose 在容量匹配后无增益，或只在训练身份内增益，则执行侧 hand synergy 不能转化为观察侧模型依据。

### H5：阶段应是带不确定性的自适应窗口，而不是固定五张帧

**假设：** 在事件持续时间、遮挡和工具速度变化时，按 boundary uncertainty 自适应分配帧预算，比固定五张关键帧稳健；untouch/effect 不确定时应保留更宽上下文。

**Benchmark：** 同任务速度缩放、停顿插入、接触遮挡、透明对象、事件时刻抖动和未见工具；显式保存真实边界区间而非单点标签。

**必要消融：** fixed five frames、fixed-width windows、uncertainty-adaptive windows、uniform、full trajectory；oracle boundary 与预测 boundary 分开报告。

**指标与否证：** 事件时刻 calibration、anticipation latency、边界扰动下 success 曲线。如果 fixed-five 在所有 timing shift 下匹配 adaptive，动态预算假设失败；如果 event detector 误差吞噬所有收益，则系统层 claim 也失败，即使 oracle 有优势。

## 对 SM-FET claim 的硬限定

### 可以写

1. “人类事件分段研究提示 touching/untouching 与对象效果是物体操作的高信息候选锚点；动作观察和运动学习同时提示局部运动学不能被普遍删除。因此我们把 SM-FET 作为事件语义与局部轨迹的**待检验混合归纳偏置**。”
2. “认知证据决定了消融设计，而不是证明架构正确：必须比较单帧、多事件帧、事件窗口、均匀同预算和完整轨迹。”
3. “五个标签是可审计的工程本体；其中 regrasp 的人类观察证据最弱，阶段数和时间宽度应接受数据/干预否证。”

### 不可以写

1. “人类理解工具使用时忽略逐帧轨迹，只看 `pre-contact/contact/regrasp/effect/release`。”没有直接三臂实验支持，且动态视频、轨迹学习和连续运动共振构成反证。
2. “mirror system 证明大脑保存五个关键手姿态。”mirror/F5 同时表现出运动学敏感性和目标不变性，且动作理解可在无同构效应器经验者中保留。
3. “接触变化足以定义事件。”Pomp 数据中多数 touching–untouching motifs 未形成主观边界；goal/action changes 也不能由接触开始/结束完全解释。
4. “手部协同证明可以用少量离散 pose 替代视频。”协同证据主要来自执行/想象；低方差维度还可能携带对象和精细接触信息。
5. “观察研究证明该表征能提高机器人控制。”人类分段、眼动、MEP/fMRI、识别与机器人 rollout success 是不同终点，必须分别验证。

## 最强反证与审稿风险

- **动态而非静态的优势：** 抓型特异 MEP、轨迹观察学习、手工教学均显示单一终态或普通静态序列可能丢失有用信息。
- **接触不充分：** 69% 左右的 touching–untouching motifs 未成为主观边界；目标/动作变化有独立贡献。
- **语义可能压过手姿态：** 儿童保留目标但不复制具体手；F5 对不同工具/相反运动学实现同一抓取目标可泛化。
- **手并非动态教学收益的唯一来源：** 动画不显示手仍可优于静态，甚至在部分认知测验更好；对象变化和时空连续性可能是主因。
- **运动模拟非必要：** 先天无手者仍有典型动作识别/效应器无关表示。
- **领域迁移风险：** 婴儿眼动、体育遮挡、按钮—音调联结、猕猴神经元和静态手形协同都不是复杂工具观察后机器人执行。

如果论文只给 SM-FET 对 full video baseline 的平均成功率，而没有容量匹配的 uniform/random/event-window、goal/effect-only、hybrid 和 phase intervention，认知故事不会补足因果识别。真正有价值的贡献应是：预注册哪类任务需要哪些事件与局部轨迹，然后允许结果否证固定五阶段。

## 最终判定

**Cognitive rationale：Conditional Go。Biological/mechanistic claim：No-Go。**

可以用人类证据解释为什么值得试验 contact/release/effect-centered sampling，并据此设计反事实消融；不能声称人类已经被证明使用 SM-FET，不能声称 exact hand pose 或固定 regrasp stage 是已知关键，也不能声称稠密轨迹冗余。最稳妥的模型原则是“事件锚定、目标与效果分层、局部运动保真、阶段不确定性显式化”，而不是“认知启发的五关键帧替代视频”。

## 选读原始文献索引

- Pomp, J. et al. (2021). Touching events predict human action segmentation in brain and behavior. *NeuroImage, 243*, 118534. https://doi.org/10.1016/j.neuroimage.2021.118534
- Su, X., & Swallow, K. M. (2024). People can reliably detect action changes and goal changes during naturalistic perception. *Memory & Cognition*. https://doi.org/10.3758/s13421-024-01525-8
- Gangitano, M. et al. (2001). Phase-specific modulation of cortical motor output during movement observation. *NeuroReport, 12*, 1489–1492. https://doi.org/10.1097/00001756-200105250-00038
- Bunday, K. L. et al. (2016). Grasp-specific motor resonance is influenced by the visibility of the observed actor. *Cortex, 84*, 43–54. https://doi.org/10.1016/j.cortex.2016.09.002
- Flanagan, J. R., & Johansson, R. S. (2003). Action plans used in action observation. *Nature, 424*, 769–771. https://doi.org/10.1038/nature01861
- Hamilton, A. F. de C., & Grafton, S. T. (2006). Goal representation in human anterior intraparietal sulcus. *Journal of Neuroscience, 26*, 1133–1137. https://doi.org/10.1523/JNEUROSCI.4551-05.2006
- Hamilton, A. F. de C., & Grafton, S. T. (2008). Action outcomes are represented in human inferior frontoparietal cortex. *Cerebral Cortex, 18*, 1160–1168. https://doi.org/10.1093/cercor/bhm150
- Cattaneo, L. et al. (2010). Responses of mirror neurons in area F5 to hand and tool grasping observation. *Experimental Brain Research, 204*, 605–616. https://doi.org/10.1007/s00221-010-2329-9
- Santello, M. et al. (1998). Postural hand synergies for tool use. *Journal of Neuroscience, 18*, 10105–10115. https://doi.org/10.1523/JNEUROSCI.18-23-10105.1998
- Mattar, A. A. G., & Gribble, P. L. (2005). Motor learning by observing. *Neuron, 46*, 153–160. https://doi.org/10.1016/j.neuron.2005.02.009
- Roberts, J. W. et al. (2015). Motion trajectory information and agency influence motor learning during observational practice. *Acta Psychologica, 159*, 76–84. https://doi.org/10.1016/j.actpsy.2015.05.011
- Ayres, P. et al. (2009). Learning hand manipulative tasks: When instructional animations are superior to equivalent static representations. *Computers in Human Behavior, 25*, 348–353. https://doi.org/10.1016/j.chb.2008.12.013
