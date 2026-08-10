# Zotero-first 证据与协议影响

## Idea-discussion 运行记录

- `idea_id`: `30e9176d-f97d-49a7-b0dc-10f98e8a57da`
- `discussion_id`: `64931002-fc27-4945-8070-64bf8cbdf78a`
- Zotero 同步条数：100；idea card 已更新。
- 运行时返回的主要 Zotero IDs：`M2RF5GE4`（RoboBrain 2.5）、`CBLGABU2`（FAST）、`HXIANH32`（RoboBrain 2.0）。这些材料涉及时序理解、动作 token 或具身规划，但没有给出 exhaustive/positive-only/censoring 的 contact 标注语义。因此它们只能证明“时序表征与控制相关”，不能作为本 CSV 的直接标注依据。

本地 Zotero/Obsidian 中更直接的条目是 `EgoLoc`：`/home/ykj/Documents/ObsidianVault-dailypaper-skills/PaperNotes/ego/ego 手/EgoLoc.md`。它把接触/分离定位为逐帧 temporal interaction localization，并报告 onset/separation MAE、MoF、IoU；同时用 RGB-D/手速/VLM 产生候选。协议影响：边界误差必须独立报告；自动候选只能作 proposal，不能直接成为 GT。

## 原始资料及实际约束

- EgoLoc，https://arxiv.org/abs/2506.03662 ：contact/separation 是独立时间边界，而非动作段边界。约束：使用独立 event ID，报告 boundary MAE 和 segment tIoU。
- Ego4D FHO 官方标注指南，https://ego4d-data.org/docs/data/annotation-guidelines/ ：当物体已在手中时可以没有 CONTACT frame，且无交互/对象不可见的 clip 会被拒绝。约束：前者应记 left-censored 而不是伪造 onset；被拒绝或不可见不能当 exhaustive negative。
- Ego4D Hands & Objects，https://ego4d-data.org/docs/benchmarks/hands-and-objects/ ：contact frame、pre/post 与状态变化是不同标注实体。约束：contact event ID 与 physical segment ID 分离。
- VISOR，https://arxiv.org/abs/2209.13064 ：区分人工稀疏 mask 与插值得到的 dense mask，并显式建 hand-object relation。约束：自动/插值结果保留 proposal 来源，不能直接标 `human_adjudicated`。
- Ego contact representation，https://arxiv.org/abs/2102.00649 ：以 contact making/breaking 组织 egocentric action，并使用 time-to-contact。约束：跨窗口保留同一物理事件身份，不能将每个 TTC/window 样本计为独立事件。
- 双人视频标注 best-practice，https://pmc.ncbi.nlm.nih.gov/articles/PMC12799097/ ：两名训练标注者独立标注、第三人解决分歧是 gold-standard 流程。约束：模板分 A/B/adjudicated 三套，builder 只接受 adjudicated。
- Event agreement 方法，https://pmc.ncbi.nlm.nih.gov/articles/PMC4439396/ ：事件一致性应采用一对一重叠匹配，不能让同一个事件被重复匹配。约束：使用一对一 tIoU matching 与 event F1，不只算逐帧 accuracy。

以上证据不证明当前 Ego 倒液数据已经具备真值。episode 0 的 JSON 只有一个 contact/separation pair，缺少 annotator、coverage mode、可见性、side/site/role/mode、删失状态与独立复核，因此只能进入 `existing_timing_hint`，不能进入 adjudicated CSV。
