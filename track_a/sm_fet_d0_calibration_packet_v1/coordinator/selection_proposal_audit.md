# D0 calibration episode selection proposal（coordinator only）

- 日期：2026-08-12
- 状态：`COORDINATOR_ONLY / PROPOSAL_ONLY / NON_GT`
- recipient allowlist：明确排除本文件
- formal blind12：`0, 9, 12, 34, 36, 38, 43, 50, 51, 53, 54, 58`
- calibration：`46, 65, 1`
- 交集：空

| task | episode | 只用于选择的 proposal 理由 |
|---|---:|---|
| `sm-fet-d0-cal-00` | 46 | 单手、时序较短、彩色内容对比度较高，作为普通可见/较清楚边界候选；未见明显 regrasp 仅是采样审计印象，须由 A/B 自行判断 |
| `sm-fet-d0-cal-01` | 65 | 原始 RGB 采样中可见两只手同时进入操作区，一手操作、一手靠近或稳定透明容器，手—杯—rim 互遮挡，覆盖双手 bundle、hand roster 与 occluded/missing 字段 |
| `sm-fet-d0-cal-02` | 1 | 透明杯与低对比度或近无色液体使容器边缘、stream、meniscus、effect/outcome 的可见证据弱；与 46/65 的紫色高对比内容形成 visibility 差异 |

## Selection bias

- 这是 purposive calibration sampling，不是随机样本；主动放大清楚边界、双手遮挡/roster 和透明液体可见性差异。
- 三段均来自同一 human-pour 数据源，不能估计自然 prevalence、OOD、跨任务、跨人、跨 session、跨容器或跨液体泛化。
- 当前 66 段缺 authoritative subject/session/operator/object/liquid/batch provenance；episode ID 或颜色不能替代 group 字段。
- coordinator 已看过 selection-level cues，不能充当未来 D1 blind A/B。
- calibration 不进入 blind agreement、validation/test、阈值调参或 GT；协议重开时必须使用新的 disjoint calibration。

选择只依据 coordinator 对原始 RGB 的只读时间采样、MP4 技术 metadata、formal blind episode
清单与现有 proposal-level 规范审计。没有读取或使用受禁的 sensor/proxy/visualization、模型输出、
旧 GT 或人工标签。选择理由不得预填进任何 annotation CSV，也不得进入 recipient packet。
