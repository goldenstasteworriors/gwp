# 项目数据监督约束

- SONIC 关节策略训练集的输出监督必须来自未来实测 `observation.state`，不得使用 `action.wbc`。
- 对时间步 `t`，数据集行内的 `action` 必须等于同一 episode 的 `observation.state[t + 1]`；episode 最后一帧使用自身状态，禁止跨 episode 取值。
- 左臂相对动作由训练变换使用未来实测关节角减去当前实测关节角得到；左手灵巧手保持未来实测关节角的绝对值。两者的原始监督源都必须是 `observation.state`。
- 新建或转换 SONIC 关节数据集时，元数据必须记录监督源、前移帧数和末帧策略；训练前必须数值验证 `action[t] == observation.state[t + 1]`，并确认它不是从 `action.wbc` 复制得到。
