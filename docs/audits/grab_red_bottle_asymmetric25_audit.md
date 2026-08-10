# `grab_red_bottle` 对 asymmetric25 的只读审计

结论：**No-Go，当前 88 集只能审计，默认禁止转换或冒充 RH56DFTP +
OmniPicker 的 25D 数据。** 数据实际是 G1 本体 29 维加左右各 6 维 Inspire
通道；右侧不是单通道 OmniPicker aperture。

## 仓库实测事实

| 项目 | 结果 | 证据 |
|---|---:|---|
| episode / parquet / video | 88 / 88 / 88 | 数据目录及 `meta/episodes.jsonl` |
| 总帧数 | 40,901 | 88 个 parquet 行数求和 |
| 频率 | `meta` 声明 50 Hz | `meta/info.json` |
| 时间戳步长 | min 0.01999855 s, median 0.01999998 s, max 0.02000046 s | 全量 parquet `timestamp` 差分；无非递增 |
| `observation.state` / `action.wbc` | 41 / 41 | 29 body + left Inspire6 + right Inspire6 |
| `observation.eef_state` | 14 | 左右各 xyz + quaternion；meta 未声明父坐标系 |
| 右手直接列名 | little, ring, middle, index, thumb_bend, thumb_rotate | `meta/info.json` |
| 右手直接观测范围 | 四指约 0.198–1.0；thumb_bend 约 0.997–1.0；thumb_rotate 恒 0 | 全量 state/action/teleop 列审计 |

SONIC 的 feature 定义和 DDS 适配器把该 41D 布局明确定义为
`body29 + left Inspire6 + right Inspire6`。适配器注释里的 `0=closed,
1=open` 是**仓库归一化接口语义**，不能当成 RH56 官方 raw 值。RH56 官方
手册的 `POS_SET/POS_ACT` 是六路 `0..2000`，且 raw 方向为 0 对应最小行程/
手指张开、2000 对应最大行程/弯曲；两者必须经独立校准映射。

此外，数据导出器在未传入 timestamp 时会用 `frame_index / fps` 生成时间戳。
因此这里观察到的规则 50 Hz 只能证明数据表时基连续，不能证明硬件时钟、
状态/命令采样偏差、相机同步或执行器延迟。

对应代码证据：

- `gear_sonic/data/features_sonic_vla.py:373-407` 定义 41D 及六路手通道顺序；
- `decoupled_wbc/control/envs/g1/utils/command_sender.py:150-190` 定义 DDS
  顺序、仓库归一化极性、`rt/inspire/cmd`、左右各六路及 `0..1` clip；
- `decoupled_wbc/control/envs/g1/utils/state_processor.py:148-168` 从
  `rt/inspire/state` 读取 6DoF，并按右 0、左 6 偏移；
- `gear_sonic/data/exporter.py:260-277` 说明缺省 timestamp 可由
  `frame_index / fps` 合成；
- Ego 极性证据为 `/home/ykj/project/EgoHumanoid/scripts/deploy.py:432-435,
  504-528` 以及 `data_collection/robot_data/teleop/run_g1_data_exporter.py:43`。

## 官方已确认（不代表当前仓库已接入）

- RH56DFTP：12 关节、6 主动自由度；六路 `POS_SET/POS_ACT` 顺序为小指、
  无名指、中指、食指、拇指弯曲、拇指旋转，raw 范围 `0..2000`。来源：
  [因时机器人 RH56DFTP 官方中文手册](https://www.inspire-robots.com/d/file/p/2025/01-16/%E5%9B%A0%E6%97%B6%E6%9C%BA%E5%99%A8%E4%BA%BA%E7%81%B5%E5%B7%A7%E6%89%8B--%E8%A7%A6%E8%A7%89%E6%89%8B%E7%94%A8%E6%88%B7%E6%89%8CV1.0.0cn.pdf)。
- OmniPicker：双夹爪、1 个主动自由度；AimDK `set_pos(x)` 接口定义
  `0.0=closed, 1.0=open`。这只是官方 X2/AimDK 接口定义，不证明当前 G1
  驱动或设备 raw 极性。来源：[智元 OmniPicker 官方手册](https://x2-aimdk.agibot.com/zh-cn/v0.8.1/_downloads/9179a383c97b0f37b1f0fbf51ed2cb35/omni_picker_introduction_for_x2.pdf)、
  [AimDK 官方末端执行器接口](https://x2-aimdk.agibot.com/en/latest/Interface/control_mod/endeffector.html)。

## 距离可训练/部署 25D 的缺口

必须重新采集目标实机的具名 measured state 与 absolute command：左 RH56 六路、
右 OmniPicker aperture 一路、双腕 pose、29 路 WBC、相机与硬件/同步时钟。
校准 capture 至少要记录并结构化复核：设备 serial/固件/驱动、state 与 command
各自单位及 closed/open/range、双腕 parent/child/FK frame、旋转布局、29 路顺序、
采样时序、state-action/camera skew、命令延迟、限幅、watchdog 和急停。

数据转换/训练最低要求 `bench_verified`；部署要求 `robot_verified`。两者还要分别
显式批准，并使 hardware manifest、capture、转换数据和 norm stats 的 SHA-256
一致。模板默认无真值且校验失败。

## 补采协议与 Go 门槛

1. 静态端点校准：每通道至少覆盖 open/closed/安全区间，state 与 command 分开。
2. 单轴小步进：验证方向、绝对语义、单步限幅、状态跟随与停止行为。
3. 双腕 frame/FK 对照：已知位姿和小旋转下复核 parent/child、四元数/rot6d 约定。
4. 同步/延迟：使用硬件或已同步时钟测 state-command-camera skew 和尾延迟。
5. bench 成功后采最小真机 episode；converter、norm、inference 三处 fingerprint
   完全一致，且 legacy 右手列被拒绝，方可训练 Go。
6. 只有 `robot_verified`、急停/watchdog 实测、限幅与 `[B,T,25]` 时间轴检查通过，
   才可部署 Go。

机器可读摘要见 `grab_red_bottle_asymmetric25_audit.json`。
