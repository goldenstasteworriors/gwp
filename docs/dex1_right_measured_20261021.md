# Dex1 右侧实测关节角三组实验

源数据：`dex1_three_camera_20261021_180822`，30 FPS，5 个 episode、6413 帧；episode0 为 1600 帧。

| 物理 GPU | 启动参数 | 数据 | 记忆 |
| --- | --- | --- | --- |
| 0 | `all` | episode 0–4 | 无 |
| 1 | `episode0` | episode 0 | 无 |
| 2 | `episode0_mem` | episode 0 | 4 秒右腕视觉历史 |

三组均使用 `observation.images.right_wrist`，只保留右臂 7 个关节和右手 Dex1 电机角。状态与动作监督均取自源 `observation.state`，单位 rad，输出为绝对关节角，不读取 `action.wbc`，不做末端位姿或角度差分。物理动作维度为 8，模型内部按原配置补齐到 32 维，其余维度不参与动作损失。

前两组继承 `giga_world_policy_0_5_sonic_g1_finetune` 的训练超参数：batch size 4、学习率 6e-5、CAME8Bit、BF16、EMA、50,000 步、48 帧动作窗口、每 5000 步保存。各数据集独立计算归一化统计，episode0 两组共享 episode0 统计。

服务器实测原配置在单卡 A800 80GB 上显存溢出，因此三组均开启模型激活重算与 `expandable_segments` 显存分配；以上训练超参数和损失不变。日志频率调整为每步，便于验证实际更新。

## 短时记忆适配

参考 [Cortex 附录 A.5.2](https://arxiv.org/html/2607.05377v1#A5.SS2) 的数秒短时记忆执行器，以及 [MEM III-C](https://arxiv.org/html/2603.03596v2#S3.SS3) 的历史视频压缩思路。本实验是对 GWP/WAN 的适配，不是对论文中 PaliGemma/ViT 结构的逐层复现。

历史偏移为 -120、-105、-90、-75、-60、-45、-30、-15 帧，即 4 秒、2 Hz；加上当前帧，共 9 帧。沿用 WAN 冻结的因果视频 VAE 编码成 3 个时间 latent，再与独立的当前帧参考 latent 拼接为干净的视觉前缀。MoT 将整个历史前缀作为参考条件，动作分支不能访问未来视频 token。历史 latent 不计入未来视频去噪损失。

历史和当前/未来图像共享一次时序一致的图像变换；episode 起点处只重复本 episode 的第一帧，不跨 episode 采样。当前状态输入、动作预测窗口和原训练目标保持不变。

## 路径与启动

服务器工作目录为 `/workspace/project/giga-world-policy`，模型资产位于同级 `giga-world-policy-assets`。由于 `/workspace` 对象文件系统不支持符号链接和正常 Git 解包，Conda 环境、Git 元数据和需要 POSIX 文件操作的运行缓存分别位于 `/nix/conda_envs/gigaworld-policy`、`/nix/giga-world-policy-git`、`/nix/gigaworld-policy-cache`。训练数据和输出仍位于工作目录下。

在项目根目录逐行运行，以下每行是一条独立命令：

```bash
bash scripts/run_dex1_right.sh all
bash scripts/run_dex1_right.sh episode0
bash scripts/run_dex1_right.sh episode0_mem
```

各命令是前台训练，三个进程需要独立终端或 tmux 窗口。添加 `--dry-run` 可仅解析配置。输出目录包含启动日期，目录中保存 `launch_config.json` 和 `console.log`。

## 已完成的本地验证

- 逐帧比较所有导出的状态和动作，均等于源数据的右侧 8 维实测角度。
- 用实际 LeRobot/PyAV 读取右腕视频，检查起点、中间和终点采样与边界填充。
- 改变未来像素不会改变记忆输入，动作损失只启用前 8 个物理维度。
- 用缩小的 MoT 验证历史条件的梯度、未来视频与动作的注意力隔离，以及生产 trainer 的损失对齐和反向传播。
- 三组启动配置解析通过，训练超参数一致；临时测试脚本已删除。

环境安装保留项目指定的 Diffusers 0.36.0 与 LeRobot 0.4.4。两者的 PyPI 依赖声明冲突，因此 LeRobot 采用 `--no-deps` 安装，并单独安装实际数据读取路径所需的依赖；服务器已验证相关类能够导入。

无图形界面的服务器使用同版本 `opencv-python-headless==4.11.0.86`，并补充数据模块需要的 `h5py`。完整实际依赖记录在服务器 `outputs/setup_dex1_20261021/pip-freeze.txt`。
