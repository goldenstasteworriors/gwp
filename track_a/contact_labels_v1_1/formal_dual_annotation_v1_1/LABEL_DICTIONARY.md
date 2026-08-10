# Contact segment v1.1 标注词典

## 标注对象与时间

标注“任务相关手/末端执行器—物体或物体—物体的真实物理接触持续区间”，不是动作阶段、
物体靠近、轮廓重叠、抓取意图、液流事件或手状态变化。每个区间使用 processed 20 Hz 时钟
的半开区间 `[start_frame,end_frame)`：`start_frame` 是第一帧确认接触，`end_frame` 是第一帧
确认已不接触。证据索引提供 processed frame、原视频 frame、两者 timestamp 与 sync error。

## Coverage

- `exhaustive`：完整看过整个 coverage，当前证据足以同时声明“列出的 segment 是全部正例”
  和“其余帧没有接触”。透明、遮挡或画外区域可能藏有接触时不能使用。
- `positive_only`：只对明确写出的 segment 声明正例；未写出的帧和空 query 都是 unknown，
  不是负例。
- `unknown`：该 coverage 无法形成可用复核。不得写 segment，也不得推断正/负例。

无接触 episode 只需一行 `exhaustive` coverage 和零行 segment；禁止虚构 negative segment。

## 边界、unknown 与 censor

- `start_boundary=observed`：看到真实接触发生边界；必须新建 `onset_event_id`。
- `start_boundary=left_censored`：接触在 coverage 开始前已存在；`start_frame` 必须等于 coverage
  start，`onset_event_id` 留空。
- `start_boundary=unknown`：能确认一段保守正区间，但无法知道接触是否恰在/早于其起点发生；
  `onset_event_id` 留空。这不是 left-censored 的同义词。
- `end_boundary=observed`：看到真实分离边界；必须新建 `release_event_id`。
- `end_boundary=right_censored`：coverage 结束时接触仍持续；`end_frame` 必须等于 coverage end，
  `release_event_id` 留空。
- `end_boundary=unknown`：能确认一段保守正区间，但真实分离边界不清楚；event ID 留空。这不是
  right-censored 的同义词。

类别不知道时用空字符串，只屏蔽该类别；不应清除已知 presence 或其他类别。不能写字面值
`unknown`、`none`、`N/A` 代替空字段。censor 只表示真实边界在复核区间外，不表示画面模糊。

## 类别

`side`：

- `left` / `right`：发生接触的左/右手或对应末端执行器。
- `object_object`：两个任务物体直接接触，当前 segment 不归属于某只手。
- 空：侧别看不清或不能可靠归属。

`site`（接触执行方的表面）：

- `opposition`：与主要抓持面相对、形成对合的手指/夹爪面。
- `primary`：承担主要施力或主要抓持的手指/夹爪面。
- `support`：稳定物体但不是主要施力面的辅助手指/支撑面。
- `palm_or_jaw`：只能可靠辨认掌面或夹爪 jaw 的宽接触，无法细分前三类。
- `tool_tip`：由工具末端而非手/夹爪本体接触。
- `other_surface`：以上均不适用、但接触表面可见且可说明。
- 空：证据不足。禁止凭抓取习惯猜测。

`object_role`（被接触的任务物体在该时刻的角色）：

- `manipulated`：当前被直接拿持、移动或转动的物体。
- `tool`：以其功能端作用于其他物体的中介物体。
- `receptacle`：承担接收/盛放作用的容器。
- `fixture`：固定环境或固定支撑。
- 空：角色在证据中不明确；不要仅根据 episode 名推断。

`contact_mode`：

- `stick`：接触面相对静止或近似无切向滑移。
- `slide`：保持接触且有清楚的切向相对位移。
- `roll`：保持接触且由滚动主导，接触点沿表面迁移。
- `impact`：短时碰撞/敲击，接触由明显冲击发生。
- 空：看不清、模式混合或证据不足。接近/分离本身不是一种 mode。

## ID 与证据

- A 使用只属于 A 的 ID 前缀，B 使用只属于 B 的前缀；不要交换或照抄 ID。
- `segment_id` 指一个全局物理持续区间；同一 segment 在多个窗口出现仍是同一个 ID。
- observed onset/release 各有独立 event ID；unknown/censored 边界不能伪造 event ID。
- `rgb_direct` 只用于当前帧直接可见；依赖前后变化则用 `rgb_temporal`。
- `confidence` 是对已提交这行人工判断的置信度 `(0,1]`，不能用 0 表示 unknown；unknown 应按
  上述字段规则编码。
- 独立轮 `source_proposal_id`、`parent_segment_ids` 必须为空。proposal/prescreen 不属于证据。

## 数据使用限制

- 隔离视频可能包含可识别的人体、声音或环境，只能用于本次获批的 contact 标注任务。
- 只使用 coordinator 提供的 packet 与 `media/` 相对路径；不得尝试访问 source 根目录、HDF5、
  另一标注员目录、proposal、prescreen 或 GT。
- 不得转发、上传个人云盘、截图、录屏或二次使用媒体。访问到期后删除本地工作副本并向
  coordinator 确认；发现错误分发或越权访问时立即停止标注并报告。
