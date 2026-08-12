# SM-FET `minimal-r1d` codec fix2 最终独立复审

- 日期：2026-08-12
- 复审对象：`2743cbdaec18e03a4c42f7444806584488837611`
- fix2 基线：`aa13981e5959611ad39574dca381fdede48db343`
- 仓库 canonical 基线：`e0ebc6929ab25bd5793ef64939b4b831e15b12c2`
- 复审方式：目标 worktree 只读；独立 detached worktree；Python 3.13.9 标准库；CPU only
- 规范优先级：freeze addendum 高于主合同；最终只读 review 确定验收边界
- 严重度：**P0 = 0；P1 = 0**
- 最终裁决：**Go（CPU-conformant Freeze，仅限 frozen structural codec/checker gates）**
- G1 裁决：**No-Go；profile states/bits 仍为 UNDEFINED**

## 0. 一句话结论

fix2 忠实关闭了 fix1 独立复审剩余的两类 P1：wrist compatibility table现在是 resource-validation-context canonical artifact的一部分，经SHA-256绑定static profile receipt，公开 compatibility validator不再接受逐调用override；axis/roll量化删除8-ULP非零吸附带，严格按实际binary64 angle结果执行half-open internal-edge判定，decoded roll构造性exact tie只用零宽tuple equality闭合。

本轮独立CPU套件执行 **410,146** 项检查，完整通过 toy813、37,000 production global rank全链、2,000 rank mutation、全部24个production header边界、全部36个topology block首尾、wire/compact/topology/JSON/resource负例、全部UV/depth/extent/axis internal edges上下1–8 ULP、全512 axis上的全部31个roll internal edges上下1 ULP、代表轴roll edges上下1–8 ULP、vertical epsilon²等值及两侧8个输入ULP、512 axis与16,384 frame/roll round-trip和geometry golden hash。未发现新的P0/P1。

```text
written contract + normative addendum: Doc structural Freeze
fix2 frozen codec/checker gates:        CPU-conformant Freeze Go
G1 executable profile:                  No-Go（独立于CPU codec结论）
G1 profile states/bits:                 UNDEFINED
```

## 1. 变更边界与无漂移检查

fix2相对`aa13981`只修改：

```text
M docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md
M world_action_model/sm_fet_minimal_r1d_schema.py
```

变更量为135 insertions、35 deletions。没有修改model、trainer、训练/推理config、dataset、checkpoint、rollout、CUDA/GPU、driver或其它受保护路径。目标worktree在复审前后均clean；临时脚本只存在于detached worktree，测试后删除并移除整个隔离worktree。

fix2文件SHA-256：

```text
c4fa9bc3b6f674703efc1926f15116266bd2d19c76c4290533e250149c1b8224  world_action_model/sm_fet_minimal_r1d_schema.py
f69a428a191c126f804232c07dfc7c29b2d2552c0ad54132f16b3cff56007737  docs/research/sm_fet_minimal_r1d_codec_conformance_20260812.md
```

核心结构无漂移：

```text
schema_id         = sm_fet_hard_grammar_v2_minimal_r1d_struct_v1
wire bytes        = 32
compact bytes     = 13, unsigned little-endian
P                 = 131,072
L_nonverify       = 70,934,200,320
L_verify          = 2,214,723,584
N                 = 43,695,467,397,120
V                 = 194,895,675,392
B_nonverify       = 14,319,704,032,895,892,051,583,303,680
B_verify          = 151,937,297,146,405,128,765,440
|G_struct_r1d|    = 157,516,896,299,151,958,972,545,105,921
bit_length(G - 1) = 97
```

32-byte wire字段、topology顺序、global block interval、dense rank、13-byte container及profile-capacity公式均未改变。`compatible_wrist_target_pairs`加入的是auxiliary resource-context validation artifact；它闭合static compatibility的profile绑定，不进入或改变structural alphabet。真实G1 profile仍不存在，因此没有把该artifact伪装为G1 calibration/capability数据。

## 2. 独立CPU复测

### 2.1 隔离与规模

- detached worktree固定在`2743cbd`；目标worktree只读且前后clean；
- Python 3.13.9标准库，未安装依赖，未使用conda项目环境、GPU或CUDA；
- 临时测试脚本及`__pycache__`随隔离worktree删除；
- 单轮累计410,146项检查/断言，全部通过。

### 2.2 覆盖结果

| 项目 | 覆盖 | 结果 |
|---|---|---|
| toy topology/global | `n=3,v=2`全部813 symbolic states，无洞、无重、双向inverse | Pass |
| production wire/rank | exact 32B、NULL/PAD、13B LE、7个冻结vectors、24个header边界、36个topology block首尾 | Pass |
| random property | seed`20260812`；37,000 global ranks；packet/wire/compact/rank/equal-bit全链 | Pass |
| rank mutation | 2,000次13-byte单bit mutation；reserved、越界或精确mutated integer | Pass |
| noncanonical/negative | wrong length/header/PAD/global rank/reserved bits；fix1完整VERIFY/topology/field负例路径未改 | Pass |
| strict JSON | raw UTF-8 bytes、nested duplicate property、unknown key、numeric/bool enum、NaN/Inf | Pass |
| resource context | accept/reject pair、hash错配、receipt self-hash、直接dataclass字段伪造、canonical bytes伪造 | Pass |
| resource conflicts | exact selector、duplicate template、generic same-resource、aperture恒mutex、MONITOR READ_ONLY | Pass |
| UV edges | u/v全部63个internal edges，exact及上下1–8 ULP | Pass |
| depth/extent edges | depth全部31个、extent全部7个internal edges，exact及上下1–8 ULP | Pass |
| axis angle edges | azimuth全部31个、elevation全部15个internal edges，actual binary64结果及上下1–8 ULP | Pass |
| vertical epsilon | `rho2 == 0x1p-80`及正负z；横向分量上下1–8输入ULP，逐项重算rho2 | Pass |
| roll edges | 全512 axis × 31 edges上下1 ULP；6个代表axis全部edges上下1–8 ULP | Pass |
| decoded surfaces | 512 axis、16,384 frame/roll round-trip；4个RASTER vertex上下1–8 ULP | Pass |
| golden hash | axis/frame/path三个完整surface digest | Pass |
| schema/capacity/protected paths | constants逐项、diff path allowlist、G1 UNDEFINED/No-Go | Pass |

geometry hashes保持冻结值：

```text
axis  d9d9319054b954fceb759dd6c9c348f588813176b50466e54950691f0d14a5de
frame 0e0e05bad6118cd3b597f895a6ebf997842208f99a1f3580e834f1edd80073f9
path  191a9bdbdb35e030de8b915a441a1bdfce8d0bf691132d95c1e359c5c38eb41a
```

## 3. 剩余两类P1的关闭证据

### 3.1 Resource compatibility binding：关闭

实现位置：

- `2743cbd:world_action_model/sm_fet_minimal_r1d_schema.py:1712-1861`
- `2743cbd:world_action_model/sm_fet_minimal_r1d_schema.py:1864-1876`
- `2743cbd:world_action_model/sm_fet_minimal_r1d_schema.py:2019-2060`

`compatible_wrist_target_pairs`现在是`sm_fet_r1d_resource_validation_context_v1` exact-key JSON字段。validator要求：pair恰有两个WRIST_POSE target；target enum及key encoding形状合法；pair内两个bytewise target identity严格排序且不同；pair数组排序、无重复；整个canonical JSON digest等于调用者给定hash。随后`_validate_resource_context()`要求该hash等于self-hashed static profile receipt中的`resource_validation_context_sha256`，并从`canonical_bytes`重新解析出新的immutable context。

公开`validate_resource_compatibility()`已删除逐调用allowlist，只读取重新验证后的`context.compatible_wrist_target_pairs`。独立probe结果：

1. 同一对wrist claims在空表profile下返回`PROFILE_RESOURCE_MUTEX`；
2. 只有换成包含该pair且hash匹配的新context/profile时才通过；
3. yes-context配no-profile、context hash字段错配、receipt未重算self-hash均拒绝；
4. 直接构造frozen dataclass，把内存中的pair字段替换为yes但保留no canonical bytes/hash，仍经reparse恢复no表并拒绝；
5. 替换canonical bytes却伪报no hash也拒绝；duplicate/reverse/same target/unknown key/uppercase hex/wrong namespace均fail closed。

每条实际claim仍在比较前由canonical wire constraint重算selector key，因此仅有profile artifact字节不能绕过locator/controller identity。原P1“相同profile因逐调用参数改变mutex结果”不再成立。

### 3.2 Exact internal-edge semantics：关闭

实现位置：

- `2743cbd:world_action_model/sm_fet_minimal_r1d_schema.py:2260-2282`
- `2743cbd:world_action_model/sm_fet_minimal_r1d_schema.py:2342-2371`

原`_snap_binary64_edge()`已完全删除。Axis将正规化后的binary64 `(x,y,z)`按规定顺序计算`rho2=x*x+y*y`；`rho2 <= exact 0x1p-80`进入vertical，等值取vertical。非vertical的actual binary64 `atan2/asin`结果直接交给`bisect_right`，没有容差或吸附。Roll同样直接对actual projected angle做half-open判定。

Roll额外枚举32个materialized decoded directions做exact tuple equality，使规范定义的`x_r`构造性tie稳定回到`roll_code=1+r`。该比较是零宽的：没有epsilon、ULP窗口或近似比较；相邻representable direction不会仅因“接近”而被改写。独立测试也对全512个axis逐一验证了internal edge两侧。

fix1反例在fix2的结果为：

```text
axis：actual atan2严格低于edge -> 低格；exact edge -> 高格
roll：actual atan2严格低于edge -> 低格；exact edge -> 高格
```

此前“edge下方2 ULP被8-ULP吸附到高格”的非零错误带不再存在。

## 4. 完整合同回归与边界说明

本轮没有只测试两个patch点。wire exact32B、rank/unrank/state integer、13B LE、nonVERIFY/VERIFY topology、same-type combinadic、human/text JSON、resource receipt/context与geometry surface均做了动态回归。VERIFY PATH拒绝、MONITOR 25维zero ownership、public-state/profile/codebook receipt、strict clock、missing/transparent等fix1已通过路径在fix2差分中未被修改；静态复核未见接线或语义漂移。

compatibility artifact可以包含不会匹配任何canonical claim的dead target identity；这类条目不能进入由逐claim validator枚举的compatibility graph，因为每条实际claim会先按当前wire、selector domain和context enum精确复算并验证。它不是profile伪造旁路；外部profile generator也不得以raw table长度替代实际枚举计数，真实profile仍应只输出oracle证明的disjoint channel pair。

本Go结论严格限定于本提交宣称的frozen structural codec/checker gates及本轮Python 3.13.9 CPU执行面。它不表示真实G1 metric tables、working-surface/resource allowlist、hardware capability、runtime reachability/WBC/collision receipt或controller profile已经存在。

## 5. 最终gate

```text
P0: 0
P1: 0
CPU-conformant Freeze: Go

G1 executable profile: No-Go
G1 profile states:     UNDEFINED
G1 profile bits:       UNDEFINED
```

因此，可以把最终review中的CPU implementation/conformance从PENDING升级为**CPU-conformant Freeze Go**；不得把该结论外推为G1 executable profile Go。
