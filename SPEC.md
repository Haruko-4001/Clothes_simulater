# 任意衣服实时虚拟试衣 · 形式化项目规格书

> **本文件是给 AI Coding Agent（Claude Code）的唯一权威执行规格。**
> 放到空仓库根目录，命名为 `SPEC.md`；`CLAUDE.md` 只写一行：
> `先完整阅读 SPEC.md，严格按第 0 节 AGENT_INSTRUCTIONS 与附录 F 歧义消解规则执行。`
>
> 版本 **v2.0（学术增强版）** ｜ 结论基线 2026-11-02 ｜ 语言：中文（标识符 / 数学符号用英文）
> 取代 v1.1；v1.1 归档，**不得同时读取两份**，避免约束冲突。
> 本文件自包含，不依赖任何前序调研文件。

---

# Part I · 形式化基础

## 0. AGENT_INSTRUCTIONS（READ FIRST）

### 0.1 执行契约

1. 完整读完本文件，再创建任何文件。
2. 严格按 **第 4 节目录结构** 建文件；按 **第 15 节 Phase 0→5** 推进，每 Phase 的 `ACCEPT_CMD` 全绿才进下一 Phase。
3. 默认 `mode: mock`。**端到端主循环必须在无 GPU / 无权重 / 无网络 / 无 GUI 环境下连续跑 300 帧不崩**。真实模型仅通过 `scripts/install_real.sh --only=...` 可选接入。
4. 每个外部依赖包成 `Adapter`：`Real*` 与 `Stub*` 实现同一 `Protocol`。调用方只依赖 `Protocol`。`Stub` 必须返回**结构合法且数值自洽**的数据（例如 `physics/stubs.py` 必须是真弹簧质点积分器，不是 `pass`）。
5. 所有伪代码块带 `# IMPL: <file>::<symbol>` 标注，实现时必须落到该文件的该符号。
6. 结束时输出 **STRUCTURED_REPORT**（附录 G 格式），逐文件声明 `STUB / REAL / NOT_IMPLEMENTED / TODO`，禁止把未实现项写成已完成。

### 0.2 歧义消解优先级（冲突时按此裁决）

```
附录 A 符号表 / 附录 B 单位与常量表
  > 第 2 节数学定义与第 10 节数值形式（最高业务权威）
  > 第 8 节函数签名（含 raises / 不变式）
  > 第 9 节 dataclass 字段与取值范围
  > 伪代码块（# IMPL 标注）
  > 第 12 节 configs 数值
  > 表格 / 散文描述
  > 前序调研文件或模型 README（仅作背景，不覆盖本文件）
```
**单位一律以附录 B 为准；阈值一律以 `configs/*.yaml` 为准，代码内不得硬编码第二份。**

### 0.3 严禁事项（完整版见第 17 节）

- 严禁虚构仓库、star、fps、许可证、权重 URL、实测数字；
- 严禁把五类帧率口径混用；
- 严禁用生成模型覆盖物理层已锁定的接触点 / 速度 / 回弹；
- 严禁在 `MaterialState != CALIBRATED` 时输出「物理已就绪」；
- 严禁在 `FramingVerdict != OK` 时提交全图换装或外推画外区域；
- 严禁把「摄像头任意部位采集」实现成硬猜。

---

## 1. 符号约定与术语

### 1.1 通用记号

| 记号 | 含义 | 类型 / 维度 |
|---|---|---|
| $k, t_k$ | 离散帧索引与时间，$t_k=k\Delta t$，$\Delta t=1/30\,\mathrm{s}$ | $\mathbb{N},\mathbb{R}$ |
| $I^g$ | 用户上传的任意衣服图（单张 RGB） | $\mathbb{R}^{H_g\times W_g\times3}$ |
| $I_k$ | 摄像头第 $k$ 帧 | $\mathbb{R}^{1080\times1920\times3}$ |
| $K$ | 相机内参 | $\mathbb{R}^{3\times3}$ |
| ${}^C T_W=(R,t)$ | 世界系→相机系刚体变换 | $SE(3)$，$R\in SO(3)$ |
| $\Pi_K(\cdot)$ | 针孔投影 $\mathbb{R}^3\to\mathbb{R}^2$ | 见 2.2 |
| $\beta,\theta$ | SMPL-X 形状 / 姿态参数 | $\mathbb{R}^{10},\mathbb{R}^{|\theta|}$ |
| $\mathcal{A}$ | 服装资产 | 见 2.3 |
| $\mathcal{M}=(V,F)$ | 三角网格，$V\!\in\!\mathbb{R}^{N\times3}$，$F\!\in\!\mathbb{N}^{M\times3}$ | — |
| $S_k=(x_k,v_k,\Lambda_k)$ | 物理状态：位置 / 速度 / 拉格朗日乘子 | $x,v\in\mathbb{R}^{N\times3}$ |
| $\Theta$ | 布料材料参数向量 | $\mathbb{R}^{7}$，见 2.3 |
| $e_k$ | 接触事件 | 见 9.3 |
| $\mathcal{D}_{a:b}$ | 观测窗口 $\{(e_\tau,\hat x_\tau,I_\tau)\}_{\tau=a}^{b}$ | — |
| $\|\cdot\|_\Sigma$ | 马氏范数 $\|r\|_\Sigma^2=r^\top\Sigma^{-1}r$ | — |
| $\mathbb{1}[\cdot]$ | 指示函数 | — |

### 1.2 坐标系（**实现必须统一，禁止隐式假设**）

| 系 | 记号 | 手性 / 轴向 | 单位 | 备注 |
|---|---|---|---|---|
| 世界 | $W$ | 右手，**Y-up** | m | 重力 $\mathbf{g}=(0,-9.81,0)\,\mathrm{m/s^2}$ |
| 相机 | $C$ | 右手，**Z-forward** | m | ${}^CT_W$，深度为正 |
| 人体 | $B$ | SMPL-X 根关节 | m | 由 $\beta,\theta$ 定义 |
| 服装静止态 | $G$ | 与 $B$ 对齐 | m | rest shape $V^0$ |
| 图像 | $\mathbb{I}$ | 左上原点，x→右 y→下 | px | $\Pi_K$ 输出 |

变换链：$^{C}p = K\,[{}^CT_W]\,{}^W\!p$，齐次坐标；旋转用**单位四元数** $q\in\mathbb{H}_1$ 存储，对外提供 `quat_to_R(q)`，禁止直接拼欧拉角（避免万向锁）。蒙皮：$^{\cdot}x_i = \sum_j \mathcal{W}_{ij}\,{}^BT_j\,{}^G\!V^0_i$。

### 1.3 五个帧率口径（**任何日志/指标禁止互推**）

| 变量 | 定义 | 目标 |
|---|---|---|
| `camera_capture_fps` | 采集帧率 | 30 |
| `display_fps` | 可见层刷新，用户感知 | 30 |
| `physics_hz` | 物理状态步/s | 60 |
| `keyframe_fps` | 异步高画质关键帧完成率 | 0.2–1 |
| `generation_throughput_fps` | 模型吞吐，**仅记录，禁止对外宣称** | 不设目标 |

「生成了 30fps 视频」$\neq$ 摄像头 30fps；「91 fps img2img」$\neq$ 试衣 91 fps。**延迟基准是 `Frame.pts_ms` 到显示时刻，报 p50 / p95 / max，不报均值。**

---

## 2. 问题形式化与目标函数

### 2.1 任务定义

给定单张衣服图 $I^g$ 与摄像头流 $\{I_k\}_{k\ge0}$，求：

- **资产** $\mathcal{A}$（离线推理，管线 A）；
- **逐帧物理状态** $\{S_k\}$（管线 B，实时）；
- **材料后验** $\hat\Theta$（管线 C，在线辨识）；

使得输出显示帧 $\hat I_k$ 同时满足：

$$
\min_{\mathcal{A},\{S_k\},\hat\Theta}\;
\underbrace{\mathcal{L}_{\text{vis}}}_{\text{衣服外观/身份}}
+\lambda_p\underbrace{\mathcal{L}_{\text{phys}}}_{\text{物理可实现}}
+\lambda_t\underbrace{\mathcal{L}_{\text{temp}}}_{\text{时序一致}}
+\lambda_c\underbrace{\mathcal{L}_{\text{cal}}}_{\text{材料可辨识}}
$$

约束：$\text{display\_fps}\ge30$，$\text{p95}(\text{capture}\to\text{display})\le100\,\mathrm{ms}$，且 $\hat I_k$ 在接触区**不得由生成模型覆盖** $S_k$ 的 $x,v,\Lambda$。

分项：

$$
\mathcal{L}_{\text{vis}}=\mathbb{E}_k\Big[\|I^g_{\text{feat}}-\mathcal{F}(\hat I_k)\|_1+\big(1-\mathrm{ID}(\hat I_k,I^g)\big)\Big]
$$

$$
\mathcal{L}_{\text{phys}}=\sum_k\Big[\|M\ddot x_k+f_{\text{int}}(x_k;\Theta)+f_{\text{ext}}-J^\top\Lambda_k\|^2+\Phi_{\text{pen}}(x_k)\Big]
$$

$$
\mathcal{L}_{\text{temp}}=\sum_k\|\hat I_k-\mathcal{W}(\hat I_{k-1},F_{k-1\to k})\|_1
$$

$$
\mathcal{L}_{\text{cal}}=\|\hat\Theta-\Theta^\ast\|^2_{P^{-1}}+\lambda\|\hat\Theta-\Theta_0\|^2_{P_0^{-1}}
$$

### 2.2 命题 1：单视图几何病态性（必须写成代码注释）

设针孔投影 $\Pi_K(p)=K[R|t]\tilde p / d(p)$。对可见像素 $u$，其反投影集

$$
\Pi_K^{-1}(u)=\{p={}^WT_C\,(d\,K^{-1}\tilde u):d\in(0,\infty)\}
$$

是一维射线，故零空间 $\mathcal{N}(\Pi_K)\neq\{0\}$：**深度不可由单图唯一确定**。记可见顶点集

$$
V_{\text{vis}}=\{i:\mathrm{visible}(V_i^0;K,{}^CT_W,\mathcal{M})\},\qquad
V_{\text{occ}}=\{1..N\}\setminus V_{\text{vis}}
$$

$V_{\text{occ}}$（背面 / 袖口内侧 / 领口内侧 / 下摆内侧 / 缝合次序）**无任何观测约束**，只能由条件先验采样：

$$
p(V_{\text{occ}}\mid I^g)=\int p(V_{\text{occ}}\mid z)\,p(z\mid I^g)\,dz
$$

> **推论 1.1**：$\mathcal{A}$ 是**概率资产**，必须携带 `confidence` 与 `provenance`；$\mathrm{conf}(V_{\text{occ}})<\tau_{\text{back}}$ 时 HUD 必须显示「AI 补全」。任何「单图唯一确定背面」的实现都违反本命题。

### 2.3 命题 2：静态材质秩亏

材料参数（附录 B 单位）：

$$
\Theta=(\rho_A,\;E,\;k_b,\;\nu,\;\mu,\;c_d,\;h)^\top\in\mathbb{R}^7
$$

依次为面密度、拉伸刚度（杨氏模量代理）、弯曲刚度、泊松比、摩擦系数、阻尼系数、厚度。

静态平衡 $f_{\text{int}}(x;\Theta)+f_{\text{ext}}=0$，观测 $y=h(x)$。灵敏度矩阵与 Fisher 信息：

$$
J_\Theta=\frac{\partial h}{\partial\Theta}\in\mathbb{R}^{m\times7},\qquad
\mathcal{I}(\Theta)=J_\Theta^\top\Sigma^{-1}J_\Theta
$$

静态下 $\dot x=\ddot x=0$，含 $c_d$ 的耗散项与含 $\rho_A$ 的惯性项在观测方程中列线性相关；同时 $E$ 与 $k_b$ 对静态轮廓的作用可由 $h$ 缩放补偿 ⇒ $\mathrm{rank}(\mathcal{I})<7$。

> **推论 2.1**：单张静态图**不可辨识** $\Theta$，只能给出先验档位 $\Theta_0\sim\mathcal{N}(\bar\Theta,P_0)$。

### 2.4 命题 3：动态激励恢复秩（在线辨识的理论依据）

当用户施加抓取/拉扯，$\dot x\neq0,\ddot x\neq0$，耗散项 $c_d\dot x$ 与惯性项 $\rho_A\ddot x$ 进入观测方程，$\mathcal{I}$ 的列独立性随激励类型恢复：

| 激励 | 主要恢复的参数列 | 判别量 |
|---|---|---|
| 单轴拉动 | $E,\;c_d$ | $\|\partial h/\partial E\|,\|\partial h/\partial c_d\|>\epsilon_J$ |
| 自由悬垂 | $\rho_A,\;k_b$ | $\|\ddot x\|>\epsilon_a$ |
| 相对滑动 | $\mu$ | 切向相对位移 $> \epsilon_s$ |
| 松手回弹 | $c_d,\;E$ | 能量衰减率可测 |

> **推论 3.1**：可信物理必须依赖**在线系统辨识**；这仍是「推理」，但把 §2.3 的秩亏问题转成序列估计问题。
> **推论 3.2（可辨识性闸门）**：仅当 $\mathrm{cond}(\mathcal{I})<\kappa_{\max}$ 且 $\mathrm{tr}(P_k)$ 单调下降时才允许 `COMMIT`；否则保持 $\Theta_0$。

### 2.5 「任意」二字的精确边界（写进 README）

$$
\text{任意衣服}\;\triangleq\;\neg\exists\,\text{SKU 级人工登记},\quad
\text{但}\;\exists\,\text{类别先验}\;\mathcal{C}_{\text{cls}}
$$

- 免降级集合：$I^g$ 属于训练覆盖的缝纫图案分布，且取景合规；
- 降级集合 $\Omega_{\text{deg}}$：透明 / 多层嵌套 / 极端遮挡 / 非标准穿法 / 镜面 / 局部裁切 / 侧背面；
- 对 $\Omega_{\text{deg}}$ 必须走 `policy/degrade.py`，**不得硬猜**。

---

## 3. 已固化技术判定（不得重新论证）

| ID | 判定 | 工程动作 |
|---|---|---|
| D1 | 单图几何可冷启动，属概率资产 | `confidence` + `provenance` 必填 |
| D2 | 静态图不可唯一反演 $\Theta$ | 必须实现管线 C |
| D3 | 随意交互需维护 $x,v,\Lambda$ 与接触 | 物理层不可省略 |
| D4 | 高画质换装秒级 | 关键帧异步 0.2–1 fps |
| D5 | 五类 fps 分离 | `TimingBook` 分字段 |
| D6 | 任意衣服 ≠ 无需先验 | 降级通道 |
| D7 | 在线辨识无现成成品 | 自行实现 §10.3 |
| D8 | StreetTryOn（WACV 2025）从 DeepFashion2 筛除 >90% 非正面 / 大遮挡 / 暗光图 | 侧背不得当 in-domain |
| D9 | DensePose / 人体解析在局部裁切下肢体检测不全，SAM2 / SAPIENS 改进有限且逐帧抖动（RealVVT 附录 B） | `FramingGate` 门禁 |
| D10 | 任意观看方向需多视图 / 源视频：VTON 360（CVPR 2025，代码+权重开源，需 nerfstudio/gsplat）；TryOnCrafter（ECCV 2026，4D 代理 + 视频 DiT，GitHub 代码未核到） | 仅离线 / 异步渲染 |
| D11 | mask-free 降低解析依赖：FASHN VTON v1.5（Apache-2.0，972M≈2GB，~8GB，H100≈5s/图）；OmniTry（NeurIPS 2025，12 类可穿戴，≥28GB） | 默认后端 FASHN |
| D12 | SMPL 拟合需足够全身关键点；局部裁切下单帧不稳 | 多帧时序累积 |
| D13 | 通用 3D（InstantMesh / TRELLIS / Hunyuan3D）只出视觉 mesh，无厚度/缝合/rest shape | 双资产分离 |
| D14 | 几何误差会被材料优化「吸收」（提高 $E$ 补偿缺失布料 / 降低 $\rho_A$ 制造假悬垂） | 冻结 + 消融 |

### 3.1 取景与视角能力分级（C9 兑现口径）

| 能力 | 判定 | 硬限制 |
|---|---|---|
| 任意衣服图（tops/bottoms/one-pieces） | 可用 | 类别枚举；秒级非实时 |
| 任意穿戴品类（首饰/包/帽/腰带/眼镜） | 可用（图像级） | 12 类；≥28GB；非流式 |
| 正面 + 全身/半身 + 无大遮挡 | **可实时** | 必须过门禁 |
| 局部裁切（只腿 / 只 torso / 膝盖截断） | **不可免降级** | 只试可见区，画外不外推 |
| 侧身 / 背面 / 转身 | 需异步 3D 资产 | 单图背面 = AI 补全 |
| 任意观看方向自由环绕 | 离线可做，实时不可 | 多视图 / 源视频 + Video DiT |

> 所有 fps / 耗时数字均为**公开 README / 模型卡 / 论文口径，需目标硬件实测**，禁止写入「已实测」表。

---

# Part II · 系统规范

## 4. 目录结构（精确到文件）

```
.
├── CLAUDE.md                      # 单行指向 SPEC.md
├── SPEC.md                        # 本文件
├── README.md
├── Makefile
├── pyproject.toml
├── environment.yml
├── docker/Dockerfile
├── configs/
│   ├── default.yaml  mock.yaml  real.yaml
│   └── hardware/{rtx4090,rtx3060}.yaml
├── src/tryon/
│   ├── __init__.py  types.py  protocols.py  factory.py  config.py  logging.py
│   ├── asset/
│   │   ├── __init__.py  segment.py  geometry.py  mesh_repair.py  drape.py
│   │   ├── material_prior.py  asset_builder.py  registry.py  stubs.py
│   ├── perception/
│   │   ├── __init__.py  camera.py  body.py  framing.py  hands.py  contact.py  stubs.py
│   ├── physics/
│   │   ├── __init__.py  state.py  solver.py  constraints.py  integrator.py  stubs.py
│   ├── render/
│   │   ├── __init__.py  gaussian.py  mesh_render.py  stubs.py
│   ├── keyframe/
│   │   ├── __init__.py  trigger.py  tryon.py  fuse.py  stubs.py
│   ├── interaction/
│   │   ├── __init__.py  state_machine.py  gestures.py  feedback.py
│   ├── calibration/
│   │   ├── __init__.py  collector.py  loss.py  identifier.py  gates.py  stubs.py
│   ├── display/
│   │   ├── __init__.py  compositor.py  hud.py
│   ├── policy/
│   │   ├── __init__.py  license_gate.py  degrade.py
│   ├── eval/
│   │   ├── __init__.py  metrics.py  report.py
│   └── math/
│       ├── __init__.py  se3.py  projection.py  quaternion.py  numerics.py
├── app/{main.py,cli.py}
├── adapters_vendor/README.md
├── scripts/{install_real.sh,fetch_weights.sh,smoke.sh}
├── tests/
│   ├── test_math.py            test_types.py        test_asset_stub.py
│   ├── test_physics_stub.py    test_state_machines.py  test_framing_gate.py
│   ├── test_calibration_gates.py  test_identifiability.py  test_fps_bookkeeping.py
│   ├── test_fusion.py          test_license_gate.py  test_end_to_end_mock.py
│   └── conftest.py             # 共享 fixture：mock_cfg / stub_asset / synth_trajectory
└── assets/samples/
```

`src/tryon/math/` 是 §2 数学定义的**唯一实现位置**，其他模块禁止自行实现投影/四元数/SE(3)。

---

## 5. 三条管线（严格隔离）

```
【A 离线 / 换衣触发，非实时】
I^g → 分割/类别 (0.05–1s) → 几何冷启动 (10s~数分钟) → 材料先验 Θ₀ (<1s)
    → GarmentAsset v0 (rest mesh + 缝纫 + 碰撞体 + 纹理 + Θ₀ + confidence + version)

【B 运行时 30fps 主循环】
I_k → Body/Hand → FramingGate → Contact → Physics.step → Render
    → Compositor(异步关键帧) → HUD → Display
    ↕ 请求/结果队列（max_queue=2，过期丢弃）

【C 后台，边跑边收敛】
𝓓_{k-W:k} → 激励判别 → 灵敏度列选择 → LM/EKF 更新 Θ → 三闸门 → asset v1 或保持 v0
```

**三条不可违反的工程约束**：

1. 管线 C **绝不阻塞** B。B 只用当前 $\Theta$ 积分；C 低频批量、抓取结束后提交。
2. **几何与参数分步冻结**（D14）：先固定 $V^0$ 与缝合，再逐个放开 $\Theta_j$；支持消融 `disable_back_prior / disable_material_opt / disable_friction_opt`。
3. 校准参数**不永久写入 SKU**：带 `scene_tag` 与 TTL，跨速度 / 抓取方式 / 人体尺度失效。

### 5.1 数据流 schema（跨模块 JSON 契约）

```jsonc
// 每帧唯一 ID，贯穿日志
{"frame_id": 1234, "pts_ms": 41123.7, "asset_id": "g_8f3c", "asset_version": 3}
// 日志必含，禁止省略
{"timing": {"camera_capture_fps":30,"display_fps":30,"physics_hz":60,
            "keyframe_fps":0.5,"generation_throughput_fps":0.0,
            "capture_to_display_ms_p50":0,"capture_to_display_ms_p95":0,
            "keyframe_latency_ms_p95":0},
 "framing":{"verdict":"OK","visible_keypoint_ratio":0.93,"guide_text":""},
 "material_state":"CALIBRATING","confidence":{"back_geometry":0.61},
 "physics":{"attached_count":2,"penetration_count":0,"energy":0.42}}
```

---

## 6. 状态机（形式化转移表）

### 6.1 资产状态机 $\mathcal{M}_A=(Q_A,\Sigma_A,\delta_A,q_0)$

$Q_A=\{\texttt{BOOTSTRAP, GEOMETRY\_READY, MATERIAL\_PRIOR, NEEDS\_REVIEW, CALIBRATING, CALIBRATED, DEGRADED}\}$

| 当前 | 事件 / 守卫 | 下一 | 副作用 |
|---|---|---|---|
| BOOTSTRAP | `G1.pass` | GEOMETRY_READY | — |
| BOOTSTRAP | `G1.fail` | DEGRADED | `needs_human_review=True` |
| GEOMETRY_READY | `conf.overall ≥ τ_review` | MATERIAL_PRIOR | 写 $\Theta_0$ |
| GEOMETRY_READY | `conf.overall < τ_review` | NEEDS_REVIEW | HUD 提示 |
| MATERIAL_PRIOR | `is_excited ∧ cond(𝓘)<κ_max` | CALIBRATING | 启动后台线程 |
| CALIBRATING | `G3.COMMIT` | CALIBRATED | `version+=1`, `scene_tag`, TTL |
| CALIBRATING | `n_interaction > 50 ∧ ¬converged` | DEGRADED | 保留 $\Theta_0$ |
| NEEDS_REVIEW | 人工确认 / 补图 | GEOMETRY_READY | 重跑 G1 |

### 6.2 交互状态机 $\mathcal{M}_I$

```
IDLE ──GRASP──► GRASPED ──PULL──► PULLING ──RELEASE──► REBOUNDING ──E<E_min──► IDLE
                   └──LIFT──► LIFTING ──┘                    ▲
                                    └──SLIDE───────────────┘
```
不变式：`RELEASE` 时 `attached.clear()` 但 **`velocities` 必须保留**；`REBOUNDING` 由 $v$ + 阻尼 + 弹性连续演化，**禁止生成模型补帧**。能量必须满足 $E_{k+1}\le E_k(1-\gamma\Delta t)$。

### 6.3 关键帧四态机 $\mathcal{M}_K$

设姿态加权差 $d_k=\|\theta_k-\theta_{k-1}\|_W$，稳定判据
$\bar d_k^{(m)}=\frac1m\sum_{j=0}^{m-1}d_{k-j}<\epsilon_{\text{stable}}$，其中 $m=\lceil T_{\text{stable}}/\Delta t\rceil$。

```
TRACKING ──[d̄<ε_in, 持续 m 帧]──► STANDBY ──[衣服ID变化 ∨ 超时 ∨ bbox跨阈值]──► GENERATING
   ▲                                                                          │
   └──────────────[Alpha 融合 6 帧完成] ◄── FUSING ◄──[结果返回 ∧ ¬expired]────┘
```
**迟滞防抖**：进入用 $\epsilon_{\text{in}}$，退出用 $\epsilon_{\text{out}}>\epsilon_{\text{in}}$。
**过期判据**：$\|\theta_k-\theta_{\text{req}}\|_2>\delta_{\text{pose}}$ ⇒ 丢弃（队列 ≤2）。
**背面禁令**：`facing != frontal` ⇒ 延长等待，禁止把正面换装仿射到背面。

---

## 7. 评测指标数学定义（先定义再实现）

| 指标 | 公式 | 阈值 |
|---|---|---|
| IoU | $\|A\cap B\|/\|A\cup B\|$（轮廓掩膜） | G1 ≥0.90 |
| Chamfer CD | $\frac1{\vert A\vert}\sum_{a}\min_b\|a-b\|_2+\frac1{\vert B\vert}\sum_b\min_a\|b-a\|_2$ | 相对比较用 |
| 法向误差 | $\frac1N\sum_i\arccos(n_i\cdot \hat n_i)$ | 预设阈值 |
| ADE@$H$ | $\frac1H\sum_{\tau=1}^{H}\|\hat x_{k+\tau}-x_{k+\tau}\|_2$ | $H=0.4$s≈12帧 |
| FDE@$H$ | $\|\hat x_{k+H}-x_{k+H}\|_2$ | 低于未校准基线 |
| 穿模率 | $\#\{i:g_i(x_k)<0\}/N$ | 0 |
| 参数误差 | $\varepsilon_\Theta=\|\hat\Theta-\Theta^\ast\|_2/\|\Theta^\ast\|_2$ | 合成数据可测 |
| 能量衰减 | $E_{k+1}/E_k\le1-\gamma\Delta t$ | 严格 |
| 收敛 | $\mathrm{tr}(P_k)$ 连续 5 窗下降且相对变化 <2% | G3 |
| p95 延迟 | 经验 95 分位（**不用均值**） | ≤100ms |

**禁止的判定**：主观观感、单帧好看、$\mathcal{L}$ 单独下降（几何错误也会降 loss）、用户轻晃即宣称已校准。

---

## 8. 模块契约（完整函数签名）

> 全部 `@runtime_checkable`；`raises` 为实现必须抛出的异常（附录 C 错误码）；`INV` 为不变式，必须写单测。

```python
# IMPL: src/tryon/math/se3.py
def quat_to_R(q: np.ndarray) -> np.ndarray: ...
    # INV: R.T@R=I, det(R)=1; q 单位四元数; raises E_MATH_QUAT
def compose_T(R: np.ndarray, t: np.ndarray) -> np.ndarray: ...   # 4x4 SE(3)
def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray: ...  # (N,3)->(N,3)

# IMPL: src/tryon/math/projection.py
def project(K: np.ndarray, T_CW: np.ndarray, p_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (uv (N,2), depth (N,)); INV: depth>0; raises E_MATH_DEPTH"""
def backproject(K: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray: ...

# IMPL: src/tryon/asset/segment.py::Segmenter
def run(self, image: RGBImage) -> Segmentation: ...
    # INV: mask.shape==image.shape[:2]; 类别 ∈ enum GarmentClass

# IMPL: src/tryon/asset/geometry.py::GeometryEstimator
def estimate(self, image: RGBImage, seg: Segmentation) -> GeometryDraft: ...
    # INV: len(stitches)>0; rest_mesh 顶点有限; provenance 标注 V_occ

# IMPL: src/tryon/asset/mesh_repair.py::MeshRepairer
def repair(self, draft: GeometryDraft) -> SimReadyMesh: ...
    # INV: is_closed ∧ ¬has_isolated_edge ∧ self_intersection_count==0 ∧ thickness>0
    # raises E_ASSET_GEOM_FAIL 时禁止进入物理层

# IMPL: src/tryon/asset/drape.py::DrapeAligner
def align(self, mesh: SimReadyMesh, seg: Segmentation) -> DrapeReport: ...

# IMPL: src/tryon/asset/material_prior.py::MaterialPrior
def prior(self, image: RGBImage, seg: Segmentation) -> MaterialParams: ...
    # INV: state==AI_PRIOR; 所有分量 ∈ 附录 B 可行域; 禁止声称真值

# IMPL: src/tryon/asset/asset_builder.py::AssetBuilder
def build(self, image: RGBImage, *, timeout_s: float = 300.0) -> GarmentAsset: ...
    # 编排 A; raises E_ASSET_TIMEOUT / E_ASSET_GEOM_FAIL; version=0

# IMPL: src/tryon/perception/body.py::BodyTracker
def track(self, frame: Frame, history: deque[BodyState] | None = None) -> BodyState: ...
    # D12: 累积 temporal_accumulate_frames 帧再拟合; 禁止单帧硬拟合

# IMPL: src/tryon/perception/framing.py::FramingGate
def check(self, frame: Frame, body: BodyState) -> FramingCompliance: ...
    # INV: verdict ∈ 4 态; GUIDE/PARTIAL ⇒ guide_text != ""
    # INV: PARTIAL ⇒ 画外区域不得产生新生成内容（由调用方保证，单测断言）

# IMPL: src/tryon/perception/hands.py::HandTracker
def track(self, frame: Frame) -> HandState: ...

# IMPL: src/tryon/perception/contact.py::ContactExtractor
def extract(self, hands: HandState, body: BodyState,
            phys: PhysicsState) -> list[ContactEvent]: ...
    # 时序滤波 + 世界坐标; INV: world_pos 为 W 系, 单位 m

# IMPL: src/tryon/physics/solver.py::PhysicsSolver
def step(self, state: PhysicsState, asset: GarmentAsset,
         contacts: list[ContactEvent], dt: float) -> PhysicsState: ...
    # 复杂度 O(N); 预算 ≤5ms @ N=6000 (mock 口径测框架)
    # INV: 附着约束持续; 松手保留 velocities; 能量单调衰减
    # INV: 生成模型不得覆盖 x/v/Λ

# IMPL: src/tryon/keyframe/trigger.py::KeyframeTrigger
def should_fire(self, ctx: TriggerContext) -> bool: ...
    # INV: framing.verdict != OK ⇒ 恒 False (硬门禁)

# IMPL: src/tryon/keyframe/tryon.py::KeyframeTryOn
def generate(self, req: KeyframeRequest) -> KeyframeResult: ...
    # 绝不进每帧循环; raises E_KF_TIMEOUT

# IMPL: src/tryon/calibration/collector.py::TrajectoryCollector
def observe(self, ev: ContactEvent, state: PhysicsState) -> None: ...
def is_excited(self) -> bool: ...
    # 轻晃 ⇒ False; 只更新置信度不改参数

# IMPL: src/tryon/calibration/loss.py::compute
def compute(batch: ObservationBatch, w: LossWeights, theta: np.ndarray,
            theta0: np.ndarray, P_inv: np.ndarray,
            sim: PhysicsSolver) -> tuple[float, np.ndarray, np.ndarray]:
    """-> (L, g=JᵀΣ⁻¹r, J=∂r/∂Θ); 公式见 §10.3 (1)(2)
    INV: L 有限且 ≥0; J.shape == (Στ dim(rτ), 7); 全分量在可行域外时 raises E_CALIB_GATE_REJECT
    注意: r 由 batch 经 sim 前向积分得到，禁止直接用观测残差代替可微残差。"""

# IMPL: src/tryon/calibration/identifier.py::SystemIdentifier
def update(self, batch: ObservationBatch) -> IdentificationResult: ...
    # 后台线程; 禁止阻塞主循环; 限幅 ‖ΔΘ‖_{P⁻¹} ≤ δ_max

# IMPL: src/tryon/calibration/gates.py::CalibrationGate
def evaluate(self, res: IdentificationResult, hist: CalibHistory) -> GateVerdict: ...
    # COMMIT 需同时满足: ΔL/L₀>ε_L ∧ tr(P)下降 ∧ cond(𝓘)<κ_max ∧ predicts_next_frame

# IMPL: src/tryon/display/hud.py::HUD
def draw(self, frame: RGBImage, status: SystemStatus) -> RGBImage: ...
    # INV: 恒显示 MaterialState 之一 + framing.verdict + guide_text
```

**工厂**：`factory.build(cfg) -> dict[str, object]`，键为 Protocol 名；`cfg.mode` 决定 `Real*/Stub*`；探测失败回退 `Stub` 并 `logger.warning(..., code=E_ADAPTER_FALLBACK)`。

---

## 9. 数据结构（字段 / 单位 / 取值范围 / 不变式）

### 9.1 枚举

```python
class MaterialState(str, Enum):
    AI_PRIOR="AI_PRIOR"; CALIBRATING="CALIBRATING"; CALIBRATED="CALIBRATED"; DEGRADED="DEGRADED"
class GeometryProvenance(str, Enum):
    OBSERVED="OBSERVED"; PRIOR_BACK="PRIOR_BACK"; PRIOR_PATTERN="PRIOR_PATTERN"
class FramingVerdict(str, Enum):
    OK="OK"; PARTIAL="PARTIAL"; GUIDE="GUIDE"; REJECT="REJECT"
class ContactKind(str, Enum):
    GRASP="GRASP"; PULL="PULL"; LIFT="LIFT"; RELEASE="RELEASE"; SLIDE="SLIDE"
class GateVerdict(str, Enum):
    KEEP_PRIOR="KEEP_PRIOR"; CALIBRATING="CALIBRATING"; COMMIT="COMMIT"; DEGRADE="DEGRADE"
class GarmentClass(str, Enum):
    TOPS="tops"; BOTTOMS="bottoms"; ONE_PIECES="one-pieces"; ACCESSORY="accessory"; UNKNOWN="unknown"
```

### 9.2 资产

```python
@dataclass
class MaterialParams:
    areal_density: float      # kg/m^2   [0.02, 1.20]
    youngs_modulus: float     # N/m      [1e2, 1e6]   (拉伸刚度代理)
    bending_stiffness: float  # N·m      [1e-6, 1e-1]
    poisson_ratio: float      # -         [0.0, 0.49]
    friction: float           # -         [0.0, 1.5]
    damping: float            # N·s/m     [0.0, 50.0]
    thickness: float          # m         [1e-4, 5e-3]
    state: MaterialState = MaterialState.AI_PRIOR
    variance: dict[str, float] = field(default_factory=dict)
    scene_tag: str = ""       # 禁止跨场景复用
    calibrated_at: float | None = None
    ttl_s: float = 1800.0     # 过期策略
    # INV: 全分量在可行域内；state!=CALIBRATED ⇒ calibrated_at is None

@dataclass
class Confidence:
    segmentation: float; visible_geometry: float; back_geometry: float
    topology: float; material: float; overall: float   # 全部 [0,1]
    # INV: overall = 加权几何平均, 权重见 configs; 单项<τ ⇒ 对应 provenance 标注

@dataclass
class GarmentAsset:
    asset_id: str; version: int; garment_class: GarmentClass
    rest_mesh: SimReadyMesh       # 物理 shell
    appearance_mesh: object      # 视觉 mesh / 3DGS（D13：与 shell 分离，须有对应关系）
    sewing_pattern: dict; material: MaterialParams; collision_body: object
    confidence: Confidence; provenance: dict[str, GeometryProvenance]
    created_at: float; needs_human_review: bool = False
    # INV: version 单调递增; COMMIT 才 +1; 外观 mesh 禁止直接进物理层
```

### 9.3 运行时

```python
@dataclass
class Frame:
    frame_id: int; pts_ms: float; image: np.ndarray   # uint8 HxWx3

@dataclass
class FramingCompliance:
    verdict: FramingVerdict
    visible_keypoint_ratio: float   # [0,1]
    bbox_clipped: bool
    facing_estimate: str            # frontal|side|back|unknown
    occlusion_ratio: float          # [0,1]
    framing_confidence: float       # [0,1]
    guide_text: str = ""
    # INV: verdict in (GUIDE,PARTIAL) ⇒ guide_text != ""

@dataclass
class ContactEvent:
    t_ms: float; kind: ContactKind; world_pos: np.ndarray  # W 系, m
    vertex_id: int | None; force_proxy: float; hand_id: str
    # force_proxy 是视觉代理，不是真实力传感器，禁止写入物理解算器作为力输入

@dataclass
class PhysicsState:
    positions: np.ndarray   # (N,3) m
    velocities: np.ndarray  # (N,3) m/s
    attached: dict[int, str]   # vertex_id -> hand_id
    lam: np.ndarray            # 约束乘子 (M_c,)
    step: int; energy: float   # J
    # INV: positions/velocities 有限; RELEASE 后 attached 清空但 velocities 非零

@dataclass
class TimingBook:
    camera_capture_fps: float=0.0; display_fps: float=0.0
    physics_hz: float=0.0; keyframe_fps: float=0.0
    generation_throughput_fps: float=0.0
    capture_to_display_ms_p50: float=0.0; capture_to_display_ms_p95: float=0.0
    keyframe_latency_ms_p95: float=0.0
    # INV: 五字段独立赋值，禁止由任一字段推算另一字段

@dataclass
class KeyframeRequest:
    req_id: str; garment_asset_id: str
    person_image: np.ndarray; garment_image: np.ndarray; mask: np.ndarray
    pose: BodyState; enqueued_at_ms: float

@dataclass
class KeyframeResult:
    req_id: str; image: np.ndarray; mask: np.ndarray
    finished_at_ms: float; expired: bool=False

@dataclass
class IdentificationResult:
    params: MaterialParams; loss_before: float; loss_after: float
    param_variance: dict[str, float]; fisher_cond: float
    predicts_next_frame: bool
```

### 9.4 网格辅助

```python
@dataclass
class SimReadyMesh:
    vertices: np.ndarray; faces: np.ndarray
    thickness: float          # m >0
    is_closed: bool; has_isolated_edge: bool
    self_intersection_count: int; collision_ready: bool

@dataclass
class DrapeReport:
    silhouette_iou: float; chamfer_distance: float
    normal_error: float; flyaway_faces: int
```

### 9.5 其余被引用的类型（**全部必须定义，禁止在 §8 签名里使用未定义符号**）

```python
# IMPL: src/tryon/types.py
RGBImage = np.ndarray                  # uint8 (H,W,3)，RGB 顺序，禁止 BGR

@dataclass
class Segmentation:
    mask: np.ndarray                   # uint8 (H,W), 0/1
    garment_class: GarmentClass
    bbox: tuple[int, int, int, int]    # x0,y0,x1,y1, px
    confidence: float                  # [0,1]
    # INV: mask.shape == 输入 image.shape[:2]

@dataclass
class BodyState:
    smpl_params: dict                  # {"beta": (10,), "theta": (|θ|), "transl": (3,)}
    pose_keypoints: np.ndarray         # (J,3) 或 (J,2)+conf，W 系 m
    bbox: tuple[int, int, int, int]
    confidence: float
    fitted_from_frames: int = 1        # D12: 累积帧数，禁止为 1 时宣称高置信拟合

@dataclass
class HandState:
    landmarks: np.ndarray              # (21,3) 或 MANO mesh 顶点
    mesh: object | None
    handedness: str                    # left|right
    occluded: bool

@dataclass
class GeometryDraft:
    panels: list[object]
    stitches: list[tuple[int, int]]
    rest_mesh: object
    back_appearance: object | None     # 仅视觉，D13
    class_prior: GarmentClass
    provenance: dict[str, GeometryProvenance]

@dataclass
class ObservationBatch:
    contacts: list[ContactEvent]
    states: list[PhysicsState]
    camera_views: list[RGBImage]
    # INV: len(contacts)==len(states)==len(camera_views)==window_frames(裁剪后允许更短)

@dataclass
class LossWeights:
    w_r: float = 1.0                   # 重投影
    w_v: float = 0.5                   # 速度
    w_s: float = 0.3                   # 轮廓 IoU
    w_p: float = 2.0                   # 穿模惩罚
    lam: float = 1.0e-3                # 先验正则
    sigma_inv: np.ndarray | None = None  # Σ⁻¹；None ⇒ 单位阵

@dataclass
class Residuals:
    reproj: np.ndarray                 # (T,2N)
    velocity: np.ndarray               # (T,3N)
    silhouette: np.ndarray             # (T,)
    penetration: np.ndarray            # (T,)

@dataclass
class CalibHistory:
    losses: list[float]
    variances: list[float]             # tr(P_k)
    conds: list[float]                 # cond(𝓘)
    n_interactions: int = 0
    # INV: 三个列表等长；G3 用连续 5 窗判定收敛

@dataclass
class TriggerContext:
    pose_history: deque                # 最近 m 帧 BodyState
    framing: FramingCompliance
    garment_id: str
    elapsed_since_kf_ms: float
    bbox_crossed: bool
    state: str                         # TRACKING|STANDBY|GENERATING|FUSING
    # INV: should_fire 在 framing.verdict != OK 时恒返回 False

@dataclass
class SystemStatus:
    asset_state: str
    material_state: MaterialState
    confidence: Confidence
    timing: TimingBook
    framing: FramingCompliance
    notice: str = ""
    # INV: HUD 必须渲染 material_state / framing.verdict / framing.guide_text
```

**符号一致性**：§2 用观测算子 $h$、§10 用残差 $r$；二者满足 $J_\Theta=\partial r/\partial\Theta = -\partial h/\partial\Theta$（差一个符号与 $\Sigma^{-1/2}$ 缩放），$\mathcal{I}=J^\top\Sigma^{-1}J$ 在两种写法下等价。**实现以 §10.3 的 $J=\partial r/\partial\Theta$ 为准**。

---

## 10. 数值方法（**核心学术章节，伪代码可直接翻译**）

### 10.1 物理积分：半隐式欧拉 + XPBD

```
# IMPL: src/tryon/physics/integrator.py::integrate
输入 x_k, v_k, Θ, dt, n_sub=4, contacts, mesh(V,F), W(蒙皮)
dt_s = dt / n_sub
for s in 1..n_sub:
    # (a) 外力：重力 + Rayleigh 阻尼 D = α_M·M + α_K·K
    #     α_M [1/s], α_K [s]，使 D 量纲为 kg/s = N·s/m
    #     M 为对角质量矩阵 diag(m_i)，m_i = ρ_A · A_i（A_i 为顶点关联面积，m²）
    f_ext[i] = m_i * g - (D @ v)[i]           # g=(0,-9.81,0) m/s²
    # (b) 预测（半隐式欧拉）
    v_pred[i] = v[i] + dt_s * (f_ext[i] + f_int(x;Θ)[i]) / m_i
    x_pred = x + dt_s * v_pred
    # (c) XPBD 约束投影，对每个约束 C_j(x)=0
    for j, λ_j in enumerate(λ):
        for inner in 1..iters:
            g_i = ∇_{x_i} C_j
            α̃ = α_j / dt_s²                     # α_j = 1/k_j, compliance
            Δλ = (-C_j(x_pred) - α̃*λ_j) / (Σ_i w_i‖g_i‖² + α̃)
            x_pred[i] += w_i * g_i * Δλ
            λ_j += Δλ
    # (d) 接触非穿透 g_i(x) ≥ 0 与库仑摩擦 ‖f_t‖ ≤ μ‖f_n‖
    apply_contact(x_pred, contacts, μ=Θ.friction)
    # (e) 速度更新
    v = (x_pred - x) / dt_s
    x = x_pred
返回 x, v, λ, E = ½ vᵀMv + U(x;Θ)
```

约束类型与刚度映射：距离约束 $C_{ij}=\|x_i-x_j\|-l^0_{ij}\leftarrow E,h$；二面角约束 $\leftarrow k_b$；接触 $\leftarrow \mu$；$\alpha_j=k_j^{-1}$。**稳定性**：$\Delta t_s\le\sqrt{\alpha_j}/\|\nabla C_j\|$ 量级；实现用固定 `n_sub=4` 并对 $\|\Delta x\|_\infty$ 做钳位，出现 `NaN`/发散即 `E_PHYS_DIVERGE` 并回退上一稳定状态。

**能量不变式**（单测）：松手后 $E_{k+1}\le E_k(1-\gamma\Delta t)$。$\gamma$ 取 `configs.physics.energy_decay_gamma`（单位 $\mathrm{s^{-1}}$，默认 0.15），与 $c_d$ 正相关但**不是** $c_d/\rho_A$（该比值量纲为 $\mathrm{m^2/s}$，不可直接作衰减率）；实现中 $\gamma$ 由阻尼比标定，标定方法记为 TODO，禁止在代码里用 $c_d/\rho_A$ 冒充。

### 10.2 接触附着

```
# IMPL: src/tryon/physics/constraints.py::apply_contact
GRASP  : attached[i_v] = hand_id;  x[i_v] = p_hand^W;  v[i_v] = 0
PULL/LIFT: 保持 attached；x[i_v] 跟随 p_hand^W（软约束，非硬置位）
RELEASE: attached.pop(i_v); 保留 v[i_v]（禁止清零）
遮挡   : 手不可见时保持 attached（时序外推 ≤ t_hold=0.3s），超时才释放
穿模惩罚 Φ_pen = Σ_i max(0, -g_i(x))²
```

### 10.3 在线系统辨识

**(1) 残差与损失**（`calibration/loss.py::compute`）

$$
r_\tau=\begin{bmatrix}
\sqrt{w_r}\big(\Pi_K(\hat x_\tau)-\Pi_K(x_\tau(\Theta))\big)\\
\sqrt{w_v}\big(v_\tau^{obs}-v_\tau(\Theta)\big)\\
\sqrt{w_s}\big(1-\mathrm{IoU}_\tau(\Theta)\big)\\
\sqrt{w_p}\,\Phi_{\text{pen}}^{1/2}(\Theta)
\end{bmatrix},\quad
\mathcal{L}(\Theta)=\sum_{\tau=t_0}^{t_1}\|r_\tau\|^2_{\Sigma}+\lambda\|\Theta-\Theta_0\|^2_{P_0^{-1}}
$$

**(2) 灵敏度与 Fisher**（`identifier.py`）

$$
J_\Theta=\frac{\partial r}{\partial\Theta},\qquad \mathcal{I}=J_\Theta^\top\Sigma^{-1}J_\Theta,\qquad \kappa=\mathrm{cond}(\mathcal{I})
$$

**(3) LM 更新（默认）**

```
# IMPL: src/tryon/calibration/identifier.py::update
Θ = Θ_prev
for it in 1..max_iter(=8):
    L, g, J = compute(batch, w, Θ, Θ_0, P_prev⁻¹, sim)   # §8 签名
    A = JᵀΣ⁻¹J + λ_LM * I + P_prev⁻¹        # 含先验正则（协方差限幅）
    Δ = solve(A, -g)                          # Cholesky，失败转 SVD
    # 投影到可行域 + 马氏步长限幅（D14 防参数跳跃）
    Δ = clip_mahalanobis(Δ, δ_max, P_prev)
    Θ_trial = project_box(Θ + Δ, Θ_min, Θ_max)
    if L(Θ_trial) < L(Θ): Θ = Θ_trial; λ_LM /= 3
    else: λ_LM *= 3
    if ‖Δ‖ < ε_Δ: break
# 协方差更新（EKF 形式）
P_new = (P_prev⁻¹ + 𝓘)⁻¹
# 只放开激励对应的列（§2.4），其余冻结
Θ = freeze_unexcited(Θ, sensitivity_mask(J_Θ, ε_J))
返回 IdentificationResult(Θ, L_before, L_after, var(P_new), κ, predicts_next_frame)
```

**`predicts_next_frame`**：用更新后的 $\Theta$ 前向积分 $H=0.4$s（≈12 帧），计算 ADE@$H$；仅当 $\mathrm{ADE}(\hat\Theta)<\mathrm{ADE}(\Theta_0)-\epsilon_{\text{ade}}$ 才为 `True`。

**(4) 激励充分性**（`collector.py::is_excited`）

```
轻晃  : ‖Δp_hand‖ < ε_s ∧ ‖a‖ < ε_a          ⇒ False（只更新置信度）
单轴拉: 主轴方差 / 次轴方差 > ρ_excitation     ⇒ mask=[E, c_d]
悬垂  : ‖a‖ > ε_a ∧ 无接触                    ⇒ mask=[ρ_A, k_b]
滑动  : 切向相对位移 > ε_s                    ⇒ mask=[μ]
松手  : RELEASE 后能量衰减可测                 ⇒ mask=[c_d, E]
```

**(5) 三闸门**（`gates.py::evaluate`，与 §11 一致）

```
G1 几何 : 正视图 IoU≥0.90 ∧ ¬has_isolated_edge ∧ self_intersection_count==0 ∧ flyaway_faces==0
          ∧ conf.back_geometry<τ_back ⇒ HUD 必含「AI 补全」
G2 悬垂 : silhouette_iou / CD / normal_error 达阈值；只验可见侧
G3 交互 : ΔL/L₀>ε_L ∧ tr(P) 连续下降 ∧ κ<κ_max ∧ predicts_next_frame ∧ ADE 优于基线
          ⇒ COMMIT；否则 KEEP_PRIOR；n>50 未收敛 ⇒ DEGRADE
```

**(6) 消融开关**（`configs.calibration.ablation`）必须能分别关闭 `back_prior / material_opt / friction_opt`，用于检测 D14 参数—几何污染。

### 10.4 取景合规门禁

```
# IMPL: src/tryon/perception/framing.py::check
r = 可见关键点数 / 期望关键点数
if occlusion_ratio > τ_occ 或 多主角 或 透明/多层:  REJECT  → policy.degrade
elif bbox_clipped 或 r < τ_vis:                     PARTIAL → 只试可见区 + guide_text
elif facing ∈ {side, back}:                         GUIDE   → 切异步 3D 渲染，禁止提交 2D 换装
elif r ≥ τ_vis ∧ ¬bbox_clipped ∧ facing=frontal:    OK
# D12: 累积 temporal_accumulate_frames(=30) 帧关键点再拟合 SMPL，禁止单帧硬拟合
```

### 10.5 异步融合

```
# IMPL: src/tryon/keyframe/fuse.py
Warp : Î_warp = W(I_kf, F_{kf→k})                 # 光流
α_k  = α₀ · exp(-(t_k - t_kf)/τ_fuse)             # 融合 6 帧递减
Î_disp = (1-α_k)·Î_render + α_k·Î_warp
过期 : ‖θ_k-θ_req‖₂ > δ_pose ⇒ discard，队列 ≤2
```

### 10.6 关键帧触发（带迟滞）

```
# IMPL: src/tryon/keyframe/trigger.py::should_fire
if framing.verdict != OK: return False            # 硬门禁
d̄_k = mean(d_{k-m+1..k}); m = ceil(T_stable/Δt)
if state==TRACKING and d̄_k<ε_in for m frames: state=STANDBY
if state==STANDBY and (garment_id_changed or elapsed>T_kf or bbox_crossed): fire=True
if d̄_k>ε_out: state=TRACKING                      # 迟滞，防抖
```

---

## 11. 三道验收闸门（形式化断言）

| 闸门 | 断言（可直接写 pytest） |
|---|---|
| G1 | `iou>=0.90 and not has_isolated_edge and self_intersection_count==0 and flyaway_faces==0`；`conf.back_geometry<τ_back ⇒ "AI 补全" in hud.text` |
| G2 | `silhouette_iou>=τ ∧ CD<=τ ∧ normal_error<=τ`；**只验可见侧**，不得替代动态验证 |
| G3 | `ΔL/L₀>ε_L ∧ tr(P)↓ ∧ κ<κ_max ∧ predicts_next_frame ∧ ADE(Θ̂)<ADE(Θ₀)-ε_ade` |

---

## 12. 配置模板与参数语义

```yaml
# configs/default.yaml —— 代码内禁止硬编码第二份数值
mode: mock                        # mock|real|hybrid
hardware: rtx4090
camera: {width: 1920, height: 1080, target_fps: 30}
math:                            # §2 / 附录 B
  gravity: [0.0, -9.81, 0.0]
  dt: 0.033333                  # 1/30 s
  world_up: Y
  camera_forward: Z
  rotation_repr: unit_quaternion
framing:                         # §10.4
  enabled: true
  min_visible_keypoint_ratio: 0.70
  max_occlusion_ratio: 0.35
  reject_bbox_clipped: false     # 裁切→PARTIAL，不直接拒
  allow_side_back: false         # 侧背必须走异步 3D
  require_class: [tops, bottoms, one-pieces]
  guide_on_partial: true
  temporal_accumulate_frames: 30  # D12
  contact_hold_s: 0.30
physics:                         # §10.1
  target_hz: 60
  substeps: 4
  dt: 0.016667
  max_vertices: 6000
  solver: stub_xpbd              # stub_xpbd|diffcloth|warp
  xpbd_iters: 6
  alpha_M: 0.02                  # Rayleigh
  alpha_K: 0.01
  energy_decay_gamma: 0.15
  penetration_clip_m: 0.002
render: {backend: stub}          # stub|mesh|3dgs
keyframe:
  max_queue: 2
  stable_ms: 800                 # T_stable
  fuse_frames: 6
  min_interval_ms: 1000
  epsilon_in: 0.015              # ε_in  (rad, 加权姿态差)
  epsilon_out: 0.035             # ε_out > ε_in，迟滞
  delta_pose: 0.12               # δ_pose，过期判据
  alpha0: 0.55
  tau_fuse_ms: 260
  backend: stub                  # stub|fashn_vton_1_5|catvton|leffa|omnitry
asset:
  geometry: stub                 # stub|sewformer|instantmesh|garmentrec
  material_prior: dictionary
  offline_timeout_s: 300
  back_confidence_threshold: 0.65   # τ_back
  review_threshold: 0.55            # τ_review
calibration:                     # §10.3
  params: [youngs_modulus, damping, bending_stiffness, areal_density]
  max_params_per_batch: 6
  window_frames: 90
  max_iter: 8
  lambda_LM: 1.0e-3
  delta_max: 0.35                # δ_max，马氏步长限幅
  epsilon_J: 1.0e-4              # 灵敏度列阈值
  epsilon_s: 0.02                # m
  epsilon_a: 0.5                 # m/s²
  rho_excitation: 3.0
  kappa_max: 1.0e6               # cond(𝓘) 上限
  commit_min_loss_drop: 0.15     # ε_L
  epsilon_ade: 0.005             # m
  predict_horizon_ms: 400        # H
  max_interactions_before_degrade: 50
  ablation: {disable_back_prior: false, disable_material_opt: false, disable_friction_opt: false}
policy:
  allow_nc_weights: false        # 生产必须 false
  require_human_review_below: 0.55
  reject_on: [transparent, multi_layer, extreme_occlusion, nonstandard_wear]
hud: {show_material_state: true, show_confidence: true, show_degrade_notice: true,
      show_framing: true, show_guide_text: true}
```

`real.yaml` 仅覆盖 `mode` 与各 `backend` 名；`mock.yaml` 全 stub。**不得把任何需联网下载的权重路径设为必需项。**

---

## 13. 外部仓库集成表

> git submodule 挂载，不复制源码；adapter 内先探测依赖，缺失回退 Stub 并记 `E_ADAPTER_FALLBACK`。查不到写「未公开/需实测」，**禁止补造**。

| 层 | 仓库 | 用途 / 公开口径 | 许可 | 约束 |
|---|---|---|---|---|
| 几何 | `sail-sg/sewformer` | 单图→面板/缝合；HF `liulj/sewformer` | 需复核 | Maya 链路需实测 |
| 几何 | `worryDes/GarmentRec` | 开放/闭合拓扑显式模板 | MIT（权重条款需复核） | 类别受限 |
| 几何 | `TencentARC/InstantMesh` | 背面外观先验 | Apache-2.0 | **只出视觉 mesh**（D13） |
| 几何 | `jby1993/BCNet` | 分层服装 mesh | **仅研究用途** | 禁 real 商用 |
| 几何 | `maria-korosteleva/Garment-Pattern-Estimation` | NeuralTailor 面板 | MIT | 输入是 3D 点云，非单图 |
| 参考 | `dress-1-to-3.github.io` | 图案+物理参数联合优化（SIGGRAPH 2025） | 需实测 | **代码未公开，禁止写成依赖** |
| 材质 | `chatgarment.github.io` | 面料档位 / 缝纫图案 | 需实测 | 只给先验（§2.3） |
| 物理 | `omegaiota/DiffCloth` | 可微 PD + 干摩擦 + 参数反演 | MIT | 核心；CMake |
| 物理 | `gradsim/gradsim` | 可微系统辨识 `demo_cloth.py` | 需复核 | 备选 |
| 物理 | `williamljb/DifferentiableCloth` | `demo_wind.py` 材料匹配 | 需复核 | 备选 |
| 物理 | `XPandora/PhysGaussian` | MPM + 3DGS | 需复核 | 渲染层 |
| 物理 | `Colmar-zlicheng/Spring-Gaus` | 弹簧质点 + 3DGS | Apache-2.0 | 渲染层 |
| 形变 | `0zgur0/HOOD` | 分层图服装动力学 | MIT | 需 `.obj` + SMPL |
| 形变 | `IGL-HKUST/UNIC` | 实时神经形变场 | 需复核 | 角色动画，非任意外力 |
| 手部 | MediaPipe Hands | 21 点 | 需复核模型卡 | **默认感知** |
| 手部 | `rolpotamias/WiLoR` | 3D 手重建，`--fast` | **CC-BY-NC-ND** + MANO | 仅实验 |
| 手部 | `geopavlakos/hamer` | 手部 mesh | MIT + MANO 受限 | 仅实验 |
| 人体 | `vchoutas/smplx` | 身体+手先验 | **非商用科研**，需注册 | 仅实验 |
| 高画质 | `fashn-AI/fashn-vton-1.5` | **默认后端**：mask-free，972M≈2GB，~8GB，H100≈5s | **Apache-2.0**，`fashn-human-parser` 另议 | 商业原型优先 |
| 任意穿戴 | `random2fly/OmniTry` | 12 类，无掩码自动定位 | LoRA Apache-2.0，基座需复核 | ≥28GB；图像级 |
| 任意视角 | `scnuhealthy/VTON360` | CVPR 2025，代码+权重 | checkpoint 为 **Open RAIL-M** 衍生 | 多视图 + nerfstudio/gsplat，**禁写实时** |
| 任意视角 | `TryOnCrafter_web` | ECCV 2026，4D 代理 + Video DiT | 需实测 | **GitHub 代码未核到，只作设计参考** |
| 评测 | `cuiaiyu/street-tryon-benchmark` | WACV 2025，>90% 非正面图被筛除 | 继承 DeepFashion2 | 验证 D8 |
| 高画质 | `Zheng-Chong/CatVTON` | 1024×768，<8GB | **CC BY-NC-SA 4.0** | 仅实验 |
| 高画质 | `franciszzj/Leffa` | A100≈6s/图 | MIT（训练数据 NC） | 备选 |
| 实时基座 | `cumulo-autumn/StreamDiffusion` | 论文 91.07 img2img fps @4090 | Apache-2.0 | **吞吐≠试衣 fps** |
| 视频 | `fudan-generative-vision/champ` | 姿态驱动 | 需复核 | 离线增强 |
| 视频 | `Tencent/MimicMotion` | 576×1024 / 72 帧 | 需复核 | 离线增强 |

**链接修正**：Champ 官方 `fudan-generative-vision/champ`；AnyDoor `ali-vilab/AnyDoor`；HOOD `0zgur0/HOOD`；`humanai-group/Champ`、`OpenGVLab/AnyDoor` 为 404；`github.org`/`github.co.uk` 为畸形域名。

**许可红线**：`CatVTON / IDM-VTON / OOTDiffusion / StableVITON / CausVid = CC BY-NC-SA 4.0`；`WiLoR = CC-BY-NC-ND`；`SMPL-X / MANO` 非商用且需注册；`RTV` 商用需书面授权；`SDXL-Turbo` 非商用。`policy/license_gate.py` 在 `mode: real` 必须拦截。**仓库 MIT/Apache ≠ 权重可商用。**

---

# Part III · 验证与执行

## 14. 不可承诺清单（写进 README 与 HUD）

```
✗ 首次拉扯即真实布料手感（Θ 是先验，需交互校准）
✗ 4K/60fps 与高细节布料同时成立
✗ 一帧一扩散生成
✗ 单图唯一确定背面 / 内侧 / 缝合次序 / 杨氏模量
✗ 透明、多层、极端遮挡、非标准穿法免降级
✗ 纯前馈生成模型充当物理真值层
✗ NC 许可权重可直接商用
✗ 摄像头随手拍哪块就试哪块（局部裁切 / 半个人出画 / 侧背面）免降级
✗ 单张摄像头图直接输出任意观看方向（需多视图或源视频，离线）
✗ 任意非正面视角下衣物 ID 与纹理完全一致
```
产品口径：**任意衣服自动冷启动；合规取景下越拉越准；任意视角由 3D 资产离线/异步渲染；HUD 显式标注材质状态与取景引导。**

## 15. 分阶段实施（每 Phase 带 ACCEPT_CMD）

### Phase 0 · 形式化骨架
产出：`math/` 全部 + `types.py` / `protocols.py` / `factory.py` / 全部 `stubs.py`；主循环 `Camera→Body/Hand→FramingGate→Contact→Physics→Render→Compositor→HUD`。`physics/stubs.py` 实现 §10.1 真积分器（重力/阻尼/回弹），`render/stubs.py` 用 PIL 画变形网格。
`ACCEPT_CMD`: `pytest tests/test_end_to_end_mock.py tests/test_math.py -q && make mock-run`
断言：300 帧不崩；`TimingBook` 五字段非零且互不串用；`R.T@R==I`；`det(R)==1`。

### Phase 1 · 资产推理层
实现 `asset/` Stub + Real（SewFormer / InstantMesh / 面料字典）；`mesh_repair.py` 非流形/厚度/缝合/碰撞体；G1 + `registry.py` 版本/置信度/TTL。
`ACCEPT_CMD`: `pytest tests/test_asset_stub.py tests/test_license_gate.py -q && make bootstrap-asset IMG=assets/samples/garment.png`
断言：封闭网格、无孤立边、无自交；`Θ` 默认 `AI_PRIOR` 且全分量在可行域。

### Phase 2 · 感知、取景门禁与物理真值
`perception/` Real（MediaPipe 优先）；`framing.py` 四态；D12 多帧累积；`constraints.py` 附着/碰撞/穿模。
`ACCEPT_CMD`: `pytest tests/test_physics_stub.py tests/test_framing_gate.py tests/test_state_machines.py -q`
断言：拉扯 2–5s 附着不丢；松手 `attached` 清空、`velocities` 非零、`E` 单调衰减；遮挡下约束保持 0.3s；四种 `FramingVerdict` 全覆盖；`PARTIAL` 时画外无新生成内容。

### Phase 3 · 在线辨识（**成败点，无现成成品**）
实现 §10.3；先合成 60–120 帧已知 $\Theta^\ast$ 轨迹，遮住参数反演，验证 $\varepsilon_\Theta$ 下降；再迁真实相机轨迹。
`ACCEPT_CMD`: `pytest tests/test_calibration_gates.py tests/test_identifiability.py -q`
断言：轻晃→`KEEP_PRIOR`；不收敛→`DEGRADE`；`cond(𝓘)` 高时禁止 COMMIT；几何错误不被参数补偿（消融对照）；`predicts_next_frame` 为真才提交。

### Phase 4 · 画质、帧率与视角
`keyframe/` 四态机 + 默认 FASHN adapter + 光流/Alpha；`render/` 3DGS；侧背只走异步 3D。
`ACCEPT_CMD`: `pytest tests/test_fps_bookkeeping.py tests/test_fusion.py -q`
断言：`capture_to_display_ms_p95≤100ms`（mock 口径测框架开销）；队列≤2；过期丢弃；`α_k` 递减。

### Phase 5 · 真实接入与交付
`install_real.sh --only=sewformer|fashn_vton|omnitry|catvton|wilor|diffcloth|physgaussian`；许可清单；`eval/report.py` 输出 p50/p95、IoU、CD、ADE/FDE、穿模数、$\varepsilon_\Theta$、显存峰值、取景合规率、降级次数。
`ACCEPT_CMD`: `make test && make eval-report`
断言：实测数据表**留空标 TODO**，禁止填虚构数字。

---

## 16. 测试矩阵（12 个文件，全部必须生成且跑通）

| 文件 | 关键断言 |
|---|---|
| `test_math.py` | `quat_to_R` 正交/行列式；`project` 深度>0；`backproject∘project` 射线一致；SE(3) 复合结合律 |
| `test_types.py` | 五类 fps 独立；`MaterialParams` 可行域；`state!=CALIBRATED ⇒ calibrated_at is None`；§9.5 全部类型可实例化 |
| `test_asset_stub.py` | 封闭无孤立边无自交；低背面置信度置 `PRIOR_BACK`；外观 mesh 不进物理层 |
| `test_physics_stub.py` | §6.2 不变式；能量衰减；穿模率 0；`NaN` 触发 `E_PHYS_DIVERGE` 回退 |
| `test_state_machines.py` | 三状态机全部转移可达；迟滞 $\epsilon_{in}<\epsilon_{out}$；关键帧队列≤2 |
| `test_framing_gate.py` | 全身正面→`OK`；膝盖截断→`PARTIAL` 且画外不外推；侧背→异步非实时；极端遮挡→`REJECT`；`GUIDE` 必带文案；多帧累积后拟合置信度上升 |
| `test_calibration_gates.py` | 轻晃/不收敛/`predicts_next_frame` 三分支；消融开关生效 |
| `test_identifiability.py` | 静态 $\mathrm{rank}(\mathcal{I})<7$；动态激励后 `cond` 下降；$\varepsilon_\Theta$ 低于基线 |
| `test_fps_bookkeeping.py` | 五字段互不污染；p50/p95 由 `pts_ms` 计算 |
| `test_fusion.py` | `α_k` 逐帧递减；融合恰 6 帧；`‖θ_k-θ_req‖₂>δ_pose` 时丢弃；队列≤2 |
| `test_license_gate.py` | `mode: real` + NC 权重抛异常；`allow_nc_weights:false` 生效 |
| `test_end_to_end_mock.py` | 无 GPU/网络/权重跑 300 帧；HUD 恒含材质状态 + 取景 verdict + 引导文案 |

> 合计 **12** 个文件；`DoD` 第 19 节与附录 E 均记为 12，禁止写成 8/9/10/11。

---

## 17. 禁止清单（护栏，可复制进 `CLAUDE.md`）

```
1. 禁止虚构仓库名、star、fps、许可证、权重 URL、论文/实测数字；查不到写「未公开/需实测」。
2. 禁止把 camera_capture_fps/display_fps/physics_hz/keyframe_fps/generation_throughput_fps 合并或互推。
3. 禁止用生成模型覆盖物理层已锁定的 x/v/Λ/接触点/回弹。
4. 禁止在 MaterialState != CALIBRATED 时输出「物理已就绪/材质已校准」。
5. 禁止把 Dress-1-to-3 / TryOnCrafter 写成可 clone 依赖（代码未公开）。
6. 禁止把 InstantMesh 等通用 3D 视觉 mesh 直接喂进物理引擎（D13）。
7. 禁止把校准参数永久写入 SKU；必须带 scene_tag 与 ttl_s。
8. 禁止让管线 C 跑进主循环或阻塞渲染。
9. 禁止低置信度 / 透明 / 多层 / 极端遮挡时硬猜，必须走 degrade。
10. 禁止 README 实测表填未实测数字；留空标 TODO。
11. 禁止为「任意衣服」引入逐件人工建模（违反 C1）。
12. 禁止声称首次交互即真实、4K/60fps 与高细节布料同时成立。
13. 禁止 framing.verdict != OK 时提交全图换装或外推画外区域。
14. 禁止把 VTON 360 / TryOnCrafter 写成「摄像头实时任意视角」（D10）。
15. 禁止把 OmniTry 12 类写成无限品类、FASHN 三类写成无限品类（D11）。
16. 禁止在 HUD 省略取景引导文案；不合规必须给可执行提示。
17. 禁止在 src/ 下自行实现投影/四元数/SE(3)，统一走 math/。
18. 禁止把 force_proxy 作为真实力输入物理解算器。
19. 禁止用均值代替 p95；禁止单帧观感或 loss 单独下降作为验收。
20. 禁止同时读取 v1.1 与本文件；本文件为唯一权威。
```

---

## 18. 风险与降级矩阵

| 风险 | 检测量 | 降级动作 |
|---|---|---|
| 几何依赖装不上 | adapter 探测失败 | Stub 网格 + HUD「占位资产」 |
| 背面置信度低 | `conf.back_geometry<τ_back` | 「AI 补全」；可请求补背面图 |
| 拓扑/缝合失败 | `has_isolated_edge` / 自交数 | `needs_human_review=True`，不进物理层 |
| 材质不辨识 | `cond(𝓘)≥κ_max` | 保持 $\Theta_0$，请求更多激励 |
| 辨识不收敛 | 交互>50 且 `tr(P)` 不降 | `DEGRADED` |
| 几何—参数污染（D14） | 消融对照异常 | 冻结几何，单参数重估 |
| 物理发散 | `NaN` / $E$ 上升 | `E_PHYS_DIVERGE`，回退上一稳定状态 |
| 取景局部裁切 | `PARTIAL` | 只试可见区 + 引导文案，画外不外推 |
| 侧身/背面 | `facing!=frontal` | 异步 3D 渲染，标注 AI 补全 |
| 关键帧超时 | `p95>2000ms` | 显示旧帧 + 「处理中」 |
| 显存不足 | 峰值监控 | 降 `max_vertices` / 分辨率 / 关 3DGS，OmniTry 优先降级 |
| 许可冲突 | `license_gate` | `mode: real` 拒绝启动 |

---

## 19. 交付定义（DoD）

- [ ] `make mock-run` 无 GPU / 网络 / 权重 / GUI 连续跑 300 帧不崩
- [ ] `make test` 全绿（**12 个**测试文件）
- [ ] `README.md` 含三条命令、不可承诺清单、空置实测表
- [ ] `SPEC.md` 与 `CLAUDE.md` 在根目录，`CLAUDE.md` 指向本文件
- [ ] 五类 fps 分字段输出；HUD 恒显示 `MaterialState` + `FramingVerdict` + `guide_text`
- [ ] `math/` 是投影/四元数/SE(3) 唯一实现位置
- [ ] 所有外部依赖有 Stub / Real 双路径，`factory.build(cfg)` 可切换
- [ ] `policy/license_gate.py` 在 real 模式拦截 NC 权重
- [ ] 代码无任何「已实测」虚构数字
- [ ] 输出 **STRUCTURED_REPORT**（附录 G）

---

# 附录

## A · 符号速查

$\Delta t$ 帧间隔 · $\Pi_K$ 投影 · ${}^CT_W\in SE(3)$ · $\mathcal{A}$ 资产 · $V^0$ rest 顶点 · $\Theta\in\mathbb{R}^7$ · $S_k=(x_k,v_k,\Lambda_k)$ · $\mathcal{I}$ Fisher 信息 · $P$ 参数协方差 · $\kappa=\mathrm{cond}(\mathcal{I})$ · $H=0.4$s 预测窗 · $\tau_{\text{back}}=0.65$ · $\tau_{\text{vis}}=0.70$

## B · 单位与常量表（**唯一权威**）

| 量 | 单位 | 可行域 / 值 |
|---|---|---|
| 长度 / 顶点 $x$ | m | — |
| 速度 $v$ | m/s | — |
| 加速度 | m/s² | $\mathbf{g}=(0,-9.81,0)$ |
| 质量 / 面密度 $\rho_A$ | kg / kg·m⁻² | $[0.02,1.20]$ |
| 拉伸刚度 $E$ | N/m | $[10^2,10^6]$ |
| 弯曲刚度 $k_b$ | N·m | $[10^{-6},10^{-1}]$ |
| 泊松比 $\nu$ | — | $[0,0.49]$ |
| 摩擦 $\mu$ | — | $[0,1.5]$ |
| 阻尼 $c_d$ | N·s/m = kg/s | $[0,50]$ |
| 顶点质量 $m_i=\rho_A A_i$ | kg | $A_i$ 单位 m² |
| Rayleigh $\alpha_M$ / $\alpha_K$ | $\mathrm{s^{-1}}$ / s | 0.02 / 0.01；使 $D=\alpha_M M+\alpha_K K$ 量纲 kg/s |
| 能量衰减率 $\gamma$ | $\mathrm{s^{-1}}$ | 0.15；**禁止用 $c_d/\rho_A$ 代替** |
| 厚度 $h$ | m | $[10^{-4},5\cdot10^{-3}]$ |
| 能量 $E_k$ | J | 单调衰减 |
| 时间 | s | $\Delta t=1/30$，$\Delta t_s=1/120$ |
| 图像 | px | 1920×1080 |
| 角度/姿态 | rad | 单位四元数存储 |

## C · 错误码

```
E_MATH_QUAT / E_MATH_DEPTH      数学层
E_ASSET_GEOM_FAIL / E_ASSET_TIMEOUT / E_ASSET_LOW_CONF
E_ADAPTER_FALLBACK              依赖缺失回退 Stub（warning）
E_FRAMING_REJECT / E_FRAMING_PARTIAL
E_PHYS_DIVERGE / E_PHYS_PENETRATION
E_KF_TIMEOUT / E_KF_EXPIRED
E_CALIB_UNIDENTIFIABLE / E_CALIB_NO_EXCITATION / E_CALIB_GATE_REJECT
E_LICENSE_NC_BLOCKED
```

## D · 日志 schema（JSONL，每行一帧）

必含 `frame_id / pts_ms / asset_id / asset_version / timing{9字段} / framing{verdict,visible_keypoint_ratio,guide_text} / material_state / confidence.back_geometry / physics{attached_count,penetration_count,energy} / events[]`。禁止把 `generation_throughput_fps` 写入对外指标端点。

## E · 文件级验收命令映射

| 文件 | 命令 |
|---|---|
| `math/*` | `pytest tests/test_math.py -q` |
| `asset/*` | `pytest tests/test_asset_stub.py -q && make bootstrap-asset IMG=...` |
| `perception/framing.py` | `pytest tests/test_framing_gate.py -q` |
| `physics/*` | `pytest tests/test_physics_stub.py -q` |
| `calibration/*` | `pytest tests/test_calibration_gates.py tests/test_identifiability.py -q` |
| `keyframe/*` | `pytest tests/test_fusion.py tests/test_state_machines.py -q` |
| `policy/*` | `pytest tests/test_license_gate.py -q` |
| 全量 | `make test && make eval-report` |

## F · 歧义消解规则（复刻自 0.2）

```
附录 A/B > §2 数学定义 & §10 数值形式 > §8 函数签名 > §9 字段取值
  > 伪代码块(# IMPL) > §12 configs > 表格/散文 > 外部 README
```
**冲突时取左侧；实现者不得自行「取中间值」或「按常识调整」，必须在 STRUCTURED_REPORT 里列为 AMBIGUITY 并请求裁决。**

## G · STRUCTURED_REPORT（结束时必须输出）

```jsonc
{
  "spec_version": "v2.0",
  "mode": "mock",
  "files": [ {"path":"src/tryon/physics/solver.py","status":"STUB","symbols":["PhysicsSolver.step"]} ],
  "phases": [ {"phase":0,"accept_cmd":"...","result":"PASS"} ],
  "not_implemented": ["Real* adapters", "DiffCloth gradient", "3DGS renderer"],
  "ambiguities": [],
  "measured_numbers": [],          // 必须为空；禁止填虚构实测值
  "license_flags": ["CatVTON NC", "WiLoR ND", "smplx research-only"]
}
```

## H · 参考（可核查；数字均为公开口径，需实测）

- Street TryOn https://github.com/cuiaiyu/street-tryon-benchmark （>90% 非正面图筛除）
- RealVVT 附录 B https://ar5iv.labs.arxiv.org/html/2501.08682 （DensePose 肢体检测不全）
- Image-Based VTON Survey https://arxiv.org/html/2311.04811v3 （人体解析瓶颈）
- VTON 360 https://github.com/scnuhealthy/VTON360 ｜ TryOnCrafter https://sunhao242.github.io/TryOnCrafter_web.github.io/
- FASHN VTON v1.5 https://github.com/fashn-AI/fashn-vton-1.5 ｜ OmniTry https://github.com/random2fly/OmniTry
- AnyFit https://arxiv.org/html/2405.18172v1 ｜ Dress-1-to-3 https://dress-1-to-3.github.io/
- SewFormer https://github.com/sail-sg/sewformer ｜ DiffCloth https://github.com/omegaiota/DiffCloth
- XPBD (Macklin et al. 2016) ｜ Projective Dynamics (Bouaziz et al. 2014) ｜ Levenberg–Marquardt / EKF 标准形式

**本文件自包含；上述链接仅作背景，不覆盖本文任何约束。**
