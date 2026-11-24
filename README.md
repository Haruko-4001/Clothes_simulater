# 任意衣服实时虚拟试衣 · Mock-first 参考实现

> 严格实现 `SPEC.md`（v2.0，学术增强版）。默认 `mode: mock`：**无 GPU / 无权重 / 无网络 / 无 GUI**，端到端主循环可连续跑 300 帧。
> 权威规格是 `SPEC.md`；本 README 只说明如何运行与边界，不覆盖规格。
> 产品定位、形态选择与商业化路线见 **[`PRODUCT.md`](PRODUCT.md)**。

## 产品形态（已落地）

这是一个**浏览器优先的「面料/版型可交互预览」服务**，不是试衣 App。启动：

```bash
make serve          # 或 python -m app.serve --physics-solver warp --render-backend opencv
# 打开 http://127.0.0.1:8000
```

- 实时 MJPEG 流（`/stream`）+ 状态面板（`/api/status`）
- 拖入衣服图片即换装（`/api/garment`，管线 A 同步重建资产）
- 能力自检（`/api/doctor` 与 `python -m app.cli doctor`）会逐项告诉你哪些后端在本机真的可用、哪些会回退 Stub

## 三条命令

```bash
# 1) 300 帧 mock 主循环（离线、无 GUI）
make mock-run

# 2) 管线 A：单张衣服图冷启动资产（G1 门禁 + 注册表）
make bootstrap-asset IMG=assets/samples/garment.png

# 3) 全量测试（SPEC §16 的 12 个规格文件 + 3 个回归文件）
make test
```

附加：

```bash
make eval-report     # 读取 var/logs/run.jsonl 生成 var/eval/report.{json,md}
make bench           # 引擎基准，产物 var/eval/bench.json
python -m app.cli selfcheck

# 可选引擎依赖（warp-lang + opencv-python-headless），缺失时自动回退 stub
python -m pip install -e ".[engine]"     # 或 make install-engine
# 可选感知依赖（MediaPipe；无 CPython 3.13 轮子，需 3.10–3.12）
python -m pip install -e ".[perception]"

# P0 引擎栈
python -m app.main --frames 300 --physics-solver warp --render-backend opencv --geometry silhouette
```

### 保证边界（必须读）

`无 GPU / 无权重 / 无网络` 是**默认 mock 配置**的保证。显式选择 `--physics-solver warp` 或 `render.backend: opencv` 会加载本地库并初始化 CUDA（若可用），这在 mock 模式下同样生效；但**新后端从不引入权重下载或网络访问**。FASHN 关键帧是唯一的联网路径，见下文。

**FASHN 前置条件**：`mode: real|hybrid` + `keyframe.backend: fashn` + API key（`configs/local.yaml`，已 git-ignore；或环境变量 `FASHN_API_KEY`）。启用后 `person_image` 与 `garment_image` 会被上传至 `api.fashn.ai`，本仓库不提供任何隐私或数据处理保证。

### Windows

仓库提供标准 `Makefile`（GNU make）。当前机器没有 GNU make 时，可使用：

```powershell
python scripts\make.py mock-run
python scripts\make.py bootstrap-asset IMG=assets\samples\garment.png
python scripts\make.py test
```

或把仓库根目录加入 PATH 后，`make.cmd` 提供同名 `make` 入口。

## 运行结构

```
管线 A（离线，非实时）
  I^g → 分割 → 几何冷启动 → 封闭双面物理壳 + 缝纫图案 + 碰撞体 + Θ₀
      → GarmentAsset v0（confidence / provenance / version）

管线 B（30 fps 主循环）
  Camera → Body/Hand → FramingGate → Contact → XPBD Physics → Mesh Render
         → 异步关键帧融合 → HUD → Display

管线 C（后台，独立进程）
  𝓓_{k-W:k} → 激励判别 → 公平差分雅可比 → LM/EKF → 三闸门 → COMMIT / KEEP_PRIOR / DEGRADE
```

- 管线 C 使用 `multiprocessing` 守护进程，主循环只提交/轮询；慢任务在退出时终止，绝不阻塞渲染。
- 关键帧使用有界队列（`max_queue=2`）的工作线程；只处理 `framing.verdict == OK` 的请求。
- `MaterialState != CALIBRATED` 时 HUD 明确显示「未校准」；背面置信度低于 `τ_back` 时显示「AI 补全」。
- mock 模式内部有一个**隐藏的合成真值材料**（`configs/mock.yaml: mock.true_material_preset`）用于生成观测，供管线 C 做合成辨识；它不是实测值，也不会出现在任何对外指标中。

## 目录

见 `SPEC.md` 第 4 节。核心入口：

- `src/tryon/math/`：投影 / 四元数 / SE(3) 的唯一实现位置
- `src/tryon/physics/`：半隐式欧拉 + XPBD + 接触/摩擦/能量钳位
- `src/tryon/calibration/`：残差、有限差分雅可比、LM、EKF 协方差、三闸门
- `src/tryon/policy/license_gate.py`：NC / ND / 研究许可拦截
- `app/main.py`：300 帧 mock 主循环；`app/cli.py`：bootstrap / eval / selfcheck
- `configs/`：所有阈值与数值参数；代码内不保存第二份阈值

## 五个帧率口径（禁止互推）

`camera_capture_fps`、`display_fps`、`physics_hz`、`keyframe_fps`、`generation_throughput_fps` 在 `TimingBook` 中独立记录。
延迟报 `Frame.pts_ms → 显示` 的 p50 / p95 / max，**不报均值**。
`generation_throughput_fps` 仅内部记录，不对外宣称，不等于试衣帧率。

## 「任意衣服」的精确边界

```
任意衣服 ≜ 无需 SKU 级人工登记，但存在类别先验 C_cls
```

- 免降级：`I^g` 属于训练覆盖的缝纫图案分布，且取景合规（正面、全身/半身、无大遮挡）。
- 降级集合：透明 / 多层嵌套 / 极端遮挡 / 非标准穿法 / 镜面 / 局部裁切 / 侧背面。必须走 `policy/degrade.py`，不得硬猜。
- 单图无法唯一确定背面、内侧、缝合次序、杨氏模量；资产是概率资产，必须携带 `confidence` 与 `provenance`。

## 不可承诺清单

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

## 许可证红线

- `CatVTON / IDM-VTON / OOTDiffusion / StableVITON / CausVid`：CC BY-NC-SA 4.0
- `WiLoR`：CC-BY-NC-ND；`SMPL-X / MANO`：非商用科研、需注册
- `RTV` 商用需书面授权；`SDXL-Turbo` 非商用
- 仓库 MIT/Apache **不等于**权重可商用。`mode: real` 时 `policy/license_gate.py` 会拦截；生产必须 `allow_nc_weights: false`。

## 实测数据表

> 目标硬件实测尚未进行。下面所有单元格保持 TODO；禁止填入任何未实测数字。

| 指标 | 目标 | 实测 |
|---|---|---|
| capture→display p50 / p95 | p95 ≤ 100 ms | TODO |
| display_fps | 30 | TODO |
| physics_hz | 60 | TODO |
| keyframe_fps | 0.2–1 | TODO |
| geometry IoU / CD / normal error | G1 IoU ≥ 0.90 | TODO |
| ADE@0.4s / FDE@0.4s | 低于未校准基线 | TODO |
| penetration count | 0 | TODO |
| ε_Θ（合成数据） | 下降 | TODO |
| peak VRAM | 依硬件档位 | TODO |

`make eval-report` 只在 `mock_run_counters` 中输出 mock 框架计数，并把目标硬件表留为 TODO。§7 的全部指标实现集中在 `src/tryon/eval/metrics.py`，由 `eval/report.py` 接线；调用方提供真值时经 `evaluate_ground_truth()` 计算并放入 `synthetic_validation`，`measured_numbers` 恒为空。

> 口径说明：mock 框架计数不构成 §2.1 的目标硬件性能声明。`display_fps` / `physics_hz` 在纯 Python stub 下不等于规格目标，按附录 F 不得自行调整，已作为 **AMBIGUITY** 登记在 `STRUCTURED_REPORT.json` 中请求裁决（mock 口径是否豁免）。

## 已实现 / 未实现（诚实声明）

**已实现（mock 路径）**：数学层、封闭双面壳几何与 G1/G2、**柱面 UV 贴图（花纹保真）**、XPBD 物理、接触/释放/回弹、四态取景门禁、异步关键帧与 6 帧融合、HUD、三状态机、有限差分雅可比、LM/EKF、三闸门、许可门禁、注册表版本/TTL、**Web 产品壳（实时流/状态/换衣）**、12 个规格测试文件 + 3 个回归文件、300 帧主循环。

**P0–P4 引擎升级（2026-11，本机已验证）**：

| 层 | 新后端 | 配置 / CLI | 状态 |
|---|---|---|---|
| 物理 | **Warp XPBD**（`physics/warp_solver.py`，CUDA 或 CPU，无权重无网络；可选 hash-grid 顶点自碰撞） | `physics.solver: warp` 或 `--physics-solver warp` | 已实现 + 测试 |
| 渲染 | **OpenCV 光栅化**（`render/cv2_render.py`，逐面 `fillConvexPoly` + AA，规避 fillPoly 的 even-odd 重叠空洞） | `render.backend: opencv` 或 `--render-backend opencv` | 已实现 + 测试 |
| 感知 | **MediaPipe Body/Hand**（33→17 关节映射、D12 时序平滑、单目深度先验为已声明局限） | `mode: real/hybrid`（mediapipe 缺失自动回退 Stub） | 已实现 |
| 关键帧 | **FASHN v1 REST 客户端**（提交+轮询+下载+差异掩膜；缺 API key 自动回退 Stub） | `keyframe.backend: fashn` + `FASHN_API_KEY` | 已实现 + mock-HTTP 测试 |
| 几何 | **轮廓驱动放样**（`asset/geometry.py::SilhouetteGeometryEstimator`，掩膜逐行宽度驱动壳体轮廓） | `asset.geometry: silhouette` | 已实现 + G1 IoU 测试 |
| 几何 | **版型冷启动**（`asset/pattern.py`：measurements → panels → seams；躯干+双袖，缝合线以 5× 刚度的 XPBD 距离约束实现） | `asset.geometry: pattern` 或 `--geometry pattern` | 已实现 + 3 项测试（袖长随肩宽行、缝合线存活、稳定性） |
| 贴图 | **柱面 UV + 逐面采样**（`asset/uv.py` + `render/texture.py`），花纹/条纹不再被平均成单色 | 自动生效 | 已实现 + 条纹保真测试 |
| 辨识 | **Warp 反传梯度**（`calibration/warp_grad.py`，1 次前向+反向替代 LM 内环 8 次前向差分） | `calibration.grad_backend: warp` | 已实现 + 方向一致性测试 |

全部新后端遵循既有 Protocol/工厂模式：依赖缺失或**初始化期运行时失败**时 `异常 → E_ADAPTER_FALLBACK → Stub`，主循环不会因某个后端不可用而崩溃。warp/cv2 为可选本地依赖，不含权重、不访问网络。

**本机实测基准**（RTX 5060 Laptop 8GB / warp-lang 1.19 / CUDA 13.4）。复现命令：

```bash
python scripts/bench_engines.py --out var/eval/bench.json     # 或 make bench
```

口径：固定 dt=1/60 s 连续步进，预热 20 步，**重复 3 次取中位数**（同时记录 min/max 以暴露主机波动）。这是**本机工程证据**，不是 §17.10 目标硬件实测表——那张表仍全为 TODO，`measured_numbers` 保持为空。

| 网格 (nu×nv) | 顶点 | numpy XPBD | Warp XPBD | 加速比 | Warp ms/步 |
|---|---|---|---|---|---|
| 12×9 | 240 | 369 步/s | **661 步/s** | 1.8× | 1.51 |
| 16×12 | 416 | 221 步/s | **470 步/s** | 2.1× | 2.13 |
| 28×20 | 1176 | 84 步/s | **324 步/s** | 3.9× | 3.09 |
| 40×35 | 2880 | 32 步/s | **257 步/s** | 8.1× | 3.89 |
| 56×52 | 5936 | 15 步/s | **190 步/s** | **12.6×** | **5.27** |

渲染（1080p、2352 面）：PIL 24.0 ms/帧 → OpenCV **10.3 ms/帧（2.3×）**。

**端到端 300 帧 A/B**（`[28,20]` 1176 顶点、1080p、`--no-calibration`，重复 3 次取中位数）：

| | 基线 numpy+PIL+stub | P0 栈 warp+cv2+silhouette | 变化 |
|---|---|---|---|
| display_fps | 13.7 | **25.5** | **1.86×** |
| p50 / p95 延迟 | 64.7 / 77.5 ms | **30.4 / 44.5 ms** | **−53% / −43%** |

复现：

```bash
python -m app.main --frames 300 --headless --log var/logs/ab.jsonl --no-calibration \
  --width 1920 --height 1080 --mesh-resolution 28 20
# 追加 --physics-solver warp --render-backend opencv --geometry silhouette 即 P0 栈
```

结论与边界：
- Warp 全规模领先（此前小网格更慢的问题已随每步固定开销的消除而消失）。在 `max_vertices: 6000` 预算点（5936 顶点）为 **12.6×，5.27 ms/步**——`190 步/s` 是 60 Hz 需求的 3.2 倍；相对 SPEC §8 的「≤5 ms @ N=6000」仍高约 5%，这一差额如实保留、未用调参抹平。
- 端到端 p95 从 77.5 ms 降到 44.5 ms，**在 ≤100 ms 界内且有 55% 余量**；display_fps 25.5 仍略低于 30 目标，剩余瓶颈是 1080p 合成相机与 HUD（各约 7–12 ms），已不属于物理/渲染引擎层。
- 数字随主机负载波动（同配置重复运行可见 ±30%），因此以中位数 + min/max 呈现，并附脚本与产物 `var/eval/bench.json`。
- 本轮修掉并防回归的两个真实性能缺陷（均有测试锁定）：
  1. 能量钳位曾用 40 步二分，每步调用 `elastic_energy` 43 次、占约 85% 物理时间 → 改为闭式二次型 + 预分配缓冲的单次向量化搜索（`_contract_scale`），最终能量仍由 float64 精确复核，§7 能量不变式不变。
  2. 每步重复分配 10 个设备数组、每轮迭代额外一次清零内核 → 改为持久缓冲 + `_apply_delta_kernel` 内联清零。

**未实现（Real 适配器，均回退 Stub 并记 `E_ADAPTER_FALLBACK`）**：SewFormer / GarmentRec / InstantMesh 真实几何、SMPL-X β/θ 网格拟合、WiLoR / HaMeR 手部、DiffCloth 求解器（管线 C 已由 warp_grad 提供反传梯度）、3DGS 真实渲染、本地 CatVTON / IDM-VTON / OmniTry / Leffa（NC 许可，未捆绑）、VTON 360 / TryOnCrafter（代码未公开/仅设计参考）、点-三角自碰撞（现为顶点级 hash-grid 近似，`physics.self_collision` 门控）。逐文件状态、各 Phase 验收结果、歧义裁决请求与许可证标记见 **[`STRUCTURED_REPORT.json`](STRUCTURED_REPORT.json)**（SPEC 附录 G 要求的交付物）。

**真实接入**：

```bash
bash scripts/install_real.sh --only=engine        # 本地引擎依赖（warp-lang + opencv）
bash scripts/install_real.sh --only=perception    # MediaPipe（需 Python < 3.13）
bash scripts/install_real.sh --only=fashn_vton    # 可选，需要网络
# 权重获取见 scripts/fetch_weights.sh；不得把任何权重路径设为必需
```

## 测试

```bash
make test
```

SPEC §16 / §19 / 附录 E 规定 **12 个规格测试文件**，数量不变：

`test_math / test_types / test_asset_stub / test_physics_stub / test_state_machines / test_framing_gate / test_calibration_gates / test_identifiability / test_fps_bookkeeping / test_fusion / test_license_gate / test_end_to_end_mock`

另有 **3 个额外回归文件**（不属于 §16 的规格矩阵，已在 `STRUCTURED_REPORT.json` 的 `deviations` 中登记）：

`test_json_safety`（严格 JSON 输出）/ `test_make_runner`（Windows 入口参数透传）/ `test_engine_adapters`（P0–P4 引擎后端：Warp 物理、OpenCV 渲染、MediaPipe 追踪、FASHN 客户端、轮廓几何、Warp 反传梯度）

## 逐 Phase 验收（无 GNU make / Windows）

SPEC §15 的 `ACCEPT_CMD` 使用 `make`。本机没有 GNU make 时，用 `scripts/make.py` 等价执行；下表逐 Phase 给出等价命令。**`make.py mock-run` 会覆盖已跟踪的 `var/logs/run.jsonl`，验收时务必透传 `--log` 指向仓库外。**

| Phase | 等价命令 |
|---|---|
| 0 | `python scripts\make.py test` 后 `python -m app.main --mode mock --frames 300 --headless --log <外部路径>` |
| 1 | `python scripts\make.py bootstrap-asset IMG=assets\samples\garment.png` |
| 2 | `python -m pytest tests/test_physics_stub.py tests/test_framing_gate.py tests/test_state_machines.py -q` |
| 3 | `python -m pytest tests/test_calibration_gates.py tests/test_identifiability.py -q` |
| 4 | `python -m pytest tests/test_fps_bookkeeping.py tests/test_fusion.py -q` |
| 5 | `python scripts\make.py test` 后 `python -m app.cli eval-report` |

`make.cmd` 提供同名入口（要求 `python` 指向装齐依赖的解释器）。当前环境的 Bash `python` 默认指向不含 pytest 的受管 Python，需把系统 Python 目录前置到 PATH。

## 日志

`var/logs/run.jsonl` 每行一帧，含 `frame_id / pts_ms / asset_id / asset_version / timing(9 字段) / framing / material_state / confidence / physics / events`。
`generation_throughput_fps` 不得写入任何对外指标端点。

## 引用与背景

- Street TryOn (WACV 2025)、RealVVT 附录 B、Image-Based VTON Survey、VTON 360、TryOnCrafter、FASHN VTON v1.5、OmniTry、SewFormer、DiffCloth、XPBD (Macklin et al. 2016)、Projective Dynamics (Bouaziz et al. 2014)、Levenberg–Marquardt / EKF 标准形式。
- 链接与许可判定以 `SPEC.md` 第 13 节与附录 H 为准；外部 README 不覆盖本仓库约束。
