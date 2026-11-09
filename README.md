# 任意衣服实时虚拟试衣 · Mock-first 参考实现

> 严格实现 `SPEC.md`（v2.0，学术增强版）。默认 `mode: mock`：**无 GPU / 无权重 / 无网络 / 无 GUI**，端到端主循环可连续跑 300 帧。
> 权威规格是 `SPEC.md`；本 README 只说明如何运行与边界，不覆盖规格。

## 三条命令

```bash
# 1) 300 帧 mock 主循环（离线、无 GUI）
make mock-run

# 2) 管线 A：单张衣服图冷启动资产（G1 门禁 + 注册表）
make bootstrap-asset IMG=assets/samples/garment.png

# 3) 全量测试（SPEC §16 的 12 个规格文件 + 2 个回归文件）
make test
```

附加：

```bash
make eval-report     # 读取 var/logs/run.jsonl 生成 var/eval/report.{json,md}
python -m app.cli selfcheck
```

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

**已实现（mock 路径）**：数学层、封闭双面壳几何与 G1/G2、XPBD 物理、接触/释放/回弹、四态取景门禁、异步关键帧与 6 帧融合、HUD、三状态机、有限差分雅可比、LM/EKF、三闸门、许可门禁、注册表版本/TTL、12 个测试文件、300 帧主循环。

**未实现（Real 适配器，均回退 Stub 并记 `E_ADAPTER_FALLBACK`）**：SewFormer / GarmentRec / InstantMesh 真实几何、MediaPipe / SMPL-X 真实人体与手部、DiffCloth 可微梯度、3DGS 真实渲染、FASHN / OmniTry / CatVTON 真实生成、VTON 360 / TryOnCrafter（代码未公开/仅设计参考）。逐文件状态、各 Phase 验收结果、歧义裁决请求与许可证标记见 **[`STRUCTURED_REPORT.json`](STRUCTURED_REPORT.json)**（SPEC 附录 G 要求的交付物）。

**真实接入**：

```bash
bash scripts/install_real.sh --only=fashn_vton   # 可选，需要网络
# 权重获取见 scripts/fetch_weights.sh；不得把任何权重路径设为必需
```

## 测试

```bash
make test
```

SPEC §16 / §19 / 附录 E 规定 **12 个规格测试文件**，数量不变：

`test_math / test_types / test_asset_stub / test_physics_stub / test_state_machines / test_framing_gate / test_calibration_gates / test_identifiability / test_fps_bookkeeping / test_fusion / test_license_gate / test_end_to_end_mock`

另有 **2 个额外回归文件**（不属于 §16 的规格矩阵，已在 `STRUCTURED_REPORT.json` 的 `deviations` 中登记）：

`test_json_safety`（严格 JSON 输出）/ `test_make_runner`（Windows 入口参数透传）

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
