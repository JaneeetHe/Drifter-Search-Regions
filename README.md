# 漂流轨迹预测：数据、代码与实验结果

本包对应手稿 `main.pdf`，包含 NOAA Global Drifter Program 逐时数据、实验代码、正式结果及图表复现入口。可以直接查看论文结果，也可以在统一的 Python 3.10 环境中重新绘图或从原始数据完整训练。

为避免覆盖正式结果，所有重新运行的输出均应写入新建的 `work/` 子目录。

## 快速开始

| 目标 | 入口 | 输出 |
|---|---|---|
| 查看已保存结果 | 打开 `main.pdf`、`paper/figures/` 和 `paper/tables/` | 无需安装环境 |
| 从保存结果重画图表 | 配置环境后运行“重新生成图表”中的3条命令 | `work/replot/paper/` |
| 从原始数据完整重训 | 先预览实验计划，再加入 `--execute` | `work/retrain/results/` |

## 统一环境

训练和绘图使用同一个 `drift_repro_py310` 环境。建议预留至少 20 GB 磁盘空间；完整实验应顺序执行。

主要版本如下，完整依赖以 `env/requirements.txt` 为准：

| 组件 | 版本 |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.4.1；Windows GPU 实测为 2.4.1+cu121 |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| scikit-learn / pandas | 1.3.2 / 2.0.3 |
| Matplotlib / Cartopy | 3.7.5 / 0.22.0 |

### GPU 环境

在包根目录执行：

```powershell
conda env create -f env/environment.yml
conda activate drift_repro_py310
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r env/requirements.txt
python -m pip check
```

`env/requirements-win-cu121.lock.txt` 是 Windows CUDA 12.1 实测环境的完整版本锁定。需要严格采用该配置时，在安装上述 CUDA 版 PyTorch 后执行：

```powershell
python -m pip install -r env/requirements-win-cu121.lock.txt
python -m pip check
```

### CPU 环境

没有兼容的 NVIDIA GPU 时，将 PyTorch 安装命令替换为：

```powershell
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r env/requirements.txt
```

CPU 可以运行，但完整训练会明显更慢；不要在 CPU 环境中使用 CUDA 锁定文件。

可用以下命令检查解释器和计算设备：

```powershell
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Cartopy 首次绘制图1时需要联网下载 Natural Earth 50m 底图，后续使用本地缓存。

## 重新生成图表

在已激活的统一环境中，从保存结果重画图1、3–7及表1–5：

```powershell
python reproduce/prepare_workspace.py work/replot --mode figures
python work/replot/scripts/paper_figures.py
python reproduce/export_table5.py --results-root work/replot/results --output-dir work/replot/paper/tables
```

结果位于 `work/replot/paper/figures/` 和 `work/replot/paper/tables/`。图2是最终方法框架 PDF，采用visio绘制。

准备程序会在工作副本中完成以下操作，不修改包内原始脚本和正式结果：

- 复制绘图所需结果，并为 NWP 校准文件创建绘图入口所需的同内容别名；
- 创建仅用于接收自动表格片段的 `paper/main.tex`；
- 同步最终手稿中的图中文字；
- 应用 `paper_figures.py` 的 Python 3.10 语法兼容修复，不改变数值计算。

目标目录必须是尚不存在的新目录。如果需要再次运行，请改用新的目录名或先妥善处理已有工作目录。

只导出表5无需安装科学计算依赖：

```powershell
python reproduce/export_table5.py
```

默认输出到 `work/table5/`。

## 从数据完整重训

先创建训练工作区并预览实验计划：

```powershell
python reproduce/prepare_workspace.py work/retrain --mode training
python reproduce/run_experiments.py --workdir work/retrain
```

确认计划后加入 `--execute` 开始顺序执行：

```powershell
python reproduce/run_experiments.py --workdir work/retrain --execute
```

每个海域运行以下实验，数据年份统一为 2015–2022：

| 实验 | 主要参数 | 结果子目录 |
|---|---|---|
| 速度与物理基线 | `pilot_t1_t2.py` | `t2/` |
| 主模型 | `paper_final_run.py --seeds 3` | `final/` |
| 校准 | `pilot_conformal.py --seeds 3` | `conformal/` |
| 消融 A/B/C | 种子数分别为 2/2/3 | `ablation/` |
| 未来位置敏感性 | `pilot_leak_oracle.py --seeds 2` | `oracle/` |
| 浮标数小样本 | `tiny_data.py --draws 5` | `tiny/` |
| 模型比较 | `compare_models.py --seeds 3` | `compare/` |

辅助入口已自动加入北大西洋实验所需的经度分界和纬度带参数。若只想先运行西北太平洋，可使用：

```powershell
python reproduce/run_experiments.py --workdir work/retrain --basins nwp --skip-figures --execute
```

单海域运行必须配合 `--skip-figures`，因为原绘图程序需要两个海域的结果。完整重训结束后，入口会继续绘图并导出表5。

数据始终从包内 `data/` 读取，新结果写入工作区；程序拒绝覆盖已有实验输出。随机性和硬件差异可能造成小幅数值变化。

单独运行主模型的命令为：

```powershell
python -I work/retrain/scripts/paper_final_run.py data/gdp_hourly_nwp work/retrain/results/ext_nwp/final --years 2015 2022 --seeds 3
python -I work/retrain/scripts/paper_final_run.py data/gdp_hourly_na work/retrain/results/ext_na/final --years 2015 2022 --seeds 3 --split-lon -45 --lat-bands 25 33 41 50
```

较短年份试点使用同一脚本，将 NWP 年份设为 `--years 2020 2022 --seeds 3`，并输出到新工作区的 `results/paper_final/`。

## 数据与方法

数据来自 NOAA Global Drifter Program hourly v2.01，DOI：<https://doi.org/10.25921/x46c-3620>。数据已包含在包内，无需重新下载。

- 西北太平洋（NWP）：20–45°N、120–180°E；
- 北大西洋（NA）：25–50°N、80–10°W；
- 两个海域均包含 2015年1月至2022年10月的 94 个月度文件；
- GPS 筛选使用 `data/meta/gdp_v2.01.1_location_type.csv`。

模型每6小时起报，输入48小时历史并按3小时采样，共17步；预测 6/12/24/48/72 小时位移。定位间隔超过6小时的窗口会被剔除。主模型使用历史位置和后向差分速度等特征，产品自带的 `ve/vn` 仅用于速度对比实验。

训练、早停、校准和测试按起报年份划分，测试年为2022年。划分并非全部按浮标 ID 互斥，解读泛化结果时应注意这一点。

主要实现文件：

| 文件 | 内容 |
|---|---|
| `pilot_t1_t2.py` | 数据读取、窗口构建、速度与物理基线 |
| `pilot_ou_lstm.py` | 特征、OU先验、LSTM、概率损失及集成 |
| `paper_final_run.py` | 主实验、时间划分及逐样本预测保存 |
| `pilot_conformal.py` | 静态、滚动及分组校准 |
| `compare_models.py` | 替代模型比较 |
| `pilot_ablation.py` | 特征消融、区域迁移及小样本实验 |
| `tiny_data.py` | 1/2/4/8 个浮标的重复抽样实验 |
| `pilot_leak_oracle.py` | 加入未来位置的敏感性实验 |
| `paper_figures.py` | 从保存结果生成论文图表 |

## 图表与结果文件

下表中的 `R` 表示 `results/ext_nwp` 或 `results/ext_na`。

| 手稿内容 | 主要输入 | 生成入口与输出 |
|---|---|---|
| 表1：数据规模 | `data/`、`R/final/final_predictions.npz` | `paper_figures.py` → `data_summary.tex` |
| 表2：主模型 | `R/final/final_predictions.npz` | `paper_figures.py` → `main_results.tex` |
| 表3：模型比较 | `R/compare/`及主实验 NPZ | `compare_models.py`、`paper_figures.py` → `comparison.tex` |
| 表4：72h校准 | `R/conformal/*.csv` | `pilot_conformal.py`、`paper_figures.py` → `calibration.tex` |
| 表5：24h覆盖率与面积 | 校准 CSV | `export_table5.py` → `table5_24h.tex/.json` |
| 图1：研究区轨迹 | 主实验 NPZ 中的 `track_*` | `fig1_study_area.pdf/.png` |
| 图2：方法框架 | `fig2_framework_v8.pdf` | 直接复制最终 PDF |
| 图3：速度与未来位置敏感性 | `R/t2/`、`R/oracle/` | `fig3_leakage.pdf/.png` |
| 图4：精度与分组改善 | 主实验 NPZ | `fig4_accuracy.pdf/.png` |
| 图5：小样本与迁移 | `R/tiny/`、`R/ablation/results_B.csv` | `fig5_prior_role.pdf/.png` |
| 图6：覆盖率与搜索面积 | 主实验概率参数、校准 CSV | `fig6_calibration.pdf/.png` |
| 图7：预测案例 | 历史、未来轨迹及概率预测 | `fig7_cases.pdf/.png` |

NWP 校准文件名为 `conformal_results_nwp.csv`，NA 为 `conformal_results.csv`；准备程序会处理绘图入口所需的文件名差异。表1–4的 TeX 文件是表格数据片段，表5是可直接插入论文的 `tabular`。

主预测 NPZ 保存逐样本目标、概率预测、校准阈值、分组和案例轨迹，不包含模型权重。其中 `y` 为局地位移目标，`HORIZONS` 为预测时效，`mu/sx/sy/rho` 为高斯预测参数，`q_rolling/q_mondrian` 为校准阈值。

## 保存结果与复现范围

| 海域 | 测试窗口 | OU+LSTM 24h误差/km | 72h误差/km | 72h分组校准覆盖率 |
|---|---:|---:|---:|---:|
| NWP | 56,539 | 10.3036 | 40.3112 | 0.899715 |
| NA | 192,841 | 7.5369 | 28.7684 | 0.900332 |

正式结果包含逐样本预测，可用于重新计算主要指标。替代模型和部分迁移实验仅保存均值或汇总指标，应结合相应 CSV 使用。

## 目录结构

```text
main.pdf                   手稿
data/gdp_hourly_nwp/       西北太平洋逐月 NetCDF 数据（94个）
data/gdp_hourly_na/        北大西洋逐月 NetCDF 数据（94个）
data/meta/                 GPS/Argos 定位类型元数据
scripts/                   方法、实验和绘图代码
results/ext_nwp/           西北太平洋正式结果
results/ext_na/            北大西洋正式结果
results/paper_final/       2020–2022年试点预测
paper/figures/             手稿图1–7最终 PDF
paper/tables/              表1–5及表5未舍入 JSON
env/                       统一环境及 Windows CUDA 锁定文件
reproduce/                 工作区准备、实验运行和表5导出入口
```
