# Surface Drifter Trajectory Prediction: Data, Code and Results

This package accompanies the manuscript `main.pdf`. It contains the NOAA Global Drifter Program hourly data, the experiment code, the final results and entry points for reproducing the figures and tables. You can inspect the published results directly, regenerate the figures, or retrain everything from the raw data in a single Python 3.10 environment.

To avoid overwriting the final results, all re-run outputs should be written to a newly created `work/` subdirectory.

## Quick Start

| Goal | Entry point | Output |
|---|---|---|
| View saved results | Open `main.pdf`, `paper/figures/` and `paper/tables/` | No environment needed |
| Regenerate figures and tables from saved results | Set up the environment, then run the three commands in "Regenerating Figures and Tables" | `work/replot/paper/` |
| Retrain from raw data | Preview the experiment plan first, then add `--execute` | `work/retrain/results/` |

## Environment

Training and plotting use the same `drift_repro_py310` environment. Reserve at least 20 GB of disk space; the full set of experiments should be run sequentially.

Main package versions are listed below; `env/requirements.txt` is the authoritative list of dependencies.

| Component | Version |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.4.1 (tested on Windows GPU as 2.4.1+cu121) |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| scikit-learn / pandas | 1.3.2 / 2.0.3 |
| Matplotlib / Cartopy | 3.7.5 / 0.22.0 |

### GPU Environment

From the package root, run:

```powershell
conda env create -f env/environment.yml
conda activate drift_repro_py310
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r env/requirements.txt
python -m pip check
```

`env/requirements-win-cu121.lock.txt` pins every package version of the tested Windows CUDA 12.1 environment. To reproduce that configuration exactly, run the following after installing the CUDA build of PyTorch above:

```powershell
python -m pip install -r env/requirements-win-cu121.lock.txt
python -m pip check
```

### CPU Environment

Without a compatible NVIDIA GPU, replace the PyTorch installation command with:

```powershell
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r env/requirements.txt
```

Everything runs on CPU, but full training is considerably slower. Do not use the CUDA lock file in a CPU environment.

To check the interpreter and compute device:

```powershell
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The first time Figure 1 is drawn, Cartopy needs an internet connection to download the Natural Earth 50m base map; later runs use the local cache.

## Regenerating Figures and Tables

With the environment activated, regenerate Figures 1 and 3–7 and Tables 1–5 from the saved results:

```powershell
python reproduce/prepare_workspace.py work/replot --mode figures
python work/replot/scripts/paper_figures.py
python reproduce/export_table5.py --results-root work/replot/results --output-dir work/replot/paper/tables
```

Outputs are written to `work/replot/paper/figures/` and `work/replot/paper/tables/`. Figure 2 is the final framework diagram, drawn in Visio and provided as a PDF.

The preparation script performs the following steps in the working copy, without modifying the original scripts or the final results in the package:

- copies the results needed for plotting and creates an identical-content alias of the NWP calibration file under the name the plotting script expects;
- creates a `paper/main.tex` used only to receive the auto-generated table snippets;
- syncs the in-figure text with the final manuscript;
- applies a Python 3.10 syntax-compatibility fix to `paper_figures.py`, without changing any numerical computation.

The target directory must not already exist. To run again, use a new directory name or move the existing working directory out of the way first.

Exporting Table 5 alone requires no scientific dependencies:

```powershell
python reproduce/export_table5.py
```

The output goes to `work/table5/` by default.

## Retraining from Raw Data

First create a training workspace and preview the experiment plan:

```powershell
python reproduce/prepare_workspace.py work/retrain --mode training
python reproduce/run_experiments.py --workdir work/retrain
```

Once the plan looks right, add `--execute` to run the experiments sequentially:

```powershell
python reproduce/run_experiments.py --workdir work/retrain --execute
```

The following experiments are run for each basin, all using data from 2015–2022:

| Experiment | Main script / parameters | Results subdirectory |
|---|---|---|
| Input velocities and physical baselines | `pilot_t1_t2.py` | `t2/` |
| Main model | `paper_final_run.py --seeds 3` | `final/` |
| Calibration | `pilot_conformal.py --seeds 3` | `conformal/` |
| Ablations A/B/C | 2/2/3 seeds, respectively | `ablation/` |
| Future-position sensitivity | `pilot_leak_oracle.py --seeds 2` | `oracle/` |
| Very-small-data (number of drifters) | `tiny_data.py --draws 5` | `tiny/` |
| Model comparison | `compare_models.py --seeds 3` | `compare/` |

The runner automatically adds the longitude split and latitude-band arguments required for the North Atlantic experiments. To run the Northwest Pacific first on its own, use:

```powershell
python reproduce/run_experiments.py --workdir work/retrain --basins nwp --skip-figures --execute
```

Single-basin runs must use `--skip-figures`, because the plotting script needs results from both basins. After a full retraining, the runner continues with plotting and exports Table 5.

Data are always read from `data/` inside the package and new results are written to the workspace; the scripts refuse to overwrite existing experiment outputs. Randomness and hardware differences may cause small numerical deviations.

To run the main model on its own:

```powershell
python -I work/retrain/scripts/paper_final_run.py data/gdp_hourly_nwp work/retrain/results/ext_nwp/final --years 2015 2022 --seeds 3
python -I work/retrain/scripts/paper_final_run.py data/gdp_hourly_na work/retrain/results/ext_na/final --years 2015 2022 --seeds 3 --split-lon -45 --lat-bands 25 33 41 50
```

The shorter-period pilot uses the same script with `--years 2020 2022 --seeds 3` for the NWP, writing to `results/paper_final/` in a new workspace.

## Data and Methods

The data are from the NOAA Global Drifter Program hourly dataset, version 2.01 (DOI: <https://doi.org/10.25921/x46c-3620>). They are included in the package and do not need to be downloaded again.

- Northwest Pacific (NWP): 20–45°N, 120–180°E;
- North Atlantic (NA): 25–50°N, 80–10°W;
- each basin contains 94 monthly files, from January 2015 to October 2022;
- GPS filtering uses `data/meta/gdp_v2.01.1_location_type.csv`.

Forecasts are issued every 6 h. The input is the 48-h history sampled every 3 h (17 time steps), and the targets are the displacements at 6, 12, 24, 48 and 72 h. Windows containing fix gaps longer than 6 h are discarded. The main model uses historical positions and backward-difference velocities, among other features; the product velocities `ve/vn` are used only in the input-velocity comparison.

Training, early-stopping, calibration and test sets are separated by forecast issue year, with 2022 as the test year. The split is not fully disjoint by drifter ID (drifters alive across the year boundary contribute to both periods), which should be kept in mind when interpreting generalisation results.

Main implementation files:

| File | Contents |
|---|---|
| `pilot_t1_t2.py` | Data loading, window construction, input velocities and physical baselines |
| `pilot_ou_lstm.py` | Features, OU prior, LSTM, probabilistic loss and ensembling |
| `paper_final_run.py` | Main experiment, temporal split and per-window prediction output |
| `pilot_conformal.py` | Static, rolling and group-conditional (Mondrian) calibration |
| `compare_models.py` | Comparison with alternative models |
| `pilot_ablation.py` | Feature ablation, spatial transfer and small-data experiments |
| `tiny_data.py` | Repeated-draw experiments with 1/2/4/8 drifters |
| `pilot_leak_oracle.py` | Sensitivity experiment with future positions added to the input |
| `paper_figures.py` | Generates the manuscript figures and tables from saved results |

## Figures, Tables and Result Files

In the table below, `R` stands for `results/ext_nwp` or `results/ext_na`.

| Manuscript item | Main inputs | Script and output |
|---|---|---|
| Table 1: data summary | `data/`, `R/final/final_predictions.npz` | `paper_figures.py` → `data_summary.tex` |
| Table 2: main results | `R/final/final_predictions.npz` | `paper_figures.py` → `main_results.tex` |
| Table 3: model comparison | `R/compare/` and the main-experiment NPZ | `compare_models.py`, `paper_figures.py` → `comparison.tex` |
| Table 4: 72-h calibration | `R/conformal/*.csv` | `pilot_conformal.py`, `paper_figures.py` → `calibration.tex` |
| Table 5: 24-h coverage and area | Calibration CSVs | `export_table5.py` → `table5_24h.tex/.json` |
| Figure 1: study area and trajectories | `track_*` arrays in the main-experiment NPZ | `fig1_study_area.pdf/.png` |
| Figure 2: framework | `fig2_framework_v8.pdf` | Final PDF copied as is |
| Figure 3: input velocity and future-position sensitivity | `R/t2/`, `R/oracle/` | `fig3_leakage.pdf/.png` |
| Figure 4: accuracy and improvement by group | Main-experiment NPZ | `fig4_accuracy.pdf/.png` |
| Figure 5: small data and spatial transfer | `R/tiny/`, `R/ablation/results_B.csv` | `fig5_prior_role.pdf/.png` |
| Figure 6: coverage and search-region area | Probabilistic parameters from the main-experiment NPZ, calibration CSVs | `fig6_calibration.pdf/.png` |
| Figure 7: example forecasts | History, future trajectories and probabilistic forecasts | `fig7_cases.pdf/.png` |

The NWP calibration file is named `conformal_results_nwp.csv` and the NA file `conformal_results.csv`; the preparation script handles the file-name difference expected by the plotting script. The TeX files for Tables 1–4 are table-body snippets, while Table 5 is a complete `tabular` that can be inserted directly into the manuscript.

The main prediction NPZ files store, for every window, the targets, probabilistic forecasts, calibration thresholds, group labels and example trajectories; they do not contain model weights. In these files, `y` is the displacement target in local coordinates, `HORIZONS` the forecast horizons, `mu/sx/sy/rho` the Gaussian forecast parameters and `q_rolling/q_mondrian` the calibration thresholds.

## Saved Results and Scope of Reproduction

| Basin | Test windows | OU+LSTM 24-h error (km) | 72-h error (km) | 72-h coverage (rolling Mondrian) |
|---|---:|---:|---:|---:|
| NWP | 56,539 | 10.3036 | 40.3112 | 0.899715 |
| NA | 192,841 | 7.5369 | 28.7684 | 0.900332 |

The final results include per-window predictions, from which the main metrics can be recomputed. For the alternative models and some transfer experiments only means or summary metrics are saved; use them together with the corresponding CSV files.

## Directory Structure

```text
main.pdf                   Manuscript
data/gdp_hourly_nwp/       Northwest Pacific monthly NetCDF files (94)
data/gdp_hourly_na/        North Atlantic monthly NetCDF files (94)
data/meta/                 GPS/Argos location-type metadata
scripts/                   Method, experiment and plotting code
results/ext_nwp/           Northwest Pacific final results
results/ext_na/            North Atlantic final results
results/paper_final/       2020–2022 pilot predictions
paper/figures/             Final PDFs of manuscript Figures 1–7
paper/tables/              Tables 1–5 and unrounded Table 5 JSON
env/                       Environment files and Windows CUDA lock file
reproduce/                 Workspace preparation, experiment runner and Table 5 export
```
