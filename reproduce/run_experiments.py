"""Plan (default) or sequentially run the original experiments in a separate workspace."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(work, basins, train_python, plot_python, skip_figures):
    plan = []
    for basin in basins:
        data = ROOT/"data"/("gdp_hourly_"+basin)
        out = work/"results"/("ext_"+basin)
        region = ["--split-lon", "-45", "--lat-bands", "25", "33", "41", "50"] if basin == "na" else []
        split = ["--split-lon", "-45"] if basin == "na" else []
        specifications = [
            ("t2", "pilot_t1_t2.py", "t2", []),
            ("final", "paper_final_run.py", "final", ["--seeds", "3"] + region),
            ("conformal", "pilot_conformal.py", "conformal", ["--seeds", "3"] + region),
            ("ablation_A", "pilot_ablation.py", "ablation", ["--exp", "A", "--seeds", "2"] + split),
            ("ablation_B", "pilot_ablation.py", "ablation", ["--exp", "B", "--seeds", "2"] + split),
            ("ablation_C", "pilot_ablation.py", "ablation", ["--exp", "C", "--seeds", "3"] + split),
            ("oracle", "pilot_leak_oracle.py", "oracle", ["--seeds", "2"]),
            ("tiny", "tiny_data.py", "tiny", ["--draws", "5"]),
            ("compare", "compare_models.py", "compare", ["--seeds", "3"] + region),
        ]
        for name, script, folder, flags in specifications:
            cmd = [train_python, "-I", str(work/"scripts"/script), str(data), str(out/folder),
                   "--years", "2015", "2022"] + flags
            plan.append({"name": basin+"_"+name, "command": cmd})
    if not skip_figures:
        plan.append({"name": "figures", "command": [plot_python, "-I", str(work/"scripts/paper_figures.py")]})
        plan.append({"name": "table5", "command": [plot_python, "-I", str(ROOT/"reproduce/export_table5.py"),
                                                     "--results-root", str(work/"results"),
                                                     "--output-dir", str(work/"paper/tables")]})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--train-python", default=sys.executable)
    ap.add_argument("--plot-python", default=sys.executable)
    ap.add_argument("--basins", nargs="+", choices=["nwp", "na"], default=["nwp", "na"])
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Without this flag, only print the plan")
    a = ap.parse_args()
    work = a.workdir.resolve()
    marker = work/".reproduction_workspace.json"
    if not marker.exists() or json.loads(marker.read_text(encoding="utf-8"))["mode"] != "training":
        ap.error("Prepare a new --mode training workspace first")
    if len(set(a.basins)) != len(a.basins):
        ap.error("Each basin must be listed once")
    if not a.skip_figures and set(a.basins) != {"nwp", "na"}:
        ap.error("One-basin runs need --skip-figures; the original plotter requires both basins")
    plan = commands(work, a.basins, a.train_python, a.plot_python, a.skip_figures)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not a.execute:
        print("PLAN ONLY. Add --execute to start sequential training.")
        return
    for basin in a.basins:
        if (work/"results"/("ext_"+basin)).exists():
            ap.error("Experiment outputs already exist; use a new workspace to preserve prior results")
    for python in ({a.train_python} if a.skip_figures else {a.train_python, a.plot_python}):
        if not Path(python).is_file() and not shutil.which(python):
            ap.error("Python executable not found: " + python)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    for step in plan:
        print("START", step["name"], flush=True)
        subprocess.run(step["command"], cwd=str(work), env=environment, check=True)
        print("DONE", step["name"], flush=True)


if __name__ == "__main__":
    main()
