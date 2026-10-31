"""Export current-manuscript Table 5 from existing conformal CSVs; standard library only."""
from pathlib import Path
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [("In-basin", "in-region (all NWP)"), ("West to East", "cross-region W->E"), ("East to West", "cross-region E->W")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=ROOT/"results")
    ap.add_argument("--output-dir", type=Path, default=ROOT/"work/table5")
    a = ap.parse_args()
    rows = []
    for basin in ["nwp", "na"]:
        p = a.results_root/("ext_"+basin)/"conformal/conformal_results.csv"
        if not p.exists() and basin == "nwp":
            p = p.with_name("conformal_results_nwp.csv")
        with p.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        for label, tag in SETTINGS:
            get = lambda model: next(r for r in records if r["tag"] == tag and r["model"] == model and r["calib"] == "M-lat×drg")
            hybrid, ou = get("LSTM + OU (ens)"), get("OU prior")
            rows.append({"basin": basin.upper(), "setting": label,
                         "hybrid_coverage": float(hybrid["cov24"]), "hybrid_area_km2": float(hybrid["area24"]),
                         "ou_coverage": float(ou["cov24"]), "ou_area_km2": float(ou["area24"]),
                         "area_reduction_percent": 100*(1-float(hybrid["area24"])/float(ou["area24"]))})
    tex = ["% Table 5: 24-hour calibrated coverage and search area.",
           r"\begin{tabular}{llrrrrr}", r"\hline",
           r"Basin & Setting & Hybrid coverage & Hybrid area & OU coverage & OU area & Reduction \\",
           r"\hline"]
    for r in rows:
        tex.append("{} & {} & {:.2f} & {:.0f} & {:.2f} & {:.0f} & {:.0f}\\% \\\\".format(
            r["basin"], r["setting"], r["hybrid_coverage"], r["hybrid_area_km2"], r["ou_coverage"], r["ou_area_km2"], r["area_reduction_percent"]))
    tex += [r"\hline", r"\end{tabular}"]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/"table5_24h.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    (a.output_dir/"table5_24h.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
