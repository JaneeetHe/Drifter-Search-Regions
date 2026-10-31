"""Create a separate workspace; never edit the packaged reference scripts/results."""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(dest, mode):
    dest = dest.resolve()
    allowed_parent = ROOT / "work"
    if dest == ROOT or dest in ROOT.parents or (ROOT in dest.parents and allowed_parent not in dest.parents):
        raise ValueError("Inside the package, choose a new directory under work/")
    if dest.exists():
        raise FileExistsError("Destination exists; choose a new directory: " + str(dest))
    dest.mkdir(parents=True)
    for name in ["scripts", "paper"]:
        shutil.copytree(str(ROOT/name), str(dest/name), ignore=shutil.ignore_patterns("__pycache__"))
    # Synchronize final-manuscript wording in the working copy only.
    # Numerical computation and the packaged original plotter remain unchanged.
    plotter = dest/"scripts/paper_figures.py"
    source = plotter.read_text(encoding="utf-8")
    labels = [
        ("Hours of true future positions given", "Hours of future positions given"),
        ("Upper bound on smoothing leakage\\n(oracle inputs)",
         "Sensitivity to added future\\npositions (oracle inputs)"),
        ("(10³ km², equal coverage)", "(10³ km²)"),
    ]
    for old, new in labels:
        if source.count(old) != 1:
            raise ValueError("Expected one original figure label: " + old)
        source = source.replace(old, new)
    # Python 3.10 does not allow a backslash inside an f-string expression.
    # Keep the original source in the package; fix only this working copy.
    lines = source.splitlines()
    matches = [i for i, line in enumerate(lines)
               if "labels.append(" in line and "tname.replace(" in line]
    if len(matches) != 1:
        raise ValueError("Expected one original f-string compatibility edit")
    lines[matches[0]:matches[0]+1] = [
        '        display_name = tname.replace(r"$\\to$", "→")',
        '        labels.append(f"{k}\\n{display_name}")',
    ]
    source = "\n".join(lines)+"\n"
    compile(source, str(plotter), "exec")
    plotter.write_text(source, encoding="utf-8")
    shutil.copytree(str(ROOT/"data/meta"), str(dest/"data/meta"))
    shutil.copy2(str(ROOT/"main.pdf"), str(dest/"main.pdf"))
    table_blocks = ["% Table output container for paper_figures.py; not manuscript source."]
    for name in ["data_summary", "main_results", "comparison", "calibration"]:
        table_blocks.extend(["% BEGIN AUTO " + name, "% END AUTO " + name])
    (dest/"paper/main.tex").write_text("\n".join(table_blocks)+"\n", encoding="utf-8")
    aliases = []
    if mode == "figures":
        for basin in ["nwp", "na"]:
            shutil.copytree(str(ROOT/"results"/("ext_"+basin)), str(dest/"results"/("ext_"+basin)))
        source = dest/"results/ext_nwp/conformal/conformal_results_nwp.csv"
        alias = source.with_name("conformal_results.csv")
        if source.exists() and not alias.exists():
            shutil.copy2(str(source), str(alias))
            assert source.read_bytes() == alias.read_bytes()
            aliases.append({"source": source.relative_to(dest).as_posix(), "alias": alias.relative_to(dest).as_posix()})
    marker = {"package_root": str(ROOT), "workspace": str(dest), "mode": mode, "aliases": aliases,
              "data_source": str(ROOT/"data"), "raw_monthly_data_copied": False}
    (dest/".reproduction_workspace.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(marker, ensure_ascii=False, indent=2))
    return marker


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("destination", type=Path)
    ap.add_argument("--mode", choices=["figures", "training"], required=True)
    a = ap.parse_args()
    prepare(a.destination, a.mode)
