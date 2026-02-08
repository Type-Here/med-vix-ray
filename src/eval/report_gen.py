import json
from pathlib import Path
from typing import Dict, List, Union


def fmt(x: float, lo: Union[float, None] = None, hi: Union[float, None] = None, nd: int = 3) -> str:
    if x != x:  # NaN
        return "NA"
    if lo is None or hi is None or lo != lo or hi != hi:
        return f"{x:.{nd}f}"
    return f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def load_results(path: str) -> Dict:
    return json.loads(Path(path).read_text())


def build_table(results: Dict):
    point = results["point"]["metrics"]
    ci = results["ci"]

    label_names: List[str] = results["meta"]["label_names"]

    rows = []

    # per-label rows
    for name in label_names:
        row = {
            "Label": name,
            "AUROC": fmt(
                point["auroc"][name],
                ci["per_label"]["auroc"][name]["lo"],
                ci["per_label"]["auroc"][name]["hi"],
            ),
            "AUPRC": fmt(
                point["auprc"][name],
                ci["per_label"]["auprc"][name]["lo"],
                ci["per_label"]["auprc"][name]["hi"],
            ),
            "Brier": fmt(
                point["brier"][name],
                ci["per_label"]["brier"][name]["lo"],
                ci["per_label"]["brier"][name]["hi"],
            ),
            "LogLoss": fmt(
                point["logloss"][name],
                ci["per_label"]["logloss"][name]["lo"],
                ci["per_label"]["logloss"][name]["hi"],
            ),
        }
        rows.append(row)

    # macro row
    rows.append({
        "Label": "Macro",
        "AUROC": fmt(
            point["auroc_macro"],
            ci["scalars"]["auroc_macro"]["lo"],
            ci["scalars"]["auroc_macro"]["hi"],
        ),
        "AUPRC": fmt(
            point["auprc_macro"],
            ci["scalars"]["auprc_macro"]["lo"],
            ci["scalars"]["auprc_macro"]["hi"],
        ),
        "Brier": fmt(
            point["brier_macro"],
            ci["scalars"]["brier_macro"]["lo"],
            ci["scalars"]["brier_macro"]["hi"],
        ),
        "LogLoss": fmt(
            point["logloss_macro"],
            ci["scalars"]["logloss_macro"]["lo"],
            ci["scalars"]["logloss_macro"]["hi"],
        ),
    })

    # micro row (only AUROC/AUPRC make sense)
    rows.append({
        "Label": "Micro",
        "AUROC": fmt(
            point["auroc_micro"],
            ci["scalars"]["auroc_micro"]["lo"],
            ci["scalars"]["auroc_micro"]["hi"],
        ),
        "AUPRC": fmt(
            point["auprc_micro"],
            ci["scalars"]["auprc_micro"]["lo"],
            ci["scalars"]["auprc_micro"]["hi"],
        ),
        "Brier": "",
        "LogLoss": "",
    })

    return rows


def write_csv(rows, out_csv: str):
    import csv
    fieldnames = ["Label", "AUROC", "AUPRC", "Brier", "LogLoss"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_latex(rows, out_tex: str):
    lines = []
    header = r"\begin{tabular}{lcccc}"
    lines.append(header)
    lines.append(r"\hline")
    lines.append(r"Label & AUROC & AUPRC & Brier & LogLoss \\")
    lines.append(r"\hline")

    for r in rows:
        line = f"{r['Label']} & {r['AUROC']} & {r['AUPRC']} & {r['Brier']} & {r['LogLoss']} \\\\"
        lines.append(line)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    Path(out_tex).write_text("\n".join(lines))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to test_eval.json")
    p.add_argument("--out_csv", default="results/table_metrics.csv")
    p.add_argument("--out_tex", default="results/table_metrics.tex")
    args = p.parse_args()

    results = load_results(args.json)
    rows = build_table(results)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_csv)
    write_latex(rows, args.out_tex)

    print("Wrote:")
    print(" -", args.out_csv)
    print(" -", args.out_tex)


if __name__ == "__main__":
    main()
