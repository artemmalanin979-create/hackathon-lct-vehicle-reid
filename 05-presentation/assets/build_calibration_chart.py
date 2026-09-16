"""Render the existing calibration CSV; do not run inference or calibration."""

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "04-solution/service/calib"
DEST = HERE
BRANCH = "rerank_market_presence"


def render():
    curve_path = SOURCE / f"curve_{BRANCH}.csv"
    summary_path = SOURCE / "summary.json"
    with curve_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = json.loads(summary_path.read_text())[BRANCH]["selected"]["robust_balanced"]
    threshold = selected["threshold"]

    # Pin the plotted result to the report used for this presentation draft.
    assert threshold == 0.49937235233589916
    assert tuple(selected[k] for k in ("tp", "fp", "fn", "tn_unknown")) == (603, 62, 229, 216)
    matching = [row for row in rows if float(row["threshold"]) == threshold]
    assert len(matching) == 1
    for key in ("f1", "tnr"):
        assert abs(float(matching[0][key]) - selected[key]) < 1e-12

    rows.sort(key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in rows]
    f1 = [float(row["f1"]) for row in rows]
    tnr = [float(row["tnr"]) for row in rows]
    assert all(0 <= value <= 1 for value in f1 + tnr)

    matplotlib.rcParams.update({
        "font.family": "Montserrat",
        "font.size": 11,
        "text.color": "#1C1D22",
        "axes.labelcolor": "#1C1D22",
        "xtick.color": "#1C1D22",
        "ytick.color": "#1C1D22",
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.24, top=0.79)
    fig.text(0.09, 0.94, "F1 и доля корректных отказов", fontsize=19, weight="bold")
    fig.text(0.09, 0.885,
             "Пакетное переранжирование · market / presence · наша валидация",
             fontsize=10.5)

    # For acceptance s >= t, the value between two observed scores equals
    # the value at the higher threshold. No smoothing or invented samples.
    ax.step(thresholds, f1, where="pre", color="#520978", linewidth=2.3, label="F1")
    ax.step(thresholds, tnr, where="pre", color="#FF0053", linewidth=2.3,
            linestyle=(0, (5, 2)), label="TNR — корректные отказы")
    ax.axvline(threshold, color="#1C1D22", linewidth=1.1, linestyle=":")
    ax.scatter([threshold, threshold], [selected["f1"], selected["tnr"]],
               color=["#520978", "#FF0053"], edgecolors="white", s=54, zorder=5)
    ax.annotate("Порог 0,4994\nF1 0,806 · TNR 0,777",
                xy=(threshold, selected["tnr"]), xytext=(0.41, 0.37),
                fontsize=11, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "#1C1D22", "lw": 0.9},
                bbox={"boxstyle": "square,pad=0.35", "fc": "white", "ec": "none"})
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Порог близости s = 1 − d",
           ylabel="Значение метрики")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    formatter = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(axis="y", color="#DAD7DE", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8E8993")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), borderaxespad=0,
              ncol=2, frameon=False, fontsize=10)
    fig.text(0.09, 0.105, "Валидация: 1110 запросов; 832 с парой, 278 без пары.", fontsize=10)
    fig.text(0.09, 0.058, "Порог выбран на этой валидации; независимого замера нет.", fontsize=10)

    DEST.mkdir(exist_ok=True)
    svg_path = DEST / "f1_tnr_threshold.svg"
    png_path = DEST / "f1_tnr_threshold.png"
    fig.savefig(svg_path, metadata={"Description":
        "Existing calibration data, no new model run. Source: " + str(curve_path)})
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    provenance = {
        "branch": BRANCH,
        "selection": "selected.robust_balanced",
        "curve_rows": len(rows),
        "selected": selected,
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (curve_path, summary_path)
        },
        "rendering": "steps-pre, existing CSV rows only, no smoothing",
        "notice": "Calibration validation; untrained weights; not a holdout result.",
        "outputs": [str(svg_path), str(png_path)],
    }
    (DEST / "calibration-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"curve_rows": len(rows), "selected": {
        key: selected[key] for key in ("threshold", "f1", "tnr", "tp", "fp", "fn", "tn_unknown")
    }, "outputs": [str(svg_path), str(png_path)]}, ensure_ascii=False))


if __name__ == "__main__":
    render()
