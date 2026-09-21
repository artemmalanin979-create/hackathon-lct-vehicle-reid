"""Render saved calibration measurements; do not run inference or calibration."""

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
SOURCE = HERE.parent.parent / "04-solution/service/calib-d1_j48"
DEST = HERE
BRANCH = "rerank_market_presence"


def render():
    curve_path = SOURCE / f"curve_{BRANCH}.csv"
    summary_path = SOURCE / "summary.json"
    saved_points = json.loads(summary_path.read_text())[BRANCH]["selected"]
    selected = saved_points["robust_balanced"]
    threshold = selected["threshold"]
    sources = [summary_path]
    full_curve = curve_path.is_file()
    if full_curve:
        with curve_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        sources.append(curve_path)
    else:
        # The committed d1_j48 summary contains measured operating points, not
        # the full threshold sweep. Plot only those points: connecting them
        # would imply unmeasured values between the saved thresholds.
        rows = list({point["threshold"]: point for point in saved_points.values()}.values())

    # Pin the plotted result to the report used for this presentation draft.
    assert threshold == 0.5282812306342437
    assert tuple(selected[k] for k in ("tp", "fp", "fn", "tn_unknown")) == (606, 60, 226, 218)
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
        "svg.hashsalt": "d1_j48",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.24, top=0.79)
    fig.text(0.09, 0.94, "F1 и доля корректных отказов", fontsize=19, weight="bold")
    fig.text(0.09, 0.885,
             "d1_j48 · переранжирование · market / presence · наша валидация",
             fontsize=10.5)

    if full_curve:
        # For acceptance s >= t, the value between two observed scores equals
        # the value at the higher threshold. No smoothing or invented samples.
        ax.step(thresholds, f1, where="pre", color="#520978", linewidth=2.3, label="F1")
        ax.step(thresholds, tnr, where="pre", color="#FF0053", linewidth=2.3,
                linestyle=(0, (5, 2)), label="TNR — корректные отказы")
    else:
        ax.scatter(thresholds, f1, color="#520978", s=25, label="F1")
        ax.scatter(thresholds, tnr, color="#FF0053", marker="s", s=23,
                   label="TNR — корректные отказы")
    ax.axvline(threshold, color="#1C1D22", linewidth=1.1, linestyle=":")
    ax.scatter([threshold, threshold], [selected["f1"], selected["tnr"]],
               color=["#520978", "#FF0053"], edgecolors="white", s=54, zorder=5)
    ax.annotate("Порог 0,52828\nF1 0,809 · TNR 0,784",
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
    if full_curve:
        fig.text(0.09, 0.105, "Валидация: 1110 запросов; 832 с парой, 278 без пары.", fontsize=10)
        fig.text(0.09, 0.058, "Порог выбран на этой валидации; независимого замера нет.", fontsize=10)
    else:
        fig.text(0.09, 0.105, f"{len(rows)} сохранённых точек; 1110 запросов: 832 с парой, 278 без пары.", fontsize=10)
        fig.text(0.09, 0.058, "Без интерполяции. Порог выбран здесь; независимого замера нет.", fontsize=10)

    DEST.mkdir(exist_ok=True)
    svg_path = DEST / "f1_tnr_threshold.svg"
    png_path = DEST / "f1_tnr_threshold.png"
    fig.savefig(svg_path, metadata={"Date": None, "Description":
        "Existing calibration data, no new model run. Sources: " + ", ".join(str(p) for p in sources)})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    provenance = {
        "configuration": "d1_j48",
        "branch": BRANCH,
        "selection": "selected.robust_balanced",
        "full_curve_available": full_curve,
        "plotted_points": len(rows),
        "selected": selected,
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "rendering": ("steps-pre, existing CSV rows only, no smoothing" if full_curve else
                      "saved measured operating points only, no lines or interpolation"),
        "notice": "d1_j48 calibration validation; two models plus whitening; not a holdout result.",
        "outputs": [str(svg_path), str(png_path)],
    }
    (DEST / "calibration-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"full_curve_available": full_curve, "plotted_points": len(rows), "selected": {
        key: selected[key] for key in ("threshold", "f1", "tnr", "tp", "fp", "fn", "tn_unknown")
    }, "outputs": [str(svg_path), str(png_path)]}, ensure_ascii=False))


if __name__ == "__main__":
    render()
