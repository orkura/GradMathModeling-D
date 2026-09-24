"""Generate two paper-sized Q3 figures from the unchanged final solution.

Run from any working directory using Group/Lcn/.venv/python.exe.  Only
resource_schedule.pdf and delivery_performance.pdf beside this script are
written.  The solver decisions, source attachments, and Results/Q3 are read-only.
All displayed times are exact saved seconds converted to minutes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


LCN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(LCN / "Code"))
from q3.data import load_data


COLORS = {"A": "#187b9b", "B": "#c9821e", "C": "#6654a4", "R": "#ba4353"}
OUTPUT = Path(__file__).resolve().parent


def charge_seconds(soc: float, full_seconds: float) -> float:
    """Appendix 2: independent resource recharge from return SOC to 100%."""
    if not 0.0 <= soc <= 1.0:
        raise ValueError("Stored SOC is outside [0, 1]")
    if soc < 0.90:
        return full_seconds * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return full_seconds * 0.35 * (1.0 - soc) / 0.10


def resource_schedule(result: dict, data: dict) -> None:
    all_trips = [*result["trips"], *result["relay_trips"]]
    drone_ids = sorted({t["drone"] for t in all_trips}, key=lambda name: (name.startswith("R"), name))
    energy_rows = []
    for trip in all_trips:
        typ = trip.get("type", "R")
        spec = data["relay"] if typ == "R" else data["types"][typ]
        energy_rows.append({
            "type": typ,
            "id": trip["component"] if typ == "R" else trip["battery"],
            "start": trip["start"],
            "end": trip["return"],
            "charge_end": trip["return"] + charge_seconds(trip["soc"], spec["charge_full"]),
        })
    energy_rows.sort(key=lambda row: (row["type"], row["id"], row["start"]))
    resource_ids = list(dict.fromkeys(row["id"] for row in energy_rows))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [len(drone_ids) + 1, len(resource_ids) + 1]})
    for trip in all_trips:
        row = drone_ids.index(trip["drone"])
        typ = trip.get("type", "R")
        start, takeoff, end = [trip[key] / 60 for key in ("start", "takeoff", "return")]
        axes[0].barh(row, takeoff - start, left=start, height=0.70,
                     color=COLORS[typ], alpha=0.28, linewidth=0)
        axes[0].barh(row, end - takeoff, left=takeoff, height=0.70,
                     color=COLORS[typ], linewidth=0)
        axes[0].text((takeoff + end) / 2, row, trip["id"],
                     ha="center", va="center", fontsize=9.5, color="white", weight="bold")
        if typ == "R":
            axes[0].barh(row, data["relay"]["turnaround"] / 60, left=end, height=0.70,
                         color="#e3e8eb", edgecolor="#97a6ae", hatch="///", linewidth=0.4)
    for resource in energy_rows:
        row = resource_ids.index(resource["id"])
        axes[1].barh(row, (resource["end"] - resource["start"]) / 60,
                     left=resource["start"] / 60, height=0.70,
                     color=COLORS[resource["type"]], linewidth=0)
        axes[1].barh(row, (resource["charge_end"] - resource["end"]) / 60,
                     left=resource["end"] / 60, height=0.70,
                     color="#e3e8eb", edgecolor="#97a6ae", hatch="///", linewidth=0.4)
    for ax, labels, title in zip(axes, (drone_ids, resource_ids),
                                 ("Aircraft: preparation and mission", "Energy resources: mission and recharge")):
        ax.set_yticks(range(len(labels)), labels, fontsize=10)
        ax.set_ylim(len(labels) - 0.4, -0.7)
        ax.set_title(title, loc="left", fontsize=11.5, weight="bold", pad=9)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)
    max_time = max(row["charge_end"] for row in energy_rows) / 60
    axes[1].set_xlim(0, math.ceil(max_time / 20) * 20)
    axes[1].set_xlabel("Elapsed time (min)", fontsize=11)
    legend = [Patch(facecolor=COLORS[g], label=f"Type {g}" if g != "R" else "Relay") for g in "ABCR"]
    legend.extend([
        Patch(facecolor="#b7cbd4", alpha=0.55, label="Preparation"),
        Patch(facecolor="#e3e8eb", edgecolor="#97a6ae", hatch="///", label="Charge / turnaround"),
    ])
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.56, 0.992),
               ncol=3, fontsize=9.5, frameon=False, handlelength=1.5, columnspacing=1.6)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.90, bottom=0.075, hspace=0.22)
    fig.savefig(OUTPUT / "resource_schedule.pdf", facecolor="white")
    plt.close(fig)


def delivery_performance(result: dict, data: dict) -> None:
    completion = {box: timestamp for trip in result["trips"]
                  for box, timestamp in trip["deliveries"].items()}
    if set(completion) != {box["id"] for box in data["boxes"]}:
        raise ValueError("Figure input must contain every box exactly once")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8))
    actual = sorted(timestamp / 60 for timestamp in completion.values())
    deadlines = sorted(box["deadline"] / 60 for box in data["boxes"] if math.isfinite(box["deadline"]))
    axes[0].step([0, *actual], [0, *range(1, len(actual) + 1)], where="post",
                 color=COLORS["A"], linewidth=2.1, label="Delivered boxes")
    axes[0].step([0, *deadlines], [0, *range(1, len(deadlines) + 1)], where="post",
                 color=COLORS["R"], linewidth=1.5, linestyle="--", label="Cumulative hard deadlines")
    axes[0].set_title("Delivery progress", loc="left", weight="bold", fontsize=12)
    axes[0].set(xlabel="Elapsed time (min)", ylabel="Number of boxes", ylim=(0, len(actual) + 5))
    axes[0].legend(loc="upper left", fontsize=10, frameon=False)
    for hard, label, color, marker in [(True, "Has hard deadline", COLORS["R"], "o"),
                                       (False, "Soft due time", COLORS["A"], "x")]:
        boxes = [box for box in data["boxes"] if math.isfinite(box["deadline"]) == hard]
        axes[1].scatter([box["due"] / 60 for box in boxes],
                        [completion[box["id"]] / 60 for box in boxes],
                        s=34, c=color, marker=marker, alpha=0.74, label=label)
    last_expected = max(box["due"] / 60 for box in data["boxes"])
    axis_limit = max(last_expected, max(actual)) * 1.05
    axes[1].plot([0, axis_limit], [0, axis_limit], linestyle="--", color="#8597a0",
                 linewidth=1.0, label="At expected due time", zorder=0)
    axes[1].set_title("Actual versus expected delivery time", loc="left", weight="bold", fontsize=12)
    axes[1].set(xlabel="Expected due time (min)", ylabel="Actual delivery time (min)",
                xlim=(0, axis_limit), ylim=(0, max(actual) * 1.13))
    axes[1].legend(loc="upper left", fontsize=10, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.95, bottom=0.085, hspace=0.42)
    fig.savefig(OUTPUT / "delivery_performance.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    result_dir = LCN / "Results" / "Q3"
    result = json.loads((result_dir / "solution.json").read_text(encoding="utf-8"))
    validation = json.loads((result_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise ValueError("Refuse to present a final schedule whose independent validation failed")
    data = load_data()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": "#8fa1ab", "text.color": "#163441",
                         "axes.labelcolor": "#324e5c", "xtick.color": "#536a77", "ytick.color": "#536a77"}):
        resource_schedule(result, data)
        delivery_performance(result, data)
    print("Generated resource_schedule.pdf and delivery_performance.pdf from the saved final solution.")


if __name__ == "__main__":
    main()
