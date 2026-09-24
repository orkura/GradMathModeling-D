"""CSV tables, publication figures, and a self-contained Chinese Q3 report.

This module reports an already-computed result; it does not change the solver
decisions. Figure text is English for portable PDF embedding; the surrounding
HTML, captions, and submission-compatible CSV headers are Chinese.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


COLORS = {"A": "#187b9b", "B": "#d18c25", "C": "#6654a4", "R": "#ba4353"}
PHASES = {"climb": "爬升", "cruise": "巡航", "descent": "下降",
          "service": "交付", "delivery": "交付", "up": "爬升", "down": "下降"}
MODE = {"direct": "直连", "relay": "中继", "outage": "中断"}


def _number(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(value) else round(float(value), digits)
    return value


def _csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows([_number(value) for value in row] for row in rows)


def _charge(soc: float, full: float) -> float:
    if not -1e-9 <= soc <= 1 + 1e-9:
        raise ValueError(f"Cannot report charging time for SOC outside [0,1]: {soc}")
    soc = min(1.0, max(0.0, float(soc)))
    if soc < 0.90:
        return full * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return full * 0.35 * (1.0 - soc) / 0.10


def _route(value: list[str]) -> str:
    nodes = list(value)
    if not nodes or nodes[0] != "O01":
        nodes.insert(0, "O01")
    if nodes[-1] != "O01":
        nodes.append("O01")
    return " → ".join(nodes)


def _save(fig, out: Path, stem: str, outputs: list[Path]) -> Path:
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    outputs.extend([out / f"{stem}.png", out / f"{stem}.pdf"])
    return out / f"{stem}.png"


def _energy_rows(result: dict, data: dict) -> list[dict]:
    rows = []
    for is_relay, trips in [(False, result.get("trips", [])), (True, result.get("relay_trips", []))]:
        for trip in trips:
            typ = "R" if is_relay else trip["type"]
            parameters = data["relay"] if is_relay else data["types"][typ]
            end = float(trip["return"])
            soc = float(trip.get("soc", 1.0 - trip["energy"] / parameters["energy"]))
            charge_end = end + _charge(soc, parameters["charge_full"])
            rows.append({
                "category": "中继能源组件" if is_relay else "运输电池",
                "type": typ, "resource": trip["component"] if is_relay else trip["battery"],
                "drone": trip["drone"], "trip": trip["id"], "start": trip["start"],
                "end": end, "energy": trip["energy"], "soc": soc,
                "charge_start": end, "charge_end": charge_end,
                "ready": charge_end,
            })
    return sorted(rows, key=lambda r: (r["type"], r["resource"], r["start"]))


def _write_tables(result: dict, data: dict, terrain, out: Path, outputs: list[Path]) -> list[dict]:
    trips = result.get("trips", [])
    relay_trips = result.get("relay_trips", [])
    path = out / "Q3_运输架次.csv"
    _csv(path, ["架次编号", "无人机编号", "机型编号", "电池编号", "开始时刻（s）", "访问服务区顺序",
                "返回O01时刻（s）", "架次能耗（kWh）", "起飞时刻（s）", "返航SOC", "货箱编号"],
         [[p["id"], p["drone"], p["type"], p["battery"], p["start"], _route(p["route"]),
           p["return"], p["energy"], p["takeoff"], p["soc"], ";".join(p["box_ids"])] for p in trips])
    outputs.append(path)
    delivered = {
        box: (p["id"], t) for p in trips for box, t in p.get("deliveries", {}).items()
    }
    box_rows = []
    for box in data["boxes"]:
        trip, completed = delivered.get(box["id"], ("", None))
        late = max(0, completed - box["due"]) if completed is not None else None
        hard_ok = (completed <= box["deadline"] + 1e-7) if completed is not None else False
        box_rows.append([box["id"], trip, box["node"], completed, box["kind"],
                         "是" if box["first"] else "否", box["first_deadline"], box["due"],
                         box["deadline"], late, box["weight"], "是" if hard_ok else "否"])
    path = out / "Q3_逐箱交付.csv"
    _csv(path, ["货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）", "物资类型", "是否首批保障",
                "首批截止时间（s）", "期望送达时间（s）", "硬截止时间（s）", "延迟（s）", "应急优先系数", "硬约束满足"], box_rows)
    outputs.append(path)
    relay_rows = []
    for p in relay_trips:
        x, y, z = p["position"]
        lon, lat = terrain.lonlat(x, y)
        relay_rows.append([p["id"], p["drone"], p["component"], p["start"], lon, lat, z,
                           p["ready"], p["service_end"], p["return"], p["energy"],
                           z - terrain.ground(x, y), p["soc"]])
    path = out / "Q3_中继架次.csv"
    _csv(path, ["中继架次编号", "中继无人机编号", "能源组件编号", "开始时刻（s）", "悬停经度（°）", "悬停纬度（°）",
                "悬停海拔（m）", "建链完成时刻（s）", "服务结束时刻（s）", "返回O01时刻（s）", "架次能耗（kWh）",
                "悬停离地高度（m）", "返航SOC"], relay_rows)
    outputs.append(path)
    path = out / "Q3_通信保障.csv"
    _csv(path, ["运输架次编号", "通信阶段", "开始时刻（s）", "结束时刻（s）", "保障方式", "中继架次编号",
                "链路裕量下界（dB）", "区间连续性已认证", "记录语义"],
         [[c["trip_id"], PHASES.get(c["phase"], c["phase"]), c["t0"], c["t1"], MODE.get(c["mode"], c["mode"]),
           c.get("relay_trip_id") or "", c.get("margin_db"), "是" if c.get("certified") else "否",
           "区间保底保障预约；实际通信状态优先直连，见validation.json采样统计"]
          for c in result.get("communication", [])])
    outputs.append(path)
    energy_rows = _energy_rows(result, data)
    path = out / "Q3_能源资源占用.csv"
    _csv(path, ["资源类别", "机型", "资源编号", "无人机编号", "架次编号", "任务占用开始（s）", "任务占用结束（s）",
                "本次能耗（kWh）", "返航SOC", "充电开始（s）", "充电完成（s）", "下次满电可用（s）"],
         [[r[k] for k in ["category", "type", "resource", "drone", "trip", "start", "end", "energy", "soc",
                          "charge_start", "charge_end", "ready"]] for r in energy_rows])
    outputs.append(path)
    return energy_rows


def _map(result: dict, data: dict, terrain, out: Path, outputs: list[Path]) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 8))
    bounds = terrain.bounds
    dem = np.asarray(terrain.dem)
    if getattr(terrain, "nodata", None) is not None:
        dem = np.where(dem == terrain.nodata, np.nan, dem)
    image = ax.imshow(dem, extent=[bounds[0] / 1000, bounds[2] / 1000, bounds[1] / 1000, bounds[3] / 1000],
                      origin="upper", cmap="terrain", alpha=0.70, interpolation="nearest", zorder=0)
    all_xy = []
    for p in result.get("trips", []):
        for s in p.get("segments", []):
            a, b = np.asarray(s["p0"]), np.asarray(s["p1"])
            if np.linalg.norm(a[:2] - b[:2]) < 1e-6:
                continue
            ax.plot([a[0] / 1000, b[0] / 1000], [a[1] / 1000, b[1] / 1000],
                    color=COLORS[p["type"]], lw=1.15, alpha=0.60, zorder=2)
    depot = data["nodes"]["O01"]
    depot_xy = np.asarray(terrain.xy(depot["lon"], depot["lat"]))
    for node in data["nodes"].values():
        xy = np.asarray(terrain.xy(node["lon"], node["lat"])) / 1000
        all_xy.append(xy)
        if node["id"] == "O01":
            ax.scatter(*xy, marker="*", s=190, c="#193742", edgecolors="white", linewidths=0.9, zorder=6)
        else:
            ax.scatter(*xy, s=36, c="white", edgecolors="#193742", linewidths=0.85, zorder=5)
        ax.annotate(node["id"], xy, xytext=(5, 5), textcoords="offset points", fontsize=8, weight="bold", zorder=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
    seen_positions = set()
    for p in result.get("relay_trips", []):
        x, y, z = p["position"]
        point_key = (round(x, 4), round(y, 4), round(z, 4))
        if point_key in seen_positions:
            continue
        seen_positions.add(point_key)
        all_xy.append(np.array([x, y]) / 1000)
        ax.plot([depot_xy[0] / 1000, x / 1000], [depot_xy[1] / 1000, y / 1000],
                color=COLORS["R"], linestyle="--", lw=1.5, alpha=0.85, zorder=3)
        ax.scatter(x / 1000, y / 1000, marker="^", s=110, color=COLORS["R"], edgecolors="white", linewidths=0.8, zorder=8)
        ax.annotate(f"{p['drone']}  {z:.0f} m ASL", (x / 1000, y / 1000), xytext=(7, -12),
                    textcoords="offset points", fontsize=8, color="#8f2636", weight="bold", zorder=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1})
    xy = np.asarray(all_xy)
    ax.set_xlim(xy[:, 0].min() - 0.7, xy[:, 0].max() + 0.7)
    ax.set_ylim(xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("Local east coordinate (km)")
    ax.set_ylabel("Local north coordinate (km)")
    ax.set_title("Transport routes and relay locations", loc="left", weight="bold", pad=14)
    handles = [Line2D([0], [0], color=COLORS[g], lw=2, label=f"Type {g}") for g in "ABC"]
    handles.append(Line2D([0], [0], color=COLORS["R"], lw=2, ls="--", label="Relay flight"))
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=8, framealpha=0.95)
    fig.colorbar(image, ax=ax, shrink=0.72, pad=0.03, label="DEM elevation (m ASL)")
    fig.tight_layout()
    return _save(fig, out, "01_routes_and_relays", outputs)


def _resources(result: dict, energy_rows: list[dict], data: dict, out: Path, outputs: list[Path]) -> Path:
    all_trips = [*result.get("trips", []), *result.get("relay_trips", [])]
    drone_ids = sorted({t["drone"] for t in all_trips}, key=lambda v: (v.startswith("R"), v))
    resource_ids = list(dict.fromkeys(r["resource"] for r in energy_rows))
    ratio = [max(3, len(drone_ids)), max(5, len(resource_ids))]
    fig, axes = plt.subplots(2, 1, figsize=(12, max(8, sum(ratio) * 0.34)),
                             sharex=True, gridspec_kw={"height_ratios": ratio, "hspace": 0.16})
    for p in all_trips:
        i = drone_ids.index(p["drone"])
        typ = p.get("type", "R")
        start, takeoff, end = [p[k] / 60 for k in ["start", "takeoff", "return"]]
        axes[0].barh(i, takeoff - start, left=start, color=COLORS[typ], alpha=0.3, height=0.65)
        axes[0].barh(i, end - takeoff, left=takeoff, color=COLORS[typ], height=0.65)
        if end - start >= 5:
            axes[0].text((start + end) / 2, i, p["id"], ha="center", va="center", fontsize=7, color="white", weight="bold")
        if typ == "R":
            axes[0].barh(i, data["relay"]["turnaround"] / 60, left=end, color="#dce2e5", height=0.65, hatch="//", edgecolor="#aab4ba", lw=0.3)
    for r in energy_rows:
        i = resource_ids.index(r["resource"])
        axes[1].barh(i, (r["end"] - r["start"]) / 60, left=r["start"] / 60, color=COLORS[r["type"]], height=0.65)
        axes[1].barh(i, (r["charge_end"] - r["charge_start"]) / 60, left=r["charge_start"] / 60,
                     color="#dce2e5", edgecolor="#aab4ba", hatch="//", lw=0.3, height=0.65)
    for ax, ids, title in zip(axes, [drone_ids, resource_ids], ["Aircraft occupation", "Energy resources: mission and recharge"]):
        ax.set_yticks(range(len(ids)), ids, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.20)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Elapsed time (min)")
    handles = [Patch(facecolor=COLORS[g], label=f"Type {g}" if g != "R" else "Relay") for g in "ABCR"]
    handles.append(Patch(facecolor="#dce2e5", hatch="//", edgecolor="#aab4ba", label="Recharge / relay turnaround"))
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=5, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.06)
    return _save(fig, out, "02_resource_schedule", outputs)


def _deliveries(result: dict, data: dict, out: Path, outputs: list[Path]) -> Path:
    completed = {b: t for p in result.get("trips", []) for b, t in p.get("deliveries", {}).items()}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1, 1.15]})
    actual = sorted(completed.values())
    axes[0].step([0, *[t / 60 for t in actual]], [0, *range(1, len(actual) + 1)], where="post", color=COLORS["A"], lw=2.2, label="Delivered")
    hard = sorted(b["deadline"] / 60 for b in data["boxes"] if math.isfinite(b["deadline"]))
    axes[0].step([0, *hard], [0, *range(1, len(hard) + 1)], where="post", color=COLORS["R"], lw=1.5, linestyle="--", label="Hard-due boxes")
    axes[0].set(xlabel="Elapsed time (min)", ylabel="Cumulative number of boxes", ylim=(0, len(data["boxes"]) + 4))
    axes[0].set_title("Delivery progress", loc="left", weight="bold")
    axes[0].legend(fontsize=8, frameon=False)
    due = [b["due"] / 60 for b in data["boxes"] if b["id"] in completed]
    actual_due = [completed[b["id"]] / 60 for b in data["boxes"] if b["id"] in completed]
    hard_due = [math.isfinite(b["deadline"]) for b in data["boxes"] if b["id"] in completed]
    for flag, label, color, marker in [(True, "Has hard deadline", COLORS["R"], "o"), (False, "Soft due time", COLORS["A"], "x")]:
        idx = [i for i, value in enumerate(hard_due) if value == flag]
        axes[1].scatter([due[i] for i in idx], [actual_due[i] for i in idx], s=36,
                        c=color, marker=marker, alpha=0.72, label=label)
    maximum = max([1, *due, *actual_due]) * 1.06
    axes[1].plot([0, maximum], [0, maximum], "--", color="#82929b", lw=1, label="Delivered at expected due time")
    axes[1].set(xlabel="Expected due time (min)", ylabel="Actual completion time (min)", xlim=(0, maximum))
    axes[1].set_title("Delivery time versus expected due time", loc="left", weight="bold")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.20)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out, "03_delivery_performance", outputs)


def _communications(result: dict, out: Path, outputs: list[Path]) -> Path:
    trips = sorted(result.get("trips", []), key=lambda t: (t["takeoff"], t["id"]))
    positions = {p["id"]: i for i, p in enumerate(trips)}
    fig, axes = plt.subplots(2, 1, figsize=(12, max(6.4, len(trips) * 0.28 + 3.2)),
                             sharex=True, gridspec_kw={"height_ratios": [max(3, len(trips) * 0.24), 1.6], "hspace": 0.19})
    comm_colors = {"direct": COLORS["A"], "relay": COLORS["R"], "outage": "#111111"}
    for c in result.get("communication", []):
        if c["trip_id"] not in positions:
            continue
        axes[0].barh(positions[c["trip_id"]], (c["t1"] - c["t0"]) / 60, left=c["t0"] / 60,
                     height=0.70, color=comm_colors.get(c["mode"], "#aaaaaa"), linewidth=0)
        margin = c.get("margin_db")
        if margin is not None and math.isfinite(margin):
            axes[1].plot([c["t0"] / 60, c["t1"] / 60], [margin, margin],
                         color=comm_colors.get(c["mode"], "#aaaaaa"), lw=1.0, alpha=0.48)
    axes[0].set_yticks(range(len(trips)), [p["id"] for p in trips], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Guaranteed communication provider reservations", loc="left", weight="bold", pad=12)
    axes[0].legend(handles=[Patch(facecolor=comm_colors[k], label=v) for k, v in [("direct", "Gateway guarantee"), ("relay", "Relay reservation")]],
                   loc="upper right", bbox_to_anchor=(1, 1.10), ncol=2, frameon=False, fontsize=8)
    axes[1].axhline(0, color="#111111", ls="--", lw=0.9)
    axes[1].set(xlabel="Elapsed time (min)", ylabel="Margin lower bound (dB)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.20)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.08)
    return _save(fig, out, "04_communication_guarantee", outputs)


def _history(result: dict, out: Path, outputs: list[Path]) -> Path | None:
    history = result.get("search_history", [])
    if isinstance(history, dict):
        history = list(history.values())
    candidates = []
    for i, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        item = {**item, **item.get("metrics", {})}
        if item.get("makespan_s") is not None and item.get("total_energy_kwh") is not None:
            candidates.append((i, item))
    if not candidates:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    x = [i + 1 for i, _ in candidates]
    axes[0].plot(x, [p["makespan_s"] / 60 for _, p in candidates], "o-", lw=1.4, ms=4, color=COLORS["A"])
    axes[1].scatter([p["makespan_s"] / 60 for _, p in candidates], [p["total_energy_kwh"] for _, p in candidates],
                    c=np.arange(len(candidates)), cmap="viridis", s=48, alpha=0.8)
    selected = result.get("metrics", {})
    if "makespan_s" in selected and "total_energy_kwh" in selected:
        axes[1].scatter(selected["makespan_s"] / 60, selected["total_energy_kwh"], marker="*", s=210,
                        c=COLORS["R"], edgecolors="white", linewidth=0.8, label="Selected solution", zorder=5)
        axes[1].legend(frameon=False, fontsize=8)
    axes[0].set(xlabel="Evaluated candidate", ylabel="Joint makespan (min)")
    axes[1].set(xlabel="Joint makespan (min)", ylabel="Total energy (kWh)")
    axes[0].set_title("Candidate schedule comparison", loc="left", weight="bold")
    axes[1].set_title("Completion time and energy trade-off", loc="left", weight="bold")
    for ax in axes:
        ax.grid(alpha=0.20)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out, "05_search_comparison", outputs)


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value):
        return html.escape(str(_number(value)))
    return ("<div class='table-scroll'><table><thead><tr>" + "".join(f"<th>{cell(h)}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _write_html(result: dict, data: dict, terrain, out: Path, outputs: list[Path], figures: list[tuple[Path, str, str]]) -> None:
    metrics, validation = result.get("metrics", {}), result.get("validation", {})
    passed = validation.get("passed", validation.get("feasible")) if isinstance(validation, dict) else None
    state = "通过全部已实现核验" if passed is True else ("发现未通过核验项" if passed is False else "核验状态未声明")
    state_class = "pass" if passed is True else "review"
    cards = [
        ("已交付货箱", f"{metrics.get('boxes_delivered', '—')} / {len(data['boxes'])}", "不可拆货箱"),
        ("联合完成时间", f"{metrics.get('makespan_s', 0) / 60:.2f}", "分钟，包含中继返航"),
        ("总能耗", f"{metrics.get('total_energy_kwh', 0):.3f}", "kWh，运输 + 中继"),
        ("使用架次", f"{metrics.get('transport_sorties', '—')} + {metrics.get('relay_sorties', '—')}", "运输架次 + 中继架次"),
        ("硬时限违约", str(metrics.get("hard_deadline_violations", "—")), "医疗及首批保障货箱"),
        ("加权延迟", f"{metrics.get('weighted_tardiness', 0):.2f}", "优先系数 × 延迟秒数"),
    ]
    card_html = "".join(f"<div class='metric'><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><small>{html.escape(unit)}</small></div>" for title, value, unit in cards)
    figures_html = ""
    for path, title, caption in figures:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        figures_html += f"<section><h2>{html.escape(title)}</h2><figure><img src='data:image/png;base64,{encoded}' alt='{html.escape(title)}'><figcaption>{html.escape(caption)}</figcaption></figure></section>"
    relay_rows = []
    for p in result.get("relay_trips", []):
        x, y, z = p["position"]
        lon, lat = terrain.lonlat(x, y)
        relay_rows.append([p["id"], p["drone"], f"{lon:.7f}, {lat:.7f}", f"{z:.1f}", f"{z - terrain.ground(x, y):.1f}",
                           f"{p['ready'] / 60:.2f}–{p['service_end'] / 60:.2f}", p["component"], f"{p['energy']:.3f}"])
    relay_table = _html_table(["架次", "无人机", "经纬度", "海拔 / m", "离地 / m", "服务区间 / min", "能源组件", "能耗 / kWh"], relay_rows)
    details = html.escape(json.dumps(validation, ensure_ascii=False, indent=2, default=str))
    config = html.escape(json.dumps({"construction": result.get("config", {}),
        "search_bounds": result.get("search_bounds", {}),
        "timing_refinement": result.get("refinement")}, ensure_ascii=False, indent=2, default=str))
    communication = validation.get("communication", {}) if isinstance(validation, dict) else {}
    if communication:
        sample_step = html.escape(str(communication.get("sample_step_s", "未声明")))
        sample_total = html.escape(str(communication.get("samples", "未声明")))
        direct_count = html.escape(str(communication.get("direct_samples", "未声明")))
        relay_count = html.escape(str(communication.get("relay_samples", "未声明")))
        interruption_count = html.escape(str(communication.get("interruptions", "未声明")))
        continuous = communication.get("continuous_certified")
        continuous_state = "通过" if continuous is True else ("未通过" if continuous is False else "未声明")
        interval_count = html.escape(str(communication.get("certified_intervals", "未声明")))
        communication_evidence = (
            f"<p><strong>直连优先的实际状态采样：</strong>独立核验以 {sample_step} 秒为采样步长，"
            f"共检查 {sample_total} 个时刻记录，其中直连 {direct_count}、中继 {relay_count}、"
            f"通信中断 {interruption_count}。这些数字表示采样记录数量，不是连续服务时长；"
            "网关直连可用时始终优先记为直连，即使该区间已预约中继保障。</p>"
            f"<p><strong>独立的全时连续性认证：</strong>{continuous_state}，已认证区间 {interval_count} 个。"
            "该结论来自原始 DEM 像元模型上的区间链路裕量下界检验，而非仅凭有限时刻采样；"
            "完整失败项及无法认证区间见核验明细。</p>"
        )
    else:
        communication_evidence = "<p>核验结果未提供直连优先采样统计或独立的全时连续性认证；不可仅依据预约图认定全程通信通过。</p>"
    source_rows = [[Path(path).name, data.get("source_sha256", {}).get(key, "未记录")] for key, path in data.get("source_paths", {}).items()]
    sources = _html_table(["原始输入文件", "SHA256"], source_rows)
    downloads = "".join(f"<li><a href='{html.escape(p.name)}'>{html.escape(p.name)}</a></li>" for p in outputs if p.suffix == ".csv")
    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题三 · 运输与通信中继联合仿真</title>
<style>
:root{{--ink:#163441;--muted:#667986;--teal:#187b9b;--line:#dae4e8;--page:#f0f5f6;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;line-height:1.75}}
main{{max-width:1240px;margin:0 auto;padding:44px 32px 70px}}header{{padding-bottom:26px;border-bottom:3px solid var(--ink)}}
.eyebrow{{font-size:12px;letter-spacing:2px;color:var(--teal);font-weight:700}}h1{{font-size:32px;margin:8px 0 12px;line-height:1.35}}header p{{max-width:950px;margin:0;color:var(--muted)}}
.status{{display:inline-block;margin-top:18px;padding:5px 12px;font-size:13px;border-radius:4px;font-weight:700}}.pass{{background:#dff0e8;color:#226046}}.review{{background:#fff1d9;color:#885b15}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}}.metric{{background:white;padding:18px 22px;border:1px solid var(--line);border-radius:5px}}.metric span,.metric small{{display:block;color:var(--muted);font-size:13px}}.metric strong{{display:block;font-size:31px;font-weight:600;line-height:1.5}}
section{{margin-top:26px;padding:24px;background:white;border:1px solid var(--line);border-radius:5px}}h2{{font-size:20px;margin:0 0 14px}}p{{margin:10px 0}}figure{{margin:0}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px;margin-top:12px}}
.table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{background:#eef4f6;font-weight:600}}pre{{overflow:auto;font-size:12px;background:#f4f7f8;padding:16px;line-height:1.6}}details{{margin-top:16px}}summary{{cursor:pointer;font-weight:600}}a{{color:var(--teal)}}.notes{{font-size:14px}}.notes li{{margin:7px 0}}footer{{margin-top:22px;font-size:12px;color:var(--muted)}}
@media(max-width:700px){{main{{padding:24px 14px}}h1{{font-size:25px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.metric{{padding:14px}}.metric strong{{font-size:25px}}section{{padding:15px}}}}@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main>
<header><div class="eyebrow">Q3 / COMPUTATIONAL EXPERIMENT</div><h1>山区洪涝灾害下运输与通信中继联合仿真</h1>
<p>使用题目给定的 15 个服务区、80 个货箱、3 类运输机、2 架中继机及原始 30 米 DEM，联合安排运输、中继服务和能源资源。结果适用于本文列出的物理口径与候选搜索范围。</p>
<span class="status {state_class}">{state}</span></header>
<div class="metrics">{card_html}</div>
<section><h2>结果解释与适用范围</h2><p><strong>这是已实现候选搜索得到的方案，不构成连续空间或全部组批、路线、时序的全局最优证明。</strong>可行性状态以独立核验明细为准；通信保证针对原始 DEM 的分片常高像元模型，不代表未观测地形、天气或真实无线干扰下的实测可靠性。</p>
<p>候选方案依次比较加权配送延迟、联合完成时间、总能耗和总架次数。构造搜索之后，程序可用 CP-SAT 在固定组批、路线及中继覆盖关系下重新安排时序与实体资源；该阶段的最优界仅适用于固定结构的整数秒排程模型，求解状态与预算见下方配置。</p>
<p>返航安全余量为各机型给定值；医疗物资与首批货箱分别检查硬时限。无人机、电池和中继能源组件按独立资源记录，每次再次使用能源资源前须充至 100%。任务完成时刻包含两类无人机最终返航。</p>
<details><summary>查看核验明细</summary><pre>{details}</pre></details><details><summary>查看本次求解配置</summary><pre>{config}</pre></details></section>
<section><h2>通信记录口径与独立核验</h2><p>通信保障 CSV 及对应甘特图记录的是<strong>区间保底保障预约</strong>：指定主体须在整个区间内提供可用链路。其“保障方式”字段表示保底提供者，不等同于每个时刻按直连优先规则得到的实际通信状态。</p>{communication_evidence}</section>
{figures_html}
<section><h2>中继位置与服务安排</h2>{relay_table}</section>
<section class="notes"><h2>建模口径与补充假设</h2><ul>
<li>经纬度以 DEM 中心的 WGS84 局部线性近似换算为米；地形按原始像元取常值。航段经过的全部像元参与净空和遮挡判断。</li>
<li>运输水平能耗补全为 E<sub>use</sub> × d / L(q)，爬升能耗为 (m+q)gΔh / (η × 3.6×10<sup>6</sup>) kWh；下降不另加能耗。运输交接阶段按题目运输能耗定义不另加悬停能耗。</li>
<li>中继建链阶段按悬停功率与通信附加功率计能耗；到较高悬停点的巡航海拔取航路净空海拔与端点海拔的最大值，避免负下降高度。</li>
<li>附件未规定工位数量、充电器数量或中继并发接入容量，因此准备与充电允许并行，中继允许同时服务多架运输机；运输换电耗时包含在固定准备时间内。</li>
<li>中继返航后的周转时间单独约束实体机再次可用；能源组件依据实际 SOC 按 90% 分界的两阶段模型充电。甘特图中的灰色斜线表示能源充电或中继机周转。</li>
<li>CSV 为 UTF-8 BOM 编码，时间统一为秒，能量为 kWh；图中为便于阅读换算成分钟。经纬度、海拔及服务起止时刻均可从结果表复核。</li>
</ul></section>
<section><h2>可复核结果表</h2><ul>{downloads}</ul><p>图像已嵌入本页面，可离线单独查看；CSV 链接需与报告文件保存在同一目录。</p></section>
<section><h2>原始输入与数据指纹</h2>{sources}</section>
<footer>图表及表格由可复现程序生成。原始输入文件未修改；完整数值以同目录求解结果与 CSV 为准。</footer>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    outputs.append(path)


def write_report(result: dict, data: dict, terrain, outdir: str | Path) -> list[str]:
    """Write five CSV tables, four/five PNG+PDF figures, and standalone HTML."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    energy_rows = _write_tables(result, data, terrain, out, outputs)
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
             "axes.edgecolor": "#8fa1ab", "text.color": "#163441",
             "axes.labelcolor": "#324e5c", "xtick.color": "#536a77", "ytick.color": "#536a77"}
    with plt.rc_context(style):
        figures = [
            (_map(result, data, terrain, out, outputs), "运输路线与中继位置", "底图为原始 DEM。不同颜色表示不同运输机型；红色虚线为中继往返航路。地图位置采用 DEM 中心的局部坐标。"),
            (_resources(result, energy_rows, data, out, outputs), "实体无人机与能源资源占用", "上图为实体机，下图为独立电池与能源组件；浅色段为准备，灰色斜线为充电或中继周转。图中包括最后架次之后的恢复满电时间，联合完成时间仍以最终返航计。"),
            (_deliveries(result, data, out, outputs), "货箱交付进度与期望时限", "累计曲线展示交付进度，散点图展示逐箱实际交付与期望送达时间。带硬约束货箱另以医疗期望时限和首批截止时限的较小值核验，详见逐箱表。"),
            (_communications(result, out, outputs), "运输全过程通信保障预约", "每条横线对应一个运输架次，蓝色表示网关保底保障，红色表示中继保底保障预约；下图展示预约主体的区间链路裕量下界。实际通信状态优先直连，独立采样统计和全时区间认证结论见上文，不能将预约颜色直接当作实际状态或服务时长。"),
        ]
        history = _history(result, out, outputs)
        if history is not None:
            figures.append((history, "候选方案比较", "候选方案的完成时间与总能耗比较。星号为最终选择；有限候选比较不能证明全局最优性。"))
    _write_html(result, data, terrain, out, outputs, figures)
    return [str(p) for p in outputs]
