"""Template-compatible tables, paper figures and a Chinese Q2 report.

All summaries use the accepted trip records. This module neither reroutes nor
reschedules tasks; physical verification belongs to the independent auditor.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import openpyxl

from q3.solver import charge_time


COLORS = {"A": "#247f99", "B": "#ce8a25", "C": "#7854a1"}
INK = "#193c4b"
GRAY = "#dce4e8"


def _csv(path, headers, rows):
    def value(item):
        if isinstance(item, (float, np.floating)):
            return "" if not math.isfinite(item) else format(float(item), ".12g")
        return item
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows([value(item) for item in row] for row in rows)


def _template_headers(data):
    template = Path(data["repo_root"])/"D题"/"结果提交模板.xlsx"
    workbook = openpyxl.load_workbook(template, read_only=True, data_only=True)
    try:
        headers = {name: list(next(workbook[name].iter_rows(values_only=True)))
                   for name in ("Q2_运输架次", "Q2_逐箱交付")}
    finally:
        workbook.close()
    if len(headers["Q2_运输架次"]) != 8 or len(headers["Q2_逐箱交付"]) != 4:
        raise ValueError("Unexpected Q2 submission-template columns; review the changed template")
    return headers


def _cycles(result, data):
    audited = {row["trip_id"]: row for row in result.get("validation", {}).get("resources", {}).get("cycles", [])}
    cycles = []
    for trip in result["trips"]:
        end = audited.get(trip["id"], {}).get("battery_ready")
        if end is None:
            end = trip["return"]+charge_time(trip["soc"], data["types"][trip["type"]]["charge_full"])
        cycles.append({"trip": trip, "battery_ready": end})
    return cycles


def _tables(result, data, output, paths, cycles):
    headers = _template_headers(data)
    trips = sorted(result["trips"], key=lambda trip: trip["id"])
    path = output/"Q2_运输架次.csv"
    _csv(path, headers["Q2_运输架次"], [[trip["id"], trip["drone"], trip["type"], trip["battery"], trip["start"],
          " → ".join(["O01", *trip["route"], "O01"]), trip["return"], trip["energy"]] for trip in trips])
    paths.append(path)
    delivered = {bid: (trip["id"], time) for trip in trips for bid, time in trip["deliveries"].items()}
    path = output/"Q2_逐箱交付.csv"
    _csv(path, headers["Q2_逐箱交付"], [[box["id"], *delivered[box["id"]][:1], box["node"], delivered[box["id"]][1]]
                                      for box in sorted(data["boxes"], key=lambda box: box["id"])])
    paths.append(path)
    occupations = []
    for row in cycles:
        trip = row["trip"]
        occupations.append(["运输无人机", trip["type"], trip["drone"], trip["id"], trip["start"],
                            trip["takeoff"], trip["return"], trip["return"], 0.0, trip["energy"], trip["soc"], "[开始,可用)"])
        occupations.append(["共享电池", trip["type"], trip["battery"], trip["id"], trip["start"],
                            trip["takeoff"], trip["return"], row["battery_ready"], row["battery_ready"]-trip["return"],
                            trip["energy"], trip["soc"], "[开始,可用)"])
    path = output/"Q2_资源占用.csv"
    _csv(path, ["资源类别", "机型编号", "资源编号", "架次编号", "准备开始（s）", "起飞时刻（s）", "返回O01时刻（s）",
                "下次可用时刻（s）", "返航后充电时间（s）", "架次能耗（kWh）", "返航SOC", "占用区间口径"],
         sorted(occupations, key=lambda row: (row[0], row[2], row[4])))
    paths.append(path)
    selected = result["metrics"]
    rows = []
    for row in result.get("search_history", []):
        feasible = bool(row.get("feasible"))
        same = feasible and all(abs(row.get(key, math.inf)-selected[key]) <= 1e-8
                                for key in ("weighted_tardiness", "makespan_s", "total_energy_kwh", "transport_sorties"))
        rows.append([row["label"], "是" if feasible else "否", row.get("seed", ""), row.get("transport_sorties", ""),
                     row.get("multi_stop_sorties", ""), row.get("weighted_tardiness", ""), row.get("late_boxes", ""),
                     row.get("makespan_s", ""), row.get("last_delivery_s", ""), row.get("total_energy_kwh", ""),
                     row.get("runtime_s", row.get("refinement", {}).get("runtime_s", "")), "是" if same else "否",
                     row.get("reason", ""), json.dumps(row.get("config", row.get("refinement", {})), ensure_ascii=False, sort_keys=True)])
    path = output/"Q2_候选比较.csv"
    _csv(path, ["候选编号", "构造可行", "随机种子", "运输架次数", "多服务区架次数", "加权迟到（加权s）", "迟到箱数",
                "全部返航时刻（s）", "末箱交付时刻（s）", "总能耗（kWh）", "计算时间（s）", "指标与入选方案相同",
                "失败原因", "搜索参数或精修记录"], rows)
    paths.append(path)


def _save(fig, output, stem, paths):
    for suffix in ("png", "pdf"):
        path = output/f"{stem}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white")
        paths.append(path)
    plt.close(fig)


def _style_axis(ax):
    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def _map(result, data, terrain, output, paths):
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    bounds = terrain.bounds
    dem = np.asarray(terrain.dem)
    if getattr(terrain, "nodata", None) is not None:
        dem = np.where(dem == terrain.nodata, np.nan, dem)
    image = ax.imshow(dem, extent=np.asarray([bounds[0], bounds[2], bounds[1], bounds[3]])/1000,
                      origin="upper", cmap="terrain", alpha=0.63, interpolation="nearest", zorder=0)
    seen = set()
    for trip in result["trips"]:
        for segment in trip["segments"]:
            a, b = np.asarray(segment["p0"][:2])/1000, np.asarray(segment["p1"][:2])/1000
            if np.linalg.norm(a-b) < 1e-10:
                continue
            key = (trip["type"], *sorted((tuple(a), tuple(b))))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS[trip["type"]], lw=1.7,
                    alpha=0.72, linestyle={"A": "-", "B": "--", "C": ":"}[trip["type"]], zorder=2)
    coordinates = []
    for nid, node in sorted(data["nodes"].items()):
        xy = np.asarray(terrain.xy(node["lon"], node["lat"]))/1000
        coordinates.append(xy)
        ax.scatter(*xy, marker="*" if nid == "O01" else "o", s=160 if nid == "O01" else 43,
                   color=INK if nid == "O01" else "white", edgecolors=INK, linewidths=0.9, zorder=5)
        dx, dy = (6, -16) if nid == "O01" else (6, 6)
        if nid in ("S012", "S014"):
            dx, dy = -6, 6
        ax.annotate(nid, xy, xytext=(dx, dy), textcoords="offset points", fontsize=10,
                    ha="right" if dx < 0 else "left", color=INK, zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6})
    xy = np.asarray(coordinates)
    ax.set_xlim(xy[:, 0].min()-0.7, xy[:, 0].max()+0.7)
    ax.set_ylim(xy[:, 1].min()-0.7, xy[:, 1].max()+0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Local east coordinate (km)")
    ax.set_ylabel("Local north coordinate (km)")
    ax.set_title("Q2 transport routes over the original DEM", fontsize=12, weight="bold", loc="left", pad=43)
    handles = [Line2D([], [], color=COLORS[tid], linestyle={"A": "-", "B": "--", "C": ":"}[tid],
                      lw=2, label=f"Type {tid}") for tid in "ABC"]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3,
              fontsize=10, frameon=False)
    _style_axis(ax)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, shrink=0.86)
    bar.set_label("DEM elevation (m ASL)", fontsize=10)
    bar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.12, top=0.84)
    _save(fig, output, "01_transport_routes", paths)


def _resources(result, data, cycles, output, paths):
    drones = [drone for tid in sorted(data["types"]) for drone in sorted(data["types"][tid]["drones"])]
    batteries = [battery for tid in sorted(data["types"]) for battery in sorted(data["types"][tid]["battery_ids"])]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [len(drones), len(batteries)]})
    for row in cycles:
        trip = row["trip"]
        start, takeoff, end = [trip[key]/60 for key in ("start", "takeoff", "return")]
        typ, y = trip["type"], drones.index(trip["drone"])
        axes[0].barh(y, takeoff-start, left=start, color=COLORS[typ], alpha=0.3, height=0.68)
        axes[0].barh(y, end-takeoff, left=takeoff, color=COLORS[typ], height=0.68)
        axes[0].text((start+end)/2, y, trip["id"], ha="center", va="center", color="white", fontsize=10,
                     bbox={"facecolor": COLORS[typ], "edgecolor": "none", "pad": 0.15})
        y = batteries.index(trip["battery"])
        axes[1].barh(y, end-start, left=start, color=COLORS[typ], height=0.68)
        axes[1].barh(y, row["battery_ready"]/60-end, left=end, color=GRAY, edgecolor="#a0b1ba",
                     hatch="///", linewidth=0.5, height=0.68)
    for ax, ids, title in zip(axes, (drones, batteries), ("Aircraft: preparation and transport", "Batteries: transport and full recharge")):
        ax.set_yticks(range(len(ids)), ids, fontsize=10)
        ax.set_ylim(len(ids)-0.35, -0.65)
        ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=10)
        _style_axis(ax)
        ax.grid(axis="y", visible=False)
    end = max(row["battery_ready"] for row in cycles)/60
    axes[1].set_xlim(0, end*1.035)
    axes[1].set_xlabel("Elapsed time (min)")
    handles = [Patch(facecolor=COLORS[tid], label=f"Type {tid}") for tid in "ABC"]
    handles += [Patch(facecolor="#84b0c0", alpha=0.4, label="Preparation"),
                Patch(facecolor=GRAY, edgecolor="#a0b1ba", hatch="///", label="Recharge")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.994), ncol=3,
               fontsize=10, frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.89, bottom=0.07, hspace=0.25)
    _save(fig, output, "02_resource_schedule", paths)


def _deliveries(result, data, output, paths):
    completed = {bid: t for trip in result["trips"] for bid, t in trip["deliveries"].items()}
    boxes = sorted(data["boxes"], key=lambda b: b["id"])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), gridspec_kw={"height_ratios": [1.25, 1]})
    actual = sorted(completed.values())
    hard_actual = sorted(completed[box["id"]] for box in boxes if math.isfinite(box["deadline"]))
    hard_due = sorted(box["deadline"] for box in boxes if math.isfinite(box["deadline"]))
    for times, label, color, style in ((actual, "All delivered", COLORS["A"], "-"),
                                      (hard_actual, "Hard boxes delivered", COLORS["C"], "-"),
                                      (hard_due, "Hard boxes due", "#bb4854", "--")):
        axes[0].step([0, *[t/60 for t in times]], [0, *range(1, len(times)+1)], where="post",
                     color=color, lw=2, linestyle=style, label=label)
    axes[0].set(xlim=(0, result["metrics"]["makespan_s"]/60*1.05), ylim=(0, len(boxes)+4),
                xlabel="Elapsed time (min)", ylabel="Cumulative box count")
    axes[0].set_title("Delivery progress", fontsize=12, weight="bold", loc="left", pad=40)
    axes[0].legend(fontsize=10, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18),
                   ncol=3, columnspacing=0.8, handlelength=1.5, handletextpad=0.4)
    lateness = np.asarray([max(0.0, completed[box["id"]]-box["due"])/60 for box in boxes])
    x = np.arange(1, len(boxes)+1)
    flags = np.asarray([math.isfinite(box["deadline"]) for box in boxes])
    for flag, color, marker, label in ((True, COLORS["C"], "o", "Has hard deadline"),
                                       (False, COLORS["A"], "x", "Other box")):
        mask = flags == flag
        axes[1].scatter(x[mask], lateness[mask], color=color, marker=marker, s=27, label=label, zorder=3)
    axes[1].axhline(0, color="#a0b1ba", lw=1, zorder=1)
    centers, labels = [], []
    for nid in sorted({box["node"] for box in boxes}):
        indices = [i+1 for i, box in enumerate(boxes) if box["node"] == nid]
        centers.append(sum(indices)/len(indices))
        labels.append(nid)
        if min(indices) > 1:
            axes[1].axvline(min(indices)-0.5, color="#dae2e6", lw=0.7, zorder=0)
    axes[1].set_xticks(centers, labels, rotation=65, ha="right", fontsize=10)
    axes[1].set(xlim=(0, len(boxes)+1), ylim=(-max(0.25, lateness.max()*0.07), max(1.0, lateness.max()*1.3)),
                xlabel="All boxes grouped by service area", ylabel="Tardiness (min)")
    axes[1].set_title(f"Box-level tardiness: {len(boxes)} boxes", fontsize=12, weight="bold", loc="left")
    if np.max(lateness) <= 1e-9:
        axes[1].text(0.5, 0.55, "All boxes delivered by their expected due time", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=11, color=COLORS["A"])
    axes[1].legend(fontsize=10, frameon=False, loc="upper center", ncol=2)
    for ax in axes:
        _style_axis(ax)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.85, hspace=0.35)
    _save(fig, output, "03_delivery_performance", paths)


def _candidates(result, output, paths):
    rows = [row for row in result.get("search_history", []) if row.get("feasible") and "total_energy_kwh" in row]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for multi in (False, True):
        for late in (False, True):
            group = [row for row in rows if (row.get("multi_stop_sorties", 0) > 0) == multi
                     and (row["weighted_tardiness"] > 1e-6) == late]
            if not group:
                continue
            ax.scatter([row["makespan_s"]/60 for row in group], [row["total_energy_kwh"] for row in group],
                       marker="D" if multi else "o", c="#dc9852" if late else COLORS["A"],
                       s=45, alpha=0.75, edgecolors="white", linewidths=0.6, zorder=2)
    chosen = result["metrics"]
    ax.scatter(chosen["makespan_s"]/60, chosen["total_energy_kwh"], marker="*", s=240,
               color="#bd3e50", edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate("Selected", (chosen["makespan_s"]/60, chosen["total_energy_kwh"]), xytext=(8, 7),
                textcoords="offset points", fontsize=11, weight="bold", color="#bd3e50")
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS["A"], label="Zero tardiness"),
               Line2D([], [], marker="o", linestyle="none", color="#dc9852", label="Positive tardiness"),
               Line2D([], [], marker="o", linestyle="none", color="#607987", label="Single-area sorties only"),
               Line2D([], [], marker="D", linestyle="none", color="#607987", label="Includes multi-area sorties")]
    ax.legend(handles=handles, ncol=2, fontsize=10, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.21))
    ax.set(xlabel="All aircraft returned (min)", ylabel="Total energy (kWh)")
    ax.set_title("Bounded search candidates and selected schedule", fontsize=12, weight="bold", loc="left", pad=64)
    ax.margins(x=0.13, y=0.13)
    _style_axis(ax)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.74)
    _save(fig, output, "04_candidate_tradeoff", paths)


def _markdown(result, data, output, paths, cycles):
    m, audit, bounds = result["metrics"], result.get("validation", {}), result.get("search_bounds", {})
    rows = result.get("search_history", [])
    constructed = [row for row in rows if row["label"].startswith("S")]
    feasible = [row for row in constructed if row.get("feasible")]
    multi = [row for row in feasible if row.get("multi_stop_sorties", 0) > 0]
    lines = ["# 问题二：不考虑通信约束的物资运输调度", "",
             "本方案直接读取原始货箱、机型、电池及DEM数据，独立执行货箱组批、访问路线、机型选择和资源排程，不读取问题三的方案文件，也不施加通信或中继约束。", "",
             f"独立核验：{'通过' if audit.get('passed') else '请检查 validation.json'}；检查数{audit.get('check_count', '未记录')}；错误数{audit.get('error_count', '未记录')}。", "",
             "## 核心结果", "", "| 指标 | 结果 |", "|---|---:|",
             f"| 完成交付 | {m['boxes_delivered']}箱 / {len(data['boxes'])}箱 |",
             f"| 运输架次 | {m['transport_sorties']} |",
             f"| A/B/C型架次 | {' / '.join(str(sum(t['type']==tid for t in result['trips'])) for tid in 'ABC')} |",
             f"| 全部返航时刻 | {m['makespan_s']/60:.6f} min |",
             f"| 末箱交付时刻 | {m['last_delivery_s']/60:.6f} min |",
             f"| 总能耗 | {m['total_energy_kwh']:.9f} kWh |",
             f"| 加权迟到 | {m['weighted_tardiness']:.6f} 加权秒 |",
             f"| 迟到箱数 | {m['late_boxes']} |",
             f"| 硬时限违约箱数 | {m['hard_deadline_violations']} |",
             f"| 最低返航SOC | {100*m['min_transport_soc']:.6f}% |",
             f"| 多服务区架次 | {m['multi_stop_sorties']} |", "",
             "目标按加权迟到、全部返航时刻、总能耗、运输架次数字典序比较；不同目标的优先级属于明确的建模选择。"]
    if m["weighted_tardiness"] <= 1e-9:
        lines.extend(["", "本方案所有货箱均在期望时限内完成交付，加权迟到为0，达到该非负指标的理论下界。这只证明第一层迟到目标达到下界，不证明后续完工时间、能耗或架次数的全局最优。"])
    lines.extend(["", "![运输航线与地形](01_transport_routes.png)", "",
                  "## 资源使用与充电", "", "| 机型 | 无人机库存 | 使用编号数 | 电池库存 | 使用编号数 | 架次 | 能耗/kWh |", "|---|---:|---:|---:|---:|---:|---:|"])
    for tid, spec in sorted(data["types"].items()):
        trips = [trip for trip in result["trips"] if trip["type"] == tid]
        lines.append(f"| {tid} | {len(spec['drones'])} | {len({trip['drone'] for trip in trips})} | {len(spec['battery_ids'])} | {len({trip['battery'] for trip in trips})} | {len(trips)} | {sum(trip['energy'] for trip in trips):.6f} |")
    lines.extend(["", "无人机从准备开始占用至返回O01；电池从准备开始占用至返航后按两阶段规则充满。区间为左闭右开，相同时刻释放后可再次使用。运输机返航即可使用另一块同型满电电池开展准备，原电池继续充电；没有添加题目未给定的充电端口数量约束。", "",
                  "甘特图显示全部8架运输机和14块库存电池，空行表示该编号未使用；浅色为准备装载，深色为运输任务，阴影为电池充电。", "",
                  "![机体与电池占用](02_resource_schedule.png)", "",
                  "## 逐箱交付", "",
                  "医疗箱的期望时限作为硬约束；带首批保障标记的指定货箱同时满足首批截止时间。一个箱有两个硬期限时取较早者，其余货箱允许迟到但计入应急优先系数加权的迟到目标。图中逐箱点按服务区排列，CSV可查全部80箱对应架次与准确送达时刻。", "",
                  "![交付进度及逐箱迟到](03_delivery_performance.png)", "",
                  "## 搜索范围与最优性边界", "",
                  f"固定种子{bounds.get('seed', result.get('seed'))}，共执行{len(constructed)}次构造，其中{len(feasible)}次生成完整可行候选、{len(multi)}次包含多服务区架次。候选允许每架次最多{bounds.get('max_stops', 3)}个服务区，在种子服务区最近{bounds.get('neighbors', 4)}个候选邻区内枚举访问排列，并生成贪心组批的可行前缀。每步保留静态评分前{bounds.get('retained_ranked_plans', 32)}个候选和全部单服务区候选。", "",
                  f"再对少量选定的箱集、路线和机型结构进行CP-SAT时序与实体分配精修，总预算上限{bounds.get('refinement_total_seconds', 0):g}秒。采用1秒时间网格，任务与电池占用时长及送达偏移向上取整、硬期限向下取整；最终重新构造连续时间轨迹，以实际指标排序并独立核验。CP-SAT状态与界仅适用于对应固定结构的保守整数秒模型。", ""])
    if m["multi_stop_sorties"] == 0:
        lines.extend(["入选方案每架次只服务一个区，是当前目标次序下的候选比较结果；程序并未把单区访问设为问题二硬约束。多服务区方案纳入搜索，具体数量与指标见候选比较表。", ""])
    lines.extend(["候选图同时展示构造与成功精修方案：圆点表示全部单区架次，菱形表示包含多区架次；蓝色为零迟到，橙色为正迟到。它是有界搜索记录，不是全部可行解集或全局Pareto前沿。", "",
                  "![候选比较](04_candidate_tradeoff.png)", "",
                  "## 物理口径与复现", "",
                  "采用原始30米DEM像元分片常值地形和局部WGS84米制坐标。每段使用先爬升、水平直飞、下降的航迹，巡航海拔为沿线最高DEM高程加50米与两端作业海拔的最大值；服务区交付高度30米。能耗使用载荷等效航程的水平项及重力爬升项，下降和交付悬停不另计运输能耗。这些是题面缺失公式下明确补充的口径，不能替代实机试验。", "",
                  "所有报告数据来自同目录 solution.json；validation.json 独立重建时序、能耗、SOC、逐箱期限和资源释放。manifest.json 记录原始输入与代码及输出的SHA-256。报告展示精度与机器可读原始浮点值不同，资源冲突验证始终使用未四舍五入的时刻。", "",
                  "## 文件", "", "- `Q2_运输架次.csv`：原结果模板8列。", "- `Q2_逐箱交付.csv`：原结果模板4列。",
                  "- `Q2_资源占用.csv`：逐架次机体、电池及充电释放时刻。", "- `Q2_候选比较.csv`：构造及成功精修候选的指标与参数。",
                  "- 四张图均提供同名PDF，供论文直接引用；CSV为UTF-8 BOM编码、LF换行，时间字段以秒计。", ""])
    path = output/"report.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    paths.append(path)


def write_report(result: dict, data: dict, terrain, output: Path) -> list[Path]:
    """Write four CSV tables, four PNG/PDF pairs and a Chinese report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    cycles = _cycles(result, data)
    _tables(result, data, output, paths, cycles)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#99adb8",
                         "xtick.color": "#536b77", "ytick.color": "#536b77"}):
        _map(result, data, terrain, output, paths)
        _resources(result, data, cycles, output, paths)
        _deliveries(result, data, output, paths)
        _candidates(result, output, paths)
    _markdown(result, data, output, paths, cycles)
    return paths
