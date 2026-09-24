"""Submission CSVs, paper-readable figures, and a Chinese Q4 report.

The supplied result is reported without changing any Q3/Q4 decision. Resource
counts are counts, not monetary costs. Partition maps show point membership,
never invented geographic boundaries or an implied connected service region.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


KEYS = ("drone_A", "drone_B", "drone_C", "battery_A", "battery_B", "battery_C", "relay", "module")
NAMES = ("A型运输无人机", "B型运输无人机", "C型运输无人机", "A型电池组", "B型电池组", "C型电池组", "中继无人机", "中继能源组件")
LABELS = ("A drones", "B drones", "C drones", "A batteries", "B batteries", "C batteries", "Relay drones", "Energy modules")
TEMPLATE_HEADERS = ["K（2或3）", "任务组编号", "服务区列表", "A型运输无人机数", "B型运输无人机数", "C型运输无人机数",
                    "A型电池组数", "B型电池组数", "C型电池组数", "中继无人机数", "中继能源组件数"]
COLORS = ("#177e9e", "#d68b24", "#7057a8")


def _csv(path: Path, headers: list, rows: list) -> None:
    def value(item):
        if isinstance(item, (float, np.floating)):
            if not math.isfinite(item):
                raise ValueError(f"Non-finite Q4 report value: {item}")
            return round(float(item), 9)
        return item
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows([value(item) for item in row] for row in rows)


def _group_text(candidate: dict) -> str:
    return " | ".join(f"{group['id']}:" + ",".join(group["nodes"]) for group in candidate["groups"])


def _gap_text(requirements: dict) -> str:
    pairs = [f"{name}{requirements[key]}" for key, name in zip(KEYS, NAMES) if requirements[key]]
    return "、".join(pairs) if pairs else "无"


def _write_csvs(result: dict, output: Path, paths: list[Path]) -> None:
    selected_rows, balanced_rows, candidate_rows, allocation_rows = [], [], [], []
    for k in (2, 3):
        part = result["partitions"][str(k)]
        for role, label, rows in [("selected", "资源优先", selected_rows), ("balanced", "均衡优先", balanced_rows)]:
            candidate = part.get(role)
            if candidate is None:
                continue
            for group in candidate["groups"]:
                rows.append([k, group["id"], ";".join(group["nodes"]), *[group["resources"][key] for key in KEYS]])
                for item in group.get("allocations", []):
                    allocation_rows.append([k, label, candidate["id"], group["id"], item["task_id"],
                                             item["resource_key"], item["resource_id"], item["start"], item["end"],
                                             item["end"] - item["start"], "[开始,结束)"])
        pareto = set(part.get("pareto_ids", []))
        for candidate in part["all"]:
            candidate_rows.append([k, candidate["id"], _group_text(candidate),
                                   candidate["total_resources"], candidate["shortage_total"],
                                   candidate["work_cv"], candidate["box_cv"],
                                   int(candidate["id"] in pareto),
                                   int(part.get("selected") is not None and candidate["id"] == part["selected"]["id"]),
                                   int(part.get("balanced") is not None and candidate["id"] == part["balanced"]["id"]),
                                   *[candidate["totals"][key] for key in KEYS],
                                   *[candidate["shortages"][key] for key in KEYS]])
    for filename, rows in [("Q4_分区配置.csv", selected_rows), ("Q4_均衡对照.csv", balanced_rows)]:
        path = output / filename
        _csv(path, TEMPLATE_HEADERS, rows)
        paths.append(path)
    path = output / "Q4_候选比较.csv"
    _csv(path, ["K", "候选编号", "分组", "资源总件数", "库存缺口总件数", "工作量CV", "货箱数CV",
                "是否Pareto", "是否资源优先选择", "是否均衡优先选择",
                *[name + "需求" for name in NAMES], *[name + "缺口" for name in NAMES]], candidate_rows)
    paths.append(path)
    path = output / "Q4_资源分配.csv"
    _csv(path, ["K", "选择规则", "候选编号", "任务组编号", "架次编号", "资源类型", "独立资源编号", "占用开始（s）",
                "占用结束（s）", "占用时长（s）", "区间口径"], allocation_rows)
    paths.append(path)
    compare_rows = []
    scenarios = [("未分组共享池下界", 1, "", result["baseline"]["resources"]),
                 ("现有库存", 0, "", result["inventory"])]
    for k in (2, 3):
        for role, label in [("selected", "资源优先"), ("balanced", "均衡优先")]:
            candidate = result["partitions"][str(k)].get(role)
            if candidate:
                scenarios.append((label, k, candidate["id"], candidate["totals"]))
    for label, k, cid, totals in scenarios:
        gaps = [max(0, totals[key] - result["inventory"][key]) for key in KEYS]
        spare = [max(0, result["inventory"][key] - totals[key]) for key in KEYS]
        added = [totals[key] - result["baseline"]["resources"][key] for key in KEYS]
        compare_rows.append([label, k, cid, sum(totals.values()), sum(gaps),
                             *[totals[key] for key in KEYS], *gaps, *spare, *added])
    path = output / "Q4_资源比较.csv"
    _csv(path, ["方案", "K", "候选编号", "资源总件数", "库存缺口总件数",
                *[name + "需求或库存" for name in NAMES], *[name + "缺口" for name in NAMES],
                *[name + "库存余量" for name in NAMES], *[name + "相对共享池增量" for name in NAMES]], compare_rows)
    paths.append(path)


def _save(fig, output: Path, name: str, paths: list[Path]) -> None:
    for extension in ("pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        paths.append(path)
    plt.close(fig)


def _maps(result: dict, data: dict, output: Path, paths: list[Path]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.5), sharex=True, sharey=True)
    nodes = data["nodes"]
    valid_nodes = [n for key, n in nodes.items() if key != "O01"]
    all_lon = [n["lon"] for n in valid_nodes] + [nodes["O01"]["lon"]]
    all_lat = [n["lat"] for n in valid_nodes] + [nodes["O01"]["lat"]]
    for k, ax in zip((2, 3), axes):
        candidate = result["partitions"][str(k)].get("selected")
        if not candidate:
            ax.text(0.5, 0.5, "No admissible partition", transform=ax.transAxes, ha="center")
            continue
        for i, group in enumerate(candidate["groups"]):
            own = [nodes[nid] for nid in group["nodes"]]
            ax.scatter([n["lon"] for n in own], [n["lat"] for n in own], s=58,
                       color=COLORS[i], edgecolors="white", linewidths=0.7,
                       label=f"{group['id']} ({len(own)} {'area' if len(own) == 1 else 'areas'})", zorder=3)
            for node in own:
                right_edge = node["lon"] > max(all_lon) - 0.008
                # Separate the nearby S009/S012 labels at the paper figure size.
                label_y = -13 if node["id"] == "S012" else 5
                ax.annotate(node["id"], (node["lon"], node["lat"]),
                            xytext=(-5 if right_edge else 5, label_y), textcoords="offset points",
                            fontsize=10, ha="right" if right_edge else "left", weight="normal")
        depot = nodes["O01"]
        ax.scatter(depot["lon"], depot["lat"], s=115, marker="*", color="#263f4b", zorder=4)
        ax.annotate("O01", (depot["lon"], depot["lat"]), xytext=(5, -13),
                    textcoords="offset points", fontsize=10, weight="bold")
        ax.set_title(f"{k} groups: {candidate['id']}",
                     fontsize=12, weight="bold", loc="left", pad=40)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.145), ncol=k,
                  fontsize=10, frameon=False, handletextpad=0.3, columnspacing=1.0)
        ax.set_ylabel("Latitude (degrees N)", fontsize=11)
        ax.grid(alpha=0.20)
        ax.set_axisbelow(True)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_aspect(1 / math.cos(math.radians(sum(all_lat) / len(all_lat))))
    axes[1].set_xlabel("Longitude (degrees E)", fontsize=11)
    axes[1].set_xlim(min(all_lon) - 0.008, max(all_lon) + 0.008)
    axes[1].set_ylim(min(all_lat) - 0.010, max(all_lat) + 0.012)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.875, bottom=0.09, hspace=0.45)
    _save(fig, output, "01_partition_maps", paths)


def _resources(result: dict, output: Path, paths: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    x = np.arange(len(KEYS))
    schemes = [("Shared pool minimum", result["baseline"]["resources"], COLORS[0])]
    for k, color in [(2, COLORS[1]), (3, COLORS[2])]:
        candidate = result["partitions"][str(k)].get("selected")
        if candidate:
            schemes.append((f"{k} groups (resource first)", candidate["totals"], color))
    width = 0.72 / len(schemes)
    for i, (label, resources, color) in enumerate(schemes):
        offset = (i - (len(schemes) - 1) / 2) * width
        values = [resources[key] for key in KEYS]
        bars = ax.bar(x + offset, values, width=width, color=color, label=label, zorder=3)
        ax.bar_label(bars, padding=2, fontsize=10)
    inventory = [result["inventory"][key] for key in KEYS]
    ax.plot(x, inventory, linestyle="none", marker="_", color="#122f3a", markersize=30,
            markeredgewidth=2.2, label="Existing inventory", zorder=5)
    ax.set_xticks(x, LABELS, rotation=32, ha="right", fontsize=10)
    ax.set_ylabel("Number of resources", fontsize=11)
    ax.set_title("Independent groups versus pooled resources", loc="left", fontsize=12, weight="bold", pad=13)
    maximum = max([*inventory, *[r[k] for _, r, _ in schemes for k in KEYS]])
    ax.set_ylim(0, maximum + 2.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=2, fontsize=10, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.25)
    _save(fig, output, "02_resource_requirements", paths)


def _tradeoff(result: dict, output: Path, paths: list[Path]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8))
    gap_values = [c["shortage_total"] for part in result["partitions"].values() for c in part["all"]]
    max_gap = max([1, *gap_values])
    scatter = None
    for k, ax in zip((2, 3), axes):
        part = result["partitions"][str(k)]
        candidates = part["all"]
        scatter = ax.scatter([c["total_resources"] for c in candidates], [c["work_cv"] for c in candidates],
                             c=[c["shortage_total"] for c in candidates], vmin=0, vmax=max_gap,
                             cmap="viridis", s=33, alpha=0.68, edgecolors="none", zorder=2)
        pareto = [c for c in candidates if c["id"] in part["pareto_ids"]]
        ax.scatter([c["total_resources"] for c in pareto], [c["work_cv"] for c in pareto],
                   facecolors="none", edgecolors="#263f4b", linewidths=1.2, s=83, zorder=3)
        for role, marker, color, offset in [("selected", "*", "#bd3e50", (8, 8)),
                                            ("balanced", "D", "#db8425", (8, -15))]:
            candidate = part.get(role)
            if not candidate:
                continue
            ax.scatter(candidate["total_resources"], candidate["work_cv"], marker=marker,
                       color=color, edgecolors="white", linewidths=0.7,
                       s=165 if marker == "*" else 67, zorder=5)
            ax.annotate(candidate["id"], (candidate["total_resources"], candidate["work_cv"]),
                        xytext=offset, textcoords="offset points", fontsize=10, color=color, weight="bold")
        ax.set_title(f"{k} groups: all {part['enumerated']} admissible partitions", loc="left", fontsize=12, weight="bold")
        ax.set_xlabel("Total resource count", fontsize=11)
        ax.set_ylabel("Workload CV", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.20)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.15, y=0.20)
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#263f4b", markersize=8, label="Pareto candidate"),
               Line2D([], [], marker="*", linestyle="none", color="#bd3e50", markersize=11, label="Resource first"),
               Line2D([], [], marker="D", linestyle="none", color="#db8425", markersize=6, label="Balance first")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, 0.995), ncol=3,
               fontsize=10, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.12, right=0.84, top=0.91, bottom=0.08, hspace=0.44)
    if scatter is not None:
        cax = fig.add_axes([0.88, 0.20, 0.028, 0.54])
        bar = fig.colorbar(scatter, cax=cax)
        bar.set_label("Total inventory shortage", fontsize=10)
        bar.ax.tick_params(labelsize=10)
        bar.locator = MaxNLocator(integer=True)
        bar.update_ticks()
    _save(fig, output, "03_tradeoff", paths)


def _markdown(result: dict, q3: dict, data: dict, output: Path, paths: list[Path]) -> None:
    validation = result.get("validation", {})
    status = "独立核验通过。" if validation.get("passed") is True else "独立核验状态请参见同目录核验文件。"
    atom_description = "；".join(atom["id"] + "={" + ", ".join(atom["nodes"]) + "}" for atom in result["atoms"])
    lines = ["# 问题四：固定任务下的分区与独立资源配置", "", status, "",
             "本报告固定问题三的货箱组批、访问顺序、任务时间、中继位置及通信保障关联，不复制中继架次。共享一个固定运输架次或中继架次的服务区合并为不可拆分单元，再穷举这些单元的非空分组。各组独立拥有资源，同型实体编号允许重新分配，但执行期间不跨组借用。", "",
             f"共有 **{len(result['atoms'])} 个不可拆分单元**：{atom_description}。", "",
             "资源优先规则依次最小化库存缺口总件数、资源总件数、工作量变异系数（CV）；均衡对照则先最小化工作量CV，再比较缺口和资源总件数。不同机型或资源不能互相抵扣，资源总件数仅用作透明的计数指标，并非采购成本。", "",
             "## 分区与指标", "", "| K | 规则 | 候选 | 总资源件数 | 库存缺口 | 工作量CV | 货箱数CV |", "|---:|---|---|---:|---:|---:|---:|"]
    for k in (2, 3):
        part = result["partitions"][str(k)]
        for role, label in [("selected", "资源优先"), ("balanced", "均衡优先")]:
            candidate = part.get(role)
            if candidate:
                lines.append(f"| {k} | {label} | {candidate['id']} | {candidate['total_resources']} | {candidate['shortage_total']} | {candidate['work_cv']:.6f} | {candidate['box_cv']:.6f} |")
    for k in (2, 3):
        part = result["partitions"][str(k)]
        lines.extend(["", f"**{k}组**：完整枚举{part['enumerated']}种无标签分区，其中{part['zero_gap_count']}种不超过现有库存。"])
    lines.extend(["", "工作量为组内所有运输及中继架次从准备开始至返航的时长之和；充电与中继机返航周转用于资源占用，不重复计入工作量。CV采用各组工作量的总体标准差除以均值。分区不要求地理连通，地图仅展示点位归属，不绘制虚构区域边界。", "",
                  "![资源优先分组点位](01_partition_maps.png)", "", "## 分组细节", ""])
    for k in (2, 3):
        for role, label in [("selected", "资源优先"), ("balanced", "均衡优先")]:
            candidate = result["partitions"][str(k)].get(role)
            if candidate is None:
                continue
            lines.extend([f"### {k}组 · {label} · {candidate['id']}", "",
                          "| 任务组 | 服务区 | 货箱数 | 质量/kg | 累计任务工作量/min |", "|---|---|---:|---:|---:|"])
            for group in candidate["groups"]:
                lines.append(f"| {group['id']} | {', '.join(group['nodes'])} | {group['box_count']} | {group['mass_kg']:.1f} | {group['work_s'] / 60:.2f} |")
            lines.extend(["", f"库存缺口：{_gap_text(candidate['shortages'])}。库存余量：{_gap_text(candidate['surplus'])}。",
                          f"相对未分组共享池的额外配置：{_gap_text(candidate['added_vs_pool'])}。", ""])
    lines.extend(["## 资源需求与冗余口径", "", "| 资源 | 共享池最少需要 | 库存 | 2组资源优先 | 3组资源优先 |", "|---|---:|---:|---:|---:|"])
    for key, name in zip(KEYS, NAMES):
        counts = [result["partitions"][str(k)]["selected"]["totals"][key]
                  if result["partitions"][str(k)].get("selected") else "不适用" for k in (2, 3)]
        lines.append(f"| {name} | {result['baseline']['resources'][key]} | {result['inventory'][key]} | {counts[0]} | {counts[1]} |")
    lines.extend(["", "共享池最少需要是固定全部任务后允许同型资源统一调配的峰值占用下界，不等于问题三原方案实际使用过的不同编号数量。各组资源需求取该组对应占用区间的最大重叠数；不同组需求相加会失去组间错峰共享机会。库存余量是库存减需求的非负部分；分组额外配置是分组需求减共享池下界，两者不是同一个冗余指标。", "",
                  "![分类资源需求](02_resource_requirements.png)", "", "## 工作量与资源的权衡", "",
                  "下图展示全部已枚举候选。颜色为库存缺口，轮廓圈为同时比较缺口、资源总件数和工作量CV的Pareto候选，星号与菱形分别为两种优先规则的选择。二维投影中看似被支配的点可能在库存缺口维度更优。", "",
                  "![全候选权衡](03_tradeoff.png)", "", "## 计算边界与复核", "",
                  "资源占用区间采用左闭右开 [开始,结束)，相同时刻先释放再占用，保留问题三原始浮点时刻。运输机占用至返航，运输电池占用至返航后充满；中继机占用至返航后完成周转，能源组件占用至充满。输出的独立资源编号及逐架次占用构成组内可执行的资源分配证据。", "",
                  "穷举结论仅对固定问题三方案、不复制中继任务、同型资源可重新编号及当前目标排序的有限分区模型成立。不能推广为允许重排路线、改变时序或复制通信任务后的全局最优，也不能据此证明其他问题三方案没有更低的分区代价。", "",
                  "## 文件", "", "- `Q4_分区配置.csv`：原模板11列，2组和3组资源优先配置。", "- `Q4_均衡对照.csv`：相同模板列的均衡优先对照。",
                  "- `Q4_候选比较.csv`：全部候选、需求、缺口、CV及选择标记。", "- `Q4_资源分配.csv`：两种规则下各组独立资源的逐架次占用。",
                  "- `Q4_资源比较.csv`：共享池、库存及两种规则配置的分类对照。", "- 三张图均有同名PDF版本供论文引用；CSV使用UTF-8 BOM编码，所有时间字段以秒计。", ""])
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    paths.append(path)


def export_report(result: dict, q3: dict, data: dict, output_dir: str | Path) -> list[str]:
    """Write five CSVs, three PNG/PDF figure pairs, and a Chinese Markdown report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    _write_csvs(result, output, paths)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": "#8fa1ab", "text.color": "#173746",
                         "axes.labelcolor": "#324e5c", "xtick.color": "#536a77", "ytick.color": "#536a77"}):
        _maps(result, data, output, paths)
        _resources(result, output, paths)
        _tradeoff(result, output, paths)
    _markdown(result, q3, data, output, paths)
    return [str(path) for path in paths]
