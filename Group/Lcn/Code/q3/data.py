"""Read the unmodified Q3 attachments into a unit-consistent data structure.

Distances/heights are metres; durations are seconds; energy is kWh; power is
kW.  ``deadline`` is the minimum of the applicable medical and first-delivery
deadlines, and is infinity when a box has no hard deadline.  Reading uses
openpyxl in read-only mode and never writes to the supplied workbooks.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl


def _root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        if not (root / "D题" / "数据").is_dir():
            raise FileNotFoundError(f"Repository data directory not found: {root}")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "D题" / "数据").is_dir():
            return root
    raise FileNotFoundError("Cannot find repository root containing D题/数据")


def _read_rows(path: Path, sheet: str = "数据") -> list[tuple[Any, ...]]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(book[sheet].iter_rows(values_only=True))
    finally:
        book.close()


def _table(rows: list[tuple[Any, ...]], header: str) -> list[tuple[Any, ...]]:
    """Return one table after its first-column header, stopping at a blank row."""
    for i, row in enumerate(rows):
        if row[0] == header:
            result = []
            for value in rows[i + 1 :]:
                if value[0] is None:
                    break
                result.append(value)
            return result
    raise ValueError(f"Missing workbook table header: {header}")


def _float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite input for {name}: {value}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load all five original parameter workbooks and locate the original DEM.

    ``types`` is keyed by A/B/C and ``nodes`` by O01/S001/.../S015.  Drone and
    battery identifiers are deterministic; battery/energy-component identifiers
    are generated locally because the attachments specify only their counts.
    """
    root = _root(repo_root)
    base = root / "D题" / "数据" / "无人机应急物资运输基础数据"
    source_paths = {
        "nodes": base / "调度中心与服务区.xlsx",
        "boxes": base / "物资需求与配送时限.xlsx",
        "types": base / "运输无人机数据.xlsx",
        "relay": base / "中继无人机数据.xlsx",
        "links": base / "通信链路参数.xlsx",
        "problem": root / "D题" / "山区洪涝灾害下无人机运输与通信协同优化.docx",
    }
    dem_candidates = list((root / "D题" / "数据").rglob("*30米DEM.tif"))
    if len(dem_candidates) != 1:
        raise ValueError(f"Expected one original 30 m DEM, found {dem_candidates}")
    source_paths["dem"] = dem_candidates[0]

    node_rows = _read_rows(source_paths["nodes"])
    nodes: dict[str, dict[str, Any]] = {}
    for header, is_depot in [("调度中心编号", True), ("服务区编号", False)]:
        for row in _table(node_rows, header):
            identifier = str(row[0])
            nodes[identifier] = {
                "id": identifier,
                "name": str(row[1]),
                "lon": _float(row[2], "longitude"),
                "lat": _float(row[3], "latitude"),
                "z": _float(row[4], "ground altitude"),
                "pop": 0 if is_depot else int(row[5]),
                "is_depot": is_depot,
                "work_agl": 0.0 if is_depot else 30.0,
            }

    type_rows = _read_rows(source_paths["types"])
    fields = [
        "mass_empty", "payload", "volume", "speed", "range_empty",
        "range_full", "energy", "reserve", "prepare", "load_per_box",
        "service_base", "service_per_box", "up_speed", "down_speed",
        "eta_up", "eta_down",
    ]
    types: dict[str, dict[str, Any]] = {}
    for row in _table(type_rows, "机型编号"):
        identifier = str(row[0])
        item = {key: _float(value, key) for key, value in zip(fields, row[2:])}
        item.update(id=identifier, name=str(row[1]), drones=[])
        item["reserve"] /= 100.0
        types[identifier] = item
    for row in _table(type_rows, "无人机编号"):
        if row[2] != "O01":
            raise ValueError("This solver requires all transport drones to start at O01")
        types[str(row[1])]["drones"].append(str(row[0]))
    inventory_header = next(
        i for i, row in enumerate(type_rows) if row[0] == "共享电池库存"
    )
    for row in _table(type_rows[inventory_header:], "机型编号"):
        item = types[str(row[0])]
        item["batteries"] = int(row[1])
        item["charge_full"] = _float(row[2], "charge_full")
        item["battery_ids"] = [
            f"BAT-{row[0]}-{i:02d}" for i in range(1, item["batteries"] + 1)
        ]

    box_rows = _read_rows(source_paths["boxes"], "逐箱货箱清单")
    boxes = []
    for row in _table(box_rows, "货箱编号"):
        first = str(row[5]).strip() == "是"
        if str(row[5]).strip() not in {"是", "否"}:
            raise ValueError(f"Unexpected first-delivery marker: {row[5]}")
        due = _float(row[7], "due")
        medical = str(row[2]) == "医疗物资"
        first_deadline = _float(row[6], "first_deadline") if first else math.inf
        deadline = min(first_deadline, due if medical else math.inf)
        boxes.append({
            "id": str(row[0]), "node": str(row[1]), "kind": str(row[2]),
            "mass": _float(row[3], "mass"),
            "volume": _float(row[4], "volume"), "first": first,
            "medical": medical, "first_deadline": first_deadline,
            "due": due, "weight": _float(row[8], "priority coefficient"),
            "deadline": deadline,
        })

    relay_rows = _read_rows(source_paths["relay"])
    relay_values = _table(relay_rows, "机型编号")
    if len(relay_values) != 1:
        raise ValueError("Expected exactly one relay drone type")
    row = relay_values[0]
    relay_fields = [
        "mass_empty", "module_mass", "mass", "speed", "power_cruise",
        "energy", "reserve", "prepare", "link_time", "turnaround",
        "up_speed", "down_speed", "eta_up", "eta_down", "power_hover",
        "power_comm", "max_agl",
    ]
    relay = {
        key: _float(value, key) for key, value in zip(relay_fields, row[2:])
    }
    relay.update(id=str(row[0]), name=str(row[1]), drones=[])
    relay["reserve"] /= 100.0
    for row in _table(relay_rows, "中继无人机编号"):
        if row[2] != "O01":
            raise ValueError("This solver requires all relay drones to start at O01")
        relay["drones"].append(str(row[0]))
    inventory_header = next(
        i for i, row in enumerate(relay_rows) if row[0] == "共享能源组件库存"
    )
    row = _table(relay_rows[inventory_header:], "机型编号")[0]
    relay["components"] = int(row[1])
    relay["charge_full"] = _float(row[2], "relay charge_full")
    relay["component_ids"] = [f"ENERGY-R-{i:02d}" for i in range(1, int(row[1]) + 1)]

    link_rows = _read_rows(source_paths["links"])
    link_raw = {
        (str(row[0]), str(row[3])): _float(row[4], str(row[1]))
        for row in _table(link_rows, "参数类别")
    }
    links = {
        "frequency_mhz": link_raw["传播参数", "f"],
        "system_loss": link_raw["传播参数", "Lsys"],
        "obstruction_loss": link_raw["传播参数", "Lobs"],
        "sensitivity": link_raw["接收参数", "Psens"],
        "fade_margin": link_raw["接收参数", "M"],
        "gateway_height": link_raw["固定网关 G01", "hG"],
    }
    for key, source in [
        ("transport", "运输无人机"), ("relay_access", "中继接入端"),
        ("relay_backhaul", "中继回传端"), ("gateway", "固定网关 G01"),
    ]:
        links[key] = {"pt": link_raw[source, "Pt"], "gain": link_raw[source, "G"]}
    links["thresholds"] = {}
    for code, a, b in [
        ("TG", "transport", "gateway"),
        ("TR", "transport", "relay_access"),
        ("RG", "relay_backhaul", "gateway"),
    ]:
        pa, pb = links[a], links[b]
        links["thresholds"][code] = (
            min(pa["pt"], pb["pt"]) + pa["gain"] + pb["gain"]
            - links["system_loss"] - links["sensitivity"] - links["fade_margin"]
        )

    _validate(nodes, types, boxes, relay)
    return {
        "repo_root": str(root), "nodes": nodes, "types": types, "boxes": boxes,
        "relay": relay, "links": links, "dem_path": str(source_paths["dem"]),
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "source_sha256": {key: _sha256(path) for key, path in source_paths.items()},
        "units": {"distance": "m", "time": "s", "mass": "kg", "volume": "m^3",
                  "energy": "kWh", "power": "kW", "coordinate": "degrees WGS84"},
        "totals": {
            "boxes": len(boxes), "mass": sum(b["mass"] for b in boxes),
            "volume": sum(b["volume"] for b in boxes),
            "hard_deadline_boxes": sum(math.isfinite(b["deadline"]) for b in boxes),
            "first_boxes": sum(b["first"] for b in boxes),
            "medical_boxes": sum(b["medical"] for b in boxes),
            "boxes_per_node": dict(Counter(b["node"] for b in boxes)),
        },
    }


def _validate(nodes: dict, types: dict, boxes: list, relay: dict) -> None:
    if "O01" not in nodes or len(nodes) != 16 or set(types) != {"A", "B", "C"}:
        raise ValueError("Input does not match the supplied 1-depot/15-area/3-type scenario")
    if len(boxes) != 80 or len({box["id"] for box in boxes}) != len(boxes):
        raise ValueError("Expected 80 uniquely identified boxes")
    for box in boxes:
        if box["node"] not in nodes or box["node"] == "O01":
            raise ValueError(f"Invalid destination for {box['id']}")
        if min(box["mass"], box["volume"], box["weight"], box["due"]) <= 0:
            raise ValueError(f"Invalid box input: {box['id']}")
    for item in [*types.values(), relay]:
        if not 0 <= item["reserve"] < 1 or item["energy"] <= 0:
            raise ValueError(f"Invalid energy parameters: {item['id']}")
        if not item["drones"] or item["charge_full"] <= 0:
            raise ValueError(f"Invalid resource parameters: {item['id']}")
    if sum(len(item["drones"]) for item in types.values()) != 8:
        raise ValueError("Expected eight transport drones")


if __name__ == "__main__":
    import json

    data = load_data()
    print(json.dumps({"totals": data["totals"], "links": data["links"],
                      "units": data["units"]}, ensure_ascii=False, indent=2))
