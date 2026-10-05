#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Branch-position justification using the images bundled in the repository.

The goal is not to claim that Q or R is tied to a fixed spatial region. Instead,
this script tests the manuscript's intended explanation: branch identity is a
local data-driven consequence of the distortion/robustness score. It exports:
  1) one row per selected 4x4 block (position, branch, candidate scores, texture),
  2) per-image branch counts,
  3) spatial 4x4-grid branch proportions,
  4) feature summaries for Q-selected vs R-selected blocks,
  5) a concise text report usable as evidence for the branch-choice discussion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from proposal_method import FLAG_Q, FLAG_R, MethodConfig, embed_array, load_binary_watermark  # noqa: E402


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def summarize_features(rows: List[dict]) -> List[dict]:
    out = []
    for image in sorted({r["image"] for r in rows}):
        for branch in ("Q", "R"):
            grp = [r for r in rows if r["image"] == image and r["branch_name"] == branch]
            if not grp:
                continue
            out.append({
                "image": image,
                "branch": branch,
                "count": len(grp),
                "mean_local_variance": float(np.mean([r["local_variance"] for r in grp])),
                "mean_local_gradient": float(np.mean([r["local_gradient"] for r in grp])),
                "mean_local_dynamic_range": float(np.mean([r["local_dynamic_range"] for r in grp])),
                "mean_chosen_distortion": float(np.mean([r["chosen_distortion"] for r in grp])),
                "mean_chosen_robustness": float(np.mean([r["chosen_robustness"] for r in grp])),
            })
    return out


def spatial_grid(rows: List[dict], grid: int = 4) -> List[dict]:
    agg: Dict[tuple, List[str]] = defaultdict(list)
    for r in rows:
        gr = min(grid - 1, int(float(r["row_norm"]) * grid))
        gc = min(grid - 1, int(float(r["col_norm"]) * grid))
        agg[(r["image"], gr, gc)].append(r["branch_name"])
    out = []
    for (image, gr, gc), vals in sorted(agg.items()):
        q = sum(v == "Q" for v in vals)
        rr = sum(v == "R" for v in vals)
        out.append({
            "image": image, "grid_row": gr, "grid_col": gc,
            "Q_count": q, "R_count": rr, "total": len(vals),
            "Q_percent": 100.0 * q / len(vals), "R_percent": 100.0 * rr / len(vals),
        })
    return out


def branch_counts(rows: List[dict]) -> List[dict]:
    out = []
    for image in sorted({r["image"] for r in rows}):
        grp = [r for r in rows if r["image"] == image]
        q = sum(r["branch_name"] == "Q" for r in grp)
        rr = sum(r["branch_name"] == "R" for r in grp)
        out.append({
            "image": image,
            "Q_count": q,
            "R_count": rr,
            "Q_percent": 100.0 * q / len(grp),
            "R_percent": 100.0 * rr / len(grp),
            "total": len(grp),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--watermark", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("validation/branch_position"))
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--hosts", default="", help="Comma-separated host stems; empty means all")
    args = ap.parse_args()

    cfg = MethodConfig()
    host_dir = args.repo / "data" / "hosts" / "classical"
    wm_path = args.watermark or (args.repo / "data" / "watermarks" / "watermark_1.png")
    wm = load_binary_watermark(str(wm_path), cfg.wm_size)

    all_rows: List[dict] = []
    excluded = []
    host_paths = sorted(host_dir.glob("*.bmp"))
    if args.hosts:
        keep = {x.strip().lower() for x in args.hosts.split(",") if x.strip()}
        host_paths = [p for p in host_paths if p.stem.lower() in keep]
    for host_path in host_paths:
        host = cv2.imread(str(host_path), cv2.IMREAD_COLOR)
        if host is None:
            continue
        try:
            emb = embed_array(host, wm, cfg, collect_diagnostics=True)
        except ValueError as e:
            excluded.append({"image": host_path.stem, "reason": str(e)})
            print(f"EXCLUDED {host_path.stem}: {e}")
            continue
        for d in emb.diagnostics:
            row = dict(d)
            row["image"] = host_path.stem
            row["branch_name"] = "Q" if int(row["branch"]) == FLAG_Q else "R"
            all_rows.append(row)
        print(host_path.stem, "Q", int(np.sum(emb.side_info["flags"] == FLAG_Q)), "R", int(np.sum(emb.side_info["flags"] == FLAG_R)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "branch_positions.csv", all_rows)
    count_rows = branch_counts(all_rows)
    grid_rows = spatial_grid(all_rows, args.grid)
    feature_rows = summarize_features(all_rows)
    write_csv(args.out_dir / "branch_counts.csv", count_rows)
    write_csv(args.out_dir / "branch_spatial_grid.csv", grid_rows)
    write_csv(args.out_dir / "branch_feature_summary.csv", feature_rows)
    if excluded:
        write_csv(args.out_dir / "excluded_hosts.csv", excluded)

    # Quantify whether one branch is confined to a location.  A low spread would
    # indicate spatial hard-coding; broad occupancy supports local score-driven use.
    image_reports = []
    for c in count_rows:
        image = c["image"]
        cells = [r for r in grid_rows if r["image"] == image]
        q_cells = sum(r["Q_count"] > 0 for r in cells)
        r_cells = sum(r["R_count"] > 0 for r in cells)
        q_share = [r["Q_percent"] / 100.0 for r in cells]
        image_reports.append({
            **c,
            "grid_cells": len(cells),
            "Q_occupied_cells": q_cells,
            "R_occupied_cells": r_cells,
            "Q_share_min": min(q_share) if q_share else float("nan"),
            "Q_share_max": max(q_share) if q_share else float("nan"),
        })
    write_csv(args.out_dir / "branch_position_summary.csv", image_reports)

    report = [
        "Branch-position justification (paper-aligned implementation)",
        "===========================================================",
        "",
        "Interpretation: Q/R is not assigned by image coordinates. Both candidates are",
        "built at every selected determinant-valid block, and the stored-domain local",
        "robustness/distortion score chooses the branch. The spatial-grid statistics below",
        "are therefore a sanity check for broad spatial use, not a new selection rule.",
        "",
    ]
    for r in image_reports:
        report.append(
            f"{r['image']}: Q={r['Q_count']} ({r['Q_percent']:.2f}%), "
            f"R={r['R_count']} ({r['R_percent']:.2f}%); "
            f"Q appears in {r['Q_occupied_cells']}/{r['grid_cells']} grid cells, "
            f"R appears in {r['R_occupied_cells']}/{r['grid_cells']} grid cells."
        )
    if excluded:
        report.append("")
        for e in excluded:
            report.append(f"Excluded {e['image']}: {e['reason']}")
    (args.out_dir / "branch_position_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "images": len(count_rows), "excluded": excluded}, indent=2))


if __name__ == "__main__":
    main()
