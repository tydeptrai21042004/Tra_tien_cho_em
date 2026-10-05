#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full paper-aligned proposal over deterministic attack profiles."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from attacks import get_attacks, profile_names  # noqa: E402
from proposal_method import (  # noqa: E402
    FLAG_Q, FLAG_R, MethodConfig, ber_binary, embed_array, extract_array,
    load_binary_watermark, nc_binary, psnr, ssim_color,
)


def find_data(repo: Path) -> Tuple[List[Path], List[Path]]:
    hosts = sorted((repo / "data" / "hosts" / "classical").glob("*.bmp"))
    watermarks = sorted((repo / "data" / "watermarks").glob("*.png"))
    if not hosts:
        raise FileNotFoundError(f"No hosts found under {repo / 'data/hosts/classical'}")
    if not watermarks:
        raise FileNotFoundError(f"No watermarks found under {repo / 'data/watermarks'}")
    return hosts, watermarks


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_summary(path: Path, rows: List[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    grouped = {}
    for r in ok:
        grouped.setdefault(r["attack"], []).append(r)
    summary = []
    for attack, rs in sorted(grouped.items()):
        ncs = np.asarray([float(x["nc"]) for x in rs], dtype=float)
        bers = np.asarray([float(x["ber"]) for x in rs], dtype=float)
        summary.append({
            "attack": attack,
            "n": len(rs),
            "mean_nc": float(np.mean(ncs)),
            "std_nc": float(np.std(ncs, ddof=1)) if len(ncs) > 1 else 0.0,
            "min_nc": float(np.min(ncs)),
            "mean_ber": float(np.mean(bers)),
            "max_ber": float(np.max(bers)),
        })
    write_csv(path, summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE, help="Root containing bundled data/")
    ap.add_argument("--profile", choices=profile_names(), default="extended")
    ap.add_argument("--out", type=Path, default=Path("validation/attack_suite.csv"))
    ap.add_argument("--hosts", default="", help="Comma-separated host stems")
    ap.add_argument("--watermarks", default="", help="Comma-separated watermark stems")
    args = ap.parse_args()

    cfg = MethodConfig()
    hosts, wms = find_data(args.repo)
    if args.hosts:
        keep = {x.strip().lower() for x in args.hosts.split(",") if x.strip()}
        hosts = [p for p in hosts if p.stem.lower() in keep]
    if args.watermarks:
        keep = {x.strip().lower() for x in args.watermarks.split(",") if x.strip()}
        wms = [p for p in wms if p.stem.lower() in keep]

    attack_map = get_attacks(args.profile)
    rows: List[dict] = []
    for host_path in hosts:
        host = cv2.imread(str(host_path), cv2.IMREAD_COLOR)
        if host is None:
            continue
        for wm_path in wms:
            wm = load_binary_watermark(str(wm_path), cfg.wm_size)
            try:
                emb = embed_array(host, wm, cfg, collect_diagnostics=False)
            except (ValueError, RuntimeError) as e:
                rows.append({"host": host_path.stem, "watermark": wm_path.stem, "status": "ineligible", "error": str(e)})
                continue
            q_count = int(np.sum(emb.side_info["flags"] == FLAG_Q))
            r_count = int(np.sum(emb.side_info["flags"] == FLAG_R))
            for attack_name, fn in attack_map.items():
                try:
                    attacked = fn(emb.watermarked)
                    ext = extract_array(attacked, emb.side_info, cfg)
                    row = {
                        "host": host_path.stem, "watermark": wm_path.stem,
                        "attack_profile": args.profile, "attack": attack_name, "status": "ok",
                        "embed_psnr": psnr(host, emb.watermarked),
                        "embed_ssim": ssim_color(host, emb.watermarked),
                        "nc": nc_binary(wm, ext.watermark), "ber": ber_binary(wm, ext.watermark),
                        "confidence": ext.confidence, "decoder_candidate": ext.candidate_name,
                        "valid_observations": ext.valid_observations,
                        "Q_count": q_count, "R_count": r_count,
                        "repeat": int(emb.side_info["repeat"][0]),
                        "embed_seconds": emb.elapsed_seconds, "extract_seconds": ext.elapsed_seconds,
                    }
                except Exception as e:
                    row = {"host": host_path.stem, "watermark": wm_path.stem,
                           "attack_profile": args.profile, "attack": attack_name,
                           "status": "attack_failed", "error": str(e)}
                rows.append(row)
                print(json.dumps(row))

    write_csv(args.out, rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_summary(summary_path, rows)
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
