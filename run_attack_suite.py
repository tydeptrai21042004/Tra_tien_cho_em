#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one selected Q-only/R-only candidate over a deterministic attack profile."""
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
    DISPLAY_NAMES, METHODS, MethodConfig, ber_binary, embed_array, extract_array,
    load_binary_watermark, nc_binary, psnr, ssim_color, strength_value, with_strength,
)


def find_data(repo: Path) -> Tuple[List[Path], List[Path]]:
    hosts = sorted((repo / "data" / "hosts" / "classical").glob("*.bmp"))
    watermarks = sorted((repo / "data" / "watermarks").glob("*.png"))
    if not hosts: raise FileNotFoundError(f"No hosts under {repo / 'data/hosts/classical'}")
    if not watermarks: raise FileNotFoundError(f"No watermarks under {repo / 'data/watermarks'}")
    return hosts, watermarks


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_summary(path: Path, rows: List[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    grouped = {}
    for r in ok: grouped.setdefault(r["attack"], []).append(r)
    summary = []
    for attack, rs in sorted(grouped.items()):
        ncs = np.asarray([float(x["nc"]) for x in rs]); bers = np.asarray([float(x["ber"]) for x in rs])
        summary.append({
            "attack": attack, "n": len(rs),
            "mean_nc": float(np.mean(ncs)), "std_nc": float(np.std(ncs, ddof=1)) if len(ncs)>1 else 0.0,
            "min_nc": float(np.min(ncs)), "mean_ber": float(np.mean(bers)), "max_ber": float(np.max(bers)),
        })
    write_csv(path, summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE)
    ap.add_argument("--profile", choices=profile_names(), default="extended")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--key", default="KB123")
    ap.add_argument("--selection", choices=("stability", "chaotic"), default="stability")
    ap.add_argument("--sync", choices=("none", "auto"), default="none")
    ap.add_argument("--out", type=Path, default=Path("validation/attack_suite.csv"))
    ap.add_argument("--hosts", default="")
    ap.add_argument("--watermarks", default="")
    args = ap.parse_args()

    cfg = MethodConfig(method=args.method, repetition_override=args.repeat, private_key=args.key, block_selection=args.selection, sync_mode=args.sync)
    if args.strength is not None: cfg = with_strength(cfg, args.strength)
    hosts, wms = find_data(args.repo)
    if args.hosts:
        keep = {x.strip().lower() for x in args.hosts.split(",") if x.strip()}; hosts = [p for p in hosts if p.stem.lower() in keep]
    if args.watermarks:
        keep = {x.strip().lower() for x in args.watermarks.split(",") if x.strip()}; wms = [p for p in wms if p.stem.lower() in keep]

    attacks = get_attacks(args.profile)
    rows: List[dict] = []
    for hp in hosts:
        host = cv2.imread(str(hp), cv2.IMREAD_COLOR)
        if host is None: continue
        for wp in wms:
            wm = load_binary_watermark(str(wp), cfg.wm_size)
            try:
                emb = embed_array(host, wm, cfg, collect_diagnostics=False)
            except Exception as e:
                rows.append({"host":hp.stem,"watermark":wp.stem,"status":"embed_failed","error":repr(e)}); continue
            p = psnr(host, emb.watermarked); s = ssim_color(host, emb.watermarked)
            for attack_name, fn in attacks.items():
                try:
                    attacked = fn(emb.watermarked); ext = extract_array(attacked, emb.side_info, cfg)
                    row = {
                        "host":hp.stem,"watermark":wp.stem,"method":cfg.method,"display_name":DISPLAY_NAMES[cfg.method],
                        "strength":strength_value(cfg),"attack_profile":args.profile,"attack":attack_name,"status":"ok",
                        "embed_psnr":p,"embed_ssim":s,"nc":nc_binary(wm,ext.watermark),"ber":ber_binary(wm,ext.watermark),
                        "confidence":ext.confidence,"valid_observations":ext.valid_observations,"repeat":args.repeat,
                        "embed_seconds":emb.elapsed_seconds,"extract_seconds":ext.elapsed_seconds,
                    }
                except Exception as e:
                    row={"host":hp.stem,"watermark":wp.stem,"method":cfg.method,"attack":attack_name,"status":"attack_failed","error":repr(e)}
                rows.append(row); print(json.dumps(row))
    write_csv(args.out, rows)
    summary = args.out.with_name(args.out.stem + "_summary.csv"); write_summary(summary, rows)
    print(f"Wrote {args.out} ({len(rows)} rows)\nWrote {summary}")

if __name__ == "__main__": main()
