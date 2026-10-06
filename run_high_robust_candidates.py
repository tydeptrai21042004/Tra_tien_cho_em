#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from attacks import get_attacks, profile_names
from proposal_method import (
    DISPLAY_NAMES as QR_DISPLAY_NAMES,
    MethodConfig,
    ber_binary,
    embed_array as qr_embed_array,
    extract_array as qr_extract_array,
    load_binary_watermark,
    nc_binary,
    psnr,
    ssim_color,
)
from robust_transform_method import (
    DISPLAY_NAMES,
    METHODS,
    TransformMethodConfig,
    embed_array,
    extract_array,
    method_strength_grid,
    strength_value,
    with_strength,
)

CONTROL = "q_smm_sync"
ALL_METHODS = METHODS + (CONTROL,)
ALL_NAMES = {**DISPLAY_NAMES, CONTROL: "Q-SMM-Sync control"}


def find_data(repo: Path) -> Tuple[List[Path], List[Path]]:
    hosts = sorted((repo / "data" / "hosts" / "classical").glob("*.bmp"))
    wms = sorted((repo / "data" / "watermarks").glob("*.png"))
    if not hosts or not wms:
        raise FileNotFoundError("Expected data/hosts/classical/*.bmp and data/watermarks/*.png")
    return hosts, wms


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def summarize(rows: List[dict], min_psnr: float, min_clean_nc: float) -> List[dict]:
    out = []
    methods = sorted({r.get("method") for r in rows if r.get("status") == "ok"})
    for m in methods:
        rs = [r for r in rows if r.get("status") == "ok" and r.get("method") == m]
        strengths = sorted({float(r["strength"]) for r in rs})
        for s in strengths:
            ss = [r for r in rs if float(r["strength"]) == s]
            clean = [r for r in ss if r["attack"] == "clean"]
            atk = [r for r in ss if r["attack"] != "clean"]
            if not clean or not atk:
                continue
            ncs = np.asarray([float(r["nc"]) for r in atk])
            bers = np.asarray([float(r["ber"]) for r in atk])
            p = float(np.mean([float(r["embed_psnr"]) for r in clean]))
            cnc = float(np.mean([float(r["nc"]) for r in clean]))
            out.append({
                "method": m,
                "display_name": ALL_NAMES[m],
                "strength": s,
                "mean_psnr": p,
                "clean_nc": cnc,
                "attacked_mean_nc": float(np.mean(ncs)),
                "attacked_min_nc": float(np.min(ncs)),
                "attacked_mean_ber": float(np.mean(bers)),
                "feasible": int(p >= min_psnr and cnc >= min_clean_nc),
            })
    out.sort(key=lambda r: (-int(r["feasible"]), -float(r["attacked_mean_nc"]), -float(r["attacked_min_nc"]), -float(r["mean_psnr"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--profile", choices=profile_names(), default="representative")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hosts", default="")
    ap.add_argument("--watermarks", default="")
    ap.add_argument("--key", default="KB123")
    ap.add_argument("--sync", choices=("none", "pilot"), default="pilot")
    ap.add_argument("--min-psnr", type=float, default=50.0)
    ap.add_argument("--target-mean-nc", type=float, default=0.90)
    ap.add_argument("--min-clean-nc", type=float, default=0.999999)
    ap.add_argument("--out", type=Path, default=Path("validation/high_robust_candidates.csv"))
    args = ap.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    hosts, wms = find_data(args.repo)
    if args.hosts:
        keep = {x.strip().lower() for x in args.hosts.split(",") if x.strip()}
        hosts = [p for p in hosts if p.stem.lower() in keep]
    if args.watermarks:
        keep = {x.strip().lower() for x in args.watermarks.split(",") if x.strip()}
        wms = [p for p in wms if p.stem.lower() in keep]
    if args.quick:
        hosts, wms = hosts[:1], wms[:1]
    attacks: Dict[str, object] = get_attacks(args.profile)
    if args.quick:
        keep = {"clean", "gaussian_noise_v0p003", "jpeg_q50", "scale_0p5", "rotation_once_plus3", "translation_once_4_4", "crop_10pct_resize"}
        attacks = {k: v for k, v in attacks.items() if k in keep}

    rows: List[dict] = []
    for hp in hosts:
        host = cv2.imread(str(hp), cv2.IMREAD_COLOR)
        if host is None:
            continue
        for wp in wms:
            wm = load_binary_watermark(str(wp), 64)
            for method in methods:
                if method == CONTROL:
                    strengths = [0.012]
                else:
                    strengths = method_strength_grid(method) if args.sweep else [TransformMethodConfig(method=method).step]
                for strength in strengths:
                    try:
                        if method == CONTROL:
                            cfgq = MethodConfig(method="q_smm", q_margin=0.012, private_key=args.key, block_selection="stability", sync_mode="auto")
                            emb = qr_embed_array(host, wm, cfgq, collect_diagnostics=False)
                            extractor = lambda x: qr_extract_array(x, emb.side_info, cfgq)
                        else:
                            cfg = TransformMethodConfig(method=method, private_key=args.key, sync_mode=args.sync)
                            cfg = with_strength(cfg, strength)
                            emb = embed_array(host, wm, cfg, collect_diagnostics=False)
                            extractor = lambda x, cfg=cfg: extract_array(x, emb.side_info, cfg)
                        p = psnr(host, emb.watermarked); ss = ssim_color(host, emb.watermarked)
                        for aname, attack in attacks.items():
                            attacked = attack(emb.watermarked)
                            ext = extractor(attacked)
                            row = {
                                "host": hp.stem,
                                "watermark": wp.stem,
                                "method": method,
                                "display_name": ALL_NAMES[method],
                                "strength": float(strength),
                                "attack": aname,
                                "status": "ok",
                                "embed_psnr": p,
                                "embed_ssim": ss,
                                "nc": nc_binary(wm, ext.watermark),
                                "ber": ber_binary(wm, ext.watermark),
                                "confidence": ext.confidence,
                                "candidate_name": ext.candidate_name,
                                "embed_seconds": emb.elapsed_seconds,
                                "extract_seconds": ext.elapsed_seconds,
                            }
                            rows.append(row); print(json.dumps(row))
                    except Exception as e:
                        row = {"host": hp.stem, "watermark": wp.stem, "method": method, "strength": float(strength), "status": "failed", "error": repr(e)}
                        rows.append(row); print(json.dumps(row))

    write_csv(args.out, rows)
    summary = summarize(rows, args.min_psnr, args.min_clean_nc)
    for r in summary:
        r["target_mean_nc"] = float(args.target_mean_nc)
        r["target_met"] = int(int(r["feasible"]) == 1 and float(r["attacked_mean_nc"]) >= float(args.target_mean_nc))
    summary.sort(key=lambda r: (-int(r["target_met"]), -int(r["feasible"]), -float(r["attacked_mean_nc"]), -float(r["attacked_min_nc"]), -float(r["mean_psnr"])))
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"), summary)
    print("\nFINAL SUMMARY")
    for r in summary:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
