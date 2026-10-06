#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the ten independent Q-only/R-only proposal candidates.

The default run uses one fixed default strength per method.  Pass --sweep to
run the method-specific strength grids and automatically report feasible
settings satisfying mean PSNR > --min-psnr and clean NC >= --min-clean-nc.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from attacks import get_attacks, profile_names  # noqa: E402
from proposal_method import (  # noqa: E402
    DISPLAY_NAMES,
    METHODS,
    MethodConfig,
    ber_binary,
    embed_array,
    extract_array,
    load_binary_watermark,
    method_strength_grid,
    nc_binary,
    psnr,
    ssim_color,
    strength_value,
    with_strength,
)

SCREEN_ATTACKS = (
    "clean",
    "jpeg_q50",
    "gaussian_noise_v0p003",
    "blur_sigma1",
    "scale_0p5",
    "crop_10pct_resize",
)

HARD_FAMILIES = ("jpeg", "blur", "median", "average", "noise", "scale", "crop", "rotation_once", "translation_once")


def find_data(repo: Path) -> Tuple[List[Path], List[Path]]:
    hosts = sorted((repo / "data" / "hosts" / "classical").glob("*.bmp"))
    watermarks = sorted((repo / "data" / "watermarks").glob("*.png"))
    if not hosts:
        raise FileNotFoundError(f"No host images under {repo / 'data/hosts/classical'}")
    if not watermarks:
        raise FileNotFoundError(f"No watermarks under {repo / 'data/watermarks'}")
    return hosts, watermarks


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def choose_attacks(profile: str, screening: bool) -> Dict[str, object]:
    attacks = get_attacks(profile)
    if not screening:
        return attacks
    chosen = {}
    for name in SCREEN_ATTACKS:
        if name in attacks:
            chosen[name] = attacks[name]
    # representative profile contains all except possibly names inherited differently
    if "clean" not in chosen:
        chosen["clean"] = lambda x: x.copy()
    return chosen


def configs_for_method(method: str, sweep: bool, repeat: int, key: str, selection: str, sync: str) -> List[MethodConfig]:
    base = MethodConfig(
        method=method, repetition_override=repeat, private_key=key,
        block_selection=selection, sync_mode=sync,
    )
    if not sweep:
        return [base]
    return [with_strength(base, s) for s in method_strength_grid(method)]


def run_one(host: np.ndarray, wm: np.ndarray, cfg: MethodConfig, attacks: Dict[str, object]) -> List[dict]:
    emb = embed_array(host, wm, cfg, collect_diagnostics=False)
    clean_psnr = psnr(host, emb.watermarked)
    clean_ssim = ssim_color(host, emb.watermarked)
    rows: List[dict] = []
    for attack_name, fn in attacks.items():
        attacked = fn(emb.watermarked)
        ext = extract_array(attacked, emb.side_info, cfg)
        rows.append({
            "method": cfg.method,
            "display_name": DISPLAY_NAMES[cfg.method],
            "family": "Q" if cfg.method.startswith("q_") else "R",
            "strength": strength_value(cfg),
            "attack": attack_name,
            "status": "ok",
            "embed_psnr": clean_psnr,
            "embed_ssim": clean_ssim,
            "nc": nc_binary(wm, ext.watermark),
            "ber": ber_binary(wm, ext.watermark),
            "confidence": ext.confidence,
            "valid_observations": ext.valid_observations,
            "repeat": int(emb.side_info["repeat"][0]),
            "embed_seconds": emb.elapsed_seconds,
            "extract_seconds": ext.elapsed_seconds,
        })
    return rows


def _mean(xs: Iterable[float]) -> float:
    a = list(xs)
    return float(np.mean(a)) if a else math.nan


def aggregate(rows: List[dict], min_psnr: float, min_clean_nc: float) -> List[dict]:
    grouped: Dict[Tuple[str, float], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            grouped[(str(r["method"]), float(r["strength"]))].append(r)
    out: List[dict] = []
    for (method, strength), rs in grouped.items():
        clean = [x for x in rs if x["attack"] == "clean"]
        attacked = [x for x in rs if x["attack"] != "clean"]
        clean_nc = _mean(float(x["nc"]) for x in clean)
        mean_psnr = _mean(float(x["embed_psnr"]) for x in clean)
        attacked_nc = _mean(float(x["nc"]) for x in attacked)
        worst_nc = min((float(x["nc"]) for x in attacked), default=clean_nc)
        hard = [x for x in attacked if any(tok in str(x["attack"]) for tok in HARD_FAMILIES)]
        hard_nc = _mean(float(x["nc"]) for x in hard)
        feasible = bool(mean_psnr > min_psnr and clean_nc >= min_clean_nc)
        out.append({
            "method": method,
            "display_name": DISPLAY_NAMES.get(method, method),
            "family": "Q" if method.startswith("q_") else "R",
            "strength": strength,
            "mean_psnr": mean_psnr,
            "mean_ssim": _mean(float(x["embed_ssim"]) for x in clean),
            "clean_nc": clean_nc,
            "clean_ber": _mean(float(x["ber"]) for x in clean),
            "attacked_mean_nc": attacked_nc,
            "hard_attack_mean_nc": hard_nc,
            "worst_attack_nc": worst_nc,
            "attacked_mean_ber": _mean(float(x["ber"]) for x in attacked),
            "mean_embed_seconds": _mean(float(x["embed_seconds"]) for x in clean),
            "mean_extract_seconds": _mean(float(x["extract_seconds"]) for x in rs),
            "feasible": int(feasible),
            "n_rows": len(rs),
        })
    out.sort(key=lambda r: (
        -int(r["feasible"]),
        -float(r["hard_attack_mean_nc"]) if np.isfinite(r["hard_attack_mean_nc"]) else 1e9,
        -float(r["attacked_mean_nc"]) if np.isfinite(r["attacked_mean_nc"]) else 1e9,
        -float(r["worst_attack_nc"]) if np.isfinite(r["worst_attack_nc"]) else 1e9,
        -float(r["mean_psnr"]),
    ))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def best_per_method(summary: List[dict]) -> List[dict]:
    best: Dict[str, dict] = {}
    for r in summary:
        m = str(r["method"])
        if m not in best:
            best[m] = r
        elif int(r["feasible"]) > int(best[m]["feasible"]):
            best[m] = r
        elif int(r["feasible"]) == int(best[m]["feasible"]):
            key = (float(r["hard_attack_mean_nc"]), float(r["attacked_mean_nc"]), float(r["worst_attack_nc"]), float(r["mean_psnr"]))
            old = (float(best[m]["hard_attack_mean_nc"]), float(best[m]["attacked_mean_nc"]), float(best[m]["worst_attack_nc"]), float(best[m]["mean_psnr"]))
            if key > old:
                best[m] = r
    out = list(best.values())
    out.sort(key=lambda r: (-int(r["feasible"]), -float(r["hard_attack_mean_nc"]), -float(r["attacked_mean_nc"]), -float(r["worst_attack_nc"])))
    for i, r in enumerate(out, 1):
        r["method_rank"] = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE)
    ap.add_argument("--profile", choices=profile_names(), default="representative")
    ap.add_argument("--out", type=Path, default=Path("validation/candidate_comparison.csv"))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--hosts", default="")
    ap.add_argument("--watermarks", default="")
    ap.add_argument("--sweep", action="store_true", help="Run method-specific strength grids")
    ap.add_argument("--screening", action="store_true", help="Use compact screening attacks")
    ap.add_argument("--quick", action="store_true", help="First host + first watermark + screening attacks")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--key", default="KB123")
    ap.add_argument("--selection", choices=("stability", "chaotic"), default="stability")
    ap.add_argument("--sync", choices=("none", "auto"), default="none",
                    help="Optional carrier-conformity geometric self-synchronisation")
    ap.add_argument("--min-psnr", type=float, default=50.0)
    ap.add_argument("--min-clean-nc", type=float, default=0.999999)
    args = ap.parse_args()

    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    hosts, wms = find_data(args.repo)
    if args.hosts:
        keep = {x.strip().lower() for x in args.hosts.split(",") if x.strip()}
        hosts = [p for p in hosts if p.stem.lower() in keep]
    if args.watermarks:
        keep = {x.strip().lower() for x in args.watermarks.split(",") if x.strip()}
        wms = [p for p in wms if p.stem.lower() in keep]
    if args.quick:
        hosts, wms = hosts[:1], wms[:1]

    attacks = choose_attacks(args.profile, args.screening or args.quick)
    rows: List[dict] = []
    for host_path in hosts:
        host = cv2.imread(str(host_path), cv2.IMREAD_COLOR)
        if host is None:
            continue
        for wm_path in wms:
            wm = load_binary_watermark(str(wm_path), 64)
            for method in methods:
                for cfg in configs_for_method(method, args.sweep, args.repeat, args.key, args.selection, args.sync):
                    try:
                        result_rows = run_one(host, wm, cfg, attacks)
                        for r in result_rows:
                            r["host"] = host_path.stem
                            r["watermark"] = wm_path.stem
                            r["attack_profile"] = args.profile
                            rows.append(r)
                            print(json.dumps(r))
                    except Exception as e:
                        row = {
                            "host": host_path.stem, "watermark": wm_path.stem,
                            "method": method, "display_name": DISPLAY_NAMES[method],
                            "strength": strength_value(cfg), "status": "failed", "error": repr(e),
                        }
                        rows.append(row); print(json.dumps(row))

    write_csv(args.out, rows)
    summary = aggregate(rows, args.min_psnr, args.min_clean_nc)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_csv(summary_path, summary)
    best = best_per_method(summary)
    best_path = args.out.with_name(args.out.stem + "_best_per_method.csv")
    write_csv(best_path, best)
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
