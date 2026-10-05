#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stronger component ablation for the paper-aligned proposal.

Each variant removes or replaces one distinct design contribution:
  full                  complete paper-aligned method
  q_only                removes adaptive dual-branch choice; always Q
  r_only                removes adaptive dual-branch choice; always R
  distortion_only       removes robustness from branch selection
  robustness_only       removes distortion normalization from branch selection
  raw_score             removes local perturbation robustness evaluation
  no_safe_margin        removes explicit Q/R safety margins
  hard_vote             replaces signed soft accumulation with hard +/-1 votes
  raw_decoder           removes NLM candidates and confidence-guided decoder choice
  fixed_mild_decoder    uses mild NLM unconditionally instead of confidence selection
  repeat_1              removes repetition (one embedding observation per bit)

The raw output CSV is accompanied by two summaries:
  *_summary.csv          mean/std/min NC and mean BER per variant/attack
  *_delta_vs_full.csv    paired NC/BER differences versus the full method
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from attacks import get_attacks, profile_names  # noqa: E402
from proposal_method import (  # noqa: E402
    FLAG_Q, FLAG_R, MethodConfig, ber_binary, embed_array, extract_array,
    load_binary_watermark, nc_binary, psnr, ssim_color,
)


def variants(base: MethodConfig) -> List[Tuple[str, MethodConfig]]:
    return [
        ("full", base),
        ("q_only", replace(base, branch_policy="q_only")),
        ("r_only", replace(base, branch_policy="r_only")),
        ("distortion_only", replace(base, branch_policy="distortion_only")),
        ("robustness_only", replace(base, branch_policy="robustness_only")),
        ("raw_score", replace(base, branch_policy="raw_score")),
        ("no_safe_margin", replace(base, q_margin=0.0, r_margin=0.0)),
        ("hard_vote", replace(base, soft_vote=False)),
        ("raw_decoder", replace(base, use_nlm_candidates=False, decoder_selection="raw")),
        ("fixed_mild_decoder", replace(base, decoder_selection="mild")),
        ("repeat_1", replace(base, repetition_override=1)),
    ]


def find_data(repo: Path) -> Tuple[List[Path], List[Path]]:
    host_dir = repo / "data" / "hosts" / "classical"
    wm_dir = repo / "data" / "watermarks"
    hosts = sorted(host_dir.glob("*.bmp"))
    watermarks = sorted(wm_dir.glob("*.png"))
    if not hosts:
        raise FileNotFoundError(f"No hosts found in {host_dir}")
    if not watermarks:
        raise FileNotFoundError(f"No watermarks found in {wm_dir}")
    return hosts, watermarks


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def aggregate(rows: List[dict]) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        grouped.setdefault((r["variant"], r["attack"]), []).append(r)
    out = []
    for (variant, attack), rs in sorted(grouped.items()):
        nc = np.asarray([float(x["nc"]) for x in rs], dtype=float)
        ber = np.asarray([float(x["ber"]) for x in rs], dtype=float)
        out.append({
            "variant": variant, "attack": attack, "n": len(rs),
            "mean_nc": float(np.mean(nc)),
            "std_nc": float(np.std(nc, ddof=1)) if len(nc) > 1 else 0.0,
            "min_nc": float(np.min(nc)),
            "mean_ber": float(np.mean(ber)),
            "max_ber": float(np.max(ber)),
            "mean_embed_psnr": float(np.mean([float(x["embed_psnr"]) for x in rs])),
            "mean_embed_seconds": float(np.mean([float(x["embed_seconds"]) for x in rs])),
            "mean_extract_seconds": float(np.mean([float(x["extract_seconds"]) for x in rs])),
        })
    return out


def paired_deltas(rows: List[dict]) -> List[dict]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_key = {(r["host"], r["watermark"], r["attack"], r["variant"]): r for r in ok}
    variants_seen = sorted({r["variant"] for r in ok if r["variant"] != "full"})
    attacks_seen = sorted({r["attack"] for r in ok})
    out = []
    for variant in variants_seen:
        for attack in attacks_seen:
            diffs_nc, diffs_ber = [], []
            for r in ok:
                if r["variant"] != variant or r["attack"] != attack:
                    continue
                full = by_key.get((r["host"], r["watermark"], attack, "full"))
                if full is None:
                    continue
                diffs_nc.append(float(r["nc"]) - float(full["nc"]))
                diffs_ber.append(float(r["ber"]) - float(full["ber"]))
            if diffs_nc:
                dnc = np.asarray(diffs_nc, dtype=float)
                dber = np.asarray(diffs_ber, dtype=float)
                out.append({
                    "variant": variant, "attack": attack, "paired_n": len(dnc),
                    "mean_delta_nc_vs_full": float(np.mean(dnc)),
                    "median_delta_nc_vs_full": float(np.median(dnc)),
                    "worst_delta_nc_vs_full": float(np.min(dnc)),
                    "mean_delta_ber_vs_full": float(np.mean(dber)),
                    "wins_nc": int(np.sum(dnc > 1e-12)),
                    "ties_nc": int(np.sum(np.abs(dnc) <= 1e-12)),
                    "losses_nc": int(np.sum(dnc < -1e-12)),
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE, help="Root containing bundled data/")
    ap.add_argument("--out", type=Path, default=Path("validation/paper_aligned_ablation.csv"))
    ap.add_argument("--attack-profile", choices=profile_names(), default="representative",
                    help="representative is recommended for full ablation; use extended/stress for deeper runs")
    ap.add_argument("--quick", action="store_true", help="First host/watermark; clean + JPEG50; full/raw_score/repeat_1")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--hosts", default="", help="Comma-separated host stems")
    ap.add_argument("--watermarks", default="", help="Comma-separated watermark stems")
    ap.add_argument("--variants", default="", help="Comma-separated ablation names")
    args = ap.parse_args()

    base = MethodConfig()
    hosts, wms = find_data(args.repo)
    if args.hosts:
        keep = {x.strip().lower() for x in args.hosts.split(",") if x.strip()}
        hosts = [p for p in hosts if p.stem.lower() in keep]
    if args.watermarks:
        keep = {x.strip().lower() for x in args.watermarks.split(",") if x.strip()}
        wms = [p for p in wms if p.stem.lower() in keep]

    attack_map = get_attacks(args.attack_profile)
    variant_list = variants(base)
    if args.clean_only:
        attack_map = {"clean": get_attacks("paper")["clean"]}
    if args.quick:
        hosts = hosts[:1]; wms = wms[:1]
        p = get_attacks("paper")
        attack_map = {"clean": p["clean"], "jpeg_q50": p["jpeg_q50"]}
        variant_list = [(n, c) for n, c in variant_list if n in {"full", "raw_score", "repeat_1"}]
    if args.variants:
        keep = {x.strip() for x in args.variants.split(",") if x.strip()}
        variant_list = [(n, c) for n, c in variant_list if n in keep]

    rows: List[dict] = []
    for host_path in hosts:
        host = cv2.imread(str(host_path), cv2.IMREAD_COLOR)
        if host is None:
            continue
        for wm_path in wms:
            wm = load_binary_watermark(str(wm_path), base.wm_size)
            for name, cfg in variant_list:
                try:
                    emb = embed_array(host, wm, cfg, collect_diagnostics=False)
                except ValueError as e:
                    rows.append({"host": host_path.stem, "watermark": wm_path.stem,
                                 "variant": name, "status": "ineligible", "error": str(e)})
                    continue
                except RuntimeError as e:
                    rows.append({"host": host_path.stem, "watermark": wm_path.stem,
                                 "variant": name, "status": "failed", "error": str(e)})
                    continue

                q_count = int(np.sum(emb.side_info["flags"] == FLAG_Q))
                r_count = int(np.sum(emb.side_info["flags"] == FLAG_R))
                for attack_name, fn in attack_map.items():
                    try:
                        attacked = fn(emb.watermarked)
                        ext = extract_array(attacked, emb.side_info, cfg)
                        row = {
                            "host": host_path.stem, "watermark": wm_path.stem,
                            "variant": name, "attack_profile": args.attack_profile,
                            "attack": attack_name, "status": "ok",
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
                               "variant": name, "attack_profile": args.attack_profile,
                               "attack": attack_name, "status": "attack_failed", "error": str(e)}
                    rows.append(row)
                    print(json.dumps(row))

    write_csv(args.out, rows)
    summary = args.out.with_name(args.out.stem + "_summary.csv")
    delta = args.out.with_name(args.out.stem + "_delta_vs_full.csv")
    write_csv(summary, aggregate(rows))
    write_csv(delta, paired_deltas(rows))
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print(f"Wrote {summary}")
    print(f"Wrote {delta}")


if __name__ == "__main__":
    main()
