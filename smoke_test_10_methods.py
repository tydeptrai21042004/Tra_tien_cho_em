#!/usr/bin/env python3
"""Fast structural smoke test for all ten new carriers."""
from __future__ import annotations
from pathlib import Path
import cv2
from proposal_method import METHODS, DISPLAY_NAMES, MethodConfig, embed_array, extract_array, load_binary_watermark, nc_binary, psnr

HERE = Path(__file__).resolve().parent
host = cv2.imread(str(HERE / "data/hosts/classical/airplane.bmp"), cv2.IMREAD_COLOR)
if host is None:
    raise FileNotFoundError("Bundled airplane.bmp not found")
host = host[:128, :128].copy()
wm = load_binary_watermark(HERE / "data/watermarks/watermark_1.png", 8)
for method in METHODS:
    cfg = MethodConfig(method=method, wm_size=8, repetition_override=1)
    emb = embed_array(host, wm, cfg, collect_diagnostics=False)
    assert "flags" not in emb.side_info, "old per-block Q/R flags must not exist"
    ext = extract_array(emb.watermarked, emb.side_info, cfg)
    print(f"{DISPLAY_NAMES[method]:9s} PSNR={psnr(host, emb.watermarked):7.3f} cleanNC={nc_binary(wm, ext.watermark):.4f}")
print("PASS: all ten methods executed and old branch flags are absent")
