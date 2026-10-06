#!/usr/bin/env python3
from pathlib import Path
import cv2

from proposal_method import load_binary_watermark, nc_binary, psnr
from robust_transform_method import METHODS, TransformMethodConfig, embed_array, extract_array

ROOT = Path(__file__).resolve().parent
host = cv2.imread(str(sorted((ROOT / "data/hosts/classical").glob("*.bmp"))[0]), cv2.IMREAD_COLOR)
wm = load_binary_watermark(str(sorted((ROOT / "data/watermarks").glob("*.png"))[0]), 64)
for method in METHODS:
    cfg = TransformMethodConfig(method=method, sync_mode="none")
    emb = embed_array(host, wm, cfg)
    ext = extract_array(emb.watermarked, emb.side_info, cfg)
    print(f"{method:28s} PSNR={psnr(host, emb.watermarked):8.3f} cleanNC={nc_binary(wm, ext.watermark):.6f}")
