#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic attack suite for the paper-aligned QR watermarking experiments.

Profiles
--------
paper
    Reproduces the attack types and principal settings currently stated in the
    manuscript.
representative
    Compact but diverse set used by default for ablation (noise, compression,
    filtering, geometry, photometric change, occlusion).
extended
    Adds median/average filtering, speckle noise, translation, crop-resize,
    brightness/contrast changes, and several additional compression/geometry
    settings.
stress
    Parameter sweep with multiple severities for the main attack families.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Dict

import cv2
import numpy as np

Attack = Callable[[np.ndarray], np.ndarray]


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, variance: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0
    y = x + rng.normal(0.0, math.sqrt(float(variance)), x.shape).astype(np.float32)
    return _clip_u8(y * 255.0)


def speckle_noise(img: np.ndarray, variance: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0
    n = rng.normal(0.0, math.sqrt(float(variance)), x.shape).astype(np.float32)
    return _clip_u8((x + x * n) * 255.0)


def salt_pepper(img: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape[:2])
    half = float(density) / 2.0
    out[mask < half] = 0
    out[(mask >= half) & (mask < float(density))] = 255
    return out


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("JPEG decode failed")
    return out


def jpeg2000_ratio(img: np.ndarray, compression_ratio: float) -> np.ndarray:
    if not hasattr(cv2, "IMWRITE_JPEG2000_COMPRESSION_X1000"):
        raise RuntimeError("This OpenCV build does not expose JPEG2000 compression")
    rate = max(1, min(1000, int(round(1000.0 / max(1.0, float(compression_ratio))))))
    ok, buf = cv2.imencode(".jp2", img, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), rate])
    if not ok:
        raise RuntimeError("JPEG2000 encode failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("JPEG2000 decode failed")
    return out


def scale_roundtrip(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    nw = max(1, int(round(w * float(factor))))
    nh = max(1, int(round(h * float(factor))))
    interp_down = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    tmp = cv2.resize(img, (nw, nh), interpolation=interp_down)
    return cv2.resize(tmp, (w, h), interpolation=cv2.INTER_CUBIC)


def rotate_once(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def rotate_roundtrip(img: np.ndarray, angle: float) -> np.ndarray:
    return rotate_once(rotate_once(img, float(angle)), -float(angle))


def translate_once(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def translate_roundtrip(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return translate_once(translate_once(img, dx, dy), -dx, -dy)


def crop_resize(img: np.ndarray, fraction: float) -> np.ndarray:
    """Center-crop a fraction of each border-area dimension, then resize back."""
    h, w = img.shape[:2]
    f = float(fraction)
    if not 0.0 <= f < 0.9:
        raise ValueError("crop fraction must be in [0, 0.9)")
    dh = int(round(h * f / 2.0))
    dw = int(round(w * f / 2.0))
    cropped = img[dh:h-dh if dh else h, dw:w-dw if dw else w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return _clip_u8((x ** float(g)) * 255.0)


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return _clip_u8(img.astype(np.float32) + float(delta))


def contrast(img: np.ndarray, factor: float) -> np.ndarray:
    return _clip_u8((img.astype(np.float32) - 127.5) * float(factor) + 127.5)


def occlusion(img: np.ndarray, proportion: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = out.shape[:2]
    target = int(round(float(proportion) * h * w))
    mask = np.zeros((h, w), dtype=bool)
    chosen = rng.choice(h * w, size=target, replace=False)
    mask.flat[chosen] = True
    out[mask] = 0
    return out


def sharpen(img: np.ndarray, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), float(sigma))
    return cv2.addWeighted(img, 1.0 + float(amount), blur, -float(amount), 0)


def _paper() -> "OrderedDict[str, Attack]":
    return OrderedDict([
        ("clean", lambda x: x.copy()),
        ("blur_sigma1", lambda x: cv2.GaussianBlur(x, (0, 0), 1.0)),
        ("sharpen_sigma1_a1p5", lambda x: sharpen(x, 1.0, 1.5)),
        ("gaussian_noise_v0p003", lambda x: gaussian_noise(x, 0.003, 0)),
        ("salt_pepper_d0p1", lambda x: salt_pepper(x, 0.10, 0)),
        ("jpeg_q50", lambda x: jpeg(x, 50)),
        ("jpeg2000_cr13", lambda x: jpeg2000_ratio(x, 13.0)),
        ("lowpass_9x9", lambda x: cv2.GaussianBlur(x, (9, 9), 0)),
        ("scale_0p2", lambda x: scale_roundtrip(x, 0.2)),
        ("rotation_45_roundtrip", lambda x: rotate_roundtrip(x, 45.0)),
        ("gamma_1p5", lambda x: gamma(x, 1.5)),
        ("occlusion_50pct", lambda x: occlusion(x, 0.50, 3)),
    ])


def get_attacks(profile: str = "extended") -> Dict[str, Attack]:
    profile = profile.lower().strip()
    paper = _paper()
    if profile == "paper":
        return paper

    representative = OrderedDict([
        ("clean", lambda x: x.copy()),
        ("blur_sigma1", lambda x: cv2.GaussianBlur(x, (0, 0), 1.0)),
        ("median_3x3", lambda x: cv2.medianBlur(x, 3)),
        ("gaussian_noise_v0p003", lambda x: gaussian_noise(x, 0.003, 0)),
        ("salt_pepper_d0p1", lambda x: salt_pepper(x, 0.10, 0)),
        ("jpeg_q50", lambda x: jpeg(x, 50)),
        ("scale_0p5", lambda x: scale_roundtrip(x, 0.5)),
        ("rotation_5_roundtrip", lambda x: rotate_roundtrip(x, 5.0)),
        # True unsynchronised geometry tests (do not undo the transform).
        ("rotation_once_plus3", lambda x: rotate_once(x, 3.0)),
        ("rotation_once_minus3", lambda x: rotate_once(x, -3.0)),
        ("translation_8_8_roundtrip", lambda x: translate_roundtrip(x, 8.0, 8.0)),
        ("translation_once_4_4", lambda x: translate_once(x, 4.0, 4.0)),
        ("crop_10pct_resize", lambda x: crop_resize(x, 0.10)),
        ("gamma_1p5", lambda x: gamma(x, 1.5)),
        ("occlusion_25pct", lambda x: occlusion(x, 0.25, 3)),
    ])
    if profile == "representative":
        return representative

    extended = OrderedDict(paper)
    extended.update([
        ("median_3x3", lambda x: cv2.medianBlur(x, 3)),
        ("median_5x5", lambda x: cv2.medianBlur(x, 5)),
        ("average_3x3", lambda x: cv2.blur(x, (3, 3))),
        ("bilateral_d5", lambda x: cv2.bilateralFilter(x, 5, 35, 35)),
        ("speckle_v0p003", lambda x: speckle_noise(x, 0.003, 1)),
        ("jpeg_q70", lambda x: jpeg(x, 70)),
        ("jpeg_q30", lambda x: jpeg(x, 30)),
        ("scale_0p75", lambda x: scale_roundtrip(x, 0.75)),
        ("scale_0p5", lambda x: scale_roundtrip(x, 0.5)),
        ("rotation_3_roundtrip", lambda x: rotate_roundtrip(x, 3.0)),
        ("rotation_5_roundtrip", lambda x: rotate_roundtrip(x, 5.0)),
        ("rotation_once_plus1", lambda x: rotate_once(x, 1.0)),
        ("rotation_once_minus1", lambda x: rotate_once(x, -1.0)),
        ("rotation_once_plus3", lambda x: rotate_once(x, 3.0)),
        ("rotation_once_minus3", lambda x: rotate_once(x, -3.0)),
        ("rotation_once_plus5", lambda x: rotate_once(x, 5.0)),
        ("rotation_once_minus5", lambda x: rotate_once(x, -5.0)),
        ("translation_4_4_roundtrip", lambda x: translate_roundtrip(x, 4.0, 4.0)),
        ("translation_once_4_4", lambda x: translate_once(x, 4.0, 4.0)),
        ("translation_once_minus4_4", lambda x: translate_once(x, -4.0, 4.0)),
        ("translation_8_8_roundtrip", lambda x: translate_roundtrip(x, 8.0, 8.0)),
        ("crop_10pct_resize", lambda x: crop_resize(x, 0.10)),
        ("crop_25pct_resize", lambda x: crop_resize(x, 0.25)),
        ("gamma_0p8", lambda x: gamma(x, 0.8)),
        ("brightness_plus15", lambda x: brightness(x, 15.0)),
        ("brightness_minus15", lambda x: brightness(x, -15.0)),
        ("contrast_0p8", lambda x: contrast(x, 0.8)),
        ("contrast_1p2", lambda x: contrast(x, 1.2)),
        ("occlusion_10pct", lambda x: occlusion(x, 0.10, 3)),
        ("occlusion_25pct", lambda x: occlusion(x, 0.25, 3)),
    ])
    if profile == "extended":
        return extended

    if profile in ("stress", "all"):
        stress = OrderedDict(extended)
        stress.update([
            ("blur_sigma0p5", lambda x: cv2.GaussianBlur(x, (0, 0), 0.5)),
            ("blur_sigma1p5", lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)),
            ("gaussian_noise_v0p001", lambda x: gaussian_noise(x, 0.001, 0)),
            ("gaussian_noise_v0p005", lambda x: gaussian_noise(x, 0.005, 0)),
            ("speckle_v0p001", lambda x: speckle_noise(x, 0.001, 1)),
            ("speckle_v0p005", lambda x: speckle_noise(x, 0.005, 1)),
            ("salt_pepper_d0p02", lambda x: salt_pepper(x, 0.02, 0)),
            ("salt_pepper_d0p05", lambda x: salt_pepper(x, 0.05, 0)),
            ("jpeg_q90", lambda x: jpeg(x, 90)),
            ("jpeg_q40", lambda x: jpeg(x, 40)),
            ("rotation_minus5_roundtrip", lambda x: rotate_roundtrip(x, -5.0)),
            ("translation_minus8_8_roundtrip", lambda x: translate_roundtrip(x, -8.0, 8.0)),
            ("crop_15pct_resize", lambda x: crop_resize(x, 0.15)),
            ("gamma_1p2", lambda x: gamma(x, 1.2)),
            ("brightness_plus30", lambda x: brightness(x, 30.0)),
            ("brightness_minus30", lambda x: brightness(x, -30.0)),
            ("contrast_0p6", lambda x: contrast(x, 0.6)),
            ("contrast_1p4", lambda x: contrast(x, 1.4)),
        ])
        return stress

    raise ValueError(f"Unknown attack profile: {profile}. Use paper, representative, extended, or stress.")


def profile_names() -> tuple[str, ...]:
    return ("paper", "representative", "extended", "stress")
