#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ten independent Q-only / R-only QR-watermark candidates.

This module intentionally removes the previous adaptive |q44| / r44 dual-branch
method.  One experiment uses exactly one carrier rule for every selected block.

Q-only family (q21, q31):
    q_gaqim  Givens angular QIM
    q_npm    normalized pair margin
    q_lrm    log-ratio margin
    q_lqim   2-D normalized-pair lattice QIM constrained to the Givens circle
    q_smm    symmetric minimum margin

R-only family (first row of R):
    r_spqim  spread/projection row QIM on (r12,r13,r14)
    r_ndqim  normalized differential QIM
    r_lrqim  log-ratio row QIM
    r_rqim   repeated first-row QIM on (r12,r13,r14)
    r_aqim   adaptive-step first-row QIM on (r11,r12,r13,r14)

The scalar R-QIM primitive follows the supplied source equation exactly:
    x' = x - mod(x,S) + 0.75*S  for bit 1
    x' = x - mod(x,S) + 0.25*S  for bit 0
and extraction compares mod(x,S) with 0.5*S.

The comparison protocol is semi-blind because selected block coordinates are
stored.  There are no per-block Q/R branch flags anymore.

Robust-v3 changes keep exactly the same ten Q-only/R-only carriers, while
adding carrier-stability block selection, host-normalized margin allocation for
the sign/margin Q family, stronger stored-domain guard closure, and an optional
carrier-conformity self-synchronizer for small rotation/translation/crop-scale
misalignment.  The synchronizer uses the watermark carrier itself: no Q/R
hybrid branch, pilot band, ECC, or attack classifier is introduced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as structural_similarity


METHODS = (
    "q_gaqim", "q_npm", "q_lrm", "q_lqim", "q_smm",
    "r_spqim", "r_ndqim", "r_lrqim", "r_rqim", "r_aqim",
)
Q_METHODS = frozenset(m for m in METHODS if m.startswith("q_"))
R_METHODS = frozenset(m for m in METHODS if m.startswith("r_"))

DISPLAY_NAMES = {
    "q_gaqim": "Q-GAQIM",
    "q_npm": "Q-NPM",
    "q_lrm": "Q-LRM",
    "q_lqim": "Q-LQIM",
    "q_smm": "Q-SMM",
    "r_spqim": "R-SPQIM",
    "r_ndqim": "R-NDQIM",
    "r_lrqim": "R-LRQIM",
    "r_rqim": "R-RQIM",
    "r_aqim": "R-AQIM",
}


@dataclass(frozen=True)
class MethodConfig:
    method: str = "q_gaqim"
    block_size: int = 4
    wm_size: int = 64
    det_eps: float = 1e-8
    arnold_iter: int = 10
    private_key: str = "KB123"
    repetition_override: int = 1  # fair carrier screening: one block per bit
    eps: float = 1e-9

    # Q-family strengths
    # Robust defaults.  These remain below/near the PSNR=50 screening budget
    # on the bundled data; the sweep runner should still be used for final tuning.
    q_angle_step_deg: float = 3.0
    q_margin: float = 0.007
    q_log_margin: float = 0.013
    q_lattice_step: float = 0.02

    # R-family strengths
    r_step: float = 16.0
    r_norm_step: float = 0.04
    r_log_step: float = 0.06
    r_alpha: float = 0.40
    r_s_min: float = 4.0
    r_s_max: float = 20.0
    r_projection: str = "balanced"  # balanced | average
    aqim_rounds: int = 3

    # Integer-domain closure / extraction.  QR is re-estimated after the
    # floating block is rounded to uint8, so clean decoding must be checked in
    # the stored domain rather than assumed from the pre-rounding carrier.
    soft_vote: bool = True
    closure_rounds: int = 4
    # Sign/margin Q methods benefit from a deep stored-domain guard.  Periodic
    # QIM carriers have a natural |evidence| target near one and therefore use
    # q_periodic_guard instead.
    q_sign_guard: float = 2.0
    q_periodic_guard: float = 0.80
    r_closure_guard: float = 0.80
    q_integer_repair_radius: int = 10

    # Robust carrier selection.  "stability" keeps the same q21/q31 or R-row
    # carrier, but chooses the payload blocks whose carrier changes least under
    # a fixed weak perturbation bank.  It is bit-independent and stores no new
    # per-bit flag.
    block_selection: str = "stability"   # stability | chaotic
    stability_margin_boost: float = 0.80
    stability_reference_quantile: float = 0.90

    # Optional self-synchronisation at extraction.  This is a decoder wrapper,
    # not a second embedding band.  Keep "none" for a strict no-search study;
    # use "auto" when maximum geometric robustness is desired.
    sync_mode: str = "none"              # none | auto
    sync_sample_stride: int = 4
    sync_accept_score: float = 0.52
    sync_accept_ratio: float = 1.35
    sync_translation_radius: int = 8
    sync_translation_step: int = 2
    sync_rotation_deg: float = 5.0
    sync_rotation_step_deg: float = 1.0
    sync_scale_min: float = 0.85
    sync_scale_step: float = 0.025


@dataclass
class EmbeddingResult:
    watermarked: np.ndarray
    side_info: Dict[str, np.ndarray]
    diagnostics: List[Dict[str, float]]
    elapsed_seconds: float


@dataclass
class ExtractionResult:
    watermark: np.ndarray
    confidence: float
    candidate_name: str
    valid_observations: int
    elapsed_seconds: float


def validate_config(cfg: MethodConfig) -> None:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method={cfg.method!r}. Choose one of: {', '.join(METHODS)}")
    if cfg.block_size != 4:
        raise ValueError("This implementation requires 4x4 QR blocks")
    if int(cfg.repetition_override) < 1:
        raise ValueError("repetition_override must be >= 1")
    if cfg.block_selection not in {"stability", "chaotic"}:
        raise ValueError("block_selection must be 'stability' or 'chaotic'")
    if cfg.sync_mode not in {"none", "auto"}:
        raise ValueError("sync_mode must be 'none' or 'auto'")


def method_strength_grid(method: str) -> List[float]:
    """Robustness-oriented search grid under a PSNR constraint.

    The old grids were too weak for several carriers (especially Q-GAQIM and
    R-RQIM), leaving a large unused distortion budget.  The runner still
    filters by mean PSNR and clean NC, so stronger points are safe to include.
    """
    if method == "q_gaqim":
        return [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    if method in {"q_npm", "q_smm"}:
        return [0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030]
    if method == "q_lrm":
        return [0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.040, 0.060]
    if method == "q_lqim":
        return [0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060]
    if method in {"r_spqim", "r_rqim"}:
        return [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
    if method == "r_ndqim":
        return [0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]
    if method == "r_lrqim":
        return [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24]
    if method == "r_aqim":
        return [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20]
    raise ValueError(method)


def with_strength(cfg: MethodConfig, strength: float) -> MethodConfig:
    method = cfg.method
    if method == "q_gaqim":
        return replace(cfg, q_angle_step_deg=float(strength))
    if method in {"q_npm", "q_smm"}:
        return replace(cfg, q_margin=float(strength))
    if method == "q_lrm":
        return replace(cfg, q_log_margin=float(strength))
    if method == "q_lqim":
        return replace(cfg, q_lattice_step=float(strength))
    if method in {"r_spqim", "r_rqim"}:
        return replace(cfg, r_step=float(strength))
    if method == "r_ndqim":
        return replace(cfg, r_norm_step=float(strength))
    if method == "r_lrqim":
        return replace(cfg, r_log_step=float(strength))
    if method == "r_aqim":
        return replace(cfg, r_alpha=float(strength))
    raise ValueError(method)


def strength_value(cfg: MethodConfig) -> float:
    if cfg.method == "q_gaqim": return float(cfg.q_angle_step_deg)
    if cfg.method in {"q_npm", "q_smm"}: return float(cfg.q_margin)
    if cfg.method == "q_lrm": return float(cfg.q_log_margin)
    if cfg.method == "q_lqim": return float(cfg.q_lattice_step)
    if cfg.method in {"r_spqim", "r_rqim"}: return float(cfg.r_step)
    if cfg.method == "r_ndqim": return float(cfg.r_norm_step)
    if cfg.method == "r_lrqim": return float(cfg.r_log_step)
    if cfg.method == "r_aqim": return float(cfg.r_alpha)
    raise ValueError(cfg.method)


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _read_color(path: str | os.PathLike[str]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def load_binary_watermark(path: str | os.PathLike[str], size: int = 64, bg: int = 255) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read watermark: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        bg_img = np.full_like(bgr, float(bg), dtype=np.float32)
        gray = cv2.cvtColor((bgr * alpha + bg_img * (1.0 - alpha)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8)
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return ((gray >= 128).astype(np.uint8) * 255)


def arnold_transform(image: np.ndarray, iterations: int) -> np.ndarray:
    n = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square watermark")
    out = image.copy()
    for _ in range(int(iterations)):
        nxt = np.empty_like(out)
        for x in range(n):
            for y in range(n):
                nxt[(x + y) % n, (x + 2 * y) % n] = out[x, y]
        out = nxt
    return out


def invert_arnold_transform(image: np.ndarray, iterations: int) -> np.ndarray:
    n = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square watermark")
    out = image.copy()
    for _ in range(int(iterations)):
        nxt = np.empty_like(out)
        for x in range(n):
            for y in range(n):
                nxt[(2 * x - y) % n, (-x + y) % n] = out[x, y]
        out = nxt
    return out


def canonical_qr(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_mat, r_mat = np.linalg.qr(block.astype(np.float64))
    signs = np.sign(np.diag(r_mat))
    signs[signs == 0] = 1.0
    d = np.diag(signs)
    return q_mat @ d, d @ r_mat


def determinant_valid(block: np.ndarray, eps: float) -> bool:
    try:
        return abs(float(np.linalg.det(block.astype(np.float64)))) > float(eps)
    except np.linalg.LinAlgError:
        return False


def enumerate_blocks(shape: Sequence[int], block_size: int) -> List[Tuple[int, int]]:
    h, w = int(shape[0]), int(shape[1])
    h0 = (h // block_size) * block_size
    w0 = (w // block_size) * block_size
    return [(r, c) for r in range(0, h0, block_size) for c in range(0, w0, block_size)]


def eligible_blocks(channel: np.ndarray, cfg: MethodConfig) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    bs = cfg.block_size
    for r, c in enumerate_blocks(channel.shape, bs):
        if determinant_valid(channel[r:r + bs, c:c + bs], cfg.det_eps):
            out.append((r, c))
    return out



# ----- Robust-v3 carrier-stability block selection ------------------------

_STABILITY_CACHE: Dict[Tuple[str, str, int, str], Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float], float]] = {}


def _channel_cache_key(channel: np.ndarray) -> str:
    # Small deterministic digest; avoids recomputing the same stability plan
    # across two watermarks or a strength sweep on the same host.
    return hashlib.blake2b(np.ascontiguousarray(channel, dtype=np.uint8).tobytes(), digest_size=12).hexdigest()


def _gray_jpeg(channel: np.ndarray, quality: int = 70) -> np.ndarray:
    u = np.clip(np.rint(channel), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return u.astype(np.float64)
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return (u if dec is None else dec).astype(np.float64)


def _scale_gray(channel: np.ndarray, factor: float = 0.75) -> np.ndarray:
    h, w = channel.shape
    u = np.clip(np.rint(channel), 0, 255).astype(np.uint8)
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    tmp = cv2.resize(u, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(tmp, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float64)


def _q_pair_from_spatial_block(block: np.ndarray, eps: float) -> Tuple[float, float]:
    # With canonical R11 >= 0, Q[:,0] is simply the normalized first spatial
    # column.  Avoiding a full QR here makes stability screening much faster.
    col = np.asarray(block[:, 0], dtype=np.float64)
    n = float(np.linalg.norm(col))
    if n <= eps:
        return 0.0, 0.0
    q1 = col / n
    return float(q1[1]), float(q1[2])


def _raw_carrier_feature(block: np.ndarray, cfg: MethodConfig) -> np.ndarray:
    if cfg.method in Q_METHODS:
        x, y = _q_pair_from_spatial_block(block, cfg.eps)
        if cfg.method == "q_gaqim":
            th = math.atan2(y, x)
            return np.asarray([math.cos(th), math.sin(th)], dtype=np.float64)
        if cfg.method == "q_npm":
            return np.asarray([(x - y) / (math.hypot(x, y) + cfg.eps)], dtype=np.float64)
        if cfg.method == "q_lrm":
            return np.asarray([math.log(abs(x) + cfg.eps) - math.log(abs(y) + cfg.eps)], dtype=np.float64)
        if cfg.method == "q_smm":
            return np.asarray([x - y], dtype=np.float64)
        if cfg.method == "q_lqim":
            return np.asarray([x, y], dtype=np.float64)
    q, r = canonical_qr(block)
    if cfg.method == "r_spqim":
        a = _r_projection_vector(cfg)
        return np.asarray([float(np.dot(a, r[0, 1:4]))], dtype=np.float64)
    if cfg.method == "r_ndqim":
        return np.asarray([_normdiff(float(r[0, 1]), float(r[0, 2]), cfg.eps)], dtype=np.float64)
    if cfg.method == "r_lrqim":
        return np.asarray([_logratio(float(r[0, 1]), float(r[0, 2]), cfg.eps)], dtype=np.float64)
    if cfg.method == "r_rqim":
        v = r[0, 1:4].astype(np.float64)
        return v / (float(np.linalg.norm(v)) + cfg.eps)
    if cfg.method == "r_aqim":
        v = r[0, 0:4].astype(np.float64)
        return v / (float(np.linalg.norm(v)) + cfg.eps)
    raise ValueError(cfg.method)


def _stability_select_blocks(
    channel: np.ndarray,
    positions: Sequence[Tuple[int, int]],
    required: int,
    cfg: MethodConfig,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float], float]:
    if cfg.block_selection == "chaotic":
        # Preserve the original keyed random subset when robust stability
        # screening is disabled.  A second keyed permutation below only changes
        # the payload order inside the already-selected subset.
        selected = chaotic_permute_positions(positions, cfg.private_key)[:required]
        return selected, {p: 0.0 for p in selected}, 1.0

    key = (_channel_cache_key(channel), cfg.method, int(required), cfg.r_projection)
    cached = _STABILITY_CACHE.get(key)
    if cached is not None:
        return cached

    # Fixed weak probes estimate carrier sensitivity without selecting an attack
    # at extraction time.  The same bank is used for every image/method.
    probes = [
        cv2.GaussianBlur(channel.astype(np.float32), (0, 0), 0.5).astype(np.float64),
        _gray_jpeg(channel, 70),
        _scale_gray(channel, 0.75),
    ]
    bs = cfg.block_size
    scored: List[Tuple[float, int, int]] = []
    for rr, cc in positions:
        try:
            f0 = _raw_carrier_feature(channel[rr:rr+bs, cc:cc+bs], cfg)
            drifts = []
            for probe in probes:
                fp = _raw_carrier_feature(probe[rr:rr+bs, cc:cc+bs], cfg)
                drifts.append(float(np.linalg.norm(fp - f0)))
            drift = float(np.mean(drifts))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            drift = math.inf
        scored.append((drift, int(rr), int(cc)))
    scored.sort(key=lambda z: z[0])
    chosen = scored[:required]
    selected = [(rr, cc) for _, rr, cc in chosen]
    score_map = {(rr, cc): float(d) for d, rr, cc in chosen}
    finite = np.asarray([d for d, _, _ in chosen if np.isfinite(d)], dtype=np.float64)
    if finite.size:
        q = float(np.clip(cfg.stability_reference_quantile, 0.5, 1.0))
        drift_ref = max(float(np.quantile(finite, q)), cfg.eps)
    else:
        drift_ref = 1.0
    out = (selected, score_map, drift_ref)
    _STABILITY_CACHE[key] = out
    return out


def _adaptive_block_cfg(cfg: MethodConfig, drift: float, drift_ref: float) -> MethodConfig:
    # These three Q methods decode by sign; therefore the embedder may allocate
    # a larger margin to less-stable blocks without transmitting a per-block
    # strength.  The multiplier is normalized per host to avoid runaway strength
    # on difficult images.
    u = float(np.clip(float(drift) / max(float(drift_ref), cfg.eps), 0.0, 1.0))
    mult = 1.0 + float(cfg.stability_margin_boost) * u
    if cfg.method in {"q_smm", "q_npm"}:
        return replace(cfg, q_margin=float(cfg.q_margin) * mult)
    if cfg.method == "q_lrm":
        return replace(cfg, q_log_margin=float(cfg.q_log_margin) * mult)
    return cfg


def _closure_guard_for_method(cfg: MethodConfig) -> float:
    if cfg.method in {"q_smm", "q_npm", "q_lrm"}:
        return float(cfg.q_sign_guard)
    if cfg.method in {"q_gaqim", "q_lqim"}:
        return float(cfg.q_periodic_guard)
    return float(cfg.r_closure_guard)


def _chaotic_order_count(n: int, key: str) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    digest = hashlib.sha512(key.encode("utf-8")).digest()
    x = int.from_bytes(digest[:8], "big") / float(2**64)
    if x <= 0.0 or x >= 1.0:
        x = 0.6180339887498949
    for _ in range(100):
        x = 3.99 * x * (1.0 - x)
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = 3.99 * x * (1.0 - x)
        seq[i] = x
    return np.argsort(seq, kind="mergesort")


def chaotic_permute_positions(positions: Sequence[Tuple[int, int]], key: str) -> List[Tuple[int, int]]:
    order = _chaotic_order_count(len(positions), key)
    return [positions[int(i)] for i in order]


def _wrap_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _circular_distance(a: float, b: float) -> float:
    return abs(_wrap_pi(a - b))


def _nearest_equivalent_angle(base: float, reference: float) -> float:
    k = round((reference - base) / (2.0 * math.pi))
    return base + 2.0 * math.pi * k


def _pair_angle(q_mat: np.ndarray) -> float:
    return math.atan2(float(q_mat[2, 0]), float(q_mat[1, 0]))


def _left_givens23(q_mat: np.ndarray, delta: float) -> np.ndarray:
    g = np.eye(4, dtype=np.float64)
    c, s = math.cos(delta), math.sin(delta)
    g[1, 1] = c; g[1, 2] = -s
    g[2, 1] = s; g[2, 2] = c
    return g @ q_mat


def _q_block_at_angle(q_mat: np.ndarray, r_mat: np.ndarray, target_theta: float) -> Tuple[np.ndarray, float]:
    theta = _pair_angle(q_mat)
    delta = _wrap_pi(target_theta - theta)
    q2 = _left_givens23(q_mat, delta)
    return q2 @ r_mat, delta


def _nearest_coset(theta: float, step: float, bit: int) -> float:
    if step <= 0:
        raise ValueError("Q angular step must be positive")
    phase = (0.25 if int(bit) == 0 else 0.75) * step
    k = round((theta - phase) / step)
    return phase + k * step


def _q_gaqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    step = math.radians(cfg.q_angle_step_deg)
    theta = _pair_angle(q)
    target = _nearest_coset(theta, step, bit)
    return _q_block_at_angle(q, r, target)


def _q_gaqim_evidence(q: np.ndarray, cfg: MethodConfig) -> float:
    step = math.radians(cfg.q_angle_step_deg)
    theta = _pair_angle(q)
    d0 = _circular_distance(theta, _nearest_coset(theta, step, 0))
    d1 = _circular_distance(theta, _nearest_coset(theta, step, 1))
    return (d0 - d1) / max(step * 0.5, cfg.eps)


def _nearest_angle_for_cos_minus_sin(theta: float, target_value: float) -> float:
    # cos(theta)-sin(theta) = sqrt(2)*cos(theta+pi/4)
    v = float(np.clip(target_value / math.sqrt(2.0), -1.0, 1.0))
    alpha = math.acos(v)
    bases = (-math.pi / 4.0 + alpha, -math.pi / 4.0 - alpha)
    candidates = [_nearest_equivalent_angle(b, theta) for b in bases]
    return min(candidates, key=lambda x: abs(x - theta))


def _q_npm_feature(q: np.ndarray, eps: float) -> float:
    x, y = float(q[1, 0]), float(q[2, 0])
    rho = math.hypot(x, y)
    return (x - y) / (rho + eps)


def _q_npm_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    s = 1.0 if bit else -1.0
    f = _q_npm_feature(q, cfg.eps)
    if s * f >= cfg.q_margin:
        return q @ r, 0.0
    target = float(np.clip(s * cfg.q_margin, -math.sqrt(2.0) + 1e-9, math.sqrt(2.0) - 1e-9))
    theta = _pair_angle(q)
    target_theta = _nearest_angle_for_cos_minus_sin(theta, target)
    return _q_block_at_angle(q, r, target_theta)


def _q_npm_evidence(q: np.ndarray, cfg: MethodConfig) -> float:
    return _q_npm_feature(q, cfg.eps) / max(cfg.q_margin, cfg.eps)


def _q_smm_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    x, y = float(q[1, 0]), float(q[2, 0])
    rho = math.hypot(x, y)
    s = 1.0 if bit else -1.0
    d = x - y
    if s * d >= cfg.q_margin:
        return q @ r, 0.0
    if rho <= cfg.eps:
        return q @ r, 0.0
    normalized_target = s * min(cfg.q_margin / rho, math.sqrt(2.0) - 1e-9)
    theta = _pair_angle(q)
    target_theta = _nearest_angle_for_cos_minus_sin(theta, normalized_target)
    return _q_block_at_angle(q, r, target_theta)


def _q_smm_evidence(q: np.ndarray, cfg: MethodConfig) -> float:
    return (float(q[1, 0]) - float(q[2, 0])) / max(cfg.q_margin, cfg.eps)


def _q_lrm_feature(q: np.ndarray, eps: float) -> float:
    x, y = abs(float(q[1, 0])), abs(float(q[2, 0]))
    return math.log(x + eps) - math.log(y + eps)


def _nearest_angle_for_log_ratio(theta: float, target: float) -> float:
    # ignoring eps at the boundary: log(|cos|/|sin|)=target
    ratio = math.exp(float(np.clip(target, -20.0, 20.0)))
    alpha = math.atan(1.0 / ratio)  # |tan(theta)| = 1/ratio
    bases = (alpha, -alpha, math.pi - alpha, -math.pi + alpha)
    candidates = [_nearest_equivalent_angle(b, theta) for b in bases]
    return min(candidates, key=lambda x: abs(x - theta))


def _q_lrm_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    s = 1.0 if bit else -1.0
    f = _q_lrm_feature(q, cfg.eps)
    if s * f >= cfg.q_log_margin:
        return q @ r, 0.0
    theta = _pair_angle(q)
    target_theta = _nearest_angle_for_log_ratio(theta, s * cfg.q_log_margin)
    return _q_block_at_angle(q, r, target_theta)


def _q_lrm_evidence(q: np.ndarray, cfg: MethodConfig) -> float:
    return _q_lrm_feature(q, cfg.eps) / max(cfg.q_log_margin, cfg.eps)


def _nearest_pair_lattice_point(pair: np.ndarray, step: float, bit: int) -> Tuple[np.ndarray, float]:
    """Nearest feasible point in one of two 2-D square-lattice cosets.

    Lambda_0 = step * Z^2
    Lambda_1 = step * (Z^2 + (1/2,1/2)).
    Only points strictly inside the unit disk are allowed because (q21,q31)
    belongs to a unit-norm column of an orthogonal matrix.
    """
    h = float(step)
    if h <= 0:
        raise ValueError("q_lattice_step must be positive")
    off = 0.0 if int(bit) == 0 else 0.5
    pair = np.asarray(pair, dtype=np.float64)
    center = pair / h - off
    i0, j0 = int(round(center[0])), int(round(center[1]))
    best: Optional[np.ndarray] = None
    best_d = math.inf
    # Expand enough to find a feasible nearby lattice point even near the disk edge.
    for radius in (2, 4, 8):
        for i in range(i0 - radius, i0 + radius + 1):
            for j in range(j0 - radius, j0 + radius + 1):
                v = h * np.asarray([i + off, j + off], dtype=np.float64)
                if float(np.dot(v, v)) >= 1.0 - 1e-8:
                    continue
                d = float(np.linalg.norm(pair - v))
                if d < best_d:
                    best_d, best = d, v
        if best is not None:
            break
    if best is None:
        raise RuntimeError("No feasible Q-pair lattice point inside unit disk")
    return best, best_d


def _complete_q_first_column(q: np.ndarray, pair_target: np.ndarray, eps: float) -> np.ndarray:
    """Complete target (q21,q31) to a unit first column near the original one."""
    old = q[:, 0].astype(np.float64)
    x, y = float(pair_target[0]), float(pair_target[1])
    rem2 = max(0.0, 1.0 - x*x - y*y)
    rem = math.sqrt(rem2)
    tail = np.asarray([old[0], old[3]], dtype=np.float64)
    n = float(np.linalg.norm(tail))
    if n <= eps:
        a, d = rem, 0.0
    else:
        a, d = (tail * (rem / n)).tolist()
    v = np.asarray([a, x, y, d], dtype=np.float64)
    v /= max(float(np.linalg.norm(v)), eps)
    return v


def _minimal_rotation_map(u: np.ndarray, v: np.ndarray, eps: float) -> np.ndarray:
    """Orthogonal minimum-plane rotation mapping unit vector u to unit vector v."""
    u = np.asarray(u, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    u = u / max(float(np.linalg.norm(u)), eps)
    v = v / max(float(np.linalg.norm(v)), eps)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(u.size, dtype=np.float64)
    if c < -1.0 + 1e-10:
        # Rare antipodal case: use a stable Householder reflection followed by -I
        idx = int(np.argmin(np.abs(u)))
        w = np.zeros_like(u); w[idx] = 1.0
        w = w - np.dot(w, u) * u
        w /= max(float(np.linalg.norm(w)), eps)
        # pi rotation in span(u,w): both u and w change sign.
        return np.eye(u.size) - 2.0*np.outer(u,u) - 2.0*np.outer(w,w)
    w = v - c * u
    s = float(np.linalg.norm(w))
    w /= max(s, eps)
    # In basis (u,w): [[c,-s],[s,c]], therefore R u = c*u + s*w = v.
    return (np.eye(u.size)
            + (c - 1.0) * (np.outer(u, u) + np.outer(w, w))
            + s * (np.outer(w, u) - np.outer(u, w)))


def _q_lqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    pair = np.asarray([q[1, 0], q[2, 0]], dtype=np.float64)
    target, distance = _nearest_pair_lattice_point(pair, cfg.q_lattice_step, bit)
    v = _complete_q_first_column(q, target, cfg.eps)
    rot = _minimal_rotation_map(q[:, 0], v, cfg.eps)
    q2 = rot @ q
    return q2 @ r, distance


def _q_lqim_evidence(q: np.ndarray, cfg: MethodConfig) -> float:
    pair = np.asarray([q[1, 0], q[2, 0]], dtype=np.float64)
    _, d0 = _nearest_pair_lattice_point(pair, cfg.q_lattice_step, 0)
    _, d1 = _nearest_pair_lattice_point(pair, cfg.q_lattice_step, 1)
    return (d0 - d1) / max(0.5 * cfg.q_lattice_step, cfg.eps)


# ----- R-only primitives ---------------------------------------------------

def binary_qim_embed(x: float, bit: int, step: float) -> float:
    """Exact source-paper scalar embedding rule with T1=.75S/T2=.25S."""
    s = float(step)
    if s <= 0:
        raise ValueError("QIM step must be positive")
    target = 0.75 * s if int(bit) == 1 else 0.25 * s
    return float(x) - (float(x) % s) + target


def binary_qim_evidence(x: float, step: float, eps: float = 1e-12) -> float:
    """Positive => bit 1; negative => bit 0, matching threshold S/2."""
    s = float(step)
    if s <= 0:
        return 0.0
    return ((float(x) % s) - 0.5 * s) / max(0.25 * s, eps)


def _r_projection_vector(cfg: MethodConfig) -> np.ndarray:
    if cfg.r_projection == "average":
        a = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    elif cfg.r_projection == "balanced":
        a = np.asarray([1.0, -2.0, 1.0], dtype=np.float64)
    else:
        raise ValueError("r_projection must be 'balanced' or 'average'")
    return a / np.linalg.norm(a)


def _r_spqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    r2 = r.copy()
    a = _r_projection_vector(cfg)
    row = r2[0, 1:4].copy()
    c = float(np.dot(a, row))
    c2 = binary_qim_embed(c, bit, cfg.r_step)
    row2 = row + (c2 - c) * a
    r2[0, 1:4] = row2
    return q @ r2, float(np.linalg.norm(row2 - row))


def _r_spqim_evidence(r: np.ndarray, cfg: MethodConfig) -> float:
    a = _r_projection_vector(cfg)
    c = float(np.dot(a, r[0, 1:4]))
    return binary_qim_evidence(c, cfg.r_step, cfg.eps)


def _normdiff(a: float, b: float, eps: float) -> float:
    return (a - b) / (abs(a) + abs(b) + eps)


def _r_ndqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    r2 = r.copy()
    a0, b0 = float(r2[0, 1]), float(r2[0, 2])
    d = _normdiff(a0, b0, cfg.eps)
    target = binary_qim_embed(d, bit, cfg.r_norm_step)
    target = float(np.clip(target, -0.98, 0.98))
    a, b = a0, b0
    # Symmetric minimum-change correction; iterate because denominator changes.
    for _ in range(5):
        denom = abs(a) + abs(b) + cfg.eps
        desired_difference = target * denom
        delta = 0.5 * (desired_difference - (a - b))
        a += delta
        b -= delta
        if abs(_normdiff(a, b, cfg.eps) - target) < 1e-7:
            break
    r2[0, 1], r2[0, 2] = a, b
    return q @ r2, math.hypot(a - a0, b - b0)


def _r_ndqim_evidence(r: np.ndarray, cfg: MethodConfig) -> float:
    d = _normdiff(float(r[0, 1]), float(r[0, 2]), cfg.eps)
    return binary_qim_evidence(d, cfg.r_norm_step, cfg.eps)


def _logratio(a: float, b: float, eps: float) -> float:
    return math.log(abs(a) + eps) - math.log(abs(b) + eps)


def _r_lrqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    r2 = r.copy()
    a0, b0 = float(r2[0, 1]), float(r2[0, 2])
    d = _logratio(a0, b0, cfg.eps)
    target = binary_qim_embed(d, bit, cfg.r_log_step)
    aa, bb = abs(a0) + cfg.eps, abs(b0) + cfg.eps
    g = math.sqrt(aa * bb)
    ma = max(0.0, g * math.exp(0.5 * target) - cfg.eps)
    mb = max(0.0, g * math.exp(-0.5 * target) - cfg.eps)
    sa = 1.0 if a0 >= 0 else -1.0
    sb = 1.0 if b0 >= 0 else -1.0
    a, b = sa * ma, sb * mb
    r2[0, 1], r2[0, 2] = a, b
    return q @ r2, math.hypot(a - a0, b - b0)


def _r_lrqim_evidence(r: np.ndarray, cfg: MethodConfig) -> float:
    d = _logratio(float(r[0, 1]), float(r[0, 2]), cfg.eps)
    return binary_qim_evidence(d, cfg.r_log_step, cfg.eps)


def _r_rqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    r2 = r.copy()
    before = r2[0, 1:4].copy()
    for j in (1, 2, 3):
        r2[0, j] = binary_qim_embed(float(r2[0, j]), bit, cfg.r_step)
    return q @ r2, float(np.linalg.norm(r2[0, 1:4] - before))


def _r_rqim_evidence(r: np.ndarray, cfg: MethodConfig) -> float:
    ev = [binary_qim_evidence(float(r[0, j]), cfg.r_step, cfg.eps) for j in (1, 2, 3)]
    return float(np.mean(ev))


def _adaptive_r_step(r: np.ndarray, cfg: MethodConfig) -> float:
    # Use the trailing 3x3 triangular part as the scale reference.  The R-only
    # embedder changes only row 1, so this reference is invariant before pixel
    # rounding and substantially more stable than deriving S from the row that
    # is itself being quantized.
    energy = float(np.linalg.norm(r[1:, 1:]))
    return float(np.clip(cfg.r_alpha * energy, cfg.r_s_min, cfg.r_s_max))


def _r_aqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    r2 = r.copy()
    before = r2[0, 0:4].copy()
    step = _adaptive_r_step(r, cfg)
    for j in (0, 1, 2, 3):
        r2[0, j] = binary_qim_embed(float(r2[0, j]), bit, step)
    return q @ r2, float(np.linalg.norm(r2[0, 0:4] - before))


def _r_aqim_evidence(r: np.ndarray, cfg: MethodConfig) -> float:
    step = _adaptive_r_step(r, cfg)
    ev = [binary_qim_evidence(float(r[0, j]), step, cfg.eps) for j in (0, 1, 2, 3)]
    return float(np.mean(ev))


def embed_qr_block(block: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    validate_config(cfg)
    q, r = canonical_qr(block)
    if cfg.method == "q_gaqim": return _q_gaqim_embed(q, r, bit, cfg)
    if cfg.method == "q_npm": return _q_npm_embed(q, r, bit, cfg)
    if cfg.method == "q_lrm": return _q_lrm_embed(q, r, bit, cfg)
    if cfg.method == "q_lqim": return _q_lqim_embed(q, r, bit, cfg)
    if cfg.method == "q_smm": return _q_smm_embed(q, r, bit, cfg)
    if cfg.method == "r_spqim": return _r_spqim_embed(q, r, bit, cfg)
    if cfg.method == "r_ndqim": return _r_ndqim_embed(q, r, bit, cfg)
    if cfg.method == "r_lrqim": return _r_lrqim_embed(q, r, bit, cfg)
    if cfg.method == "r_rqim": return _r_rqim_embed(q, r, bit, cfg)
    if cfg.method == "r_aqim": return _r_aqim_embed(q, r, bit, cfg)
    raise ValueError(cfg.method)


def extract_qr_evidence(block: np.ndarray, cfg: MethodConfig) -> float:
    validate_config(cfg)
    q, r = canonical_qr(block)
    if cfg.method == "q_gaqim": return _q_gaqim_evidence(q, cfg)
    if cfg.method == "q_npm": return _q_npm_evidence(q, cfg)
    if cfg.method == "q_lrm": return _q_lrm_evidence(q, cfg)
    if cfg.method == "q_lqim": return _q_lqim_evidence(q, cfg)
    if cfg.method == "q_smm": return _q_smm_evidence(q, cfg)
    if cfg.method == "r_spqim": return _r_spqim_evidence(r, cfg)
    if cfg.method == "r_ndqim": return _r_ndqim_evidence(r, cfg)
    if cfg.method == "r_lrqim": return _r_lrqim_evidence(r, cfg)
    if cfg.method == "r_rqim": return _r_rqim_evidence(r, cfg)
    if cfg.method == "r_aqim": return _r_aqim_evidence(r, cfg)
    raise ValueError(cfg.method)


def _stored_block(block: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(block), 0, 255).astype(np.uint8)


def _q_evidence_from_first_column(col: np.ndarray, cfg: MethodConfig) -> float:
    """Evaluate the q21/q31 carrier directly from the spatial first column.

    Under canonical QR with R11 >= 0, the first column of Q is the normalized
    first spatial column.  This identity makes the post-rounding repair both
    exact for the chosen Q feature and much cheaper than repeating QR hundreds
    of times during a local integer search.
    """
    v = np.asarray(col, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= cfg.eps:
        return 0.0
    q1 = v / n
    x, y = float(q1[1]), float(q1[2])
    if cfg.method == "q_gaqim":
        step = math.radians(cfg.q_angle_step_deg)
        theta = math.atan2(y, x)
        d0 = _circular_distance(theta, _nearest_coset(theta, step, 0))
        d1 = _circular_distance(theta, _nearest_coset(theta, step, 1))
        return (d0 - d1) / max(0.5 * step, cfg.eps)
    if cfg.method == "q_npm":
        return ((x - y) / (math.hypot(x, y) + cfg.eps)) / max(cfg.q_margin, cfg.eps)
    if cfg.method == "q_lrm":
        return (math.log(abs(x) + cfg.eps) - math.log(abs(y) + cfg.eps)) / max(cfg.q_log_margin, cfg.eps)
    if cfg.method == "q_smm":
        return (x - y) / max(cfg.q_margin, cfg.eps)
    if cfg.method == "q_lqim":
        pair = np.asarray([x, y], dtype=np.float64)
        _, d0 = _nearest_pair_lattice_point(pair, cfg.q_lattice_step, 0)
        _, d1 = _nearest_pair_lattice_point(pair, cfg.q_lattice_step, 1)
        return (d0 - d1) / max(0.5 * cfg.q_lattice_step, cfg.eps)
    raise ValueError(cfg.method)


def _repair_q_integer_carrier(stored: np.ndarray, bit: int, cfg: MethodConfig) -> np.ndarray:
    """Smallest local integer repair of the recovered q21/q31 decision.

    The floating Q-domain update can move back across a decision boundary when
    Q@R is rounded to uint8.  Search only the two first-column samples that
    determine q21 and q31, in increasing squared-distance order, and stop at
    the first candidate meeting the requested guard.  No extra side information
    is stored and the extractor uses the same original carrier rule.
    """
    if cfg.method not in Q_METHODS:
        return stored
    sign = 1.0 if int(bit) else -1.0
    guard = _closure_guard_for_method(cfg)
    base = stored.copy()
    ev0 = _q_evidence_from_first_column(base[:, 0], cfg)
    if np.isfinite(ev0) and sign * ev0 >= guard:
        return base

    radius = max(0, int(cfg.q_integer_repair_radius))
    if radius <= 0:
        return base
    a0, b0 = int(base[1, 0]), int(base[2, 0])
    offsets = [(da*da + db*db, da, db)
               for da in range(-radius, radius + 1)
               for db in range(-radius, radius + 1)
               if da or db]
    offsets.sort(key=lambda z: z[0])

    fallback = base
    best_margin = sign * ev0 if np.isfinite(ev0) else -math.inf
    for _cost, da, db in offsets:
        aa = int(np.clip(a0 + da, 0, 255))
        bb = int(np.clip(b0 + db, 0, 255))
        if aa == a0 and bb == b0:
            continue
        col = base[:, 0].astype(np.float64).copy()
        col[1], col[2] = aa, bb
        ev = _q_evidence_from_first_column(col, cfg)
        if not np.isfinite(ev):
            continue
        margin = sign * ev
        if margin > best_margin:
            best_margin = margin
            fallback = base.copy()
            fallback[1, 0], fallback[2, 0] = aa, bb
        if margin >= guard:
            out = base.copy()
            out[1, 0], out[2, 0] = aa, bb
            return out
    return fallback


def _close_r_integer_block(stored: np.ndarray, bit: int, cfg: MethodConfig) -> np.ndarray:
    """Re-embed an R carrier after uint8 rounding until its clean decision is safe."""
    if cfg.method not in R_METHODS:
        return stored
    sign = 1.0 if int(bit) else -1.0
    out = stored.copy()
    for _ in range(max(0, int(cfg.closure_rounds))):
        try:
            ev = float(extract_qr_evidence(out.astype(np.float64), cfg))
        except Exception:
            ev = 0.0
        if np.isfinite(ev) and sign * ev >= _closure_guard_for_method(cfg):
            break
        try:
            fb, _ = embed_qr_block(out.astype(np.float64), bit, cfg)
            nxt = _stored_block(fb)
        except Exception:
            break
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out


def embed_array(host_img: np.ndarray, watermark_binary: np.ndarray, cfg: MethodConfig = MethodConfig(), collect_diagnostics: bool = True) -> EmbeddingResult:
    validate_config(cfg)
    t0 = time.perf_counter()
    if host_img.ndim != 3 or host_img.shape[2] != 3:
        raise ValueError("Host must be a BGR color image")

    bs = cfg.block_size
    h, w = host_img.shape[:2]
    h0, w0 = (h // bs) * bs, (w // bs) * bs
    channel = host_img[:h0, :w0, 0].astype(np.float64)

    wm = watermark_binary
    if wm.shape != (cfg.wm_size, cfg.wm_size):
        wm = cv2.resize(wm, (cfg.wm_size, cfg.wm_size), interpolation=cv2.INTER_NEAREST)
    scrambled = arnold_transform(wm, cfg.arnold_iter)
    bits = (scrambled.reshape(-1) >= 128).astype(np.uint8)
    payload_bits = int(bits.size)
    repeat = int(cfg.repetition_override)
    required = payload_bits * repeat

    eligible = eligible_blocks(channel, cfg)
    if len(eligible) < required:
        raise ValueError(
            f"Insufficient determinant-valid blocks: {len(eligible)} available, {required} required "
            f"for payload={payload_bits}, repeat={repeat}."
        )
    selected, stability_scores, drift_ref = _stability_select_blocks(channel, eligible, required, cfg)
    ordered = chaotic_permute_positions(selected, cfg.private_key)

    out_channel = channel.copy()
    rows: List[int] = []
    cols: List[int] = []
    diagnostics: List[Dict[str, float]] = []

    for t, (rr, cc) in enumerate(ordered[:required]):
        bit_index = t % payload_bits
        bit = int(bits[bit_index])
        original = channel[rr:rr + bs, cc:cc + bs].copy()
        try:
            block_cfg = _adaptive_block_cfg(cfg, stability_scores.get((rr, cc), 0.0), drift_ref)
            float_block, carrier_delta = embed_qr_block(original, bit, block_cfg)
            stored = _stored_block(float_block)
            # The stored uint8 block, not the pre-rounding Q/R factors, is what
            # the decoder sees.  Repair only in that stored domain.
            if cfg.method in Q_METHODS:
                stored = _repair_q_integer_carrier(stored, bit, block_cfg)
            else:
                stored = _close_r_integer_block(stored, bit, block_cfg)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError):
            stored = original.astype(np.uint8)
            carrier_delta = 0.0

        out_channel[rr:rr + bs, cc:cc + bs] = stored
        rows.append(rr); cols.append(cc)

        if collect_diagnostics:
            diagnostics.append({
                "embedding_index": float(t),
                "payload_bit_index": float(bit_index),
                "bit": float(bit),
                "row": float(rr), "col": float(cc),
                "carrier_delta": float(carrier_delta),
                "pixel_l2": float(np.linalg.norm(stored.astype(np.float64) - original)),
                "pixel_max_abs": float(np.max(np.abs(stored.astype(np.float64) - original))),
                "stability_drift": float(stability_scores.get((rr, cc), 0.0)),
                "stability_drift_ref": float(drift_ref),
            })

    output = host_img.copy()
    output[:h0, :w0, 0] = np.clip(np.rint(out_channel), 0, 255).astype(np.uint8)
    side_info: Dict[str, np.ndarray] = {
        "rows": np.asarray(rows, dtype=np.int32),
        "cols": np.asarray(cols, dtype=np.int32),
        "payload_bits": np.asarray([payload_bits], dtype=np.int32),
        "repeat": np.asarray([repeat], dtype=np.int32),
        "block_size": np.asarray([bs], dtype=np.int32),
        "arnold_iter": np.asarray([cfg.arnold_iter], dtype=np.int32),
        "wm_size": np.asarray([cfg.wm_size], dtype=np.int32),
        "method": np.asarray([cfg.method]),
        "strength": np.asarray([strength_value(cfg)], dtype=np.float64),
        "block_selection": np.asarray([cfg.block_selection]),
    }
    return EmbeddingResult(output, side_info, diagnostics, time.perf_counter() - t0)


def _config_from_side_info(cfg: MethodConfig, side_info: Dict[str, np.ndarray]) -> MethodConfig:
    method = str(side_info.get("method", np.asarray([cfg.method]))[0])
    out = replace(cfg, method=method)
    if "strength" in side_info:
        out = with_strength(out, float(side_info["strength"][0]))
    return out


def _extract_from_channel(channel: np.ndarray, side_info: Dict[str, np.ndarray], cfg: MethodConfig) -> Tuple[np.ndarray, float, int]:
    cfg = _config_from_side_info(cfg, side_info)
    rows = side_info["rows"].astype(np.int32)
    cols = side_info["cols"].astype(np.int32)
    payload_bits = int(side_info.get("payload_bits", np.asarray([cfg.wm_size * cfg.wm_size]))[0])
    wm_size = int(side_info.get("wm_size", np.asarray([cfg.wm_size]))[0])
    bs = int(side_info.get("block_size", np.asarray([cfg.block_size]))[0])
    arnold_iter = int(side_info.get("arnold_iter", np.asarray([cfg.arnold_iter]))[0])

    sums = np.zeros(payload_bits, dtype=np.float64)
    abs_sums = np.zeros(payload_bits, dtype=np.float64)
    counts = np.zeros(payload_bits, dtype=np.int32)
    valid_obs = 0

    for t, (rr, cc) in enumerate(zip(rows, cols)):
        block = channel[int(rr):int(rr) + bs, int(cc):int(cc) + bs].astype(np.float64)
        if block.shape != (bs, bs):
            continue
        try:
            ev = float(extract_qr_evidence(block, cfg))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            continue
        if not np.isfinite(ev):
            continue
        if not cfg.soft_vote:
            ev = 1.0 if ev >= 0.0 else -1.0
        k = t % payload_bits
        sums[k] += ev
        abs_sums[k] += abs(ev)
        counts[k] += 1
        valid_obs += 1

    valid = counts > 0
    bits = np.zeros(payload_bits, dtype=np.uint8)
    bits[valid] = (sums[valid] >= 0.0).astype(np.uint8)
    bit_conf = np.zeros(payload_bits, dtype=np.float64)
    bit_conf[valid] = np.abs(sums[valid]) / (abs_sums[valid] + cfg.eps)
    confidence = float(np.mean(bit_conf[valid])) if np.any(valid) else 0.0
    wm_scrambled = (bits.reshape(wm_size, wm_size) * 255).astype(np.uint8)
    wm = invert_arnold_transform(wm_scrambled, arnold_iter)
    return wm, confidence, valid_obs



# ----- Optional carrier-conformity self-synchronisation --------------------

def _sync_evidence_score(ev: float, cfg: MethodConfig) -> float:
    if not np.isfinite(ev):
        return 0.0
    a = abs(float(ev))
    if cfg.method in {"q_smm", "q_npm", "q_lrm"}:
        g = max(_closure_guard_for_method(cfg), cfg.eps)
        return float(min(a / g, 1.0))
    # Periodic Q/R carriers are embedded near evidence +/-1.  A score based on
    # closeness to |e|=1 is far more selective than raw |e| on random blocks.
    return float(max(0.0, 1.0 - min(abs(a - 1.0), 1.0)))


def _sync_score_channel(
    channel: np.ndarray,
    side_info: Dict[str, np.ndarray],
    cfg: MethodConfig,
    dy: int = 0,
    dx: int = 0,
) -> float:
    rows = side_info["rows"].astype(np.int32)
    cols = side_info["cols"].astype(np.int32)
    bs = int(side_info.get("block_size", np.asarray([cfg.block_size]))[0])
    stride = max(1, int(cfg.sync_sample_stride))
    vals: List[float] = []
    for ii in range(0, len(rows), stride):
        rr, cc = int(rows[ii]) + int(dy), int(cols[ii]) + int(dx)
        if rr < 0 or cc < 0 or rr + bs > channel.shape[0] or cc + bs > channel.shape[1]:
            continue
        block = channel[rr:rr+bs, cc:cc+bs].astype(np.float64)
        try:
            if cfg.method in Q_METHODS:
                ev = _q_evidence_from_first_column(block[:, 0], cfg)
            else:
                ev = extract_qr_evidence(block, cfg)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            continue
        vals.append(_sync_evidence_score(float(ev), cfg))
    return float(np.mean(vals)) if vals else 0.0


def _warp_channel_rotation(channel: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = channel.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(channel, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _warp_channel_scale(channel: np.ndarray, scale: float) -> np.ndarray:
    h, w = channel.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, float(scale))
    return cv2.warpAffine(channel, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _extract_from_channel_offset(
    channel: np.ndarray,
    side_info: Dict[str, np.ndarray],
    cfg: MethodConfig,
    dy: int,
    dx: int,
) -> Tuple[np.ndarray, float, int]:
    # Same decoder as _extract_from_channel, but sample the known payload blocks
    # at one globally shifted coordinate grid.
    cfg = _config_from_side_info(cfg, side_info)
    rows = side_info["rows"].astype(np.int32)
    cols = side_info["cols"].astype(np.int32)
    payload_bits = int(side_info.get("payload_bits", np.asarray([cfg.wm_size * cfg.wm_size]))[0])
    wm_size = int(side_info.get("wm_size", np.asarray([cfg.wm_size]))[0])
    bs = int(side_info.get("block_size", np.asarray([cfg.block_size]))[0])
    arnold_iter = int(side_info.get("arnold_iter", np.asarray([cfg.arnold_iter]))[0])
    sums = np.zeros(payload_bits, dtype=np.float64)
    abs_sums = np.zeros(payload_bits, dtype=np.float64)
    counts = np.zeros(payload_bits, dtype=np.int32)
    valid_obs = 0
    for t, (rr0, cc0) in enumerate(zip(rows, cols)):
        rr, cc = int(rr0) + int(dy), int(cc0) + int(dx)
        if rr < 0 or cc < 0 or rr + bs > channel.shape[0] or cc + bs > channel.shape[1]:
            continue
        block = channel[rr:rr+bs, cc:cc+bs].astype(np.float64)
        try:
            ev = float(extract_qr_evidence(block, cfg))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            continue
        if not np.isfinite(ev):
            continue
        if not cfg.soft_vote:
            ev = 1.0 if ev >= 0.0 else -1.0
        k = t % payload_bits
        sums[k] += ev; abs_sums[k] += abs(ev); counts[k] += 1; valid_obs += 1
    valid = counts > 0
    bits = np.zeros(payload_bits, dtype=np.uint8)
    bits[valid] = (sums[valid] >= 0.0).astype(np.uint8)
    bit_conf = np.zeros(payload_bits, dtype=np.float64)
    bit_conf[valid] = np.abs(sums[valid]) / (abs_sums[valid] + cfg.eps)
    confidence = float(np.mean(bit_conf[valid])) if np.any(valid) else 0.0
    wm_scrambled = (bits.reshape(wm_size, wm_size) * 255).astype(np.uint8)
    return invert_arnold_transform(wm_scrambled, arnold_iter), confidence, valid_obs


def _auto_sync_channel(
    channel: np.ndarray,
    side_info: Dict[str, np.ndarray],
    cfg: MethodConfig,
) -> Tuple[np.ndarray, int, int, str, float, float]:
    base_score = _sync_score_channel(channel, side_info, cfg)
    best_score = base_score
    best_kind, best_a, best_b = "base", 0.0, 0.0
    best_channel = channel

    # Integer translation: first search a coarse grid, then refine by one pixel.
    rad = max(0, int(cfg.sync_translation_radius))
    step = max(1, int(cfg.sync_translation_step))
    coarse_best = (best_score, 0, 0)
    for dy in range(-rad, rad + 1, step):
        for dx in range(-rad, rad + 1, step):
            if dy == 0 and dx == 0:
                continue
            sc = _sync_score_channel(channel, side_info, cfg, dy, dx)
            if sc > coarse_best[0]:
                coarse_best = (sc, dy, dx)
    if coarse_best[0] > best_score:
        _, cy, cx = coarse_best
        for dy in range(max(-rad, cy - 1), min(rad, cy + 1) + 1):
            for dx in range(max(-rad, cx - 1), min(rad, cx + 1) + 1):
                sc = _sync_score_channel(channel, side_info, cfg, dy, dx)
                if sc > best_score:
                    best_score, best_kind, best_a, best_b = sc, "shift", float(dy), float(dx)

    # Small rotation correction.
    rmax = max(0.0, float(cfg.sync_rotation_deg))
    rstep = max(0.25, float(cfg.sync_rotation_step_deg))
    if rmax > 0:
        angles = np.arange(-rmax, rmax + 0.5*rstep, rstep)
        rot_best = (best_score, 0.0, channel)
        for a in angles:
            if abs(float(a)) < 1e-12:
                continue
            warped = _warp_channel_rotation(channel, float(a))
            sc = _sync_score_channel(warped, side_info, cfg)
            if sc > rot_best[0]:
                rot_best = (sc, float(a), warped)
        if rot_best[0] > best_score:
            best_score, best_kind, best_a, best_b, best_channel = rot_best[0], "rotation", rot_best[1], 0.0, rot_best[2]

    # Center-scale correction covers center-crop + resize attacks without pilots.
    smin = float(np.clip(cfg.sync_scale_min, 0.5, 1.0))
    sstep = max(0.005, float(cfg.sync_scale_step))
    scales = np.arange(smin, 1.0 + 0.5*sstep, sstep)
    scale_best = (best_score, 1.0, channel)
    for scale in scales:
        if abs(float(scale) - 1.0) < 1e-12:
            continue
        warped = _warp_channel_scale(channel, float(scale))
        sc = _sync_score_channel(warped, side_info, cfg)
        if sc > scale_best[0]:
            scale_best = (sc, float(scale), warped)
    if scale_best[0] > best_score:
        best_score, best_kind, best_a, best_b, best_channel = scale_best[0], "scale", scale_best[1], 0.0, scale_best[2]

    accept = best_score >= float(cfg.sync_accept_score) and best_score >= base_score * float(cfg.sync_accept_ratio)
    if not accept:
        return channel, 0, 0, "base", base_score, base_score
    if best_kind == "shift":
        return channel, int(round(best_a)), int(round(best_b)), "shift", base_score, best_score
    return best_channel, 0, 0, best_kind, base_score, best_score


def extract_array(watermarked: np.ndarray, side_info: Dict[str, np.ndarray], cfg: MethodConfig = MethodConfig()) -> ExtractionResult:
    t0 = time.perf_counter()
    cfg2 = _config_from_side_info(cfg, side_info)
    channel = watermarked[:, :, 0].astype(np.float64)
    sync_note = ""
    if cfg2.sync_mode == "auto":
        synced, dy, dx, kind, base_sc, best_sc = _auto_sync_channel(channel, side_info, cfg2)
        if kind == "shift":
            wm, conf, valid = _extract_from_channel_offset(synced, side_info, cfg2, dy, dx)
        else:
            wm, conf, valid = _extract_from_channel(synced, side_info, cfg2)
        if kind != "base":
            sync_note = f"+sync:{kind}"
    else:
        wm, conf, valid = _extract_from_channel(channel, side_info, cfg2)
    method = str(side_info.get("method", np.asarray([cfg2.method]))[0])
    return ExtractionResult(wm, conf, DISPLAY_NAMES.get(method, method) + sync_note, valid, time.perf_counter() - t0)


def save_side_info(path: str | os.PathLike[str], side_info: Dict[str, np.ndarray]) -> None:
    _ensure_parent(path)
    np.savez_compressed(str(path), **side_info)


def load_side_info(path: str | os.PathLike[str]) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def embed_file(host_path: str, watermark_path: str, output_path: str, side_info_path: str, cfg: MethodConfig = MethodConfig(), diagnostics_csv: Optional[str] = None) -> EmbeddingResult:
    host = _read_color(host_path)
    wm = load_binary_watermark(watermark_path, cfg.wm_size)
    result = embed_array(host, wm, cfg, collect_diagnostics=diagnostics_csv is not None)
    _ensure_parent(output_path)
    if not cv2.imwrite(output_path, result.watermarked):
        raise IOError(f"Failed to write watermarked image: {output_path}")
    save_side_info(side_info_path, result.side_info)
    if diagnostics_csv:
        write_diagnostics_csv(diagnostics_csv, result.diagnostics)
    return result


def extract_file(watermarked_path: str, side_info_path: str, output_path: str, cfg: MethodConfig = MethodConfig()) -> ExtractionResult:
    img = _read_color(watermarked_path)
    side = load_side_info(side_info_path)
    result = extract_array(img, side, cfg)
    _ensure_parent(output_path)
    if not cv2.imwrite(output_path, result.watermark):
        raise IOError(f"Failed to write extracted watermark: {output_path}")
    return result


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse <= 1e-15 else 10.0 * math.log10((255.0 ** 2) / mse)


def ssim_color(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(structural_similarity(a, b, data_range=255, channel_axis=-1))
    except TypeError:
        return float(structural_similarity(a, b, data_range=255, multichannel=True))


def nc_binary(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    x = (original_wm.reshape(-1) >= 128).astype(np.float64)
    y = (extracted_wm.reshape(-1) >= 128).astype(np.float64)
    num = float(np.dot(x, y))
    den = float(math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y))))
    if den <= 1e-15:
        return 1.0 if np.array_equal(x, y) else 0.0
    return num / den


def ber_binary(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    x = original_wm.reshape(-1) >= 128
    y = extracted_wm.reshape(-1) >= 128
    return float(np.mean(x != y))


def clean_metrics(host: np.ndarray, watermarked: np.ndarray, original_wm: np.ndarray, extracted_wm: np.ndarray) -> Dict[str, float]:
    return {
        "psnr": psnr(host, watermarked),
        "ssim": ssim_color(host, watermarked),
        "nc": nc_binary(original_wm, extracted_wm),
        "ber": ber_binary(original_wm, extracted_wm),
    }


def write_diagnostics_csv(path: str | os.PathLike[str], rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    _ensure_parent(path)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def save_config_json(path: str | os.PathLike[str], cfg: MethodConfig) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def _cli_cfg(args: argparse.Namespace) -> MethodConfig:
    cfg = MethodConfig(
        method=args.method, private_key=args.key, repetition_override=args.repeat,
        block_selection=getattr(args, "selection", "stability"),
        sync_mode=getattr(args, "sync", "none"),
    )
    if args.strength is not None:
        cfg = with_strength(cfg, args.strength)
    return cfg


def _cli_embed(args: argparse.Namespace) -> None:
    cfg = _cli_cfg(args)
    result = embed_file(args.host, args.watermark, args.output, args.side_info, cfg, args.diagnostics)
    print(json.dumps({
        "method": DISPLAY_NAMES[cfg.method], "strength": strength_value(cfg),
        "embed_seconds": result.elapsed_seconds,
        "selected": int(result.side_info["rows"].size),
        "repeat": int(result.side_info["repeat"][0]),
    }, indent=2))


def _cli_extract(args: argparse.Namespace) -> None:
    cfg = _cli_cfg(args)
    result = extract_file(args.watermarked, args.side_info, args.output, cfg)
    print(json.dumps({
        "extract_seconds": result.elapsed_seconds,
        "confidence": result.confidence,
        "method": result.candidate_name,
        "valid_observations": result.valid_observations,
    }, indent=2))


def _cli_clean_benchmark(args: argparse.Namespace) -> None:
    cfg = _cli_cfg(args)
    host = _read_color(args.host)
    wm = load_binary_watermark(args.watermark, cfg.wm_size)
    emb = embed_array(host, wm, cfg, collect_diagnostics=False)
    ext = extract_array(emb.watermarked, emb.side_info, cfg)
    m = clean_metrics(host, emb.watermarked, wm, ext.watermark)
    m.update({
        "method": DISPLAY_NAMES[cfg.method], "strength": strength_value(cfg),
        "embed_seconds": emb.elapsed_seconds, "extract_seconds": ext.elapsed_seconds,
        "confidence": ext.confidence, "repeat": int(emb.side_info["repeat"][0]),
    })
    print(json.dumps(m, indent=2))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ten independent Q-only/R-only QR watermark candidates")
    sub = p.add_subparsers(dest="command", required=True)
    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--method", choices=METHODS, default="q_gaqim")
        sp.add_argument("--strength", type=float)
        sp.add_argument("--repeat", type=int, default=1)
        sp.add_argument("--key", default="KB123")
        sp.add_argument("--selection", choices=("stability", "chaotic"), default="stability")
        sp.add_argument("--sync", choices=("none", "auto"), default="none")
    e = sub.add_parser("embed"); common(e)
    e.add_argument("--host", required=True); e.add_argument("--watermark", required=True)
    e.add_argument("--output", required=True); e.add_argument("--side-info", required=True)
    e.add_argument("--diagnostics"); e.set_defaults(func=_cli_embed)
    x = sub.add_parser("extract"); common(x)
    x.add_argument("--watermarked", required=True); x.add_argument("--side-info", required=True)
    x.add_argument("--output", required=True); x.set_defaults(func=_cli_extract)
    b = sub.add_parser("clean-benchmark"); common(b)
    b.add_argument("--host", required=True); b.add_argument("--watermark", required=True)
    b.set_defaults(func=_cli_clean_benchmark)
    return p


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
