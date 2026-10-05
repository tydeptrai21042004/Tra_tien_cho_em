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
    q_angle_step_deg: float = 1.5
    q_margin: float = 0.01
    q_log_margin: float = 0.02
    q_lattice_step: float = 0.02

    # R-family strengths
    r_step: float = 8.0
    r_norm_step: float = 0.04
    r_log_step: float = 0.06
    r_alpha: float = 0.035
    r_s_min: float = 2.0
    r_s_max: float = 16.0
    r_projection: str = "balanced"  # balanced | average
    aqim_rounds: int = 3

    # Extraction
    soft_vote: bool = True


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


def method_strength_grid(method: str) -> List[float]:
    """Recommended first-pass search grid for each candidate."""
    if method == "q_gaqim":
        return [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    if method in {"q_npm", "q_smm"}:
        return [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    if method == "q_lrm":
        return [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    if method == "q_lqim":
        return [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]
    if method in {"r_spqim", "r_rqim"}:
        return [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    if method == "r_ndqim":
        return [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16]
    if method == "r_lrqim":
        return [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]
    if method == "r_aqim":
        return [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
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
    energy = float(np.linalg.norm(r[0, 0:4]))
    return float(np.clip(cfg.r_alpha * energy, cfg.r_s_min, cfg.r_s_max))


def _r_aqim_embed(q: np.ndarray, r: np.ndarray, bit: int, cfg: MethodConfig) -> Tuple[np.ndarray, float]:
    r2 = r.copy()
    before = r2[0, 0:4].copy()
    # Fixed-point rounds make the received-row-derived S closer to the embed S.
    for _ in range(max(1, int(cfg.aqim_rounds))):
        step = _adaptive_r_step(r2, cfg)
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
    ordered = chaotic_permute_positions(eligible, cfg.private_key)

    out_channel = channel.copy()
    rows: List[int] = []
    cols: List[int] = []
    diagnostics: List[Dict[str, float]] = []

    for t, (rr, cc) in enumerate(ordered[:required]):
        bit_index = t % payload_bits
        bit = int(bits[bit_index])
        original = channel[rr:rr + bs, cc:cc + bs].copy()
        try:
            float_block, carrier_delta = embed_qr_block(original, bit, cfg)
            stored = _stored_block(float_block)
            if not determinant_valid(stored, cfg.det_eps):
                # preserve the deterministic coordinate list: keep original if the
                # quantized candidate becomes singular, but record zero evidence.
                stored = original.astype(np.uint8)
                carrier_delta = 0.0
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
        if block.shape != (bs, bs) or not determinant_valid(block, cfg.det_eps):
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


def extract_array(watermarked: np.ndarray, side_info: Dict[str, np.ndarray], cfg: MethodConfig = MethodConfig()) -> ExtractionResult:
    t0 = time.perf_counter()
    channel = watermarked[:, :, 0].astype(np.float64)
    wm, conf, valid = _extract_from_channel(channel, side_info, cfg)
    method = str(side_info.get("method", np.asarray([cfg.method]))[0])
    return ExtractionResult(wm, conf, DISPLAY_NAMES.get(method, method), valid, time.perf_counter() - t0)


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
    cfg = MethodConfig(method=args.method, private_key=args.key, repetition_override=args.repeat)
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
