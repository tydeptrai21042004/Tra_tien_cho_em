#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-aligned implementation of the proposed QR watermarking method.

This file replaces the notebook implementation. It follows the manuscript's
4x4 determinant-gated semi-blind method:
  * blue-channel, non-overlapping 4x4 blocks;
  * determinant eligibility before QR decomposition;
  * canonical QR with non-negative diagonal of R;
  * Q branch based on |q44| around tau_Q;
  * R branch based on r44 modulo q with periodic sine evidence;
  * stored selected block indices and Q/R branch flags;
  * local robustness/distortion branch score;
  * repeated soft-decision recovery;
  * raw/mild-NLM/strong-NLM confidence-guided extraction.

The implementation also exposes diagnostics used by the ablation and branch-
position analysis scripts.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as structural_similarity


FLAG_Q = 0
FLAG_R = 1
FLAG_NAMES = {FLAG_Q: "Q", FLAG_R: "R"}


@dataclass(frozen=True)
class MethodConfig:
    block_size: int = 4
    wm_size: int = 64
    q_step: float = 8.0
    q_tau: float = 0.5
    q_margin: float = 0.026
    r_margin: float = 1.95
    det_eps: float = 1e-8
    perturb_amp: float = 0.35
    arnold_iter: int = 10
    private_key: str = "KB123"
    score_eps: float = 1e-9
    confidence_eps: float = 1e-12
    nlm_h_mild: float = 3.0
    nlm_h_strong: float = 7.0
    nlm_template_window: int = 7
    nlm_search_window: int = 21
    branch_policy: str = "adaptive"  # adaptive | q_only | r_only | distortion_only | robustness_only | raw_score
    repetition_override: Optional[int] = None
    use_nlm_candidates: bool = True
    decoder_selection: str = "confidence"  # confidence | raw | mild | strong
    soft_vote: bool = True


@dataclass
class CandidateDiagnostics:
    mode: int
    valid: bool
    distortion: float
    raw_margin: float
    mean_margin: float
    worst_margin: float
    robustness: float
    score: float
    reliable: bool
    stored_block: Optional[np.ndarray] = None


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
    """Load a watermark and return an exactly size x size uint8 binary image."""
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
        raise ValueError("Arnold transform requires a square watermark.")
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
        raise ValueError("Arnold transform requires a square watermark.")
    out = image.copy()
    for _ in range(int(iterations)):
        nxt = np.empty_like(out)
        for x in range(n):
            for y in range(n):
                nxt[(2 * x - y) % n, (-x + y) % n] = out[x, y]
        out = nxt
    return out


def highest_odd_repeat(total_blocks: int, payload_bits: int) -> int:
    if payload_bits <= 0:
        raise ValueError("payload_bits must be positive")
    full = total_blocks // payload_bits
    if full < 1:
        raise ValueError(f"Capacity insufficient: {total_blocks} blocks for {payload_bits} bits")
    r = full if full % 2 == 1 else full - 1
    return max(1, r)


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
    bs = cfg.block_size
    out: List[Tuple[int, int]] = []
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


def q_evidence(q_mat: np.ndarray, cfg: MethodConfig) -> float:
    return (abs(float(q_mat[3, 3])) - cfg.q_tau) / 0.5


def r_evidence(r_mat: np.ndarray, cfg: MethodConfig) -> float:
    z = float(r_mat[3, 3]) % cfg.q_step
    return -math.sin(2.0 * math.pi * z / cfg.q_step)


def signed_margin(block: np.ndarray, mode: int, bit: int, cfg: MethodConfig) -> float:
    if not determinant_valid(block, cfg.det_eps):
        return -1.0
    q_mat, r_mat = canonical_qr(block)
    llr = q_evidence(q_mat, cfg) if mode == FLAG_Q else r_evidence(r_mat, cfg)
    return float(llr if bit == 1 else -llr)


def _givens_matrix(n: int, k: int, l: int, theta: float) -> np.ndarray:
    g = np.eye(n, dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    # With right multiplication: new col_l = s * old col_k + c * old col_l.
    g[k, k] = c
    g[k, l] = s
    g[l, k] = -s
    g[l, l] = c
    return g


def _q_rotation_candidates(q_mat: np.ndarray, r_mat: np.ndarray, target_abs: float) -> List[np.ndarray]:
    """All low-dimensional Givens realizations satisfying |q44| = target_abs."""
    target_abs = float(np.clip(abs(target_abs), 0.0, 0.999999999))
    out: List[np.ndarray] = []
    row = 3
    col4 = 3
    for k in (0, 1, 2):
        a = float(q_mat[row, k])
        b = float(q_mat[row, col4])
        radius = math.hypot(a, b)
        if radius <= 1e-15 or target_abs > radius + 1e-12:
            continue
        phi = math.atan2(a, b)  # a*sin(theta)+b*cos(theta)=radius*cos(theta-phi)
        for target_sign in (-1.0, 1.0):
            y = target_sign * target_abs
            ratio = float(np.clip(y / radius, -1.0, 1.0))
            alpha = math.acos(ratio)
            for theta in (phi + alpha, phi - alpha):
                g = _givens_matrix(4, k, col4, theta)
                q_new = q_mat @ g
                # Numerical guard for the manuscript constraint.
                if abs(abs(float(q_new[3, 3])) - target_abs) <= 1e-7:
                    out.append(q_new @ r_mat)
    return out


def build_q_candidate(block: np.ndarray, bit: int, cfg: MethodConfig) -> np.ndarray:
    q_mat, r_mat = canonical_qr(block)
    q44 = abs(float(q_mat[3, 3]))
    target0 = cfg.q_tau - 0.5 * cfg.q_margin
    target1 = cfg.q_tau + 0.5 * cfg.q_margin

    already_safe = (bit == 0 and q44 <= target0) or (bit == 1 and q44 >= target1)
    if already_safe:
        return block.astype(np.float64).copy()

    target = target1 if bit == 1 else target0
    realizations = _q_rotation_candidates(q_mat, r_mat, target)
    if not realizations:
        raise RuntimeError("No feasible 4x4 Givens realization for Q branch")

    # Paper: select a low-distortion realization satisfying |q44| = c_b.
    return min(realizations, key=lambda x: float(np.mean((_stored_candidate(x) - block) ** 2)))


def _nearest_positive_qim_center(value: float, target_phase: float, period: float, eps: float) -> float:
    k0 = int(round((value - target_phase) / period))
    candidates = []
    for k in range(k0 - 3, k0 + 4):
        rho = target_phase + k * period
        if rho > eps:
            candidates.append(rho)
    if not candidates:
        k = int(math.floor((eps - target_phase) / period)) + 1
        candidates = [target_phase + k * period]
    return float(min(candidates, key=lambda x: abs(x - value)))


def build_r_candidate(block: np.ndarray, bit: int, cfg: MethodConfig) -> np.ndarray:
    q_mat, r_mat = canonical_qr(block)
    r_new = r_mat.copy()
    r44 = float(r_new[3, 3])
    z = r44 % cfg.q_step

    if bit == 1:
        safe = (0.5 * cfg.q_step + cfg.r_margin) <= z <= (cfg.q_step - cfg.r_margin)
        target = 0.75 * cfg.q_step
    else:
        safe = cfg.r_margin <= z <= (0.5 * cfg.q_step - cfg.r_margin)
        target = 0.25 * cfg.q_step

    if not safe:
        r_new[3, 3] = _nearest_positive_qim_center(r44, target, cfg.q_step, cfg.det_eps)
    return q_mat @ r_new


def _stored_candidate(candidate: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(candidate), 0, 255).astype(np.float64)


def _perturbation_set(stored: np.ndarray, amplitude: float) -> List[np.ndarray]:
    """Deterministic mild local perturbations around the stored pixel block.

    The paper fixes the local perturbation amplitude at 0.35.  We use symmetric
    DC and checkerboard perturbations so the score probes both brightness and
    local high-frequency sensitivity without introducing an attack classifier.
    """
    a = float(amplitude)
    checker = np.fromfunction(lambda i, j: np.where((i + j) % 2 == 0, 1.0, -1.0), stored.shape)
    variants = [
        stored,
        np.clip(stored + a, 0, 255),
        np.clip(stored - a, 0, 255),
        np.clip(stored + a * checker, 0, 255),
    ]
    return variants


def evaluate_candidate(
    original_block: np.ndarray,
    floating_candidate: np.ndarray,
    mode: int,
    bit: int,
    cfg: MethodConfig,
    use_perturbations: bool = True,
) -> CandidateDiagnostics:
    stored = _stored_candidate(floating_candidate)
    if not determinant_valid(stored, cfg.det_eps):
        return CandidateDiagnostics(mode, False, math.inf, -1.0, -1.0, -1.0, -1.0, -math.inf, False, None)

    distortion = float(np.mean((original_block.astype(np.float64) - stored) ** 2))
    variants = _perturbation_set(stored, cfg.perturb_amp) if use_perturbations else [stored]
    margins = [signed_margin(v, mode, bit, cfg) for v in variants if determinant_valid(v, cfg.det_eps)]
    if not margins:
        return CandidateDiagnostics(mode, False, distortion, -1.0, -1.0, -1.0, -1.0, -math.inf, False, None)

    raw = float(margins[0])
    mean = float(np.mean(margins))
    worst = float(np.min(margins))
    robustness = 0.5 * raw + 0.3 * mean + 0.2 * worst
    score = robustness / (distortion + cfg.score_eps)
    reliable = raw >= 0.0 and mean >= 0.0
    return CandidateDiagnostics(mode, True, distortion, raw, mean, worst, robustness, score, reliable, stored)


def choose_branch(q_diag: CandidateDiagnostics, r_diag: CandidateDiagnostics, policy: str) -> CandidateDiagnostics:
    valid = [d for d in (q_diag, r_diag) if d.valid]
    if not valid:
        raise RuntimeError("Both Q and R candidates became determinant-invalid after storage quantization")

    if policy == "q_only":
        if not q_diag.valid:
            raise RuntimeError("Q-only ablation has no valid Q candidate")
        return q_diag
    if policy == "r_only":
        if not r_diag.valid:
            raise RuntimeError("R-only ablation has no valid R candidate")
        return r_diag
    if policy == "distortion_only":
        return min(valid, key=lambda d: (d.distortion, -d.score))
    if policy == "robustness_only":
        reliable = [d for d in valid if d.reliable]
        pool = reliable if reliable else valid
        return max(pool, key=lambda d: (d.robustness, -d.distortion))
    if policy not in ("adaptive", "raw_score"):
        raise ValueError(f"Unknown branch policy: {policy}")

    reliable = [d for d in valid if d.reliable]
    pool = reliable if reliable else valid
    return max(pool, key=lambda d: (d.score, -d.distortion))


def _local_texture_features(block: np.ndarray) -> Tuple[float, float, float]:
    b = block.astype(np.float64)
    variance = float(np.var(b))
    gx = np.diff(b, axis=1)
    gy = np.diff(b, axis=0)
    gradient = float((np.mean(np.abs(gx)) if gx.size else 0.0) + (np.mean(np.abs(gy)) if gy.size else 0.0))
    dynamic_range = float(np.max(b) - np.min(b))
    return variance, gradient, dynamic_range


def embed_array(host_img: np.ndarray, watermark_binary: np.ndarray, cfg: MethodConfig = MethodConfig(), collect_diagnostics: bool = True) -> EmbeddingResult:
    t0 = time.perf_counter()
    if host_img.ndim != 3 or host_img.shape[2] != 3:
        raise ValueError("Host must be a BGR color image")
    bs = cfg.block_size
    if bs != 4:
        raise ValueError("Paper-aligned implementation requires block_size=4")

    h, w = host_img.shape[:2]
    h0, w0 = (h // bs) * bs, (w // bs) * bs
    channel = host_img[:h0, :w0, 0].astype(np.float64)

    wm = watermark_binary
    if wm.shape != (cfg.wm_size, cfg.wm_size):
        wm = cv2.resize(wm, (cfg.wm_size, cfg.wm_size), interpolation=cv2.INTER_NEAREST)
    wm_scrambled = arnold_transform(wm, cfg.arnold_iter)
    bits = (wm_scrambled.reshape(-1) >= 128).astype(np.uint8)
    payload_bits = int(bits.size)

    nominal_positions = enumerate_blocks(channel.shape, bs)
    nominal_blocks = len(nominal_positions)
    repeat = highest_odd_repeat(nominal_blocks, payload_bits)
    if cfg.repetition_override is not None:
        repeat = int(cfg.repetition_override)
        if repeat < 1 or repeat % 2 != 1:
            raise ValueError("repetition_override must be a positive odd integer")
        if repeat * payload_bits > nominal_blocks:
            raise ValueError("repetition_override exceeds nominal block capacity")
    required = payload_bits * repeat

    eligible = eligible_blocks(channel, cfg)
    if len(eligible) < required:
        raise ValueError(
            f"Host is determinant-ineligible: {len(eligible)} eligible 4x4 blocks, "
            f"but {required} are required for payload={payload_bits}, r={repeat}."
        )
    ordered = chaotic_permute_positions(eligible, cfg.private_key)

    watermarked_channel = channel.copy()
    rows: List[int] = []
    cols: List[int] = []
    flags: List[int] = []
    diagnostics: List[Dict[str, float]] = []

    for r, c in ordered:
        if len(flags) >= required:
            break
        bit_index = len(flags) % payload_bits
        bit = int(bits[bit_index])
        original = channel[r:r + bs, c:c + bs].copy()
        if not determinant_valid(original, cfg.det_eps):
            continue

        try:
            q_float = build_q_candidate(original, bit, cfg)
            r_float = build_r_candidate(original, bit, cfg)
            use_perturb = cfg.branch_policy != "raw_score"
            q_diag = evaluate_candidate(original, q_float, FLAG_Q, bit, cfg, use_perturbations=use_perturb)
            r_diag = evaluate_candidate(original, r_float, FLAG_R, bit, cfg, use_perturbations=use_perturb)
            chosen = choose_branch(q_diag, r_diag, cfg.branch_policy)
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            continue

        if chosen.stored_block is None or not determinant_valid(chosen.stored_block, cfg.det_eps):
            continue

        watermarked_channel[r:r + bs, c:c + bs] = chosen.stored_block
        rows.append(r)
        cols.append(c)
        flags.append(chosen.mode)

        if collect_diagnostics:
            variance, gradient, dynamic_range = _local_texture_features(original)
            diagnostics.append({
                "embedding_index": float(len(flags) - 1),
                "payload_bit_index": float(bit_index),
                "bit": float(bit),
                "row": float(r),
                "col": float(c),
                "row_norm": float(r / max(1, h0 - bs)),
                "col_norm": float(c / max(1, w0 - bs)),
                "branch": float(chosen.mode),
                "q_valid": float(q_diag.valid),
                "r_valid": float(r_diag.valid),
                "q_reliable": float(q_diag.reliable),
                "r_reliable": float(r_diag.reliable),
                "q_distortion": float(q_diag.distortion),
                "r_distortion": float(r_diag.distortion),
                "q_robustness": float(q_diag.robustness),
                "r_robustness": float(r_diag.robustness),
                "q_score": float(q_diag.score),
                "r_score": float(r_diag.score),
                "chosen_distortion": float(chosen.distortion),
                "chosen_robustness": float(chosen.robustness),
                "local_variance": variance,
                "local_gradient": gradient,
                "local_dynamic_range": dynamic_range,
            })

    if len(flags) != required:
        raise RuntimeError(f"Only {len(flags)} usable embeddings were produced; {required} required")

    output = host_img.copy()
    output[:h0, :w0, 0] = np.clip(np.rint(watermarked_channel), 0, 255).astype(np.uint8)
    side_info = {
        "rows": np.asarray(rows, dtype=np.int32),
        "cols": np.asarray(cols, dtype=np.int32),
        "flags": np.asarray(flags, dtype=np.uint8),
        "payload_bits": np.asarray([payload_bits], dtype=np.int32),
        "repeat": np.asarray([repeat], dtype=np.int32),
        "block_size": np.asarray([bs], dtype=np.int32),
        "arnold_iter": np.asarray([cfg.arnold_iter], dtype=np.int32),
        "wm_size": np.asarray([cfg.wm_size], dtype=np.int32),
    }
    return EmbeddingResult(output, side_info, diagnostics, time.perf_counter() - t0)


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
    if diagnostics_csv is not None:
        write_diagnostics_csv(diagnostics_csv, result.diagnostics)
    return result


def _extract_from_channel(channel: np.ndarray, side_info: Dict[str, np.ndarray], cfg: MethodConfig) -> Tuple[np.ndarray, float, int]:
    rows = side_info["rows"].astype(np.int32)
    cols = side_info["cols"].astype(np.int32)
    flags = side_info["flags"].astype(np.uint8)
    payload_bits = int(side_info.get("payload_bits", np.asarray([cfg.wm_size * cfg.wm_size]))[0])
    wm_size = int(side_info.get("wm_size", np.asarray([cfg.wm_size]))[0])
    bs = int(side_info.get("block_size", np.asarray([cfg.block_size]))[0])
    arnold_iter = int(side_info.get("arnold_iter", np.asarray([cfg.arnold_iter]))[0])

    sums = np.zeros(payload_bits, dtype=np.float64)
    abs_sums = np.zeros(payload_bits, dtype=np.float64)
    counts = np.zeros(payload_bits, dtype=np.int32)
    valid_observations = 0

    for t, (r, c, flag) in enumerate(zip(rows, cols, flags)):
        block = channel[int(r):int(r) + bs, int(c):int(c) + bs].astype(np.float64)
        # Paper: do not reconstruct eligibility; visit stored index and ignore an
        # attacked observation only if its current determinant is invalid.
        if block.shape != (bs, bs) or not determinant_valid(block, cfg.det_eps):
            continue
        try:
            q_mat, r_mat = canonical_qr(block)
        except np.linalg.LinAlgError:
            continue
        if int(flag) == FLAG_Q:
            lam = q_evidence(q_mat, cfg)
        elif int(flag) == FLAG_R:
            lam = r_evidence(r_mat, cfg)
        else:
            continue
        if not cfg.soft_vote:
            lam = 1.0 if lam >= 0.0 else -1.0
        k = t % payload_bits
        sums[k] += lam
        abs_sums[k] += abs(lam)
        counts[k] += 1
        valid_observations += 1

    valid = counts > 0
    bits = np.zeros(payload_bits, dtype=np.uint8)
    bits[valid] = (sums[valid] >= 0.0).astype(np.uint8)
    bit_conf = np.zeros(payload_bits, dtype=np.float64)
    bit_conf[valid] = np.abs(sums[valid]) / (abs_sums[valid] + cfg.confidence_eps)
    confidence = float(np.mean(bit_conf[valid])) if np.any(valid) else 0.0
    scrambled = (bits.reshape(wm_size, wm_size) * 255).astype(np.uint8)
    return invert_arnold_transform(scrambled, arnold_iter), confidence, valid_observations


def extract_array(watermarked: np.ndarray, side_info: Dict[str, np.ndarray], cfg: MethodConfig = MethodConfig()) -> ExtractionResult:
    t0 = time.perf_counter()
    channel = watermarked[:, :, 0].astype(np.uint8)
    candidates: List[Tuple[str, np.ndarray, float, int]] = []

    wm, conf, valid = _extract_from_channel(channel.astype(np.float64), side_info, cfg)
    candidates.append(("raw", wm, conf, valid))

    if cfg.use_nlm_candidates:
        mild = cv2.fastNlMeansDenoising(
            channel, None, h=float(cfg.nlm_h_mild),
            templateWindowSize=int(cfg.nlm_template_window),
            searchWindowSize=int(cfg.nlm_search_window),
        )
        wm_m, conf_m, valid_m = _extract_from_channel(mild.astype(np.float64), side_info, cfg)
        candidates.append(("nlm_mild", wm_m, conf_m, valid_m))

        strong = cv2.fastNlMeansDenoising(
            channel, None, h=float(cfg.nlm_h_strong),
            templateWindowSize=int(cfg.nlm_template_window),
            searchWindowSize=int(cfg.nlm_search_window),
        )
        wm_s, conf_s, valid_s = _extract_from_channel(strong.astype(np.float64), side_info, cfg)
        candidates.append(("nlm_strong", wm_s, conf_s, valid_s))

    # Paper default: choose recovered watermark with the largest global confidence.
    selection = cfg.decoder_selection.lower().strip()
    if selection == "confidence":
        name, best_wm, best_conf, best_valid = max(candidates, key=lambda x: x[2])
    else:
        wanted = {"raw": "raw", "mild": "nlm_mild", "strong": "nlm_strong"}.get(selection)
        if wanted is None:
            raise ValueError(f"Unknown decoder_selection: {cfg.decoder_selection}")
        match = [x for x in candidates if x[0] == wanted]
        if not match:
            raise ValueError(f"decoder_selection={selection} requires the corresponding NLM candidate")
        name, best_wm, best_conf, best_valid = match[0]
    return ExtractionResult(best_wm, float(best_conf), name, int(best_valid), time.perf_counter() - t0)


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
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_config_json(path: str | os.PathLike[str], cfg: MethodConfig) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def _cli_embed(args: argparse.Namespace) -> None:
    cfg = MethodConfig(private_key=args.key)
    result = embed_file(args.host, args.watermark, args.output, args.side_info, cfg, args.diagnostics)
    print(json.dumps({
        "embed_seconds": result.elapsed_seconds,
        "Q_count": int(np.sum(result.side_info["flags"] == FLAG_Q)),
        "R_count": int(np.sum(result.side_info["flags"] == FLAG_R)),
        "selected": int(result.side_info["flags"].size),
        "repeat": int(result.side_info["repeat"][0]),
    }, indent=2))


def _cli_extract(args: argparse.Namespace) -> None:
    cfg = MethodConfig(private_key=args.key)
    result = extract_file(args.watermarked, args.side_info, args.output, cfg)
    print(json.dumps({
        "extract_seconds": result.elapsed_seconds,
        "confidence": result.confidence,
        "selected_candidate": result.candidate_name,
        "valid_observations": result.valid_observations,
    }, indent=2))


def _cli_clean_benchmark(args: argparse.Namespace) -> None:
    cfg = MethodConfig(private_key=args.key)
    host = _read_color(args.host)
    wm = load_binary_watermark(args.watermark, cfg.wm_size)
    emb = embed_array(host, wm, cfg, collect_diagnostics=True)
    ext = extract_array(emb.watermarked, emb.side_info, cfg)
    m = clean_metrics(host, emb.watermarked, wm, ext.watermark)
    m.update({
        "embed_seconds": emb.elapsed_seconds,
        "extract_seconds": ext.elapsed_seconds,
        "confidence": ext.confidence,
        "extract_candidate": ext.candidate_name,
        "Q_count": int(np.sum(emb.side_info["flags"] == FLAG_Q)),
        "R_count": int(np.sum(emb.side_info["flags"] == FLAG_R)),
        "repeat": int(emb.side_info["repeat"][0]),
    })
    print(json.dumps(m, indent=2))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-aligned 4x4 determinant-gated adaptive QR watermarking")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--host", required=True)
    e.add_argument("--watermark", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--side-info", required=True)
    e.add_argument("--diagnostics")
    e.add_argument("--key", default="KB123")
    e.set_defaults(func=_cli_embed)

    x = sub.add_parser("extract")
    x.add_argument("--watermarked", required=True)
    x.add_argument("--side-info", required=True)
    x.add_argument("--output", required=True)
    x.add_argument("--key", default="KB123")
    x.set_defaults(func=_cli_extract)

    b = sub.add_parser("clean-benchmark")
    b.add_argument("--host", required=True)
    b.add_argument("--watermark", required=True)
    b.add_argument("--key", default="KB123")
    b.set_defaults(func=_cli_clean_benchmark)
    return p


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
