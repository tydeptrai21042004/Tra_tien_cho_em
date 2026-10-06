#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from proposal_method import EmbeddingResult, ExtractionResult, arnold_transform, invert_arnold_transform

METHODS = (
    "dwt_schur_jmqim",
    "swt_schur_jmqim",
    "dwt_svd_ngqim",
    "dwt_normdiff_spread_qim",
)

DISPLAY_NAMES = {
    "dwt_schur_jmqim": "DWT-Schur-Spectral-JMQIM",
    "swt_schur_jmqim": "SWT-Schur-Spectral-JMQIM",
    "dwt_svd_ngqim": "DWT-SVD-NGQIM",
    "dwt_normdiff_spread_qim": "DWT-Normalized-Difference Spread-QIM",
}


@dataclass(frozen=True)
class TransformMethodConfig:
    method: str = "dwt_schur_jmqim"
    wm_size: int = 64
    private_key: str = "KB123"
    arnold_iter: int = 10
    step: float = 0.025
    eps: float = 1e-9
    weights: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    closure_rounds: int = 5
    local_step_growth: float = 1.5
    local_step_max_factor: float = 6.0
    spread_repetitions: int = 4
    pilot_count: int = 63
    pilot_step: float = 22.0
    sync_mode: str = "pilot"  # none | pilot
    sync_translation_radius: int = 8
    sync_translation_step: int = 2
    sync_rotation_deg: float = 5.0
    sync_rotation_step_deg: float = 1.0
    sync_scale_min: float = 0.84
    sync_scale_max: float = 1.08
    sync_scale_step: float = 0.02
    sync_accept_score: float = 0.57
    sync_accept_gain: float = 0.04


def validate_config(cfg: TransformMethodConfig) -> None:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method={cfg.method!r}. Choose from {METHODS}")
    if cfg.wm_size != 64:
        raise ValueError("This candidate patch currently targets a 64x64 payload")
    if cfg.step <= 0:
        raise ValueError("step must be positive")
    if cfg.sync_mode not in {"none", "pilot"}:
        raise ValueError("sync_mode must be 'none' or 'pilot'")


def method_strength_grid(method: str) -> List[float]:
    if method == "dwt_schur_jmqim":
        return [0.012, 0.016, 0.020, 0.025, 0.030, 0.040, 0.055]
    if method == "swt_schur_jmqim":
        return [0.012, 0.016, 0.020, 0.025, 0.030, 0.040, 0.055]
    if method == "dwt_svd_ngqim":
        return [0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040]
    if method == "dwt_normdiff_spread_qim":
        return [0.015, 0.020, 0.025, 0.030, 0.035, 0.050, 0.080, 0.120, 0.180]
    raise ValueError(method)


def with_strength(cfg: TransformMethodConfig, strength: float) -> TransformMethodConfig:
    return replace(cfg, step=float(strength))


def strength_value(cfg: TransformMethodConfig) -> float:
    return float(cfg.step)


def _rng(key: str, tag: str) -> np.random.Generator:
    d = hashlib.sha256((key + "|" + tag).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "big"))


def _qim_target(x: float, bit: int, step: float) -> float:
    phase = (0.75 if int(bit) else 0.25) * step
    return float(round((float(x) - phase) / step) * step + phase)


def _qim_evidence(x: float, step: float, eps: float = 1e-12) -> float:
    t0 = _qim_target(float(x), 0, step)
    t1 = _qim_target(float(x), 1, step)
    d0, d1 = abs(float(x) - t0), abs(float(x) - t1)
    return float(np.clip((d0 - d1) / max(0.5 * step, eps), -1.0, 1.0))


def _haar_dwt2(x: np.ndarray):
    a = np.asarray(x, dtype=np.float64)
    if a.shape[0] % 2 or a.shape[1] % 2:
        raise ValueError("Haar DWT requires even dimensions")
    s = math.sqrt(2.0)
    lo_r = (a[:, 0::2] + a[:, 1::2]) / s
    hi_r = (a[:, 0::2] - a[:, 1::2]) / s
    ll = (lo_r[0::2] + lo_r[1::2]) / s
    lh = (lo_r[0::2] - lo_r[1::2]) / s
    hl = (hi_r[0::2] + hi_r[1::2]) / s
    hh = (hi_r[0::2] - hi_r[1::2]) / s
    return ll, (lh, hl, hh)


def _haar_idwt2(coeffs) -> np.ndarray:
    ll, (lh, hl, hh) = coeffs
    s = math.sqrt(2.0)
    lo0, lo1 = (ll + lh) / s, (ll - lh) / s
    hi0, hi1 = (hl + hh) / s, (hl - hh) / s
    lo = np.empty((ll.shape[0] * 2, ll.shape[1]), dtype=np.float64)
    hi = np.empty_like(lo)
    lo[0::2], lo[1::2] = lo0, lo1
    hi[0::2], hi[1::2] = hi0, hi1
    out = np.empty((lo.shape[0], lo.shape[1] * 2), dtype=np.float64)
    out[:, 0::2] = (lo + hi) / s
    out[:, 1::2] = (lo - hi) / s
    return out


def _haar_swt2(x: np.ndarray):
    a = np.asarray(x, dtype=np.float64)
    s = math.sqrt(2.0)
    xr = np.roll(a, 1, axis=1)
    lo_r, hi_r = (a + xr) / s, (a - xr) / s
    lo_u, hi_u = np.roll(lo_r, 1, axis=0), np.roll(hi_r, 1, axis=0)
    ll = (lo_r + lo_u) / s
    lh = (lo_r - lo_u) / s
    hl = (hi_r + hi_u) / s
    hh = (hi_r - hi_u) / s
    return ll, (lh, hl, hh)


def _haar_iswt2(coeffs) -> np.ndarray:
    ll, (lh, hl, hh) = coeffs
    s = math.sqrt(2.0)
    return ((ll + lh) / s + (hl + hh) / s) / s


def _block_positions(shape: Sequence[int], bs: int) -> List[Tuple[int, int]]:
    h, w = int(shape[0]), int(shape[1])
    return [(r, c) for r in range(0, h - bs + 1, bs) for c in range(0, w - bs + 1, bs)]


def _spectral_gap_state(block: np.ndarray, eps: float):
    b = np.asarray(block, dtype=np.float64)
    sym = 0.5 * (b + b.T)
    skew = 0.5 * (b - b.T)
    vals, vecs = np.linalg.eigh(sym)
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]
    if vals.size < 2:
        raise ValueError("spectral gap needs at least 2x2 block")
    total = float(abs(vals[0]) + abs(vals[1])) + eps
    gap = float((abs(vals[0]) - abs(vals[1])) / total)
    return gap, (vals, vecs, skew, total), total


def _spectral_gap_set(state, target: float) -> np.ndarray:
    vals, vecs, skew, total = state
    out = np.array(vals, copy=True)
    g = float(np.clip(target, 0.0, 0.999999))
    a = 0.5 * total * (1.0 + g)
    b = 0.5 * total * (1.0 - g)
    out[0] = math.copysign(a, vals[0] if vals[0] != 0 else 1.0)
    out[1] = math.copysign(b, vals[1] if vals[1] != 0 else 1.0)
    sym2 = (vecs * out[None, :]) @ vecs.T
    return sym2 + skew


def _svd_gap_state(block: np.ndarray, eps: float):
    u, s, vt = np.linalg.svd(np.asarray(block, dtype=np.float64), full_matrices=False)
    total = float(s[0] + s[1]) + eps
    gap = float((s[0] - s[1]) / total)
    return gap, (u, s, vt, total), total


def _svd_gap_set(state, target: float) -> np.ndarray:
    u, s, vt, total = state
    s2 = np.array(s, copy=True)
    g = float(np.clip(target, 0.0, 0.999999))
    s2[0] = 0.5 * total * (1.0 + g)
    s2[1] = 0.5 * total * (1.0 - g)
    return (u * s2[None, :]) @ vt


def _feature_state(block: np.ndarray, cfg: TransformMethodConfig):
    if cfg.method == "dwt_svd_ngqim":
        return _svd_gap_state(block, cfg.eps)
    return _spectral_gap_state(block, cfg.eps)


def _feature_set(state, target: float, cfg: TransformMethodConfig) -> np.ndarray:
    if cfg.method == "dwt_svd_ngqim":
        return _svd_gap_set(state, target)
    return _spectral_gap_set(state, target)


def _layout(channel: np.ndarray, cfg: TransformMethodConfig):
    if cfg.method == "swt_schur_jmqim":
        ll, (lh, hl, hh) = _haar_swt2(channel)
        return [ll.copy(), lh.copy(), hl.copy(), hh.copy()], 4, "swt"
    ll, (lh, hl, hh) = _haar_dwt2(channel)
    return [ll.copy(), lh.copy(), hl.copy(), hh.copy()], 2, "dwt"


def _inverse_layout(bands: Sequence[np.ndarray], kind: str) -> np.ndarray:
    if kind == "swt":
        return _haar_iswt2((bands[0], (bands[1], bands[2], bands[3])))
    return _haar_idwt2((bands[0], (bands[1], bands[2], bands[3])))


def _joint_feature(bands, rr: int, cc: int, bs: int, cfg: TransformMethodConfig):
    gs, states, energies = [], [], []
    for band in bands[:3]:
        g, st, e = _feature_state(band[rr:rr + bs, cc:cc + bs], cfg)
        gs.append(g); states.append(st); energies.append(e)
    w = np.asarray(cfg.weights, dtype=np.float64)
    w /= np.sum(w)
    return float(np.dot(w, gs)), np.asarray(gs), states, float(min(energies)), w


def _select_spectral_positions(channel: np.ndarray, cfg: TransformMethodConfig, count: int):
    bands, bs, _ = _layout(channel, cfg)
    pos = _block_positions(bands[0].shape, bs)
    scored = []
    for i, (rr, cc) in enumerate(pos):
        try:
            z, _, _, energy, _ = _joint_feature(bands, rr, cc, bs, cfg)
            center = min(z, 1.0 - z)
            scored.append((energy * (0.5 + max(0.0, center)), i))
        except Exception:
            scored.append((-math.inf, i))
    scored.sort(reverse=True)
    stable = [pos[i] for _, i in scored[:count]]
    order = _rng(cfg.private_key, cfg.method + "|payload").permutation(len(stable))
    return [stable[int(i)] for i in order], bs


def _embed_spectral_once(channel: np.ndarray, bits: np.ndarray, rows: np.ndarray, cols: np.ndarray, steps: np.ndarray, cfg: TransformMethodConfig, only: np.ndarray | None = None) -> np.ndarray:
    bands, bs, kind = _layout(channel, cfg)
    w = np.asarray(cfg.weights, dtype=np.float64); w /= np.sum(w)
    w2 = float(np.dot(w, w))
    indices = range(len(bits)) if only is None else [int(x) for x in only]
    for k in indices:
        rr, cc = int(rows[k]), int(cols[k])
        try:
            z, gs, states, _, _ = _joint_feature(bands, rr, cc, bs, cfg)
            zt = _qim_target(z, int(bits[k]), float(steps[k]))
            dg = w * ((zt - z) / max(w2, cfg.eps))
            for j in range(3):
                bands[j][rr:rr + bs, cc:cc + bs] = _feature_set(states[j], float(np.clip(gs[j] + dg[j], 0.0, 0.999999)), cfg)
        except Exception:
            continue
    return _inverse_layout(bands, kind)


def _extract_spectral_bits(channel: np.ndarray, rows: np.ndarray, cols: np.ndarray, steps: np.ndarray, cfg: TransformMethodConfig):
    bands, bs, _ = _layout(channel, cfg)
    bits = np.zeros(len(rows), dtype=np.uint8)
    conf = np.zeros(len(rows), dtype=np.float64)
    for k, (rr, cc) in enumerate(zip(rows, cols)):
        try:
            z, _, _, _, _ = _joint_feature(bands, int(rr), int(cc), bs, cfg)
            ev = _qim_evidence(z, float(steps[k]), cfg.eps)
            bits[k] = 1 if ev >= 0 else 0
            conf[k] = abs(ev)
        except Exception:
            pass
    return bits, float(np.mean(conf))


def _normdiff(a: float, b: float, eps: float) -> float:
    return float((abs(a) - abs(b)) / (abs(a) + abs(b) + eps))


def _normdiff_set(a: float, b: float, target: float, eps: float):
    m = abs(a) + abs(b) + eps
    g = float(np.clip(target, -0.999999, 0.999999))
    aa, bb = 0.5 * m * (1.0 + g), 0.5 * m * (1.0 - g)
    return math.copysign(aa, a if a != 0 else 1.0), math.copysign(bb, b if b != 0 else 1.0)


def _select_spread_pairs(channel: np.ndarray, cfg: TransformMethodConfig, count: int):
    ll, _ = _haar_dwt2(channel)
    pairs = [(r, c) for r in range(ll.shape[0]) for c in range(0, ll.shape[1] - 1, 2)]
    scored = [(abs(float(ll[r, c])) + abs(float(ll[r, c + 1])), i) for i, (r, c) in enumerate(pairs)]
    scored.sort(reverse=True)
    stable = [pairs[i] for _, i in scored[:count]]
    order = _rng(cfg.private_key, cfg.method + "|spread").permutation(len(stable))
    return [stable[int(i)] for i in order]


def _embed_spread(channel: np.ndarray, bits: np.ndarray, rows: np.ndarray, cols: np.ndarray, cfg: TransformMethodConfig) -> np.ndarray:
    ll, (lh, hl, hh) = _haar_dwt2(channel)
    for t, (rr, cc) in enumerate(zip(rows, cols)):
        k = t % len(bits)
        g = _normdiff(float(ll[rr, cc]), float(ll[rr, cc + 1]), cfg.eps)
        gt = _qim_target(g, int(bits[k]), cfg.step)
        ll[rr, cc], ll[rr, cc + 1] = _normdiff_set(float(ll[rr, cc]), float(ll[rr, cc + 1]), gt, cfg.eps)
    return _haar_idwt2((ll, (lh, hl, hh)))


def _extract_spread(channel: np.ndarray, rows: np.ndarray, cols: np.ndarray, cfg: TransformMethodConfig):
    ll, _ = _haar_dwt2(channel)
    nbits = cfg.wm_size * cfg.wm_size
    sums = np.zeros(nbits, dtype=np.float64)
    abs_sums = np.zeros(nbits, dtype=np.float64)
    for t, (rr, cc) in enumerate(zip(rows, cols)):
        k = t % nbits
        g = _normdiff(float(ll[rr, cc]), float(ll[rr, cc + 1]), cfg.eps)
        ev = _qim_evidence(g, cfg.step, cfg.eps)
        sums[k] += ev; abs_sums[k] += abs(ev)
    bits = (sums >= 0).astype(np.uint8)
    conf = float(np.mean(np.abs(sums) / (abs_sums + cfg.eps)))
    return bits, conf


def _pilot_bits(cfg: TransformMethodConfig) -> np.ndarray:
    return _rng(cfg.private_key, "pilot-bits").integers(0, 2, size=cfg.pilot_count, dtype=np.uint8)


def _pilot_positions(shape: Sequence[int], cfg: TransformMethodConfig):
    h, w = int(shape[0]), int(shape[1])
    pos = [(r, c) for r in range(16, h - 23, 8) for c in range(16, w - 23, 8)]
    order = _rng(cfg.private_key, "pilot-pos").permutation(len(pos))[:cfg.pilot_count]
    return [pos[int(i)] for i in order]


def _embed_pilots(channel: np.ndarray, cfg: TransformMethodConfig) -> np.ndarray:
    out = np.asarray(channel, dtype=np.float64).copy()
    for bit, (r, c) in zip(_pilot_bits(cfg), _pilot_positions(out.shape, cfg)):
        d = cv2.dct(out[r:r + 8, c:c + 8].astype(np.float32)).astype(np.float64)
        x = float(d[1, 2] - d[2, 1])
        xt = _qim_target(x, int(bit), cfg.pilot_step)
        dd = xt - x
        d[1, 2] += 0.5 * dd; d[2, 1] -= 0.5 * dd
        out[r:r + 8, c:c + 8] = cv2.idct(d.astype(np.float32)).astype(np.float64)
    return out


def _pilot_score(channel: np.ndarray, cfg: TransformMethodConfig, dy: int = 0, dx: int = 0) -> float:
    vals = []
    for bit, (r0, c0) in zip(_pilot_bits(cfg), _pilot_positions(channel.shape, cfg)):
        r, c = r0 + dy, c0 + dx
        if r < 0 or c < 0 or r + 8 > channel.shape[0] or c + 8 > channel.shape[1]:
            continue
        d = cv2.dct(channel[r:r + 8, c:c + 8].astype(np.float32))
        ev = _qim_evidence(float(d[1, 2] - d[2, 1]), cfg.pilot_step)
        signed = ev if int(bit) else -ev
        vals.append(0.5 * (float(np.clip(signed, -1.0, 1.0)) + 1.0))
    return float(np.mean(vals)) if vals else 0.0


def _warp(channel: np.ndarray, angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    h, w = channel.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    return cv2.warpAffine(channel, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def _pilot_sync(channel: np.ndarray, cfg: TransformMethodConfig):
    base = _pilot_score(channel, cfg)
    best = (base, "base", channel, 0, 0)
    rad, st = cfg.sync_translation_radius, max(1, cfg.sync_translation_step)
    for dy in range(-rad, rad + 1, st):
        for dx in range(-rad, rad + 1, st):
            if dy == 0 and dx == 0: continue
            sc = _pilot_score(channel, cfg, dy, dx)
            if sc > best[0]: best = (sc, "shift", channel, dy, dx)
    rs = max(0.5, cfg.sync_rotation_step_deg)
    for a in np.arange(-cfg.sync_rotation_deg, cfg.sync_rotation_deg + 0.5 * rs, rs):
        if abs(float(a)) < 1e-12: continue
        y = _warp(channel, float(a), 1.0); sc = _pilot_score(y, cfg)
        if sc > best[0]: best = (sc, "rotation", y, 0, 0)
    ss = max(0.01, cfg.sync_scale_step)
    for s in np.arange(cfg.sync_scale_min, cfg.sync_scale_max + 0.5 * ss, ss):
        if abs(float(s) - 1.0) < 1e-12: continue
        y = _warp(channel, 0.0, float(s)); sc = _pilot_score(y, cfg)
        if sc > best[0]: best = (sc, "scale", y, 0, 0)
    if best[0] < cfg.sync_accept_score or best[0] < base + cfg.sync_accept_gain:
        return channel, "base"
    _, kind, y, dy, dx = best
    if kind == "shift":
        h, w = y.shape
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        y = cv2.warpAffine(y, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return y, kind


def embed_array(host_img: np.ndarray, watermark_binary: np.ndarray, cfg: TransformMethodConfig = TransformMethodConfig(), collect_diagnostics: bool = False) -> EmbeddingResult:
    validate_config(cfg)
    t0 = time.perf_counter()
    if host_img.shape[:2] != (512, 512) or host_img.ndim != 3 or host_img.shape[2] != 3:
        raise ValueError("This screening patch requires a 512x512 BGR host")
    wm = watermark_binary
    if wm.shape != (cfg.wm_size, cfg.wm_size):
        wm = cv2.resize(wm, (cfg.wm_size, cfg.wm_size), interpolation=cv2.INTER_NEAREST)
    bits = (arnold_transform(wm, cfg.arnold_iter).reshape(-1) >= 128).astype(np.uint8)
    channel = host_img[:, :, 0].astype(np.float64)

    side: Dict[str, np.ndarray] = {
        "method": np.asarray([cfg.method]), "strength": np.asarray([cfg.step], dtype=np.float64),
        "wm_size": np.asarray([cfg.wm_size], dtype=np.int32), "arnold_iter": np.asarray([cfg.arnold_iter], dtype=np.int32),
    }

    if cfg.method == "dwt_normdiff_spread_qim":
        count = len(bits) * int(cfg.spread_repetitions)
        selected = _select_spread_pairs(channel, cfg, count)
        rows = np.asarray([p[0] for p in selected], dtype=np.int32)
        cols = np.asarray([p[1] for p in selected], dtype=np.int32)
        marked = _embed_spread(channel, bits, rows, cols, cfg)
        side.update({"rows": rows, "cols": cols, "repeat": np.asarray([cfg.spread_repetitions], dtype=np.int32)})
    else:
        selected, _ = _select_spectral_positions(channel, cfg, len(bits))
        rows = np.asarray([p[0] for p in selected], dtype=np.int32)
        cols = np.asarray([p[1] for p in selected], dtype=np.int32)
        steps = np.full(len(bits), cfg.step, dtype=np.float64)
        out = host_img.copy()
        work = channel
        for rnd in range(max(1, cfg.closure_rounds + 1)):
            current_bits, _ = _extract_spectral_bits(work, rows, cols, steps, cfg)
            wrong = np.arange(len(bits), dtype=np.int32) if rnd == 0 else np.flatnonzero(current_bits != bits).astype(np.int32)
            if rnd > 0 and wrong.size == 0:
                break
            if rnd > 0:
                steps[wrong] = np.minimum(steps[wrong] * cfg.local_step_growth, cfg.step * cfg.local_step_max_factor)
            work = _embed_spectral_once(work, bits, rows, cols, steps, cfg, wrong)
            out[:, :, 0] = np.clip(np.rint(work), 0, 255).astype(np.uint8)
            work = out[:, :, 0].astype(np.float64)
        marked = work
        side.update({"rows": rows, "cols": cols, "steps": steps})

    marked = _embed_pilots(marked, cfg) if cfg.pilot_count else marked
    output = host_img.copy(); output[:, :, 0] = np.clip(np.rint(marked), 0, 255).astype(np.uint8)
    return EmbeddingResult(output, side, [], time.perf_counter() - t0)


def _cfg_from_side(cfg: TransformMethodConfig, side: Dict[str, np.ndarray]) -> TransformMethodConfig:
    return replace(cfg, method=str(side["method"][0]), step=float(side["strength"][0]))


def extract_array(watermarked: np.ndarray, side_info: Dict[str, np.ndarray], cfg: TransformMethodConfig = TransformMethodConfig()) -> ExtractionResult:
    t0 = time.perf_counter(); cfg = _cfg_from_side(cfg, side_info); validate_config(cfg)
    channel = watermarked[:, :, 0].astype(np.float64); note = ""
    if cfg.sync_mode == "pilot" and cfg.pilot_count:
        channel, kind = _pilot_sync(channel, cfg)
        if kind != "base": note = "+sync:" + kind
    rows, cols = side_info["rows"].astype(np.int32), side_info["cols"].astype(np.int32)
    if cfg.method == "dwt_normdiff_spread_qim":
        bits, conf = _extract_spread(channel, rows, cols, cfg)
    else:
        steps = side_info.get("steps", np.full(cfg.wm_size * cfg.wm_size, cfg.step)).astype(np.float64)
        bits, conf = _extract_spectral_bits(channel, rows, cols, steps, cfg)
    scrambled = (bits.reshape(cfg.wm_size, cfg.wm_size) * 255).astype(np.uint8)
    wm = invert_arnold_transform(scrambled, cfg.arnold_iter)
    return ExtractionResult(wm, conf, DISPLAY_NAMES[cfg.method] + note, int(len(rows)), time.perf_counter() - t0)
