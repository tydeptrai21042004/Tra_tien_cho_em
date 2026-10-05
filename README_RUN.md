# Q-only / R-only QR watermark candidate code

This patch **removes the old adaptive two-branch `|q44|` / `r44` proposal**. A run now uses one fixed method for all selected blocks, with no `choose_branch()` logic and no per-block Q/R flags.

## Ten candidate methods

Q-only, based on `q21,q31`:

- `q_gaqim` — Givens Angular QIM
- `q_npm` — Normalized Pair-Margin
- `q_lrm` — Log-Ratio Margin
- `q_lqim` — 2-D Pair Lattice QIM
- `q_smm` — Symmetric Minimum-Margin

R-only, based on the first row of `R`:

- `r_spqim` — Spread/Projection Row QIM on `r12,r13,r14`
- `r_ndqim` — Normalized Differential QIM
- `r_lrqim` — Log-Ratio Row QIM
- `r_rqim` — Repeated First-Row QIM on `r12,r13,r14`
- `r_aqim` — Adaptive-Step First-Row QIM on `r11:r14`

The R scalar primitive follows the supplied formula exactly: target modulo positions are `0.75*S` for bit 1 and `0.25*S` for bit 0; extraction thresholds at `0.5*S`.

## 1. Smoke test

```bash
python smoke_test_10_methods.py
```

## 2. Compare all ten at their defaults

```bash
python run_candidate_comparison.py --quick
```

## 3. First-stage strength screening

```bash
python run_strength_screen.py
```

This performs method-specific strength sweeps and writes:

- `validation/strength_screen.csv`
- `validation/strength_screen_summary.csv`
- `validation/strength_screen_best_per_method.csv`

Feasibility is reported using mean PSNR > 50 dB and clean NC approximately 1 by default.

## 4. Full comparison after screening

Use the selected strengths with `run_attack_suite.py`, for example:

```bash
python run_attack_suite.py --method q_gaqim --strength 0.75 --profile extended \
  --out validation/q_gaqim_extended.csv
```

or run a broad candidate sweep:

```bash
python run_candidate_comparison.py --sweep --profile extended \
  --out validation/candidate_extended.csv
```

## 5. Single-image embed / extract

```bash
python proposal_method.py embed \
  --method r_rqim --strength 6 \
  --host data/hosts/classical/airplane.bmp \
  --watermark data/watermarks/watermark_1.png \
  --output watermarked.png --side-info side_info.npz

python proposal_method.py extract \
  --method r_rqim --strength 6 \
  --watermarked watermarked.png --side-info side_info.npz \
  --output extracted.png
```

The side-information file stores coordinates and the run-level method/strength, but **does not store Q/R branch flags**. The current comparison remains semi-blind because block coordinates are stored; do not call it fully blind unless coordinate regeneration is redesigned separately.

## Fair comparison rules

For the carrier search, keep `repeat=1` so every bit uses one selected 4x4 block. This prevents R-RQIM's three within-block observations from being compounded with hidden across-block repetition. Use the same hosts, watermarks, selected positions, attacks, and metrics for every candidate.

The candidate search is not an ablation study. First choose the winning carrier under the PSNR/clean-NC constraints; then construct an ablation specifically around that winner.
