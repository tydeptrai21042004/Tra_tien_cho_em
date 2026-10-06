# Q-only / R-only QR watermark candidates — robust-v3

The old adaptive two-branch `|q44| / r44` proposal is removed.  A run uses **one fixed method** for every selected block, with no `choose_branch()` logic and no per-block Q/R flags.

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

The scalar R rule remains the supplied formula: modulo target `0.75*S` for bit 1, `0.25*S` for bit 0, with threshold `0.5*S`.

## Robust-v3 additions

The carrier itself is unchanged.  Robust-v3 adds:

1. bit-independent carrier-stability block selection;
2. host-normalized margin allocation for Q-SMM/Q-NPM/Q-LRM;
3. stronger uint8-domain closure;
4. optional carrier-conformity geometric self-synchronisation.

Read `README_ROBUST_V3.md` for details.

## 1. Smoke test

```bash
python smoke_test_10_methods.py
```

## 2. Compare all ten at default strengths

Strict/core mode, without transform search:

```bash
python run_candidate_comparison.py --quick --selection stability --sync none
```

Maximum-robustness mode:

```bash
python run_candidate_comparison.py --quick --selection stability --sync auto
```

## 3. Strength screening

```bash
python run_strength_screen.py --selection stability --sync none
```

This writes:

- `validation/strength_screen.csv`
- `validation/strength_screen_summary.csv`
- `validation/strength_screen_best_per_method.csv`

The default feasibility rule is mean PSNR > 50 dB and clean NC approximately 1.

For the strict carrier paper, select strength with `--sync none`.  Synchronisation should be evaluated separately rather than used to choose the underlying carrier.

## 4. Full selected-method evaluation

Example Q-SMM core run:

```bash
python run_attack_suite.py --method q_smm --strength 0.007 \
  --profile extended --selection stability --sync none \
  --out validation/q_smm_core.csv
```

Maximum-robustness Q-SMM run:

```bash
python run_attack_suite.py --method q_smm --strength 0.007 \
  --profile extended --selection stability --sync auto \
  --out validation/q_smm_sync.csv
```

## 5. Single-image embed / extract

```bash
python proposal_method.py embed \
  --method q_smm --strength 0.007 --selection stability \
  --host data/hosts/classical/airplane.bmp \
  --watermark data/watermarks/watermark_1.png \
  --output watermarked.png --side-info side_info.npz

python proposal_method.py extract \
  --method q_smm --strength 0.007 --sync auto \
  --watermarked watermarked.png --side-info side_info.npz \
  --output extracted.png
```

The side-information file stores selected coordinates and the run-level method/strength, but it stores **no Q/R branch flags**.  The current comparison is therefore semi-blind because coordinates are stored.  Do not call it fully blind unless coordinate regeneration is redesigned separately.

## Fair comparison rule

Keep `repeat=1` while screening the ten carriers.  First choose the winning carrier under the PSNR/clean-NC constraints.  Only after that should you construct a method-specific ablation or separately test synchronisation/repetition/system enhancements.
