# Paper-aligned QR watermarking code + bundled data

This package is self-contained. It contains the paper-aligned Python implementation, the host/watermark data copied unchanged from the supplied repository ZIP, a stronger component ablation, a branch-position justification experiment, and expanded deterministic attack suites.

## Bundled data

### Host images
`data/hosts/classical/`

- `airplane.bmp`
- `girl.bmp`
- `lenna.bmp`
- `manhattan.bmp`
- `pepper.bmp`
- `safari.bmp`

### Watermark images
`data/watermarks/`

- `watermark_1.png`
- `watermark_2.png`

The scripts load the watermarks as binary 64×64 payloads. Manhattan is intentionally included even though the paper-aligned capacity test rejects it for the 4096-bit, `r=3` configuration; this makes the exclusion reproducible.

## Paper-aligned implementation

`proposal_method.py` follows the manuscript method:

- blue-channel, non-overlapping 4×4 blocks;
- determinant gate `|det(A)| > 1e-8` before QR;
- nominal odd repetition factor (`r=3` for 512×512 host and 64×64 watermark);
- key-dependent chaotic ordering of determinant-eligible blocks;
- canonical QR with non-negative diagonal of `R`;
- Q branch based on `|q44|` and a 4×4 Givens rotation;
- R branch based on `r44 mod q` and the positive QIM codebook;
- branch selection after round/clip in the actual stored-pixel domain;
- robustness score from raw / mean / worst local perturbation margins;
- final score `robustness / (distortion + eps)`;
- semi-blind side information storing selected block coordinates and Q/R flags;
- repeated soft-decision recovery;
- raw, NLM h=3, and NLM h=7 extraction candidates selected by global confidence.

## Stronger ablation

`run_ablation.py` now isolates eleven versions:

1. `full` — complete proposal.
2. `q_only` — removes adaptive dual-branch selection and always uses Q.
3. `r_only` — removes adaptive dual-branch selection and always uses R.
4. `distortion_only` — branch decision ignores robustness.
5. `robustness_only` — branch decision ignores distortion normalization.
6. `raw_score` — removes the local perturbation ensemble.
7. `no_safe_margin` — sets both Q/R safety margins to zero.
8. `hard_vote` — replaces soft reliability accumulation with ±1 hard votes.
9. `raw_decoder` — removes NLM candidates / confidence-guided decoder selection.
10. `fixed_mild_decoder` — always uses mild NLM instead of confidence selection.
11. `repeat_1` — removes repetition and uses one observation per payload bit.

The runner writes:

- raw per-host / per-watermark / per-attack results;
- `*_summary.csv` with mean, standard deviation, minimum NC and BER statistics;
- `*_delta_vs_full.csv` with paired NC/BER changes against the complete method plus win/tie/loss counts.

Recommended full ablation:

```bash
python run_ablation.py --repo . --attack-profile representative
```

Clean-only ablation:

```bash
python run_ablation.py --repo . --clean-only
```

Fast smoke test:

```bash
python run_ablation.py --repo . --quick
```

To make the ablation much heavier, use:

```bash
python run_ablation.py --repo . --attack-profile extended
```

or:

```bash
python run_ablation.py --repo . --attack-profile stress
```

## Expanded attack suite

`attacks.py` provides four deterministic profiles:

- `paper`: 12 cases including clean + the manuscript attack settings.
- `representative`: 12 diverse cases intended for component ablation.
- `extended`: 34 cases.
- `stress`: 52 cases with multiple severity levels.

The attack families now include:

- Gaussian blur;
- average filtering;
- median filtering;
- bilateral filtering;
- sharpening;
- Gaussian noise;
- speckle noise;
- salt-and-pepper noise;
- JPEG at multiple quality levels;
- JPEG2000;
- low-pass filtering;
- scale down/up at multiple factors;
- rotation and inverse rotation, including small-angle and 45° conditions;
- translation and inverse translation;
- center crop followed by resize-back;
- gamma correction;
- brightness changes;
- contrast changes;
- random occlusion at several proportions.

Run the complete proposal against the extended suite:

```bash
python run_attack_suite.py --repo . --profile extended
```

Run the 52-case stress suite:

```bash
python run_attack_suite.py --repo . --profile stress
```

The attack runner produces both a raw CSV and an attack-level summary CSV.

## Branch-position justification

```bash
python run_branch_position_justification.py --repo .
```

This records branch coordinates, local image features, branch scores, and 4×4 spatial-grid Q/R occupancy. It is intended to support the paper argument that branch choice is driven by local candidate quality rather than a hard-coded spatial region.

## Single-image commands

Clean benchmark:

```bash
python proposal_method.py clean-benchmark \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png
```

Embed:

```bash
python proposal_method.py embed \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --output outputs/watermarked_girl.png \
  --side-info outputs/girl_side_info.npz \
  --diagnostics outputs/girl_branch_diagnostics.csv
```

Extract:

```bash
python proposal_method.py extract \
  --watermarked outputs/watermarked_girl.png \
  --side-info outputs/girl_side_info.npz \
  --output outputs/extracted_girl.png
```

## Install

```bash
pip install -r requirements.txt
```

## Reproducibility notes

All stochastic attacks use fixed seeds. Attack output sizes remain identical to the watermarked host so stored semi-blind block coordinates remain well-defined. The code records attack failures (for example, a platform lacking JPEG2000 support) instead of silently dropping them.
