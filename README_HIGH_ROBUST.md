# High-robustness QIM candidate patch

This patch is an **experimental screening patch** for the next robustness round. It keeps the original binary two-coset QIM rule

- bit 0: `0.25*S`
- bit 1: `0.75*S`

but moves the carrier into wavelet / spectral representations and adds a keyed geometric synchronizer.

It does **not** hard-code or fake `NC > 0.90`. The runner measures that target on the real hosts, watermarks, and attacks and reports `target_met=1` only when the measured attacked mean NC is at least the requested target while the PSNR/clean-NC constraints also pass.

## Added candidates

1. `dwt_schur_jmqim`
   - Haar DWT.
   - Symmetric-Schur spectral-gap carrier.
   - Joint LL/LH/HL QIM.
   - Host-stability block selection.
   - Stored-domain local step closure.

2. `swt_schur_jmqim`
   - Undecimated Haar/SWT representation.
   - Same Schur spectral-gap carrier.
   - Intended to test shift tolerance.

3. `dwt_svd_ngqim`
   - Haar DWT.
   - Normalized SVD gap carrier.
   - Included as a performance oracle against the Schur construction.

4. `dwt_normdiff_spread_qim`
   - Haar DWT LL normalized-difference carrier.
   - Four keyed, spatially dispersed observations per payload bit.
   - This is the strongest robustness-oriented engineering candidate in the patch.

5. `q_smm_sync`
   - Existing Q-SMM + existing auto synchronization, used by the runner as the control.

## Blind / semi-blind status

The new transform candidates are **semi-blind**: extraction does not require the original host image, but the spectral/spread candidates store the selected carrier coordinates (and, for spectral closure, local QIM steps) in side information. The geometric synchronizer itself is blind with respect to the host and uses a small keyed pilot set.

No attack classifier is used.

## Why the patch includes a spread-QIM candidate

The previous result was limited mainly by JPEG/noise plus unsynchronised crop/rotation/translation. A decomposition change by itself is unlikely to move mean NC from about 0.76 to above 0.90. The spread candidate is therefore included as the robustness ceiling test while the Schur/SVD variants test the proposed mathematical carrier.

## Run

Smoke test:

```bash
python smoke_test_high_robust.py
```

Quick one-host/one-watermark screen:

```bash
python run_high_robust_candidates.py \
  --quick \
  --sweep \
  --profile representative \
  --min-psnr 50 \
  --target-mean-nc 0.90 \
  --out validation/high_robust_quick.csv
```

Full 6-host x 2-watermark screen:

```bash
python run_high_robust_candidates.py \
  --sweep \
  --profile representative \
  --min-psnr 50 \
  --target-mean-nc 0.90 \
  --out validation/high_robust_full.csv
```

For the clean mathematical carrier comparison, run with `--sync none`. For the maximum-robustness system experiment, keep the default `--sync pilot`.

## Recommended interpretation

Do not pick the method only by rank. Keep a candidate only if all of the following are satisfied on the full run:

- clean NC = 1.0 (or the exact threshold you set),
- mean PSNR >= 50 dB,
- attacked mean NC >= 0.90,
- weak attacks such as JPEG Q50 and Gaussian noise are not near random decoding,
- the result is consistent across hosts and both watermarks.
