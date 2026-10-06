# Robust-v3: same ten QR carriers, stronger robustness

This revision keeps the ten independent Q-only/R-only proposal methods.  It does **not** restore the removed adaptive `|q44|/r44` branch and does not combine Q and R in one watermark.

## What changed

### 1. Carrier-stability block selection

Instead of taking a keyed random subset from every determinant-valid 4x4 block, the embedder first ranks blocks by how little the *same QR carrier* changes under a fixed weak perturbation bank:

- Gaussian blur, sigma 0.5
- JPEG quality 70
- resize 0.75 and restore

The ranking is independent of the embedded bit.  After the stable subset is selected, the secret-key chaotic permutation is still used for payload order.

For Q methods, the stability test uses only `q21,q31`.  For R methods, it uses the corresponding first-row R feature.  No Q/R branch flag is introduced.

### 2. Host-normalized margin allocation

For the sign-decoded Q methods (`Q-SMM`, `Q-NPM`, `Q-LRM`), less-stable selected blocks receive a somewhat larger embedding margin.  The boost is normalized by a host-specific stability quantile, so difficult images do not receive unbounded strength.

The decoder still decides from the original carrier sign.  It does not need a per-block strength table.

### 3. Stronger stored-domain closure

The first version embedded a floating QR target but decoded the rounded uint8 image.  Robust-v3 pushes sign/margin Q carriers deeper into the correct decision region *after rounding* and keeps bounded repair local to the spatial samples that determine recovered `q21,q31`.

Periodic Q/R carriers use their own smaller guard around their natural QIM codeword rather than the sign-method guard.

### 4. Optional carrier-conformity self-synchronisation

`--sync auto` adds a decoder-side alignment search for small geometric desynchronisation.  It does **not** add a pilot band.  The score is computed from how strongly the selected blocks conform to the same watermark carrier.

The search covers:

- small integer translations (default +/-8 pixels),
- small rotations (default +/-5 degrees),
- center scale correction (used for center-crop + resize attacks).

The search is accepted only if the best carrier-conformity score is both absolutely strong and substantially better than the uncorrected score.  This avoids treating blur/JPEG/noise as a geometric transform.

For a strict experiment that forbids transform search, keep `--sync none` (the default).  For maximum system robustness, use `--sync auto` and report it as a separate synchronised configuration.

## Development validation

On `airplane.bmp + watermark_1.png` with the compact screening attacks (JPEG50, Gaussian noise, blur, scale 0.5, crop 10%), the robust-v3 default Q-SMM run produced approximately:

- clean NC = 1.0000
- PSNR = 51.93 dB
- attacked screening mean NC = 0.7795

On the full representative 15-condition profile with `--sync auto`, the same Q-SMM configuration produced approximately:

- clean NC = 1.0000
- PSNR = 51.93 dB
- attacked mean NC = 0.8945
- hard-attack mean NC = 0.8658
- worst attacked NC = 0.6602

These are development sanity results for one host/watermark pair, not final paper-wide numbers.  Run the full host/watermark sweep before selecting a proposal.

## Recommended runs

Core carrier comparison (no transform search):

```bash
python smoke_test_10_methods.py
python run_strength_screen.py
python run_candidate_comparison.py --profile representative \
  --selection stability --sync none \
  --out validation/robust_v3_core.csv
```

Maximum-robustness system comparison:

```bash
python run_candidate_comparison.py --profile representative \
  --selection stability --sync auto \
  --out validation/robust_v3_sync.csv
```

For the likely leading Q-SMM candidate:

```bash
python run_attack_suite.py --method q_smm --strength 0.007 \
  --profile extended --selection stability --sync auto \
  --out validation/q_smm_robust_v3.csv
```

## What is intentionally NOT added

- no adaptive Q-versus-R branch selection;
- no `q44` / `r44` carrier;
- no Q/R branch flags;
- no ECC;
- no second watermark band;
- no attack classifier;
- no pilot band.

The core proposal remains exactly the requested comparison between Q-only `q21,q31` carriers and R-only first-row carriers.
