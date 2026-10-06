# Attacks, candidate comparison, and ablation protocol

## Candidate phase

The ten Q-only/R-only methods are competing carrier formulations, not ablations.  Use the same hosts, watermarks, 4x4 QR blocks, payload, secret key, repeat=1, attacks, metrics, and feasibility thresholds.

Run the carrier comparison with:

```bash
python run_candidate_comparison.py --sweep --profile representative \
  --selection stability --sync none \
  --min-psnr 50 --min-clean-nc 0.999999
```

Use `--sync none` to decide the carrier so the ranking is not dominated by a decoder-side geometric search.

## Robust-system phase

After selecting a carrier, evaluate the same carrier with optional self-synchronisation:

```bash
python run_attack_suite.py --method q_smm --strength <chosen> \
  --profile extended --selection stability --sync auto
```

Report core/no-sync and sync-enabled results separately.

## Geometric attacks

`rotation_*_roundtrip` and `translation_*_roundtrip` are interpolation/degradation diagnostics.  `rotation_once_*` and `translation_once_*` are true unsynchronised geometric attacks.  Center crop + resize is also a coordinate-desynchronising attack.

Robust-v3 `--sync auto` searches only small global translation, rotation, and center scale using carrier conformity.  It does not add a second embedding band or pilot pattern.

## Ablation phase

Only after one carrier wins should its components be ablated.  For Q-SMM, useful ablations are:

- stability selection vs keyed chaotic subset;
- host-normalized margin boost on/off;
- stored-domain guard 0.2 vs robust guard;
- `--sync none` vs `--sync auto` (system-level, not carrier novelty);
- optional repeat>1 only as a separate system experiment.

Do not resurrect the removed `q44/r44` adaptive branch as an ablation of the new method.
