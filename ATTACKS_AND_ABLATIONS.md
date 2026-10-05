# Candidate comparison and later ablation

## Candidate stage

The ten methods are **competing proposal formulations**, not components of one adaptive Q/R method. Run all methods with the same data and `repeat=1`.

Suggested selection order:

1. require mean PSNR > 50 dB;
2. require clean NC = 1 where possible;
3. maximize hard-attack mean NC;
4. then maximize global attacked mean NC;
5. use worst-attack NC and BER as tie-breakers.

`run_candidate_comparison.py --sweep --screening` uses a compact first-stage set including clean, JPEG, noise, blur, scaling and crop. After selecting one or two strengths per method, use the extended/stress profiles.

## Geometric attacks

The attack module now keeps round-trip rotation/translation only as interpolation diagnostics and also provides true one-way rotations/translations. One-way geometry is an unsynchronised attack; do not label a round-trip attack as ordinary rotation robustness.

## Ablation stage

Do **not** use the deleted adaptive-branch ablations. After the winning carrier is known, create an ablation around that specific method. Examples:

- Q winner: remove normalization/guard, replace Givens update with a simpler update, compare angular/lattice formulation, then study repetition separately.
- R winner: compare projection vectors, direct single-coefficient QIM, repeated row observations, normalized/relative feature, then study repetition separately.
