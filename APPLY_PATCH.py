#!/usr/bin/env python3
from pathlib import Path
import shutil, sys, time

HERE = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    raise SystemExit("Usage: python APPLY_PATCH.py /path/to/Tra_tien_cho_em")
repo = Path(sys.argv[1]).expanduser().resolve()
if not repo.exists():
    raise SystemExit(f"Repository not found: {repo}")

replace_files = [
    "proposal_method.py",
    "run_candidate_comparison.py",
    "run_strength_screen.py",
    "run_attack_suite.py",
    "smoke_test_10_methods.py",
    "attacks.py",
    "run_ablation.py",
    "README_RUN.md",
    "README_ROBUST_V3.md",
    "ATTACKS_AND_ABLATIONS.md",
    "REMOVE_OLD_FILES.txt",
]
obsolete = [
    "run_branch_position_justification.py",
    "validation_sample/branch_counts.csv",
    "validation_sample/branch_feature_summary.csv",
    "validation_sample/branch_position_report.txt",
    "validation_sample/branch_position_summary.csv",
    "validation_sample/branch_spatial_grid.csv",
    "validation_sample/ablation_quick.csv",
    "validation_sample/ablation_quick_summary.csv",
    "validation_sample/ablation_quick_delta_vs_full.csv",
]

backup = repo / ("backup_before_robust_v3_" + time.strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=True)

for name in replace_files:
    src = HERE / name
    if not src.exists():
        raise SystemExit(f"Patch file missing: {src}")
    dst = repo / name
    if dst.exists():
        b = backup / name
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, b)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

for name in obsolete:
    dst = repo / name
    if dst.exists():
        b = backup / name
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, b)
        dst.unlink()

print(f"Applied robust-v3 patch to: {repo}")
print(f"Backup of replaced/removed files: {backup}")
print("\nValidate core carriers:")
print("  python smoke_test_10_methods.py")
print("\nCarrier strength screening (strict/no transform search):")
print("  python run_strength_screen.py --selection stability --sync none")
print("\nMaximum-robustness Q-SMM example:")
print("  python run_attack_suite.py --method q_smm --strength 0.007 --profile representative --selection stability --sync auto --out validation/q_smm_v3.csv")
