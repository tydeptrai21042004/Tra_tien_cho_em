#!/usr/bin/env python3
from __future__ import annotations
import shutil
import sys
from pathlib import Path

FILES = [
    "robust_transform_method.py",
    "run_high_robust_candidates.py",
    "smoke_test_high_robust.py",
    "README_HIGH_ROBUST.md",
]

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python APPLY_PATCH.py /path/to/Tra_tien_cho_em")
    target = Path(sys.argv[1]).resolve()
    if not (target / "proposal_method.py").exists():
        raise SystemExit(f"Not a Tra_tien_cho_em repository: {target}")
    here = Path(__file__).resolve().parent
    for name in FILES:
        src = here / name
        dst = target / name
        shutil.copy2(src, dst)
        print(f"copied {name} -> {dst}")
    print("Patch applied. Run: python smoke_test_high_robust.py")

if __name__ == "__main__":
    main()
