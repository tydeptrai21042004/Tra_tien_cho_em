#!/usr/bin/env python3
"""Deprecated compatibility entry point.

The old adaptive Q/R component ablation was removed.  The ten new algorithms
are competing candidate methods, not ablations.  This wrapper launches the
candidate comparison so old notebooks do not silently evaluate the removed
|q44|/r44 method.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
print("[deprecated] run_ablation.py -> run_candidate_comparison.py --sweep", file=sys.stderr)
cmd = [sys.executable, str(HERE / "run_candidate_comparison.py"), "--repo", str(HERE), "--sweep"]
cmd.extend(sys.argv[1:])
raise SystemExit(subprocess.call(cmd))
