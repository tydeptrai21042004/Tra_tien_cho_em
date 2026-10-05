#!/usr/bin/env python3
"""Convenience wrapper for the first-stage ten-candidate strength screening."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
cmd = [
    sys.executable, str(HERE / "run_candidate_comparison.py"),
    "--repo", str(HERE),
    "--profile", "representative",
    "--screening",
    "--sweep",
    "--repeat", "1",
    "--out", str(HERE / "validation" / "strength_screen.csv"),
]
cmd.extend(sys.argv[1:])
raise SystemExit(subprocess.call(cmd))
