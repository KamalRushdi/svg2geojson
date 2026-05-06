"""Inc 13 checkpoint on the WORKING-PLANS folder (full floor SVGs, not slices).

Runs the same Inc 13 + Inc 14 pipeline as `tests.inc13_checkpoint` but
against `input/working_plans/` and writes to `output/inc13_working/`.
Output filenames mirror the input stem with `_classified.png`.

Usage:
    .venv/bin/python -m tests.inc13_working_checkpoint
"""

from __future__ import annotations

from pathlib import Path

from tests import inc13_checkpoint

WORKING_DIR = Path("input/working_plans")
OUTPUT_DIR = Path("output/inc13_working")


def main() -> None:
    samples = sorted(WORKING_DIR.glob("*.svg"))
    if not samples:
        raise SystemExit(f"No SVGs found in {WORKING_DIR}")
    # Override the module-level constants used by inc13_checkpoint.main().
    inc13_checkpoint.SAMPLES = samples
    inc13_checkpoint.OUTPUT_DIR = OUTPUT_DIR
    inc13_checkpoint.main()


if __name__ == "__main__":
    main()
