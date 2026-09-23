"""Score one system's transcripts against IndicTelephony-Bench.

    python scripts/score.py --predictions my_system.csv --by language --ci

A shim so the repository works cloned as well as installed; the implementation
is `indictelephony_bench.cli`, which `pip install` also exposes as the command
`indictelephony-score`. Run `--help` for the options.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indictelephony_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
