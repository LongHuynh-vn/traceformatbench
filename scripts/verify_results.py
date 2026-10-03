"""Check that the committed main-grid outputs follow from the committed calls and code.

Usage: uv run python scripts/verify_results.py configs/verify_v1.yaml
Writes only to a temporary directory. Exit code 0 only if every check passes.
"""

import argparse
import sys
from pathlib import Path

from traceformatbench.analysis.verify import load_verify_config, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        checks = verify(load_verify_config(args.config))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for check in checks:
        print(f"{'OK  ' if check.ok else 'FAIL'}  {check.name}: {check.detail}")
    failed = sum(not check.ok for check in checks)
    print(f"{len(checks) - failed} of {len(checks)} checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
