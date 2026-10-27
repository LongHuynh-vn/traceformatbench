"""Render the validated single-column paper figure from saved analysis tables.

Usage: uv run --locked python scripts/plot_paper_figure.py configs/paper_figure_v1.yaml
"""

import argparse
from pathlib import Path

from traceformatbench.analysis.paper_figure import export_paper_figure, load_paper_figure_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        provenance = export_paper_figure(args.config)
        config = load_paper_figure_config(args.config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
    print(f"Figure: {config.output_pdf}")
    print(f"SHA-256: {provenance['outputs']['pdf_sha256']}")
    print(f"Preview: {config.output_png}")
    print(f"Provenance: {config.provenance}")


if __name__ == "__main__":
    main()
