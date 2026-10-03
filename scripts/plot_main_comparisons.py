"""Draw the forest plot of the pre-registered main comparisons from a saved comparisons table.

Usage: uv run python scripts/plot_main_comparisons.py configs/figures_v1.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from traceformatbench.analysis.figures import load_figure_config, plot_main_comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_figure_config(args.config)
    try:
        comparisons = pd.read_csv(config.comparisons)
        Path(config.output).parent.mkdir(parents=True, exist_ok=True)
        plot_main_comparisons(comparisons, config.output)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Figure: {config.output}")


if __name__ == "__main__":
    main()
