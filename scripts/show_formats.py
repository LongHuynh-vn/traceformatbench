"""Print F0-F5 and character counts for one consecutive eligible trace window.

Usage:
    uv run python scripts/show_formats.py <case_dir> [--phase post] [--index 0]
"""

import argparse
from pathlib import Path

from traceformatbench.data.trace import load_case_traces
from traceformatbench.data.windows import PHASES, eligible, load_window_config, phase_of
from traceformatbench.serializers import SERIALIZERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--phase", choices=PHASES, default="post")
    parser.add_argument("--index", type=int, default=0, help="start index in eligible phase traces")
    parser.add_argument("--config", type=Path, default=Path("configs/windows_v1.yaml"))
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be non-negative.")
    try:
        config = load_window_config(args.config)
        inject_time_s = int((args.case_dir / "inject_time.txt").read_text().strip())
        traces = [
            trace
            for trace in load_case_traces(args.case_dir)
            if phase_of(trace, inject_time_s, config.guard_seconds) == args.phase
            and eligible(trace, config.max_trace_chars)
        ]
        window = traces[args.index : args.index + config.window_size]
        if config.window_size < 1 or len(window) != config.window_size:
            raise ValueError(
                f"Cannot form a {config.window_size}-trace {args.phase} window "
                f"at index {args.index}: {len(traces)} eligible traces."
            )
        for fmt, serializer in SERIALIZERS.items():
            text = serializer(window)
            print(f"{fmt} ({len(text)} characters)\n{text}\n")
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
