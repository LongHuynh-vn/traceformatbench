"""Call fixed T1/T2 tasks with local Ollama models; append and resume JSONL calls.

Usage: uv run python scripts/run_tasks.py configs/tasks_smoke.yaml
"""

import argparse
import json
import urllib.error
from pathlib import Path

import yaml

from traceformatbench.runners.ollama import OllamaClient
from traceformatbench.runners.tasks import TaskConfig, read_records, record_key, run_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = TaskConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
        output = Path(config.output)
        previous = read_records(output, allow_incomplete_tail=True)
        done = frozenset(record_key(record) for record in previous)
        print(f"Resuming with {len(done)} recorded calls." if done else "Starting fresh.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in previous),
            encoding="utf-8",
        )
        with output.open("a", encoding="utf-8") as stream:
            for index, record in enumerate(
                run_tasks(config, OllamaClient(config.ollama_url), done),
                len(done) + 1,
            ):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"[{index}] {record['model']} {record['case']} {record['task']} "
                    f"{record['label']} window={record['window_index']} {record['format']} "
                    f"{record['status']}",
                    flush=True,
                )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama request failed ({error.code}): {detail}") from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise SystemExit(
            f"Cannot reach Ollama at {config.ollama_url}. Start Ollama: {error}"
        ) from error
    except (LookupError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Raw calls: {output}")


if __name__ == "__main__":
    main()
