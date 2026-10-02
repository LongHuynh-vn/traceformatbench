"""Sample wrong T2 answers, rebuild their prompts and write the labeling material (spec 010).

Usage: uv run python scripts/sample_errors.py configs/error_analysis_v1.yaml
EXPLORATORY: no model calls; the student assigns every reason by hand.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from traceformatbench.analysis.errors import (
    assign_item_ids,
    automatic_flags,
    check_prompt_hash,
    item_markdown,
    load_error_config,
    output_files,
    rater2_item_ids,
    raw_ranking,
    read_jsonl,
    sample_items,
    write_files,
)
from traceformatbench.data.trace import load_case_traces
from traceformatbench.data.windows import load_window_config
from traceformatbench.runners.tasks import TaskConfig, load_task_windows, prompt_sha256
from traceformatbench.tasks.prompts import SYSTEM_PROMPT, build_t2_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = load_error_config(args.config)
        items = assign_item_ids(sample_items(read_jsonl(config.scores), config), config.seed)
        grid = TaskConfig(**yaml.safe_load(Path(config.task_config).read_text(encoding="utf-8")))
        cases = sorted({item["case"] for item in items})
        formats = sorted({item["format"] for item in items} | {"F2"})
        grid = replace(grid, cases=cases, formats=formats, tasks=["T2"])
        windows_config = load_window_config(grid.windows_config)
        windows = {
            (w["case"], w["window_index"]): w
            for w in load_task_windows(grid, windows_config)
            if w["task"] == "T2"
        }
        texts = {}
        for case in cases:
            by_id = {t.trace_id: t for t in load_case_traces(Path(windows_config.data_dir) / case)}
            for item in (i for i in items if i["case"] == case):
                window = windows[(case, item["window_index"])]
                user = build_t2_prompt(
                    window["texts"][item["format"]], len(window["trace_ids"]), window["candidates"]
                )
                check_prompt_hash(item, prompt_sha256(SYSTEM_PROMPT, user))
                ranking = raw_ranking(item)
                traces = [by_id[trace_id] for trace_id in window["trace_ids"]]
                item.update(
                    raw_ranking=ranking,
                    **automatic_flags(item, ranking, traces, config.habitual[item["system"]]),
                )
                texts[item["item_id"]] = (user, window["texts"]["F2"])
            del by_id
        item_texts = {
            item["item_id"]: item_markdown(item, SYSTEM_PROMPT, *texts[item["item_id"]])
            for item in items
        }
        rater2 = rater2_item_ids([i["item_id"] for i in items], config.seed, config.rater2_size)
        codebook = Path(config.codebook).read_text(encoding="utf-8")
        write_files(output_files(items, rater2, item_texts, codebook), config.output_dir)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"{len(items)} items, all {len(items)} prompt hashes match. Output: {config.output_dir}")


if __name__ == "__main__":
    main()
