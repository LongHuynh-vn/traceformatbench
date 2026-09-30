"""Fixed spec-005 v1 prompts, independent of serialization format."""

from collections.abc import Sequence

from traceformatbench.data.trace import Trace

PROMPT_VERSION = "v1"
SYSTEM_PROMPT = (
    "You analyze distributed traces from a microservice system. Reply with one JSON object only."
)


def build_t1_prompt(traces_text: str, n: int) -> str:
    return (
        f"Below are {n} traces recorded during one time window of a microservice system.\n\n"
        f"<traces>\n{traces_text}\n</traces>\n\n"
        "At some point, a fault was injected into one service of this system. "
        "Were these traces recorded before or after the fault injection?\n"
        'Answer with {"phase": "pre"} if before, or {"phase": "post"} if after.'
    )


def build_t2_prompt(traces_text: str, n: int, candidates: Sequence[str]) -> str:
    return (
        f"Below are {n} traces recorded after a fault was injected into exactly one service "
        "of a microservice system.\n\n"
        f"<traces>\n{traces_text}\n</traces>\n\n"
        f"Candidate services: {', '.join(sorted(candidates))}\n"
        "Which service is the root cause (the service where the fault was injected)?\n"
        'Answer with {"ranking": [first, second, third]}: the three most likely root-cause '
        "services from the candidate list, most likely first, no duplicates."
    )


def build_t1p_prompt(a_text: str, b_text: str, n: int) -> str:
    return (
        f"Below are two windows of {n} traces each, recorded in the same microservice system.\n"
        "At some point, a fault was injected into one service of this system. "
        "One window was recorded before the fault injection and the other after it.\n\n"
        f"<window_a>\n{a_text}\n</window_a>\n\n"
        f"<window_b>\n{b_text}\n</window_b>\n\n"
        "Which window was recorded after the fault injection?\n"
        'Answer with {"after": "A"} or {"after": "B"}.'
    )


def case_candidates(traces: Sequence[Trace]) -> list[str]:
    """Include every span service, before any trace eligibility filtering."""
    return sorted({span.service for trace in traces for span in trace.spans})
