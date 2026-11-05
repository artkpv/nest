"""
Multi-baseline solver: runs stego response (already generated upstream) plus
each requested baseline on the same sample, storing per-baseline outputs and
correctness flags in state.metadata without touching state.output.
"""

from inspect_ai.solver import Solver, solver, TaskState, Generate
from inspect_ai.model import GenerateConfig, get_model

from stego_inspect.utils import get_reasoning_effort
from stego_inspect.clean_protocol import (
    FILLER_TARGET_TOKENS,
    CleanProtocol,
    build_clean_messages,
    clean_generate_config,
    grade_clean_arm,
)
from stego_inspect.scorers.hidden_task import (
    CUT_STOP_REASONS,
    check_answer_correctness,
    extract_baseline_answer,
)
from stego_inspect.scorers.refusal import check_refusal_heuristic

from .baselines import (
    baseline_cot_solver,
    baseline_direct_solver,
    baseline_dual_solver,
    baseline_ttc_solver,
)


BASELINE_SOLVERS = {
    "direct": baseline_direct_solver,
    "cot": baseline_cot_solver,
    "ttc": baseline_ttc_solver,
    "dual": baseline_dual_solver,
}


def _reasoning_config(model_name: str) -> GenerateConfig | None:
    try:
        params = get_reasoning_effort(model_name)
    except NotImplementedError:
        return None
    return GenerateConfig(**params) if params else None


async def _run_clean_baselines(
    state: TaskState, baseline_names: list[str], protocol: CleanProtocol
) -> TaskState:
    """Clean protocol: tri-state `correct`, plus stop_reason, tokens and status."""
    model = get_model()
    reasoning = _reasoning_config(str(model))
    problem = state.metadata.get("problem", state.input_text)
    ground_truth = state.target if isinstance(state.target, str) else state.target.text

    # A cut or empty stego output is excluded from every rate and pair, so its
    # baselines could change nothing. Skipping them saves the calls.
    stego = state.output
    stop_reason = str(stego.stop_reason) if stego and stego.choices else None
    if stop_reason in CUT_STOP_REASONS or not (stego and stego.completion.strip()):
        for name in baseline_names:
            state.metadata[f"baseline_{name}_correct"] = None
            state.metadata[f"baseline_{name}_status"] = "skipped"
        return state

    for name in baseline_names:
        messages = build_clean_messages(name, problem, protocol)
        config = clean_generate_config(name, reasoning)
        completion, stop_reason, output_tokens, error = "", None, None, None
        try:
            if config is not None:
                result = await model.generate(input=messages, config=config)
            else:
                result = await model.generate(input=messages)
            completion = result.completion or ""
            stop_reason = result.stop_reason if result.choices else None
            output_tokens = result.usage.output_tokens if result.usage else None
            error = result.error
        except Exception as e:
            error = str(e)

        graded = grade_clean_arm(
            name, ground_truth, protocol, completion, stop_reason, error, output_tokens
        )
        state.metadata[f"baseline_{name}_output"] = completion
        if error:
            state.metadata[f"baseline_{name}_error"] = error
        for field, value in graded.items():
            state.metadata[f"baseline_{name}_{field}"] = value

    return state


@solver
def run_baselines(
    baseline_names: list[str],
    filler_words: int = 0,
    filler_target_tokens: int = FILLER_TARGET_TOKENS,
) -> Solver:
    """
    For each baseline in `baseline_names`, build the baseline prompt by
    invoking the baseline solver, then call the model directly to produce
    a completion. Store completion, refusal flag, and correctness flag in
    `state.metadata[f"baseline_{name}_output|refused|correct"]`.

    Does not modify `state.output` — the stego completion produced upstream
    remains available for stego-specific scorers.

    `filler_words` > 0 selects the clean protocol (see `clean_protocol.py`);
    0 keeps the legacy baselines every published cell used.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not baseline_names:
            return state

        if filler_words:
            protocol = CleanProtocol(filler_words, filler_target_tokens)
            return await _run_clean_baselines(state, baseline_names, protocol)

        model = get_model()
        # str(model) is "provider/name"; model.name drops the provider, and the
        # reasoning switch is route-dependent (e.g. vllm/ vs openrouter/).
        model_name = str(model)
        config = _reasoning_config(model_name)

        # Preserve stego context so scorers downstream see it unchanged.
        stego_messages = list(state.messages)
        stego_baseline_type = state.metadata.get("baseline_type")

        ground_truth = state.target if isinstance(state.target, str) else state.target.text

        for name in baseline_names:
            solver_factory = BASELINE_SOLVERS.get(name)
            if solver_factory is None:
                continue

            # Baseline solver rewrites state.messages and sets baseline_type.
            state = await solver_factory()(state, generate)

            try:
                if config is not None:
                    result = await model.generate(input=state.messages, config=config)
                else:
                    result = await model.generate(input=state.messages)
                completion = result.completion if result else ""
            except Exception as e:
                state.metadata[f"baseline_{name}_output"] = ""
                state.metadata[f"baseline_{name}_error"] = str(e)
                state.metadata[f"baseline_{name}_refused"] = False
                state.metadata[f"baseline_{name}_correct"] = False
                continue

            refused = check_refusal_heuristic(completion)
            answer = extract_baseline_answer(completion, name)
            correct = check_answer_correctness(answer, ground_truth)

            state.metadata[f"baseline_{name}_output"] = completion
            state.metadata[f"baseline_{name}_answer"] = answer
            state.metadata[f"baseline_{name}_refused"] = refused
            state.metadata[f"baseline_{name}_correct"] = correct

        # Restore stego messages / baseline_type so downstream scorers are not confused.
        state.messages = stego_messages
        if stego_baseline_type is None:
            state.metadata.pop("baseline_type", None)
        else:
            state.metadata["baseline_type"] = stego_baseline_type

        return state

    return solve
