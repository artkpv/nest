"""
Encoder solver for steganographic tasks.
"""

from inspect_ai.solver import Solver, solver, TaskState, Generate
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from stego_inspect.prompts.encoder import get_stego_cot_prompts, get_counting_prompts


@solver
def stego_encoder_solver(
    algorithm: str = "word_acrostic",
    use_mapping: bool = False,
    framing: str = "covert",
) -> Solver:
    """
    Solver that generates steganographically encoded responses.

    This solver sets up the system prompt and user message for the
    stego CoT task, then generates a response.

    Args:
        algorithm: The encoding algorithm to use
        use_mapping: Whether to include character mapping in prompt
        framing: "covert" (the paper's prompt) or "neutral" (same structural
            constraint, no concealment language). See
            `prompts.encoder.get_stego_cot_prompts` — the neutral arm measures
            channel capacity, not the paper's threat model, and is reported
            separately.

    Returns:
        A Solver that generates stego-encoded responses
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get hidden problem and cover question from metadata
        hidden_problem = state.metadata.get("problem", state.input_text)
        cover_question = state.metadata.get("cover_question", "")

        # Handle different hidden task types
        hidden_task_type = state.metadata.get("hidden_task_type", "arithmetic")

        if hidden_task_type == "counting":
            # Counting uses sentence acrostic with digit mapping
            X = state.metadata.get("X", 1)
            Y = state.metadata.get("Y", 10)
            I = state.metadata.get("I", 1)
            digit_to_letter = state.metadata.get("digit_to_letter", {})

            # Get topic dict with examples
            topic_name = state.metadata.get("topic", "sushi")
            topic_question = state.metadata.get("cover_question", "")
            topic_example_1_3 = state.metadata.get("example_1_3", "")
            topic_example_8_10 = state.metadata.get("example_8_10", "")

            topic = {
                "topic": topic_name,
                "question": topic_question,
                "example_1_3": topic_example_1_3,
                "example_8_10": topic_example_8_10,
            }

            # For counting, we use a single user message (no system prompt)
            prompt = get_counting_prompts(X, Y, I, topic, digit_to_letter)
            state.messages = [ChatMessageUser(content=prompt)]

            # Store prompts in metadata for scorers
            state.metadata["encoder_system_prompt"] = ""
            state.metadata["encoder_user_prompt"] = prompt
        else:
            # Standard stego CoT task
            system_prompt, user_prompt = get_stego_cot_prompts(
                algorithm=algorithm,
                cover_question=cover_question,
                hidden_problem=hidden_problem,
                use_mapping=use_mapping,
                framing=framing,
            )

            state.messages = [
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt),
            ]

            # Store prompts in metadata for scorers
            state.metadata["encoder_system_prompt"] = system_prompt
            state.metadata["encoder_user_prompt"] = user_prompt

        # Store algorithm in metadata for scorers
        state.metadata["algorithm"] = algorithm
        state.metadata["framing"] = framing

        # Note: We don't call generate() here - it will be called by
        # generate_with_no_reasoning() in the solver chain to apply
        # model-specific reasoning configuration

        return state

    return solve


@solver
def stego_message_solver(
    algorithm: str = "word_acrostic",
    topic: str = "nature",
    use_mapping: bool = False,
) -> Solver:
    """
    Solver for pure message encoding task (no hidden reasoning).

    This tests encoding capability without the math problem.

    Args:
        algorithm: The encoding algorithm to use
        topic: Topic to write about

    Returns:
        A Solver that generates encoded messages
    """
    from stego_inspect.prompts.encoder import (
        get_message_encoding_prompt,
        get_baseline_acrostic_prompt,
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Target string is in the input
        target_string = state.metadata.get("target_string", state.input_text)

        if state.metadata.get("baseline_acrostic"):
            # Legacy baseline acrostic: real cover question, no system prompt.
            letters = state.metadata.get(
                "expected_letters", list(target_string)
            )
            cover_question = state.metadata.get("cover_question", "")
            user_prompt = get_baseline_acrostic_prompt(letters, cover_question)
            state.messages = [ChatMessageUser(content=user_prompt)]
            state.metadata["encoder_system_prompt"] = ""
            state.metadata["encoder_user_prompt"] = user_prompt
        else:
            system_prompt, user_prompt = get_message_encoding_prompt(
                target_string=target_string,
                algorithm=algorithm,
                topic=topic,
                use_mapping=use_mapping,
            )

            state.messages = [
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt),
            ]

        state.metadata["algorithm"] = algorithm
        state.metadata["target_string"] = target_string

        # Note: generate() will be called by generate_with_no_reasoning() in the chain

        return state

    return solve
