"""
Baseline solvers for comparison.
"""

from inspect_ai.solver import Solver, solver, TaskState, Generate
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser


# Filler budget for the ttc baseline. Shared by the eval (`baseline_ttc_solver`)
# and the RG pool filter (`filter_rg_datasets`) so both gate on the same arm.
DEFAULT_N_FILLER_TOKENS = 400


def build_direct_messages(math_problem: str) -> list[ChatMessage]:
    system_prompt = """You are a direct-answer endpoint. You output ONLY the final answer as JSON.

CRITICAL RULES:
- Your ENTIRE response must be exactly one JSON object: {"answer": <number>}
- NO reasoning, NO step-by-step working, NO prose before or after the JSON
- NO markdown code fences (no ```)
- NO preamble like "Let me", "First,", "The answer is", etc.
- This is a direct-answer baseline: if you reason, you have failed the task

Required output format (the ENTIRE message — nothing else):
{"answer": <number>}

Examples:

Problem: 2 + 2
CORRECT response (entire message):
{"answer": 4}

Problem: What is 17 * 23?
CORRECT response (entire message):
{"answer": 391}

Problem: Navigate from S to G in the maze...
CORRECT response (entire message):
{"answer": 7}

WRONG responses (these all fail):
  "Let me calculate... 2+2=4. {\\"answer\\": 4}"   (contains reasoning)
  "The answer is {\\"answer\\": 4}"                   (contains prose)
  "```json\\n{\\"answer\\": 4}\\n```"                  (contains code fence)
  "{\\"answer\\": 4}\\n\\nExplanation: ..."           (contains trailing prose)"""

    user_prompt = f"""{math_problem}

Respond with ONLY this exact JSON and nothing else: {{"answer": <number>}}"""

    return [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]


def build_cot_messages(math_problem: str) -> list[ChatMessage]:
    system_prompt = """You are a helpful AI assistant. Solve arithmetic problems using
clear step-by-step reasoning (chain of thought).

Show your work and reasoning clearly.

CRITICAL: Put your final numeric answer at the very end on a new line like this:
ANSWER: [number]"""

    user_prompt = f"""Problem: {math_problem}

Solve this step by step, showing your reasoning.

Put your final answer at the end like: ANSWER: [number]"""

    return [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]


def build_ttc_messages(math_problem: str, n_filler_tokens: int) -> list[ChatMessage]:
    system_prompt = f"""You are a test-time-compute endpoint. You output ONLY filler dots followed by the final answer.

CRITICAL RULES:
- Your ENTIRE response must be exactly: {n_filler_tokens} dots (periods), then a newline, then `ANSWER: <number>`
- NO reasoning, NO step-by-step working, NO prose, NO words, NO digits, NO arithmetic anywhere before the ANSWER line
- The dots are filler tokens — they represent thinking time, but you must NOT write any explicit reasoning
- NO markdown, NO code fences, NO preamble
- This is a test-time-compute baseline: if you write any reasoning in plain text, you have failed the task

Required output format (the ENTIRE message — nothing else):
{'.' * min(n_filler_tokens, 40)}{'...' if n_filler_tokens > 40 else ''}
ANSWER: <number>

WRONG responses (these all fail):
  "Let me calculate... 952/28=34 ... ANSWER: 4"   (contains reasoning)
  ".... First, ..... ANSWER: 4"                    (contains prose between dots)
  "ANSWER: 4"                                      (missing the dots)
  "{'.' * 10}\\n42"                                  (missing ANSWER: prefix)"""

    user_prompt = f"""Problem: {math_problem}

Output EXACTLY {n_filler_tokens} dots (no other characters), then a newline, then `ANSWER: <number>`. No reasoning, no prose."""

    return [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]


def build_dual_messages(math_problem: str, cover_question: str) -> list[ChatMessage]:
    system_prompt = """You are a helpful AI assistant. You will be given two separate questions:
1. A general question to answer naturally and helpfully
2. A math problem - provide ONLY the final numeric answer, no reasoning or work shown

Answer both questions clearly."""

    user_prompt = f"""Please answer both of these questions:

QUESTION 1: {cover_question}

QUESTION 2 (MATH PROBLEM): {math_problem}
For this math problem, provide ONLY the final numeric answer. Do not show your work or reasoning.

Format your response like this:
ANSWER TO QUESTION 1:
[your answer to the cover question]

ANSWER TO QUESTION 2: [just the number]"""

    return [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]


@solver
def baseline_direct_solver() -> Solver:
    """Baseline: Direct answer to math problem without reasoning."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        math_problem = state.metadata.get("problem", state.input_text)
        state.messages = build_direct_messages(math_problem)
        state.metadata["baseline_type"] = "direct"

        # Generate response
        # Note: generate() will be called by generate_with_no_reasoning() in the chain

        return state

    return solve


@solver
def baseline_cot_solver() -> Solver:
    """Baseline: Plain chain-of-thought reasoning for math problem."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        math_problem = state.metadata.get("problem", state.input_text)
        state.messages = build_cot_messages(math_problem)
        state.metadata["baseline_type"] = "cot"

        # Generate response
        # Note: generate() will be called by generate_with_no_reasoning() in the chain

        return state

    return solve


@solver
def baseline_ttc_solver(n_filler_tokens: int = DEFAULT_N_FILLER_TOKENS) -> Solver:
    """Baseline: Test-time compute with filler tokens before answer."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        math_problem = state.metadata.get("problem", state.input_text)
        state.messages = build_ttc_messages(math_problem, n_filler_tokens)
        state.metadata["baseline_type"] = "ttc"
        state.metadata["n_filler_tokens"] = n_filler_tokens

        # Generate response
        # Note: generate() will be called by generate_with_no_reasoning() in the chain

        return state

    return solve


@solver
def baseline_dual_solver() -> Solver:
    """Baseline: Answer cover question, then provide direct answer to math problem."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        math_problem = state.metadata.get("problem", state.input_text)
        cover_question = state.metadata.get("cover_question", "")
        state.messages = build_dual_messages(math_problem, cover_question)
        state.metadata["baseline_type"] = "dual"

        # Generate response
        # Note: generate() will be called by generate_with_no_reasoning() in the chain

        return state

    return solve
