"""OpenRouter text-completions provider that prefills past the reasoning slot.

Why this exists
---------------
The counting/stego cells published so far all ran with the target model's
internal reasoning **off** (Anthropic: extended thinking off by default; GPT-5.x:
`reasoning_effort="none"`). That is load-bearing: the claim under test is that
the work happens in the *visible* cover text, so a hidden reasoning channel lets
a model compute the answer privately and merely transcribe it.

Neither open-weight model we want to add can be run that way through a chat API
(verified against OpenRouter 2026-09-16):

* `openai/gpt-oss-20b` — `reasoning={"enabled": false}` is rejected outright with
  HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled".
  `reasoning={"effort": "low"}` only shrinks the analysis channel (~31 -> ~5-13
  tokens), it does not remove it.
* `qwen/qwen3.5-9b` — `reasoning={"enabled": false}` does work, but `/no_think`,
  `chat_template_kwargs` and assistant-message prefill are all silently ignored.

Passing a trailing assistant message to `/chat/completions` does **not** prefill
either model: the message is dropped and the model answers from scratch with
reasoning intact.

What does work is the raw `/completions` endpoint: we render the chat template
ourselves and cut the prompt off *after* the point where reasoning would go —
`<|channel|>final<|message|>` for gpt-oss (harmony), a pre-closed `<think>` block
for Qwen (ChatML). The model then has no syntactic slot to reason in, and the
API reports `reasoning_tokens=0`. This is a structural guarantee, not a request
the provider may ignore.

Not every OpenRouter backend honours a raw prompt — several re-apply their own
chat template, which burns reasoning tokens and (for Qwen) returns an **empty**
string, a failure that would otherwise score as an ordinary miss. So each model
carries an allow-list of backends verified to honour the prefill, tried in order,
and every response is checked before it is returned (see `_validate`).

Provider support, measured 2026-09-16 (4-sentence acrostic probe, temp 0):

    gpt-oss-20b   OK : DeepInfra(bf16) CoreWeave(fp4) Parasail(fp4) Novita(fp4) Phala
                  BAD: AkashML(rt=256) SiliconFlow(rt=369) Groq(rt=264) Together(400)
    qwen3.5-9b    OK : Parasail(bf16) Together
                  BAD: SiliconFlow DeepInfra Darkbloom Venice  (all rt>0, empty text)

Usage:

    inspect eval ... --model prefill/openai/gpt-oss-20b
    inspect_eval(task, model="prefill/qwen/qwen3.5-9b")

The served backend and its reported `reasoning_tokens` are recorded on every
`ModelOutput.metadata`, so a run can be audited after the fact rather than
trusted.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.model._model_call import ModelCall
from inspect_ai.tool import ToolChoice, ToolInfo

OPENROUTER_COMPLETIONS_URL = "https://openrouter.ai/api/v1/completions"
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"

# Default output cap. A D=8 counting response needs ~8-12 carrier sentences;
# 2048 leaves generous headroom without letting a derailed generation run away.
DEFAULT_MAX_TOKENS = 2048


# ---- Chat templates --------------------------------------------------------
#
# Each entry renders a message list into the model's raw prompt string, stopping
# immediately after the token sequence that opens the *post-reasoning* turn.


def _render_harmony(messages: list[ChatMessage]) -> str:
    """gpt-oss / harmony. Opening the `final` channel skips `analysis` entirely."""
    system_text = "\n".join(
        _text(m) for m in messages if isinstance(m, ChatMessageSystem)
    )
    parts = [
        "<|start|>system<|message|>"
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Reasoning: low\n\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.<|end|>"
    ]
    if system_text:
        parts.append(f"<|start|>developer<|message|># Instructions\n\n{system_text}<|end|>")
    for m in messages:
        if isinstance(m, ChatMessageUser):
            parts.append(f"<|start|>user<|message|>{_text(m)}<|end|>")
        elif isinstance(m, ChatMessageAssistant):
            parts.append(f"<|start|>assistant<|channel|>final<|message|>{_text(m)}<|end|>")
    parts.append("<|start|>assistant<|channel|>final<|message|>")
    return "".join(parts)


def _render_chatml_closed_think(messages: list[ChatMessage]) -> str:
    """Qwen3.5 / ChatML. A pre-closed <think> block leaves no room to think."""
    parts = []
    for m in messages:
        if isinstance(m, ChatMessageSystem):
            parts.append(f"<|im_start|>system\n{_text(m)}<|im_end|>\n")
        elif isinstance(m, ChatMessageUser):
            parts.append(f"<|im_start|>user\n{_text(m)}<|im_end|>\n")
        elif isinstance(m, ChatMessageAssistant):
            parts.append(f"<|im_start|>assistant\n{_text(m)}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def _text(message: ChatMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(getattr(c, "text", "") for c in content)


class _Template:
    def __init__(self, render, stop: list[str], providers: list[str]):
        self.render = render
        self.stop = stop
        self.providers = providers


# Keyed by substring of the OpenRouter model id.
TEMPLATES: dict[str, _Template] = {
    "gpt-oss": _Template(
        render=_render_harmony,
        stop=["<|return|>", "<|end|>", "<|start|>"],
        # bf16 first so the default route is the unquantized one.
        providers=["DeepInfra", "CoreWeave", "Novita", "Phala", "Parasail"],
    ),
    "qwen3.5": _Template(
        render=_render_chatml_closed_think,
        stop=["<|im_end|>"],
        # Together first: Parasail is bf16 and would be preferable, but it has
        # returned 429 on every attempt since 2026-09-16, so leading with it
        # only buys a wasted round-trip per sample. It stays as fallback.
        providers=["Together", "Parasail"],
    ),
}


def _template_for(model_name: str) -> _Template:
    lowered = model_name.lower()
    for key, tmpl in TEMPLATES.items():
        if key in lowered:
            return tmpl
    raise ValueError(
        f"No prefill template registered for {model_name!r}. "
        f"Known: {', '.join(sorted(TEMPLATES))}. Add one only after verifying "
        "against the live API that the prefill yields reasoning_tokens=0."
    )


class PrefillValidationError(Exception):
    """The backend ignored the prefill (re-templated the prompt)."""


@modelapi(name="prefill")
def prefill() -> type[ModelAPI]:
    return PrefillOpenRouterAPI


class PrefillOpenRouterAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig = GenerateConfig(),
        providers: list[str] | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[OPENROUTER_API_KEY] + list(api_key_vars or []),
            config=config,
        )
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY)
        if not self.api_key:
            raise RuntimeError(f"{OPENROUTER_API_KEY} is not set")
        self.url = base_url or OPENROUTER_COMPLETIONS_URL
        self.template = _template_for(model_name)
        # `-M providers=...` overrides the verified allow-list; order is honoured.
        self.providers = list(providers) if providers else list(self.template.providers)
        self.model_args = model_args

    def max_tokens(self) -> int | None:
        return DEFAULT_MAX_TOKENS

    def connection_key(self) -> str:
        return f"prefill/{self.model_name}"

    def should_retry(self, ex: Exception) -> bool:
        if isinstance(ex, httpx.HTTPStatusError):
            return ex.response.status_code in (408, 409, 429, 500, 502, 503, 504)
        return isinstance(ex, (httpx.TimeoutException, httpx.ConnectError))

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        if tools:
            raise NotImplementedError("prefill provider does not support tools")

        prompt = self.template.render(input)
        body: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": config.max_tokens or DEFAULT_MAX_TOKENS,
            "stop": self.template.stop,
        }
        for field in ("temperature", "top_p", "seed"):
            value = getattr(config, field, None)
            if value is not None:
                body[field] = value
        # NOTE: config.reasoning_effort is deliberately ignored — the prefill is
        # what disables reasoning, and the /completions route takes no reasoning
        # parameters. See stego_inspect/utils.get_reasoning_effort.

        attempts: list[dict[str, Any]] = []
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=config.timeout or 180) as client:
            for provider in self.providers:
                request = dict(
                    body, provider={"only": [provider], "allow_fallbacks": False}
                )
                try:
                    response = await client.post(
                        self.url,
                        json=request,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                except httpx.HTTPStatusError as ex:
                    attempts.append(
                        {"provider": provider, "error": f"HTTP {ex.response.status_code}",
                         "detail": ex.response.text[:300]}
                    )
                    continue

                try:
                    text, usage, served, reasoning_tokens = self._validate(payload)
                except PrefillValidationError as ex:
                    attempts.append({"provider": provider, "error": str(ex)})
                    continue

                output = ModelOutput.from_content(
                    model=self.model_name,
                    content=text,
                    stop_reason=_stop_reason(payload),
                )
                output.usage = usage
                output.time = time.monotonic() - start
                output.metadata = {
                    "prefill": True,
                    "served_by": served,
                    "reasoning_tokens_reported": reasoning_tokens,
                    "rejected_providers": attempts,
                }
                return output, ModelCall.create(request=request, response=payload)

        raise RuntimeError(
            f"prefill/{self.model_name}: no allow-listed backend honoured the "
            f"prefill. Attempts: {attempts}"
        )

    def _validate(self, payload: dict) -> tuple[str, ModelUsage, str | None, int | None]:
        """Reject a response that shows the backend re-applied its own template.

        Two symptoms, both seen in the wild: nonzero `reasoning_tokens` (the model
        got an analysis channel after all) and empty text (Qwen backends emit the
        whole answer into a reasoning field that the completions schema drops).
        Either one silently corrupts a cell, so neither is allowed through.
        """
        served = payload.get("provider")
        choices = payload.get("choices") or []
        if not choices:
            raise PrefillValidationError("no choices in response")
        text = choices[0].get("text") or ""

        usage_raw = payload.get("usage") or {}
        details = usage_raw.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")

        if reasoning_tokens:
            raise PrefillValidationError(
                f"backend reported reasoning_tokens={reasoning_tokens} despite the "
                "prefill — it re-applied its own chat template"
            )
        if not text.strip():
            raise PrefillValidationError("empty completion text")

        usage = ModelUsage(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            reasoning_tokens=reasoning_tokens,
        )
        return text, usage, served, reasoning_tokens


def _stop_reason(payload: dict) -> str:
    finish = (payload.get("choices") or [{}])[0].get("finish_reason")
    if finish == "length":
        return "max_tokens"
    if finish == "content_filter":
        return "content_filter"
    return "stop"
