from typing import Any, Dict


def get_reasoning_effort(model: str) -> Dict[str, Any]:
    """Get reasoning effort parameter based on model type to disable/minimize internal reasoning.

    Based on official provider documentation:

    Gemini (https://ai.google.dev/gemini-api/docs/thinking):
    - Gemini 2.0 Flash: Experimental thinking, can disable with thinkingBudget=0
    - Gemini 2.5 Flash: Can disable with thinkingBudget=0 (LiteLLM: reasoning_effort="none")
    - Gemini 2.5 Pro: CANNOT disable thinking - it's mandatory
    - Gemini 3 Flash: "minimal" doesn't guarantee thinking is off
    - Gemini 3 Pro / 3.1 Pro: CANNOT disable thinking

    Claude/Anthropic (https://platform.claude.com/docs/en/build-with-claude/extended-thinking):
    - Extended thinking is DISABLED by default
    - Must explicitly enable with thinking: { "type": "enabled", "budget_tokens": ... }
    - Simply omitting the thinking parameter keeps thinking disabled
    - Supported: Claude 4.5 (Opus/Sonnet/Haiku), Claude 4 (Opus/Sonnet), Claude 3.7 Sonnet

    OpenAI (https://platform.openai.com/docs/guides/reasoning):
    - GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o: No reasoning capability, no parameter needed
    - GPT-5 (base): Default reasoning is "medium", minimum is "minimal" (cannot fully disable)
    - GPT-5.1, GPT-5.2, GPT-5-nano: Default is "none" (disabled), support "none" to keep disabled
    - o1, o3 series: Reasoning models, minimum is "low" (cannot fully disable)
    - GPT-6 (astra): Reasoning model, minimum is "low" (cannot fully disable);
      verified 2026-09-06 against the API, which rejects reasoning_effort="none"
    """
    params = {}
    model_lower = model.lower()

    # The `prefill/` provider disables reasoning structurally, by cutting the raw
    # prompt off after the point where the reasoning slot would open, so there is
    # no parameter to send — and the /completions route it uses accepts no
    # reasoning parameters at all. See stego_inspect/providers/prefill_openrouter.
    if model_lower.startswith("prefill/"):
        return params

    # Gemini models
    if "gemini" in model_lower:
        # Gemini 2.5 Pro, 3 Pro, and 3.x Pro: CANNOT disable thinking - it's mandatory
        if "2.5-pro" in model_lower or "3-pro" in model_lower or "3.1-pro" in model_lower:
            raise NotImplementedError(f"Cannot disable reasoning for {model} - thinking is mandatory")
        # Gemini 3 Flash: "minimal" doesn't guarantee thinking is off
        elif "3-flash" in model_lower:
            raise NotImplementedError(f"Cannot reliably disable reasoning for {model} - 'minimal' doesn't guarantee thinking is off")
        # Gemini 2.0 Flash and 2.5 Flash: Can disable with reasoning_effort="none"
        else:
            params["reasoning_effort"] = "none"

    # OpenAI GPT-5.1, GPT-5.2, GPT-5-nano: Default is "none", explicitly set for clarity
    elif (
        "gpt-5.1" in model_lower
        or "gpt-5.2" in model_lower
        or "gpt-5.4" in model_lower
        or "gpt-5-nano" in model_lower
    ):
        params["reasoning_effort"] = "none"

    # OpenAI GPT-6 (astra): reasoning is mandatory - minimum reasoning_effort is
    # "low" ("none" is rejected with an unsupported_value error), and even an
    # unspecified call spends reasoning tokens. Cannot be disabled.
    elif "gpt-6" in model_lower:
        raise NotImplementedError(
            f"Cannot disable reasoning for {model} - minimum reasoning_effort is 'low'"
        )

    # gpt-oss (harmony format): the analysis channel is structural, so reasoning
    # cannot be turned off at all - OpenRouter rejects reasoning={"enabled": false}
    # with 400 "Reasoning is mandatory for this endpoint and cannot be disabled"
    # (verified 2026-09-16). Prefilling a trailing assistant message does not
    # suppress it either. "low" is the floor: it cuts reasoning_tokens from ~31
    # to ~5-13 on a one-line arithmetic probe. Cells run this way are NOT
    # comparable to the reasoning-off closed-model rows - report reasoning_tokens
    # next to the rate and treat it as an upper bound.
    elif "gpt-oss" in model_lower:
        params["reasoning_effort"] = "low"

    # Qwen3.5 (hybrid thinking): reasoning IS fully disableable, but only through
    # OpenRouter's `reasoning={"enabled": false}` body field, which Inspect takes
    # as the `reasoning_enabled` MODEL ARG (-M reasoning_enabled=false), not as a
    # GenerateConfig field - so there is nothing to return here. Verified
    # 2026-09-16: reasoning_tokens=0 on 3/3 repeats with it, ~194-300 without.
    # `/no_think` and chat_template_kwargs are ignored by the OpenRouter route.
    elif "qwen3.5" in model_lower:
        pass

    # Qwen3.8 served by vLLM: the chat template has a native enable_thinking=false
    # branch that renders an empty `<think>\n\n</think>\n\n` block - the same cut
    # the prefill/ provider makes, applied server-side - and vLLM honours
    # `chat_template_kwargs`. OpenRouter backends silently ignore that field
    # (verified 2026-09-16 on Qwen3.5), which would leave reasoning ON with no
    # error, so it is only sent on the vllm/ route. Other routes raise, which
    # records `_reasoning_warning` on the sample instead of running silently.
    elif "qwen3.8" in model_lower:
        if model_lower.startswith("vllm/"):
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            raise NotImplementedError(
                f"Reasoning-off for {model} is only wired for vllm/ "
                "(chat_template_kwargs); use the prefill/ provider or a vLLM server"
            )

    # OpenAI GPT-5 (base): Default is "medium", minimum is "minimal" - cannot fully disable
    elif "gpt-5" in model_lower:
        raise NotImplementedError(f"Cannot fully disable reasoning for {model} - minimum is 'minimal'")

    # OpenAI o1, o3 series: Reasoning models, minimum is "low" - cannot fully disable
    elif "o1" in model_lower or "o3" in model_lower:
        raise NotImplementedError(f"Cannot disable reasoning for {model} - reasoning models require reasoning")

    # Gemma 3 and Llama 3.x/4: no reasoning mode at all - OpenRouter lists no
    # `reasoning` parameter on any of their endpoints (checked 2026-11-04), so
    # there is nothing to disable. Gemma 4 is hybrid-thinking and is NOT covered.
    elif "gemma-3" in model_lower or "llama-3" in model_lower or "llama-4" in model_lower:
        pass

    # Claude/Anthropic: Extended thinking is DISABLED by default
    # Simply don't pass the thinking parameter to keep it disabled
    elif "anthropic" in model_lower or "claude" in model_lower:
        pass  # No thinking parameter = thinking disabled by default

    # GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o: No reasoning capability, no parameter needed
    # Other models: Don't add reasoning parameters as they don't use reasoning by default

    return params
