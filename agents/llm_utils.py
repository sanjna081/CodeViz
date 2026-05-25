# llm_utils.py - VERSION 2
# Changes from V1:
#
#  1. .complete() added as the first probe [NEW — primary path]:
#     LLMClient (main.py V4) exposes .complete(prompt) -> str directly.
#     This is tried first because it always returns a plain string,
#     requiring no further unwrapping.
#
#  2. All V1 fallbacks retained in original order:
#     .invoke()  → unwraps .content if present  (LangChain / planner shim)
#     .run()     → LangChain LLM style
#     __call__   → plain callable
#     These remain so any LLM object that worked before still works,
#     and so the codebase stays model-agnostic: swap LLMClient for any
#     object that satisfies at least one probe and nothing else changes.

from typing import Any
import inspect


def call_llm(llm: Any, prompt: str) -> str:
    """
    Model-agnostic LLM caller. Probes the object for known call patterns
    in priority order and returns the first successful plain-string result.

    Priority:
      1. llm.complete(prompt)   → LLMClient (OpenRouter / any REST wrapper)
      2. llm.invoke(prompt)     → LangChain Runnable / ChatOllama / planner shim
      3. llm.run(prompt)        → LangChain LLM style
      4. llm(prompt)            → plain callable

    Returns:
        str — the model's reply.

    Raises:
        RuntimeError if none of the four patterns succeed.
    """

    # 1) .complete() — primary path for LLMClient
    #    Returns a plain string directly; no unwrapping needed.
    try:
        if hasattr(llm, "complete") and inspect.isroutine(llm.complete):
            return str(llm.complete(prompt))
    except Exception:
        pass

    # 2) .invoke() — LangChain Runnable / ChatOllama / _InvokeResponse shim
    #    May return an object with a .content attribute (AIMessage style),
    #    or a plain string — handle both.
    try:
        if hasattr(llm, "invoke") and inspect.isroutine(llm.invoke):
            res = llm.invoke(prompt)
            if hasattr(res, "content"):
                return str(res.content)
            return str(res)
    except Exception:
        pass

    # 3) .run() — LangChain LLM (non-chat) style
    try:
        if hasattr(llm, "run") and inspect.isroutine(llm.run):
            return str(llm.run(prompt))
    except Exception:
        pass

    # 4) Direct call — plain callable (e.g. a lambda or mock in tests)
    try:
        if callable(llm):
            return str(llm(prompt))
    except Exception:
        pass

    raise RuntimeError(
        "call_llm: could not call the LLM object with any known pattern "
        "(.complete / .invoke / .run / __call__). "
        f"Object type: {type(llm).__name__}"
    )