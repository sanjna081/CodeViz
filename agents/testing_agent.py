# testing_agent.py - VERSION 5
# Changes from V4:
#
#  1. No functional changes — this file was already compatible with the
#     OpenRouter/LLMClient backend introduced in main.py V4.
#
#     Reason: the only LLM call in this file is in _request_fix(), which
#     imports and calls call_llm() from llm_utils locally at the point of
#     use (not at module level). llm_utils V2 probes .complete() first, so
#     LLMClient is fully supported without any changes here.
#     There are no direct self.llm.invoke() calls, no langchain imports,
#     and no ChatOllama references anywhere in this file.
#
#  2. Version bump to V5 for consistency with the rest of the V5 stack
#     (main V4, llm_utils V2, planner V5, analyzer V3, analytical V5).
#
#  Note on LLM call frequency:
#     The LLM in this agent is only called during error recovery — one call
#     per failed attempt, up to max_retries times. On a clean pipeline run
#     where all generated code executes correctly, this agent makes zero
#     LLM calls.

import re
import contextlib
import io
import inspect
from typing import Dict, Any, Callable, Optional, List
import traceback
import pandas as pd

FENCED_CODE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
IMPORT_LINE_RE = re.compile(
    r'^\s*(?:from\s+[A-Za-z0-9_\.]+\s+import\s+.+|import\s+[A-Za-z0-9_\.]+(?:\s+as\s+[A-Za-z0-9_]+)?)\s*$',
    re.IGNORECASE | re.MULTILINE
)


# ══════════════════════════════════════════════════════════════════════════════
#  CODE EXTRACTION HELPERS  (unchanged from V4)
# ══════════════════════════════════════════════════════════════════════════════

def extract_fenced_code(text: str) -> str | None:
    m = FENCED_CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def gather_import_lines(text: str) -> str:
    imports = []
    seen = set()
    for ln in text.splitlines():
        if IMPORT_LINE_RE.match(ln):
            line = ln.strip()
            if line and line not in seen and not line.startswith('#'):
                seen.add(line)
                imports.append(line)
    if imports:
        return "\n".join(imports) + "\n\n"
    return ""


def extract_solution_and_imports(raw: str) -> str:
    fenced = extract_fenced_code(raw)
    imports_block = gather_import_lines(raw)

    if fenced:
        if IMPORT_LINE_RE.search(fenced):
            return fenced
        else:
            return imports_block + fenced

    idx = raw.find("def solution")
    if idx != -1:
        func_block = raw[idx:].strip()
        return imports_block + func_block

    code = raw.strip()
    wrapped = wrap_into_solution(code)
    return imports_block + wrapped


def wrap_into_solution(code: str) -> str:
    body_lines = code.splitlines()
    while body_lines and body_lines[0].strip() == "":
        body_lines.pop(0)
    while body_lines and body_lines[-1].strip() == "":
        body_lines.pop()
    indented = "\n".join("    " + ln for ln in body_lines)
    wrapper = f"def solution(df, context={{}}):\n{indented}\n"
    if "return" not in code:
        wrapper += "    return df\n"
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
#  SOLUTION DISPATCHER  (unchanged from V4)
# ══════════════════════════════════════════════════════════════════════════════

def _call_solution(fn, df: pd.DataFrame, context: dict):
    try:
        sig_params = inspect.signature(fn).parameters
        if len(sig_params) >= 2:
            return fn(df, context)
        else:
            return fn(df)
    except (ValueError, TypeError):
        try:
            return fn(df, context)
        except TypeError:
            return fn(df)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTING AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestingAgent:
    """
    Non-sandboxed testing agent for local dev.

    V4 changes (retained in V5):
      - fix_reason extended with 'wrong_context_type' for type mismatch failures.
      - run_solution() accepts optional `produces` list so _request_fix() can
        enumerate required context writes explicitly rather than parsing the
        error message.
      - _request_fix() 'missing_context_writes' diagnostic block now surfaces
        the exact validator error rather than generic hardcoded examples.
      - _request_fix() new 'wrong_context_type' diagnostic block for cases
        where a key was written but with the wrong type.
      - All other logic (retry loop, _try_run, extraction helpers) unchanged.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    # ── main entry point ──────────────────────────────────────────────────────

    def run_solution(self, raw_code: str, df: pd.DataFrame,
                     llm=None,
                     context: dict = None,
                     context_validator: Optional[Callable[[dict], None]] = None,
                     produces: Optional[List[str]] = None,
                     ) -> Dict[str, Any]:
        """
        Extract, compile, and execute the generated solution function.

        Args:
            raw_code:           Raw LLM output (may contain markdown fences).
            df:                 Input DataFrame passed to solution().
            llm:                LLM instance used for retry-with-fix.
                                If None, no retries are attempted.
            context:            Shared pipeline context dict passed by reference.
            context_validator:  Optional callable (context) -> None.
                                Should raise RuntimeError on validation failure.
                                Treated as a retriable error inside the loop.
            produces:           Optional list of 'key:type' strings from
                                task_metadata['produces'] (planner V5).
                                When provided, _request_fix() uses this list
                                to enumerate required writes explicitly rather
                                than relying solely on the error message text.

        Returns:
            On success: {'ok': True,  'result': ..., 'stdout': str,
                         'wrapped_code': str, 'attempts': int}
            On failure: {'ok': False, 'error': str, 'stage': str,
                         'traceback': str, 'wrapped_code': str,
                         'stdout': str, 'attempts': int}
        """
        if context is None:
            context = {}

        current_code = raw_code
        last_output  = None

        for attempt in range(1, self.max_retries + 2):
            print(f"[TestingAgent] Attempt {attempt}/{self.max_retries + 1}")

            output = self._try_run(current_code, df, context)
            output['attempts'] = attempt

            # ── run context validator when execution succeeded ────────────────
            if output['ok'] and context_validator is not None:
                try:
                    context_validator(context)
                except RuntimeError as validation_error:
                    error_msg = str(validation_error)
                    print(
                        f"[TestingAgent] ✗ Context validation failed: "
                        f"{error_msg[:200]}"
                    )

                    # Distinguish missing keys from wrong-type keys so
                    # _request_fix() can give the LLM the right diagnosis.
                    fix_reason = _classify_validation_error(error_msg)

                    output = {
                        'ok':           False,
                        'stage':        'context_validation',
                        'error':        error_msg,
                        'traceback':    '',
                        'wrapped_code': output.get('wrapped_code', current_code),
                        'stdout':       output.get('stdout', ''),
                        'attempts':     attempt,
                        '_fix_reason':  fix_reason,
                    }
            # ─────────────────────────────────────────────────────────────────

            if output['ok']:
                print(f"[TestingAgent] ✓ Success on attempt {attempt}")
                return output

            last_output = output

            if llm is None or attempt > self.max_retries:
                break

            fix_reason = output.get('_fix_reason', 'runtime_error')
            print(f"[TestingAgent] ✗ Failed ({fix_reason}): {output['error'][:120]}")
            print(f"[TestingAgent] Requesting LLM fix (attempt {attempt})...")

            current_code = self._request_fix(
                llm,
                broken_code=output.get('wrapped_code', current_code),
                error=output['error'],
                traceback_str=output.get('traceback', ''),
                df=df,
                context=context,
                fix_reason=fix_reason,
                produces=produces,
            )

        print(f"[TestingAgent] ✗ All attempts failed.")
        return last_output

    # ── single execution attempt  (unchanged from V4) ─────────────────────────

    def _try_run(self, raw_code: str, df: pd.DataFrame,
                 context: dict) -> Dict[str, Any]:
        try:
            to_exec = extract_solution_and_imports(raw_code)
        except Exception as e:
            return {
                'ok':           False,
                'stage':        'extract',
                'error':        str(e),
                'traceback':    traceback.format_exc(),
                'raw_code':     raw_code,
                'wrapped_code': None,
                'stdout':       '',
            }

        module_ns: Dict[str, Any] = {
            'pd':           pd,
            '__builtins__': __builtins__,
        }
        for lib_name, alias in [
            ('numpy',             'np'),
            ('matplotlib.pyplot', 'plt'),
            ('seaborn',           'sns'),
            ('json',              'json'),
            ('sklearn',           'sklearn'),
        ]:
            try:
                import importlib
                module_ns[alias] = importlib.import_module(lib_name)
            except ImportError:
                pass

        stdout_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf):
                compiled = compile(to_exec, '<solution>', 'exec')
                exec(compiled, module_ns, module_ns)
        except SyntaxError as e:
            return {
                'ok':           False,
                'stage':        'syntax_error',
                'error':        f"Syntax error at line {e.lineno}: {e.msg}",
                'traceback':    traceback.format_exc(),
                'wrapped_code': to_exec,
                'stdout':       stdout_buf.getvalue(),
            }
        except ImportError as e:
            return {
                'ok':           False,
                'stage':        'import_error',
                'error':        f"Import failed: {str(e)}",
                'traceback':    traceback.format_exc(),
                'wrapped_code': to_exec,
                'stdout':       stdout_buf.getvalue(),
                'hint':         'Check if the imported module/function exists in the library.',
            }
        except Exception as e:
            return {
                'ok':           False,
                'stage':        'compile_exec',
                'error':        str(e),
                'traceback':    traceback.format_exc(),
                'wrapped_code': to_exec,
                'stdout':       stdout_buf.getvalue(),
            }

        fn = module_ns.get('solution')
        if fn is None or not callable(fn):
            return {
                'ok':             False,
                'stage':          'no_solution',
                'error':          "No callable solution(df) found after exec",
                'wrapped_code':   to_exec,
                'stdout':         stdout_buf.getvalue(),
                'namespace_keys': list(module_ns.keys()),
            }

        try:
            with contextlib.redirect_stdout(stdout_buf):
                result = _call_solution(fn, df.copy(), context)
            return {
                'ok':           True,
                'result':       result,
                'stdout':       stdout_buf.getvalue(),
                'wrapped_code': to_exec,
            }
        except Exception as e:
            return {
                'ok':           False,
                'stage':        'runtime',
                'error':        str(e),
                'traceback':    traceback.format_exc(),
                'wrapped_code': to_exec,
                'stdout':       stdout_buf.getvalue(),
            }

    # ── context description helper  (unchanged from V4) ───────────────────────

    @staticmethod
    def _build_context_desc(context: dict) -> str:
        if not context:
            return "Context is empty (no prior pipeline state)."

        lines = []
        for k, v in context.items():
            type_name = type(v).__name__
            if hasattr(v, 'shape'):
                lines.append(f"  context['{k}'] → {type_name}, shape={v.shape}")
            elif hasattr(v, '__len__') and not isinstance(v, str):
                lines.append(f"  context['{k}'] → {type_name}, len={len(v)}")
            elif isinstance(v, str):
                preview = v[:60] + '...' if len(v) > 60 else v
                lines.append(f"  context['{k}'] → str, value='{preview}'")
            else:
                lines.append(f"  context['{k}'] → {type_name}")
        return "Available context keys:\n" + "\n".join(lines)

    # ── LLM-assisted fix  (V4: targeted diagnostics per failure type) ─────────

    def _request_fix(self, llm, broken_code: str, error: str,
                     traceback_str: str, df: pd.DataFrame,
                     context: dict,
                     fix_reason: str = 'runtime_error',
                     produces: Optional[List[str]] = None) -> str:
        """
        Ask the LLM to fix a broken solution() function.

        V4 changes (retained in V5):
          - 'missing_context_writes' block surfaces the exact validator error
            rather than generic hardcoded examples. If produces is provided,
            required writes are listed explicitly.
          - New 'wrong_context_type' block for type mismatch failures where
            a key was written but with the wrong type.
          - 'runtime_error' block unchanged.
        """
        from llm_utils import call_llm

        context_desc = self._build_context_desc(context)
        col_info     = ", ".join(f"'{c}'" for c in df.columns)

        # ── build the diagnostic block for this specific failure type ─────────

        if fix_reason == 'missing_context_writes':
            produces_block = ""
            if produces:
                lines = ["Required context writes for this task:"]
                for key_hint in produces:
                    key  = key_hint.split(':')[0]
                    hint = key_hint.split(':', 1)[1] if ':' in key_hint else ''
                    type_note = f"  # must be {hint}" if hint else ""
                    lines.append(f"    context['{key}'] = {key}{type_note}")
                produces_block = "\n".join(lines)

            diagnostic_block = f"""
{"=" * 60}
PROBLEM: MISSING CONTEXT WRITES
{"=" * 60}
The function ran successfully but did NOT store required artifacts
in the context dict. The pipeline cannot continue without them.

{error}
{f"{chr(10)}{produces_block}" if produces_block else ""}
Add the missing assignments INSIDE the function body BEFORE the
return statement. Do NOT change the return value.
"""

        elif fix_reason == 'wrong_context_type':
            produces_block = ""
            if produces:
                lines = ["Expected context types for this task:"]
                for key_hint in produces:
                    key  = key_hint.split(':')[0]
                    hint = key_hint.split(':', 1)[1] if ':' in key_hint else ''
                    if hint:
                        lines.append(f"    context['{key}']  →  must be {hint}")
                produces_block = "\n".join(lines)

            diagnostic_block = f"""
{"=" * 60}
PROBLEM: WRONG CONTEXT VALUE TYPE
{"=" * 60}
The function stored a context key but with the wrong type. The
downstream task cannot use it and will fail.

{error}
{f"{chr(10)}{produces_block}" if produces_block else ""}
Fix the assignment so the stored value has the correct type.
Common causes:
  - Storing None instead of a fitted object (forgot to fit/transform)
  - Storing a list instead of a numpy array (wrap with np.array())
  - Storing a DataFrame when ndarray is expected (use .values)
  - Calling fit() but not storing the result before return
"""

        else:
            # Standard runtime / syntax / import error — show error + traceback.
            tb_section = f"""
{"=" * 60}
TRACEBACK
{"=" * 60}
{traceback_str}
""" if traceback_str.strip() else ""

            diagnostic_block = f"""
{"=" * 60}
ERROR
{"=" * 60}
{error}
{tb_section}"""

        # ── assemble the full fix prompt ──────────────────────────────────────
        prompt = f"""You are a Python debugging expert. The following solution() function
failed. Fix the bug and return only the corrected function.

{"=" * 60}
BROKEN CODE
{"=" * 60}
{broken_code}
{diagnostic_block}
{"=" * 60}
DATASET INFORMATION
{"=" * 60}
Columns : {col_info}
Shape   : {df.shape[0]} rows × {df.shape[1]} columns

{"=" * 60}
PIPELINE CONTEXT
{"=" * 60}
{context_desc}

{"=" * 60}
INSTRUCTIONS
{"=" * 60}
- Fix ONLY what is broken. Do not restructure unnecessarily.
- Function signature must be: def solution(df, context={{}}):
- Use context.get('key') to read pipeline state (never context['key'])
- Store required artifacts in context BEFORE the return statement
- Return the same type/shape as the original function intended
- Import all libraries inside the function
- Write ONLY the corrected Python function — no markdown, no explanation.

Start with: def solution(df, context={{}}):
"""
        return call_llm(llm, prompt)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPER  (unchanged from V4)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_validation_error(error_msg: str) -> str:
    """
    Classify a RuntimeError from _validate_produces() into one of:
      'missing_context_writes' — required keys were not stored at all
      'wrong_context_type'     — keys were stored but with wrong type

    Used by run_solution() to set the _fix_reason tag so _request_fix()
    gives the LLM the right targeted diagnostic.
    """
    msg_lower = error_msg.lower()
    if 'wrong type' in msg_lower or 'expected' in msg_lower and 'got' in msg_lower:
        return 'wrong_context_type'
    return 'missing_context_writes'