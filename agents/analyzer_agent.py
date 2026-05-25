# analyzer_agent.py - VERSION 3
# Changes from V2:
#
#  1. No functional changes — this file was already compatible with the
#     OpenRouter/LLMClient backend introduced in main.py V4.
#
#     Reason: the only LLM call in this file goes through call_llm()
#     from llm_utils.py (see _llm_route()). llm_utils V2 probes .complete()
#     first, so LLMClient is fully supported without any changes here.
#     There are no direct self.llm.invoke() calls, no langchain imports,
#     and no ChatOllama references anywhere in this file.
#
#  2. Version bump to V3 for consistency with the rest of the V5 stack
#     (main V4, llm_utils V2, planner V5, analytical V5, testing V5).
#
#  Note on the normal execution path:
#     When the planner produces structured task dicts (output_type set),
#     _llm_route() is never called at all — routing is a direct dict lookup.
#     The LLM is only involved in the legacy fallback path (raw string input),
#     which means this agent is effectively zero-LLM-calls in normal operation.

from typing import Dict, List, Union
import re
from llm_utils import call_llm


# Mirror of VISUALIZATION_KEYWORDS from planner_agent — kept in sync manually.
# Used only in the heuristic fallback when no output_type is available.
_VIZ_KEYWORDS = (
    'plot', 'chart', 'graph', 'visuali', 'histogram', 'scatter',
    'heatmap', 'bar chart', 'line chart', 'boxplot', 'box plot',
    'violin', 'distribution plot', 'draw', 'display chart',
)


class AnalyzerAgent:
    def __init__(self, llm):
        self.llm = llm

    # ══════════════════════════════════════════════════════════════════════════
    #  LEGACY INPUT CLEANING  (used only when raw string input is received)
    # ══════════════════════════════════════════════════════════════════════════

    def _clean_subtask_text(self, raw: str) -> List[str]:
        lines = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r'^[\-\*\•\u2022]\s*', '', ln)
            ln = re.sub(r'^\d+\s*[\)\.\-:]\s*', '', ln)
            ln = ln.strip().strip('*').strip('_')
            if ln:
                lines.append(ln)
        if len(lines) == 1 and (',' in lines[0] and len(lines[0].split(',')) <= 6):
            cand = [s.strip() for s in lines[0].split(',') if s.strip()]
            if len(cand) > 1:
                lines = cand
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    #  LEGACY LLM ROUTING  (only called when no output_type field is present)
    # ══════════════════════════════════════════════════════════════════════════

    def _llm_route(self, subtasks_for_llm: List[str]) -> Dict[int, str]:
        """
        Ask the LLM to assign agents to tasks by index.
        Returns {task_index: agent_name}.
        Only called on the legacy path (raw string or list-of-strings input).
        """
        prompt = (
            "You are a task router that assigns subtasks to specialized agents. "
            "Assign each subtask to exactly ONE of these two agents:\n\n"
            "1. analytical_agent: computation, analysis, aggregation, filtering, "
            "grouping, statistical summaries, ML models, any non-visual output.\n"
            "2. visualization_agent: any task whose output is a chart, plot, or graph.\n\n"
            "RULES:\n"
            "- Use ONLY: analytical_agent, visualization_agent\n"
            "- Output format: <task_number> -> <agent_name>\n"
            "- Mappings only, no explanations.\n\n"
            f"Subtasks:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtasks_for_llm))
            + "\n\nOutput:"
        )

        raw_reply = call_llm(self.llm, prompt)
        print("\n[Analyzer] LLM routing reply:")
        print(raw_reply)
        print("------ end reply ------\n")

        mapping = {}
        for ln in raw_reply.splitlines():
            ln = ln.strip()
            if '->' not in ln:
                continue
            left, right = ln.split('->', 1)
            num_match = re.search(r'\d+', left.strip())
            if num_match:
                idx = int(num_match.group()) - 1
                right_lower = right.strip().lower()
                if 'visualization' in right_lower:
                    mapping[idx] = 'visualization_agent'
                elif 'analytical' in right_lower:
                    mapping[idx] = 'analytical_agent'

        return mapping

    def _heuristic_route(self, subtasks_for_llm: List[str]) -> Dict[int, str]:
        """
        Keyword-based fallback routing when LLM produces no parseable output.
        Uses the same keyword set as the planner's _classify_task_output_type().
        """
        mapping = {}
        for idx, s in enumerate(subtasks_for_llm):
            s_low = s.lower()
            if any(kw in s_low for kw in _VIZ_KEYWORDS):
                mapping[idx] = 'visualization_agent'
            else:
                mapping[idx] = 'analytical_agent'
        return mapping

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def assign(self, raw_subtasks_output: Union[List[str], List[dict], str]) -> Dict[str, dict]:
        """
        Assigns agents to subtasks and builds the enriched mapping consumed by
        main.py's _execute_task().

        Input formats:
          - List[dict]  — structured task dicts from planner (normal path)
          - List[str]   — legacy string list (fallback path)
          - str         — raw planner LLM text (fallback path)

        Returns:
          Dict[task_description -> {
              'agent':                 'analytical_agent' | 'visualization_agent',
              'output_type':           'analytical' | 'visualization',
              'ml_step':               'split'|'scale'|'train'|'evaluate'|None,
              'requires':              [context keys needed before execution],
              'produces':              [context keys written after execution],
              'target_columns':        [...],
              'column_constraint':     bool,
              'all_mentioned_columns': [...],
          }]

        Routing decision priority:
          1. output_type field in task dict  → direct lookup, no LLM call
          2. LLM routing call                → only on legacy string input
          3. Keyword heuristic               → final fallback
        """

        # ── STEP 1: Normalise input ───────────────────────────────────────────
        subtasks      = []   # description strings
        metadata_list = []   # full task dicts

        if isinstance(raw_subtasks_output, list) and raw_subtasks_output:
            if isinstance(raw_subtasks_output[0], dict):
                # Normal path: structured dicts from planner V5
                subtasks      = [t['description'] for t in raw_subtasks_output]
                metadata_list = raw_subtasks_output
                print(f"[Analyzer] Received {len(subtasks)} structured tasks with metadata")
            else:
                # Legacy path: plain strings
                subtasks      = raw_subtasks_output
                metadata_list = [_empty_meta(t) for t in subtasks]
                print(f"[Analyzer] Received {len(subtasks)} legacy string tasks (no metadata)")
        else:
            # Raw string path
            subtasks      = self._clean_subtask_text(str(raw_subtasks_output))
            metadata_list = [_empty_meta(t) for t in subtasks]
            print(f"[Analyzer] Parsed {len(subtasks)} tasks from raw text")

        # ── STEP 2: Determine routing for each task ───────────────────────────
        # Normal path: output_type is set on every dict — no LLM needed.
        # Legacy path: call LLM, fall back to heuristic if LLM fails.

        all_have_output_type = all(
            isinstance(m, dict) and 'output_type' in m
            for m in metadata_list
        )

        if all_have_output_type:
            # Direct lookup — output_type already decided by planner
            agent_by_index = {
                i: ('visualization_agent'
                    if m['output_type'] == 'visualization'
                    else 'analytical_agent')
                for i, m in enumerate(metadata_list)
            }
            print(f"[Analyzer] Using planner output_type for routing (no LLM call needed)")
        else:
            # Legacy path: truncate for LLM (max 8 words per task)
            subtasks_for_llm = [' '.join(s.split()[:8]) for s in subtasks]
            agent_by_index   = self._llm_route(subtasks_for_llm)

            if not agent_by_index:
                print("[Analyzer] LLM routing produced no output — using heuristic fallback")
                agent_by_index = self._heuristic_route(subtasks_for_llm)

        # ── STEP 3: Build enriched mapping ────────────────────────────────────
        enriched_mapping = {}

        for i, full_desc in enumerate(subtasks):
            agent_name = agent_by_index.get(i, 'analytical_agent')
            meta       = metadata_list[i]

            enriched_mapping[full_desc] = {
                'agent':                 agent_name,
                # Context contract fields — passed through from planner, not interpreted here
                'output_type':           meta.get('output_type',           'analytical'),
                'ml_step':               meta.get('ml_step',               None),
                'requires':              meta.get('requires',              []),
                'produces':              meta.get('produces',              []),
                # Column constraint fields
                'target_columns':        meta.get('target_columns',        []),
                'column_constraint':     meta.get('column_constraint',     False),
                'all_mentioned_columns': meta.get('all_mentioned_columns', []),
            }

            # Diagnostic log
            req = [k.split(':')[0] for k in meta.get('requires', [])]
            pro = [k.split(':')[0] for k in meta.get('produces', [])]
            cols_count = len(meta.get('target_columns', []))
            ml_step    = meta.get('ml_step')

            parts = [f"→ {agent_name}"]
            if ml_step:
                parts.append(f"[{ml_step}]")
            if cols_count:
                parts.append(f"cols={cols_count}")
            if req:
                parts.append(f"requires={req}")
            if pro:
                parts.append(f"produces={pro}")

            print(f"[Analyzer] '{full_desc[:55]}...' {' '.join(parts)}")

        return enriched_mapping


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _empty_meta(task_str: str) -> dict:
    """Minimal metadata dict for legacy input paths."""
    return {
        'description':           task_str,
        'output_type':           None,   # will trigger LLM routing fallback
        'ml_step':               None,
        'requires':              [],
        'produces':              [],
        'target_columns':        [],
        'column_constraint':     False,
        'all_mentioned_columns': [],
    }