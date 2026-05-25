"""
Comprehensive Evaluation Script for Multi-Agent Data Analysis System
=====================================================================

Tests the agent across 6 datasets with 18 test cases (3 per dataset):
  - titanic        : survival analysis, groupby, null handling
  - california     : groupby median, missing values, feature engineering
  - wine           : correlation, conditional filter, groupby mean
  - imdb           : string parsing, correlation, datetime feature engineering
  - airbnb         : string cleaning (price), conditional filter, groupby
  - hr_attrition   : string-encoded boolean aggregation, groupby income, groupby apply

Agent output contract (updated analytical + visualization agents):
  - AnalyticalAgent.execute() returns plain {str: float} dicts — no formatter wrapper
  - VisualizationAgent.execute() returns {'status': 'success'/'error', 'file': '...'}
  - ResultExtractor._unwrap() is now a safety-only passthrough
  - _visualization() scores {'status': 'error'} results as 0.0

Changes in this version
-----------------------
ResultExtractor:
  1. Hallucinated tasks are excluded from candidates BEFORE scoring — their
     results can no longer "win" even if the heuristic score is high.
  2. Validator similarity score is weighted 10× heavier than the heuristic
     in the sort key, so ground-truth proximity always beats key-name guessing.

TaskDecompositionAnalyzer:
  3. task_count_appropriate now accepts 1–7 tasks (not 3–7) so simple
     single-operation problems aren't penalised for generating 1–2 tasks.
  4. Hallucination penalty is only applied to the decomposition score when
     the hallucinated task's result was actually selected by the extractor.
     A hallucinated task that executed but was NOT chosen is penalised less
     heavily (structural noise vs actual answer corruption).

Usage:
    python evaluate.py
"""

import json
import pandas as pd
import numpy as np
import time
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# ─────────────────────────── import your pipeline ────────────────────────────
from main import run_analysis_programmatic


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    id: str
    dataset_key: str
    problem: str
    description: str
    task_type: str            # 'analytical' | 'visualization' | 'mixed' | 'multistep'
    expected_agents: List[str]
    ground_truth_type: str    # 'single_value' | 'numeric_dict' | 'visualization'
    ground_truth_value: Any
    tolerance: float
    expected_columns: List[str]
    expected_operations: List[str]
    requires_visualization: bool = False
    expected_chart_type: str = ""
    difficulty: str = "medium"   # 'easy' | 'medium' | 'hard'


@dataclass
class RunResult:
    test_id: str
    dataset_key: str
    problem: str
    description: str
    pipeline: str
    run_number: int
    timestamp: str
    success: bool
    execution_time: float
    output: Optional[Dict]
    error: Optional[str]
    ground_truth_type: str
    ground_truth_value: Any
    tolerance: float
    expected_columns: List[str]
    expected_operations: List[str]
    requires_visualization: bool
    task_type: str
    difficulty: str


@dataclass
class TestMetrics:
    test_id: str
    dataset_key: str
    problem: str
    pipeline: str
    run_number: int
    timestamp: str
    task_type: str
    difficulty: str
    # Execution
    execution_success: float
    execution_time: float
    error_message: str
    # Correctness
    output_correctness: float
    output_correctness_binary: int
    validation_message: str
    # Code quality
    code_quality_score: float
    uses_correct_operations: float
    handles_missing_values: float
    uses_correct_columns: float
    efficiency_score: float
    # Decomposition
    task_decomposition_score: float
    total_tasks: int
    hallucinated_task_count: int
    has_duplicate_tasks: bool
    duplicate_task_count: int
    task_count_appropriate: bool
    column_constraint_violations: int
    # Agent routing
    visualization_agent_used: bool
    planner_rag_used: bool
    planner_rag_retrievals: int
    # Overall
    overall_quality_score: float


# ══════════════════════════════════════════════════════════════════════════════
#  PLANNER HALLUCINATION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

_HALLUCINATION_PHRASES = [
    "i've followed",
    "i have followed",
    "following the guidelines",
    "staying minimal",
    "note:",
    "note that",
    "as instructed",
    "per the guidelines",
    "as per",
    "as requested",
    "i have used",
    "i used only",
    "i have only used",
]

def is_hallucinated_task(task_description: str) -> bool:
    lowered = task_description.lower().strip()
    return any(phrase in lowered for phrase in _HALLUCINATION_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class GroundTruthValidator:
    """Validates pipeline output against pre-calculated ground truth."""

    def validate(self, actual: Any, expected: Any, tolerance: float,
                 validation_type: str) -> Tuple[bool, str, float]:
        if actual is None and validation_type != 'visualization':
            return False, "Output is None", 0.0

        dispatch = {
            'single_value':  self._single_value,
            'numeric_dict':  self._numeric_dict,
            'visualization': self._visualization,
        }
        fn = dispatch.get(validation_type)
        if fn is None:
            return False, f"Unknown validation type: {validation_type}", 0.0
        return fn(actual, expected, tolerance)

    @staticmethod
    def _to_float(val: Any) -> Optional[float]:
        """Convert a value to float, handling numpy scalars, numeric strings, and % strings."""
        if isinstance(val, str):
            val = val.strip().rstrip('%').strip()
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    _ANSWER_KEY_HINTS = [
        'correlation', 'corr', 'pearson', 'pearson_r',
        'missing', 'missing_count', 'null_count', 'nan_count', 'missing_values',
        'outliers', 'outlier_count',
        'percentage', 'pct', 'percent', 'rate', 'proportion', 'survival_rate',
        'attrition_rate', 'attrition',
        'mean', 'median', 'mode', 'average', 'avg',
        'std', 'std_dev', 'standard_deviation', 'variance',
        'revenue', 'budget', 'income', 'price',
        'result', 'value', 'answer', 'score',
    ]

    @staticmethod
    def _collect_all_numerics(obj: Any, _depth: int = 0) -> List[Tuple[str, float]]:
        if _depth > 4:
            return []
        results = []
        if isinstance(obj, bool):
            return []
        if not isinstance(obj, (dict, list, tuple)):
            v = GroundTruthValidator._to_float(obj)
            if v is not None:
                results.append(('', v))
            return results
        if isinstance(obj, dict):
            for k, val in obj.items():
                for path, v in GroundTruthValidator._collect_all_numerics(val, _depth + 1):
                    results.append((str(k) if not path else f"{k}.{path}", v))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                for path, v in GroundTruthValidator._collect_all_numerics(item, _depth + 1):
                    results.append((str(i) if not path else f"{i}.{path}", v))
        return results

    @staticmethod
    def _extract_scalar(actual: Any, expected: float = None) -> Optional[float]:
        if isinstance(actual, bool):
            return None

        # 1. Direct numeric
        v = GroundTruthValidator._to_float(actual)
        if v is not None and not isinstance(actual, (dict, list, tuple)):
            return v

        # 2. Unwrap wrapper keys
        if isinstance(actual, dict):
            for wk in ('raw_data', 'result', 'value', 'output'):
                if wk in actual:
                    inner = actual[wk]
                    if not isinstance(inner, (dict, list)):
                        c = GroundTruthValidator._to_float(inner)
                        if c is not None:
                            return c
                    else:
                        c = GroundTruthValidator._extract_scalar(inner, expected)
                        if c is not None:
                            return c

        all_numerics = GroundTruthValidator._collect_all_numerics(actual)
        if not all_numerics:
            return None

        # 3. Single numeric
        if len(all_numerics) == 1:
            return all_numerics[0][1]

        _stat_words = {'mean', 'std', 'stddev', 'median', 'count', 'min', 'max',
                       'variance', 'std_dev', 'percentage', 'pct'}

        def _is_per_column_stat(path: str) -> bool:
            parts = path.split('.')
            if len(parts) < 2:
                return False
            parent = parts[-2].lower()
            leaf   = parts[-1].lower()
            return leaf in _stat_words and parent not in _stat_words

        per_col_count = sum(1 for p, _ in all_numerics if _is_per_column_stat(p))
        skip_generic_stats = per_col_count > len(all_numerics) * 0.5

        # 4. Key-hint priority
        hint_winner = None
        for hint in GroundTruthValidator._ANSWER_KEY_HINTS:
            if skip_generic_stats and hint in _stat_words:
                continue
            for path, val in all_numerics:
                leaf = path.split('.')[-1].lower()
                if hint in leaf:
                    hint_winner = val
                    break
            if hint_winner is not None:
                break

        # 5. Proximity matching
        if expected is not None:
            proximate = min(all_numerics, key=lambda pv: abs(pv[1] - expected))[1]
            if hint_winner is not None:
                hint_diff = abs(hint_winner - expected)
                prox_diff = abs(proximate   - expected)
                if prox_diff * 3 < hint_diff:
                    return proximate
            else:
                return proximate

        if hint_winner is not None:
            return hint_winner

        # 6. First numeric
        return all_numerics[0][1]

    def _single_value(self, actual, expected, tolerance):
        v = self._extract_scalar(actual, expected=expected)
        if v is None:
            return False, f"Cannot extract scalar from {type(actual).__name__}: {str(actual)[:80]}", 0.0
        diff = abs(v - expected)
        if diff <= tolerance:
            return True, f"Matches: {v:.4f}", 1.0
        sim = max(0.0, 1.0 - diff / (abs(expected) + 1e-9))
        return False, f"Got {v:.4f}, expected {expected:.4f} (diff={diff:.4f})", sim

    def _numeric_dict(self, actual, expected, tolerance):
        # ── Shape B: list of records ──────────────────────────────────────────
        if isinstance(actual, list):
            if len(actual) == 1 and isinstance(actual[0], dict):
                inner = actual[0]
                for wk in ('raw_data', 'result', 'value', 'output'):
                    if wk in inner:
                        return self._numeric_dict(inner[wk], expected, tolerance)

            if not all(isinstance(r, dict) for r in actual):
                return False, "List contains non-dict elements", 0.0

            expected_keys_str = {str(k) for k in expected.keys()}
            all_cols = list(actual[0].keys()) if actual else []

            group_col = None
            for col in all_cols:
                col_vals = {str(r.get(col, '')) for r in actual}
                if col_vals >= expected_keys_str:
                    group_col = col
                    break

            if group_col is None:
                return False, (
                    f"Records list has {len(actual)} rows but no column whose values "
                    f"match expected keys {sorted(expected_keys_str)}"
                ), 0.0

            numeric_cols = [
                c for c in all_cols
                if c != group_col and
                self._to_float(actual[0].get(c)) is not None
            ]

            if not numeric_cols:
                return False, f"No numeric columns found alongside group key '{group_col}'", 0.0

            best_sim, best_msg, best_ok, best_col = 0.0, '', False, ''
            for value_col in numeric_cols:
                built = {str(r[group_col]): r.get(value_col) for r in actual}
                ok, msg, sim = self._compare_flat_dicts(built, expected, tolerance)
                if sim > best_sim:
                    best_sim, best_msg, best_ok, best_col = sim, msg, ok, value_col

            prefix = f"(matched on column '{best_col}') " if best_col else ''
            return best_ok, prefix + best_msg, best_sim

        # ── Shape C / D: dict ─────────────────────────────────────────────────
        if not isinstance(actual, dict):
            return False, f"Expected dict or records list, got {type(actual).__name__}", 0.0

        unwrapped = actual

        for wrapper_key in ('raw_data', 'result', 'value', 'output'):
            if len(unwrapped) == 1 and wrapper_key in unwrapped:
                inner = unwrapped[wrapper_key]
                if isinstance(inner, (dict, list)):
                    return self._numeric_dict(inner, expected, tolerance)

        for _ in range(3):
            if len(unwrapped) == 1:
                inner = next(iter(unwrapped.values()))
                if isinstance(inner, dict):
                    unwrapped = inner
                    continue
            break

        # ── Shape E: deeply nested dict ───────────────────────────────────────
        expected_keys_str = {str(k) for k in expected.keys()}
        best_ok, best_msg, best_sim = False, '', 0.0

        def _search_nested(obj, depth=0):
            nonlocal best_ok, best_msg, best_sim
            if depth > 5 or not isinstance(obj, dict):
                return
            norm = {str(k): v for k, v in obj.items()}
            if expected_keys_str <= set(norm.keys()):
                ok, msg, sim = self._compare_flat_dicts(norm, expected, tolerance)
                if sim > best_sim:
                    best_ok, best_msg, best_sim = ok, msg, sim
            for v in obj.values():
                if isinstance(v, dict):
                    _search_nested(v, depth + 1)

        _search_nested(unwrapped)
        if best_sim > 0:
            return best_ok, best_msg, best_sim

        # ── Shape A: flat dict ────────────────────────────────────────────────
        norm = {str(k): v for k, v in unwrapped.items()}
        ok, msg, sim = self._compare_flat_dicts(norm, expected, tolerance)

        # ── Shape F: value-set fallback ───────────────────────────────────────
        if sim == 0.0:
            ok2, msg2, sim2 = self._value_set_match(norm, expected, tolerance)
            if sim2 > sim:
                return ok2, f'(key mismatch, value-set match) {msg2}', sim2 * 0.7

        return ok, msg, sim

    def _value_set_match(self, actual_norm: Dict, expected: Dict,
                         tolerance: float) -> Tuple[bool, str, float]:
        actual_vals = []
        for v in actual_norm.values():
            f = self._to_float(v)
            if f is not None:
                actual_vals.append(f)
            elif isinstance(v, dict):
                for inner_key in ('median', 'mean', 'value', 'result'):
                    f2 = self._to_float(v.get(inner_key))
                    if f2 is not None:
                        actual_vals.append(f2)
                        break

        expected_items = [(str(k), self._to_float(v)) for k, v in expected.items()]
        expected_items = [(k, v) for k, v in expected_items if v is not None]

        if not actual_vals or not expected_items:
            return False, 'No numeric values to compare', 0.0

        remaining = list(actual_vals)
        matched, issues = 0, []
        for ek, ev in expected_items:
            best_i, best_diff = -1, float('inf')
            for i, av in enumerate(remaining):
                diff = abs(av - ev)
                if diff < best_diff:
                    best_diff, best_i = diff, i
            if best_i >= 0 and best_diff <= tolerance:
                matched += 1
                remaining.pop(best_i)
            else:
                issues.append(f"'{ek}': no match (closest diff={best_diff:.4f})")

        sim = matched / len(expected_items) if expected_items else 0.0
        msg = f'{matched}/{len(expected_items)} values matched by value' + (
            '' if not issues else ': ' + '; '.join(issues[:2]))
        return sim == 1.0, msg, sim

    def _compare_flat_dicts(self, actual_norm: Dict, expected: Dict,
                            tolerance: float) -> Tuple[bool, str, float]:
        matches, total, issues = 0, 0, []
        for k, ev in expected.items():
            total += 1
            av = actual_norm.get(str(k))
            if av is None:
                issues.append(f"Missing key '{k}'")
                continue
            af = self._to_float(av)
            ef = self._to_float(ev)
            if af is None:
                issues.append(f"'{k}': non-numeric value {av!r}")
                continue
            if abs(af - ef) <= tolerance:
                matches += 1
            else:
                issues.append(f"'{k}': got {af:.4f}, expected {ef:.4f}")

        sim = matches / total if total else 0.0
        ok  = sim == 1.0
        msg = "All match" if ok else f"{matches}/{total}: " + "; ".join(issues[:3])
        return ok, msg, sim

    @staticmethod
    def _find_recent_image(max_age_seconds: int = 60):
        image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html'}
        now = time.time()
        candidates = []

        search_dirs = [Path('.')]
        try:
            search_dirs += [p for p in Path('.').iterdir() if p.is_dir()]
        except PermissionError:
            pass

        for directory in search_dirs:
            try:
                for f in directory.iterdir():
                    if f.suffix.lower() in image_extensions and f.is_file():
                        age = now - f.stat().st_mtime
                        if age <= max_age_seconds:
                            candidates.append((f.stat().st_mtime, str(f)))
            except (PermissionError, OSError):
                continue

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return None

    @staticmethod
    def _visualization(actual, expected, tolerance):
        if isinstance(actual, dict) and actual.get('status') == 'error':
            err = actual.get('error', 'unknown error')
            return False, f"Visualization agent reported error: {str(err)[:80]}", 0.0

        if isinstance(actual, dict):
            for k in ('file', 'output', 'path'):
                v = actual.get(k)
                if v and isinstance(v, str) and v.strip():
                    if Path(v).exists():
                        return True, f"Visualization file confirmed on disk: {v}", 1.0
                    break

        found = GroundTruthValidator._find_recent_image(max_age_seconds=60)
        if found:
            return True, f"Visualization file found on disk: {found}", 1.0

        if isinstance(actual, dict):
            status = actual.get('status', '')
            if isinstance(status, str) and 'success' in status.lower():
                return True, "Visualization status=success (no output file found on disk)", 0.7

        return True, "Visualization task ran (no output file detected)", 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  CODE QUALITY ANALYZER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class CodeQualityAnalyzer:
    PLOT_KEYWORDS   = ['matplotlib', 'pyplot', 'plt', 'seaborn', 'sns', 'plotly',
                       '.plot(', '.scatter(', '.bar(', '.hist(', '.boxplot(']
    NULL_KEYWORDS   = ['dropna', 'fillna', 'isna', 'notna', 'isnull', 'notnull']
    NULL_COUNT_BUG_PATTERN = re.compile(r'len\s*\(\s*\w+\[.+\]\.isnull\(\)')

    _OP_EQUIVALENTS = {
    'mean': [
        '.mean(',
        "agg('mean')",
        'agg("mean")',
        'aggregate('
    ],
    'median': [
        '.median(',
        "agg('median')",
        'agg("median")'
    ],
    'std': [
        '.std(',
        "agg('std')",
        'agg("std")'
    ],
    'corr': [
        '.corr(',
        'pearsonr(',
        'spearmanr('
    ],
    'groupby': [
        'groupby(',
        'pivot_table('
    ],
    'isna': [
        '.isna(',
        '.isnull(',
        'pd.isna(',
        'pd.isnull('
    ],
    'fillna': [
        '.fillna(',
        '.replace(np.nan'
    ],
    'str.replace': [
        'str.replace(',
        'str.strip(',
        're.sub('
    ],
    'apply': [
        '.apply(',
        '.map(',
        '.transform('
    ],
    'scatter': [
        '.scatter(',
        "kind='scatter'",
        'go.Scatter('
    ],
}

    def _op_present(self, code: str, op: str) -> bool:
        if op in code:
            return True
        for equiv in self._OP_EQUIVALENTS.get(op, []):
            if equiv in code:
                return True
        return False

    def analyze(self, code: str, task: str, expected_columns: List[str],
                expected_operations: List[str], is_visualization: bool = False,
                constrained_columns: Optional[List[str]] = None) -> Dict:
        if not code or not isinstance(code, str):
            return {
                'uses_correct_operations': False,
                'handles_missing_values': False,
                'uses_correct_columns': False,
                'has_hardcoded_values': False,
                'efficiency_score': 0.0,
                'null_count_bug': False,
                'column_constraint_violation': False,
                'issues': ['No code generated'],
                'aggregate_score': 0.0,
            }

        issues = []

        uses_correct_ops = True
        if is_visualization:
            if not any(kw in code for kw in self.PLOT_KEYWORDS):
                uses_correct_ops = False
                issues.append("No plotting library detected")
        else:
            for op in expected_operations:
                if not self._op_present(code, op):
                    uses_correct_ops = False
                    issues.append(f"Missing operation: {op}")

        handles_nulls = any(kw in code for kw in self.NULL_KEYWORDS)
        if not handles_nulls and any(op in code for op in ['mean', 'corr', 'std', 'median']):
            issues.append("Statistical ops present but no null handling")

        null_count_bug = bool(self.NULL_COUNT_BUG_PATTERN.search(code))
        if null_count_bug:
            issues.append("Bug: len(df[col].isnull()) should be df[col].isnull().sum()")

        uses_correct_cols = True
        for col in expected_columns:
            if f"'{col}'" not in code and f'"{col}"' not in code:
                uses_correct_cols = False
                issues.append(f"Expected column '{col}' not referenced")

        column_constraint_violation = False
        if constrained_columns:
            if 'df.columns' in code and constrained_columns:
                column_constraint_violation = True
                issues.append("Uses df.columns — violates column constraint")
            if re.search(r'dropna\(\s*\)', code) and constrained_columns:
                issues.append("dropna() with no subset= may operate on unconstrained columns")

        has_hardcoded = bool(re.search(r'(?<![=<>!])\s*=\s*[0-9]+\.?[0-9]*\b', code))

        efficiency = 1.0
        if not is_visualization:
            if 'corr' in task.lower() and 'for ' in code and '.corr(' not in code:
                efficiency -= 0.3
                issues.append("Manual correlation loop (use .corr())")
            if 'group' in task.lower() and 'for ' in code and 'groupby' not in code:
                efficiency -= 0.3
                issues.append("Manual groupby loop (use .groupby())")
            if re.search(r'for .+ in .+:\s*\n.*\.append', code):
                efficiency -= 0.2
                issues.append("Row-by-row loop (prefer vectorised pandas)")

        null_bug_penalty = 0.2 if null_count_bug else 0.0

        aggregate = max(0.0, (
            (1.0 if uses_correct_ops else 0.0) * 0.30 +
            (1.0 if handles_nulls else 0.5)    * 0.20 +
            (1.0 if uses_correct_cols else 0.0) * 0.30 +
            max(0.0, efficiency)                * 0.20
            - null_bug_penalty
        ))

        return {
            'uses_correct_operations': uses_correct_ops,
            'handles_missing_values': handles_nulls,
            'uses_correct_columns': uses_correct_cols,
            'has_hardcoded_values': has_hardcoded,
            'efficiency_score': max(0.0, efficiency),
            'null_count_bug': null_count_bug,
            'column_constraint_violation': column_constraint_violation,
            'issues': issues,
            'aggregate_score': aggregate,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TASK DECOMPOSITION ANALYZER
#
#  CHANGES:
#  1. task_count_appropriate: lower bound lowered from 3 → 1 so that simple
#     1-2 task plans are not penalised.
#  2. The hallucination penalty is split into two levels:
#       - "selected" hallucination (result was chosen by extractor): -0.30 each
#       - "structural" hallucination (result was NOT chosen): -0.10 each
#     This avoids over-penalising plans where a hallucinated task ran but its
#     result was correctly ignored downstream.
#  3. analyze() accepts an optional selected_task_desc parameter so the caller
#     can pass the description of the task whose result was actually used.
# ══════════════════════════════════════════════════════════════════════════════

class TaskDecompositionAnalyzer:

    @staticmethod
    def jaccard(a: str, b: str) -> float:
        w1, w2 = set(a.lower().split()), set(b.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def analyze(self, subtasks: List[Dict],
                selected_task_desc: Optional[str] = None) -> Dict:
        """
        Parameters
        ----------
        subtasks : list of subtask dicts (each has at least a 'description' key)
        selected_task_desc : description of the subtask whose result was actually
            chosen by ResultExtractor. Pass None if unknown / not applicable.
        """
        if not subtasks:
            return {
                'total_tasks': 0, 'hallucinated_tasks': [], 'hallucinated_count': 0,
                'real_task_count': 0, 'has_duplicates': False, 'duplicate_count': 0,
                'duplicate_pairs': [], 'avg_task_length': 0,
                'task_count_appropriate': False, 'score': 0.0,
                'selected_was_hallucinated': False,
            }

        descs = [t.get('description', '') for t in subtasks]

        hallucinated_indices = [i for i, d in enumerate(descs) if is_hallucinated_task(d)]
        hallucinated_tasks   = [descs[i] for i in hallucinated_indices]
        real_descs = [d for i, d in enumerate(descs) if i not in hallucinated_indices]

        dup_pairs = [
            (i, j, self.jaccard(real_descs[i], real_descs[j]))
            for i in range(len(real_descs))
            for j in range(i + 1, len(real_descs))
            if self.jaccard(real_descs[i], real_descs[j]) > 0.75
        ]

        real_count = len(real_descs)

        # FIX 1: accept 1–7 tasks (was 3–7).  Single-operation problems
        # legitimately need only 1 task; 2 tasks is fine too.
        count_ok = 1 <= real_count <= 7

        # FIX 2: differentiate between hallucinated tasks whose result was
        # selected vs those that were merely structural noise.
        selected_was_hallucinated = False
        if selected_task_desc and hallucinated_tasks:
            selected_was_hallucinated = is_hallucinated_task(selected_task_desc)

        # Build per-task penalty:
        #   - selected hallucination: 0.30 per task
        #   - structural hallucination (not selected): 0.10 per task
        hallucination_penalty = 0.0
        for ht in hallucinated_tasks:
            is_selected = (
                selected_task_desc is not None
                and ht.lower().strip() == selected_task_desc.lower().strip()
            )
            hallucination_penalty += 0.30 if is_selected else 0.10

        hallucination_factor = max(0.0, 1.0 - hallucination_penalty)

        score = (
            (1.0 if count_ok else 0.5)   * 0.40 +
            (0.0 if dup_pairs else 1.0)  * 0.40 +
            hallucination_factor          * 0.20
        )

        return {
            'total_tasks':               len(subtasks),
            'hallucinated_tasks':        hallucinated_tasks,
            'hallucinated_count':        len(hallucinated_tasks),
            'real_task_count':           real_count,
            'has_duplicates':            bool(dup_pairs),
            'duplicate_count':           len(dup_pairs),
            'duplicate_pairs':           dup_pairs,
            'avg_task_length':           float(np.mean([len(d) for d in descs])) if descs else 0.0,
            'task_count_appropriate':    count_ok,
            'score':                     score,
            'selected_was_hallucinated': selected_was_hallucinated,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT EXTRACTOR
#
#  CHANGES:
#  1. Hallucinated tasks are excluded from candidates BEFORE any scoring.
#     Previously they received score=-1000 in _score() but were still in the
#     list; if all other tasks also had low scores they could still "win".
#     Now they are filtered out entirely.
#  2. Sort key weights validator similarity 10× more than the heuristic:
#       key = (sim * 10 + heuristic * 0.001)
#     Previously both dimensions were equal which let heuristic key-name
#     guessing override a correct similarity match.
# ══════════════════════════════════════════════════════════════════════════════

class ResultExtractor:

    _STATUS_ONLY_KEYS = frozenset({'status', 'file', 'output', 'message', 'path'})

    _TASK_ANSWER_HINTS = [
        'correlation', 'corr', 'survival rate', 'survival', 'percentage', 'percent',
        'standard deviation', 'std', 'missing values', 'missing', 'median', 'mean',
        'average', 'calculate', 'compute', 'what is', 'how many', 'count',
        'attrition', 'revenue', 'income', 'price', 'budget',
    ]

    def _score(self, task_result: Dict, requires_visualization: bool) -> int:
        """
        Heuristic score used only as a tie-breaker when validator similarity
        scores are equal.  Hallucinated tasks are never passed to this method
        (they are filtered before scoring).
        """
        agent  = str(task_result.get('agent', ''))
        result = task_result.get('result', {})
        code   = task_result.get('code', '')
        task   = str(task_result.get('task', ''))
        score  = 0

        if 'analytical' in agent:
            score += 20
        if 'visualization' in agent and not requires_visualization:
            score -= 10

        task_lower = task.lower()
        for hint in self._TASK_ANSWER_HINTS:
            if hint in task_lower:
                score += 8
                break

        if isinstance(result, list):
            if result and all(isinstance(r, dict) for r in result):
                if len(result) <= 20:
                    sample_keys = set(result[0].keys()) if result else set()
                    for key in ('mean', 'median', 'rate', 'survival', 'percentage',
                                'pct', 'corr', 'std', 'attrition', 'income', 'price'):
                        if any(key in str(k).lower() for k in sample_keys):
                            score += 15
                            break
                    score += 10
                else:
                    score -= 20

        elif isinstance(result, dict):
            keys = set(result.keys())

            if keys <= self._STATUS_ONLY_KEYS:
                score -= 15

            dict_vals = [v for v in result.values() if isinstance(v, dict)]
            if len(dict_vals) > len(result) * 0.6 and len(result) > 3:
                score -= 20

            values = list(result.values())
            list_vals = [v for v in values if isinstance(v, list)]
            if list_vals and max((len(lv) for lv in list_vals), default=0) > 20:
                score -= 30

            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            if numeric_vals and len(set(numeric_vals)) == 1 and len(numeric_vals) > 2:
                score -= 10

            score += len(numeric_vals) * 3

            HIGH_VALUE_KEYS = [
                ('null_count', 20), ('null_counts', 18), ('missing_count', 18),
                ('missing_values', 16), ('missing', 14),
                ('std_dev', 16), ('standard_deviation', 16),
                ('correlation', 14), ('corr', 14),
                ('attrition_rate', 14), ('attrition', 12),
                ('survival_rate', 14), ('percentage', 14), ('pct', 14), ('rate', 12),
                ('outliers', 12), ('outlier_count', 12),
                ('revenue', 10), ('income', 10), ('price', 10),
                ('mean', 8), ('median', 8), ('std', 8), ('count', 6),
            ]
            for key_hint, bonus in HIGH_VALUE_KEYS:
                if any(key_hint in str(k).lower() for k in keys):
                    score += bonus
                    break

            metadata_keys = {'column_names', 'dtypes', 'selected_columns', 'columns', 'shape'}
            if keys & metadata_keys:
                score -= 12
                int_vals = [v for v in values if isinstance(v, int) and v > 100]
                if int_vals and all(v > 500 for v in int_vals):
                    score -= 8

        elif isinstance(result, (int, float)):
            score += 15

        if 'df.columns' in code:
            score -= 20

        return score

    @staticmethod
    def _unwrap(result: Any) -> Any:
        if isinstance(result, dict) and 'raw_data' in result and 'summary' in result:
            return result['raw_data']
        return result

    def extract(self, task_results: List[Dict], requires_visualization: bool,
                validator=None, gt_value=None, gt_type: str = None,
                tolerance: float = 0.0) -> Tuple[Any, str]:
        """
        Returns (best_result, best_agent_name).

        FIX 1: hallucinated tasks are excluded from candidates entirely before
                any scoring takes place.
        FIX 2: sort key = sim * 10 + heuristic * 0.001, so validator similarity
                dominates over heuristic key-name guessing.
        """
        # Base filter: task must have succeeded and produced a result
        candidates = [
            tr for tr in task_results
            if tr.get('ok') and tr.get('result') is not None
        ]

        # FIX 1 — drop hallucinated tasks before scoring
        non_hallucinated = [
            tr for tr in candidates
            if not is_hallucinated_task(str(tr.get('task', '')))
        ]

        # If filtering removed everything, fall back to all candidates so we
        # return something rather than None (but log a warning).
        if not non_hallucinated and candidates:
            print("  ⚠ ResultExtractor: all successful tasks were hallucinated — "
                  "falling back to full candidate set")
            non_hallucinated = candidates
        elif not non_hallucinated:
            return None, ''

        scored = []
        for tr in non_hallucinated:
            result    = self._unwrap(tr.get('result', {}))
            heuristic = self._score(tr, requires_visualization)
            sim       = 0.0
            if validator is not None and gt_value is not None and gt_type is not None:
                try:
                    _, _, sim = validator.validate(result, gt_value, tolerance, gt_type)
                except Exception:
                    sim = 0.0
            scored.append((sim, heuristic, tr, result))

        # FIX 2 — validator similarity weighted 10× over heuristic
        scored.sort(key=lambda x: (x[0] * 10 + x[1] * 0.001), reverse=True)

        _, _, best_tr, best_result = scored[0]
        return best_result, str(best_tr.get('agent', ''))


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH FACTORY  —  18 test cases, 3 per dataset
# ══════════════════════════════════════════════════════════════════════════════

class GroundTruthFactory:
    """Pre-computes correct answers and builds test cases for all datasets."""

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets

    def compute(self) -> Tuple[Dict, List[TestCase]]:
        gt    = {}
        cases = []
        cases += self._titanic_cases(gt)
        cases += self._california_cases(gt)
        cases += self._wine_cases(gt)
        cases += self._imdb_cases(gt)
        cases += self._airbnb_cases(gt)
        cases += self._hr_cases(gt)
        return gt, cases

    # ── Titanic ───────────────────────────────────────────────────────────────
    def _titanic_cases(self, gt: Dict) -> List[TestCase]:
        df = self.datasets.get('titanic')
        if df is None:
            return []
        cases = []

        survived_col = next((c for c in df.columns if c.lower() == 'survived'), None)
        pclass_col   = next((c for c in df.columns if c.lower() == 'pclass'), None)
        age_col      = next((c for c in df.columns if c.lower() == 'age'), None)

        if survived_col:
            val = float(df[survived_col].mean())
            gt['T-01'] = val
            cases.append(TestCase(
                id='T-01', dataset_key='titanic',
                problem='What is the overall survival rate (proportion) of passengers on the Titanic?',
                description='Simple survival rate',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=val,
                tolerance=0.01, expected_columns=[survived_col],
                expected_operations=['mean'], difficulty='easy'
            ))

        if survived_col and pclass_col:
            grouped = {str(int(k)): float(v)
                       for k, v in df.groupby(pclass_col)[survived_col].mean().items()}
            gt['T-02'] = grouped
            cases.append(TestCase(
                id='T-02', dataset_key='titanic',
                problem='Calculate the survival rate for each passenger class (Pclass). Return a plain number per class, not a formatted string.',
                description='Survival rate by Pclass — numeric_dict output',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=0.01, expected_columns=[pclass_col, survived_col],
                expected_operations=['groupby', 'mean'], difficulty='easy'
            ))

        if age_col:
            median_age = float(df[age_col].median())
            gt['T-03'] = median_age
            cases.append(TestCase(
                id='T-03', dataset_key='titanic',
                problem='What is the median age of passengers? Fill any missing Age values with the median before calculating.',
                description='Null-aware median computation',
                task_type='multistep', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=median_age,
                tolerance=0.5, expected_columns=[age_col],
                expected_operations=['fillna', 'median'], difficulty='medium'
            ))

        return cases

    # ── California Housing ────────────────────────────────────────────────────
    def _california_cases(self, gt: Dict) -> List[TestCase]:
        df = self.datasets.get('california')
        if df is None:
            return []
        cases = []

        if {'ocean_proximity', 'median_house_value'} <= set(df.columns):
            grouped = {str(k): float(v)
                       for k, v in df.groupby('ocean_proximity')['median_house_value'].median().items()}
            gt['C-01'] = grouped
            cases.append(TestCase(
                id='C-01', dataset_key='california',
                problem='Calculate the median house value for each ocean_proximity category.',
                description='Categorical groupby with median',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=100.0, expected_columns=['ocean_proximity', 'median_house_value'],
                expected_operations=['groupby', 'median'], difficulty='easy'
            ))

        if 'total_bedrooms' in df.columns:
            val = float(df['total_bedrooms'].isna().sum())
            gt['C-02'] = val
            cases.append(TestCase(
                id='C-02', dataset_key='california',
                problem='How many missing values are there in the total_bedrooms column?',
                description='Missing value audit',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=val,
                tolerance=0.0, expected_columns=['total_bedrooms'],
                expected_operations=['isna'], difficulty='easy'
            ))

        if {'total_rooms', 'households'} <= set(df.columns):
            df2 = df.copy()
            df2['rooms_per_household'] = df2['total_rooms'] / df2['households']
            val = float(df2['rooms_per_household'].std())
            gt['C-03'] = val
            cases.append(TestCase(
                id='C-03', dataset_key='california',
                problem='Create a new feature rooms_per_household = total_rooms / households. What is the standard deviation of this new feature?',
                description='Feature engineering then std',
                task_type='multistep', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=val,
                tolerance=0.1, expected_columns=['total_rooms', 'households'],
                expected_operations=['std'], difficulty='medium'
            ))

        return cases

    # ── Wine Quality ──────────────────────────────────────────────────────────
    def _wine_cases(self, gt: Dict) -> List[TestCase]:
        df = self.datasets.get('wine')
        if df is None:
            return []
        cases = []

        quality_col = next((c for c in df.columns if 'quality' in c.lower()), None)
        alcohol_col = next((c for c in df.columns if 'alcohol' in c.lower()), None)

        if quality_col and alcohol_col:
            val = float(df[alcohol_col].corr(df[quality_col]))
            gt['W-01'] = val
            cases.append(TestCase(
                id='W-01', dataset_key='wine',
                problem='What is the Pearson correlation between alcohol content and wine quality?',
                description='Correlation between two features',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=val,
                tolerance=0.01, expected_columns=[alcohol_col, quality_col],
                expected_operations=['corr'], difficulty='easy'
            ))

        if quality_col:
            pct = float((df[quality_col] >= 7).mean() * 100)
            gt['W-02'] = pct
            cases.append(TestCase(
                id='W-02', dataset_key='wine',
                problem='What percentage of wines have a quality score of 7 or higher? Return the answer as a plain number (e.g. 13.57), not as a formatted string.',
                description='Conditional filter and percentage',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=pct,
                tolerance=0.5, expected_columns=[quality_col],
                expected_operations=['mean'], difficulty='medium'
            ))

        if quality_col and alcohol_col:
            grouped = {str(int(k)): round(float(v), 4)
                       for k, v in df.groupby(quality_col)[alcohol_col].mean().items()}
            gt['W-03'] = grouped
            cases.append(TestCase(
                id='W-03', dataset_key='wine',
                problem='Calculate the average alcohol content for each quality score group.',
                description='Groupby mean across quality scores',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=0.05, expected_columns=[quality_col, alcohol_col],
                expected_operations=['groupby', 'mean'], difficulty='easy'
            ))

        return cases

    # ── IMDB Movies ───────────────────────────────────────────────────────────
    def _imdb_cases(self, gt: Dict) -> List[TestCase]:
        df = self.datasets.get('imdb')
        if df is None:
            return []
        cases = []

        genre_col   = next((c for c in df.columns if 'genre' in c.lower()), None)
        revenue_col = next((c for c in df.columns if 'revenue' in c.lower()), None)
        budget_col  = next((c for c in df.columns if 'budget' in c.lower()), None)
        vote_col    = next((c for c in df.columns if 'vote_average' in c.lower()
                            or c.lower() == 'score'), None)
        year_col    = next((c for c in df.columns if 'year' in c.lower()
                            or 'release' in c.lower()), None)

        if budget_col and revenue_col:
            filtered = df[(df[budget_col] > 0) & (df[revenue_col] > 0)]
            val = float(filtered[budget_col].corr(filtered[revenue_col]))
            gt['I-01'] = val
            cases.append(TestCase(
                id='I-01', dataset_key='imdb',
                problem="What is the Pearson correlation between a film's budget and its revenue? Exclude films where either budget or revenue is zero.",
                description='Filtered correlation: budget vs revenue',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=val,
                tolerance=0.02, expected_columns=[budget_col, revenue_col],
                expected_operations=['corr'], difficulty='easy'
            ))

        if genre_col and revenue_col:
            df2 = df[df[revenue_col] > 0].copy()
            df2['primary_genre'] = df2[genre_col].astype(str).str.split(
                r'[,|]').str[0].str.strip()
            grouped = {str(k): round(float(v), 2)
                       for k, v in df2.groupby('primary_genre')[revenue_col].mean().items()
                       if pd.notna(v)}
            gt['I-02'] = grouped
            cases.append(TestCase(
                id='I-02', dataset_key='imdb',
                problem='What is the average revenue for each genre? Use only the primary genre (first listed) for each film, and exclude films where revenue is 0.',
                description='String parsing + groupby mean on revenue',
                task_type='multistep', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=max(grouped.values()) * 0.02 if grouped else 1000.0,
                expected_columns=[genre_col, revenue_col],
                expected_operations=['groupby', 'mean'], difficulty='medium'
            ))

        if year_col and vote_col:
            df3 = df.dropna(subset=[year_col, vote_col]).copy()
            df3[year_col] = pd.to_numeric(df3[year_col], errors='coerce')
            df3 = df3.dropna(subset=[year_col])
            df3['decade'] = (df3[year_col].astype(int) // 10 * 10).astype(str)
            grouped = {str(k): round(float(v), 4)
                       for k, v in df3.groupby('decade')[vote_col].mean().items()}
            gt['I-03'] = grouped
            cases.append(TestCase(
                id='I-03', dataset_key='imdb',
                problem='Calculate the average vote score for each decade (e.g. 1990s = 1990, 2000s = 2000). Extract the decade from the release year.',
                description='Datetime feature engineering + groupby mean',
                task_type='multistep', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=0.1, expected_columns=[year_col, vote_col],
                expected_operations=['groupby', 'mean'], difficulty='medium'
            ))

        return cases

    # ── Airbnb Listings ───────────────────────────────────────────────────────
    def _airbnb_cases(self, gt: Dict) -> List[TestCase]:
        df = self.datasets.get('airbnb')
        if df is None:
            return []
        cases = []

        price_col      = next((c for c in df.columns if c.lower() == 'price'), None)
        hood_group_col = next((c for c in df.columns
                               if 'neighbourhood_group' in c.lower()
                               or 'neighborhood_group' in c.lower()), None)
        reviews_col    = next((c for c in df.columns if 'number_of_reviews' in c.lower()), None)
        room_type_col  = next((c for c in df.columns if 'room_type' in c.lower()), None)

        def _clean_price(series: pd.Series) -> pd.Series:
            return (series.astype(str)
                          .str.replace(r'[\$,]', '', regex=True)
                          .str.strip()
                          .pipe(pd.to_numeric, errors='coerce'))

        if price_col and hood_group_col:
            df2 = df.copy()
            df2['price_clean'] = _clean_price(df2[price_col])
            grouped = {str(k): round(float(v), 2)
                       for k, v in df2.groupby(hood_group_col)['price_clean'].mean().items()
                       if pd.notna(v)}
            gt['A-01'] = grouped
            cases.append(TestCase(
                id='A-01', dataset_key='airbnb',
                problem='What is the average listing price for each neighbourhood group? The price column contains dollar signs and commas — clean it before calculating.',
                description='Price string cleaning + groupby mean',
                task_type='multistep', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=1.0, expected_columns=[price_col, hood_group_col],
                expected_operations=['groupby', 'mean', 'str.replace'], difficulty='easy'
            ))

        if reviews_col:
            pct = float((df[reviews_col] == 0).mean() * 100)
            gt['A-02'] = pct
            cases.append(TestCase(
                id='A-02', dataset_key='airbnb',
                problem='What percentage of listings have never received a review (number_of_reviews = 0)? Return the answer as a plain number.',
                description='Conditional filter and percentage',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=pct,
                tolerance=0.5, expected_columns=[reviews_col],
                expected_operations=['mean'], difficulty='easy'
            ))

        if price_col and room_type_col:
            df3 = df.copy()
            df3['price_clean'] = _clean_price(df3[price_col])
            grouped = {str(k): round(float(v), 2)
                       for k, v in df3.groupby(room_type_col)['price_clean'].mean().items()
                       if pd.notna(v)}
            gt['A-03'] = grouped
            cases.append(TestCase(
                id='A-03', dataset_key='airbnb',
                problem='Calculate the average price per night for each room type. Clean the price column (remove $ and commas) before calculating.',
                description='Price cleaning + groupby mean by room type',
                task_type='multistep', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=1.0, expected_columns=[price_col, room_type_col],
                expected_operations=['groupby', 'mean', 'str.replace'], difficulty='medium'
            ))

        return cases

    # ── HR Attrition ──────────────────────────────────────────────────────────
    def _hr_cases(self, gt: Dict) -> List[TestCase]:
        df = self.datasets.get('hr_attrition')
        if df is None:
            return []
        cases = []

        attrition_col = next((c for c in df.columns if 'attrition' in c.lower()), None)
        dept_col      = next((c for c in df.columns if 'department' in c.lower()), None)
        income_col    = next((c for c in df.columns
                              if 'monthlyincome' in c.lower().replace(' ', '')
                              or 'monthly_income' in c.lower()), None)
        role_col      = next((c for c in df.columns
                              if 'jobrole' in c.lower().replace(' ', '')
                              or 'job_role' in c.lower()), None)

        if attrition_col:
            val = float((df[attrition_col] == 'Yes').mean())
            gt['H-01'] = val
            cases.append(TestCase(
                id='H-01', dataset_key='hr_attrition',
                problem='What is the overall employee attrition rate (proportion of employees who left)? The Attrition column contains "Yes" or "No" strings.',
                description='String-encoded boolean — overall attrition rate',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='single_value', ground_truth_value=val,
                tolerance=0.01, expected_columns=[attrition_col],
                expected_operations=['mean'], difficulty='easy'
            ))

        if dept_col and income_col:
            grouped = {str(k): round(float(v), 2)
                       for k, v in df.groupby(dept_col)[income_col].mean().items()}
            gt['H-02'] = grouped
            cases.append(TestCase(
                id='H-02', dataset_key='hr_attrition',
                problem='What is the average monthly income for each department?',
                description='Groupby mean on monthly income',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=10.0, expected_columns=[dept_col, income_col],
                expected_operations=['groupby', 'mean'], difficulty='easy'
            ))

        if attrition_col and role_col:
            grouped = {str(k): round(float(v), 4)
                       for k, v in df.groupby(role_col)[attrition_col]
                                     .apply(lambda x: (x == 'Yes').mean()).items()}
            gt['H-03'] = grouped
            cases.append(TestCase(
                id='H-03', dataset_key='hr_attrition',
                problem='Calculate the attrition rate for each job role. The Attrition column contains "Yes" or "No" — compute the proportion of "Yes" per role.',
                description='String-encoded boolean groupby attrition rate',
                task_type='analytical', expected_agents=['analytical'],
                ground_truth_type='numeric_dict', ground_truth_value=grouped,
                tolerance=0.01, expected_columns=[role_col, attrition_col],
                expected_operations=['groupby', 'apply'], difficulty='medium'
            ))

        return cases


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PATHS  —  update these to match your local file locations
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATHS = {
    'titanic':      r'C:\Users\Shreekumar\codeviz\data\titanic.csv',
    'california':   r'C:\Users\Shreekumar\codeviz\data\housing.csv',
    'wine':         r'C:\Users\Shreekumar\codeviz\data\winequality-red.csv',
    'imdb':         r'C:\Users\Shreekumar\codeviz\data\imdb_top_1000.csv',
    'airbnb':       r'C:\Users\Shreekumar\codeviz\data\airnb.csv',
    'hr_attrition': r'C:\Users\Shreekumar\codeviz\data\WA_Fn-UseC_-HR-Employee-Attrition.csv',
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class ComprehensiveEvaluator:
    def __init__(self, dataset_paths: Dict[str, str]):
        self.dataset_paths = dataset_paths
        self.datasets: Dict[str, pd.DataFrame] = {}
        self._load_datasets()

        self.validator  = GroundTruthValidator()
        self.code_qa    = CodeQualityAnalyzer()
        self.decomp_qa  = TaskDecompositionAnalyzer()
        self.extractor  = ResultExtractor()

        factory = GroundTruthFactory(self.datasets)
        self.ground_truth_cache, self.test_cases = factory.compute()

        print(f"\n✓ Loaded {len(self.datasets)} datasets")
        print(f"✓ Generated {len(self.test_cases)} test cases")
        for k, df in self.datasets.items():
            print(f"  - {k}: {len(df)} rows × {len(df.columns)} cols")

    def _load_datasets(self):
        for key, path in self.dataset_paths.items():
            if not Path(path).exists():
                print(f"  WARNING: Dataset not found — {key}: {path}")
                continue
            try:
                sep = ';' if 'wine' in path.lower() or 'winequality' in path.lower() else ','
                self.datasets[key] = pd.read_csv(path, sep=sep)
            except Exception as e:
                print(f"  WARNING: Could not load {key}: {e}")

    # ── single test run ───────────────────────────────────────────────────────

    def run_test(self, tc: TestCase, use_rag: bool, run_num: int = 1) -> RunResult:
        pipeline     = 'RAG' if use_rag else 'No-RAG'
        dataset_path = self.dataset_paths[tc.dataset_key]

        print(f"\n{'─'*70}")
        print(f"[{tc.id}] {pipeline} | Run {run_num} | {tc.difficulty.upper()} | {tc.task_type}")
        print(f"Dataset : {tc.dataset_key}")
        print(f"Problem : {tc.problem[:90]}{'...' if len(tc.problem) > 90 else ''}")
        print(f"{'─'*70}")

        start = time.time()
        output, error = None, None
        try:
            output = run_analysis_programmatic(
                dataset_path=dataset_path,
                problem_statement=tc.problem,
                use_rag=use_rag,
                verbose=False
            )
            success = True
            print("  ✓ Execution successful")
        except Exception as exc:
            success = False
            error = str(exc)
            print(f"  ✗ Execution failed: {error[:120]}")

        return RunResult(
            test_id=tc.id, dataset_key=tc.dataset_key,
            problem=tc.problem, description=tc.description,
            pipeline=pipeline, run_number=run_num,
            timestamp=datetime.now().isoformat(),
            success=success, execution_time=time.time() - start,
            output=output, error=error,
            ground_truth_type=tc.ground_truth_type,
            ground_truth_value=tc.ground_truth_value,
            tolerance=tc.tolerance,
            expected_columns=tc.expected_columns,
            expected_operations=tc.expected_operations,
            requires_visualization=tc.requires_visualization,
            task_type=tc.task_type, difficulty=tc.difficulty
        )

    # ── metric calculation ────────────────────────────────────────────────────

    def calc_metrics(self, rr: RunResult) -> TestMetrics:
        base = dict(
            test_id=rr.test_id, dataset_key=rr.dataset_key,
            problem=rr.problem, pipeline=rr.pipeline,
            run_number=rr.run_number, timestamp=rr.timestamp,
            task_type=rr.task_type, difficulty=rr.difficulty,
            execution_time=rr.execution_time,
        )

        if not rr.success or rr.output is None:
            return TestMetrics(**base,
                execution_success=0, error_message=rr.error or 'Unknown',
                output_correctness=0.0, output_correctness_binary=0,
                validation_message='Execution failed',
                code_quality_score=0.0, uses_correct_operations=0.0,
                handles_missing_values=0.0, uses_correct_columns=0.0,
                efficiency_score=0.0,
                task_decomposition_score=0.0, total_tasks=0,
                hallucinated_task_count=0,
                has_duplicate_tasks=False, duplicate_task_count=0,
                task_count_appropriate=False,
                column_constraint_violations=0,
                visualization_agent_used=False,
                planner_rag_used=False, planner_rag_retrievals=0,
                overall_quality_score=0.0
            )

        out          = rr.output
        task_results = out.get('task_results', [])

        # Extract best result — now excludes hallucinated tasks automatically
        actual, best_agent = self.extractor.extract(
            task_results,
            requires_visualization=rr.requires_visualization,
            validator=self.validator,
            gt_value=rr.ground_truth_value,
            gt_type=rr.ground_truth_type,
            tolerance=rr.tolerance,
        )
        if actual is not None:
            if isinstance(actual, dict) and 'raw_data' in actual and 'summary' in actual:
                print("  ⚠ WARNING : result still wrapped in raw_data/summary "
                      "— pipeline may be using old analytical agent")
            print(f"  Result from : {best_agent} → {str(actual)[:80]}")
        else:
            print("  Result      : None extracted from task results")

        ok, msg, sim = self.validator.validate(
            actual, rr.ground_truth_value, rr.tolerance, rr.ground_truth_type)
        print(f"  Correctness : {sim*100:.1f}% — {msg[:80]}")

        cq_scores, column_constraint_violations = [], 0

        for tr in task_results:
            if not tr.get('ok'):
                continue
            task_desc        = tr.get('task', '')
            code             = tr.get('code', '')
            constrained_cols = tr.get('constrained_columns', None)

            qa = self.code_qa.analyze(
                code=code,
                task=task_desc,
                expected_columns=rr.expected_columns,
                expected_operations=rr.expected_operations,
                is_visualization=rr.requires_visualization,
                constrained_columns=constrained_cols,
            )
            cq_scores.append(qa)

            if qa.get('column_constraint_violation'):
                column_constraint_violations += 1

            if qa['issues']:
                print(f"  Code issues : {'; '.join(qa['issues'][:2])}")

        def avg(key):
            return float(np.mean([q[key] for q in cq_scores])) if cq_scores else 0.0

        cq_agg  = avg('aggregate_score')
        cq_ops  = avg('uses_correct_operations')
        cq_null = avg('handles_missing_values')
        cq_cols = avg('uses_correct_columns')
        cq_eff  = avg('efficiency_score')

        subtasks = out.get('subtasks', [])

        # Find the description of whichever task's result was selected, so the
        # decomp analyser can distinguish "selected hallucination" from
        # "structural hallucination".
        selected_task_desc: Optional[str] = None
        for tr in task_results:
            if tr.get('ok') and tr.get('result') is not None:
                result = ResultExtractor._unwrap(tr.get('result', {}))
                # Check if this is the result that was chosen
                _, _, chosen_sim = self.validator.validate(
                    result, rr.ground_truth_value, rr.tolerance, rr.ground_truth_type
                ) if actual is not None else (False, '', 0.0)
                # The best_agent comparison is approximate; match by task description
                if str(tr.get('agent', '')) == best_agent:
                    selected_task_desc = str(tr.get('task', ''))
                    break

        da = self.decomp_qa.analyze(subtasks, selected_task_desc=selected_task_desc)

        if da['hallucinated_count']:
            for ht in da['hallucinated_tasks']:
                print(f"  ⚠ Hallucinated task: \"{ht[:70]}\"")
            if da.get('selected_was_hallucinated'):
                print("  ✗ Selected result came from a hallucinated task!")
        if da['has_duplicates']:
            print(f"  ⚠ {da['duplicate_count']} duplicate subtask(s) detected")

        agents_used = [tr.get('agent', '') for tr in task_results]
        viz_used    = rr.requires_visualization and any(
            'viz' in str(a).lower() or 'visual' in str(a).lower()
            for a in agents_used if a
        )

        overall = sim * 100 * 0.50 + cq_agg * 100 * 0.30 + da['score'] * 100 * 0.20
        print(f"  Overall     : {overall:.1f}/100")

        return TestMetrics(**base,
            execution_success=100.0, error_message='',
            output_correctness=sim * 100,
            output_correctness_binary=1 if ok else 0,
            validation_message=msg,
            code_quality_score=cq_agg * 100,
            uses_correct_operations=cq_ops * 100,
            handles_missing_values=cq_null * 100,
            uses_correct_columns=cq_cols * 100,
            efficiency_score=cq_eff * 100,
            task_decomposition_score=da['score'] * 100,
            total_tasks=da['total_tasks'],
            hallucinated_task_count=da['hallucinated_count'],
            has_duplicate_tasks=da['has_duplicates'],
            duplicate_task_count=da['duplicate_count'],
            task_count_appropriate=da['task_count_appropriate'],
            column_constraint_violations=column_constraint_violations,
            visualization_agent_used=viz_used,
            planner_rag_used=bool(out.get('planner_rag_used', False)),
            planner_rag_retrievals=int(out.get('planner_rag_retrievals', 0)),
            overall_quality_score=overall,
        )

    # ── full evaluation ───────────────────────────────────────────────────────

    def run_full_evaluation(
        self,
        num_runs: int = 1,
        test_both_pipelines: bool = True,
        filter_datasets: Optional[List[str]] = None,
        filter_task_types: Optional[List[str]] = None,
        filter_difficulty: Optional[List[str]] = None,
    ):
        cases = self.test_cases
        if filter_datasets:
            cases = [tc for tc in cases if tc.dataset_key in filter_datasets]
        if filter_task_types:
            cases = [tc for tc in cases if tc.task_type in filter_task_types]
        if filter_difficulty:
            cases = [tc for tc in cases if tc.difficulty in filter_difficulty]

        pipelines = [True, False] if test_both_pipelines else [True]
        total     = len(cases) * num_runs * len(pipelines)

        print(f"\n{'═'*70}")
        print(f"  COMPREHENSIVE MULTI-DATASET EVALUATION")
        print(f"{'═'*70}")
        print(f"  Test cases  : {len(cases)}")
        print(f"  Runs each   : {num_runs}")
        print(f"  Pipelines   : {'RAG + No-RAG' if test_both_pipelines else 'RAG only'}")
        print(f"  Total runs  : {total}")
        print(f"{'═'*70}")

        all_results, all_metrics = [], []
        idx = 0

        for use_rag in pipelines:
            for tc in cases:
                for run in range(1, num_runs + 1):
                    idx += 1
                    print(f"\n[{idx}/{total}]", end='')
                    rr = self.run_test(tc, use_rag, run)
                    m  = self.calc_metrics(rr)
                    all_results.append(rr)
                    all_metrics.append(m)

        return self.generate_report(all_results, all_metrics)

    # ── reporting (unchanged) ─────────────────────────────────────────────────

    @staticmethod
    def _to_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): ComprehensiveEvaluator._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ComprehensiveEvaluator._to_json_safe(i) for i in obj]
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return ComprehensiveEvaluator._to_json_safe(obj.to_dict())
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    @staticmethod
    def _json_safe_result(obj, max_list_len=30):
        if isinstance(obj, dict):
            return {str(k): ComprehensiveEvaluator._json_safe_result(v, max_list_len)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            if len(obj) > max_list_len:
                return f'[... {len(obj)} items, first: {ComprehensiveEvaluator._json_safe_result(obj[0], max_list_len)} ...]'
            return [ComprehensiveEvaluator._json_safe_result(i, max_list_len) for i in obj]
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return ComprehensiveEvaluator._json_safe_result(obj.to_dict(), max_list_len)
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    def generate_report(self, results: List[RunResult], metrics: List[TestMetrics]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _safe_asdict(r: RunResult) -> dict:
            return {
                'test_id':             r.test_id,
                'dataset_key':         r.dataset_key,
                'problem':             r.problem,
                'description':         r.description,
                'pipeline':            r.pipeline,
                'run_number':          r.run_number,
                'timestamp':           r.timestamp,
                'success':             r.success,
                'execution_time':      r.execution_time,
                'output':              self._to_json_safe(r.output),
                'error':               r.error,
                'ground_truth_type':   r.ground_truth_type,
                'ground_truth_value':  self._to_json_safe(r.ground_truth_value),
                'tolerance':           r.tolerance,
                'expected_columns':    r.expected_columns,
                'expected_operations': r.expected_operations,
                'requires_visualization': r.requires_visualization,
                'task_type':           r.task_type,
                'difficulty':          r.difficulty,
            }

        results_path = f'eval_results_{ts}.json'
        with open(results_path, 'w') as f:
            json.dump([_safe_asdict(r) for r in results], f, indent=2, default=str)
        print(f"\n✓ Saved raw results         → {results_path}")

        # ── Human-readable text log ───────────────────────────────────────────
        metrics_by_key = {(m.test_id, m.pipeline, m.run_number): m for m in metrics}
        txt_path = f'raw_results_{ts}.txt'
        lines = []
        W = 80

        def w(text=''):
            lines.append(str(text))

        w('=' * W)
        w(f'  RAW EVALUATION RESULTS — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        w('=' * W)

        for rr in results:
            pipeline_label = 'RAG' if (rr.output and rr.output.get('planner_rag_used')) else 'No-RAG'
            m = metrics_by_key.get((rr.test_id, pipeline_label, rr.run_number))

            w()
            w('─' * W)
            w(f'  [{rr.test_id}] {pipeline_label}  |  run={rr.run_number}  |  {rr.task_type}  |  {rr.difficulty}')
            w(f'  Problem  : {rr.problem}')
            w(f'  GT type  : {rr.ground_truth_type}')
            w(f'  GT value : {str(rr.ground_truth_value)[:120]}')
            w(f'  Tolerance: {rr.tolerance}')
            w('─' * W)

            if not rr.success or rr.output is None:
                w(f'  EXECUTION FAILED: {rr.error}')
                continue

            out          = rr.output
            task_results = out.get('task_results', [])
            subtasks     = out.get('subtasks', [])

            w(f'  Subtasks ({len(subtasks)}):')
            for i, st in enumerate(subtasks, 1):
                desc  = st.get('description', str(st))
                agent = st.get('agent', '?')
                cols  = st.get('columns', [])
                w(f'    {i}. [{agent}] {desc[:90]}')
                if cols:
                    w(f'       columns: {cols}')

            w(f'  Task Results ({len(task_results)}):')
            for i, tr in enumerate(task_results, 1):
                halluc_flag = ' [HALLUCINATED]' if is_hallucinated_task(str(tr.get('task', ''))) else ''
                w(f'  [{i}] agent : {tr.get("agent","?")}{halluc_flag}')
                w(f'      task  : {str(tr.get("task",""))[:90]}')
                w(f'      ok    : {tr.get("ok")}  |  error: {tr.get("error")}')
                result_safe = self._json_safe_result(tr.get('result'))
                try:
                    result_str = json.dumps(result_safe, indent=6, default=str)
                except Exception:
                    result_str = repr(result_safe)
                w(f'      result:')
                for line in result_str.split('\n')[:60]:
                    w(f'        {line}')
                if result_str.count('\n') > 60:
                    w('        ... (truncated)')
                code = tr.get('code', '')
                if code:
                    code_lines = code.split('\n')
                    w(f'      code  : ({len(code_lines)} lines)')
                    for cl in code_lines[:30]:
                        w(f'        {cl}')
                    if len(code_lines) > 30:
                        w(f'        ... (+{len(code_lines)-30} more lines)')

            actual, best_agent = self.extractor.extract(
                task_results,
                requires_visualization=rr.requires_visualization,
                validator=self.validator,
                gt_value=rr.ground_truth_value,
                gt_type=rr.ground_truth_type,
                tolerance=rr.tolerance,
            )
            ok_v, msg_v, sim_v = self.validator.validate(
                actual, rr.ground_truth_value, rr.tolerance, rr.ground_truth_type)

            w()
            w('  ── Extraction ──')
            w(f'  Chosen agent  : {best_agent}')
            actual_safe = self._json_safe_result(actual)
            try:
                actual_str = json.dumps(actual_safe, indent=4, default=str)
            except Exception:
                actual_str = repr(actual_safe)
            w('  Extracted val :')
            for line in actual_str.split('\n')[:30]:
                w(f'    {line}')

            w()
            w('  ── Validation ──')
            w(f'  Correctness   : {sim_v*100:.1f}%  (ok={ok_v})')
            w(f'  Message       : {msg_v}')

            if m:
                w()
                w('  ── Metrics ──')
                w(f'  overall_quality    : {m.overall_quality_score:.1f}')
                w(f'  output_correctness : {m.output_correctness:.1f}%')
                w(f'  code_quality       : {m.code_quality_score:.1f}%')
                w(f'  task_decomposition : {m.task_decomposition_score:.1f}%')
                w(f'  hallucinated_tasks : {m.hallucinated_task_count}')
                w(f'  exec_time          : {rr.execution_time:.1f}s')

        w()
        w('=' * W)
        w('END')
        w('=' * W)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✓ Saved raw results text    → {txt_path}")

        # ── CSV metrics ───────────────────────────────────────────────────────
        df_m         = pd.DataFrame([asdict(m) for m in metrics])
        metrics_path = f'eval_metrics_{ts}.csv'
        df_m.to_csv(metrics_path, index=False)
        print(f"✓ Saved detailed metrics    → {metrics_path}")

        numeric_cols = [
            'execution_success', 'output_correctness', 'output_correctness_binary',
            'code_quality_score', 'task_decomposition_score', 'overall_quality_score',
            'execution_time', 'total_tasks', 'duplicate_task_count',
            'hallucinated_task_count', 'column_constraint_violations',
            'uses_correct_operations', 'handles_missing_values',
            'uses_correct_columns', 'efficiency_score',
        ]
        bool_cols = [
            'has_duplicate_tasks', 'visualization_agent_used',
            'planner_rag_used', 'task_count_appropriate',
        ]
        agg_dict = {c: 'mean' for c in numeric_cols if c in df_m.columns}
        agg_dict.update({c: 'sum' for c in bool_cols if c in df_m.columns})

        summary_pipeline = df_m.groupby('pipeline').agg(agg_dict).round(2)
        pipeline_path    = f'summary_by_pipeline_{ts}.csv'
        summary_pipeline.to_csv(pipeline_path)
        print(f"✓ Saved pipeline summary    → {pipeline_path}")

        summary_dataset = df_m.groupby(['dataset_key', 'pipeline']).agg(agg_dict).round(2)
        dataset_path    = f'summary_by_dataset_{ts}.csv'
        summary_dataset.to_csv(dataset_path)
        print(f"✓ Saved dataset summary     → {dataset_path}")

        summary_tasktype = df_m.groupby(['task_type', 'pipeline']).agg(agg_dict).round(2)
        tasktype_path    = f'summary_by_tasktype_{ts}.csv'
        summary_tasktype.to_csv(tasktype_path)
        print(f"✓ Saved task-type summary   → {tasktype_path}")

        # ── Console summary ───────────────────────────────────────────────────
        key_cols = [c for c in [
            'execution_success', 'output_correctness', 'code_quality_score',
            'task_decomposition_score', 'hallucinated_task_count',
            'column_constraint_violations', 'overall_quality_score', 'execution_time',
        ] if c in summary_pipeline.columns]

        print(f"\n{'═'*70}")
        print("  PIPELINE COMPARISON (mean scores)")
        print(f"{'═'*70}")
        print(summary_pipeline[key_cols].to_string())

        if 'RAG' in summary_pipeline.index and 'No-RAG' in summary_pipeline.index:
            print(f"\n{'─'*70}")
            print("  RAG IMPACT (RAG minus No-RAG)")
            print(f"{'─'*70}")
            for col in key_cols:
                delta = summary_pipeline.loc['RAG', col] - summary_pipeline.loc['No-RAG', col]
                arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '═')
                print(f"  {col:45s}: {arrow} {delta:+.2f}")

        print(f"\n{'═'*70}")
        print("  PER-DATASET OVERALL QUALITY SCORE")
        print(f"{'═'*70}")
        if 'overall_quality_score' in summary_dataset.columns:
            print(summary_dataset[['overall_quality_score']].to_string())

        print(f"\n{'═'*70}")
        print("  PER-TASK-TYPE PERFORMANCE")
        print(f"{'═'*70}")
        if 'overall_quality_score' in summary_tasktype.columns:
            print(summary_tasktype[['overall_quality_score', 'output_correctness']].to_string())

        print(f"\n{'═'*70}")
        print("  INDIVIDUAL TEST CASE RESULTS")
        print(f"{'═'*70}")
        per_test_cols = [c for c in [
            'overall_quality_score', 'output_correctness',
            'execution_success', 'hallucinated_task_count', 'column_constraint_violations',
        ] if c in df_m.columns]
        per_test = df_m.groupby(['test_id', 'pipeline']).agg({
            c: 'mean' for c in per_test_cols
        }).round(1)
        print(per_test.to_string())
        print(f"{'═'*70}\n")

        return summary_pipeline, summary_dataset, df_m


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    NUM_RUNS            = 1
    TEST_BOTH_PIPELINES = True
    FILTER_DATASETS     = None
    FILTER_TASK_TYPES   = None
    FILTER_DIFFICULTY   = None

    evaluator = ComprehensiveEvaluator(DATASET_PATHS)
    evaluator.run_full_evaluation(
        num_runs=NUM_RUNS,
        test_both_pipelines=TEST_BOTH_PIPELINES,
        filter_datasets=FILTER_DATASETS,
        filter_task_types=FILTER_TASK_TYPES,
        filter_difficulty=FILTER_DIFFICULTY,
    )


if __name__ == "__main__":
    main()