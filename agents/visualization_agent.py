# visualization_agent.py - VERSION 5
# Architectural changes from V4:
#
#  1. generate_code() — COLUMN RESTRICTION ACTIVE suppressed for full-dataset tasks:
#     ML tasks (and any task where target_columns equals all dataset columns)
#     no longer trigger the COLUMN RESTRICTION ACTIVE block. Previously the
#     block fired whenever column_constraint=True, which is always set for ML
#     tasks even though they receive the full column list. The block now only
#     renders when target_columns is a genuine subset of the dataset columns.
#     This mirrors the same fix applied in analytical_agent V4.
#
#  2. generate_code() — DECLARED DEPENDENCIES block injected from requires/produces:
#     When task_metadata carries a non-empty 'requires' list (set by planner V4),
#     a concise DECLARED DEPENDENCIES section is injected into the prompt.
#     This tells the LLM exactly which context keys are guaranteed present before
#     the function runs, replacing the vague "check context for artifacts from
#     prior tasks" guidance. Particularly important for ML visualization tasks
#     (feature importance, confusion matrix, predictions vs actuals) where
#     requires=['model:fitted', 'X_test_scaled:ndarray', 'y_test:Series'] etc.
#     The produces list is also shown when non-empty (rare for viz tasks but
#     present for completeness).
#
#  All other behaviour (normalise, output contract, _get_task_guidance,
#  _build_context_hint, rag_hint injection, _call_solution, run()) unchanged from V4.

from llm_utils import call_llm
import pandas as pd
import numpy as np
import traceback
import inspect


class VisualizationAgent:
    """
    Generates Python code to produce visualization files.
    Execution is delegated to TestingAgent (which handles retries).

    Output contract (unchanged):
        {'status': 'success', 'file': 'output.png'}   on success
        {'status': 'error',   'error': '<message>'}    on failure
    """

    DEFAULT_OUTPUT = 'output.png'

    def __init__(self, llm):
        self.llm = llm

    # ── context description  (unchanged from V4) ──────────────────────────────

    @staticmethod
    def _build_context_hint(context: dict) -> str:
        if not context:
            return "  (empty — no prior pipeline state available)"

        lines = []
        for key, val in context.items():
            type_name = type(val).__name__

            if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                lines.append(
                    f"  context['{key}'] → {type_name}  "
                    f"shape={val.shape}  dtype={val.dtype}"
                )
            elif isinstance(val, pd.DataFrame):
                lines.append(
                    f"  context['{key}'] → DataFrame  "
                    f"shape={val.shape}  columns={list(val.columns)}"
                )
            elif isinstance(val, pd.Series):
                lines.append(
                    f"  context['{key}'] → Series  "
                    f"len={len(val)}  dtype={val.dtype}  name={val.name}"
                )
            elif isinstance(val, pd.Index):
                lines.append(f"  context['{key}'] → Index  len={len(val)}")
            elif isinstance(val, (list, tuple)):
                lines.append(f"  context['{key}'] → {type_name}  len={len(val)}")
            elif isinstance(val, dict):
                lines.append(f"  context['{key}'] → dict  keys={list(val.keys())}")
            elif isinstance(val, (int, float, np.integer, np.floating)):
                lines.append(f"  context['{key}'] → {type_name}  value={val}")
            elif hasattr(val, 'fit') or hasattr(val, 'predict') or hasattr(val, 'transform'):
                extra = ""
                if hasattr(val, 'coef_'):
                    extra = f"  coef_shape={np.array(val.coef_).shape}"
                elif hasattr(val, 'scale_'):
                    extra = f"  n_features={len(val.scale_)}"
                elif hasattr(val, 'n_features_in_'):
                    extra = f"  n_features_in={val.n_features_in_}"
                lines.append(
                    f"  context['{key}'] → {type_name}{extra}  [fitted sklearn object]"
                )
            elif isinstance(val, str):
                preview = val[:60] + "..." if len(val) > 60 else val
                lines.append(f"  context['{key}'] → str  value='{preview}'")
            else:
                lines.append(f"  context['{key}'] → {type_name}")

        return "\n".join(lines)

    # ── task-specific guidance  (unchanged from V4) ───────────────────────────

    @staticmethod
    def _get_task_guidance(task: str) -> str:
        t = task.lower()

        if any(kw in t for kw in (
            'predicted vs actual', 'predictions vs actual',
            'actual vs predicted', 'scatter.*predict',
            'predict.*scatter', 'model prediction'
        )):
            return (
                "TASK TYPE: Predictions vs Actuals plot.\n"
                "Read from context:\n"
                "    model  = context.get('model')\n"
                "    X_test = context.get('X_test_scaled') or context.get('X_test')\n"
                "    y_test = context.get('y_test')\n"
                "Generate predictions: y_pred = model.predict(X_test)\n"
                "Plot y_test (x-axis) vs y_pred (y-axis) as a scatter plot.\n"
                "Add a diagonal reference line (perfect prediction line).\n"
                "If context keys are missing, fall back to computing predictions from df."
            )

        if any(kw in t for kw in ('residual', 'error plot', 'prediction error')):
            return (
                "TASK TYPE: Residual plot.\n"
                "Read from context:\n"
                "    model  = context.get('model')\n"
                "    X_test = context.get('X_test_scaled') or context.get('X_test')\n"
                "    y_test = context.get('y_test')\n"
                "Compute residuals: residuals = y_test - model.predict(X_test)\n"
                "Plot fitted values (x-axis) vs residuals (y-axis).\n"
                "Add a horizontal reference line at y=0.\n"
                "If context keys are missing, fall back to computing from df."
            )

        if any(kw in t for kw in (
            'feature importance', 'feature weight',
            'coefficient', 'coef plot', 'variable importance'
        )):
            return (
                "TASK TYPE: Feature importance / coefficient plot.\n"
                "Read from context:\n"
                "    model = context.get('model')\n"
                "For linear models use model.coef_; for tree models use model.feature_importances_.\n"
                "Get feature names from context.get('X_train') columns if available,\n"
                "otherwise use generic labels.\n"
                "Plot as a horizontal bar chart sorted by absolute importance.\n"
                "If context keys are missing, fall back to fitting a fresh model from df."
            )

        if any(kw in t for kw in ('confusion matrix', 'confusion_matrix', 'classification report')):
            return (
                "TASK TYPE: Confusion matrix plot.\n"
                "Read from context:\n"
                "    model  = context.get('model')\n"
                "    X_test = context.get('X_test_scaled') or context.get('X_test')\n"
                "    y_test = context.get('y_test')\n"
                "Compute: from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n"
                "Use ConfusionMatrixDisplay to render the matrix.\n"
                "If context keys are missing, fall back to computing from df."
            )

        if any(kw in t for kw in ('learning curve', 'training curve', 'loss curve', 'train.*loss')):
            return (
                "TASK TYPE: Learning curve plot.\n"
                "Use sklearn.model_selection.learning_curve to compute train/val scores.\n"
                "Read model type from context.get('model') if available.\n"
                "Plot training score and cross-validation score against training set size.\n"
                "Add shaded regions for score variance."
            )

        if any(kw in t for kw in ('distribution', 'histogram', 'hist', 'density', 'kde')):
            return (
                "TASK TYPE: Distribution plot.\n"
                "Use seaborn.histplot() or seaborn.kdeplot() as appropriate.\n"
                "Handle missing values with dropna() before plotting.\n"
                "Use a sensible number of bins (bins='auto' or explicit count).\n"
                "Add a KDE overlay when plotting histograms if appropriate."
            )

        if any(kw in t for kw in ('heatmap', 'correlation matrix', 'corr matrix')):
            return (
                "TASK TYPE: Correlation heatmap.\n"
                "Compute df[numeric_cols].corr() and pass to seaborn.heatmap().\n"
                "Use annot=True, fmt='.2f', cmap='coolwarm'.\n"
                "Set figure size large enough to read all labels."
            )

        if any(kw in t for kw in ('time series', 'line chart', 'line plot', 'over time', 'trend')):
            return (
                "TASK TYPE: Time series / line plot.\n"
                "Parse the date/time column with pd.to_datetime() before plotting.\n"
                "Sort by the date column before plotting.\n"
                "Use plt.plot() or seaborn.lineplot().\n"
                "Rotate x-axis labels if there are many date ticks."
            )

        if any(kw in t for kw in ('box plot', 'boxplot', 'violin', 'box and whisker')):
            return (
                "TASK TYPE: Box / violin plot.\n"
                "Use seaborn.boxplot() or seaborn.violinplot().\n"
                "Handle outliers gracefully — do not clip data unless specified.\n"
                "Label axes clearly with units where applicable."
            )

        if any(kw in t for kw in ('bar chart', 'bar plot', 'bar graph', 'grouped bar')):
            return (
                "TASK TYPE: Bar chart.\n"
                "Aggregate data with groupby() before plotting.\n"
                "Use seaborn.barplot() or plt.bar().\n"
                "Sort bars by value unless a natural order exists.\n"
                "Add value labels on bars if there are fewer than 15 categories."
            )

        if any(kw in t for kw in ('scatter', 'scatter plot', 'scatter chart')):
            return (
                "TASK TYPE: Scatter plot.\n"
                "Use seaborn.scatterplot() or plt.scatter().\n"
                "Handle overplotting with alpha transparency if many points.\n"
                "Add axis labels and a title with the variable names."
            )

        return ""

    # ── declared dependencies prompt block [NEW in V5] ────────────────────────

    @staticmethod
    def _build_dependencies_block(requires: list, produces: list) -> str:
        """
        Build a concise prompt section listing which context keys are
        guaranteed present (requires) and which must be written (produces).

        Mirrors the same method in analytical_agent V4.
        For visualization tasks, requires is the important list — it tells
        the LLM which ML artifacts are already available so it doesn't
        unnecessarily fall back to recomputing from df.
        produces is rare for visualization tasks but included for completeness.
        Only rendered when at least one list is non-empty.
        """
        if not requires and not produces:
            return ""

        lines = ["=" * 70,
                 "CONTEXT CONTRACT FOR THIS TASK",
                 "=" * 70]

        if requires:
            lines.append("The following context keys are GUARANTEED to be present.")
            lines.append("USE THEM DIRECTLY — do not recompute from df:\n")
            for key_hint in requires:
                key  = key_hint.split(':')[0]
                hint = key_hint.split(':', 1)[1] if ':' in key_hint else ''
                type_note = f"  ({hint})" if hint else ""
                lines.append(f"  context['{key}']{type_note}")

        if requires and produces:
            lines.append("")

        if produces:
            lines.append("This task must write the following keys to context")
            lines.append("BEFORE the return statement:\n")
            for key_hint in produces:
                key  = key_hint.split(':')[0]
                hint = key_hint.split(':', 1)[1] if ':' in key_hint else ''
                type_note = f"  ({hint})" if hint else ""
                lines.append(f"  context['{key}'] = ...{type_note}")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── code generation ───────────────────────────────────────────────────────

    def generate_code(self, task: str, df: pd.DataFrame,
                      task_metadata: dict = None,
                      context: dict = None,
                      rag_hint: str = "") -> str:
        """
        Generate a solution(df, context={}) function that produces and saves a chart.

        V5 changes:
          (a) COLUMN RESTRICTION ACTIVE suppressed when target_columns equals
              the full dataset column list — no genuine restriction is active.
          (b) DECLARED DEPENDENCIES block injected when task_metadata carries
              non-empty requires/produces lists from planner V4.
        """
        if context is None:
            context = {}

        target_columns    = []
        column_constraint = False
        requires          = []
        produces          = []
        all_df_columns    = df.columns.tolist()

        if task_metadata:
            raw_cols = task_metadata.get('target_columns', [])
            target_columns    = [c for c in raw_cols if c in df.columns]
            column_constraint = bool(task_metadata.get('column_constraint', False))
            requires          = task_metadata.get('requires', [])
            produces          = task_metadata.get('produces', [])
            # Pick up rag_hint from metadata if not passed directly
            if not rag_hint:
                rag_hint = task_metadata.get('rag_hint', '')

        # V5: suppress COLUMN RESTRICTION ACTIVE when target_columns is the
        # full dataset — identical logic to analytical_agent V4.
        is_genuine_restriction = (
            column_constraint
            and target_columns
            and set(target_columns) != set(all_df_columns)
        )

        if is_genuine_restriction:
            col_info   = self._column_info(df, target_columns)
            dtype_info = self._dtype_info(df, target_columns)
            sample     = df[target_columns].head(3).to_string()
            constraint_block = (
                "COLUMN RESTRICTION ACTIVE\n"
                f"Visualise ONLY these columns: {', '.join(target_columns)}\n"
                "Using any other column violates the task specification.\n"
            )
            dataset_note = (
                f"Dataset shape: {df.shape[0]} rows "
                f"(showing {len(target_columns)} of {len(all_df_columns)} columns)"
            )
            print(f"[VisualizationAgent] Genuine column restriction active: {target_columns}")
        else:
            col_info   = self._column_info(df)
            dtype_info = self._dtype_info(df)
            sample     = df.head(3).to_string()
            constraint_block = ""
            dataset_note = f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns"
            if column_constraint and target_columns:
                print(f"[VisualizationAgent] Full-dataset task — no column restriction applied")

        task_guidance      = self._get_task_guidance(task)
        context_hint       = self._build_context_hint(context)
        dependencies_block = self._build_dependencies_block(requires, produces)
        has_context        = bool(context)

        # RAG hint section — only rendered when a hint is provided (unchanged from V4)
        rag_section = (
            f"{'=' * 70}\n"
            f"RAG HINT — insight from a similar solved problem\n"
            f"{'=' * 70}\n"
            f"{rag_hint.strip()}\n"
            f"{'=' * 70}\n"
        ) if rag_hint and rag_hint.strip() else ""

        if has_context:
            context_section = f"""
{"=" * 70}
PIPELINE CONTEXT — artifacts produced by previous tasks
{"=" * 70}
{context_hint}

RULES FOR USING CONTEXT IN VISUALIZATIONS:
  • Access values with context.get('key') — never context['key'] (avoids KeyError)
  • For ML plots: USE pre-computed artifacts (model, X_test_scaled, y_test)
    directly from context — do NOT recompute from df unless keys are absent.
  • Visualization tasks should generally READ from context, not write to it.
  • If needed keys are absent, fall back gracefully to computing from df.
"""
        else:
            context_section = f"""
{"=" * 70}
PIPELINE CONTEXT
{"=" * 70}
{context_hint}
"""

        prompt = f"""You are a data visualisation expert. \
Write Python code to produce a clear, well-labelled chart and save it to disk.

{constraint_block}
{task_guidance}

{rag_section}
{"=" * 70}
DATASET INFORMATION
{"=" * 70}
Available columns : {col_info}
Data types        : {dtype_info}
{dataset_note}

Sample data (first 3 rows):
{sample}
{context_section}
{dependencies_block}
{"=" * 70}
TASK
{"=" * 70}
{task}

{"=" * 70}
INSTRUCTIONS — follow every point exactly
{"=" * 70}
1.  Write ONLY Python code — no prose, no markdown backticks, no explanations.
2.  Define a function exactly named solution(df, context={{}}) — always include context.
3.  Import libraries INSIDE the function (matplotlib, seaborn, etc.).
4.  Use EXACT column names as shown above. Use bracket notation: df['col name'].
5.  If the CONTEXT CONTRACT above lists guaranteed keys, USE THEM directly.
6.  If context keys are missing, fall back gracefully to computing from df.
7.  Handle missing values before plotting: dropna(subset=[...]) on relevant columns.
8.  Set a clear figure size: plt.figure(figsize=(10, 6))
9.  Add a descriptive title, axis labels, and a legend where appropriate.
10. Save the figure:  plt.savefig('{self.DEFAULT_OUTPUT}', bbox_inches='tight', dpi=150)
11. Close the figure: plt.close()
12. Do NOT call plt.show() — the code runs in a non-interactive environment.
13. Return EXACTLY this dict (nothing else):
        {{'status': 'success', 'file': '{self.DEFAULT_OUTPUT}'}}

Write ONLY the Python function — start with: def solution(df, context={{}}):
"""

        code = call_llm(self.llm, prompt)
        return self._clean_code(code)

    # ── normalisation  (unchanged from V4) ────────────────────────────────────

    @staticmethod
    def normalise(raw) -> dict:
        if isinstance(raw, dict):
            if 'status' in raw:
                if raw.get('status') == 'success' and 'file' not in raw:
                    raw['file'] = VisualizationAgent.DEFAULT_OUTPUT
                return raw
            for k in ('file', 'output', 'path'):
                if k in raw:
                    return {'status': 'success', 'file': str(raw[k])}
            return {'status': 'success', 'file': VisualizationAgent.DEFAULT_OUTPUT}

        if isinstance(raw, str):
            fname = raw.split(':')[-1].strip() if ':' in raw else raw.strip()
            return {'status': 'success', 'file': fname or VisualizationAgent.DEFAULT_OUTPUT}

        return {'status': 'success', 'file': VisualizationAgent.DEFAULT_OUTPUT}

    # ── run()  (unchanged from V4) ────────────────────────────────────────────

    def run(self, task: str, df: pd.DataFrame,
            task_metadata: dict = None,
            testing_agent=None,
            context: dict = None) -> dict:
        if context is None:
            context = {}

        rag_hint = ""
        if task_metadata:
            rag_hint = task_metadata.get('rag_hint', '')

        code = self.generate_code(task, df, task_metadata,
                                  context=context, rag_hint=rag_hint)

        if testing_agent is not None:
            output = testing_agent.run_solution(code, df, llm=self.llm, context=context)
            if output['ok']:
                return self.normalise(output['result'])
            else:
                return {'status': 'error', 'error': output['error']}
        else:
            return self._execute_direct(code, df, context=context)

    def _execute_direct(self, code: str, df: pd.DataFrame,
                        context: dict = None) -> dict:
        if context is None:
            context = {}

        namespace = {}
        try:
            exec(compile(code, '<solution>', 'exec'), namespace)
        except SyntaxError as e:
            return {'status': 'error', 'error': f"SyntaxError: {e}"}

        fn = namespace.get('solution')
        if fn is None:
            return {'status': 'error',
                    'error': "No function named 'solution' found in generated code."}

        try:
            raw = _call_solution(fn, df.copy(), context)
        except Exception:
            return {'status': 'error', 'error': traceback.format_exc(limit=5)}

        return self.normalise(raw)

    # ── helpers  (unchanged from V4) ──────────────────────────────────────────

    def _column_info(self, df: pd.DataFrame, columns=None) -> str:
        cols = columns if columns else df.columns
        return ", ".join(f"'{c}'" for c in cols if c in df.columns)

    def _dtype_info(self, df: pd.DataFrame, columns=None) -> str:
        cols = columns if columns else df.columns
        pairs = [f"'{c}': {df[c].dtype}" for c in cols if c in df.columns]
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def _clean_code(code: str) -> str:
        code = str(code).strip()
        if code.startswith('```'):
            lines = code.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            code = '\n'.join(lines)
        return code.strip()

    def _get_column_info(self, df: pd.DataFrame) -> str:
        return self._column_info(df)

    def _get_column_info_filtered(self, df: pd.DataFrame, columns: list) -> str:
        return self._column_info(df, columns)

    def _get_dtypes_info(self, df: pd.DataFrame) -> str:
        return self._dtype_info(df)

    def _get_dtypes_info_filtered(self, df: pd.DataFrame, columns: list) -> str:
        return self._dtype_info(df, columns)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPER  (unchanged from V4)
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