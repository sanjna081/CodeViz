# analytical_agent.py - VERSION 5
# Changes from V4:
#
#  1. No functional changes — this file was already compatible with the
#     OpenRouter/LLMClient backend introduced in main.py V4.
#
#     Reason: the only LLM call in this file is in generate_code(), which
#     routes through call_llm() from llm_utils.py (see bottom of
#     generate_code()). llm_utils V2 probes .complete() first, so LLMClient
#     is fully supported without any changes here.
#     There are no direct self.llm.invoke() calls, no langchain imports,
#     and no ChatOllama references anywhere in this file.
#
#  2. Version bump to V5 for consistency with the rest of the V5 stack
#     (main V4, llm_utils V2, planner V5, analyzer V3, testing V5).

from llm_utils import call_llm
import pandas as pd
import numpy as np
import os
import inspect


# ══════════════════════════════════════════════════════════════════════════════
#  FEW-SHOT EXAMPLES  (unchanged from V4)
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = r"""
CORE EXAMPLES
=============

# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1 — Simple correlation between two numeric columns
# ──────────────────────────────────────────────────────────────────────────────
# TASK: What is the Pearson correlation between alcohol and quality?

def solution(df, context={}):
    import pandas as pd
    corr_val = float(df['alcohol'].corr(df['quality']))
    return {'correlation': corr_val}

# CORRECT OUTPUT SHAPE: {'correlation': 0.4762}
# NEVER return: {'correlation': '0.48'} or scipy.stats.PearsonRResult


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2 — Groupby aggregation: one metric per group → flat dict
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Calculate the survival rate for each passenger class.

def solution(df, context={}):
    import pandas as pd
    result = df.groupby('Pclass')['Survived'].mean()
    return {str(k): float(v) for k, v in result.items()}

# CORRECT OUTPUT SHAPE: {'1': 0.6296, '2': 0.4728, '3': 0.2424}
# NEVER return: {'1': {'survival_rate': '63%', 'count': 216}}
# NEVER return: {'1': '63.00%'}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3 — Conditional filter then percentage
# ──────────────────────────────────────────────────────────────────────────────
# TASK: What percentage of wines have a quality score of 7 or higher?

def solution(df, context={}):
    import pandas as pd
    pct = float((df['quality'] >= 7).mean() * 100)
    return {'percentage': pct}

# CORRECT OUTPUT SHAPE: {'percentage': 13.5710}
# NEVER return: {'percentage': '13.57%'}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 4 — Missing value count
# ──────────────────────────────────────────────────────────────────────────────
# TASK: How many missing values are in the total_bedrooms column?

def solution(df, context={}):
    import pandas as pd
    count = int(df['total_bedrooms'].isna().sum())
    return {'missing_count': count}

# CORRECT OUTPUT SHAPE: {'missing_count': 207}
# NEVER return: len(df['total_bedrooms'].isnull())


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 5 — Null handling before a statistic
# ──────────────────────────────────────────────────────────────────────────────
# TASK: What is the median age? Fill missing Age values with the median first.

def solution(df, context={}):
    import pandas as pd
    median_age = df['Age'].median()
    age_filled = df['Age'].fillna(median_age)
    result = float(age_filled.median())
    return {'median': result}

# CORRECT OUTPUT SHAPE: {'median': 28.0}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 6 — Feature engineering then aggregate
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Create rooms_per_household = total_rooms / households.
#        What is the standard deviation of this new feature?

def solution(df, context={}):
    import pandas as pd
    df = df.copy()
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    std_val = float(df['rooms_per_household'].std())
    return {'std': std_val}

# CORRECT OUTPUT SHAPE: {'std': 2.4742}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 7 — Groupby with median (categorical groups)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Calculate the median house value for each ocean_proximity category.

def solution(df, context={}):
    import pandas as pd
    result = df.groupby('ocean_proximity')['median_house_value'].median()
    return {str(k): float(v) for k, v in result.items()}

# CORRECT OUTPUT SHAPE:
#   {'<1H OCEAN': 179700.0, 'INLAND': 119600.0, 'ISLAND': 380440.0, ...}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 8 — String-encoded boolean in groupby (Yes/No attrition column)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Calculate the attrition rate for each job role.
#        The Attrition column contains "Yes" or "No" strings.

def solution(df, context={}):
    import pandas as pd
    result = df.groupby('JobRole')['Attrition'].apply(
        lambda x: float((x == 'Yes').mean())
    )
    return {str(k): float(v) for k, v in result.items()}

# CORRECT OUTPUT SHAPE:
#   {'Healthcare Representative': 0.0698, 'Human Resources': 0.2308, ...}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 9 — String cleaning before aggregation (price column with $ and ,)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: What is the average listing price for each neighbourhood group?

def solution(df, context={}):
    import pandas as pd
    df = df.copy()
    df['price_clean'] = (
        df['price'].astype(str)
                   .str.replace(r'[\$,]', '', regex=True)
                   .str.strip()
                   .pipe(pd.to_numeric, errors='coerce')
    )
    result = df.groupby('neighbourhood_group')['price_clean'].mean()
    return {str(k): float(v) for k, v in result.items()}

# CORRECT OUTPUT SHAPE:
#   {'Bronx': 87.49, 'Brooklyn': 124.38, 'Manhattan': 196.87, ...}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 10 — String parsing to extract primary genre, then groupby
# ──────────────────────────────────────────────────────────────────────────────
# TASK: What is the average revenue per primary genre?

def solution(df, context={}):
    import pandas as pd
    df = df.copy()
    df = df[df['revenue'] > 0].copy()
    df['primary_genre'] = (
        df['genres'].astype(str)
                    .str.split(r'[,|]')
                    .str[0]
                    .str.strip()
    )
    result = df.groupby('primary_genre')['revenue'].mean()
    return {str(k): float(v) for k, v in result.items()}

# CORRECT OUTPUT SHAPE:
#   {'Action': 226812866.54, 'Comedy': 112453102.21, 'Drama': 67334891.08, ...}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 11 — Decade extraction then groupby
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Calculate the average vote score for each decade.

def solution(df, context={}):
    import pandas as pd
    df = df.copy()
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df = df.dropna(subset=['release_year', 'vote_average'])
    df['decade'] = (df['release_year'].astype(int) // 10 * 10)
    result = df.groupby('decade')['vote_average'].mean()
    return {str(k): float(v) for k, v in result.items()}

# CORRECT OUTPUT SHAPE:
#   {'1960': 6.812, '1970': 6.541, '1980': 6.234, ...}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 12 — Filtered correlation (exclude zero values)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: What is the correlation between budget and revenue?

def solution(df, context={}):
    import pandas as pd
    filtered = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    corr_val = float(filtered['budget'].corr(filtered['revenue']))
    return {'correlation': corr_val}

# CORRECT OUTPUT SHAPE: {'correlation': 0.7314}


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 13 — DATASET SPLITTING (stateful — MUST write to context)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Split the data into training and testing sets.

def solution(df, context={}):
    from sklearn.model_selection import train_test_split

    feature_cols = ['feature1', 'feature2', 'feature3']   # replace with actual features
    target_col   = 'target'                                # replace with actual target

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── MANDATORY: write all four splits to context ──────────────────────────
    context['X_train'] = X_train
    context['X_test']  = X_test
    context['y_train'] = y_train
    context['y_test']  = y_test
    # ────────────────────────────────────────────────────────────────────────

    return {'train_size': float(len(y_train)), 'test_size': float(len(y_test))}

# KEY RULES FOR SPLIT TASKS:
#   - ALWAYS store X_train, X_test, y_train, y_test in context
#   - NEVER include the target column inside X
#   - Write to context BEFORE the return statement
#   - Return only display-friendly scalars in the dict


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 14 — FEATURE SCALING (stateful — reads AND writes context)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Scale features using StandardScaler.

def solution(df, context={}):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np

    X_train = context.get('X_train')
    X_test  = context.get('X_test')

    if X_train is None or X_test is None:
        feature_cols = [c for c in df.columns if c != 'target']
        X = df[feature_cols]
        y = df['target']
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── MANDATORY: write scaled arrays and fitted scaler back to context ─────
    context['X_train_scaled'] = X_train_scaled
    context['X_test_scaled']  = X_test_scaled
    context['scaler']         = scaler
    # ────────────────────────────────────────────────────────────────────────

    return {
        'train_mean': float(X_train_scaled.mean()),
        'train_std':  float(X_train_scaled.std()),
    }

# KEY RULES FOR SCALING TASKS:
#   - ALWAYS use context.get('X_train') and context.get('X_test') — never df
#   - scaler.fit() on X_train ONLY
#   - scaler.transform() on both X_train and X_test separately
#   - Store context['X_train_scaled'], context['X_test_scaled'], context['scaler']
#   - Return scalars only: .mean() and .std() with no axis argument


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 15 — MODEL TRAINING (stateful — reads context, writes model)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Train a model using the scaled training data from the previous step.

def solution(df, context={}):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X_train_scaled = context.get('X_train_scaled')
    if X_train_scaled is not None:
        X_train = X_train_scaled
    else:
        X_train = context.get('X_train')
    y_train = context.get('y_train')

    if X_train is None or y_train is None:
        from sklearn.model_selection import train_test_split
        feature_cols = [c for c in df.columns if c != 'target']
        X_train, _, y_train, _ = train_test_split(
            df[feature_cols], df['target'], test_size=0.2, random_state=42
        )

    if context.get('X_train_scaled') is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = context.get('X_test')
        if X_test is not None:
            context['X_test_scaled'] = scaler.transform(X_test)
        context['scaler']         = scaler
        context['X_train_scaled'] = X_train

    model = LinearRegression()
    model.fit(X_train, y_train)

    # ── MANDATORY: store fitted model in context ─────────────────────────────
    context['model'] = model
    # ────────────────────────────────────────────────────────────────────────

    return {'coefficient_count': len(model.coef_), 'intercept': float(model.intercept_)}

# KEY RULES FOR TRAINING TASKS:
#   - Prefer X_train_scaled over X_train when both are present
#   - NEVER use X_test or y_test during training
#   - Store context['model'] before returning


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE 16 — MODEL EVALUATION (stateful — reads context only)
# ──────────────────────────────────────────────────────────────────────────────
# TASK: Evaluate the trained model using RMSE and R² on the test set.

def solution(df, context={}):
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    model  = context.get('model')
    y_test = context.get('y_test')

    X_test_scaled = context.get('X_test_scaled')
    if X_test_scaled is not None:
        X_test = X_test_scaled
    else:
        X_test = context.get('X_test')

    if model is None or X_test is None or y_test is None:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        feature_cols = [c for c in df.columns if c != 'target']
        X = df[feature_cols]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))

    return {'rmse': rmse, 'r2': r2}

# KEY RULES FOR EVALUATION TASKS:
#   - ALWAYS evaluate on X_test / y_test — NEVER on training data
#   - Prefer X_test_scaled if the scaler was applied upstream
#   - Evaluation tasks only READ from context — do not write new artifacts

END OF EXAMPLES
"""


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT FORMATTER  (unchanged from V4)
# ══════════════════════════════════════════════════════════════════════════════

class ResultFormatter:

    @staticmethod
    def format_for_display(task: str, raw_result) -> str:
        if raw_result is None:
            return "No result produced."

        lines = [f"Task: {task}", ""]

        if isinstance(raw_result, dict):
            scalar_keys = {k: v for k, v in raw_result.items()
                           if isinstance(v, (int, float))}
            if len(raw_result) == 1 and scalar_keys:
                key, val = next(iter(scalar_keys.items()))
                lines.append(ResultFormatter._format_scalar_line(key, val))
            elif scalar_keys and len(scalar_keys) == len(raw_result):
                lines.append("Results by group:")
                max_key_len = max(len(str(k)) for k in scalar_keys)
                for k, v in sorted(scalar_keys.items()):
                    lines.append(
                        f"  {str(k):<{max_key_len}}  →  {ResultFormatter._fmt_val(k, v)}"
                    )
            else:
                for k, v in raw_result.items():
                    if isinstance(v, (int, float)):
                        lines.append(f"  {k}: {ResultFormatter._fmt_val(k, v)}")
                    else:
                        lines.append(f"  {k}: {v}")

        elif isinstance(raw_result, (int, float)):
            lines.append(f"Result: {ResultFormatter._fmt_val('result', raw_result)}")
        elif isinstance(raw_result, list):
            lines.append(f"Returned {len(raw_result)} item(s).")
        else:
            lines.append(str(raw_result))

        return "\n".join(lines)

    @staticmethod
    def _fmt_val(key: str, val: float) -> str:
        key_lower = str(key).lower()
        if any(kw in key_lower for kw in ('pct', 'percent', 'percentage', 'rate', 'proportion')):
            return f"{val:.2f}%"
        if any(kw in key_lower for kw in ('corr', 'correlation')):
            return f"{val:.4f}"
        if any(kw in key_lower for kw in ('count', 'missing', 'null', 'outlier')):
            return f"{int(round(val)):,}"
        if abs(val) >= 10_000:
            return f"{val:,.2f}"
        return f"{val:.4f}"

    @staticmethod
    def _format_scalar_line(key: str, val: float) -> str:
        label = key.replace('_', ' ').capitalize()
        return f"{label}: {ResultFormatter._fmt_val(key, val)}"


# ══════════════════════════════════════════════════════════════════════════════
#  CONTEXT VALIDATION HELPERS  [V4 — module level for reuse]
# ══════════════════════════════════════════════════════════════════════════════

# Maps the type-hint suffix used in produces/requires lists to a callable
# that checks whether a context value satisfies that type.
_TYPE_CHECKERS = {
    'DataFrame': lambda v: isinstance(v, pd.DataFrame),
    'Series':    lambda v: isinstance(v, pd.Series),
    'ndarray':   lambda v: hasattr(v, 'shape') and hasattr(v, 'dtype'),
    'fitted':    lambda v: (hasattr(v, 'predict') or hasattr(v, 'transform'))
                           and hasattr(v, '__class__'),
}


def _check_context_value_type(key_with_hint: str, value) -> bool:
    """
    Given a key string like 'X_train:DataFrame' or 'model:fitted',
    check whether the value in context satisfies the declared type hint.
    Returns True if the check passes or if no hint is present.
    """
    if ':' not in key_with_hint:
        return True
    _, hint = key_with_hint.split(':', 1)
    checker = _TYPE_CHECKERS.get(hint)
    if checker is None:
        return True   # unknown hint — don't block
    return checker(value)


def _validate_produces(produces: list, context: dict, task_desc: str) -> None:
    """
    Check that every key declared in `produces` is present in context and
    has a value matching the declared type hint.
    Raises RuntimeError listing all failures so the testing agent can retry.
    """
    if not produces:
        return

    missing    = []
    wrong_type = []

    for key_hint in produces:
        key = key_hint.split(':')[0]
        if key not in context:
            missing.append(key)
        elif not _check_context_value_type(key_hint, context[key]):
            hint   = key_hint.split(':', 1)[1] if ':' in key_hint else 'unknown'
            actual = type(context[key]).__name__
            wrong_type.append(f"'{key}' expected {hint}, got {actual}")

    if missing or wrong_type:
        lines = [f"Task '{task_desc[:60]}' completed but context contract was violated."]
        if missing:
            lines.append(f"  Missing keys: {missing}")
            lines.append("  Add to solution() before return:")
            for key_hint in produces:
                key = key_hint.split(':')[0]
                if key in missing:
                    lines.append(f"      context['{key}'] = {key}")
        if wrong_type:
            lines.append(f"  Wrong types: {wrong_type}")
        raise RuntimeError("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICAL AGENT
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticalAgent:
    """
    Generates Python code to solve analytical tasks on a DataFrame.
    Execution is delegated to TestingAgent (which handles retries).

    V4 changes (retained in V5):
      - _validate_context_writes() reads 'produces' from task_metadata when
        available, falling back to keyword matching for legacy paths.
      - generate_code() suppresses COLUMN RESTRICTION ACTIVE when
        target_columns equals the full dataset (no real restriction).
      - generate_code() injects a DECLARED DEPENDENCIES section when
        task_metadata carries a non-empty 'requires' list.
      - run() passes task_metadata to the context validator.
    """

    def __init__(self, llm, examples_file_path: str = None):
        self.llm       = llm
        self.formatter = ResultFormatter()
        self.few_shot_examples = self._load_examples(examples_file_path)

    # ── example loading ───────────────────────────────────────────────────────

    def _load_examples(self, path: str) -> str:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'CORE EXAMPLES' in content and 'END OF EXAMPLES' in content:
                    start = content.find('CORE EXAMPLES')
                    end   = content.find('END OF EXAMPLES')
                    content = content[start:end].strip()
                print(f"[AnalyticalAgent] Loaded few-shot examples from {path}")
                return content
            except Exception as e:
                print(f"[AnalyticalAgent] Could not load examples file ({e}); using built-in.")
        else:
            print("[AnalyticalAgent] Using built-in few-shot examples.")
        return FEW_SHOT_EXAMPLES

    # ── context description ───────────────────────────────────────────────────

    @staticmethod
    def _build_context_hint(context: dict) -> str:
        if not context:
            return "  (empty — this is the first task, no prior state available)"

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

    # ── task-specific guidance ────────────────────────────────────────────────

    def _get_task_guidance(self, task: str) -> str:
        t = task.lower()

        if any(kw in t for kw in (
            'next day', 'next-day', 'next value', 'forecast',
            'future', 't+1', 'tomorrow', 'predict next'
        )):
            return (
                "TASK TYPE: Time-series forecasting.\n"
                "If predicting a future value (e.g., next day), shift the target:\n"
                "    df['target_shifted'] = df['target'].shift(-1)\n"
                "Drop resulting NaN rows. Use chronological split, not random shuffle."
            )

        if any(kw in t for kw in ('train', 'fit', 'model training')):
            return (
                "TASK TYPE: Model training.\n"
                "Step 1 — Check context for scaled data:\n"
                "    X_train = context.get('X_train_scaled') or context.get('X_train')\n"
                "    y_train = context.get('y_train')\n"
                "Step 2 — If not found, fall back to splitting df directly.\n"
                "Step 3 — Fit the model on X_train and y_train.\n"
                "Step 4 — MANDATORY: store model in context BEFORE returning:\n"
                "    context['model'] = model\n"
                "NEVER include the target column in features.\n"
                "NEVER use X_test or y_test during training.\n"
                "See EXAMPLE 15 above for the required pattern."
            )

        if any(kw in t for kw in ('split', 'training set', 'test set')):
            return (
                "TASK TYPE: Dataset splitting.\n"
                "Step 1 — Separate features (X) and target (y).\n"
                "          NEVER include the target column inside X.\n"
                "Step 2 — Split: train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "Step 3 — MANDATORY: write all four arrays to context BEFORE returning:\n"
                "    context['X_train'] = X_train\n"
                "    context['X_test']  = X_test\n"
                "    context['y_train'] = y_train\n"
                "    context['y_test']  = y_test\n"
                "Step 4 — Return only display scalars: {'train_size': ..., 'test_size': ...}\n"
                "See EXAMPLE 13 above for the required pattern."
            )

        if any(kw in t for kw in ('scale', 'scaling', 'normalize', 'normalise')):
            return (
                "TASK TYPE: Feature scaling.\n"
                "Step 1 — Read X_train and X_test from context (set by the split task).\n"
                "Step 2 — If either is None, fall back: split df directly.\n"
                "Step 3 — Fit scaler ONLY on X_train — never on X_test or full df.\n"
                "Step 4 — Transform both sets with the fitted scaler.\n"
                "Step 5 — MANDATORY: store results in context BEFORE returning:\n"
                "    context['X_train_scaled'] = X_train_scaled\n"
                "    context['X_test_scaled']  = X_test_scaled\n"
                "    context['scaler']         = scaler\n"
                "See EXAMPLE 14 above for the required pattern."
            )

        if any(kw in t for kw in ('evaluate', 'rmse', 'r2', 'r²', 'score')):
            return (
                "TASK TYPE: Model evaluation.\n"
                "Step 1 — Read model, X_test (prefer X_test_scaled), y_test from context.\n"
                "Step 2 — If any are None, fall back: re-run pipeline from df.\n"
                "Step 3 — y_pred = model.predict(X_test)\n"
                "Step 4 — Return metrics as plain floats: {'rmse': ..., 'r2': ...}\n"
                "NEVER evaluate on training data.\n"
                "See EXAMPLE 16 above for the required pattern."
            )

        if any(kw in t for kw in ('missing', 'null', 'nan')):
            if any(kw in t for kw in ('count', 'how many', 'identify', 'find', 'check')):
                return (
                    "TASK TYPE: Missing value COUNT.\n"
                    "Use df['col'].isna().sum() — never len(df['col'].isnull()).\n"
                    "Return: {'missing_count': int_value}"
                )
            if any(kw in t for kw in ('fill', 'impute', 'clean', 'handle')):
                return (
                    "TASK TYPE: Missing value IMPUTATION.\n"
                    "Use median for numeric columns, mode for categorical.\n"
                    "After filling, return the requested statistic as a plain float."
                )

        if any(kw in t for kw in ('correlation', 'corr', 'pearson')):
            return (
                "TASK TYPE: Correlation.\n"
                "Use df['col1'].corr(df['col2']) for a single pair.\n"
                "Return: {'correlation': float_value}\n"
                "If filtering is required (e.g. exclude zeros), apply it before .corr()."
            )

        if any(kw in t for kw in (
            'each', 'per', 'group', 'by category', 'by class',
            'by type', 'by role', 'by department', 'by genre',
            'by decade', 'by neighbourhood'
        )):
            return (
                "TASK TYPE: Grouped aggregation.\n"
                "Use df.groupby('group_col')['value_col'].agg_func().\n"
                "ALWAYS return: {str(k): float(v) for k, v in result.items()}\n"
                "Keys must be strings. Values must be plain floats.\n"
                "NEVER nest dicts. NEVER format values as strings."
            )

        if any(kw in t for kw in (
            'percentage', 'percent', 'proportion', 'rate',
            'fraction', 'how many', 'what share'
        )):
            return (
                "TASK TYPE: Conditional percentage or rate.\n"
                "Use boolean_series.mean() * 100 for percentages.\n"
                "Return a plain float — never a formatted string like '13.57%'."
            )

        if any(kw in t for kw in (
            'create', 'new feature', 'derive', 'calculate',
            'engineer', 'ratio', 'per household', 'per capita'
        )):
            return (
                "TASK TYPE: Feature engineering.\n"
                "Create the column, then compute the requested statistic.\n"
                "Return the statistic as a plain float."
            )

        if any(kw in t for kw in ('$', 'dollar', 'clean', 'strip',
                                   'remove symbol', 'remove comma', 'price column')):
            return (
                "TASK TYPE: String cleaning before aggregation.\n"
                r"Use .str.replace(r'[\$,]', '', regex=True) then pd.to_numeric(errors='coerce')."
                "\nAfter cleaning, apply the groupby or aggregation as normal."
            )

        if any(kw in t for kw in ('yes', 'no', 'attrition', 'string', '"yes"', '"no"')):
            return (
                "TASK TYPE: String-encoded boolean aggregation.\n"
                "Use (series == 'Yes').mean() to compute rates — never map or replace first.\n"
                "For grouped rates: groupby(...).apply(lambda x: float((x == 'Yes').mean()))."
            )

        return ""

    # ── column info helpers ───────────────────────────────────────────────────

    def _column_info(self, df: pd.DataFrame, columns=None) -> str:
        cols = columns if columns else df.columns
        return ", ".join(f"'{c}'" for c in cols if c in df.columns)

    def _dtype_info(self, df: pd.DataFrame, columns=None) -> str:
        cols = columns if columns else df.columns
        pairs = [f"'{c}': {df[c].dtype}" for c in cols if c in df.columns]
        return "{" + ", ".join(pairs) + "}"

    # ── declared dependencies prompt block [V4] ───────────────────────────────

    @staticmethod
    def _build_dependencies_block(requires: list, produces: list) -> str:
        """
        Build a concise prompt section that tells the LLM exactly which context
        keys are guaranteed to be present (requires) and which it must write
        (produces). Only rendered when requires or produces is non-empty.
        """
        if not requires and not produces:
            return ""

        lines = ["=" * 70,
                 "CONTEXT CONTRACT FOR THIS TASK",
                 "=" * 70]

        if requires:
            lines.append("The following context keys are GUARANTEED to be present")
            lines.append("because upstream tasks have already written them.")
            lines.append("USE THEM DIRECTLY — do not recompute from df:\n")
            for key_hint in requires:
                key  = key_hint.split(':')[0]
                hint = key_hint.split(':', 1)[1] if ':' in key_hint else ''
                type_note = f"  ({hint})" if hint else ""
                lines.append(f"  context['{key}']{type_note}")

        if requires and produces:
            lines.append("")

        if produces:
            lines.append("This task MUST write the following keys to context")
            lines.append("BEFORE the return statement — the pipeline will fail if missing:\n")
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
                      context: dict = None) -> str:
        """
        Generate a solution(df, context={}) function for the given task.

        V4 changes (retained in V5):
          (a) COLUMN RESTRICTION ACTIVE is suppressed when target_columns
              equals all dataset columns — only fires for genuine subsets.
          (b) DECLARED DEPENDENCIES block injected when requires/produces
              are present in task_metadata.
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

        # Suppress COLUMN RESTRICTION ACTIVE when target_columns is the full
        # dataset — that is not a restriction, it is the default.
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
                f"USE ONLY THESE COLUMNS: {', '.join(target_columns)}\n"
                "Using any other column violates the task specification.\n"
            )
            dataset_note = (
                f"Dataset shape: {df.shape[0]} rows "
                f"(showing {len(target_columns)} of {len(all_df_columns)} columns)"
            )
            print(f"[AnalyticalAgent] Genuine column restriction active: {target_columns}")
        else:
            col_info   = self._column_info(df)
            dtype_info = self._dtype_info(df)
            sample     = df.head(3).to_string()
            constraint_block = ""
            dataset_note = (
                f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns"
            )
            if column_constraint and target_columns:
                print(f"[AnalyticalAgent] Full-dataset ML task — no restriction applied")

        task_guidance      = self._get_task_guidance(task)
        context_hint       = self._build_context_hint(context)
        dependencies_block = self._build_dependencies_block(requires, produces)
        has_context        = bool(context)

        if has_context:
            context_section = f"""
{"=" * 70}
PIPELINE CONTEXT — artifacts produced by previous tasks
{"=" * 70}
{context_hint}

RULES FOR USING CONTEXT:
  • Access values with context.get('key') — never context['key'] (avoids KeyError)
  • USE pre-computed artifacts directly — do not reprocess df from scratch
  • Store new artifacts in context BEFORE the return statement
  • Never store the full df in context
"""
        else:
            context_section = f"""
{"=" * 70}
PIPELINE CONTEXT
{"=" * 70}
{context_hint}
"""

        prompt = f"""You are a data analysis expert. Write a Python function that solves \
an analytical task and returns a machine-readable result.

{constraint_block}
{task_guidance}

{"=" * 70}
FEW-SHOT EXAMPLES — study the output shapes carefully
{"=" * 70}

{self.few_shot_examples}

{"=" * 70}
YOUR TASK
{"=" * 70}

DATASET INFORMATION
Available columns : {col_info}
Data types        : {dtype_info}
{dataset_note}

Sample data (first 3 rows):
{sample}
{context_section}
{dependencies_block}
TASK: {task}

OUTPUT CONTRACT:
  • Signature MUST be: def solution(df, context={{}}):
  • Return a plain float OR a flat dict {{str: float}}
  • For grouped results: {{str(k): float(v) for k, v in result.items()}}
  • NEVER return formatted strings, nested dicts, or raw row data
  • Import libraries inside the function
  • For stateful tasks: write ALL declared produces keys to context BEFORE return

Write ONLY the Python function — no markdown, no explanation, no backticks.
Start with: def solution(df, context={{}}):
"""

        code = call_llm(self.llm, prompt)
        return self._clean_code(code)

    # ── context validation [V4 — dict-driven, with type checking] ────────────

    @staticmethod
    def _validate_context_writes(task: str, context: dict,
                                  task_metadata: dict = None) -> None:
        """
        Validate that all required context keys were written after a stateful
        task completes.

        Two-tier validation:
          Tier 1 (normal path): reads 'produces' from task_metadata and calls
            _validate_produces(), which checks both key presence and value type.
          Tier 2 (legacy fallback): keyword matching on the task string when
            task_metadata is absent or has no 'produces' field.
        """
        produces = None
        if task_metadata and 'produces' in task_metadata:
            produces = task_metadata['produces']

        if produces is not None:
            # Tier 1: dict-driven validation with type checking
            _validate_produces(produces, context, task)
            return

        # Tier 2: legacy keyword-matching fallback
        t = task.lower()

        if any(kw in t for kw in ('split', 'training set', 'test set')):
            required = ['X_train', 'X_test', 'y_train', 'y_test']
            missing  = [k for k in required if k not in context]
            if missing:
                raise RuntimeError(
                    f"Split task completed but did NOT write required context keys: "
                    f"{missing}. Add to solution() before return:\n"
                    + "\n".join(f"    context['{k}'] = {k}" for k in missing)
                )

        elif any(kw in t for kw in ('scale', 'scaling', 'normalize', 'normalise')):
            required = ['X_train_scaled', 'X_test_scaled', 'scaler']
            missing  = [k for k in required if k not in context]
            if missing:
                raise RuntimeError(
                    f"Scaling task completed but did NOT write required context keys: "
                    f"{missing}. Add to solution() before return:\n"
                    + "\n".join(f"    context['{k}'] = {k}" for k in missing)
                )

        elif any(kw in t for kw in ('train', 'fit', 'model training')):
            if 'model' not in context:
                raise RuntimeError(
                    "Training task completed but did NOT write context['model']. "
                    "Add before return:  context['model'] = model"
                )

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_row_level_data(d: dict) -> bool:
        if len(d) < 10:
            return False
        keys = list(d.keys())
        try:
            int_keys = [int(k) for k in keys]
            if int_keys == list(range(len(keys))):
                return True
        except (ValueError, TypeError):
            pass
        float_key_count = sum(1 for k in keys if _is_float_str(k))
        if float_key_count / len(keys) > 0.8:
            return True
        return False

    @staticmethod
    def normalise(raw) -> dict:
        if isinstance(raw, (int, float, np.integer, np.floating)):
            return {'result': float(raw)}

        if isinstance(raw, pd.Series):
            d = {str(k): v for k, v in raw.items() if pd.notna(v)}
            if AnalyticalAgent._is_row_level_data(d):
                return {}
            return {k: float(v) for k, v in d.items()
                    if isinstance(v, (int, float, np.number))}

        if isinstance(raw, dict):
            if AnalyticalAgent._is_row_level_data(raw):
                return {}
            cleaned = {}
            for k, v in raw.items():
                coerced = AnalyticalAgent._coerce_value(v)
                if isinstance(coerced, (int, float)):
                    cleaned[str(k)] = coerced
                elif isinstance(coerced, str):
                    cleaned[str(k)] = coerced
            return cleaned

        return {'result': str(raw)}

    @staticmethod
    def _coerce_value(val):
        if isinstance(val, (np.integer, np.floating)):
            return float(val)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            cleaned = val.strip().rstrip('%').replace(',', '').strip()
            try:
                f = float(cleaned)
                if val.strip().endswith('%') and f > 1:
                    return f / 100.0
                return f
            except ValueError:
                return val
        if isinstance(val, dict) and len(val) == 1:
            inner = next(iter(val.values()))
            return AnalyticalAgent._coerce_value(inner)
        return val

    # ── run() [V4 — passes task_metadata to validator] ───────────────────────

    def run(self, task: str, df: pd.DataFrame,
            task_metadata: dict = None,
            testing_agent=None,
            context: dict = None,
            display: bool = False) -> dict:
        """
        Generate code and execute via testing_agent (with retries).
        task_metadata is forwarded to the context validator so it can use
        the 'produces' list from the planner dict.
        """
        if context is None:
            context = {}

        code = self.generate_code(task, df, task_metadata, context=context)

        def validator(ctx):
            self._validate_context_writes(task, ctx, task_metadata=task_metadata)

        if testing_agent is not None:
            output = testing_agent.run_solution(
                code, df,
                llm=self.llm,
                context=context,
                context_validator=validator,
            )
            if output['ok']:
                result = self.normalise(output['result'])
            else:
                result = {'error': output['error']}
        else:
            result = self._execute_direct(code, df, context=context)
            if 'error' not in result:
                self._validate_context_writes(task, context,
                                              task_metadata=task_metadata)

        if display:
            print(self.formatter.format_for_display(task, result))

        return result

    def _execute_direct(self, code: str, df: pd.DataFrame,
                        context: dict = None) -> dict:
        if context is None:
            context = {}
        namespace = {}
        try:
            exec(compile(code, '<solution>', 'exec'), namespace)
        except SyntaxError as e:
            return {'error': f"SyntaxError: {e}"}
        fn = namespace.get('solution')
        if fn is None:
            return {'error': "No function named 'solution' found."}
        try:
            raw = _call_solution(fn, df.copy(), context)
        except Exception as e:
            return {'error': f"RuntimeError: {e}"}
        return self.normalise(raw)

    # ── code cleaning ─────────────────────────────────────────────────────────

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

    # ── backwards-compatible aliases ──────────────────────────────────────────

    def _get_column_info(self, df: pd.DataFrame) -> str:
        return self._column_info(df)

    def _get_column_info_filtered(self, df: pd.DataFrame, columns: list) -> str:
        return self._column_info(df, columns)

    def _get_dtypes_info(self, df: pd.DataFrame) -> str:
        return self._dtype_info(df)

    def _get_dtypes_info_filtered(self, df: pd.DataFrame, columns: list) -> str:
        return self._dtype_info(df, columns)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
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


def _is_float_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False