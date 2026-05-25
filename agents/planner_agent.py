# planner_agent.py - VERSION 8
# Changes from V7:
#
#  1. _repair_ml_pipeline() removed entirely [UPDATED]:
#     The method and all its keyword-detection logic have been deleted.
#     Every version since V6 has had a different false-positive bug:
#       V6: 'scale' substring matched 'scaled' in train task descriptions
#       V7: 'test set' substring matched evaluate tasks as split tasks
#     The root cause is that keyword matching on free-form LLM-generated
#     task descriptions is inherently fragile — the LLM uses varied
#     phrasing that will always find new ways to fool a keyword list.
#
#  2. _validate_and_repair_tasks() simplified [UPDATED]:
#     Now only handles visualization coverage (check (a)).
#     The ML pipeline check (check (b)) is removed entirely.
#     If the LLM omits a required ML step, the dependency check in
#     main.py._check_dependencies() will catch it at execution time
#     with a clear error — which is honest data for RQ3 (reliability).
#
#  3. ML_STEP_KEYWORDS, _match_step_keywords(), _detect_step_from_task()
#     retained for _detect_ml_step() / _attach_column_metadata() use only.
#     These classify tasks for context contract assignment (requires/produces)
#     and are NOT used for coverage checking anymore.
#
#  4. _SAFE_STEP_DEFAULTS removed — no longer needed without repair logic.
#
#  All other logic unchanged from V7.

from sentence_transformers import SentenceTransformer
from llm_utils import call_llm
import faiss
import numpy as np
import json
import re
import os
from column_extractor import ColumnExtractor


# ── Output type classification keywords ───────────────────────────────────────
VISUALIZATION_KEYWORDS = (
    'plot', 'chart', 'graph', 'visuali', 'histogram', 'scatter',
    'heatmap', 'bar chart', 'line chart', 'boxplot', 'box plot',
    'violin', 'distribution plot', 'draw', 'display chart',
)

# ── ML pipeline task detection ────────────────────────────────────────────────
ML_PIPELINE_KEYWORDS = (
    'split', 'train', 'fit', 'model', 'logistic', 'linear', 'regression',
    'classifier', 'classification', 'random forest', 'xgboost', 'svm',
    'decision tree', 'neural', 'scale', 'scaling', 'normalize', 'normalise',
    'standardscaler', 'minmaxscaler', 'evaluate', 'evaluation', 'rmse',
    'r2', 'roc', 'auc', 'accuracy', 'precision', 'recall', 'f1',
    'predict', 'cross-val', 'cross_val', 'hyperparameter',
)

# ── ML pipeline step keywords ─────────────────────────────────────────────────
# Used ONLY by _detect_ml_step() / _attach_column_metadata() to assign
# requires/produces context contracts to each task.
# NOT used for coverage checking (that logic has been removed).
# Each entry is (keyword, use_word_boundary).
ML_STEP_KEYWORDS = {
    'split':    [('split', True), ('train/test', False), ('training set', False),
                 ('train test', False)],
    'scale':    [('scale', True), ('scaling', True), ('normalize', True),
                 ('normalise', True), ('standardscaler', False),
                 ('minmaxscaler', False), ('standard scaler', False)],
    'train':    [('train a', False), ('train the', False), ('fit a', False),
                 ('fit the', False), ('logistic regression', False),
                 ('random forest', False), ('svm', True), ('decision tree', False),
                 ('xgboost', False), ('neural', True), ('classifier', True),
                 ('regressor', True), ('regression model', False),
                 ('build a model', False)],
    'evaluate': [('evaluat', False), ('rmse', True), ('r2', True), ('r²', False),
                 ('roc', True), ('auc', True), ('accuracy', True),
                 ('precision', True), ('recall', True), ('f1', True),
                 ('performance', True), ('test score', False)],
}

# ── Protected ML steps: never merged with each other ─────────────────────────
PROTECTED_ML_TASKS = (
    'split', 'train', 'scale', 'evaluat', 'predict', 'fit',
    'normalize', 'normalise', 'standardscaler',
)

# ── RAG influence threshold ───────────────────────────────────────────────────
HIGH_INFLUENCE_THRESHOLD = 0.65


def _match_step_keywords(task_lower: str, keywords: list) -> bool:
    """
    Check whether any keyword in the list matches task_lower.
    Keywords marked use_word_boundary=True use \\b word boundaries.
    Keywords marked use_word_boundary=False use plain substring matching.
    """
    for kw, use_boundary in keywords:
        if use_boundary:
            if re.search(r'\b' + re.escape(kw) + r'\b', task_lower):
                return True
        else:
            if kw in task_lower:
                return True
    return False


def _detect_step_from_task(task_str: str) -> str | None:
    """
    Detect which canonical ML pipeline step a task string represents.
    Returns 'split', 'scale', 'train', 'evaluate', or None.
    Used only for assigning context contracts — not for coverage checking.
    """
    t = task_str.lower()
    for step, keywords in ML_STEP_KEYWORDS.items():
        if _match_step_keywords(t, keywords):
            return step
    return None


class PlannerAgent:
    def __init__(self, llm=None, rag_index_path=None, rag_metadata_path=None, use_rag=True):
        if llm is None:
            raise ValueError(
                "PlannerAgent requires an LLM instance. "
                "Pass an LLMClient from main.py: PlannerAgent(llm=llm, ...)"
            )
        self.llm = llm

        self.use_rag = use_rag
        self.embedder = None
        self.index = None
        self.metadata = []

        if use_rag and rag_index_path and rag_metadata_path:
            self._initialize_rag(rag_index_path, rag_metadata_path)

    # ══════════════════════════════════════════════════════════════════════════
    #  RAG INITIALISATION
    # ══════════════════════════════════════════════════════════════════════════

    def _initialize_rag(self, index_path, metadata_path):
        print(f"[Planner] Initializing RAG")
        print(f"  Index: {index_path}")
        print(f"  Metadata: {metadata_path}")

        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                print(f"  ✓ Loaded FAISS index: {self.index.ntotal} vectors")
            else:
                print(f"  ✗ Error: Index file not found: {index_path}")
                self.use_rag = False
                return

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"  ✓ Loaded metadata: {len(self.metadata)} problems")

                categories = {}
                for item in self.metadata:
                    cat = item.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1

                print(f"  Breakdown:")
                for cat, count in categories.items():
                    print(f"    - {cat}: {count}")
            else:
                print(f"  ✗ Error: Metadata file not found: {metadata_path}")
                self.use_rag = False
                return

            print("[Planner] RAG initialized successfully")

        except Exception as e:
            print(f"[Planner] Error initializing RAG: {str(e)}")
            self.use_rag = False

    # ══════════════════════════════════════════════════════════════════════════
    #  RAG RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════

    def _retrieve_similar_cases(self, problem_description, top_k=3,
                                similarity_threshold=0.40):
        if not self.use_rag or self.index is None:
            return []

        try:
            query_embedding = self.embedder.encode([problem_description])
            query_embedding = np.array(query_embedding).astype('float32')

            distances, indices = self.index.search(query_embedding, top_k)

            retrieved_cases = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    cosine_sim = float(1.0 - dist / 2.0)

                    if cosine_sim < similarity_threshold:
                        print(
                            f"[Planner] Skipping case (similarity={cosine_sim:.2f} "
                            f"< threshold={similarity_threshold:.2f}): "
                            f"{self.metadata[idx].get('title', 'N/A')[:60]}"
                        )
                        continue

                    case = self.metadata[idx].copy()
                    case['retrieval_distance'] = float(dist)
                    case['cosine_similarity']  = cosine_sim
                    retrieved_cases.append(case)
                    print(
                        f"[Planner] Accepted case (similarity={cosine_sim:.2f}): "
                        f"{case.get('title', 'N/A')[:60]}"
                    )

            if not retrieved_cases:
                print(
                    f"[Planner] No cases met the similarity threshold "
                    f"({similarity_threshold:.0%}) — proceeding without RAG context."
                )

            return retrieved_cases

        except Exception as e:
            print(f"[Planner] Error during retrieval: {str(e)}")
            return []

    # ══════════════════════════════════════════════════════════════════════════
    #  RAG CONTEXT BUILDER
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_approach_template(self, template):
        if not template or '→' not in template:
            return []

        steps = [s.strip() for s in template.split('→')]
        tasks = []

        step_mapping = {
            'JOIN': 'Merge/join the relevant dataframes',
            'FILTER': 'Filter data based on specified conditions',
            'GROUP': 'Group data by key column(s)',
            'AGG': 'Calculate aggregations',
            'GROUP + AGG': 'Group data and compute aggregations',
            'COMPARE': 'Compare the calculated values',
            'SORT': 'Sort data appropriately',
            'WINDOW/CUMULATIVE': 'Apply window functions or cumulative calculations',
            'WINDOW': 'Apply window functions',
            'CUMULATIVE': 'Calculate cumulative values',
            'AGGREGATE NUMERATOR': 'Calculate the numerator value',
            'AGGREGATE DENOMINATOR': 'Calculate the denominator value',
            'DIVIDE': 'Compute the final ratio/rate',
            'TRANSFORM': 'Transform/reshape the data',
            'PIVOT': 'Pivot the data structure',
            'UNPIVOT': 'Unpivot/melt the data',
            'DEDUPLICATE': 'Remove duplicate records',
            'VALIDATE': 'Validate data quality',
            'CLEAN': 'Clean and handle missing values',
        }

        for step in steps:
            step_upper = step.upper().strip()
            tasks.append(step_mapping.get(step_upper, step.capitalize()))

        return tasks

    def _get_strategy_hint(self, problem_type):
        strategies = {
            'aggregation_comparison': (
                "Strategy: (1) Group data by key (2) Calculate aggregates (3) Compare results across groups"
            ),
            'behavioral_analysis': (
                "Strategy: (1) Sort by time/sequence (2) Apply window functions (3) Filter by behavior pattern"
            ),
            'metric_calculation': (
                "Strategy: (1) Calculate numerator (2) Calculate denominator (3) Compute ratio/percentage"
            ),
            'data_cleaning': (
                "Strategy: (1) Identify issues (2) Apply transformations (3) Validate results"
            ),
            'time_series': (
                "Strategy: (1) Sort by time (2) Calculate rolling/cumulative metrics (3) Identify patterns"
            ),
            'ranking': (
                "Strategy: (1) Calculate ranking metric (2) Sort/rank (3) Select top/bottom N"
            ),
            'join_analysis': (
                "Strategy: (1) Merge datasets (2) Filter combined data (3) Aggregate results"
            ),
        }
        return strategies.get(problem_type, "")

    def _build_rag_context(self, retrieved_cases):
        if not retrieved_cases:
            return ""

        context = "\n" + "=" * 70 + "\n"
        context += "KNOWLEDGE BASE: SIMILAR PROBLEMS & SOLUTION PATTERNS\n"
        context += "=" * 70 + "\n\n"

        problem_types = [c.get('problem_type', 'unknown') for c in retrieved_cases]
        problem_types_clean = [pt for pt in problem_types if pt and pt != 'unknown']

        if problem_types_clean:
            most_common_type = max(set(problem_types_clean), key=problem_types_clean.count)
            context += f"**DETECTED PROBLEM TYPE: {most_common_type.upper()}**\n"
            strategy = self._get_strategy_hint(most_common_type)
            if strategy:
                context += f"{strategy}\n\n"

        context += "**HOW TO USE THESE EXAMPLES:**\n"
        context += "1. Study the TASK BREAKDOWN to understand the solution structure\n"
        context += "2. Count the number of tasks - use SIMILAR count for your problem\n"
        context += "3. Apply the KEY INSIGHT to your specific problem\n"
        context += "4. Adapt the pattern, don't copy verbatim\n\n"

        high_conf = [c for c in retrieved_cases
                     if c.get('cosine_similarity', 0) >= HIGH_INFLUENCE_THRESHOLD]
        low_conf  = [c for c in retrieved_cases
                     if c.get('cosine_similarity', 0) <  HIGH_INFLUENCE_THRESHOLD]

        for i, case in enumerate(retrieved_cases, 1):
            sim = case.get('cosine_similarity', 0)
            conf_label = "(high confidence)" if sim >= HIGH_INFLUENCE_THRESHOLD else "(reference only)"
            context += f"--- EXAMPLE {i} {conf_label} ---\n"
            context += f"Problem: {case.get('title', 'N/A')}\n"
            context += f"Similarity: {sim:.2f} | Difficulty: {case.get('difficulty', 'N/A')}\n"

            problem_type = case.get('problem_type', 'unknown')
            if problem_type != 'unknown':
                context += f"Type: {problem_type}\n"
            context += "\n"

            approach = case.get('approach_template', '')
            if approach:
                context += f"**APPROACH PATTERN:** {approach}\n\n"
                parsed_tasks = self._parse_approach_template(approach)
                if parsed_tasks:
                    context += f"**RECOMMENDED TASK BREAKDOWN ({len(parsed_tasks)} tasks):**\n"
                    for idx, task in enumerate(parsed_tasks, 1):
                        context += f"  {idx}. {task}\n"
                    context += "\n"

            insight = case.get('key_insight', '')
            if insight:
                context += f"**KEY INSIGHT:** {insight}\n"
                context += f"→ Apply this principle to your current problem\n\n"

            operations = case.get('operations', [])
            if operations and len(operations) <= 6:
                context += f"Core Operations: {', '.join(operations[:6])}\n"

            skills = case.get('core_skills', [])
            if skills and len(skills) <= 5:
                context += f"Skills Required: {', '.join(skills[:5])}\n"

            context += "\n"

        task_counts = []
        for case in high_conf:
            approach = case.get('approach_template', '')
            if approach:
                parsed = self._parse_approach_template(approach)
                if parsed:
                    task_counts.append(len(parsed))

        context += "=" * 70 + "\n"
        if task_counts:
            avg_tasks = int(np.mean(task_counts))
            context += f"**TASK COUNT GUIDANCE (from {len(high_conf)} high-confidence example(s)):**\n"
            context += f"Examples used {min(task_counts)}-{max(task_counts)} tasks (average: {avg_tasks})\n"
            context += f"Your problem should use a SIMILAR number of tasks.\n"
        elif low_conf and not high_conf:
            context += "**TASK COUNT GUIDANCE:**\n"
            context += "Retrieved examples are reference patterns only (low similarity).\n"
            context += "Use your own judgment for task count — do not copy example counts blindly.\n"
        else:
            context += "**CRITICAL: Match the complexity shown in examples above.**\n"
            context += "If examples are simple, keep your breakdown simple too.\n"
        context += "=" * 70 + "\n\n"

        return context

    # ══════════════════════════════════════════════════════════════════════════
    #  TASK COUNT DETERMINATION
    # ══════════════════════════════════════════════════════════════════════════

    def _determine_max_tasks(self, problem_description, retrieved_cases):
        problem_lower = problem_description.lower()

        ml_keywords = [
            'train', 'predict', 'model', 'pipeline', 'cross-validation',
            'hyperparameter', 'classifier', 'regression', 'random forest',
            'logistic', 'svm', 'xgboost', 'neural',
        ]
        ml_keyword_count = sum(kw in problem_lower for kw in ml_keywords)
        is_ml_pipeline = ml_keyword_count >= 2

        high_conf_cases = [
            c for c in retrieved_cases
            if c.get('cosine_similarity', 0) >= HIGH_INFLUENCE_THRESHOLD
        ]
        example_task_counts = []
        for case in high_conf_cases:
            approach = case.get('approach_template', '')
            if approach:
                parsed = self._parse_approach_template(approach)
                if parsed:
                    example_task_counts.append(len(parsed))

        if example_task_counts:
            avg_example_tasks = int(np.mean(example_task_counts))
            base_max = min(avg_example_tasks + 1, 5)
            print(f"[Planner] High-confidence RAG examples suggest ~{avg_example_tasks} tasks, "
                  f"setting base max to {base_max}")
        else:
            base_max = 4
            print(f"[Planner] No high-confidence RAG examples, using default base max: {base_max}")

        if is_ml_pipeline:
            if base_max < 4:
                print(f"[Planner] ML pipeline detected — raising base max from {base_max} to 4 "
                      f"(minimum for split→scale→train→evaluate)")
            base_max = max(base_max, 4)

        complexity_boost = 0

        if is_ml_pipeline:
            complexity_boost += 1
            print(f"[Planner] Detected ML pipeline, adding +1 task")

        viz_count = (problem_lower.count('plot') + problem_lower.count('visualiz') +
                     problem_lower.count('chart') + problem_lower.count('graph'))
        if viz_count > 2:
            complexity_boost += 1
            print(f"[Planner] Multiple visualizations detected ({viz_count}), adding +1 task")

        sequential_keywords = ['and then', 'after that', 'next', 'finally', 'subsequently']
        if any(kw in problem_lower for kw in sequential_keywords):
            complexity_boost += 1
            print(f"[Planner] Sequential steps detected, adding +1 task")

        max_tasks = min(base_max + complexity_boost, 7)

        if max_tasks <= 3:
            target = "2-3"
        elif max_tasks <= 5:
            target = "3-5"
        else:
            target = "5-7"

        print(f"[Planner] Final max tasks: {max_tasks} (target range: {target})")
        return max_tasks, target

    # ══════════════════════════════════════════════════════════════════════════
    #  OUTPUT TYPE CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _classify_task_output_type(task_str: str) -> str:
        t = task_str.lower()
        if any(kw in t for kw in VISUALIZATION_KEYWORDS):
            return 'visualization'
        return 'analytical'

    # ══════════════════════════════════════════════════════════════════════════
    #  DEDUPLICATION
    # ══════════════════════════════════════════════════════════════════════════

    def _merge_most_similar_pair(self, tasks, sim_matrix):
        n = len(tasks)
        best_i, best_j, best_score = 0, 1, -1.0

        for i in range(n):
            for j in range(i + 1, n):
                i_protected = any(kw in tasks[i].lower() for kw in PROTECTED_ML_TASKS)
                j_protected = any(kw in tasks[j].lower() for kw in PROTECTED_ML_TASKS)
                if i_protected and j_protected:
                    continue

                if _classify_task_output_type(tasks[i]) != _classify_task_output_type(tasks[j]):
                    continue

                if sim_matrix[i][j] > best_score:
                    best_score = sim_matrix[i][j]
                    best_i, best_j = i, j

        if best_score == -1.0:
            print(f"[Planner] No mergeable pair found — truncating last non-protected task")
            for idx in range(len(tasks) - 1, -1, -1):
                t = tasks[idx]
                is_protected = any(kw in t.lower() for kw in PROTECTED_ML_TASKS)
                is_viz = _classify_task_output_type(t) == 'visualization'
                if not is_protected and not is_viz:
                    print(f"[Planner] Truncating: '{t[:60]}'")
                    return [t2 for k, t2 in enumerate(tasks) if k != idx]
            return tasks[:-1]

        merged = f"{tasks[best_i]} and {tasks[best_j]}"
        print(f"[Planner] Merging tasks (similarity={best_score:.2f}):")
        print(f"  '{tasks[best_i][:60]}'")
        print(f"  '{tasks[best_j][:60]}'")
        print(f"  → '{merged[:80]}'")

        return [t for idx, t in enumerate(tasks) if idx not in (best_i, best_j)] + [merged]

    def _deduplicate_tasks(self, tasks):
        if len(tasks) <= 2:
            return tasks

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            embeddings = self.embedder.encode(tasks)
            sim_matrix = cosine_similarity(embeddings)

            duplicates = set()
            for i in range(len(tasks)):
                if i in duplicates:
                    continue
                for j in range(i + 1, len(tasks)):
                    if j in duplicates:
                        continue

                    i_protected = any(kw in tasks[i].lower() for kw in PROTECTED_ML_TASKS)
                    j_protected = any(kw in tasks[j].lower() for kw in PROTECTED_ML_TASKS)
                    if i_protected and j_protected:
                        continue

                    if _classify_task_output_type(tasks[i]) != _classify_task_output_type(tasks[j]):
                        continue

                    if sim_matrix[i][j] > 0.62:
                        print(f"[Planner] Deduplication: merging similar tasks:")
                        print(f"  Task {i+1}: {tasks[i][:60]}...")
                        print(f"  Task {j+1}: {tasks[j][:60]}...")
                        print(f"  Similarity: {sim_matrix[i][j]:.2f}")
                        duplicates.add(j)

            deduplicated = [task for idx, task in enumerate(tasks) if idx not in duplicates]

            if len(deduplicated) < len(tasks):
                print(f"[Planner] Removed {len(tasks) - len(deduplicated)} duplicate task(s)")

            return deduplicated

        except Exception as e:
            print(f"[Planner] Deduplication failed: {e}, returning original tasks")
            return tasks

    # ══════════════════════════════════════════════════════════════════════════
    #  COVERAGE VALIDATION
    #  V8: only checks visualization coverage.
    #  ML pipeline coverage check removed — see module header for rationale.
    # ══════════════════════════════════════════════════════════════════════════

    def _validate_and_repair_tasks(self, tasks: list, problem_description: str,
                                   dataset) -> list:
        problem_lower = problem_description.lower()

        viz_requested = any(kw in problem_lower for kw in (
            'plot', 'chart', 'graph', 'visuali', 'histogram', 'scatter',
            'heatmap', 'bar chart', 'line chart', 'draw',
        ))

        if viz_requested:
            has_viz_task = any(
                _classify_task_output_type(t) == 'visualization' for t in tasks
            )
            if not has_viz_task:
                viz_task = self._infer_visualization_task(problem_description, dataset)
                tasks.append(viz_task)
                print(f"[Planner] Coverage repair: added missing visualization task: "
                      f"'{viz_task[:70]}'")

        return tasks

    def _infer_visualization_task(self, problem_description: str, dataset) -> str:
        p = problem_description.lower()

        if 'correlation' in p or 'relationship' in p:
            return "Plot the correlation results as a scatter plot with axis labels and title"
        if 'distribution' in p or 'histogram' in p:
            return "Plot the distribution of the target variable as a histogram"
        if 'trend' in p or 'over time' in p or 'time series' in p:
            return "Plot the trend over time as a line chart with date on the x-axis"
        if 'comparison' in p or 'compare' in p or 'by group' in p:
            return "Plot the comparison results as a bar chart with group labels"
        if 'bar' in p:
            return "Plot the results as a bar chart with appropriate axis labels"

        return "Visualize the analysis results as an appropriate chart with axis labels and title"

    # ══════════════════════════════════════════════════════════════════════════
    #  COLUMN METADATA ATTACHMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_columns_from_task(self, task_str, mentioned_columns):
        if not mentioned_columns:
            return []
        task_lower = task_str.lower()
        return [col for col in mentioned_columns if col.lower() in task_lower]

    @staticmethod
    def _detect_ml_step(task_str: str) -> str | None:
        """
        Classify a task's ML pipeline step for context contract assignment.
        Returns 'split', 'scale', 'train', 'evaluate', or None.
        """
        return _detect_step_from_task(task_str)

    @staticmethod
    def _get_context_contract(ml_step: str | None) -> tuple[list, list]:
        contracts = {
            'split': (
                [],
                ['X_train:DataFrame', 'X_test:DataFrame',
                 'y_train:Series',    'y_test:Series'],
            ),
            'scale': (
                ['X_train:DataFrame', 'X_test:DataFrame'],
                ['X_train_scaled:ndarray', 'X_test_scaled:ndarray',
                 'scaler:fitted'],
            ),
            'train': (
                ['X_train_scaled:ndarray', 'y_train:Series'],
                ['model:fitted'],
            ),
            'evaluate': (
                ['model:fitted', 'X_test_scaled:ndarray', 'y_test:Series'],
                [],
            ),
        }
        if ml_step is None:
            return [], []
        return contracts.get(ml_step, ([], []))

    def _attach_column_metadata(self, subtasks, mentioned_columns, all_columns):
        structured_tasks = []

        for task_str in subtasks:
            task_lower    = task_str.lower()
            is_ml_task    = any(kw in task_lower for kw in ML_PIPELINE_KEYWORDS)
            output_type   = _classify_task_output_type(task_str)
            ml_step       = self._detect_ml_step(task_str) if is_ml_task else None
            requires, produces = self._get_context_contract(ml_step)

            if is_ml_task:
                target_columns    = all_columns
                column_constraint = True
                print(
                    f"[Planner] Task '{task_str[:50]}...' "
                    f"→ ML/{ml_step or 'general'} [{output_type}]: "
                    f"ALL {len(all_columns)} cols | "
                    f"requires={[k.split(':')[0] for k in requires]} | "
                    f"produces={[k.split(':')[0] for k in produces]}"
                )
            elif mentioned_columns:
                task_specific_columns = self._extract_columns_from_task(
                    task_str, mentioned_columns
                )
                target_columns    = task_specific_columns if task_specific_columns else mentioned_columns
                column_constraint = True
                print(
                    f"[Planner] Task '{task_str[:50]}...' "
                    f"→ [{output_type}] constrained to: "
                    f"{task_specific_columns if task_specific_columns else mentioned_columns}"
                )
            else:
                target_columns    = []
                column_constraint = False
                print(
                    f"[Planner] Task '{task_str[:50]}...' "
                    f"→ [{output_type}] no column constraint"
                )

            structured_tasks.append({
                'description':           task_str,
                'output_type':           output_type,
                'ml_step':               ml_step,
                'requires':              requires,
                'produces':              produces,
                'target_columns':        target_columns,
                'column_constraint':     column_constraint,
                'all_mentioned_columns': mentioned_columns,
                'is_ml_task':            is_ml_task,
            })

        return structured_tasks

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def create_subtasks(self, problem_description, dataset):
        print(f"\n[Planner] Creating subtasks for problem...")

        # STEP 1: Extract mentioned columns
        extractor = ColumnExtractor(dataset.columns.tolist())
        mentioned_columns = extractor.extract_with_llm_verification(problem_description, self.llm)
        all_columns = dataset.columns.tolist()

        if mentioned_columns:
            print(f"[Planner] Detected specific columns: {mentioned_columns}")
        else:
            print(f"[Planner] No specific columns detected — agents will select relevant columns")

        # STEP 2: Retrieve similar cases
        retrieved_cases = []
        rag_context = ""

        if self.use_rag:
            print(f"[Planner] Retrieving similar problems from knowledge base...")
            retrieved_cases = self._retrieve_similar_cases(problem_description, top_k=3)
            rag_context = self._build_rag_context(retrieved_cases)
            print(f"[Planner] Retrieved {len(retrieved_cases)} similar cases")

        # STEP 3: Determine max tasks
        max_tasks, target = self._determine_max_tasks(problem_description, retrieved_cases)

        # STEP 4: Build column-specific context for prompt
        column_context = ""
        if mentioned_columns:
            column_context  = f"\n**SPECIFIC COLUMNS MENTIONED:**\n"
            column_context += f"The user referenced: {', '.join(mentioned_columns)}\n"
            column_context += f"Prioritize these columns in your subtasks where relevant.\n\n"

        # STEP 5: Generate subtasks via LLM
        prompt = f"""You are an expert planning agent that breaks down data science problems into clear, actionable subtasks.

{rag_context}

**CURRENT PROBLEM TO SOLVE:**
{problem_description}

**DATASET INFORMATION:**
- Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns
- Available columns: {', '.join(all_columns[:10])}{'...' if len(all_columns) > 10 else ''}
{column_context}

**CRITICAL RULES**

1. **MATCH THE EXAMPLES**: Use the knowledge base patterns to guide task count and structure.

2. **STAY MINIMAL**: Generate EXACTLY {target} subtasks. No more, no less.
   Never add optional, bonus, or exploratory tasks. Every task must be explicitly required by the problem statement.

3. **BE SPECIFIC**: Reference columns explicitly. BAD: "Analyze correlations" → GOOD: "Calculate correlation between Age and Fare"

4. **NO REDUNDANCY**: Merge related operations into one task. BAD: "Calculate mean Age" + "Calculate median Age" → GOOD: "Calculate Age statistics (mean, median, std)"

5. **LOGICAL ORDER**: Tasks must build sequentially. Data cleaning → Analysis → Visualization → Modeling.

6. **NO PREP TASKS**: Every task must produce a terminal output: a trained model, a plot, a summary table, or a metric. Intermediate steps belong inside the task that uses them.

7. **ML PIPELINES**: If the problem involves modeling, always include ALL FOUR of these as SEPARATE tasks in this exact order:
   (a) Split data into train/test sets
   (b) Scale features
   (c) Train model
   (d) Evaluate model — use EXACTLY the metric(s) requested in the problem statement
   Never merge any two of these four steps into a single task.
   Never omit any of these four steps.

8. **ML TASKS — NAME ALL COLUMNS EXPLICITLY**: For any training, scaling, splitting, or evaluation task,
   you MUST name every feature column AND the target column in the task description.
   BAD:  "Train a logistic regression model with Outcome as the target"
   GOOD: "Train a logistic regression model using Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DiabetesPedigreeFunction, Age as features and Outcome as the target"

9. **METRICS — USE EXACTLY WHAT IS ASKED**: If the problem asks for RMSE, the evaluate task must
   say RMSE. If it asks for ROC-AUC, say ROC-AUC. Never substitute or add a different metric.

**OUTPUT FORMAT — READ CAREFULLY:**
- Output ONLY a numbered list. Nothing else.
- Do NOT write preamble, reasoning, explanations, or commentary of any kind.
- Start your response with "1." and end it with the last numbered task. Nothing before, nothing after.
- Format: "1. [task]", "2. [task]", etc.
- Each task: 25–300 characters.

10. **CATEGORICAL ENCODING**: If the dataset has categorical columns, the scale task MUST include
    one-hot encoding for categorical columns AND StandardScaler for numerical columns.
    Use sklearn's ColumnTransformer to apply both in one step.
    Never drop categorical columns or apply StandardScaler to string columns.

Begin your numbered list now (start with "1."):"""

        raw_response = call_llm(self.llm, prompt)
        raw_lines = [line.strip() for line in raw_response.split("\n") if line.strip()]

        # STEP 6: Clean and extract numbered tasks
        cleaned = []
        for line in raw_lines:
            if not re.match(r'^\d+', line):
                continue
            clean = re.sub(r'^(?:STEP\s*|Task\s*|Step\s*)?\d+\s*[\.\)\-\:]\s*', '',
                           line, flags=re.IGNORECASE).strip()
            if clean and 25 <= len(clean) <= 300:
                cleaned.append(clean)
            elif clean and len(clean) < 25:
                print(f"[Planner] Skipped short line: '{clean}'")
            elif clean and len(clean) > 300:
                print(f"[Planner] Skipped overly long line ({len(clean)} chars): '{clean[:60]}..'")

        print(f"[Planner] Extracted {len(cleaned)} initial tasks from LLM")

        # STEP 7: Deduplicate
        if len(cleaned) > 2:
            cleaned = self._deduplicate_tasks(cleaned)

        # STEP 8: Enforce max_tasks via similarity-based merging
        if len(cleaned) > max_tasks:
            print(f"[Planner] {len(cleaned)} tasks exceed max ({max_tasks}), "
                  f"merging most similar pairs...")
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                while len(cleaned) > max_tasks:
                    embeddings = self.embedder.encode(cleaned)
                    sim_matrix = cosine_similarity(embeddings)
                    np.fill_diagonal(sim_matrix, 0.0)
                    cleaned = self._merge_most_similar_pair(cleaned, sim_matrix)
            except Exception as e:
                print(f"[Planner] Similarity merging failed ({e}), falling back to truncation")
                cleaned = cleaned[:max_tasks]

        if not cleaned:
            print(f"[Planner] WARNING: No tasks extracted — falling back to generic task")
            print(f"Raw LLM response: {raw_response[:300]}")
            cleaned = ["Analyze the dataset to address the problem statement"]

        # STEP 9: Coverage validation (visualization only)
        cleaned = self._validate_and_repair_tasks(cleaned, problem_description, dataset)

        # STEP 10: Attach column metadata
        structured_tasks = self._attach_column_metadata(cleaned, mentioned_columns, all_columns)

        print(f"[Planner] ✓ Final subtask count: {len(structured_tasks)}")
        for i, task in enumerate(structured_tasks, 1):
            cols_info = f" [Columns: {len(task['target_columns'])} cols]" if task['target_columns'] else ""
            ml_tag    = " [ML]"  if task.get('is_ml_task') else ""
            out_type  = _classify_task_output_type(task['description'])
            viz_tag   = " [VIZ]" if out_type == 'visualization' else ""
            print(f"  {i}. {task['description'][:70]}{cols_info}{ml_tag}{viz_tag}")

        return {
            'subtasks':               structured_tasks,
            'mentioned_columns':      mentioned_columns,
            'column_constraint':      bool(mentioned_columns),
            'rag_used':               self.use_rag and len(retrieved_cases) > 0,
            'rag_retrievals':         len(retrieved_cases),
            'retrieved_cases':        retrieved_cases if self.use_rag else [],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _classify_task_output_type(task_str: str) -> str:
    t = task_str.lower()
    if any(kw in t for kw in VISUALIZATION_KEYWORDS):
        return 'visualization'
    return 'analytical'