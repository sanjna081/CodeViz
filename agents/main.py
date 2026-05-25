# main.py - VERSION 7
# Changes from V6:
#
#  1. run_analysis_programmatic() accepts test_case_name and results_dir [UPDATED]:
#     Previously used global constants TEST_CASE_NAME and RESULTS_DIR.
#     Now accepts these as optional parameters so runner.py can pass the
#     correct filename for each test case without modifying the config block.
#     Defaults to the global constants so interactive main() is unchanged.

import os
import sys
import time
import json
import datetime

import numpy as np
import requests
import pandas as pd

from planner_agent import PlannerAgent
from analyzer_agent import AnalyzerAgent
from analytical_agent import AnalyticalAgent
from visualization_agent import VisualizationAgent
from testing_agent import TestingAgent
from dotenv import load_dotenv
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODEL_NAME   = "qwen/qwen3-coder"
TEMPERATURE  = 0.2
MAX_TOKENS   = 8192

RAG_INDEX_PATH    = r'C:\Users\Shreekumar\codeviz\rag\faiss_index_unified.bin'
RAG_METADATA_PATH = r'C:\Users\Shreekumar\codeviz\rag\faiss_metadata_unified.json'

RESULTS_DIR    = r'C:\Users\Shreekumar\codeviz\results'
TEST_CASE_NAME = "test_case_1"


# ══════════════════════════════════════════════════════════════════════════════
#  LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class _InvokeResponse:
    def __init__(self, content: str):
        self.content = content


class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key:     str   = OPENROUTER_API_KEY,
        model:       str   = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_tokens:  int   = MAX_TOKENS,
    ):
        if not api_key or api_key == "YOUR_OPENROUTER_API_KEY_HERE":
            raise ValueError(
                "OpenRouter API key not set. "
                "Either set the OPENROUTER_API_KEY environment variable or "
                "update OPENROUTER_API_KEY in main.py."
            )
        self.api_key     = api_key
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/codeviz",
            "X-Title":       "codeviz",
        }

        self._calls             = 0
        self._prompt_tokens     = 0
        self._completion_tokens = 0
        self._total_tokens      = 0
        self._total_cost_usd    = 0.0
        self._total_latency     = 0.0

        print(f"[LLMClient] Initialised — model: {self.model}")

    def complete(self, prompt: str) -> str:
        payload = {
            "model":       self.model,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "messages":    [{"role": "user", "content": prompt}],
        }
        t0 = time.time()
        try:
            resp = requests.post(
                self.OPENROUTER_BASE_URL,
                headers=self._headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data    = resp.json()
            latency = time.time() - t0

            usage             = data.get("usage", {})
            prompt_tokens     = usage.get("prompt_tokens",     0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens      = usage.get("total_tokens",      prompt_tokens + completion_tokens)
            call_cost         = float(usage.get("cost", 0.0))

            self._calls             += 1
            self._prompt_tokens     += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens      += total_tokens
            self._total_cost_usd    += call_cost
            self._total_latency     += latency

            print(
                f"[LLMClient] call #{self._calls} — "
                f"{total_tokens} tokens "
                f"({prompt_tokens} in / {completion_tokens} out) | "
                f"latency: {latency:.1f}s | "
                f"cost: ${call_cost:.5f}"
            )

            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"[LLMClient] Request timed out after 120 s (model: {self.model})"
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            body   = e.response.text[:400]  if e.response is not None else ""

            if status == 429:
                try:
                    retry_after = int(
                        e.response.json()['error']['metadata']['retry_after_seconds']
                    ) + 2
                except Exception:
                    retry_after = 30
                print(f"[LLMClient] Rate limited — waiting {retry_after}s then retrying...")
                time.sleep(retry_after)
                return self.complete(prompt)

            raise RuntimeError(
                f"[LLMClient] HTTP {status} from OpenRouter: {body}"
            ) from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"[LLMClient] Unexpected response shape: {e}"
            ) from e

    def invoke(self, prompt: str) -> _InvokeResponse:
        return _InvokeResponse(self.complete(prompt))

    def get_usage_dict(self) -> dict:
        return {
            'model':             self.model,
            'llm_calls':         self._calls,
            'prompt_tokens':     self._prompt_tokens,
            'completion_tokens': self._completion_tokens,
            'total_tokens':      self._total_tokens,
            'total_cost_usd':    self._total_cost_usd,
            'llm_latency_s':     round(self._total_latency, 2),
            'avg_latency_s':     round(self._total_latency / self._calls, 2) if self._calls else 0,
        }

    def print_usage_summary(self, pipeline_wall_time: float = None):
        print("\n" + "=" * 60)
        print("USAGE SUMMARY")
        print("=" * 60)
        print(f"  Model          : {self.model}")
        print(f"  LLM calls      : {self._calls}")
        print(f"  Prompt tokens  : {self._prompt_tokens:,}")
        print(f"  Completion tok : {self._completion_tokens:,}")
        print(f"  Total tokens   : {self._total_tokens:,}")
        print(f"  Total cost     : ${self._total_cost_usd:.5f}")
        if self._calls:
            print(f"  LLM latency    : {self._total_latency:.1f}s "
                  f"(avg {self._total_latency/self._calls:.1f}s/call)")
        else:
            print(f"  LLM latency    : 0s")
        if pipeline_wall_time is not None:
            print(f"  Pipeline time  : {pipeline_wall_time:.1f}s total "
                  f"(LLM={self._total_latency:.1f}s, "
                  f"other={pipeline_wall_time - self._total_latency:.1f}s)")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  JSON RESULTS SAVER
# ══════════════════════════════════════════════════════════════════════════════

def _make_serialisable(obj):
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame shape={obj.shape} columns={list(obj.columns)}>"
    if isinstance(obj, pd.Series):
        return f"<Series len={len(obj)} dtype={obj.dtype} name={obj.name}>"
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    return f"<{type(obj).__name__}>"


def _save_results(output: dict, llm: LLMClient,
                  pipeline_wall_time: float,
                  test_case_name: str = TEST_CASE_NAME,
                  results_dir: str    = RESULTS_DIR) -> str:

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    filepath = os.path.join(results_dir, f"results_{test_case_name}.json")

    task_results = output.get('task_results', [])
    n_tasks      = len(task_results)
    n_success    = sum(1 for t in task_results if t.get('ok'))
    n_failed     = n_tasks - n_success
    n_retried    = sum(1 for t in task_results if t.get('attempts', 1) > 1)

    context = output.get('context', {})
    context_summary = {}
    for k, v in context.items():
        if hasattr(v, 'shape'):
            context_summary[k] = f"<{type(v).__name__} shape={v.shape}>"
        elif isinstance(v, pd.DataFrame):
            context_summary[k] = f"<DataFrame shape={v.shape}>"
        elif isinstance(v, pd.Series):
            context_summary[k] = f"<Series len={len(v)}>"
        elif hasattr(v, '__len__') and not isinstance(v, str):
            context_summary[k] = f"<{type(v).__name__} len={len(v)}>"
        else:
            context_summary[k] = f"<{type(v).__name__}>"

    subtasks_clean = []
    for t in output.get('subtasks', []):
        if isinstance(t, dict):
            subtasks_clean.append({
                'description': t.get('description', ''),
                'output_type': t.get('output_type', ''),
                'ml_step':     t.get('ml_step'),
                'requires':    t.get('requires', []),
                'produces':    t.get('produces', []),
            })
        else:
            subtasks_clean.append({'description': str(t)})

    task_records = []
    for t in task_results:
        task_records.append({
            'task':           t.get('task', ''),
            'agent':          t.get('agent', ''),
            'ok':             t.get('ok', False),
            'attempts':       t.get('attempts', 1),
            'execution_time': t.get('execution_time'),
            'result':         _make_serialisable(t.get('result')),
            'error':          t.get('error'),
            'code':           t.get('code'),
            'wrapped_code':   t.get('wrapped_code'),
            'stdout':         t.get('stdout'),
            'metadata': {
                'output_type': t.get('metadata', {}).get('output_type'),
                'ml_step':     t.get('metadata', {}).get('ml_step'),
                'requires':    t.get('metadata', {}).get('requires', []),
                'produces':    t.get('metadata', {}).get('produces', []),
            },
        })

    usage = llm.get_usage_dict()
    usage['pipeline_wall_time_s'] = round(pipeline_wall_time, 2)
    usage['non_llm_time_s']       = round(pipeline_wall_time - llm._total_latency, 2)

    document = {
        'test_case':         test_case_name,
        'timestamp':         datetime.datetime.now().isoformat(),
        'model':             llm.model,
        'temperature':       llm.temperature,
        'max_tokens':        llm.max_tokens,
        'dataset_path':      output.get('dataset_path', ''),
        'dataset_shape':     output.get('dataset_shape', []),
        'dataset_columns':   output.get('dataset_columns', []),
        'problem_statement': output.get('problem', ''),
        'usage':             usage,
        'pipeline_stats': {
            'total_tasks':      n_tasks,
            'successful_tasks': n_success,
            'failed_tasks':     n_failed,
            'retried_tasks':    n_retried,
            'rag_used':         output.get('planner_rag_used', False),
            'rag_retrievals':   output.get('planner_rag_retrievals', 0),
        },
        'subtasks':     subtasks_clean,
        'task_results': task_records,
        'errors':       output.get('errors', []),
        'context_keys': context_summary,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    print(f"\n[Results] Saved to: {filepath}")
    return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(2048)
            sep = ';' if sample.count(';') > sample.count(',') else ','
            df = pd.read_csv(file_path, sep=sep)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                "Supported formats: .csv, .xlsx, .xls, .json, .parquet"
            )

        original_cols = list(df.columns)
        df.columns = df.columns.str.strip()
        stripped = [o for o, n in zip(original_cols, df.columns) if o != n]
        if stripped:
            print(f"  ⚠ Stripped whitespace from {len(stripped)} column name(s): {stripped}")

        print(f"\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        return df

    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  USER INPUT
# ══════════════════════════════════════════════════════════════════════════════

def get_user_input():
    print("=" * 60)
    print("Welcome to the Multi-Agent Data Analysis System")
    print("=" * 60)

    print("\n[Dataset Input]")
    print("Enter the path to your dataset file.")
    print("Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet")

    while True:
        dataset_path = input("\nDataset file path: ").strip()
        if not dataset_path:
            print("Error: Please provide a file path.")
            continue
        try:
            df = load_dataset(dataset_path)
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            retry = input("Would you like to try again? (yes/no): ").strip().lower()
            if retry not in ['yes', 'y']:
                print("Exiting...")
                sys.exit(0)

    print("\n" + "=" * 60)
    print("[Problem Statement]")
    print("Describe what you want to analyze or visualize.")
    print("Examples:")
    print("  - Analyze the relationship between features X and Y")
    print("  - Find patterns in the data and visualize them")
    print("  - Predict target variable using available features")
    print("=" * 60)

    while True:
        problem = input("\nYour problem statement: ").strip()
        if not problem:
            print("Error: Please provide a problem statement.")
            continue

        print("\n" + "-" * 60)
        print("Dataset:", dataset_path)
        print("Problem:", problem)
        print("-" * 60)

        confirm = input("\nProceed with this configuration? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            break

    return df, problem


# ══════════════════════════════════════════════════════════════════════════════
#  TASK METADATA BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_task_metadata(assignment_data: dict) -> dict:
    return {
        'target_columns':        assignment_data.get('target_columns',        []),
        'column_constraint':     assignment_data.get('column_constraint',      False),
        'all_mentioned_columns': assignment_data.get('all_mentioned_columns', []),
        'requires':              assignment_data.get('requires',              []),
        'produces':              assignment_data.get('produces',              []),
        'output_type':           assignment_data.get('output_type',           'analytical'),
        'ml_step':               assignment_data.get('ml_step',               None),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-EXECUTION DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _check_dependencies(task_desc: str, requires: list, context: dict) -> None:
    if not requires:
        return

    missing = []
    for key_hint in requires:
        key = key_hint.split(':')[0]
        if key not in context:
            missing.append(key)

    if missing:
        raise RuntimeError(
            f"Pre-execution dependency check failed for task:\n"
            f"  '{task_desc[:80]}'\n\n"
            f"Required context keys are missing: {missing}\n\n"
            f"This means an upstream task did not write these values to context.\n"
            f"Check the task that was supposed to produce: {missing}\n"
            f"Current context keys: {list(context.keys()) or '(empty)'}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TASK EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def _execute_task(agent_name: str, task_desc: str, df: pd.DataFrame,
                  task_metadata: dict, context: dict,
                  analytical: AnalyticalAgent, visual: VisualizationAgent,
                  tester: TestingAgent, llm) -> dict:
    task_result = {
        'task':         task_desc,
        'agent':        agent_name,
        'code':         None,
        'ok':           False,
        'result':       None,
        'error':        None,
        'wrapped_code': None,
        'stdout':       None,
        'metadata':     task_metadata,
        'attempts':     1,
    }

    is_analytical    = 'analytical' in agent_name.lower()
    is_visualization = ('visualization' in agent_name.lower()
                        or 'visual' in agent_name.lower())
    _task_start = time.time()

    requires = task_metadata.get('requires', [])
    produces = task_metadata.get('produces', [])

    try:
        try:
            _check_dependencies(task_desc, requires, context)
        except RuntimeError as dep_error:
            task_result['error'] = str(dep_error)
            print(f"[main] ✗ Dependency check failed: {str(dep_error)[:200]}")
            task_result['execution_time'] = round(time.time() - _task_start, 2)
            return task_result

        if is_visualization:
            code = visual.generate_code(task_desc, df, task_metadata, context=context)
        else:
            code = analytical.generate_code(task_desc, df, task_metadata, context=context)

        task_result['code'] = code

        context_validator = None
        if is_analytical:
            def context_validator(ctx, _task=task_desc, _meta=task_metadata):
                AnalyticalAgent._validate_context_writes(_task, ctx, task_metadata=_meta)

        test_output = tester.run_solution(
            code, df,
            llm=llm,
            context=context,
            context_validator=context_validator,
            produces=produces,
        )

        task_result['wrapped_code'] = test_output.get('wrapped_code')
        task_result['stdout']       = test_output.get('stdout', '')
        task_result['attempts']     = test_output.get('attempts', 1)

        if not test_output['ok']:
            task_result['error'] = test_output['error']
            return task_result

        if is_analytical:
            task_result['result'] = AnalyticalAgent.normalise(test_output['result'])
            task_result['ok']     = True
        elif is_visualization:
            task_result['result'] = VisualizationAgent.normalise(test_output['result'])
            task_result['ok']     = task_result['result'].get('status') == 'success'
            if not task_result['ok']:
                task_result['error'] = task_result['result'].get('error')
        else:
            task_result['result'] = test_output['result']
            task_result['ok']     = True

    except Exception as e:
        task_result['error'] = str(e)

    task_result['execution_time'] = round(time.time() - _task_start, 2)
    return task_result


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_task_result(task_result: dict, analytical: AnalyticalAgent):
    agent_name = task_result.get('agent', '')
    attempts   = task_result.get('attempts', 1)

    if task_result.get('code'):
        print("Generated Code:")
        print(task_result['code'])
        print()

    if attempts > 1:
        print(f"  (succeeded after {attempts} attempt(s))")

    if task_result['ok']:
        print("✓ Result: SUCCESS")
        if 'analytical' in agent_name.lower():
            display = analytical.formatter.format_for_display(
                task_result['task'],
                task_result['result']
            )
            print(display)
        elif 'visual' in agent_name.lower() or 'visualization' in agent_name.lower():
            res = task_result['result']
            if isinstance(res, dict):
                print(f"  File saved: {res.get('file', 'unknown')}")
            else:
                print(f"  {res}")
        else:
            print(task_result['result'])
    else:
        print("✗ Result: FAILED")
        print(f"  Error: {task_result.get('error', 'Unknown error')}")

    print("-" * 60)


def _print_context_summary(context: dict):
    if not context:
        print("  (context is empty)")
        return
    for key, val in context.items():
        type_name = type(val).__name__
        if hasattr(val, 'shape'):
            print(f"  context['{key}'] → {type_name}, shape={val.shape}")
        elif hasattr(val, '__len__') and not isinstance(val, str):
            print(f"  context['{key}'] → {type_name}, len={len(val)}")
        else:
            print(f"  context['{key}'] → {type_name}")


def _print_task_plan(task_desc: str, task_metadata: dict):
    output_type = task_metadata.get('output_type', 'analytical')
    ml_step     = task_metadata.get('ml_step')
    requires    = [k.split(':')[0] for k in task_metadata.get('requires', [])]
    produces    = [k.split(':')[0] for k in task_metadata.get('produces', [])]
    cols        = task_metadata.get('target_columns', [])

    print(f"  Type: {output_type}" + (f"  [{ml_step}]" if ml_step else ""))
    if cols:
        print(f"  Columns: {len(cols)} cols")
    if requires:
        print(f"  Requires: {requires}")
    if produces:
        print(f"  Produces: {produces}")


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRAMMATIC PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis_programmatic(dataset_path: str, problem_statement: str,
                               use_rag: bool = True, verbose: bool = False,
                               test_case_name: str = TEST_CASE_NAME,
                               results_dir: str    = RESULTS_DIR) -> dict:
    """
    Programmatic pipeline — called by runner.py.
    test_case_name controls the output filename: results_<test_case_name>.json
    results_dir controls where the file is saved.
    Both default to the global config constants when called without arguments.
    """
    _pipeline_start = time.time()
    llm = LLMClient()

    if use_rag:
        planner = PlannerAgent(
            llm=llm,
            rag_index_path=RAG_INDEX_PATH,
            rag_metadata_path=RAG_METADATA_PATH,
            use_rag=True
        )
    else:
        planner = PlannerAgent(llm=llm, use_rag=False)

    analyzer   = AnalyzerAgent(llm)
    tester     = TestingAgent(max_retries=2)
    analytical = AnalyticalAgent(llm)
    visual     = VisualizationAgent(llm)

    if verbose:
        df = load_dataset(dataset_path)
    else:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2048)
        sep = ';' if sample.count(';') > sample.count(',') else ','
        df = pd.read_csv(dataset_path, sep=sep)
        df.columns = df.columns.str.strip()

    output = {
        'dataset_path':           dataset_path,
        'problem':                problem_statement,
        'use_rag':                use_rag,
        'dataset_shape':          list(df.shape),
        'dataset_columns':        list(df.columns),
        'subtasks':               [],
        'task_assignments':       {},
        'task_results':           [],
        'errors':                 [],
        'raw_subtask_output':     None,
        'raw_analyzer_output':    None,
        'planner_rag_used':       False,
        'planner_rag_retrievals': 0,
        'context':                {},
    }

    try:
        if verbose:
            print(f"\n{'='*60}\nStarting Analysis Pipeline\n{'='*60}")

        planner_result = planner.create_subtasks(problem_statement, df)
        subtasks = planner_result['subtasks']

        output['subtasks']               = subtasks
        output['raw_subtask_output']     = str(subtasks)
        output['planner_rag_used']       = planner_result.get('rag_used', False)
        output['planner_rag_retrievals'] = planner_result.get('rag_retrievals', 0)

        if verbose:
            print("\n[Subtasks]")
            for i, t in enumerate(subtasks, 1):
                if isinstance(t, dict):
                    ml_tag  = f" [{t.get('ml_step')}]" if t.get('ml_step') else ""
                    viz_tag = " [VIZ]" if t.get('output_type') == 'visualization' else ""
                    req     = [k.split(':')[0] for k in t.get('requires', [])]
                    pro     = [k.split(':')[0] for k in t.get('produces', [])]
                    print(f"{i}. {t['description'][:70]}{ml_tag}{viz_tag}")
                    if req:
                        print(f"   → requires: {req}")
                    if pro:
                        print(f"   → produces: {pro}")
                else:
                    print(f"{i}. {t}")
            if output['planner_rag_used']:
                print(f"\n✓ Planner used RAG: {output['planner_rag_retrievals']} cases retrieved")

        assignments = analyzer.assign(subtasks)
        output['task_assignments']    = assignments
        output['raw_analyzer_output'] = str(assignments)

        if verbose:
            print("\n[Task Assignments]")
            for task_desc, data in assignments.items():
                req = [k.split(':')[0] for k in data.get('requires', [])]
                pro = [k.split(':')[0] for k in data.get('produces', [])]
                print(f"  {task_desc[:55]}... → {data['agent']}")
                if req:
                    print(f"    requires: {req}")
                if pro:
                    print(f"    produces: {pro}")
            print(f"\n{'='*60}\nExecuting Tasks\n{'='*60}")

        context = {}

        for task_desc, assignment_data in assignments.items():
            agent_name    = assignment_data['agent']
            task_metadata = _build_task_metadata(assignment_data)

            if verbose:
                print(f"\n[{agent_name}] {task_desc}")
                _print_task_plan(task_desc, task_metadata)
                print("-" * 60)

            task_result = _execute_task(
                agent_name, task_desc, df, task_metadata, context,
                analytical, visual, tester, llm
            )

            if verbose:
                _print_task_result(task_result, analytical)

            output['task_results'].append(task_result)
            if task_result.get('error') and not task_result['ok']:
                output['errors'].append({
                    'task':  task_desc,
                    'agent': agent_name,
                    'error': task_result['error'],
                })

        output['context'] = context

        if verbose:
            print(f"\n{'='*60}\nAnalysis Complete\n{'='*60}")
            rag_msg = ("✓ Planner RAG was used" if output['planner_rag_used']
                       else "✗ Planner RAG was NOT used")
            print(rag_msg)
            if context:
                print("\n[Pipeline Context — accumulated artifacts]")
                _print_context_summary(context)

        pipeline_wall_time = time.time() - _pipeline_start
        llm.print_usage_summary(pipeline_wall_time=pipeline_wall_time)
        _save_results(output, llm, pipeline_wall_time,
                      test_case_name=test_case_name,
                      results_dir=results_dir)

    except Exception as e:
        output['errors'].append({'stage': 'pipeline', 'error': str(e)})
        if verbose:
            print(f"\n✗ Pipeline error: {str(e)}")

    return output


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive mode — prompts user for dataset and problem statement."""
    _pipeline_start = time.time()
    llm = LLMClient()

    planner = PlannerAgent(
        llm=llm,
        rag_index_path=RAG_INDEX_PATH,
        rag_metadata_path=RAG_METADATA_PATH,
        use_rag=True
    )

    analyzer   = AnalyzerAgent(llm)
    analytical = AnalyticalAgent(llm)
    visual     = VisualizationAgent(llm)
    tester     = TestingAgent(max_retries=2)

    try:
        df, problem = get_user_input()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Starting Analysis Pipeline")
    print("=" * 60)

    planner_result = planner.create_subtasks(problem, df)
    subtasks = planner_result['subtasks']

    print("\n[Subtasks]")
    for i, t in enumerate(subtasks, 1):
        if isinstance(t, dict):
            ml_tag  = f" [{t.get('ml_step')}]" if t.get('ml_step') else ""
            viz_tag = " [VIZ]" if t.get('output_type') == 'visualization' else ""
            req     = [k.split(':')[0] for k in t.get('requires', [])]
            pro     = [k.split(':')[0] for k in t.get('produces', [])]
            print(f"{i}. {t['description'][:70]}{ml_tag}{viz_tag}")
            if req:
                print(f"   → requires: {req}")
            if pro:
                print(f"   → produces: {pro}")
        else:
            print(f"{i}. {t}")

    if planner_result.get('rag_used', False):
        print(f"\n✓ Planner used RAG: {planner_result.get('rag_retrievals', 0)} cases retrieved")

    assignments = analyzer.assign(subtasks)

    print("\n[Task Assignments]")
    for task_desc, data in assignments.items():
        print(f"  {task_desc[:60]}... → {data['agent']}")

    print("\n" + "=" * 60)
    print("Executing Tasks")
    print("=" * 60)

    context = {}

    output = {
        'dataset_path':           '',
        'problem':                problem,
        'dataset_shape':          list(df.shape),
        'dataset_columns':        list(df.columns),
        'subtasks':               subtasks,
        'task_results':           [],
        'errors':                 [],
        'planner_rag_used':       planner_result.get('rag_used', False),
        'planner_rag_retrievals': planner_result.get('rag_retrievals', 0),
        'context':                {},
    }

    for task_desc, assignment_data in assignments.items():
        agent_name    = assignment_data['agent']
        task_metadata = _build_task_metadata(assignment_data)

        req = [k.split(':')[0] for k in task_metadata.get('requires', [])]
        pro = [k.split(':')[0] for k in task_metadata.get('produces', [])]

        print(f"\n[{agent_name}] Task: {task_desc}")
        if req:
            print(f"  Requires: {req}")
        if pro:
            print(f"  Produces: {pro}")
        print("-" * 60)

        task_result = _execute_task(
            agent_name, task_desc, df, task_metadata, context,
            analytical, visual, tester, llm
        )
        _print_task_result(task_result, analytical)

        output['task_results'].append(task_result)
        if task_result.get('error') and not task_result['ok']:
            output['errors'].append({
                'task':  task_desc,
                'agent': agent_name,
                'error': task_result['error'],
            })

    output['context'] = context

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    if context:
        print("\n[Pipeline Context — accumulated artifacts]")
        _print_context_summary(context)

    pipeline_wall_time = time.time() - _pipeline_start
    llm.print_usage_summary(pipeline_wall_time=pipeline_wall_time)
    _save_results(output, llm, pipeline_wall_time)


if __name__ == "__main__":
    main()