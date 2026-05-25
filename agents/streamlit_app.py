import streamlit as st
import pandas as pd
import os
import sys
import shutil
from io import StringIO
import json
import base64
from pathlib import Path

# Import agents
from langchain_ollama import ChatOllama
from planner_agent import PlannerAgent
from analyzer_agent import AnalyzerAgent
from analytical_agent import AnalyticalAgent
from visualization_agent import VisualizationAgent
from testing_agent import TestingAgent

# Page configuration
st.set_page_config(
    page_title="CodeViz: Multi-Agent Data Science System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _strip_columns(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all column names.
    Logs a warning if any names were changed.
    """
    original = list(df.columns)
    df.columns = df.columns.str.strip()
    stripped = [o for o, n in zip(original, df.columns) if o != n]
    if stripped:
        st.warning(
            f"⚠ Stripped whitespace from {len(stripped)} column name(s)"
            + (f" in {source}" if source else "")
            + f": {stripped}"
        )
    return df


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from a file path and normalise column names."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

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
        raise ValueError(f"Unsupported file format: {file_ext}")

    return _strip_columns(df, source=os.path.basename(file_path))


def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    """Load dataset from a Streamlit uploaded file and normalise column names."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext == '.csv':
        df = pd.read_csv(uploaded_file)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(uploaded_file)
    elif file_ext == '.json':
        df = pd.read_json(uploaded_file)
    elif file_ext == '.parquet':
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    return _strip_columns(df, source=uploaded_file.name)


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def initialize_agents():
    """Initialize all agents once per session."""
    if 'agents_initialized' not in st.session_state:
        with st.spinner("Initializing agents..."):
            llm = ChatOllama(
                model="llama3.1:8b",
                temperature=0.2,
                num_gpu=0,
                num_ctx=4096
            )

            RAG_INDEX_PATH    = r'C:\Users\Shreekumar\codeviz\rag\faiss_index_unified.bin'
            RAG_METADATA_PATH = r'C:\Users\Shreekumar\codeviz\rag\faiss_metadata_unified.json'

            st.session_state.planner = PlannerAgent(
                llm=llm,
                rag_index_path=RAG_INDEX_PATH,
                rag_metadata_path=RAG_METADATA_PATH,
                use_rag=True
            )
            st.session_state.analyzer   = AnalyzerAgent(llm)
            st.session_state.analytical = AnalyticalAgent(llm)
            st.session_state.visual     = VisualizationAgent(llm)
            st.session_state.tester     = TestingAgent(max_retries=2)
            st.session_state.llm        = llm

            st.session_state.agents_initialized = True


# ══════════════════════════════════════════════════════════════════════════════
#  TASK METADATA BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_task_metadata(assignment_data: dict) -> dict:
    """
    Build the full task_metadata dict from analyzer assignment_data.
    Captures all fields set by the planner and passed through by the analyzer,
    including requires, produces, output_type, ml_step.
    """
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

def _check_dependencies(task_desc: str, requires: list, context: dict) -> str | None:
    """
    Check that every key in requires is present in context before execution.
    Returns an error string if anything is missing, None if all clear.
    """
    if not requires:
        return None

    missing = [kh.split(':')[0] for kh in requires
               if kh.split(':')[0] not in context]

    if missing:
        return (
            f"Pre-execution dependency check failed.\n"
            f"Required context keys missing: {missing}\n"
            f"An upstream task did not write these values.\n"
            f"Current context: {list(context.keys()) or '(empty)'}"
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(df: pd.DataFrame, problem_statement: str) -> dict:
    """
    Run the full multi-agent analysis pipeline.

    Key fixes vs original:
      - Shared context dict passed through every task (ML pipelines work).
      - generate_code() called with context= so LLM sees prior artifacts.
      - task_metadata built via _build_task_metadata() — includes requires,
        produces, output_type, ml_step.
      - Pre-execution dependency check before each task.
      - context_validator wired into tester.run_solution() for analytical tasks.
      - produces passed to tester.run_solution() for targeted fix prompts.
    """
    results = {
        'subtasks':    [],
        'assignments': {},
        'task_results': [],
        'rag_info':    {},
        'errors':      [],
    }

    progress_bar = st.progress(0)
    status_text  = st.empty()

    try:
        # ── Step 1: Plan ──────────────────────────────────────────────────────
        status_text.text("Planner Agent: Breaking down the problem...")
        progress_bar.progress(20)

        planner_result = st.session_state.planner.create_subtasks(
            problem_statement, df
        )
        results['subtasks'] = planner_result['subtasks']
        results['rag_info'] = {
            'used':       planner_result.get('rag_used',         False),
            'retrievals': planner_result.get('rag_retrievals',   0),
            'cases':      planner_result.get('retrieved_cases',  []),
        }

        # ── Step 2: Assign ────────────────────────────────────────────────────
        status_text.text("Analyzer Agent: Assigning tasks...")
        progress_bar.progress(40)

        assignments = st.session_state.analyzer.assign(results['subtasks'])
        results['assignments'] = assignments

        # ── Step 3: Execute ───────────────────────────────────────────────────
        # Single context dict shared across ALL tasks for the entire pipeline.
        # Because dicts are passed by reference, artifacts written by one task
        # (e.g. X_train, fitted scaler, trained model) are automatically
        # visible to every subsequent task without any extra plumbing.
        context = {}

        total_tasks = len(assignments)
        for idx, (task_desc, assignment_data) in enumerate(assignments.items(), 1):
            agent_name    = assignment_data['agent']
            task_metadata = _build_task_metadata(assignment_data)
            requires      = task_metadata.get('requires', [])
            produces      = task_metadata.get('produces', [])

            is_viz = (
                'visualization' in agent_name.lower()
                or 'visual' in agent_name.lower()
            )

            status_text.text(
                f"{agent_name}: Executing task {idx}/{total_tasks}..."
            )
            progress_bar.progress(40 + int((idx / total_tasks) * 55))

            task_result = {
                'task':             task_desc,
                'agent':            agent_name,
                'metadata':         task_metadata,
                'code':             None,
                'ok':               False,
                'result':           None,
                'error':            None,
                'stdout':           None,
                'attempts':         1,
                'is_visualization': is_viz,
                'image_path':       None,
            }

            try:
                # ── Pre-execution dependency check ────────────────────────────
                dep_error = _check_dependencies(task_desc, requires, context)
                if dep_error:
                    task_result['error'] = dep_error
                    results['errors'].append({
                        'task':  task_desc,
                        'agent': agent_name,
                        'error': dep_error,
                    })
                    results['task_results'].append(task_result)
                    continue   # skip execution — don't waste retry budget

                # ── Code generation — context passed so LLM sees prior state ──
                if is_viz:
                    code = st.session_state.visual.generate_code(
                        task_desc, df, task_metadata, context=context
                    )
                else:
                    code = st.session_state.analytical.generate_code(
                        task_desc, df, task_metadata, context=context
                    )

                task_result['code'] = code

                # ── Remove stale output.png before viz execution ───────────────
                output_png = 'output.png'
                if is_viz and os.path.exists(output_png):
                    os.remove(output_png)

                # ── Build context_validator for analytical tasks ───────────────
                context_validator = None
                if not is_viz:
                    def context_validator(ctx,
                                          _task=task_desc,
                                          _meta=task_metadata):
                        AnalyticalAgent._validate_context_writes(
                            _task, ctx, task_metadata=_meta
                        )

                # ── Execute via TestingAgent ───────────────────────────────────
                test_result = st.session_state.tester.run_solution(
                    code, df,
                    llm=st.session_state.llm,
                    context=context,
                    context_validator=context_validator,
                    produces=produces,
                )

                task_result['ok']       = test_result.get('ok',       False)
                task_result['result']   = test_result.get('result')
                task_result['error']    = test_result.get('error')
                task_result['stdout']   = test_result.get('stdout',   '')
                task_result['attempts'] = test_result.get('attempts', 1)

                # ── Capture visualization output ───────────────────────────────
                if is_viz and task_result['ok']:
                    if os.path.exists(output_png):
                        unique_path = f"viz_output_{idx}_{abs(hash(task_desc))}.png"
                        shutil.copy(output_png, unique_path)
                        task_result['image_path'] = unique_path
                    else:
                        task_result['error'] = (
                            "Visualization task reported success but output.png "
                            "was not created."
                        )
                        task_result['ok'] = False

            except Exception as e:
                task_result['error'] = str(e)
                results['errors'].append({
                    'task':  task_desc,
                    'agent': agent_name,
                    'error': str(e),
                })

            results['task_results'].append(task_result)

        progress_bar.progress(100)
        status_text.text("Analysis complete!")

    except Exception as e:
        results['errors'].append({'stage': 'pipeline', 'error': str(e)})
        status_text.text(f"Pipeline error: {str(e)}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display_results(results: dict, df: pd.DataFrame):
    """Display analysis results."""
    st.markdown(
        "<div class='sub-header'>Analysis Results</div>",
        unsafe_allow_html=True
    )

    # ── Subtasks ──────────────────────────────────────────────────────────────
    with st.expander("**Subtasks Generated**", expanded=True):
        if results['rag_info'].get('used'):
            st.info(
                f"✓ RAG used: {results['rag_info']['retrievals']} "
                f"similar case(s) retrieved"
            )

        for i, subtask in enumerate(results['subtasks'], 1):
            if isinstance(subtask, dict):
                ml_tag  = f" [{subtask['ml_step']}]"  if subtask.get('ml_step')              else ""
                viz_tag = " [VIZ]"                     if subtask.get('output_type') == 'visualization' else ""
                req     = [k.split(':')[0] for k in subtask.get('requires', [])]
                pro     = [k.split(':')[0] for k in subtask.get('produces', [])]

                st.write(f"{i}. {subtask['description']}{ml_tag}{viz_tag}")
                if subtask.get('target_columns') and set(subtask['target_columns']) != set(df.columns):
                    st.caption(f"   → Columns: {', '.join(subtask['target_columns'])}")
                if req:
                    st.caption(f"   → Requires: {req}")
                if pro:
                    st.caption(f"   → Produces: {pro}")
            else:
                st.write(f"{i}. {subtask}")

    # ── Task assignments ──────────────────────────────────────────────────────
    with st.expander("**Task Assignments**"):
        for task_desc, assignment_data in results['assignments'].items():
            agent_name = assignment_data['agent']
            req = [k.split(':')[0] for k in assignment_data.get('requires', [])]
            pro = [k.split(':')[0] for k in assignment_data.get('produces', [])]
            st.write(f"**{agent_name}**: {task_desc}")
            if req:
                st.caption(f"   Requires: {req}")
            if pro:
                st.caption(f"   Produces: {pro}")

    # ── Per-task outputs ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='sub-header'>Outputs</div>",
        unsafe_allow_html=True
    )

    for idx, task_result in enumerate(results['task_results'], 1):
        task     = task_result['task']
        agent    = task_result['agent']
        is_viz   = task_result.get('is_visualization', False)
        attempts = task_result.get('attempts', 1)

        with st.expander(f"**Task {idx}: {task[:80]}**", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Agent:** {agent}")
                meta = task_result.get('metadata', {})
                ml_step = meta.get('ml_step')
                if ml_step:
                    st.caption(f"ML step: {ml_step}")
                if attempts > 1:
                    st.caption(f"Succeeded after {attempts} attempt(s)")
            with col2:
                if task_result['ok']:
                    st.success("✓ Success")
                else:
                    st.error("✗ Failed")

            # ── Result display ────────────────────────────────────────────────
            if task_result['ok']:
                if is_viz and task_result.get('image_path'):
                    img_path = task_result['image_path']
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                        st.caption("Visualization generated successfully")
                    else:
                        st.warning(f"Image file not found: {img_path}")

                elif task_result['result'] is not None:
                    result = task_result['result']

                    if isinstance(result, dict):
                        if any(k in result for k in ('plot_path', 'image')):
                            path = result.get('plot_path') or result.get('image')
                            if path and os.path.exists(str(path)):
                                st.image(str(path), use_container_width=True)
                            else:
                                st.json(result)
                        else:
                            st.json(result)

                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)

                    elif isinstance(result, str):
                        if result.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                            if os.path.exists(result):
                                st.image(result, use_container_width=True)
                            else:
                                st.info(result)
                        else:
                            st.write(result)

                    else:
                        st.write(str(result))

                if task_result.get('stdout'):
                    with st.expander("Console Output"):
                        st.code(task_result['stdout'], language='text')

            else:
                st.error(f"Error: {task_result.get('error', 'Unknown error')}")

            # ── Generated code ────────────────────────────────────────────────
            if task_result.get('code'):
                with st.expander("View Generated Code"):
                    st.code(task_result['code'], language='python')

    # ── Pipeline errors ───────────────────────────────────────────────────────
    if results['errors']:
        st.markdown(
            "<div class='sub-header'>Errors</div>",
            unsafe_allow_html=True
        )
        for error in results['errors']:
            label = error.get('task') or error.get('stage') or 'Unknown'
            st.error(f"**{label}**: {error['error']}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(
        "<div class='main-header'>CodeViz: Multi-Agent Data Science System</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Session state initialisation
    for key, default in [
        ('dataset_loaded',    False),
        ('analysis_complete', False),
        ('df',                None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    initialize_agents()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Dataset Input")

        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Enter File Path"],
        )

        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            )
            if uploaded_file is not None:
                try:
                    st.session_state.df             = load_dataset_from_upload(uploaded_file)
                    st.session_state.dataset_loaded = True
                    st.success(f"Dataset loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
                    st.session_state.dataset_loaded = False
        else:
            file_path = st.text_input(
                "Enter file path:",
                placeholder="C:/path/to/your/dataset.csv"
            )
            if file_path and st.button("Load Dataset"):
                try:
                    st.session_state.df             = load_dataset(file_path)
                    st.session_state.dataset_loaded = True
                    st.success("Dataset loaded!")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
                    st.session_state.dataset_loaded = False

        if st.session_state.dataset_loaded and st.session_state.df is not None:
            st.markdown("---")
            st.subheader("Dataset Info")
            df = st.session_state.df
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            preview_cols = list(df.columns[:5])
            st.write(f"**Columns:** {', '.join(preview_cols)}")
            if len(df.columns) > 5:
                st.write(f"... and {len(df.columns) - 5} more")

    # ── Main area ─────────────────────────────────────────────────────────────
    if not st.session_state.dataset_loaded:
        st.info("Please load a dataset from the sidebar to begin.")
        st.markdown("### Example Usage")
        st.markdown("""
        1. **Upload your dataset** or provide a file path in the sidebar
        2. **Enter your problem statement**
        3. **Click Start Analysis**
        4. **View results** including visualizations and generated code
        """)
        return

    st.markdown("### Dataset Preview")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.markdown("---")

    st.markdown("### Problem Statement")
    problem_statement = st.text_area(
        "Your problem statement:",
        placeholder=(
            "Example: Analyze the relationship between features X and Y, "
            "find patterns, and create visualizations"
        ),
        height=100,
        label_visibility="collapsed",
    )

    if problem_statement:
        st.markdown("---")
        st.markdown("### Confirmation")
        col1, col2 = st.columns(2)
        with col1:
            df = st.session_state.df
            st.write(
                "**Dataset:**",
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )
        with col2:
            preview = (problem_statement[:100] + "..."
                       if len(problem_statement) > 100
                       else problem_statement)
            st.write("**Problem:**", preview)

        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("Start Analysis", type="primary", use_container_width=True):
                st.session_state.analysis_complete = False
                with st.spinner("Running analysis..."):
                    results = run_analysis(st.session_state.df, problem_statement)
                    st.session_state.results          = results
                    st.session_state.analysis_complete = True
                st.rerun()

        with col2:
            if st.button("Reset", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.dataset_loaded    = False
                st.session_state.df                = None
                st.rerun()

    if st.session_state.analysis_complete and 'results' in st.session_state:
        st.markdown("---")
        display_results(st.session_state.results, st.session_state.df)


if __name__ == "__main__":
    main()