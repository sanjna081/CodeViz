# analyse_results.py
# Reads all results_test_case_*.json files from RESULTS_DIR and produces:
#   1. A printed table of all timing/cost metrics per test case
#   2. A bar chart of pipeline wall time per test case
#   3. A stacked bar chart showing LLM time vs non-LLM time per test case
#   4. A bar chart of total cost per test case
#
# Usage:
#   python analyse_results.py

import os
import json
import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR  = r'C:\Users\Shreekumar\codeviz\results'
PLOTS_DIR    = r'C:\Users\Shreekumar\codeviz\results\plots'

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ALL RESULT FILES
# ══════════════════════════════════════════════════════════════════════════════

def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, 'results_test_case_*.json')
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No results files found in {results_dir}.\n"
            f"Expected pattern: results_test_case_*.json"
        )

    print(f"Found {len(files)} result file(s):")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print()

    rows = []
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        usage = data.get('usage', {})
        stats = data.get('pipeline_stats', {})

        # Per-task execution times
        task_results     = data.get('task_results', [])
        task_times       = [t.get('execution_time', 0) or 0 for t in task_results]
        total_task_time  = sum(task_times)
        avg_task_time    = total_task_time / len(task_times) if task_times else 0

        rows.append({
            'test_case':           data.get('test_case', ''),
            'dataset':             os.path.basename(data.get('dataset_path', '')),
            'problem_type':        _infer_problem_type(data.get('problem_statement', '')),
            'pipeline_wall_time_s': usage.get('pipeline_wall_time_s', 0),
            'llm_latency_s':       usage.get('llm_latency_s', 0),
            'non_llm_time_s':      usage.get('non_llm_time_s', 0),
            'avg_llm_latency_s':   usage.get('avg_latency_s', 0),
            'total_task_exec_s':   round(total_task_time, 2),
            'avg_task_exec_s':     round(avg_task_time, 2),
            'llm_calls':           usage.get('llm_calls', 0),
            'total_tokens':        usage.get('total_tokens', 0),
            'prompt_tokens':       usage.get('prompt_tokens', 0),
            'completion_tokens':   usage.get('completion_tokens', 0),
            'total_cost_usd':      usage.get('total_cost_usd', 0),
            'total_tasks':         stats.get('total_tasks', 0),
            'successful_tasks':    stats.get('successful_tasks', 0),
            'failed_tasks':        stats.get('failed_tasks', 0),
            'retried_tasks':       stats.get('retried_tasks', 0),
        })

    return pd.DataFrame(rows).sort_values('test_case').reset_index(drop=True)


def _infer_problem_type(problem_statement: str) -> str:
    p = problem_statement.lower()
    if 'time series' in p or 'forecasting' in p or 'mape' in p:
        return 'Time Series'
    if 'multi-label' in p:
        return 'Multi-Label'
    if 'multi-class' in p or 'multiclass' in p:
        return 'Multi-Class'
    if 'regression' in p and 'classification' not in p:
        return 'Regression'
    if 'binary classification' in p or 'roc-auc' in p:
        return 'Binary Classification'
    return 'Unknown'


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_latency_cost_table(df: pd.DataFrame):
    print("=" * 100)
    print("LATENCY & COST TABLE")
    print("=" * 100)

    display = df[[
        'test_case',
        'dataset',
        'problem_type',
        'pipeline_wall_time_s',
        'llm_latency_s',
        'non_llm_time_s',
        'avg_llm_latency_s',
        'total_task_exec_s',
        'avg_task_exec_s',
        'llm_calls',
        'total_tokens',
        'total_cost_usd',
    ]].copy()

    display.columns = [
        'Test Case',
        'Dataset',
        'Problem Type',
        'Wall Time (s)',
        'LLM Time (s)',
        'Non-LLM (s)',
        'Avg LLM/call (s)',
        'Task Exec (s)',
        'Avg Task (s)',
        'LLM Calls',
        'Tokens',
        'Cost (USD)',
    ]

    display['Cost (USD)'] = display['Cost (USD)'].map('${:.5f}'.format)
    display['Wall Time (s)'] = display['Wall Time (s)'].map('{:.1f}'.format)
    display['LLM Time (s)'] = display['LLM Time (s)'].map('{:.1f}'.format)
    display['Non-LLM (s)'] = display['Non-LLM (s)'].map('{:.1f}'.format)
    display['Avg LLM/call (s)'] = display['Avg LLM/call (s)'].map('{:.2f}'.format)
    display['Task Exec (s)'] = display['Task Exec (s)'].map('{:.1f}'.format)
    display['Avg Task (s)'] = display['Avg Task (s)'].map('{:.2f}'.format)
    display['Tokens'] = display['Tokens'].map('{:,}'.format)

    print(display.to_string(index=False))
    print()

    # Summary row
    print("-" * 100)
    print(f"  Total cost across all test cases : ${df['total_cost_usd'].sum():.5f}")
    print(f"  Average wall time per test case  : {df['pipeline_wall_time_s'].mean():.1f}s")
    print(f"  Average LLM calls per test case  : {df['llm_calls'].mean():.1f}")
    print(f"  Average tokens per test case     : {df['total_tokens'].mean():,.0f}")
    print("=" * 100)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'llm':     '#4C72B0',
    'non_llm': '#DD8452',
    'cost':    '#55A868',
    'wall':    '#4C72B0',
}


def _save_or_show(fig, filename: str, plots_dir: str):
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        path = os.path.join(plots_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.show()
    plt.close(fig)


def plot_wall_time(df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(df['test_case'], df['pipeline_wall_time_s'],
                  color=COLORS['wall'], edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, df['pipeline_wall_time_s']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=8)

    ax.set_title('Pipeline Wall Time per Test Case', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Test Case', fontsize=10)
    ax.set_ylabel('Wall Time (seconds)', fontsize=10)
    ax.set_xticklabels(df['test_case'], rotation=45, ha='right', fontsize=8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    _save_or_show(fig, 'wall_time.png', plots_dir)


def plot_stacked_time(df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(df['test_case'], df['llm_latency_s'],
           label='LLM Latency', color=COLORS['llm'], edgecolor='white', linewidth=0.5)
    ax.bar(df['test_case'], df['non_llm_time_s'],
           bottom=df['llm_latency_s'],
           label='Non-LLM Time (execution)', color=COLORS['non_llm'],
           edgecolor='white', linewidth=0.5)

    ax.set_title('LLM vs Non-LLM Time per Test Case', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Test Case', fontsize=10)
    ax.set_ylabel('Time (seconds)', fontsize=10)
    ax.set_xticklabels(df['test_case'], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    _save_or_show(fig, 'stacked_time.png', plots_dir)


def plot_cost(df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(df['test_case'], df['total_cost_usd'],
                  color=COLORS['cost'], edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, df['total_cost_usd']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.00005,
                f'${val:.4f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_title('Total API Cost per Test Case (USD)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Test Case', fontsize=10)
    ax.set_ylabel('Cost (USD)', fontsize=10)
    ax.set_xticklabels(df['test_case'], rotation=45, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.4f'))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    _save_or_show(fig, 'cost.png', plots_dir)


def plot_tokens(df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(df['test_case'], df['prompt_tokens'],
           label='Prompt Tokens', color=COLORS['llm'], edgecolor='white', linewidth=0.5)
    ax.bar(df['test_case'], df['completion_tokens'],
           bottom=df['prompt_tokens'],
           label='Completion Tokens', color=COLORS['non_llm'],
           edgecolor='white', linewidth=0.5)

    ax.set_title('Token Usage per Test Case', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Test Case', fontsize=10)
    ax.set_ylabel('Tokens', fontsize=10)
    ax.set_xticklabels(df['test_case'], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    _save_or_show(fig, 'tokens.png', plots_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE TABLE TO CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_table_csv(df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, 'latency_cost_table.csv')
    df.to_csv(path, index=False)
    print(f"  Table saved to CSV: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_results(RESULTS_DIR)

    print_latency_cost_table(df)

    print("Generating plots...")
    plot_wall_time(df, PLOTS_DIR)
    plot_stacked_time(df, PLOTS_DIR)
    plot_cost(df, PLOTS_DIR)
    plot_tokens(df, PLOTS_DIR)

    save_table_csv(df, PLOTS_DIR)

    print("\nDone.")