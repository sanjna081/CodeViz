# runner.py
# Automated test case runner for CodeViz experiments.
#
# Usage:
#   python runner.py

import sys
import os

# ── Make sure agents folder is on the path ────────────────────────────────────
AGENTS_DIR = r'C:\Users\Shreekumar\codeviz\agents'
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from main import run_analysis_programmatic

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS OUTPUT DIRECTORY
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = r'C:\Users\Shreekumar\codeviz\results'

# ══════════════════════════════════════════════════════════════════════════════
#  TEST CASES 17 – 18
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        'name':              'test_case_17',
        'dataset_path':      r'C:\Users\Shreekumar\codeviz\data\enzyme.csv',
        'problem_statement': (
            "Act as a data scientist and provide a working solution using Python3 "
            "for the problem described below. You are given a dataset of enzyme "
            "substrate records with molecular feature columns. The target columns "
            "are 'EC1' and 'EC2', representing two binary enzyme class labels. "
            "Train a multi-label classification model to predict both 'EC1' and "
            "'EC2' simultaneously and report the Label Ranking Average Precision "
            "(LRAP) score."
        ),
    },
    {
        'name':              'test_case_18',
        'dataset_path':      r'C:\Users\Shreekumar\codeviz\data\mini_course.csv',
        'problem_statement': (
            "Act as a data scientist and provide a working solution using Python3 "
            "for the problem described below. You are given a dataset of historical "
            "mini-course sales records containing features such as 'date', 'country', "
            "'store', and 'course'. The target column is 'num_sold', representing "
            "the number of courses sold. Build a time series forecasting model to "
            "predict future sales using date-based features and report the Symmetric "
            "Mean Absolute Percentage Error (SMAPE) score."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    total  = len(TEST_CASES)
    passed = 0
    failed = 0

    print("=" * 60)
    print(f"CodeViz Experiment Runner — {total} test case(s)")
    print("=" * 60)

    for i, tc in enumerate(TEST_CASES, 1):
        name    = tc['name']
        dataset = tc['dataset_path']
        problem = tc['problem_statement']

        print(f"\n[Runner] ── Test case {i}/{total}: {name} ──")
        print(f"[Runner] Dataset : {dataset}")
        print(f"[Runner] Problem : {problem[:80]}...")
        print()

        if not os.path.exists(dataset):
            print(f"[Runner] ✗ SKIPPED — dataset not found: {dataset}")
            failed += 1
            continue

        try:
            output = run_analysis_programmatic(
                dataset_path=dataset,
                problem_statement=problem,
                use_rag=True,
                verbose=True,
                test_case_name=name,
                results_dir=RESULTS_DIR,
            )

            errors = output.get('errors', [])
            if errors:
                print(f"\n[Runner] ⚠ Test case {name} completed with {len(errors)} error(s):")
                for e in errors:
                    print(f"  - {e}")
                failed += 1
            else:
                print(f"\n[Runner] ✓ Test case {name} completed successfully.")
                passed += 1

        except Exception as e:
            print(f"\n[Runner] ✗ Test case {name} crashed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"[Runner] Done — {passed}/{total} passed, {failed}/{total} failed")
    print(f"[Runner] Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()