# api.py
# FastAPI wrapper for CodeViz.
#
# Endpoints:
#   GET  /          - API info
#   GET  /health    - Health check
#   POST /analyse   - Run full CodeViz pipeline (uploads CSV + problem statement)
#   GET  /results/{test_case_name} - Retrieve a saved result by name
#
# Usage (local):
#   uvicorn api:app --reload
#
# Then open http://localhost:8000/docs for the interactive UI.

import os
import sys
import json
import uuid
import tempfile
import glob

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# ── Make sure agents folder is on the path ────────────────────────────────────
AGENTS_DIR = os.path.join(os.path.dirname(__file__), 'agents')
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from main import run_analysis_programmatic

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CodeViz API",
    description=(
        "CodeViz is a task-aware multi-agent framework that automates end-to-end "
        "data science pipelines from natural language problem statements. "
        "Upload a CSV dataset and describe your problem — CodeViz will plan, "
        "generate, execute, and evaluate the full pipeline automatically."
    ),
    version="1.0.0",
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    """API info and available endpoints."""
    return {
        "name":        "CodeViz API",
        "version":     "1.0.0",
        "description": "Multi-agent framework for automated data science pipelines",
        "endpoints": {
            "POST /analyse":                  "Run CodeViz pipeline on uploaded CSV",
            "GET  /results/{test_case_name}": "Retrieve saved result by name",
            "GET  /results":                  "List all saved results",
            "GET  /health":                   "Health check",
            "GET  /docs":                     "Interactive API documentation",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "CodeViz API is running"}


@app.post("/analyse", tags=["Pipeline"])
async def analyse(
    file: UploadFile = File(
        ...,
        description="CSV dataset file to analyse"
    ),
    problem_statement: str = Form(
        ...,
        description=(
            "Natural language description of the data science task. "
            "Example: 'Train a regression model to predict gemstone price and report RMSE.'"
        )
    ),
    use_rag: bool = Form(
        default=True,
        description="Whether to use RAG-augmented planning (recommended)"
    ),
    test_case_name: str = Form(
        default=None,
        description="Optional name for this run. Used as the result filename. "
                    "Defaults to a unique ID if not provided."
    ),
):
    """
    Run the full CodeViz pipeline on an uploaded CSV file.

    - Upload your dataset as a CSV file
    - Provide a natural language problem statement
    - CodeViz will plan, generate, execute, and evaluate the pipeline automatically
    - Returns the full pipeline result including metric score, generated code, and usage stats
    """

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are supported. Received: {file.filename}"
        )

    # Generate a run name if not provided
    run_name = test_case_name or f"api_run_{uuid.uuid4().hex[:8]}"

    # Save uploaded file to a temp location
    try:
        suffix = f"_{file.filename}"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode='wb'
        ) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Run the pipeline
    try:
        output = run_analysis_programmatic(
            dataset_path=tmp_path,
            problem_statement=problem_statement,
            use_rag=use_rag,
            verbose=False,
            test_case_name=run_name,
            results_dir=RESULTS_DIR,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Extract the final metric for a clean top-level summary
    metric_name  = None
    metric_value = None
    for task in output.get('task_results', []):
        if (task.get('metadata', {}).get('ml_step') == 'evaluate'
                and task.get('ok')):
            result = task.get('result', {})
            if result:
                metric_name  = list(result.keys())[0]
                metric_value = list(result.values())[0]

    stats   = output.get('pipeline_stats', {}) if isinstance(output, dict) else {}
    usage   = output.get('usage', {})          if isinstance(output, dict) else {}
    errors  = output.get('errors', [])         if isinstance(output, dict) else []

    return JSONResponse(content={
        "run_name":          run_name,
        "status":            "completed" if not errors else "completed_with_errors",
        "dataset":           file.filename,
        "problem_statement": problem_statement,

        # Top-level metric result
        "result": {
            "metric":  metric_name,
            "score":   metric_value,
        },

        # Pipeline stats
        "pipeline_stats": {
            "total_tasks":      stats.get('total_tasks', 0),
            "successful_tasks": stats.get('successful_tasks', 0),
            "failed_tasks":     stats.get('failed_tasks', 0),
            "retried_tasks":    stats.get('retried_tasks', 0),
        },

        # Usage / cost
        "usage": {
            "llm_calls":        usage.get('llm_calls', 0),
            "total_tokens":     usage.get('total_tokens', 0),
            "total_cost_usd":   usage.get('total_cost_usd', 0),
            "wall_time_s":      usage.get('pipeline_wall_time_s', 0),
            "llm_latency_s":    usage.get('llm_latency_s', 0),
        },

        # Errors if any
        "errors": errors,

        # Where the full result was saved
        "result_saved_to": os.path.join(RESULTS_DIR, f"results_{run_name}.json"),
    })


@app.get("/results", tags=["Results"])
def list_results():
    """List all saved pipeline results."""
    pattern = os.path.join(RESULTS_DIR, "results_*.json")
    files   = sorted(glob.glob(pattern))

    results = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append({
                "name":      data.get('test_case', os.path.basename(f)),
                "timestamp": data.get('timestamp', ''),
                "dataset":   os.path.basename(data.get('dataset_path', '')),
                "tasks":     data.get('pipeline_stats', {}).get('total_tasks', 0),
                "success":   data.get('pipeline_stats', {}).get('successful_tasks', 0),
                "cost_usd":  data.get('usage', {}).get('total_cost_usd', 0),
                "file":      os.path.basename(f),
            })
        except Exception:
            continue

    return {"total": len(results), "results": results}


@app.get("/results/{test_case_name}", tags=["Results"])
def get_result(test_case_name: str):
    """
    Retrieve a saved pipeline result by test case name.
    The name is the value passed as test_case_name when running /analyse.
    """
    filepath = os.path.join(RESULTS_DIR, f"results_{test_case_name}.json")

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"No result found for '{test_case_name}'. "
                   f"Check /results for available runs."
        )

    with open(filepath) as f:
        data = json.load(f)

    return JSONResponse(content=data)
