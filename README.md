## CodeViz: A Multi-Agent Framework for Automating Data Science Workflows

CodeViz is a task-aware multi-agent framework that automates data science pipelines. Given a dataset and a plain English description of the task, CodeViz plans, generates, executes, and evaluates a complete machine learning pipeline.
It is built with Python, LangChain, OpenRouter API, and a RAG-augmented planning system backed by FAISS vector search.

## Try the app
Live demo: https://codeviz.streamlit.app/

## Flowchart

![Flowchart](images/codeviz_flowchart.png)

## Solution Walkthrough

### Planner Agent
The Planner Agent decomposes the natural language problem statement into an ordered sequence of subtasks, with metadata specifying its ML step  type and the context keys it must read and write. For this query, five subtasks are generated: handling missing values, splitting the data, normalising features, training the model, and computing evaluation metrics.

![Planner Agent Subtasks](figures/subtasks.png)

### Analyzer Agent
The Analyzer Agent routes each subtask to the appropriate execution agent based on its output type. Since this problem involves no visualization, all five tasks are routed to the Analytical Agent.

![Analyzer Agent](figures/analyzer.png)


### Analytical Agent
The Analytical Agent generates a `solution(df, context)` function for each subtask, executing them sequentially and propagating outputs through a shared context dictionary. Missing values are handled via median imputation; the data is split 80:20; features are standardised; the model is trained and its coefficients written to context; finally, the evaluation task reads the model and held-out test data from context and reports RMSE and R².

![Output 1](figures/output1.png)

![Output 2](figures/output2.png)

![Output 3](figures/output3.png)

![Output 4](figures/output4.png)
