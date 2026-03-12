# Agent Evals Dashboard 📊

A quality measurement system for AI agents.
Ask questions to a RAG agent and track answer quality over time.

## What it measures
- Relevance — Did it answer what was asked?
- Faithfulness — Did it stick to facts?
- Completeness — Was the answer complete?
- Latency — How fast did it respond?

## Tech Stack
- LangGraph + LangChain — agent framework
- Groq (LLaMA 3.1) — free LLM
- SQLite — stores all eval runs
- Plotly — charts and visualizations
- Streamlit — dashboard UI

## How to run
1. Clone the repo
2. pip install -r requirements.txt
3. Add GROQ_API_KEY to .env
4. python rag_agent.py (first time only)
5. streamlit run app.py