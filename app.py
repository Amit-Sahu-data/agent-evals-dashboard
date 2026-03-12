# app.py
# Evals Dashboard — the final piece!

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from rag_agent import create_rag_agent, run_agent, ingest_pdf
from evaluator import evaluate_response
from database import init_db, save_evaluation, get_all_evaluations, get_stats, flag_evaluation

# -----------------------------------------
# Page config
# -----------------------------------------
st.set_page_config(
    page_title="Agent Evals Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize database
init_db()

st.title("📊 Agent Evals Dashboard")
st.caption("Ask questions, evaluate answers, track quality over time")

# -----------------------------------------
# Sidebar — PDF upload + agent setup
# -----------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        temp_path = f"./data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Ingesting PDF..."):
            ingest_pdf(temp_path)
        st.success("PDF ready!")

    if st.button("🤖 Initialize Agent"):
        with st.spinner("Loading agent..."):
            st.session_state.agent = create_rag_agent()
        st.success("Agent ready!")

    st.divider()
    st.caption("Agent initialized automatically on first question")

# Initialize agent in session state
if "agent" not in st.session_state:
    if os.path.exists("./chromadb"):
        st.session_state.agent = create_rag_agent()

# -----------------------------------------
# Main area — tabs
# -----------------------------------------
tab1, tab2, tab3 = st.tabs(["💬 Ask & Evaluate", "📈 Dashboard", "📋 History"])

# ── TAB 1: Ask & Evaluate ──
with tab1:
    st.subheader("Ask a Question")

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What is attention mechanism?"
    )

    if st.button("🚀 Ask + Evaluate", type="primary"):
        if not question:
            st.warning("Please enter a question!")
        elif "agent" not in st.session_state:
            st.warning("Please initialize agent from sidebar first!")
        else:
            with st.spinner("Agent thinking..."):
                run_result = run_agent(question, st.session_state.agent)

            with st.spinner("Evaluating response..."):
                eval_result = evaluate_response(
                    question=run_result['question'],
                    answer=run_result['answer'],
                    latency=run_result['latency'],
                    tool_used=run_result['tool_used']
                )

            # Save to database
            save_evaluation(
                question=run_result['question'],
                answer=run_result['answer'],
                latency=run_result['latency'],
                tool_used=run_result['tool_used'],
                eval_results=eval_result
            )

            # Show answer
            st.subheader("Answer")
            st.write(run_result['answer'])

            # Show metrics
            st.subheader("Evaluation Scores")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Overall", f"{eval_result['overall']}/10")
            with col2:
                st.metric("Relevance", f"{eval_result['relevance']['score']}/10")
            with col3:
                st.metric("Faithfulness", f"{eval_result['faithfulness']['score']}/10")
            with col4:
                st.metric("Completeness", f"{eval_result['completeness']['score']}/10")
            with col5:
                st.metric("Latency", f"{run_result['latency']}s")

            # Show reasons
            with st.expander("See evaluation reasoning"):
                st.write(f"**Relevance:** {eval_result['relevance']['reason']}")
                st.write(f"**Faithfulness:** {eval_result['faithfulness']['reason']}")
                st.write(f"**Completeness:** {eval_result['completeness']['reason']}")
                st.write(f"**Latency:** {eval_result['latency']['reason']}")
                st.write(f"**Tool used:** {run_result['tool_used']}")

# ── TAB 2: Dashboard ──
with tab2:
    st.subheader("Performance Dashboard")

    stats = get_stats()

    if stats['total_runs'] == 0:
        st.info("No evaluations yet. Ask some questions in the Ask & Evaluate tab!")
    else:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", stats['total_runs'])
        with col2:
            st.metric("Avg Overall Score", f"{stats['avg_overall']}/10")
        with col3:
            st.metric("Avg Latency", f"{stats['avg_latency']}s")
        with col4:
            st.metric("Flagged Responses", stats['total_flagged'])

        st.divider()

        # Get all data
        evals = get_all_evaluations()
        df = pd.DataFrame(evals)

        # Score over time chart
        st.subheader("Overall Score Over Time")
        fig1 = px.line(
            df,
            x='timestamp',
            y='overall_score',
            markers=True,
            title="Agent Quality Over Time"
        )
        fig1.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig1, use_container_width=True)

        # Score breakdown chart
        st.subheader("Score Breakdown")
        fig2 = go.Figure(data=go.Bar(
            x=['Relevance', 'Faithfulness', 'Completeness', 'Latency'],
            y=[
                stats['avg_relevance'],
                stats['avg_faithfulness'],
                stats['avg_completeness'],
                stats['avg_overall']
            ],
            marker_color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        ))
        fig2.update_layout(
            title="Average Scores by Dimension",
            yaxis_range=[0, 10]
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: History ──
with tab3:
    st.subheader("Evaluation History")

    evals = get_all_evaluations()

    if not evals:
        st.info("No history yet!")
    else:
        for eval in evals:
            with st.expander(
                f"[{eval['timestamp']}] {eval['question'][:60]}... "
                f"— Overall: {eval['overall_score']}/10"
                f" {'🚩' if eval['flagged'] else ''}"
            ):
                st.write(f"**Answer:** {eval['answer']}")
                st.divider()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Relevance", f"{eval['relevance_score']}/10")
                with col2:
                    st.metric("Faithfulness", f"{eval['faithfulness_score']}/10")
                with col3:
                    st.metric("Completeness", f"{eval['completeness_score']}/10")
                with col4:
                    st.metric("Latency", f"{eval['latency']}s")

                if not eval['flagged']:
                    if st.button(f"🚩 Flag for review", key=f"flag_{eval['id']}"):
                        flag_evaluation(eval['id'])
                        st.rerun()
                else:
                    st.error("🚩 Flagged for review")