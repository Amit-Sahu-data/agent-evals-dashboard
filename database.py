# database.py
# Saves every agent run and eval scores to SQLite
# SQLite = free local database, no setup needed

import sqlite3
import json
from datetime import datetime

DB_PATH = "./evals.db"

# -----------------------------------------
# Create database and table
# -----------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            latency REAL,
            tool_used INTEGER,
            relevance_score REAL,
            faithfulness_score REAL,
            completeness_score REAL,
            latency_score REAL,
            overall_score REAL,
            relevance_reason TEXT,
            faithfulness_reason TEXT,
            completeness_reason TEXT,
            latency_reason TEXT,
            flagged INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized!")

# -----------------------------------------
# Save one evaluation run
# -----------------------------------------
def save_evaluation(question: str, answer: str,
                    latency: float, tool_used: bool,
                    eval_results: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO evaluations (
            timestamp, question, answer, latency, tool_used,
            relevance_score, faithfulness_score,
            completeness_score, latency_score, overall_score,
            relevance_reason, faithfulness_reason,
            completeness_reason, latency_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question, answer, latency, int(tool_used),
        eval_results['relevance']['score'],
        eval_results['faithfulness']['score'],
        eval_results['completeness']['score'],
        eval_results['latency']['score'],
        eval_results['overall'],
        eval_results['relevance']['reason'],
        eval_results['faithfulness']['reason'],
        eval_results['completeness']['reason'],
        eval_results['latency']['reason']
    ))

    conn.commit()
    conn.close()
    print("Evaluation saved to database!")

# -----------------------------------------
# Get all evaluations
# -----------------------------------------
def get_all_evaluations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM evaluations
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    columns = [
        'id', 'timestamp', 'question', 'answer',
        'latency', 'tool_used',
        'relevance_score', 'faithfulness_score',
        'completeness_score', 'latency_score', 'overall_score',
        'relevance_reason', 'faithfulness_reason',
        'completeness_reason', 'latency_reason', 'flagged'
    ]

    return [dict(zip(columns, row)) for row in rows]

# -----------------------------------------
# Flag a bad response for review
# -----------------------------------------
def flag_evaluation(eval_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE evaluations SET flagged = 1 WHERE id = ?",
        (eval_id,)
    )
    conn.commit()
    conn.close()
    print(f"Evaluation {eval_id} flagged for review!")

# -----------------------------------------
# Get summary statistics
# -----------------------------------------
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_runs,
            ROUND(AVG(overall_score), 2) as avg_overall,
            ROUND(AVG(relevance_score), 2) as avg_relevance,
            ROUND(AVG(faithfulness_score), 2) as avg_faithfulness,
            ROUND(AVG(completeness_score), 2) as avg_completeness,
            ROUND(AVG(latency), 2) as avg_latency,
            SUM(flagged) as total_flagged
        FROM evaluations
    """)

    row = cursor.fetchone()
    conn.close()

    return {
        'total_runs': row[0],
        'avg_overall': row[1],
        'avg_relevance': row[2],
        'avg_faithfulness': row[3],
        'avg_completeness': row[4],
        'avg_latency': row[5],
        'total_flagged': row[6]
    }

# -----------------------------------------
# Test it
# -----------------------------------------
if __name__ == "__main__":
    init_db()

    # Save a fake evaluation
    save_evaluation(
        question="What is attention mechanism?",
        answer="The attention mechanism allows model to focus on input.",
        latency=10.84,
        tool_used=True,
        eval_results={
            'relevance': {'score': 10, 'reason': 'Perfectly answered'},
            'faithfulness': {'score': 9, 'reason': 'Factual answer'},
            'completeness': {'score': 6, 'reason': 'Missing some details'},
            'latency': {'score': 5, 'reason': 'Slow response'},
            'overall': 7.5
        }
    )

    # Get stats
    stats = get_stats()
    print("\nDatabase Stats:")
    print(f"Total runs: {stats['total_runs']}")
    print(f"Avg overall score: {stats['avg_overall']}/10")
    print(f"Avg latency: {stats['avg_latency']} seconds")