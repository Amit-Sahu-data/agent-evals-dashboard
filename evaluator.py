# evaluator.py
# Scores every agent response on 4 dimensions

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# -----------------------------------------
# Score 1: Relevance
# Did the agent answer what was actually asked?
# -----------------------------------------
def score_relevance(question: str, answer: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation expert.
Score how relevant the answer is to the question.
Score from 1-10 where:
10 = perfectly answers the question
5  = partially answers the question
1  = completely off topic

Respond in EXACTLY this format:
SCORE: <number>
REASON: <one sentence>"""),
        ("human", f"Question: {question}\nAnswer: {answer}")
    ])
    chain = prompt | llm
    response = chain.invoke({}).content

    lines = response.strip().split('\n')
    score = int(lines[0].replace('SCORE:', '').strip())
    reason = lines[1].replace('REASON:', '').strip()
    return {"score": score, "reason": reason}

# -----------------------------------------
# Score 2: Faithfulness
# Did the agent stick to facts or hallucinate?
# -----------------------------------------
def score_faithfulness(answer: str, tool_used: bool) -> dict:
    if not tool_used:
        return {
            "score": 7,
            "reason": "Agent answered from general knowledge without document search"
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation expert checking for hallucination.
Analyze if the answer appears factual and grounded.
Score from 1-10 where:
10 = completely factual, no hallucination
5  = some uncertain claims
1  = clearly hallucinated or made up

Respond in EXACTLY this format:
SCORE: <number>
REASON: <one sentence>"""),
        ("human", f"Answer to evaluate:\n{answer}")
    ])
    chain = prompt | llm
    response = chain.invoke({}).content

    lines = response.strip().split('\n')
    score = int(lines[0].replace('SCORE:', '').strip())
    reason = lines[1].replace('REASON:', '').strip()
    return {"score": score, "reason": reason}

# -----------------------------------------
# Score 3: Completeness
# Did the agent give a complete answer?
# -----------------------------------------
def score_completeness(question: str, answer: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluation expert.
Score how complete the answer is.
Score from 1-10 where:
10 = fully complete answer covering all aspects
5  = partially complete
1  = very incomplete or too short

Respond in EXACTLY this format:
SCORE: <number>
REASON: <one sentence>"""),
        ("human", f"Question: {question}\nAnswer: {answer}")
    ])
    chain = prompt | llm
    response = chain.invoke({}).content

    lines = response.strip().split('\n')
    score = int(lines[0].replace('SCORE:', '').strip())
    reason = lines[1].replace('REASON:', '').strip()
    return {"score": score, "reason": reason}

# -----------------------------------------
# Score latency
# -----------------------------------------
def score_latency(latency: float) -> dict:
    if latency < 3:
        score = 10
        reason = "Very fast response under 3 seconds"
    elif latency < 7:
        score = 7
        reason = "Acceptable response time under 7 seconds"
    elif latency < 15:
        score = 5
        reason = "Slow response between 7-15 seconds"
    else:
        score = 2
        reason = "Very slow response over 15 seconds"
    return {"score": score, "reason": reason}

# -----------------------------------------
# Main eval function — runs all 4 scores
# -----------------------------------------
def evaluate_response(question: str, answer: str,
                      latency: float, tool_used: bool) -> dict:
    print("\nEvaluating response...")

    relevance = score_relevance(question, answer)
    faithfulness = score_faithfulness(answer, tool_used)
    completeness = score_completeness(question, answer)
    latency_score = score_latency(latency)

    overall = round(
        (relevance['score'] + faithfulness['score'] +
         completeness['score'] + latency_score['score']) / 4, 1
    )

    return {
        "relevance": relevance,
        "faithfulness": faithfulness,
        "completeness": completeness,
        "latency": latency_score,
        "overall": overall
    }

# -----------------------------------------
# Test it
# -----------------------------------------
if __name__ == "__main__":
    result = evaluate_response(
        question="What is attention mechanism?",
        answer="The attention mechanism allows the model to focus on different parts of input sequence when generating output.",
        latency=10.84,
        tool_used=True
    )

    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    print(f"Relevance:    {result['relevance']['score']}/10 — {result['relevance']['reason']}")
    print(f"Faithfulness: {result['faithfulness']['score']}/10 — {result['faithfulness']['reason']}")
    print(f"Completeness: {result['completeness']['score']}/10 — {result['completeness']['reason']}")
    print(f"Latency:      {result['latency']['score']}/10 — {result['latency']['reason']}")
    print(f"\nOVERALL SCORE: {result['overall']}/10")