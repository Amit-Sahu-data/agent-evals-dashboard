# rag_agent.py
import time
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Global document store — simple list of chunks
DOCUMENT_CHUNKS = []

# -----------------------------------------
# Step 1: Ingest PDF into memory
# No embeddings needed — we use LLM to find relevant chunks
# -----------------------------------------
def ingest_pdf(pdf_path: str):
    global DOCUMENT_CHUNKS
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    DOCUMENT_CHUNKS = [chunk.page_content for chunk in chunks]
    print(f"Loaded {len(DOCUMENT_CHUNKS)} chunks into memory")
    return DOCUMENT_CHUNKS

# -----------------------------------------
# Step 2: Load chunks from saved file
# -----------------------------------------
def load_chunks():
    global DOCUMENT_CHUNKS
    if DOCUMENT_CHUNKS:
        return DOCUMENT_CHUNKS
    chunks_file = "./chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r') as f:
            DOCUMENT_CHUNKS = json.load(f)
    return DOCUMENT_CHUNKS

def save_chunks():
    with open("./chunks.json", 'w') as f:
        json.dump(DOCUMENT_CHUNKS, f)

# -----------------------------------------
# Step 3: Simple keyword search
# No embeddings — just find chunks containing query words
# -----------------------------------------
def simple_search(query: str, k: int = 3) -> list:
    chunks = load_chunks()
    if not chunks:
        return []

    query_words = query.lower().split()
    scored = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:k]]

# -----------------------------------------
# Step 4: Search tool
# -----------------------------------------
@tool
def search_document(query: str) -> str:
    """
    Search the document for relevant information.
    Use this when question is technical or document specific.
    """
    results = simple_search(query, k=3)
    if not results:
        return "No relevant information found in the document."
    context = "\n\n---\n\n".join(results)
    return f"Relevant content from document:\n\n{context}"

# -----------------------------------------
# Step 5: Create agent
# -----------------------------------------
def create_rag_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    tools = [search_document]

    system_prompt = """You are a helpful assistant with access to a document.

CRITICAL RULES:
1. For ANY technical question about AI, transformers, attention, neural networks
   → ALWAYS call search_document tool FIRST
2. For simple general knowledge (capitals, math) → answer directly
3. After searching, answer ONLY from the search results
4. If search returns nothing relevant → say I could not find this in the document
5. NEVER skip the search_document tool for technical questions"""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    return agent

# -----------------------------------------
# Step 6: Run agent and measure
# -----------------------------------------
def run_agent(question: str, agent):
    start_time = time.time()

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    end_time = time.time()
    latency = round(end_time - start_time, 2)
    answer = result['messages'][-1].content

    tool_used = any(
        hasattr(msg, 'type') and msg.type == "tool"
        for msg in result['messages']
    )

    return {
        "question": question,
        "answer": answer,
        "latency": latency,
        "tool_used": tool_used
    }

# -----------------------------------------
# Test
# -----------------------------------------
if __name__ == "__main__":
    ingest_pdf("./data/sample.pdf")
    save_chunks()

    agent = create_rag_agent()
    result = run_agent("What is attention mechanism?", agent)

    print("\n" + "="*50)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Latency: {result['latency']} seconds")
    print(f"Tool used: {result['tool_used']}")