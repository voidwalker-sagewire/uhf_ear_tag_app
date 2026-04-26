#!/usr/bin/env python3
"""
HerdMate Vet AI — FastAPI Backend
Deploy this on your DigitalOcean server alongside your existing apps.

pip install fastapi uvicorn chromadb sentence-transformers anthropic requests
uvicorn herdmate_vet_api:app --host 0.0.0.0 --port 8001
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import chromadb
    from chromadb.utils import embedding_functions
    import anthropic
except ImportError:
    os.system("pip install fastapi uvicorn chromadb sentence-transformers anthropic --break-system-packages -q")
    import chromadb
    from chromadb.utils import embedding_functions
    import anthropic

app = FastAPI(title="HerdMate Vet AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://scanner.herdmate.ag", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ──
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
VET_COLLECTION = "herdmate_vet_knowledge"
MEMORY_COLLECTION = "herdmate_vet_memory"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── CLIENTS ──
def get_chroma():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        return client
    except:
        return chromadb.PersistentClient(path="./herdmate_vet_db")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma = get_chroma()
vet_collection = chroma.get_or_create_collection(VET_COLLECTION, embedding_function=ef)
memory_collection = chroma.get_or_create_collection(MEMORY_COLLECTION, embedding_function=ef)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── MODELS ──
class VetQuestion(BaseModel):
    question: str
    operation: Optional[str] = "HerdMate"
    tag_epc: Optional[str] = None       # link to specific animal if scanning
    pasture: Optional[str] = None       # current pasture context
    weather: Optional[str] = None       # current weather
    user_id: Optional[str] = "default"

class VetAnswer(BaseModel):
    answer: str
    sources: list
    similar_past_cases: list
    confidence: str
    timestamp: str

# ── RAG QUERY ──
def search_knowledge(question: str, n_results: int = 5):
    """Search veterinary knowledge base."""
    try:
        results = vet_collection.query(
            query_texts=[question],
            n_results=min(n_results, vet_collection.count() or 1)
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except Exception as e:
        return []

def search_memory(question: str, user_id: str, n_results: int = 3):
    """Search user's personal field history."""
    try:
        if memory_collection.count() == 0:
            return []
        results = memory_collection.query(
            query_texts=[question],
            n_results=min(n_results, memory_collection.count()),
            where={"user_id": user_id} if user_id != "default" else None
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except:
        return []

def save_to_memory(question: str, answer: str, metadata: dict):
    """Save Q&A to personal field memory so it learns over time."""
    try:
        doc_id = hashlib.md5(f"{question}_{datetime.now().isoformat()}".encode()).hexdigest()
        memory_collection.add(
            ids=[doc_id],
            documents=[f"Q: {question}\nA: {answer}"],
            metadatas=[{
                **metadata,
                "timestamp": datetime.now().isoformat(),
                "type": "field_case"
            }]
        )
    except Exception as e:
        pass  # Memory save failure shouldn't break the response

# ── SYSTEM PROMPT ──
VET_SYSTEM_PROMPT = """You are HerdMate Vet AI — a cattle health assistant built for working ranchers in the field.

You are NOT a replacement for a veterinarian. You are a knowledgeable field reference that helps ranchers make informed decisions, especially when a vet isn't immediately available.

Your style:
- Plain language. No jargon unless necessary.
- Direct. Get to the point fast.
- Practical. What do I do RIGHT NOW.
- Honest about uncertainty. If you don't know, say so.
- Always recommend calling a vet for serious conditions.

You have access to:
1. Veterinary knowledge base — from the MSD Veterinary Manual and beef cattle extension publications
2. The rancher's personal field history — past cases and outcomes from their operation

When answering:
- Lead with the most likely cause and immediate action
- Note warning signs that require a vet call
- Reference similar past cases from their history if relevant
- Keep it short enough to read on a phone in bad light at 2am

You are built by HerdMate — sovereign ag-tech for working ranchers. The rancher owns their data."""

@app.post("/vet/ask", response_model=VetAnswer)
async def ask_vet(q: VetQuestion):
    """Main veterinary question endpoint."""

    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Build context from knowledge base
    vet_results = search_knowledge(q.question)
    past_cases = search_memory(q.question, q.user_id)

    # Build context string
    vet_context = ""
    sources = []
    if vet_results:
        vet_context = "\n\n--- VETERINARY KNOWLEDGE BASE ---\n"
        for doc, meta in vet_results:
            source = meta.get("source", "veterinary reference")
            vet_context += f"\n[Source: {source}]\n{doc}\n"
            if source not in sources:
                sources.append(source)

    memory_context = ""
    past_case_summaries = []
    if past_cases:
        memory_context = "\n\n--- YOUR PAST FIELD CASES ---\n"
        for doc, meta in past_cases:
            ts = meta.get("timestamp", "")[:10]
            memory_context += f"\n[{ts}] {doc}\n"
            past_case_summaries.append(f"{ts}: {doc[:100]}...")

    # Build the full prompt
    field_context = ""
    if q.pasture:
        field_context += f"Current pasture: {q.pasture}\n"
    if q.weather:
        field_context += f"Current weather: {q.weather}\n"
    if q.tag_epc:
        field_context += f"Animal tag: {q.tag_epc}\n"
    if q.operation:
        field_context += f"Operation: {q.operation}\n"

    user_message = f"""Field context:
{field_context if field_context else "No specific field context provided"}

Question: {q.question}
{vet_context}
{memory_context}"""

    # Call Claude
    try:
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast and cheap for field queries
            max_tokens=600,
            system=VET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        answer = response.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

    # Save to memory for future learning
    save_to_memory(
        question=q.question,
        answer=answer,
        metadata={
            "user_id": q.user_id,
            "operation": q.operation or "",
            "pasture": q.pasture or "",
            "tag_epc": q.tag_epc or "",
            "weather": q.weather or ""
        }
    )

    return VetAnswer(
        answer=answer,
        sources=sources[:3],
        similar_past_cases=past_case_summaries[:2],
        confidence="high" if vet_results else "low",
        timestamp=datetime.now().isoformat()
    )

@app.get("/vet/status")
async def vet_status():
    """Check knowledge base status."""
    return {
        "status": "online",
        "vet_knowledge_docs": vet_collection.count(),
        "field_memory_docs": memory_collection.count(),
        "ready": vet_collection.count() > 0
    }

@app.get("/vet/health")
async def health():
    return {"status": "ok", "service": "HerdMate Vet AI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

