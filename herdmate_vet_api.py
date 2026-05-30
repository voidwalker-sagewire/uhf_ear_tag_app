#!/usr/bin/env python3
"""
HerdMate DAVE Vet AI — FastAPI Backend v3 (restored canonical)
Uses Google Service Account for permanent server-side auth.
No OAuth tokens. No browser dependency. Works forever.

This is the full RAG-powered DAVE: ChromaDB knowledge base + field memory,
Google Sheets animal lookup via service account, the DAVE triage prompt,
photo support, and conversation history. NOT a bare Claude wrapper.

Run:
    pip install fastapi uvicorn chromadb sentence-transformers anthropic \
        google-auth requests --break-system-packages
    export ANTHROPIC_API_KEY='sk-ant-...'        # required, never hardcode
    export CREDENTIALS_FILE='/root/credentials.json'   # service account JSON
    python3 herdmate_vet_api.py                  # serves on port 8001

Optional env vars: CHROMA_HOST, CHROMA_PORT, CLAUDE_MODEL, PORT.
Front it with HTTPS (Certbot/Cloudflare) so the field app's Bluetooth and
microphone work — browsers block those on plain HTTP.
"""

import os
import json
import hashlib
import requests as http_requests
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb
from chromadb.utils import embedding_functions
import anthropic
from google.oauth2.service_account import Credentials
import google.auth.transport.requests

app = FastAPI(title="HerdMate DAVE Vet AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://scanner.herdmate.ag",
        "https://api.herdmate.ag",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ──
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
VET_COLLECTION = "herdmate_vet_knowledge"
MEMORY_COLLECTION = "herdmate_vet_memory"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CREDENTIALS_FILE = os.environ.get("CREDENTIALS_FILE", "/root/credentials.json")
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
SERVER_PORT = int(os.environ.get("PORT", "8001"))

# Fail fast and loud if the key is missing. This is the single guard that
# prevents DAVE from silently degrading into a keyless wrapper that returns
# 500s on every question. The real DAVE always has its key in the environment.
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Export it before starting DAVE:\n"
        "    export ANTHROPIC_API_KEY='sk-ant-...'\n"
        "Never hardcode the key into this file."
    )

# ── SERVICE ACCOUNT AUTH ──
_service_creds = None

def get_service_token():
    global _service_creds
    try:
        if _service_creds is None:
            _service_creds = Credentials.from_service_account_file(
                CREDENTIALS_FILE, scopes=SHEETS_SCOPES
            )
        if not _service_creds.valid:
            auth_req = google.auth.transport.requests.Request()
            _service_creds.refresh(auth_req)
        return _service_creds.token
    except Exception as e:
        print(f"Service account auth error: {e}")
        return None

# ── SIMPLE CACHE ──
_animal_cache: dict = {}
CACHE_TTL_SECONDS = 300

def get_cached_animal(key: str):
    import time
    if key in _animal_cache:
        record, ts = _animal_cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return record
    return "MISS"

def set_cached_animal(key: str, record):
    import time
    _animal_cache[key] = (record, time.time())

# ── CHROMA CLIENT ──
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
    tag_epc: Optional[str] = None
    pasture: Optional[str] = None
    weather: Optional[str] = None
    user_id: Optional[str] = "default"
    conversation_history: list = Field(default_factory=list)
    image_base64: Optional[str] = None
    image_type: Optional[str] = "image/jpeg"
    google_access_token: Optional[str] = None   # deprecated - server uses service account
    herdmate_sheet_id: Optional[str] = None
    google_user_email: Optional[str] = None

class VetAnswer(BaseModel):
    answer: str
    sources: list
    similar_past_cases: list
    confidence: str
    timestamp: str
    animal_context: Optional[dict] = None

# ── GOOGLE SHEETS LOOKUP ──
def sheets_get(token: str, sheet_id: str, range_name: str):
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/{range_name}"
    try:
        resp = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        if resp.ok:
            return resp.json().get("values", [])
        else:
            print(f"Sheets error {resp.status_code}: {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"Sheets request error: {e}")
        return []

def find_animal(sheet_id: str, tag_identifier: str):
    """
    Look up an animal by tag number or UHF EPC.
    Searches Calf Tracker first, then Ranch Tracker.
    tag_identifier can be a visual tag number (3476) or UHF EPC.
    """
    if not tag_identifier or not sheet_id:
        return None

    cache_key = f"{sheet_id}_{tag_identifier}"
    cached = get_cached_animal(cache_key)
    if cached != "MISS":
        return cached

    token = get_service_token()
    if not token:
        return None

    tag = str(tag_identifier).strip()

    # ── SEARCH CALF TRACKER ──
    try:
        calf_data = sheets_get(token, sheet_id, "Calf Tracker!A:AF")
        if calf_data and len(calf_data) > 1:
            headers = calf_data[0]
            for row in calf_data[1:]:
                if not row:
                    continue
                row_dict = dict(zip(headers, row + [""] * max(0, len(headers) - len(row))))
                calf_tag = str(row_dict.get("Calf Tag", "")).strip()
                uhf = str(row_dict.get("UHF#", "")).strip()
                if calf_tag == tag or (uhf and uhf == tag):
                    result = {
                        "source": "Calf Tracker",
                        "tag": calf_tag,
                        "uhf": uhf,
                        "date": str(row_dict.get("Date", ""))[:10],
                        "color": row_dict.get("Calf Color", ""),
                        "sex": row_dict.get("Calf Sex", ""),
                        "type": row_dict.get("Calf Type", ""),
                        "birth_weight": row_dict.get("Birth Weight", ""),
                        "dam_tag": row_dict.get("Cow Tag", ""),
                        "season": row_dict.get("Calving Season", ""),
                        "status": row_dict.get("Status", ""),
                        "assisted": row_dict.get("Assisted Y/N", ""),
                        "is_twin": row_dict.get("Is Twin", ""),
                        "dam_bcs": row_dict.get("Dam BCS", ""),
                        "udder": row_dict.get("Udder Condition", ""),
                        "notes": row_dict.get("Calving Notes", ""),
                        "gps": row_dict.get("User Location", ""),
                        "tagger": row_dict.get("Created By", ""),
                    }
                    set_cached_animal(cache_key, result)
                    return result
    except Exception as e:
        print(f"Calf Tracker search error: {e}")

    # ── SEARCH RANCH TRACKER ──
    try:
        ranch_data = sheets_get(token, sheet_id, "Ranch Tracker!A:BZ")
        if ranch_data and len(ranch_data) > 1:
            headers = ranch_data[0]
            for row in ranch_data[1:]:
                if not row:
                    continue
                row_dict = dict(zip(headers, row + [""] * max(0, len(headers) - len(row))))
                tag_num = str(row_dict.get("Tag #", "")).strip()
                uhf = str(row_dict.get("UHF#", "")).strip()
                if tag_num == tag or (uhf and uhf == tag):
                    result = {
                        "source": "Ranch Tracker",
                        "tag": tag_num,
                        "uhf": uhf,
                        "display_id": row_dict.get("DisplayID", ""),
                        "sex": row_dict.get("Sex", ""),
                        "breed": row_dict.get("Breed", ""),
                        "type": row_dict.get("Type", ""),
                        "color": row_dict.get("Color", ""),
                        "birth_date": str(row_dict.get("Birth Date", ""))[:10],
                        "age": row_dict.get("Age", ""),
                        "pasture": row_dict.get("Pasture", ""),
                        "status": row_dict.get("Status", ""),
                        "weight": row_dict.get("Weight", ""),
                        "dam": row_dict.get("Dam #", ""),
                        "sire": row_dict.get("Sire #", ""),
                        "due_date": str(row_dict.get("Due Date", ""))[:10],
                        "palp_result": row_dict.get("Palp. Result", ""),
                        "months_preg": row_dict.get("Mth. Preg.", ""),
                        "bcs": row_dict.get("BCS", ""),
                        "disposition": row_dict.get("Disposition", ""),
                        "notes": row_dict.get("Notes", ""),
                    }
                    set_cached_animal(cache_key, result)
                    return result
    except Exception as e:
        print(f"Ranch Tracker search error: {e}")

    set_cached_animal(cache_key, None)
    return None

def format_animal_context(animal: dict) -> str:
    if not animal:
        return ""
    lines = [f"--- ANIMAL RECORD ({animal.get('source', 'HerdMate')}) ---"]
    lines.append(f"Tag #: {animal.get('tag', 'Unknown')}")
    if animal.get('uhf'): lines.append(f"UHF EPC: {animal['uhf']}")
    if animal.get('sex'): lines.append(f"Sex: {animal['sex']}")
    if animal.get('breed'): lines.append(f"Breed: {animal['breed']}")
    if animal.get('color'): lines.append(f"Color: {animal['color']}")
    if animal.get('type'): lines.append(f"Type: {animal['type']}")
    if animal.get('date'): lines.append(f"Born: {animal['date']}")
    if animal.get('birth_date'): lines.append(f"Born: {animal['birth_date']}")
    if animal.get('age'): lines.append(f"Age: {animal['age']} years")
    if animal.get('season'): lines.append(f"Season: {animal['season']}")
    if animal.get('status'): lines.append(f"Status: {animal['status']}")
    if animal.get('birth_weight'): lines.append(f"Birth Weight: {animal['birth_weight']} lbs")
    if animal.get('weight'): lines.append(f"Current Weight: {animal['weight']} lbs")
    if animal.get('dam_tag') or animal.get('dam'): lines.append(f"Dam Tag: {animal.get('dam_tag') or animal.get('dam')}")
    if animal.get('sire'): lines.append(f"Sire Tag: {animal['sire']}")
    if animal.get('due_date') and str(animal['due_date']) not in ['', 'None', '1900-01-00']: lines.append(f"Due Date: {animal['due_date']}")
    if animal.get('palp_result'): lines.append(f"Palpation: {animal['palp_result']}")
    if animal.get('months_preg'): lines.append(f"Months Pregnant: {animal['months_preg']}")
    if animal.get('bcs'): lines.append(f"BCS: {animal['bcs']}")
    if animal.get('dam_bcs'): lines.append(f"Dam BCS at birth: {animal['dam_bcs']}")
    if animal.get('udder'): lines.append(f"Dam Udder: {animal['udder']}")
    if animal.get('assisted') and str(animal['assisted']).lower() in ['yes', 'y', 'true']: lines.append(f"Assisted Birth: Yes")
    if animal.get('is_twin') and str(animal['is_twin']).lower() in ['yes', 'y', 'true', '1']: lines.append(f"Twin: Yes")
    if animal.get('pasture'): lines.append(f"Pasture: {animal['pasture']}")
    if animal.get('disposition'): lines.append(f"Disposition: {animal['disposition']}")
    if animal.get('notes') and str(animal['notes']) not in ['', 'None']: lines.append(f"Notes: {animal['notes']}")
    return "\n".join(lines)

# ── RAG ──
def search_knowledge(question: str, n_results: int = 5):
    try:
        results = vet_collection.query(
            query_texts=[question],
            n_results=min(n_results, vet_collection.count() or 1)
        )
        return list(zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]))
    except Exception as e:
        print(f"Knowledge search error: {e}")
        return []

def search_memory(question: str, user_id: str, n_results: int = 3):
    try:
        if memory_collection.count() == 0:
            return []
        where_filter = {"user_id": {"$eq": user_id}}
        results = memory_collection.query(
            query_texts=[question],
            n_results=min(n_results, memory_collection.count()),
            where=where_filter
        )
        return list(zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]))
    except Exception as e:
        print(f"Memory search error: {e}")
        return []

def save_to_memory(question: str, answer: str, metadata: dict):
    try:
        doc_id = hashlib.md5(f"{question}_{datetime.now().isoformat()}".encode()).hexdigest()
        memory_collection.add(
            ids=[doc_id],
            documents=[f"Q: {question}\nA: {answer}"],
            metadatas=[{**metadata, "timestamp": datetime.now().isoformat(), "type": "field_case"}]
        )
    except Exception as e:
        print(f"Memory save error: {e}")

# ── SYSTEM PROMPT ──
VET_SYSTEM_PROMPT = """You are DAVE — Don't Always Visit the Emergency Vet.
You are a cattle health assistant built for working ranchers in the field by HerdMate.

You are NOT a replacement for a veterinarian. You are a knowledgeable field reference.

Your style:
- Plain language. Direct. Get to the point fast.
- Practical. What do I do RIGHT NOW.
- Honest about uncertainty.

Only recommend calling a vet when it genuinely warrants it:
- EMERGENCY (say it first, loud): not breathing, severe bleeding, prolapse, broken bones, downer cow that can't rise, bloat with distress, difficult calving over 2 hours
- URGENT (mention once at end): fever over 104, eye cloudiness or corneal ulcer, calf not nursing after 6 hours, signs of BRD
- MONITOR (no vet mention needed): mild lameness, early scours with alert calf, minor wounds, routine questions

The disclaimer at the top of the app already covers liability. Do not repeat it in every response.
If you do recommend a vet call, say it once clearly and move on.

When you have an animal record, use it. Reference specific details — tag number, age, dam, birth weight.
Make your answers personal to that specific animal.

You have access to:
1. Veterinary knowledge base — MSD Veterinary Manual and beef cattle extension publications
2. The rancher's personal field history — past cases and outcomes
3. Animal records from HerdMate Google Sheet when a tag is provided"""

# ── MAIN ENDPOINT ──
@app.post("/vet/ask", response_model=VetAnswer)
async def ask_vet(q: VetQuestion):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    effective_user_id = q.google_user_email or q.user_id or "default"

    # Look up animal if tag provided
    animal_record = None
    animal_context = ""
    if q.tag_epc and q.herdmate_sheet_id:
        animal_record = find_animal(q.herdmate_sheet_id, q.tag_epc)
        if animal_record:
            animal_context = format_animal_context(animal_record)

    # RAG search
    vet_results = search_knowledge(q.question)
    past_cases = search_memory(q.question, effective_user_id)

    # Build dynamic system prompt
    dynamic_system = VET_SYSTEM_PROMPT

    if animal_context:
        dynamic_system += "\n\n" + animal_context

    ctx_parts = []
    if q.operation: ctx_parts.append(f"Operation: {q.operation}")
    if q.pasture: ctx_parts.append(f"Pasture: {q.pasture}")
    if q.weather: ctx_parts.append(f"Weather: {q.weather}")
    if q.tag_epc and not animal_record: ctx_parts.append(f"Scanned tag: {q.tag_epc} (no record found)")
    if ctx_parts:
        dynamic_system += "\n\n--- FIELD CONTEXT ---\n" + "\n".join(ctx_parts)

    sources = []
    if vet_results:
        vet_ctx = "\n\n--- VETERINARY KNOWLEDGE ---"
        for doc, meta in vet_results:
            source = meta.get("source", "veterinary reference")
            vet_ctx += f"\n[{source}]\n{doc}\n"
            if source not in sources:
                sources.append(source)
        dynamic_system += vet_ctx

    past_case_summaries = []
    if past_cases:
        mem_ctx = "\n\n--- YOUR PAST FIELD CASES ---"
        for doc, meta in past_cases:
            ts = meta.get("timestamp", "")[:10]
            mem_ctx += f"\n[{ts}] {doc}\n"
            past_case_summaries.append(f"{ts}: {doc[:100]}...")
        dynamic_system += mem_ctx

    # Build conversation
    claude_messages = []
    for msg in q.conversation_history[-8:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            claude_messages.append({"role": role, "content": content})

    if q.image_base64:
        current_content = [
            {"type": "image", "source": {"type": "base64", "media_type": q.image_type or "image/jpeg", "data": q.image_base64}},
            {"type": "text", "text": q.question}
        ]
    else:
        current_content = q.question

    claude_messages.append({"role": "user", "content": current_content})

    try:
        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=dynamic_system,
            messages=claude_messages
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        answer = text_blocks[0].text if text_blocks else "DAVE could not generate a response."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

    save_to_memory(
        question=q.question,
        answer=answer,
        metadata={
            "user_id": effective_user_id,
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
        timestamp=datetime.now().isoformat(),
        animal_context=animal_record
    )

@app.get("/vet/status")
async def vet_status():
    return {
        "status": "online",
        "vet_knowledge_docs": vet_collection.count(),
        "field_memory_docs": memory_collection.count(),
        "ready": vet_collection.count() > 0,
        "service_account": os.path.exists(CREDENTIALS_FILE)
    }

@app.get("/vet/health")
async def health():
    return {"status": "ok", "service": "HerdMate DAVE Vet AI v3"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
