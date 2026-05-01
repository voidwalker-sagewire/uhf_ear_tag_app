#!/usr/bin/env python3
"""
HerdMate DAVE Vet AI — FastAPI Backend with Cow Context
Reads animal records from Google Sheets when a tag EPC is provided.
"""

import os
import json
import hashlib
import requests
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

app = FastAPI(title="HerdMate DAVE Vet AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://scanner.herdmate.ag", "https://api.herdmate.ag", "http://localhost", "*"],
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

# Google Sheet IDs
DCC_SHEET_ID = "1ziqvEJRYmqf4IvYLa4Ij3z4I-swln4nAMJI6gLlzlGI"  # DCC animal records
HERDMATE_SHEET_ID_KEY = "hm_sheet_id"  # stored per user in request

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
    conversation_history: Optional[list] = []
    image_base64: Optional[str] = None
    image_type: Optional[str] = "image/jpeg"
    google_access_token: Optional[str] = None   # passed from browser for Sheets lookup
    herdmate_sheet_id: Optional[str] = None     # user's HerdMate sheet ID

class VetAnswer(BaseModel):
    answer: str
    sources: list
    similar_past_cases: list
    confidence: str
    timestamp: str
    animal_context: Optional[dict] = None      # animal record if found

# ── GOOGLE SHEETS LOOKUP ──
def sheets_get(access_token: str, sheet_id: str, range_name: str):
    """Fetch a range from Google Sheets."""
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/{range_name}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
    if resp.ok:
        return resp.json().get("values", [])
    return []

def find_animal_by_epc(access_token: str, herdmate_sheet_id: str, tag_epc: str):
    """
    Step 1: Look up EPC in HerdMate Animals tab to find visual Tag #
    Step 2: Look up Tag # in DCC sheet across Calf Tracker and Ranch Tracker
    Returns a dict with animal context or None
    """
    if not access_token or not tag_epc:
        return None

    try:
        # Step 1: HerdMate Animals tab — EPC → Tag #
        animal_tag = None
        if herdmate_sheet_id:
            animals_data = sheets_get(access_token, herdmate_sheet_id, "Animals!A:I")
            if animals_data and len(animals_data) > 1:
                headers = animals_data[0]
                for row in animals_data[1:]:
                    if row and len(row) > 0 and str(row[0]).strip() == tag_epc.strip():
                        # Found EPC match
                        row_dict = dict(zip(headers, row))
                        animal_tag = str(row_dict.get("Animal ID", "")).strip()
                        break

        # Also check if tag_epc looks like a visual tag number directly
        if not animal_tag:
            # Try treating EPC as animal tag directly for backwards compat
            animal_tag = tag_epc

        if not animal_tag:
            return None

        # Step 2: Search Calf Tracker in DCC sheet by Calf Tag or UHF#
        calf_data = sheets_get(access_token, DCC_SHEET_ID, "Calf Tracker!A:AX")
        if calf_data and len(calf_data) > 1:
            ct_headers = calf_data[0]
            for row in calf_data[1:]:
                if not row:
                    continue
                row_dict = dict(zip(ct_headers, row + [""] * (len(ct_headers) - len(row))))
                calf_tag = str(row_dict.get("Calf Tag", "")).strip()
                uhf = str(row_dict.get("UHF#", "")).strip()
                if calf_tag == animal_tag or uhf == tag_epc:
                    return {
                        "source": "Calf Tracker",
                        "tag": calf_tag,
                        "epc": uhf or tag_epc,
                        "sex": row_dict.get("Calf Sex", ""),
                        "color": row_dict.get("Calf Color", ""),
                        "type": row_dict.get("Calf Type", ""),
                        "birth_weight": row_dict.get("Birth Weight", ""),
                        "dam_tag": row_dict.get("Cow Tag", ""),
                        "season": row_dict.get("Calving Season", ""),
                        "status": row_dict.get("Status", ""),
                        "herd": row_dict.get("Herd", ""),
                        "notes": row_dict.get("Calving Notes", ""),
                        "date": str(row_dict.get("Date", ""))[:10],
                        "is_twin": row_dict.get("Is Twin", ""),
                        "dam_bcs": row_dict.get("Dam BCS", ""),
                        "assisted": row_dict.get("Assisted Y/N", ""),
                        "sire": row_dict.get("Sire", ""),
                    }

        # Step 3: Search Ranch Tracker by Tag #
        ranch_data = sheets_get(access_token, DCC_SHEET_ID, "Ranch Tracker!A:CJ")
        if ranch_data and len(ranch_data) > 1:
            rt_headers = ranch_data[0]
            for row in ranch_data[1:]:
                if not row:
                    continue
                row_dict = dict(zip(rt_headers, row + [""] * (len(rt_headers) - len(row))))
                tag_num = str(row_dict.get("Tag #", "")).strip()
                uhf = str(row_dict.get("UHF#", "")).strip()
                if tag_num == animal_tag or uhf == tag_epc:
                    return {
                        "source": "Ranch Tracker",
                        "tag": tag_num,
                        "epc": uhf or tag_epc,
                        "display_id": row_dict.get("DisplayID", ""),
                        "sex": row_dict.get("Sex", ""),
                        "breed": row_dict.get("Breed", ""),
                        "type": row_dict.get("Type", ""),
                        "color": row_dict.get("Color", ""),
                        "birth_date": str(row_dict.get("Birth Date", ""))[:10],
                        "age": row_dict.get("Age", ""),
                        "pasture": row_dict.get("Pasture", ""),
                        "herd": row_dict.get("Herd", ""),
                        "status": row_dict.get("Status", ""),
                        "weight": row_dict.get("Weight (lbs)", ""),
                        "birth_weight": row_dict.get("Birth Weight (lbs)", ""),
                        "dam": row_dict.get("Dam #", ""),
                        "sire": row_dict.get("Sire #", ""),
                        "due_date": str(row_dict.get("Due Date", ""))[:10],
                        "palp_result": row_dict.get("Palp. Result", ""),
                        "months_preg": row_dict.get("Mth. Preg.", ""),
                        "cull": row_dict.get("Cull Y/N", ""),
                        "bcs": row_dict.get("Body Condition Score (BCS)", ""),
                        "disposition": row_dict.get("Disposition", ""),
                        "notes": row_dict.get("Notes", ""),
                        "calving_2023": row_dict.get("2023\nCalving Date", ""),
                        "calving_2024": row_dict.get("2024\nCalving Date", ""),
                        "calving_2025": row_dict.get("2025\nCalving Date", ""),
                    }

    except Exception as e:
        print(f"Animal lookup error: {e}")

    return None

def format_animal_context(animal: dict) -> str:
    """Format animal record into readable context for DAVE."""
    if not animal:
        return ""

    lines = [f"--- ANIMAL RECORD (from {animal.get('source', 'HerdMate')}) ---"]
    lines.append(f"Tag #: {animal.get('tag', 'Unknown')}")

    if animal.get('epc'):
        lines.append(f"UHF EPC: {animal['epc']}")
    if animal.get('sex'):
        lines.append(f"Sex: {animal['sex']}")
    if animal.get('breed'):
        lines.append(f"Breed: {animal['breed']}")
    if animal.get('color'):
        lines.append(f"Color: {animal['color']}")
    if animal.get('type'):
        lines.append(f"Type: {animal['type']}")
    if animal.get('birth_date'):
        lines.append(f"Born: {animal['birth_date']}")
    if animal.get('age'):
        lines.append(f"Age: {animal['age']} years")
    if animal.get('herd'):
        lines.append(f"Herd: {animal['herd']}")
    if animal.get('pasture'):
        lines.append(f"Pasture: {animal['pasture']}")
    if animal.get('status'):
        lines.append(f"Status: {animal['status']}")
    if animal.get('weight'):
        lines.append(f"Current Weight: {animal['weight']} lbs")
    if animal.get('birth_weight'):
        lines.append(f"Birth Weight: {animal['birth_weight']} lbs")
    if animal.get('dam') or animal.get('dam_tag'):
        lines.append(f"Dam Tag: {animal.get('dam') or animal.get('dam_tag')}")
    if animal.get('sire'):
        lines.append(f"Sire Tag: {animal['sire']}")
    if animal.get('season'):
        lines.append(f"Season: {animal['season']}")
    if animal.get('due_date') and str(animal['due_date']) not in ['', 'None', '1900-01-00']:
        lines.append(f"Due Date: {animal['due_date']}")
    if animal.get('palp_result'):
        lines.append(f"Palpation Result: {animal['palp_result']}")
    if animal.get('months_preg'):
        lines.append(f"Months Pregnant: {animal['months_preg']}")
    if animal.get('bcs'):
        lines.append(f"Body Condition Score: {animal['bcs']}")
    if animal.get('cull') and str(animal['cull']).lower() in ['yes', 'y', 'true']:
        lines.append(f"⚠️ MARKED FOR CULL")
    if animal.get('disposition'):
        lines.append(f"Disposition: {animal['disposition']}")
    if animal.get('assisted'):
        lines.append(f"Birth Assistance: {animal['assisted']}")
    if animal.get('dam_bcs'):
        lines.append(f"Dam BCS at birth: {animal['dam_bcs']}")
    if animal.get('is_twin') and str(animal['is_twin']).lower() in ['yes', 'y', 'true']:
        lines.append(f"Twin: Yes")

    # Calving history
    calving = []
    for yr in ['calving_2023', 'calving_2024', 'calving_2025']:
        if animal.get(yr) and str(animal[yr]) not in ['', 'None']:
            calving.append(f"{yr[-4:]}: {animal[yr]}")
    if calving:
        lines.append(f"Calving History: {', '.join(calving)}")

    if animal.get('notes') and str(animal['notes']) not in ['', 'None']:
        lines.append(f"Notes: {animal['notes']}")

    return "\n".join(lines)

# ── RAG ──
def search_knowledge(question: str, n_results: int = 5):
    try:
        results = vet_collection.query(
            query_texts=[question],
            n_results=min(n_results, vet_collection.count() or 1)
        )
        return list(zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]))
    except:
        return []

def search_memory(question: str, user_id: str, n_results: int = 3):
    try:
        if memory_collection.count() == 0:
            return []
        results = memory_collection.query(
            query_texts=[question],
            n_results=min(n_results, memory_collection.count()),
            where={"user_id": user_id} if user_id != "default" else None
        )
        return list(zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]))
    except:
        return []

def save_to_memory(question: str, answer: str, metadata: dict):
    try:
        doc_id = hashlib.md5(f"{question}_{datetime.now().isoformat()}".encode()).hexdigest()
        memory_collection.add(
            ids=[doc_id],
            documents=[f"Q: {question}\nA: {answer}"],
            metadatas=[{**metadata, "timestamp": datetime.now().isoformat(), "type": "field_case"}]
        )
    except:
        pass

# ── SYSTEM PROMPT ──
VET_SYSTEM_PROMPT = """You are DAVE — Don't Always Visit the Emergency Vet.
You are a cattle health assistant built for working ranchers in the field by HerdMate.

You are NOT a replacement for a veterinarian. You are a knowledgeable field reference.

Your style:
- Plain language. Direct. Get to the point fast.
- Practical. What do I do RIGHT NOW.
- Honest about uncertainty.

Only recommend calling a vet when it genuinely warrants it. Use this scale:
- EMERGENCY (say it first, loud): not breathing, severe bleeding, prolapse, broken bones, downer cow that can't rise, bloat with distress, difficult calving over 2 hours
- URGENT (mention once at end): fever over 104, eye cloudiness or corneal ulcer, calf not nursing after 6 hours, signs of BRD
- MONITOR (no vet mention needed): mild lameness, early scours with alert calf, minor wounds, routine questions

The disclaimer at the top of the app already covers liability. Do not repeat it in every response.
If you do recommend a vet call, say it once clearly and move on.

When you have an animal record, use it. Reference specific details — "This cow is due July 15" or "Her BCS was 5 at last check." Make it personal to that animal.

You have access to:
1. Veterinary knowledge base — MSD Veterinary Manual and beef cattle extension publications
2. The rancher's personal field history — past cases and outcomes
3. Animal records from HerdMate when a tag is scanned"""

# ── MAIN ENDPOINT ──
@app.post("/vet/ask", response_model=VetAnswer)
async def ask_vet(q: VetQuestion):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Look up animal record if tag EPC provided
    animal_context = None
    animal_record = None
    if q.tag_epc and q.google_access_token:
        animal_record = find_animal_by_epc(
            q.google_access_token,
            q.herdmate_sheet_id or "",
            q.tag_epc
        )
        if animal_record:
            animal_context = format_animal_context(animal_record)

    # RAG search
    vet_results = search_knowledge(q.question)
    past_cases = search_memory(q.question, q.user_id)

    # Build dynamic system prompt
    dynamic_system = VET_SYSTEM_PROMPT

    if animal_context:
        dynamic_system = dynamic_system + "\n\n" + animal_context

    if q.pasture or q.weather or q.tag_epc or q.operation:
        ctx = "\n\n--- CURRENT FIELD CONTEXT ---"
        if q.operation:
            ctx += f"\nOperation: {q.operation}"
        if q.pasture:
            ctx += f"\nPasture: {q.pasture}"
        if q.weather:
            ctx += f"\nWeather: {q.weather}"
        if q.tag_epc and not animal_record:
            ctx += f"\nScanned tag EPC: {q.tag_epc} (no matching animal record found)"
        dynamic_system = dynamic_system + ctx

    sources = []
    if vet_results:
        vet_ctx = "\n\n--- VETERINARY KNOWLEDGE BASE ---"
        for doc, meta in vet_results:
            source = meta.get("source", "veterinary reference")
            vet_ctx += f"\n[{source}]\n{doc}\n"
            if source not in sources:
                sources.append(source)
        dynamic_system = dynamic_system + vet_ctx

    past_case_summaries = []
    if past_cases:
        mem_ctx = "\n\n--- YOUR PAST FIELD CASES ---"
        for doc, meta in past_cases:
            ts = meta.get("timestamp", "")[:10]
            mem_ctx += f"\n[{ts}] {doc}\n"
            past_case_summaries.append(f"{ts}: {doc[:100]}...")
        dynamic_system = dynamic_system + mem_ctx

    # Build conversation messages
    claude_messages = []
    if q.conversation_history:
        for msg in q.conversation_history[-8:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                claude_messages.append({"role": role, "content": content})

    # Current message — with optional image
    if q.image_base64:
        current_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": q.image_type or "image/jpeg",
                    "data": q.image_base64
                }
            },
            {"type": "text", "text": q.question}
        ]
    else:
        current_content = q.question

    claude_messages.append({"role": "user", "content": current_content})

    # Call Claude
    try:
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            system=dynamic_system,
            messages=claude_messages
        )
        answer = response.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

    # Save to memory
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
        timestamp=datetime.now().isoformat(),
        animal_context=animal_record
    )

@app.get("/vet/status")
async def vet_status():
    return {
        "status": "online",
        "vet_knowledge_docs": vet_collection.count(),
        "field_memory_docs": memory_collection.count(),
        "ready": vet_collection.count() > 0
    }

@app.get("/vet/health")
async def health():
    return {"status": "ok", "service": "HerdMate DAVE Vet AI v2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
