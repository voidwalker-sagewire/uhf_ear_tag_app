#!/usr/bin/env python3
"""
HerdMate Sentinel Animal Lookup API.

This file extends the existing DAVE FastAPI application without replacing
DAVE's veterinary endpoints. Put it beside herdmate_vet_api.py and start this
file instead of starting herdmate_vet_api.py directly.

Run:
    python3 sentinel_animal_api.py

Existing environment variables used by herdmate_vet_api.py still apply,
including ANTHROPIC_API_KEY, CREDENTIALS_FILE, CHROMA_HOST, CHROMA_PORT,
and PORT.
"""

from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel

from herdmate_vet_api import SERVER_PORT, app, find_animal


class AnimalLookupRequest(BaseModel):
    epc: str
    herdmate_sheet_id: str
    operation: Optional[str] = "HerdMate"


@app.post("/animal/lookup")
async def animal_lookup(request: AnimalLookupRequest):
    epc = request.epc.strip()
    sheet_id = request.herdmate_sheet_id.strip()

    if not epc:
        raise HTTPException(status_code=400, detail="EPC is required")
    if not sheet_id:
        raise HTTPException(status_code=400, detail="herdmate_sheet_id is required")

    try:
        animal = find_animal(sheet_id, epc)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Animal lookup failed: {exc}",
        ) from exc

    now = datetime.now(timezone.utc).isoformat()

    if not animal:
        return {
            "found": False,
            "epc": epc,
            "operation": request.operation or "HerdMate",
            "animal": None,
            "ambiguous": False,
            "other_matches": [],
            "timestamp": now,
        }

    return {
        "found": True,
        "epc": epc,
        "operation": request.operation or "HerdMate",
        "animal": animal,
        "ambiguous": bool(animal.get("_ambiguous")),
        "other_matches": animal.get("_other_matches", []),
        "timestamp": now,
    }


@app.get("/animal/health")
async def animal_health():
    return {
        "status": "ok",
        "service": "HerdMate Sentinel Animal Lookup",
        "dave_routes_preserved": True,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
