"""Memory API — bootstrap files and vector-searchable memory entries."""

import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.services import memory_service

router = APIRouter(prefix="/memory", tags=["memory"])

# UUID v4 pattern for scope validation
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

VALID_CATEGORIES = {"fact", "preference", "procedure", "context", "session_summary"}


def _validate_scope(scope: str) -> None:
    """Validate that scope is 'global' or a UUID. Raises HTTPException on invalid input."""
    if scope != "global" and not _UUID_RE.match(scope):
        raise HTTPException(status_code=400, detail="scope must be 'global' or a valid UUID")


def _validate_category(category: Optional[str]) -> Optional[str]:
    """Validate category is in allowed set."""
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
    return category


# ─── Request / Response Models ──────────────────────────────────

class BootstrapFileContent(BaseModel):
    content: str


class MemoryEntryCreate(BaseModel):
    content: str = Field(..., max_length=4096)
    agent_id: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = "manual"

    @field_validator("category")
    @classmethod
    def check_category(cls, v: Optional[str]) -> Optional[str]:
        return _validate_category(v)


class MemoryEntryUpdate(BaseModel):
    content: Optional[str] = Field(None, max_length=4096)
    category: Optional[str] = None

    @field_validator("category")
    @classmethod
    def check_category(cls, v: Optional[str]) -> Optional[str]:
        return _validate_category(v)


class MemorySearchRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


# ─── Bootstrap File Endpoints ───────────────────────────────────

@router.get("/files")
async def list_files(agent_id: Optional[str] = Query(None)):
    """List bootstrap files (global + per-agent)."""
    if agent_id is not None and not _UUID_RE.match(agent_id):
        raise HTTPException(status_code=400, detail="agent_id must be a valid UUID")
    files = memory_service.list_bootstrap_files(agent_id)
    return {"files": files}


@router.get("/files/{scope}/{filename}")
async def read_file(scope: str, filename: str):
    """Read a bootstrap file."""
    _validate_scope(scope)
    content = memory_service.read_bootstrap_file(scope, filename)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    return {"content": content, "scope": scope, "filename": filename}


@router.put("/files/{scope}/{filename}")
async def write_file(scope: str, filename: str, body: BootstrapFileContent):
    """Create or update a bootstrap file."""
    _validate_scope(scope)
    if filename not in memory_service.BOOTSTRAP_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid filename. Must be one of: {memory_service.BOOTSTRAP_FILES}")
    ok = memory_service.write_bootstrap_file(scope, filename, body.content)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write file")
    return {"success": True, "scope": scope, "filename": filename}


@router.delete("/files/{scope}/{filename}")
async def delete_file(scope: str, filename: str):
    """Delete a bootstrap file."""
    _validate_scope(scope)
    ok = memory_service.delete_bootstrap_file(scope, filename)
    if not ok:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True}


# ─── Memory Entry Endpoints ─────────────────────────────────────

@router.get("/entries")
async def list_entries(
    agent_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memory entries with optional filters."""
    if agent_id is not None and not _UUID_RE.match(agent_id):
        raise HTTPException(status_code=400, detail="agent_id must be a valid UUID")
    result = await memory_service.list_entries(
        agent_id=agent_id,
        category=category,
        limit=limit,
        offset=offset,
    )
    return {"entries": result["entries"], "total": result["total"]}


@router.post("/entries")
async def create_entry(body: MemoryEntryCreate):
    """Create a memory entry (auto-embeds)."""
    entry = await memory_service.create_entry(
        content=body.content,
        agent_id=body.agent_id,
        category=body.category,
        source=body.source,
    )
    return entry


@router.put("/entries/{entry_id}")
async def update_entry(entry_id: str, body: MemoryEntryUpdate):
    """Update a memory entry (re-embeds if content changed)."""
    entry = await memory_service.update_entry(
        entry_id=entry_id,
        content=body.content,
        category=body.category,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str):
    """Delete a memory entry."""
    ok = await memory_service.delete_entry(entry_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"success": True}


# ─── Search Endpoint ────────────────────────────────────────────

@router.post("/search")
async def search(body: MemorySearchRequest):
    """Semantic search over memory entries."""
    results = await memory_service.search_memory(
        query=body.query,
        agent_id=body.agent_id,
        top_k=body.top_k,
    )
    return {"results": results, "query": body.query}
