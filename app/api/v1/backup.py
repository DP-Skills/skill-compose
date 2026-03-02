"""
Backup & Restore API - Full system backup and restore.

Uses pg_dump/psql for database operations instead of custom JSON serialization.
Backup scope: PostgreSQL database + disk files (skills/, config/) + .env (optional).
Restore mode: pg_dump --clean drops and recreates all tables automatically.
"""

import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.db.database import AsyncSessionLocal
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backup", tags=["backup"])

# ============ Constants ============

BACKUP_VERSION = "2.0"

# File filtering for skills directory
SKIP_PATTERNS = {"__pycache__", ".pyc", ".backup", "UPDATE_REPORT"}
MAX_FILE_SIZE = 1024 * 1024  # 1MB

# Config files that start with "." but should still be backed up
CONFIG_DOTFILE_WHITELIST = {".env", ".env.custom.keys"}


def _should_skip_file(path: Path, is_config: bool = False) -> bool:
    """Check if a file should be skipped during backup.

    Args:
        is_config: If True, use config-specific rules (whitelist certain dotfiles).
    """
    name = path.name
    path_str = str(path)
    if name.startswith(".") and not (is_config and name in CONFIG_DOTFILE_WHITELIST):
        return True
    for pattern in SKIP_PATTERNS:
        if pattern in path_str or name.endswith(pattern):
            return True
    return False


def _get_backups_dir() -> Path:
    """Get and ensure backups directory exists."""
    backups_dir = Path(settings.backups_dir).resolve()
    backups_dir.mkdir(parents=True, exist_ok=True)
    return backups_dir


def _parse_db_url() -> dict:
    """Parse DATABASE_URL into pg_dump/psql connection params."""
    url = settings.effective_database_url
    # postgresql+asyncpg://user:pass@host:port/dbname
    # Strip any SQLAlchemy driver suffix (e.g. +asyncpg, +psycopg2, +psycopg)
    clean_url = re.sub(r"\+\w+", "", url, count=1)
    parsed = urlparse(clean_url)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "skills",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "skills_api",
    }


def _pg_env(db_params: dict) -> dict:
    """Build environment dict with PGPASSWORD for pg_dump/psql."""
    env = os.environ.copy()
    env["PGPASSWORD"] = db_params["password"]
    # Force English messages so error filtering works regardless of DB locale
    env["LC_MESSAGES"] = "C"
    return env


# ============ Response Models ============


class BackupStats(BaseModel):
    skills: int = 0
    agents: int = 0
    traces: int = 0
    sessions: int = 0
    memory_entries: int = 0


class BackupListItem(BaseModel):
    filename: str
    size_bytes: int
    created_at: str
    backup_version: Optional[str] = None
    stats: Optional[BackupStats] = None


class BackupListResponse(BaseModel):
    backups: List[BackupListItem]
    total: int


class RestoreResponse(BaseModel):
    success: bool
    message: str
    snapshot_filename: Optional[str] = None
    restored: BackupStats = BackupStats()
    errors: List[str]


# ============ Internal Helpers ============


async def _get_db_stats() -> BackupStats:
    """Query DB for table counts."""
    stats = BackupStats()
    async with AsyncSessionLocal() as db:
        for attr, table in [
            ("skills", "skills"),
            ("agents", "agent_presets"),
            ("traces", "agent_traces"),
            ("sessions", "published_sessions"),
            ("memory_entries", "memory_entries"),
        ]:
            try:
                result = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                setattr(stats, attr, result.scalar() or 0)
            except Exception:
                pass
    return stats


def _run_pg_dump(db_params: dict) -> bytes:
    """Run pg_dump and return SQL bytes."""
    cmd = [
        "pg_dump",
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        "--exclude-table=executors",
        "--exclude-table=background_tasks",
        "-h", db_params["host"],
        "-p", db_params["port"],
        "-U", db_params["user"],
        db_params["dbname"],
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        env=_pg_env(db_params),
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"pg_dump failed (exit {result.returncode}): {stderr}")
    return result.stdout


def _run_psql(db_params: dict, sql: bytes) -> str:
    """Run psql with SQL input. Returns stderr output (errors only)."""
    cmd = [
        "psql",
        "-h", db_params["host"],
        "-p", db_params["port"],
        "-U", db_params["user"],
        "-d", db_params["dbname"],
        "--no-psqlrc",
        "--single-transaction",
    ]
    result = subprocess.run(
        cmd,
        input=sql,
        capture_output=True,
        env=_pg_env(db_params),
        timeout=300,
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"psql failed (exit {result.returncode}): {stderr}")
    # Filter: only return actual ERROR lines, ignore NOTICE/WARNING
    error_lines = [
        line for line in stderr.strip().split("\n")
        if line and line.startswith("ERROR:")
    ]
    return "\n".join(error_lines)


async def _create_backup_zip(
    include_env: bool = True,
    filename_prefix: str = "backup",
) -> tuple[io.BytesIO, BackupStats, str]:
    """Create a backup ZIP archive. Returns (zip_buffer, stats, filename)."""

    db_params = _parse_db_url()

    # 1. Get stats before dump
    stats = await _get_db_stats()

    # 2. Run pg_dump (in thread to avoid blocking event loop)
    sql_bytes = await asyncio.to_thread(_run_pg_dump, db_params)

    # 3. Build manifest
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.zip"

    manifest = {
        "backup_version": BACKUP_VERSION,
        "created_at": now.isoformat(),
        "stats": stats.model_dump(),
    }

    # 4. Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr("database.sql", sql_bytes)

        # Config files (with dotfile whitelist for .env.custom.keys etc.)
        config_dir = Path(settings.config_dir).resolve()
        if config_dir.exists():
            for fp in config_dir.rglob("*"):
                if not fp.is_file() or _should_skip_file(fp, is_config=True):
                    continue
                # .env is handled separately via include_env flag
                if fp.name == ".env":
                    continue
                try:
                    rel = fp.relative_to(config_dir)
                    zf.writestr(f"config/{rel}", fp.read_bytes())
                except Exception:
                    pass

        # .env file (optional, controlled by include_env)
        if include_env:
            env_path = Path(settings.config_dir).resolve() / ".env"
            if env_path.exists():
                try:
                    zf.writestr("env/.env", env_path.read_bytes())
                except Exception:
                    pass

        # Skill files from disk
        skills_dir = Path(settings.effective_skills_dir).resolve()
        if skills_dir.exists():
            for fp in skills_dir.rglob("*"):
                if not fp.is_file():
                    continue
                if _should_skip_file(fp):
                    continue
                try:
                    if fp.stat().st_size > MAX_FILE_SIZE:
                        continue
                    rel = fp.relative_to(skills_dir)
                    zf.writestr(f"skills/{rel}", fp.read_bytes())
                except Exception:
                    pass

    zip_buffer.seek(0)
    return zip_buffer, stats, filename


async def _restore_from_zip(zip_bytes: bytes) -> RestoreResponse:
    """Restore system from a backup ZIP. Creates auto-snapshot first."""
    errors: List[str] = []

    # Validate ZIP
    try:
        zip_buffer = io.BytesIO(zip_bytes)
        zf = zipfile.ZipFile(zip_buffer, "r")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")

    with zf:
        file_list = zf.namelist()
        if "manifest.json" not in file_list:
            raise HTTPException(status_code=400, detail="Invalid backup: missing manifest.json")

        # Read manifest
        try:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid manifest.json: {e}")

        backup_version = manifest.get("backup_version", "1.0")
        if backup_version == "1.0":
            raise HTTPException(
                status_code=400,
                detail="v1.0 JSON-based backups are no longer supported. Please create a new backup with v2.0 format.",
            )
        if backup_version != "2.0":
            raise HTTPException(status_code=400, detail=f"Unsupported backup version: {backup_version}")

        if "database.sql" not in file_list:
            raise HTTPException(status_code=400, detail="Invalid backup: missing database.sql")

        # Auto-snapshot before restore
        snapshot_filename = None
        try:
            snap_buf, _, snap_fname = await _create_backup_zip(
                include_env=True, filename_prefix="pre_restore"
            )
            backups_dir = _get_backups_dir()
            snap_path = backups_dir / snap_fname
            snap_path.write_bytes(snap_buf.getvalue())
            snapshot_filename = snap_fname
            logger.info(f"Pre-restore snapshot saved: {snap_fname}")
        except Exception as e:
            errors.append(f"Warning: Failed to create pre-restore snapshot: {e}")
            logger.warning(f"Pre-restore snapshot failed: {e}")

        # Restore database via psql (in thread to avoid blocking event loop)
        db_params = _parse_db_url()
        try:
            sql_bytes = zf.read("database.sql")
            stderr = await asyncio.to_thread(_run_psql, db_params, sql_bytes)
            if stderr:
                for line in stderr.strip().split("\n"):
                    if line:
                        errors.append(f"psql: {line}")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=f"Database restore failed: {e}")

        # Get restored stats
        restored_stats = await _get_db_stats()

        # Clear and restore skills directory
        skills_dir = Path(settings.effective_skills_dir).resolve()
        if skills_dir.exists():
            for child in skills_dir.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except Exception as e:
                    errors.append(f"Failed to clear skill dir '{child.name}': {e}")

        # Clear config directory (except .env which is restored separately)
        config_dir = Path(settings.config_dir).resolve()
        if config_dir.exists():
            for child in config_dir.iterdir():
                if child.name == ".env":
                    continue  # handled by env/.env restore
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except Exception as e:
                    errors.append(f"Failed to clear config '{child.name}': {e}")

        # Extract files from ZIP
        for zip_path in file_list:
            if zip_path.endswith("/"):
                continue

            try:
                content = zf.read(zip_path)
            except Exception:
                continue

            if zip_path.startswith("skills/"):
                rel = zip_path[len("skills/"):]
                if not rel:
                    continue
                out_path = skills_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    out_path.write_bytes(content)
                except Exception as e:
                    errors.append(f"Failed to write {zip_path}: {e}")

            elif zip_path.startswith("config/"):
                rel = zip_path[len("config/"):]
                if not rel:
                    continue
                out_path = config_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    out_path.write_bytes(content)
                except Exception as e:
                    errors.append(f"Failed to write {zip_path}: {e}")

            elif zip_path == "env/.env":
                env_path = config_dir / ".env"
                try:
                    env_path.write_bytes(content)
                except PermissionError:
                    logger.info("Skipped .env restore (read-only mount, typical in Docker)")
                except Exception as e:
                    errors.append(f"Failed to restore .env: {e}")

    return RestoreResponse(
        success=True,
        message=f"Restore completed. Restored {restored_stats.skills} skills, {restored_stats.agents} agents, {restored_stats.traces} traces, {restored_stats.memory_entries} memory entries.",
        snapshot_filename=snapshot_filename,
        restored=restored_stats,
        errors=errors,
    )


# ============ Endpoints ============


@router.post("/create")
async def create_backup(
    include_env: bool = Query(True, description="Include .env file in backup"),
):
    """Create a full system backup.

    Uses pg_dump for database and includes disk files (skills/, config/, .env).
    The backup is saved to the backups directory and returned for download.
    """
    try:
        zip_buffer, stats, filename = await _create_backup_zip(include_env=include_env)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")

    # Save a copy to backups directory
    backups_dir = _get_backups_dir()
    backup_path = backups_dir / filename
    backup_path.write_bytes(zip_buffer.getvalue())
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/list", response_model=BackupListResponse)
async def list_backups():
    """List all available backups from the backups directory."""
    backups_dir = _get_backups_dir()
    items: List[BackupListItem] = []

    for fp in sorted(backups_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat = fp.stat()
            backup_version = None
            stats = None

            try:
                with zipfile.ZipFile(fp, "r") as zf:
                    if "manifest.json" in zf.namelist():
                        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                        backup_version = manifest.get("backup_version")
                        if "stats" in manifest:
                            raw = manifest["stats"]
                            # Handle both v1.0 (8-field) and v2.0 (4-field) stats
                            stats = BackupStats(
                                skills=raw.get("skills", 0),
                                agents=raw.get("agents", raw.get("agent_presets", 0)),
                                traces=raw.get("traces", raw.get("agent_traces", 0)),
                                sessions=raw.get("sessions", raw.get("published_sessions", 0)),
                            )
            except Exception:
                pass

            items.append(BackupListItem(
                filename=fp.name,
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                backup_version=backup_version,
                stats=stats,
            ))
        except Exception:
            pass

    return BackupListResponse(backups=items, total=len(items))


@router.get("/download/{filename}")
async def download_backup(filename: str):
    """Download a backup file from the backups directory."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backups_dir = _get_backups_dir()
    backup_path = backups_dir / filename

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        backup_path.resolve().relative_to(backups_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return StreamingResponse(
        open(backup_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/restore", response_model=RestoreResponse)
async def restore_from_upload(
    file: UploadFile = File(..., description="Backup ZIP file to restore"),
):
    """Restore system from an uploaded backup ZIP file.

    This will:
    1. Create an auto-snapshot of current state
    2. Restore database via psql (pg_dump --clean drops and recreates tables)
    3. Restore disk files (skills/, config/, .env)
    """
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    content = await file.read()
    return await _restore_from_zip(content)


@router.post("/restore/{filename}", response_model=RestoreResponse)
async def restore_from_server(filename: str):
    """Restore system from a backup file stored on the server."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backups_dir = _get_backups_dir()
    backup_path = backups_dir / filename

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        backup_path.resolve().relative_to(backups_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    content = backup_path.read_bytes()
    return await _restore_from_zip(content)
