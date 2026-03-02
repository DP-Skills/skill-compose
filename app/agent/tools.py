"""
Agent Tools - Tools that the agent can call

Each tool is defined as a dict with:
- name: Tool name for Claude
- description: What the tool does
- input_schema: JSON schema for parameters
- function: The actual function to execute
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import fnmatch
import time
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

from sqlalchemy import select

from app.core.skill_manager import generate_skills_xml
from app.tools.code_executor import AgentWorkspace
from app.services.executor_client import ExecutorClient
from app.config import get_settings
from app.tools.mcp_client import (
    call_mcp_tool,
    list_mcp_servers,
    get_mcp_client,
    discover_mcp_tools,
    get_default_enabled_mcp_servers,
)
from app.db.database import SyncSessionLocal
from app.db.models import SkillDB, SkillVersionDB

# Get settings (loads from .env automatically via pydantic_settings)
_settings = get_settings()
# Skills directory — used as default path for glob/grep and in tool descriptions
_SKILLS_DIR = str(Path(_settings.skills_dir or _settings.custom_skills_dir).resolve())
# Persistent directory for output files (survives workspace cleanup)
_PERSIST_DIR = Path(_settings.upload_dir)

logger = logging.getLogger(__name__)


def _write_via_subprocess(filepath: Path, content: str) -> None:
    """Write file via subprocess for Docker overlay2 filesystem consistency.

    In Docker overlay2, files written by a parent process (API uvicorn worker)
    may not be immediately visible to child subprocesses (bash, execute_code).
    By writing via subprocess, the file is created in the same FS layer that
    other subprocesses see, eliminating the visibility inconsistency.

    Content is passed via stdin pipe to avoid shell quoting issues.
    """
    subprocess.run(
        ['mkdir', '-p', str(filepath.parent)],
        check=True, capture_output=True,
    )
    proc = subprocess.run(
        ['sh', '-c', 'cat > "$1"', 'sh', str(filepath)],
        input=content.encode('utf-8'),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise IOError(
            f"Subprocess write failed (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')}"
        )


# ============ Registry Database Helpers ============
# Direct sync database queries — safe to call from any thread (agent runs in thread pool)


def _fetch_skills_from_registry() -> List[Dict[str, Any]]:
    """Fetch all skills from the registry database (sync)."""
    try:
        with SyncSessionLocal() as session:
            result = session.execute(
                select(SkillDB).order_by(SkillDB.created_at.desc()).limit(100)
            )
            skills = result.scalars().all()
            return [
                {
                    "name": skill.name,
                    "description": skill.description or "",
                    "current_version": skill.current_version,
                }
                for skill in skills
            ]
    except Exception as e:
        print(f"Warning: Failed to fetch skills from registry: {e}")
        return []


def _fetch_skill_content_from_registry(skill_name: str) -> Optional[Dict[str, Any]]:
    """Fetch skill content (SKILL.md) from the registry database (sync)."""
    try:
        with SyncSessionLocal() as session:
            # Get skill
            result = session.execute(
                select(SkillDB).where(SkillDB.name == skill_name)
            )
            skill = result.scalar_one_or_none()
            if not skill or not skill.current_version:
                return None

            # Get version content
            version_result = session.execute(
                select(SkillVersionDB).where(
                    SkillVersionDB.skill_id == skill.id,
                    SkillVersionDB.version == skill.current_version
                )
            )
            version = version_result.scalar_one_or_none()
            if not version:
                return None

            return {
                "name": skill_name,
                "description": skill.description or "",
                "content": version.skill_md or "",
                "version": skill.current_version,
            }
    except Exception as e:
        print(f"Warning: Failed to fetch skill '{skill_name}' from registry: {e}")
        return None


# ============ Tool Functions ============

def list_skills(allowed_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """List all available skills from the registry database, optionally filtered by allowed_skills."""
    # Fetch skills from registry database
    registry_skills = _fetch_skills_from_registry()

    # Filter if allowed_skills is specified
    if allowed_skills is not None:
        registry_skills = [s for s in registry_skills if s.get("name") in allowed_skills]

    return {
        "skills": [{"name": s.get("name"), "description": s.get("description", "")} for s in registry_skills],
        "count": len(registry_skills),
    }


def list_registry_skills() -> Dict[str, Any]:
    """List ALL registered skills in the system (not filtered by equipped skills)."""
    registry_skills = _fetch_skills_from_registry()
    return {
        "skills": [{"name": s.get("name"), "description": s.get("description", "")} for s in registry_skills],
        "count": len(registry_skills),
    }


def get_skill(skill_name: str) -> Dict[str, Any]:
    """Get the content of a specific skill from the registry database."""
    # Fetch from database (single source of truth)
    registry_skill = _fetch_skill_content_from_registry(skill_name)
    if not registry_skill:
        return {"error": f"Skill '{skill_name}' not found"}

    return {
        "name": registry_skill["name"],
        "content": registry_skill["content"],
        "version": registry_skill.get("version"),
        "resources": {
            "scripts": [],
            "references": [],
            "assets": [],
        },
    }


def execute_code(code: str, workspace: Optional[AgentWorkspace] = None) -> Dict[str, Any]:
    """Execute Python code in the workspace."""
    if workspace is None:
        return {"success": False, "output": "", "error": "No workspace available"}

    start = time.time()
    result = workspace.execute(code)
    elapsed = round(time.time() - start, 2)
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "duration_seconds": elapsed,
    }


def bash(command: str, workspace: Optional[AgentWorkspace] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Execute a shell command in the workspace."""
    if workspace is None:
        return {"success": False, "output": "", "error": "No workspace available"}

    start = time.time()
    result = workspace.execute_command(command, timeout=timeout)
    elapsed = round(time.time() - start, 2)
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "duration_seconds": elapsed,
    }


# ============ Code Exploration Tools ============
# These tools help the agent explore skill source code (similar to Claude Code's tools)

# Constants for code exploration tools
GLOB_LIMIT = 100
GREP_LIMIT = 100
READ_DEFAULT_LIMIT = 2000
MAX_LINE_LENGTH = 3000
MAX_BYTES = 50 * 1024

# Binary file extensions that should not be read
BINARY_EXTENSIONS = {
    '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.class', '.jar', '.war',
    '.7z', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods',
    '.odp', '.bin', '.dat', '.obj', '.o', '.a', '.lib', '.wasm', '.pyc', '.pyo',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac', '.pdf'
}


def _is_binary_file(filepath: Path) -> bool:
    """Check if a file is binary based on extension and content."""
    # Check extension first
    if filepath.suffix.lower() in BINARY_EXTENSIONS:
        return True

    # Sample file content for binary detection
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(4096)
            if not chunk:
                return False
            # Check for null bytes (strong indicator of binary)
            if b'\x00' in chunk:
                return True
            # Check for high ratio of non-printable characters
            non_printable = sum(1 for b in chunk if b < 9 or (13 < b < 32))
            return non_printable / len(chunk) > 0.3
    except Exception:
        return True


def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available."""
    try:
        subprocess.run(['rg', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def glob(pattern: str, path: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for files matching a glob pattern.
    Similar to Claude Code's Glob tool.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "*.md")
        path: Directory to search in (defaults to skills directory)
    """
    # Determine search path
    if path:
        search_path = Path(path)
    else:
        # Default to skills directory
        search_path = Path(_SKILLS_DIR)

    if not search_path.exists():
        return {"error": f"Path not found: {search_path}", "files": [], "count": 0}

    # Collect matching files
    files = []
    truncated = False

    try:
        # Use rglob for recursive patterns, glob for simple ones
        if '**' in pattern:
            matches = search_path.rglob(pattern.replace('**/', '').replace('**', '*'))
        else:
            matches = search_path.rglob(pattern)

        for filepath in matches:
            if filepath.is_file():
                if len(files) >= GLOB_LIMIT:
                    truncated = True
                    break
                try:
                    mtime = filepath.stat().st_mtime
                    files.append({
                        "path": str(filepath),
                        "mtime": mtime
                    })
                except OSError:
                    continue
    except Exception as e:
        return {"error": f"Glob search failed: {str(e)}", "files": [], "count": 0}

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x["mtime"], reverse=True)

    # Format output
    output_lines = []
    if not files:
        output_lines.append("No files found")
    else:
        output_lines.extend([f["path"] for f in files])
        if truncated:
            output_lines.append("")
            output_lines.append(f"(Results truncated at {GLOB_LIMIT}. Consider using a more specific pattern.)")

    return {
        "files": [f["path"] for f in files],
        "count": len(files),
        "truncated": truncated,
        "output": "\n".join(output_lines)
    }


def grep(pattern: str, path: Optional[str] = None, include: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for content in files using regex pattern.
    Similar to Claude Code's Grep tool. Uses ripgrep if available, falls back to Python.

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in (defaults to skills directory)
        include: File pattern to include (e.g., "*.py", "*.ts")
    """
    # Determine search path
    if path:
        search_path = Path(path)
    else:
        search_path = Path(_SKILLS_DIR)

    if not search_path.exists():
        return {"error": f"Path not found: {search_path}", "matches": 0, "output": ""}

    # Try ripgrep first (much faster for large codebases)
    if _check_ripgrep_available():
        return _grep_with_ripgrep(pattern, search_path, include)
    else:
        return _grep_with_python(pattern, search_path, include)


def _grep_with_ripgrep(pattern: str, search_path: Path, include: Optional[str]) -> Dict[str, Any]:
    """Use ripgrep for fast searching."""
    args = ['rg', '-nH', '--hidden', '--follow', '--field-match-separator=|', '--regexp', pattern]

    if include:
        args.extend(['--glob', include])

    args.append(str(search_path))

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "matches": 0, "output": ""}
    except Exception as e:
        return {"error": f"ripgrep failed: {str(e)}", "matches": 0, "output": ""}

    if result.returncode == 1:  # No matches
        return {"matches": 0, "truncated": False, "output": "No matches found"}

    if result.returncode != 0:
        return {"error": f"ripgrep error: {result.stderr}", "matches": 0, "output": ""}

    # Parse results
    matches = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|', 2)
        if len(parts) >= 3:
            filepath, line_num_str, line_text = parts[0], parts[1], parts[2]
            try:
                line_num = int(line_num_str)
                mtime = Path(filepath).stat().st_mtime
                matches.append({
                    "path": filepath,
                    "line_num": line_num,
                    "line_text": line_text,
                    "mtime": mtime
                })
            except (ValueError, OSError):
                continue

    # Sort by modification time and limit
    matches.sort(key=lambda x: x["mtime"], reverse=True)
    truncated = len(matches) > GREP_LIMIT
    matches = matches[:GREP_LIMIT]

    # Format output
    output_lines = [f"Found {len(matches)} matches"]
    current_file = ""
    for match in matches:
        if current_file != match["path"]:
            if current_file:
                output_lines.append("")
            current_file = match["path"]
            output_lines.append(f"{match['path']}:")

        line_text = match["line_text"]
        if len(line_text) > MAX_LINE_LENGTH:
            line_text = line_text[:MAX_LINE_LENGTH] + "..."
        output_lines.append(f"  Line {match['line_num']}: {line_text}")

    if truncated:
        output_lines.append("")
        output_lines.append(f"(Results truncated at {GREP_LIMIT}. Consider using a more specific pattern.)")

    return {
        "matches": len(matches),
        "truncated": truncated,
        "output": "\n".join(output_lines)
    }


def _grep_with_python(pattern: str, search_path: Path, include: Optional[str]) -> Dict[str, Any]:
    """Fallback Python implementation for grep."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex pattern: {str(e)}", "matches": 0, "output": ""}

    matches = []

    # Determine file pattern
    file_pattern = include if include else "*"

    for filepath in search_path.rglob(file_pattern):
        if not filepath.is_file() or _is_binary_file(filepath):
            continue

        if len(matches) >= GREP_LIMIT:
            break

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({
                            "path": str(filepath),
                            "line_num": line_num,
                            "line_text": line.rstrip('\n\r'),
                            "mtime": filepath.stat().st_mtime
                        })
                        if len(matches) >= GREP_LIMIT:
                            break
        except Exception:
            continue

    # Sort by modification time
    matches.sort(key=lambda x: x["mtime"], reverse=True)
    truncated = len(matches) >= GREP_LIMIT

    # Format output
    if not matches:
        return {"matches": 0, "truncated": False, "output": "No matches found"}

    output_lines = [f"Found {len(matches)} matches"]
    current_file = ""
    for match in matches:
        if current_file != match["path"]:
            if current_file:
                output_lines.append("")
            current_file = match["path"]
            output_lines.append(f"{match['path']}:")

        line_text = match["line_text"]
        if len(line_text) > MAX_LINE_LENGTH:
            line_text = line_text[:MAX_LINE_LENGTH] + "..."
        output_lines.append(f"  Line {match['line_num']}: {line_text}")

    if truncated:
        output_lines.append("")
        output_lines.append(f"(Results truncated at {GREP_LIMIT}. Consider using a more specific pattern.)")

    return {
        "matches": len(matches),
        "truncated": truncated,
        "output": "\n".join(output_lines)
    }


def write(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file. Creates the file if it doesn't exist, overwrites if it does.
    Similar to Claude Code's Write tool.

    Args:
        file_path: Path to the file to write (relative to working directory or absolute)
        content: Content to write to the file
    """
    filepath = Path(file_path)

    # Security: prevent writing to sensitive locations
    sensitive_patterns = ['.env', 'credentials', 'secrets', '.git/']
    filepath_str = str(filepath).lower()
    for pattern in sensitive_patterns:
        if pattern in filepath_str:
            return {"error": f"Cannot write to sensitive location: {file_path}"}

    try:
        _write_via_subprocess(filepath, content)

        expected_bytes = len(content.encode('utf-8'))
        return {
            "success": True,
            "path": str(filepath),
            "bytes_written": expected_bytes,
            "message": f"Successfully wrote {expected_bytes} bytes to {filepath}"
        }
    except Exception as e:
        logger.error("Failed to write file %s: %s", filepath, e)
        return {"error": f"Failed to write file: {str(e)}"}


# Unicode normalization table: each replacement is 1:1 character mapping
# so string positions are preserved after normalization.
_UNICODE_CHAR_MAP = str.maketrans({
    # Smart/curly quotes → ASCII
    '\u2018': "'",   # '
    '\u2019': "'",   # '
    '\u201C': '"',   # "
    '\u201D': '"',   # "
    '\u2032': "'",   # ′ (prime)
    '\u2033': '"',   # ″ (double prime)
    # Dashes → ASCII hyphen
    '\u2014': '-',   # — (em dash)
    '\u2013': '-',   # – (en dash)
    '\u2015': '-',   # ― (horizontal bar)
    # Special spaces → regular space
    '\u00A0': ' ',   # NBSP
    '\u3000': ' ',   # fullwidth space
    '\u202F': ' ',   # narrow no-break space
    '\u2009': ' ',   # thin space
})


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to ASCII equivalents (1:1 char mapping)."""
    return text.translate(_UNICODE_CHAR_MAP)


def edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> Dict[str, Any]:
    """
    Edit a file by replacing exact string matches.
    Similar to Claude Code's Edit tool.

    Args:
        file_path: Path to the file to edit
        old_string: The exact string to find and replace
        new_string: The string to replace it with
        replace_all: If True, replace all occurrences; if False, old_string must be unique

    Returns:
        Success/error status with details
    """
    filepath = Path(file_path)

    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}

    if not filepath.is_file():
        return {"error": f"Not a file: {filepath}"}

    # Check for binary file
    if _is_binary_file(filepath):
        return {"error": f"Cannot edit binary file: {filepath}"}

    # Security: prevent editing sensitive files
    sensitive_patterns = ['.env', 'credentials', 'secrets']
    filepath_str = str(filepath).lower()
    for pattern in sensitive_patterns:
        if pattern in filepath_str:
            return {"error": f"Cannot edit sensitive file: {file_path}"}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    # Check if old_string exists (exact match first)
    fuzzy_matched = False
    if old_string not in content:
        # Try fuzzy match with Unicode normalization (1:1 char mapping preserves positions)
        norm_content = _normalize_unicode(content)
        norm_old = _normalize_unicode(old_string)
        if norm_old in norm_content:
            norm_count = norm_content.count(norm_old)
            if not replace_all and norm_count > 1:
                return {
                    "error": f"The old_string (after Unicode normalization) appears {norm_count} times in the file. Either provide a larger unique string with more context, or set replace_all=true to replace all occurrences.",
                    "occurrences": norm_count
                }
            # Since all replacements are 1:1 char mapping, positions map directly
            # Find position(s) in normalized space and extract original text
            fuzzy_matched = True
        else:
            return {
                "error": f"String not found in file. The old_string must match exactly (including whitespace and indentation).",
                "hint": "Use read_file first to see the exact content, then copy the exact string to replace."
            }

    if fuzzy_matched:
        # Perform replacement using normalized position mapping
        norm_content = _normalize_unicode(content)
        norm_old = _normalize_unicode(old_string)
        new_content = content
        replacements = 0
        offset = 0
        while True:
            idx = _normalize_unicode(new_content).find(norm_old, offset)
            if idx == -1:
                break
            # Extract the original text at this position and replace it
            original_fragment = new_content[idx:idx + len(old_string)]
            new_content = new_content[:idx] + new_string + new_content[idx + len(original_fragment):]
            replacements += 1
            offset = idx + len(new_string)
            if not replace_all:
                break
    else:
        # Check uniqueness if not replace_all
        count = content.count(old_string)
        if not replace_all and count > 1:
            return {
                "error": f"The old_string appears {count} times in the file. Either provide a larger unique string with more context, or set replace_all=true to replace all occurrences.",
                "occurrences": count
            }

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

    # Check if anything changed
    if new_content == content:
        return {"error": "No changes made - old_string equals new_string"}

    try:
        _write_via_subprocess(filepath, new_content)
    except Exception as e:
        logger.error("Failed to write edited file %s: %s", filepath, e)
        return {"error": f"Failed to write file: {str(e)}"}

    result = {
        "success": True,
        "path": str(filepath),
        "replacements": replacements,
        "message": f"Successfully replaced {replacements} occurrence(s) in {filepath}"
    }
    if fuzzy_matched:
        result["message"] += " (matched via Unicode normalization: smart quotes, dashes, or special spaces were treated as ASCII equivalents)"
    return result


def read(file_path: str, offset: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Read file contents with line numbers.
    Similar to Claude Code's Read tool.

    Args:
        file_path: Path to the file to read
        offset: Line number to start reading from (0-based)
        limit: Number of lines to read (defaults to 2000)
    """
    filepath = Path(file_path)

    if not filepath.exists():
        # Try to suggest similar files
        parent = filepath.parent
        if parent.exists():
            base = filepath.name.lower()
            suggestions = [
                str(parent / entry.name)
                for entry in parent.iterdir()
                if base in entry.name.lower() or entry.name.lower() in base
            ][:3]
            if suggestions:
                return {
                    "error": f"File not found: {filepath}\n\nDid you mean one of these?\n" + "\n".join(suggestions)
                }
        return {"error": f"File not found: {filepath}"}

    if not filepath.is_file():
        return {"error": f"Not a file: {filepath}"}

    # Check for binary file
    if _is_binary_file(filepath):
        return {"error": f"Cannot read binary file: {filepath}"}

    # Read file
    limit = limit or READ_DEFAULT_LIMIT

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    total_lines = len(all_lines)

    # Apply offset and limit with byte size check
    raw_lines = []
    bytes_count = 0
    truncated_by_bytes = False

    for i in range(offset, min(total_lines, offset + limit)):
        line = all_lines[i].rstrip('\n\r')
        # Truncate long lines
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "..."

        line_bytes = len(line.encode('utf-8')) + 1  # +1 for newline
        if bytes_count + line_bytes > MAX_BYTES:
            truncated_by_bytes = True
            break

        raw_lines.append(line)
        bytes_count += line_bytes

    # Format output with line numbers (1-based, like cat -n)
    content_lines = [
        f"{str(i + offset + 1).zfill(5)}| {line}"
        for i, line in enumerate(raw_lines)
    ]

    last_read_line = offset + len(raw_lines)
    has_more_lines = total_lines > last_read_line
    truncated = has_more_lines or truncated_by_bytes

    output = "<file>\n"
    output += "\n".join(content_lines)

    if truncated_by_bytes:
        output += f"\n\n(Output truncated at {MAX_BYTES} bytes. Use 'offset' parameter to read beyond line {last_read_line})"
    elif has_more_lines:
        output += f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {last_read_line})"
    else:
        output += f"\n\n(End of file - total {total_lines} lines)"
    output += "\n</file>"

    return {
        "content": "\n".join(raw_lines),
        "total_lines": total_lines,
        "lines_read": len(raw_lines),
        "offset": offset,
        "truncated": truncated,
        "output": output
    }


# ============ Web Tools ============

def web_fetch(url: str, prompt: str) -> Dict[str, Any]:
    """
    Fetch content from a URL and process it.
    Similar to Claude Code's WebFetch tool.

    Args:
        url: The URL to fetch content from
        prompt: What information to extract from the page

    Returns:
        Processed content from the URL
    """
    import requests
    from bs4 import BeautifulSoup
    import html2text

    # Validate URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        # Fetch the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()

        # Check for redirect to different host
        if response.history:
            original_host = url.split('/')[2]
            final_host = response.url.split('/')[2]
            if original_host != final_host:
                return {
                    "success": False,
                    "error": f"URL redirected to different host: {response.url}",
                    "redirect_url": response.url
                }

        content_type = response.headers.get('content-type', '').lower()

        # Handle different content types
        if 'text/html' in content_type:
            # Convert HTML to markdown
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Convert to markdown
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0  # No wrapping
            markdown = h.handle(str(soup))

            # Truncate if too long
            max_chars = 50000
            if len(markdown) > max_chars:
                markdown = markdown[:max_chars] + "\n\n... (content truncated)"

            return {
                "success": True,
                "url": response.url,
                "content": markdown,
                "content_type": "text/html",
                "prompt": prompt,
                "message": f"Successfully fetched {response.url}. Use the content above to answer: {prompt}"
            }

        elif 'application/json' in content_type:
            return {
                "success": True,
                "url": response.url,
                "content": response.text[:50000],
                "content_type": "application/json",
                "prompt": prompt
            }

        elif 'text/' in content_type:
            return {
                "success": True,
                "url": response.url,
                "content": response.text[:50000],
                "content_type": content_type,
                "prompt": prompt
            }

        else:
            return {
                "success": False,
                "error": f"Unsupported content type: {content_type}",
                "url": response.url
            }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch URL: {str(e)}"}


def web_search(query: str) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    Similar to Claude Code's WebSearch tool.

    Args:
        query: The search query

    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        from ddgs import DDGS

        results = []
        ddgs = DDGS()
        for r in ddgs.text(query, max_results=10):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })

        if not results:
            return {
                "success": True,
                "query": query,
                "results": [],
                "count": 0,
                "message": "No results found"
            }

        # Format output
        output_lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(f"{i}. [{r['title']}]({r['url']})")
            if r['snippet']:
                output_lines.append(f"   {r['snippet'][:200]}")
            output_lines.append("")

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "output": "\n".join(output_lines)
        }

    except ImportError:
        return {
            "success": False,
            "error": "ddgs package not installed. Run: pip install ddgs"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}"
        }


# ============ Tool Definitions for Claude ============

# Base tools that are always available (non-MCP tools)
BASE_TOOLS = [
    {
        "name": "list_skills",
        "description": "List all available skills. Use this first to see what skills are available before reading one.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_skill",
        "description": "Get the full documentation of a specific skill. Use this to learn how to use a library or perform a task before writing code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to read (e.g., 'data-analyzer', 'pdf-converter')",
                },
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "execute_code",
        "description": f"""Execute Python code. Variables, imports, and state persist across calls within the same session (powered by IPython kernel).

IMPORTANT: Code runs in an isolated workspace directory, NOT the project root.
To access project files, use absolute paths (e.g., "{_SKILLS_DIR}/my-skill/scripts/main.py").""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "bash",
        "description": f"""Execute a shell command. Use for git, npm, pip, and other CLI tools.

IMPORTANT:
- Commands run in an isolated workspace directory, NOT the project root
- To access project files, use absolute paths (e.g., "python {_SKILLS_DIR}/my-skill/scripts/main.py")
- Use for system commands, not for file operations (use read/write/edit instead)
- Supports optional timeout parameter
Examples:
- bash(command="pip install pandas")
- bash(command="python {_SKILLS_DIR}/my-skill/scripts/main.py")
- bash(command="ls -la")""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (default: 120)",
                },
            },
            "required": ["command"],
        },
    },
    # Code exploration tools - for reading skill source code
    {
        "name": "glob",
        "description": """Search for files matching a glob pattern. Use this to find source code files in skill directories.

Examples:
- glob(pattern="**/*.py") - Find all Python files
- glob(pattern="*.md", path="skills/data-analyzer") - Find markdown files in a skill
- glob(pattern="**/*test*.py") - Find test files

Results are sorted by modification time (newest first), limited to 100 files.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.py', '*.md', '**/test_*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to 'skills' directory. Can be relative or absolute path.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": """Search for content in files using regex pattern. Use this to find function definitions, class names, or specific code patterns.

Examples:
- grep(pattern="def calculate") - Find function definitions
- grep(pattern="class.*Molecule", include="*.py") - Find Molecule classes in Python files
- grep(pattern="import pandas", path="skills/data-analyzer") - Find pandas imports

Results are sorted by modification time (newest first), limited to 100 matches.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to 'skills' directory.",
                },
                "include": {
                    "type": "string",
                    "description": "File pattern to include (e.g., '*.py', '*.ts', '*.{py,pyx}')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read",
        "description": """Read file contents with line numbers. Use this to read source code files after finding them with glob or grep.

Features:
- Shows line numbers for easy reference
- Supports reading large files in chunks using offset/limit
- Automatically detects and rejects binary files
- Suggests similar files if the requested file is not found

Examples:
- read(file_path="skills/data-analyzer/scripts/main.py") - Read the main module
- read(file_path="...", offset=100, limit=50) - Read lines 101-150""",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to working directory or absolute)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Default: 0",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Default: 2000",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write",
        "description": """Write content to a file. Creates the file if it doesn't exist, overwrites if it does.

IMPORTANT: This will overwrite the entire file. For modifying existing files, prefer using edit instead.

Examples:
- write(file_path="output/report.md", content="# Report\\n...")
- write(file_path="scripts/helper.py", content="def helper():\\n    pass")

Security: Cannot write to sensitive locations (.env, credentials, secrets, .git/)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write (relative to working directory or absolute)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit",
        "description": """Edit a file by replacing exact string matches. More precise than write for modifications.

IMPORTANT:
- You MUST read the file first using read before editing
- The old_string must match EXACTLY (including whitespace and indentation)
- By default, old_string must be unique in the file. Use replace_all=true to replace all occurrences.

Examples:
- edit(file_path="app.py", old_string="def old_func():", new_string="def new_func():")
- edit(file_path="config.py", old_string="DEBUG = True", new_string="DEBUG = False")
- edit(file_path="app.py", old_string="old_name", new_string="new_name", replace_all=true)

Security: Cannot edit sensitive files (.env, credentials, secrets)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace (must match exactly including whitespace)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace it with",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences. If false (default), old_string must be unique.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    # Web tools
    {
        "name": "web_fetch",
        "description": """Fetch content from a URL and convert it to markdown.

Use this to read web pages, documentation, or API responses.

Examples:
- web_fetch(url="https://docs.python.org/3/library/json.html", prompt="How to parse JSON?")
- web_fetch(url="https://api.github.com/repos/owner/repo", prompt="Get repo info")

Notes:
- HTML is converted to markdown for easier reading
- Content is truncated at 50KB
- Some sites may block automated requests""",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from",
                },
                "prompt": {
                    "type": "string",
                    "description": "What information to extract from the page",
                },
            },
            "required": ["url", "prompt"],
        },
    },
    {
        "name": "web_search",
        "description": """Search the web using DuckDuckGo.

Returns up to 10 search results with titles, URLs, and snippets.

Examples:
- web_search(query="Python asyncio tutorial 2024")
- web_search(query="FastAPI best practices")

Notes:
- Results include title, URL, and snippet
- Use web_fetch to read full content of interesting results""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
    # Interactive tool
    {
        "name": "ask_user",
        "description": """Ask the user for a decision.

Use this PROACTIVELY when:
- You need to confirm a decision with the user — e.g., "Should I proceed with installation?"
- You encounter multiple candidates or possibilities that the user should pick from — e.g., a search returns several matching skills, a dataset has multiple date columns, there are several files that could be the target. Present them as options instead of guessing.

Examples:
- ask_user(question="Proceed with installation?", options=["yes", "no"])
- ask_user(question="What format would you like for the output report?")
- ask_user(question="I found two matching skills. Which one should I use?", options=["data-analyzer", "csv-processor"])""",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 6,
                    "description": "Optional predefined answer choices (rendered as buttons). If omitted, user gets a free text input.",
                },
            },
            "required": ["question"],
        },
    },
]

# Meta tools — only available to meta agents (is_system=True).
# NOT included in BASE_TOOLS, TOOLS, or TOOLS_REGISTRY (invisible to UI).
META_TOOLS = [
    {
        "name": "list_registry_skills",
        "description": "List ALL registered skills in the system, including skills not equipped to the current agent. Returns name and description. Use this to discover what skills exist before planning an agent.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# Legacy TOOLS list for backward compatibility (includes all MCP tools by default)
# New code should use get_tools_for_agent() instead
TOOLS = BASE_TOOLS.copy()


# write wrapper that auto-detects the written file as an output file
def _write_with_output_detection(file_path, content, **kwargs):
    from app.tools.file_scanner import build_output_file_infos
    result = write(file_path, content)
    if result.get("success") and result.get("path"):
        file_infos = build_output_file_infos([Path(result["path"])], persist_dir=_PERSIST_DIR)
        if file_infos:
            result["new_files"] = file_infos
    return result


# Base tool functions that don't need workspace
# Note: execute_code and bash are created dynamically with workspace binding
BASE_TOOL_FUNCTIONS: Dict[str, Callable] = {
    "list_skills": list_skills,
    "get_skill": get_skill,
    "glob": lambda pattern, path=None, **kwargs: glob(pattern, path),
    "grep": lambda pattern, path=None, include=None, **kwargs: grep(pattern, path, include),
    "read": lambda file_path, offset=0, limit=None, **kwargs: read(file_path, offset, limit),
    "write": _write_with_output_detection,
    "edit": lambda file_path, old_string, new_string, replace_all=False, **kwargs: edit(file_path, old_string, new_string, replace_all),
    "web_fetch": lambda url, prompt, **kwargs: web_fetch(url, prompt),
    "web_search": lambda query, **kwargs: web_search(query),
}

# Meta tool functions — only added for meta agents
META_TOOL_FUNCTIONS: Dict[str, Callable] = {
    "list_registry_skills": list_registry_skills,
}


def create_workspace_bound_tools(workspace: AgentWorkspace) -> Dict[str, Callable]:
    """
    Create tool functions bound to a specific workspace.

    Args:
        workspace: The AgentWorkspace instance to bind

    Returns:
        Dict of tool name -> callable with workspace bound
    """
    from app.tools.file_scanner import snapshot_files, diff_new_files, build_output_file_infos

    def execute_code_with_scan(code, **kwargs):
        before = snapshot_files(workspace.workspace_dir, recursive=True)
        start = time.time()
        result = execute_code(code, workspace=workspace)
        elapsed = round(time.time() - start, 2)
        result["duration_seconds"] = elapsed
        after = snapshot_files(workspace.workspace_dir, recursive=True)
        new_files = diff_new_files(before, after)
        if new_files:
            result["new_files"] = build_output_file_infos(new_files, persist_dir=_PERSIST_DIR)
        return result

    def bash_with_scan(command, timeout=None, **kwargs):
        before = snapshot_files(workspace.workspace_dir, recursive=True)
        start = time.time()
        result = bash(command, workspace=workspace, timeout=timeout)
        elapsed = round(time.time() - start, 2)
        result["duration_seconds"] = elapsed
        after = snapshot_files(workspace.workspace_dir, recursive=True)
        new_files = diff_new_files(before, after)
        if new_files:
            result["new_files"] = build_output_file_infos(new_files, persist_dir=_PERSIST_DIR)
        return result

    def read_in_workspace(file_path, offset=0, limit=None, **kwargs):
        """Read that resolves relative paths to workspace_dir."""
        filepath = Path(file_path)
        if not filepath.is_absolute():
            filepath = workspace.workspace_dir / file_path
        return read(str(filepath), offset, limit)

    def write_in_workspace(file_path, content, **kwargs):
        """Write that resolves relative paths to workspace_dir (same cwd as execute_code)."""
        from app.tools.file_scanner import build_output_file_infos
        filepath = Path(file_path)
        if not filepath.is_absolute():
            filepath = workspace.workspace_dir / file_path
        result = write(str(filepath), content)
        if result.get("success") and result.get("path"):
            file_infos = build_output_file_infos([Path(result["path"])], persist_dir=_PERSIST_DIR)
            if file_infos:
                result["new_files"] = file_infos
        return result

    def glob_in_workspace(pattern, path=None, **kwargs):
        """Glob that resolves relative paths to workspace_dir."""
        if path:
            p = Path(path)
            if not p.is_absolute():
                path = str(workspace.workspace_dir / path)
        return glob(pattern, path)

    def grep_in_workspace(pattern, path=None, include=None, **kwargs):
        """Grep that resolves relative paths to workspace_dir."""
        if path:
            p = Path(path)
            if not p.is_absolute():
                path = str(workspace.workspace_dir / path)
        return grep(pattern, path, include)

    def edit_in_workspace(file_path, old_string, new_string, replace_all=False, **kwargs):
        """Edit that resolves relative paths to workspace_dir."""
        filepath = Path(file_path)
        if not filepath.is_absolute():
            filepath = workspace.workspace_dir / file_path
        return edit(str(filepath), old_string, new_string, replace_all)

    return {
        "execute_code": execute_code_with_scan,
        "bash": bash_with_scan,
        "read": read_in_workspace,
        "write": write_in_workspace,
        "glob": glob_in_workspace,
        "grep": grep_in_workspace,
        "edit": edit_in_workspace,
    }


def _collect_env_for_executor(skill_env_vars: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Collect environment variables to pass to a remote executor container.

    Reads user-configured keys from the .env file and fetches their current
    values from os.environ (which may have been updated at runtime via the
    Settings / Environment page).  Merges in skill-specific env vars on top.
    """
    env: Dict[str, str] = {}

    # 1. User-configured env vars (API keys etc.) from .env file (multi-worker safe)
    try:
        from app.config import read_env_all
        env.update(read_env_all())
    except Exception:
        pass

    # 2. Skill-specific env vars (override)
    if skill_env_vars:
        env.update(skill_env_vars)

    return env


def create_executor_bound_tools(
    executor_name: str,
    workspace_id: str,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Callable]:
    """
    Create tool functions that execute code in a remote executor container.

    Args:
        executor_name: Name of the executor (e.g., 'base', 'data-analysis')
        workspace_id: Workspace ID for isolation
        env_vars: Environment variables to forward to the executor

    Returns:
        Dict of tool name -> callable that calls executor container
    """
    import httpx
    from app.services.executor_config import get_executor_url
    from app.tools.file_scanner import snapshot_files, diff_new_files, build_output_file_infos

    base_url = get_executor_url(executor_name)
    wks_path = Path("/app/workspaces") / workspace_id

    # Collect env vars to pass to executor (API keys + skill env vars)
    executor_env = _collect_env_for_executor(env_vars) or None

    def execute_code_remote(code: str, **kwargs) -> Dict[str, Any]:
        """Execute Python code in remote executor container."""
        before = snapshot_files(wks_path, recursive=True) if wks_path.exists() else {}
        start = time.time()
        try:
            # Use synchronous httpx client to avoid asyncio issues
            with httpx.Client(timeout=kwargs.get('timeout', 300) + 30) as client:
                payload = {
                    "code": code,
                    "workspace_id": workspace_id,
                    "timeout": kwargs.get('timeout', 300),
                }
                if executor_env:
                    payload["env"] = executor_env
                response = client.post(
                    f"{base_url}/execute/python",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            elapsed = round(time.time() - start, 2)
            output = {
                "success": result.get("exit_code", -1) == 0,
                "output": result.get("stdout", ""),
                "error": result.get("stderr") if result.get("exit_code", -1) != 0 else None,
                "duration_seconds": elapsed,
            }
        except httpx.TimeoutException:
            return {"success": False, "output": "", "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

        after = snapshot_files(wks_path, recursive=True) if wks_path.exists() else {}
        new_files = diff_new_files(before, after)
        if new_files:
            output["new_files"] = build_output_file_infos(new_files, persist_dir=_PERSIST_DIR)
        return output

    def bash_remote(command: str, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Execute shell command in remote executor container."""
        before = snapshot_files(wks_path, recursive=True) if wks_path.exists() else {}
        start = time.time()
        try:
            effective_timeout = timeout or 300
            with httpx.Client(timeout=effective_timeout + 30) as client:
                payload = {
                    "command": command,
                    "workspace_id": workspace_id,
                    "timeout": effective_timeout,
                }
                if executor_env:
                    payload["env"] = executor_env
                response = client.post(
                    f"{base_url}/execute/bash",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            elapsed = round(time.time() - start, 2)
            output = {
                "success": result.get("exit_code", -1) == 0,
                "output": result.get("stdout", ""),
                "error": result.get("stderr") if result.get("exit_code", -1) != 0 else None,
                "duration_seconds": elapsed,
            }
        except httpx.TimeoutException:
            return {"success": False, "output": "", "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

        after = snapshot_files(wks_path, recursive=True) if wks_path.exists() else {}
        new_files = diff_new_files(before, after)
        if new_files:
            output["new_files"] = build_output_file_infos(new_files, persist_dir=_PERSIST_DIR)
        return output

    def read_in_workspace(file_path, offset=0, limit=None, **kwargs):
        """Read that resolves relative paths to workspace_dir."""
        filepath = Path(file_path)
        if not filepath.is_absolute():
            filepath = wks_path / file_path
        return read(str(filepath), offset, limit)

    def write_in_workspace(file_path, content, **kwargs):
        """Write that resolves relative paths to workspace_dir (same cwd as execute_code)."""
        from app.tools.file_scanner import build_output_file_infos
        filepath = Path(file_path)
        if not filepath.is_absolute():
            filepath = wks_path / file_path
        result = write(str(filepath), content)
        if result.get("success") and result.get("path"):
            file_infos = build_output_file_infos([Path(result["path"])], persist_dir=_PERSIST_DIR)
            if file_infos:
                result["new_files"] = file_infos
        return result

    def glob_in_workspace(pattern, path=None, **kwargs):
        """Glob that resolves relative paths to workspace_dir."""
        if path:
            p = Path(path)
            if not p.is_absolute():
                path = str(wks_path / path)
        return glob(pattern, path)

    def grep_in_workspace(pattern, path=None, include=None, **kwargs):
        """Grep that resolves relative paths to workspace_dir."""
        if path:
            p = Path(path)
            if not p.is_absolute():
                path = str(wks_path / path)
        return grep(pattern, path, include)

    def edit_in_workspace(file_path, old_string, new_string, replace_all=False, **kwargs):
        """Edit that resolves relative paths to workspace_dir."""
        filepath = Path(file_path)
        if not filepath.is_absolute():
            filepath = wks_path / file_path
        return edit(str(filepath), old_string, new_string, replace_all)

    return {
        "execute_code": execute_code_remote,
        "bash": bash_remote,
        "read": read_in_workspace,
        "write": write_in_workspace,
        "glob": glob_in_workspace,
        "grep": grep_in_workspace,
        "edit": edit_in_workspace,
    }


# Legacy TOOL_FUNCTIONS for backward compatibility (without workspace - will fail for execute_code/bash)
TOOL_FUNCTIONS: Dict[str, Callable] = {
    **BASE_TOOL_FUNCTIONS,
    "execute_code": lambda code, **kwargs: execute_code(code, workspace=None),
    "bash": lambda command, timeout=None, **kwargs: bash(command, workspace=None, timeout=timeout),
}


def _create_mcp_tool_function(server_name: str, tool_name: str) -> Callable:
    """Create a function that calls an MCP tool."""
    def mcp_tool_func(**kwargs) -> Dict[str, Any]:
        return call_mcp_tool(server_name, tool_name, kwargs)
    return mcp_tool_func


# ─── Memory Tools ────────────────────────────────────────────────

MEMORY_TOOLS = [
    {
        "name": "memory_search",
        "description": "Search the agent's memory for relevant information using semantic similarity. Use this to recall previously saved facts, preferences, procedures, and context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_save",
        "description": "Save important information to the agent's long-term memory. Use this to remember facts, user preferences, procedures, or important context for future conversations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to save (1-3 concise sentences)"
                },
                "category": {
                    "type": "string",
                    "description": "Category of the memory entry",
                    "enum": ["fact", "preference", "procedure", "context"]
                }
            },
            "required": ["content"]
        }
    },
]


def create_memory_tool_functions(agent_id: Optional[str] = None) -> Dict[str, Callable]:
    """Create bound memory tool functions for a specific agent."""
    from app.services.memory_service import search_memory_sync, save_memory_sync

    def memory_search(query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        results = search_memory_sync(query, agent_id=agent_id, top_k=top_k)
        if not results:
            return {"results": [], "message": "No matching memories found."}
        return {"results": results}

    def memory_save(content: str, category: str = "context", **kwargs) -> Dict[str, Any]:
        valid_categories = {"fact", "preference", "procedure", "context", "session_summary"}
        if category not in valid_categories:
            category = "context"
        if len(content) > 4096:
            content = content[:4096]
        entry = save_memory_sync(
            content=content,
            agent_id=agent_id,
            category=category,
            source="agent_tool",
        )
        return {"saved": True, "id": entry["id"], "content": content, "category": category}

    return {
        "memory_search": memory_search,
        "memory_save": memory_save,
    }


def get_tools_for_agent(
    equipped_mcp_servers: Optional[List[str]] = None,
    skill_names: Optional[List[str]] = None,
    executor_name: Optional[str] = None,
    workspace_id: Optional[str] = None,
    is_meta_agent: bool = False,
    agent_id: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Callable], AgentWorkspace]:
    """
    Get tools, tool functions, and workspace for an agent.

    Creates a new AgentWorkspace for this request. The caller is responsible
    for calling workspace.cleanup() when the agent request is complete.

    Args:
        equipped_mcp_servers: List of MCP server names to include.
                              None = use default enabled servers only.
                              Empty list = no MCP servers.
        skill_names: List of skill names to load env vars for.
        executor_name: Optional executor name for remote code execution.
                       If provided, execute_code and bash will run in
                       the specified executor container instead of locally.
        is_meta_agent: If True, include meta tools (e.g. list_registry_skills)
                       that are only available to system/meta agents.
        agent_id: Optional agent preset ID for memory tools scope.

    Returns:
        Tuple of (tools list for Claude, tool_functions dict, workspace)
    """
    # Create workspace
    env_vars: Dict[str, str] = {}
    workspace = AgentWorkspace(env_vars=env_vars, workspace_id=workspace_id)

    # Start with base tools
    tools = BASE_TOOLS.copy()

    # Add execution tools - either remote (executor container) or local (workspace)
    if executor_name:
        # Use remote executor container for code execution
        tool_functions = {
            **BASE_TOOL_FUNCTIONS,
            **create_executor_bound_tools(executor_name, workspace.workspace_id, env_vars),
        }
    else:
        # Use local workspace for code execution
        tool_functions = {
            **BASE_TOOL_FUNCTIONS,
            **create_workspace_bound_tools(workspace),
        }

    # Add meta tools for system/meta agents
    if is_meta_agent:
        tools.extend(META_TOOLS)
        tool_functions.update(META_TOOL_FUNCTIONS)

    # Add memory tools when agent_id is provided
    if agent_id:
        tools.extend(MEMORY_TOOLS)
        tool_functions.update(create_memory_tool_functions(agent_id))

    # Get MCP client
    mcp_client = get_mcp_client()

    # Determine which MCP servers to include
    if equipped_mcp_servers is None:
        # Use default enabled servers only (not all servers)
        server_names = get_default_enabled_mcp_servers()
    else:
        # Only include specified servers
        server_names = equipped_mcp_servers

    # Add tools from each equipped MCP server
    for server_name in server_names:
        server = mcp_client.get_server(server_name)
        if not server:
            continue
        for mcp_tool in server.tools:
            # Add tool definition
            tools.append({
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "input_schema": mcp_tool.input_schema
            })

            # Add tool function
            tool_functions[mcp_tool.name] = _create_mcp_tool_function(server_name, mcp_tool.name)

    return tools, tool_functions, workspace


# Tool parameter requirements for validation
TOOL_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "get_skill": ["skill_name"],
    "execute_code": ["code"],
    "bash": ["command"],
    "glob": ["pattern"],
    "grep": ["pattern"],
    "read": ["file_path"],
    "write": ["file_path", "content"],
    "edit": ["file_path", "old_string", "new_string"],
    "web_fetch": ["url", "prompt"],
    "web_search": ["query"],
    "ask_user": ["question"],
    "memory_search": ["query"],
    "memory_save": ["content"],
}


def call_tool(
    name: str,
    arguments: Dict[str, Any],
    allowed_skills: Optional[List[str]] = None,
    tool_functions: Optional[Dict[str, Callable]] = None
) -> str:
    """Call a tool by name with arguments, return JSON string result.

    Args:
        name: Tool name
        arguments: Tool arguments
        allowed_skills: Optional list of skill names to filter (for list_skills/get_skill)
        tool_functions: Optional dict of tool functions (if None, uses global TOOL_FUNCTIONS)
    """
    # Use provided tool_functions or fall back to global
    funcs = tool_functions if tool_functions is not None else TOOL_FUNCTIONS

    if name not in funcs:
        return json.dumps({"error": f"Unknown tool: {name}"})

    # Validate required parameters before calling the tool
    if name in TOOL_REQUIRED_PARAMS:
        missing_params = [p for p in TOOL_REQUIRED_PARAMS[name] if p not in arguments or arguments[p] is None]
        if missing_params:
            return json.dumps({
                "error": f"Missing required parameter(s) for '{name}': {', '.join(missing_params)}. "
                         f"Please provide: {', '.join(TOOL_REQUIRED_PARAMS[name])}."
            })

    try:
        # Special handling for skill-related tools to pass allowed_skills
        if name == "list_skills":
            result = list_skills(allowed_skills=allowed_skills)
        elif name == "get_skill":
            result = get_skill(arguments.get("skill_name"))
        else:
            result = funcs[name](**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


async def acall_tool(
    name: str,
    arguments: Dict[str, Any],
    allowed_skills: Optional[List[str]] = None,
    tool_functions: Optional[Dict[str, Callable]] = None,
) -> str:
    """Async wrapper: runs sync call_tool in a thread pool executor.

    All tool functions are synchronous (subprocess, sync DB, MCP stdio).
    This wrapper makes them awaitable without blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: call_tool(name, arguments, allowed_skills, tool_functions),
    )
