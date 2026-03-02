"""
Skills Agent - An async agent that uses skills and tools to complete tasks

The agent:
1. Receives a user request
2. Uses an LLM (via native SDK) to decide what to do
3. Calls tools (list_skills, get_skill, execute_code, etc.)
4. Loops until task is complete
5. Returns the final result

Supports multiple LLM providers:
- Anthropic (Claude)
- OpenRouter (access to many models)
- OpenAI (GPT-4o, o1)
- Google (Gemini)
"""
import asyncio
import copy
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field, asdict

from app.config import settings
from app.agent.tools import TOOLS, call_tool, acall_tool, get_tools_for_agent, BASE_TOOLS, META_TOOLS, get_mcp_client, _SKILLS_DIR
from app.core.tools_registry import get_all_tools, get_tools_by_ids, tools_to_claude_format
from app.llm import LLMClient, LLMTextBlock, LLMToolCall
from app.llm.models import MODEL_CONTEXT_LIMITS, DEFAULT_CONTEXT_LIMIT, get_context_limit

if TYPE_CHECKING:
    from app.agent.event_stream import EventStream

# Context window compression constants
COMPRESSION_THRESHOLD_RATIO = 0.7  # Trigger compression when input tokens exceed 70% of limit
MAX_RECENT_TURNS = 5               # Keep at most 5 recent logical turns
RECENT_TURNS_TOKEN_BUDGET = 0.25   # Recent turns can use up to 25% of context limit
CHARS_PER_TOKEN = 3.5              # Conservative estimate for mixed CJK/English text


SUMMARY_SYSTEM_PROMPT = """You have been given a partial transcript of a conversation between a user and an AI assistant. Write a summary that provides continuity so the assistant can continue making progress in a future context where the raw history is replaced by this summary.

You must wrap your summary in a <summary></summary> block with the following sections:

<summary>
## Primary Request and Intent
The user's explicit goals and overall task. Include any clarifications or constraints the user provided.

## Key Technical Concepts
Technologies, frameworks, models, APIs, and domain-specific terms discussed. Include exact model names, package versions, and configuration values.

## Files and Code Sections
Specific files read, created, or modified, with brief notes on what was done. Include exact file paths. For critical code changes, preserve the key snippets verbatim.

## Problem Solving
Completed troubleshooting efforts — what was tried, what worked, what failed and why. Include exact error messages if relevant.

## All User Messages
List every non-tool-use user statement **verbatim** (or near-verbatim for very long messages >500 chars). Number them chronologically. This is the most critical section — user intent must be preserved precisely, not paraphrased.

## Current State
What was just completed immediately before this summary. Be specific about the last action taken and its result.

## Pending Tasks
Outstanding work items and next steps, in priority order. Include any blockers.
</summary>

Be concise but thorough. Preserve exact file paths, variable names, model names, API parameters, and configuration values. Do not omit details that would be needed to continue the work.

Note: File tracking sections (<read-files> and <modified-files>) will be appended automatically — do not duplicate them in your summary.

{file_tracking_section}"""


SUMMARY_UPDATE_PROMPT = """You have been given NEW conversation messages that occurred after a previous summary. Update the existing summary with the new information.

<previous-summary>
{previous_summary}
</previous-summary>

Rules:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, user messages, and context from the new messages
- UPDATE "Current State" and "Pending Tasks" based on what was accomplished
- APPEND new user statements to "All User Messages" (preserve existing entries verbatim)
- PRESERVE exact file paths, function names, error messages, and configuration values
- If something is no longer relevant, you may remove it
- Use the same <summary> section structure as the original

Note: File tracking sections (<read-files> and <modified-files>) will be appended automatically — do not duplicate them in your summary.

{file_tracking_section}"""


TURN_PREFIX_SUMMARY_PROMPT = """This is the PREFIX of a conversation turn that was too large to keep in full. The SUFFIX (recent work) is retained verbatim. Summarize the prefix to provide context for the retained suffix.

Write a brief summary with these sections:
## Original Request
What did the user ask for in this turn?

## Early Progress
Key decisions and work done in the prefix

## Context for Suffix
Information needed to understand the retained recent work"""


def _extract_file_operations(messages: List[Dict]) -> tuple:
    """Extract file read and modify operations from messages.

    Scans tool_use and tool_result blocks for file operations.

    Returns:
        (read_files: set, modified_files: set)
    """
    read_files: set = set()
    modified_files: set = set()

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name in ("read", "read_file"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        read_files.add(fp)
                elif tool_name in ("glob", "glob_files"):
                    # glob is a read-like operation
                    path = tool_input.get("path", "")
                    pattern = tool_input.get("pattern", "")
                    if path:
                        read_files.add(f"{path}/{pattern}" if pattern else path)
                elif tool_name in ("grep", "grep_search"):
                    path = tool_input.get("path", "")
                    if path:
                        read_files.add(path)
                elif tool_name in ("write", "write_file"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        modified_files.add(fp)
                elif tool_name in ("edit", "edit_file"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        modified_files.add(fp)
                elif tool_name in ("execute_code", "bash", "execute_command"):
                    # For code execution, check new_files in the result later
                    pass

            elif block.get("type") == "tool_result":
                # Check for new_files in tool results (from execute_code/bash/write)
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    try:
                        parsed = json.loads(result_content)
                        if isinstance(parsed, dict):
                            for nf in parsed.get("new_files", []):
                                filename = nf.get("filename", "")
                                if filename:
                                    modified_files.add(filename)
                    except (json.JSONDecodeError, TypeError):
                        pass

    return read_files, modified_files


def _build_file_tracking_section(read_files: set, modified_files: set) -> str:
    """Build the file tracking XML section for appending to summaries."""
    parts = []
    if read_files:
        files_list = "\n".join(sorted(read_files))
        parts.append(f"<read-files>\n{files_list}\n</read-files>")
    if modified_files:
        files_list = "\n".join(sorted(modified_files))
        parts.append(f"<modified-files>\n{files_list}\n</modified-files>")
    return "\n".join(parts)


def _extract_previous_file_tracking(summary_text: str) -> tuple:
    """Extract existing file tracking from a previous summary.

    Returns:
        (read_files: set, modified_files: set)
    """
    import re
    read_files: set = set()
    modified_files: set = set()

    read_match = re.search(r"<read-files>\s*(.*?)\s*</read-files>", summary_text, re.DOTALL)
    if read_match:
        for line in read_match.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                read_files.add(line)

    mod_match = re.search(r"<modified-files>\s*(.*?)\s*</modified-files>", summary_text, re.DOTALL)
    if mod_match:
        for line in mod_match.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                modified_files.add(line)

    return read_files, modified_files


def _serialize_messages_for_summary(messages: List[Dict]) -> str:
    """Serialize messages into readable text for summarization.

    Truncates tool_use inputs to 500 chars and tool_results to 1000 chars.
    If total text exceeds 100K chars, takes first and last halves with a truncation marker.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        parts.append(f"[{role}]: {block.get('text', '')}")
                    elif block_type == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = json.dumps(block.get("input", {}), ensure_ascii=False)
                        if len(tool_input) > 500:
                            tool_input = tool_input[:500] + "...(truncated)"
                        parts.append(f"[{role} -> tool_use({tool_name})]: {tool_input}")
                    elif block_type == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str) and len(tool_content) > 1000:
                            tool_content = tool_content[:1000] + "...(truncated)"
                        parts.append(f"[tool_result]: {tool_content}")
                elif isinstance(block, str):
                    parts.append(f"[{role}]: {block}")

    text = "\n\n".join(parts)

    # If total text exceeds 100K characters, take head + tail
    max_chars = 100_000
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n\n[... truncated middle section ...]\n\n" + text[-half:]

    return text


async def compress_messages_standalone(
    messages: List[Dict],
    model_provider: str,
    model_name: str,
    verbose: bool = False,
) -> tuple:
    """Compress old messages into a structured summary, keeping recent turns.

    Standalone version that doesn't require a SkillsAgent instance.
    Used for pre-compressing session context before starting the agent.

    Supports iterative summaries and cumulative file tracking (same as
    SkillsAgent._compress_messages).

    Returns:
        (compressed_messages, summary_input_tokens, summary_output_tokens)
    """
    import re as _re

    context_limit = get_context_limit(model_provider, model_name)

    # Find logical turn boundaries: indices of real user messages (not tool_result).
    turn_boundaries = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            turn_boundaries.append(i)
        elif isinstance(content, list):
            is_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )
            if not is_tool_result:
                turn_boundaries.append(i)

    # Need at least 2 turn boundaries (1 to compress + 1 to keep)
    if len(turn_boundaries) < 2:
        if verbose:
            print("[Pre-Compress] Not enough logical turns to compress, skipping")
        return messages, 0, 0

    # Dynamically select how many recent turns to keep within token budget.
    max_recent_tokens = int(context_limit * RECENT_TURNS_TOKEN_BUDGET)
    accumulated_tokens = 0
    keep_turns = 0

    for idx in range(len(turn_boundaries) - 1, -1, -1):
        turn_start = turn_boundaries[idx]
        turn_end = turn_boundaries[idx + 1] if idx + 1 < len(turn_boundaries) else len(messages)
        turn_chars = sum(
            len(json.dumps(messages[i].get("content", ""), ensure_ascii=False))
            for i in range(turn_start, turn_end)
        )
        turn_tokens = turn_chars / CHARS_PER_TOKEN
        if accumulated_tokens + turn_tokens > max_recent_tokens and keep_turns >= 1:
            break
        accumulated_tokens += turn_tokens
        keep_turns += 1
        if keep_turns >= MAX_RECENT_TURNS:
            break

    # Ensure we have something left to compress
    if keep_turns >= len(turn_boundaries):
        if verbose:
            print("[Pre-Compress] All turns fit in budget, skipping")
        return messages, 0, 0

    split_point = turn_boundaries[-keep_turns]

    if verbose:
        print(f"[Pre-Compress] Keeping {keep_turns} recent logical turns (~{int(accumulated_tokens)} tokens)")

    old_messages = messages[:split_point]
    recent_messages = messages[split_point:]

    if verbose:
        print(f"[Pre-Compress] Compressing {len(old_messages)} old messages, keeping {len(recent_messages)} recent messages")

    # Extract file operations for cumulative tracking
    read_files, modified_files = _extract_file_operations(old_messages)

    # Check if this is an iterative compression
    has_previous_summary = False
    previous_summary_text = ""
    if old_messages and old_messages[0].get("role") == "user":
        first_content = old_messages[0].get("content", "")
        if isinstance(first_content, str) and "<summary>" in first_content:
            has_previous_summary = True
            match = _re.search(r"<summary>(.*?)</summary>", first_content, _re.DOTALL)
            if match:
                previous_summary_text = match.group(1).strip()
            prev_read, prev_mod = _extract_previous_file_tracking(first_content)
            read_files |= prev_read
            modified_files |= prev_mod

    file_tracking = _build_file_tracking_section(read_files, modified_files)

    # Call LLM to generate a structured summary
    client = LLMClient(provider=model_provider, model=model_name)
    summary_input_tokens = 0
    summary_output_tokens = 0
    try:
        if has_previous_summary and previous_summary_text:
            new_messages_only = [m for m in old_messages[1:] if not (
                m.get("role") == "assistant" and isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "text"
                       and "I understand the context" in b.get("text", "")
                       for b in m["content"])
            )]
            serialized = _serialize_messages_for_summary(new_messages_only) if new_messages_only else ""
            system_prompt = SUMMARY_UPDATE_PROMPT.format(
                previous_summary=previous_summary_text,
                file_tracking_section=file_tracking,
            )
            user_content = f"Please update the summary with these new conversation messages:\n\n{serialized}" if serialized else "No new messages to add."

            if verbose:
                print("[Pre-Compress] Using iterative summary update (previous summary detected)")
        else:
            serialized = _serialize_messages_for_summary(old_messages)
            system_prompt = SUMMARY_SYSTEM_PROMPT.format(file_tracking_section=file_tracking)
            user_content = f"Please summarize the following conversation:\n\n{serialized}"

        summary_response = await client.acreate(
            messages=[{
                "role": "user",
                "content": user_content,
            }],
            system=system_prompt,
            max_tokens=4096,
        )
        summary_input_tokens = summary_response.usage.input_tokens
        summary_output_tokens = summary_response.usage.output_tokens
        summary_text = summary_response.text_content

        if verbose:
            print(f"[Pre-Compress] Summary generated ({summary_input_tokens} in / {summary_output_tokens} out)")
    except Exception as e:
        # Fallback: use truncated serialized text as summary
        if verbose:
            print(f"[Pre-Compress] Summary API call failed: {e}, using fallback")
        if has_previous_summary and previous_summary_text:
            summary_text = previous_summary_text
        else:
            serialized = _serialize_messages_for_summary(old_messages)
            summary_text = serialized
        if len(summary_text) > 10000:
            summary_text = summary_text[:5000] + "\n\n[... truncated ...]\n\n" + summary_text[-5000:]

    # Build the compression message.
    if "<summary>" not in summary_text:
        summary_text = f"<summary>\n{summary_text}\n</summary>"

    # Append file tracking to summary if not already present
    if file_tracking and "<read-files>" not in summary_text and "<modified-files>" not in summary_text:
        summary_text = summary_text.rstrip()
        if summary_text.endswith("</summary>"):
            summary_text = summary_text[:-len("</summary>")] + f"\n\n{file_tracking}\n</summary>"
        else:
            summary_text += f"\n\n{file_tracking}"

    compression_content = (
        "This session is being continued from a previous conversation that ran out of context. "
        "The summary below covers the earlier portion of the conversation.\n\n"
        f"{summary_text}\n\n"
        "Please continue the conversation from where we left off without asking the user any further questions. "
        "Continue with the last task that you were asked to work on."
    )

    # Build compressed messages: summary as first user message + recent messages
    compressed = [{"role": "user", "content": compression_content}]

    # If recent_messages starts with a user message, we need an assistant acknowledgment
    if recent_messages and recent_messages[0].get("role") == "user":
        compressed.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}]
        })

    compressed.extend(recent_messages)

    if verbose:
        print(f"[Pre-Compress] Messages reduced from {len(messages)} to {len(compressed)}")

    return compressed, summary_input_tokens, summary_output_tokens


@dataclass
class LLMCall:
    """Record of a single LLM API call."""
    turn: int
    timestamp: str
    model: str
    request_messages: List[Dict]
    response_content: List[Dict]
    stop_reason: str
    input_tokens: int
    output_tokens: int


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    role: str  # "assistant" or "tool"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_result: Optional[str] = None


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    llm_calls: List[LLMCall] = field(default_factory=list)
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error: Optional[str] = None
    log_file: Optional[str] = None
    skills_used: List[str] = field(default_factory=list)
    output_files: List[Dict] = field(default_factory=list)
    final_messages: List[Dict] = field(default_factory=list)


@dataclass
class StreamEvent:
    """Event emitted during streaming execution."""
    event_type: str  # "turn_start", "text_delta", "assistant", "tool_call", "tool_result", "output_file", "turn_complete", "complete", "error"
    turn: int
    data: Dict[str, Any] = field(default_factory=dict)


BASE_SYSTEM_PROMPT = """You are a helpful assistant with access to skills and tools.
{custom_instructions}
## Current Date
Today is {current_date}. Use this date when searching for recent news or time-sensitive information.

## Equipped Skills
{equipped_skills_section}

**IMPORTANT:** When user request matches a skill's trigger words, you MUST:
1. Call `get_skill(skill_name)` to read the full documentation
2. Follow the skill's workflow exactly as documented
3. Use the tools specified in the skill (e.g., tavily_search for topic-collector)

## Available Tools

### Skill Management
- list_skills: List equipped skills for this agent
- get_skill: Read skill documentation (can read any skill, not just equipped ones)

### Code Execution
- execute_code: Execute Python code (variables persist across calls via IPython kernel)
- execute_command: Execute shell commands

### Code Exploration (for reading skill source code)
- glob_files: Search for files by pattern (e.g., "**/*.py")
- grep_search: Search for content in files using regex
- read_file: Read file contents with line numbers

{mcp_tools_section}

### User Interaction
- ask_user: Ask the user for a decision. Provide predefined options when possible.

**When to use ask_user proactively:**
- You need to confirm a decision with the user — e.g., "Should I proceed with installation?"
- You encounter multiple candidates or possibilities that the user needs to pick from — e.g., a search returns several matching skills, a dataset has multiple date columns, or there are several files that could be the target. Present them as options and let the user decide instead of guessing.

## Workflow
1. First, list available skills to see what's available
2. Read relevant skill documentation to learn the API
3. **If you need more details about the implementation**, use the code exploration tools:
   - Use `glob_files` to find source files in the skill directory
   - Use `grep_search` to find specific functions, classes, or patterns
   - Use `read_file` to read the actual source code
4. Write and execute code based on what you learned
5. If code fails, debug and retry
6. Return the final result to the user

## Code Exploration Tips
When using skills, you can explore the source code to understand:
- Function signatures and parameters
- Implementation details
- Available classes and methods
- Example usage patterns

Example workflow:
```
# Find all Python files in a skill
glob_files(pattern="**/*.py", path="{skills_dir}/data-analyzer")

# Search for a specific function
grep_search(pattern="def analyze", include="*.py", path="{skills_dir}/data-analyzer")

# Read the source file
read_file(file_path="{skills_dir}/data-analyzer/scripts/main.py")
```

## Working Directory
Your workspace is `{workspace_dir}`. All tools share this directory — relative paths resolve here.
- **Saving output files:** Use relative paths directly (e.g., `open("output.png", "wb")`, `df.to_csv("result.csv")`). They will be auto-detected as downloadable output files. **NEVER use `/tmp/` for output files**.
- **Accessing project files:** Use absolute paths (e.g., `read(file_path="{skills_dir}/my-skill/SKILL.md")`, `bash(command="python {skills_dir}/my-skill/scripts/main.py")`).
- `glob` and `grep` default to the skills directory when no path is specified.

## Skills Directory
The skills directory is `{skills_dir}`. All SKILL.md files, references, scripts, assets, etc. are stored in this directory.

## Important Notes
- Always read skill documentation before writing code
- Variables, imports, and state persist across execute_code calls within the same session
- If the kernel crashes, it automatically falls back to subprocess mode (variables won't persist)
- Use code exploration tools when skill docs are insufficient
- When task is complete, provide a clear final answer
"""


def _build_mcp_tools_section(tools: list) -> str:
    """Build the MCP tools section for system prompt."""
    # Get MCP tool names (tools not in BASE_TOOLS or META_TOOLS)
    builtin_names = set(t["name"] for t in BASE_TOOLS) | set(t["name"] for t in META_TOOLS)
    mcp_tools = [t for t in tools if t["name"] not in builtin_names]

    if not mcp_tools:
        return ""

    lines = ["### MCP Tools (External Services)"]
    for tool in mcp_tools:
        # Get first line of description
        desc = tool.get("description", "").split("\n")[0].strip()
        lines.append(f"- {tool['name']}: {desc}")

    return "\n".join(lines)


def _make_steering_event(turn: int, message: str) -> StreamEvent:
    """Create a steering_received StreamEvent with a unique steering_id."""
    return StreamEvent(
        event_type="steering_received",
        turn=turn,
        data={"message": message, "steering_id": f"steer-{uuid.uuid4().hex[:12]}"},
    )


class SkillsAgent:
    """Agent that uses skills and tools to complete tasks."""

    def __init__(
        self,
        model: str = None,
        model_provider: str = None,
        max_turns: int = 60,
        verbose: bool = True,
        log_dir: str = "./logs",
        allowed_skills: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        equipped_mcp_servers: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        executor_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        is_meta_agent: bool = False,
        agent_id: Optional[str] = None,
    ):
        # Model configuration
        self.model_provider = model_provider or settings.default_model_provider
        self.model = model or settings.default_model_name
        self.max_turns = max_turns
        self.verbose = verbose
        self.log_dir = log_dir
        self.allowed_skills = allowed_skills  # None means all skills are available
        self.allowed_tools = allowed_tools  # None means all tools are available
        self.equipped_mcp_servers = equipped_mcp_servers  # None means default MCP servers
        self.executor_name = executor_name  # Remote executor for code execution
        self.agent_id = agent_id  # Agent preset ID for memory scope
        self._background_tasks: set = set()  # prevent GC of fire-and-forget tasks

        # Initialize LLM client with provider-specific configuration
        self.client = LLMClient(
            provider=self.model_provider,
            model=self.model,
        )

        # Build the tools list and create workspace
        # Workspace is created per-agent instance for request isolation
        # If executor_name is provided, code execution runs in remote container
        all_tools, all_tool_functions, self.workspace = get_tools_for_agent(
            equipped_mcp_servers=equipped_mcp_servers,
            skill_names=allowed_skills,
            executor_name=executor_name,
            workspace_id=workspace_id,
            is_meta_agent=is_meta_agent,
            agent_id=agent_id,
        )

        # Filter tools if allowed_tools is specified
        # Note: MCP tools from equipped servers and meta tools are always included (not filtered out)
        if allowed_tools is not None:
            # Get MCP tool names from equipped servers
            always_include = set()
            if equipped_mcp_servers:
                mcp_client = get_mcp_client()
                for server_name in equipped_mcp_servers:
                    server = mcp_client.get_server(server_name)
                    if server:
                        for tool in server.tools:
                            always_include.add(tool.name)
            # Meta tools are always included for meta agents (not filtered by allowed_tools)
            if is_meta_agent:
                always_include.update(t["name"] for t in META_TOOLS)

            # Include tool if: in allowed_tools OR always-included (MCP/meta)
            self.tools = [t for t in all_tools if t["name"] in allowed_tools or t["name"] in always_include]
            self.tool_functions = {k: v for k, v in all_tool_functions.items() if k in allowed_tools or k in always_include}
        else:
            self.tools = all_tools
            self.tool_functions = all_tool_functions

        # Build dynamic system prompt with equipped skills and MCP tools info
        skills_section = self._build_equipped_skills_section()
        mcp_section = _build_mcp_tools_section(self.tools)
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Build custom instructions section
        custom_instructions = ""
        if custom_system_prompt:
            custom_instructions = f"\n## Custom Instructions\n{custom_system_prompt}\n"

        self.system_prompt = BASE_SYSTEM_PROMPT.format(
            current_date=current_date,
            equipped_skills_section=skills_section,
            mcp_tools_section=mcp_section,
            custom_instructions=custom_instructions,
            workspace_dir=str(self.workspace.workspace_dir),
            skills_dir=_SKILLS_DIR,
        )

        # Append memory bootstrap files to system prompt
        memory_section = self._build_memory_section()
        if memory_section:
            self.system_prompt += memory_section

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

    def _build_memory_section(self) -> str:
        """Build the memory section for system prompt from bootstrap files."""
        try:
            from app.services.memory_service import load_bootstrap_files
            files = load_bootstrap_files(self.agent_id)
            if not files:
                return ""
            parts = ["\n\n## Agent Memory"]
            for filename, content in files.items():
                parts.append(f"\n### {filename}\n{content}")
            return "\n".join(parts)
        except Exception as e:
            if self.verbose:
                print(f"[Memory] Failed to load bootstrap files: {e}")
            return ""

    def _build_equipped_skills_section(self) -> str:
        """Build the equipped skills section for system prompt."""
        from app.agent.tools import _fetch_skill_content_from_registry

        if not self.allowed_skills:
            return "No skills equipped. Use `list_skills` to see available skills."

        lines = ["The following skills are equipped and ready to use:\n"]

        for skill_name in self.allowed_skills:
            try:
                registry_skill = _fetch_skill_content_from_registry(skill_name)
                if not registry_skill:
                    lines.append(f"### {skill_name}")
                    lines.append("(Skill not found)")
                    lines.append("")
                    continue

                description = registry_skill.get("description", "")
                content_text = registry_skill.get("content", "")

                triggers = []
                # Look for trigger words in content
                content_lower = content_text.lower()
                if "trigger" in content_lower:
                    # Extract trigger section
                    for line in content_text.split("\n"):
                        if "- \"" in line or "- '" in line or '- "' in line:
                            trigger = line.strip().lstrip("-").strip().strip('"\'')
                            if trigger and len(trigger) < 50:
                                triggers.append(trigger)

                lines.append(f"### {skill_name}")
                if description:
                    lines.append(f"**Description:** {description}")
                if triggers:
                    lines.append(f"**Triggers:** {', '.join(triggers[:5])}")

                lines.append("")
            except Exception as e:
                lines.append(f"### {skill_name}")
                lines.append(f"(Error loading skill: {e})")
                lines.append("")

        return "\n".join(lines)

    def _get_context_limit(self) -> int:
        """Get the context window limit for the current model."""
        return get_context_limit(self.model_provider, self.model)

    def _should_compress(self, last_input_tokens: int) -> bool:
        """Check if context compression is needed based on the last API call's input token count."""
        threshold = int(self._get_context_limit() * COMPRESSION_THRESHOLD_RATIO)
        return last_input_tokens > threshold

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Check if an LLM error is retryable (transient network/server issue)."""
        err_str = str(error).lower()
        retryable_patterns = [
            "connection", "timeout", "rate limit", "rate_limit",
            "429", "500", "502", "503", "504",
            "overloaded", "service unavailable", "service_unavailable",
            "server error", "internal error",
            "incomplete chunked read", "peer closed",
            "reset by peer", "broken pipe", "fetch failed",
        ]
        return any(p in err_str for p in retryable_patterns)

    def _serialize_messages_for_summary(self, messages: List[Dict]) -> str:
        """Serialize messages into readable text for summarization. Delegates to module-level function."""
        return _serialize_messages_for_summary(messages)

    async def _compress_messages(self, messages: List[Dict]) -> tuple:
        """Compress old messages into a structured summary, keeping recent turns.

        Supports:
        - Iterative summaries: if old_messages starts with a previous summary, uses
          SUMMARY_UPDATE_PROMPT to merge new info into the existing summary.
        - Split turn handling: if the budget only allows 1 oversized turn, splits it
          at assistant message boundaries and summarizes the prefix.
        - Cumulative file tracking: extracts read/modified files and appends XML tags.

        Returns:
            (compressed_messages, summary_input_tokens, summary_output_tokens)
        """
        # Find logical turn boundaries: indices of real user messages (not tool_result).
        turn_boundaries = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                turn_boundaries.append(i)
            elif isinstance(content, list):
                is_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                )
                if not is_tool_result:
                    turn_boundaries.append(i)

        # Need at least 2 turn boundaries (1 to compress + 1 to keep)
        if len(turn_boundaries) < 2:
            if self.verbose:
                print("[Context Compression] Not enough logical turns to compress, skipping")
            return messages, 0, 0

        # Dynamically select how many recent turns to keep within token budget.
        max_recent_tokens = int(self._get_context_limit() * RECENT_TURNS_TOKEN_BUDGET)
        accumulated_tokens = 0
        keep_turns = 0

        for idx in range(len(turn_boundaries) - 1, -1, -1):
            turn_start = turn_boundaries[idx]
            turn_end = turn_boundaries[idx + 1] if idx + 1 < len(turn_boundaries) else len(messages)
            turn_chars = sum(
                len(json.dumps(messages[i].get("content", ""), ensure_ascii=False))
                for i in range(turn_start, turn_end)
            )
            turn_tokens = turn_chars / CHARS_PER_TOKEN
            if accumulated_tokens + turn_tokens > max_recent_tokens and keep_turns >= 1:
                break
            accumulated_tokens += turn_tokens
            keep_turns += 1
            if keep_turns >= MAX_RECENT_TURNS:
                break

        # Ensure we have something left to compress
        if keep_turns >= len(turn_boundaries):
            if self.verbose:
                print("[Context Compression] All turns fit in budget, skipping")
            return messages, 0, 0

        split_point = turn_boundaries[-keep_turns]

        if self.verbose:
            print(f"[Context Compression] Keeping {keep_turns} recent logical turns (~{int(accumulated_tokens)} tokens)")

        old_messages = messages[:split_point]
        recent_messages = messages[split_point:]

        # Split turn handling: if only 1 turn kept and it's oversized, try to split it
        turn_prefix_summary = None
        if keep_turns == 1:
            turn_start = turn_boundaries[-1]
            turn_end = len(messages)
            turn_chars = sum(
                len(json.dumps(messages[i].get("content", ""), ensure_ascii=False))
                for i in range(turn_start, turn_end)
            )
            turn_tokens = turn_chars / CHARS_PER_TOKEN
            if turn_tokens > max_recent_tokens * 0.5:
                # Try to find valid cut points within the turn (assistant message boundaries)
                cut_points = []
                for i in range(turn_start + 1, turn_end):
                    msg = messages[i]
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Don't cut right before a tool_result (keep tool_use/tool_result pairs)
                        if i + 1 < turn_end:
                            next_msg = messages[i + 1]
                            next_content = next_msg.get("content", "")
                            if isinstance(next_content, list) and any(
                                isinstance(b, dict) and b.get("type") == "tool_result"
                                for b in next_content
                            ):
                                continue
                        cut_points.append(i)

                if cut_points:
                    # Walk backwards from end, accumulating tokens until budget
                    best_cut = None
                    acc = 0
                    for i in range(turn_end - 1, turn_start, -1):
                        msg_chars = len(json.dumps(messages[i].get("content", ""), ensure_ascii=False))
                        acc += msg_chars / CHARS_PER_TOKEN
                        if acc > max_recent_tokens and i in cut_points:
                            best_cut = i
                            break
                        elif i in cut_points:
                            best_cut = i  # Keep updating as we go back

                    if best_cut and best_cut > turn_start:
                        # Split: prefix goes to summarization, suffix kept verbatim
                        turn_prefix = messages[turn_start:best_cut]
                        recent_messages = messages[best_cut:]

                        if self.verbose:
                            print(f"[Context Compression] Split oversized turn: prefix={len(turn_prefix)} msgs, suffix={len(recent_messages)} msgs")

                        # Generate turn prefix summary
                        prefix_serialized = self._serialize_messages_for_summary(turn_prefix)
                        try:
                            prefix_response = await self.client.acreate(
                                messages=[{
                                    "role": "user",
                                    "content": f"Summarize this turn prefix:\n\n{prefix_serialized}"
                                }],
                                system=TURN_PREFIX_SUMMARY_PROMPT,
                                max_tokens=2048,
                            )
                            turn_prefix_summary = prefix_response.text_content
                        except Exception:
                            turn_prefix_summary = prefix_serialized
                            if len(turn_prefix_summary) > 5000:
                                turn_prefix_summary = turn_prefix_summary[:2500] + "\n...\n" + turn_prefix_summary[-2500:]

        if self.verbose:
            print(f"\n[Context Compression] Compressing {len(old_messages)} old messages, keeping {len(recent_messages)} recent messages")

        # Extract file operations for cumulative tracking
        read_files, modified_files = _extract_file_operations(old_messages)

        # Check if this is an iterative compression (old_messages starts with previous summary)
        has_previous_summary = False
        previous_summary_text = ""
        if old_messages and old_messages[0].get("role") == "user":
            first_content = old_messages[0].get("content", "")
            if isinstance(first_content, str) and "<summary>" in first_content:
                has_previous_summary = True
                # Extract the summary text
                import re
                match = re.search(r"<summary>(.*?)</summary>", first_content, re.DOTALL)
                if match:
                    previous_summary_text = match.group(1).strip()
                # Merge file tracking from previous summary
                prev_read, prev_mod = _extract_previous_file_tracking(first_content)
                read_files |= prev_read
                modified_files |= prev_mod

        # Also extract from recent messages (for completeness of tracking)
        recent_read, recent_mod = _extract_file_operations(recent_messages)
        # Don't add recent to the summary tracking — those will be in the raw messages

        file_tracking = _build_file_tracking_section(read_files, modified_files)

        # Call LLM to generate a structured summary
        summary_input_tokens = 0
        summary_output_tokens = 0
        try:
            if has_previous_summary and previous_summary_text:
                # Iterative: update existing summary with new messages
                new_messages_only = [m for m in old_messages[1:] if not (
                    m.get("role") == "assistant" and isinstance(m.get("content"), list)
                    and any(isinstance(b, dict) and b.get("type") == "text"
                           and "I understand the context" in b.get("text", "")
                           for b in m["content"])
                )]
                serialized = self._serialize_messages_for_summary(new_messages_only) if new_messages_only else ""
                system_prompt = SUMMARY_UPDATE_PROMPT.format(
                    previous_summary=previous_summary_text,
                    file_tracking_section=file_tracking,
                )
                user_content = f"Please update the summary with these new conversation messages:\n\n{serialized}" if serialized else "No new messages to add."

                if self.verbose:
                    print("[Context Compression] Using iterative summary update (previous summary detected)")
            else:
                # First-time compression
                serialized = self._serialize_messages_for_summary(old_messages)
                system_prompt = SUMMARY_SYSTEM_PROMPT.format(file_tracking_section=file_tracking)
                user_content = f"Please summarize the following conversation:\n\n{serialized}"

            summary_response = await self.client.acreate(
                messages=[{
                    "role": "user",
                    "content": user_content,
                }],
                system=system_prompt,
                max_tokens=4096,
            )
            summary_input_tokens = summary_response.usage.input_tokens
            summary_output_tokens = summary_response.usage.output_tokens

            summary_text = summary_response.text_content

            if self.verbose:
                print(f"[Context Compression] Summary generated ({summary_input_tokens} in / {summary_output_tokens} out)")

        except Exception as e:
            # Fallback: use truncated serialized text as summary
            if self.verbose:
                print(f"[Context Compression] Summary API call failed: {e}, using fallback")
            if has_previous_summary and previous_summary_text:
                summary_text = previous_summary_text
            else:
                serialized = self._serialize_messages_for_summary(old_messages)
                summary_text = serialized
            if len(summary_text) > 10000:
                summary_text = summary_text[:5000] + "\n\n[... truncated ...]\n\n" + summary_text[-5000:]

        # Build the compression message.
        if "<summary>" not in summary_text:
            summary_text = f"<summary>\n{summary_text}\n</summary>"

        # Append file tracking to summary if not already present
        if file_tracking and "<read-files>" not in summary_text and "<modified-files>" not in summary_text:
            # Insert before </summary> closing tag
            summary_text = summary_text.rstrip()
            if summary_text.endswith("</summary>"):
                summary_text = summary_text[:-len("</summary>")] + f"\n\n{file_tracking}\n</summary>"
            else:
                summary_text += f"\n\n{file_tracking}"

        # Include turn prefix summary if we split an oversized turn
        if turn_prefix_summary:
            summary_text += f"\n\n[Recent turn prefix context]:\n{turn_prefix_summary}"

        compression_content = (
            "This session is being continued from a previous conversation that ran out of context. "
            "The summary below covers the earlier portion of the conversation.\n\n"
            f"{summary_text}\n\n"
            "Please continue the conversation from where we left off without asking the user any further questions. "
            "Continue with the last task that you were asked to work on."
        )

        # Build compressed messages: summary as first user message + recent messages
        compressed = [{"role": "user", "content": compression_content}]

        # If recent_messages starts with a user message, we need an assistant acknowledgment
        if recent_messages and recent_messages[0].get("role") == "user":
            compressed.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}]
            })

        compressed.extend(recent_messages)

        if self.verbose:
            print(f"[Context Compression] Messages reduced from {len(messages)} to {len(compressed)}")

        return compressed, summary_input_tokens, summary_output_tokens

    def _save_log(self, request: str, result: AgentResult) -> str:
        """Save the conversation log to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"agent_log_{timestamp}.json")

        log_data = {
            "timestamp": timestamp,
            "request": request,
            "model": self.model,
            "success": result.success,
            "total_turns": result.total_turns,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "answer": result.answer,
            "llm_calls": [asdict(call) for call in result.llm_calls],
            "steps": [asdict(step) for step in result.steps],
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return log_file

    async def run(
        self,
        request: str,
        conversation_history: Optional[List[Dict]] = None,
        image_contents: Optional[List[Dict]] = None,
        event_stream: Optional["EventStream"] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AgentResult:
        """
        Run the agent on a user request (async).

        When event_stream is provided, pushes StreamEvent objects for SSE streaming.
        When event_stream is None, runs silently and returns the result.

        Args:
            request: The user's request/task
            conversation_history: Optional list of previous messages for multi-turn conversation.
            image_contents: Optional list of Anthropic-format image blocks.
            event_stream: Optional EventStream for pushing streaming events.
            cancellation_event: Optional asyncio.Event — set to signal cancellation.

        Returns:
            AgentResult with the final answer and execution steps
        """
        streaming = event_stream is not None

        # Build the user message content (text or multipart with images)
        if image_contents:
            user_content = image_contents + [{"type": "text", "text": request}]
        else:
            user_content = request

        # Build messages from conversation history + new request
        if conversation_history:
            messages = list(conversation_history)  # Copy to avoid mutation
            messages.append({"role": "user", "content": user_content})
        else:
            messages = [{"role": "user", "content": user_content}]

        steps: List[AgentStep] = []
        llm_calls: List[LLMCall] = []
        used_skills: set = set()
        output_files: List[Dict] = []
        output_file_paths: set = set()  # For deduplication
        turns = 0
        total_input_tokens = 0
        total_output_tokens = 0
        last_input_tokens = 0

        while turns < self.max_turns:
            # Check cancellation
            if cancellation_event and cancellation_event.is_set():
                if self.verbose:
                    print("\n[Cancelled] Agent execution cancelled by user")
                break

            turns += 1

            # Context compression check
            if last_input_tokens > 0 and self._should_compress(last_input_tokens):
                # Flush memory before compression (fire-and-forget on a deep-copied snapshot,
                # so compression can safely mutate/replace `messages` concurrently)
                if self.agent_id:
                    try:
                        from app.services.memory_service import flush_memory
                        snapshot = copy.deepcopy(messages[-20:])
                        task = asyncio.create_task(flush_memory(
                            snapshot, self.agent_id,
                            self.model_provider, self.model,
                        ))
                        self._background_tasks.add(task)
                        def _on_flush_done(t, _tasks=self._background_tasks):
                            _tasks.discard(t)
                            if not t.cancelled():
                                exc = t.exception()
                                if exc and self.verbose:
                                    print(f"\n[Memory Flush] Background error: {exc}")
                        task.add_done_callback(_on_flush_done)
                    except Exception as flush_err:
                        if self.verbose:
                            print(f"\n[Memory Flush] Failed (non-fatal): {flush_err}")
                if self.verbose:
                    print(f"\n[Context Compression] Input tokens ({last_input_tokens}) exceeded threshold, compressing...")
                messages, s_in, s_out = await self._compress_messages(messages)
                total_input_tokens += s_in
                total_output_tokens += s_out

                if streaming:
                    await event_stream.push(StreamEvent(
                        event_type="context_compressed",
                        turn=turns,
                        data={
                            "previous_tokens": last_input_tokens,
                            "context_limit": self._get_context_limit(),
                        }
                    ))

            # Emit turn start event
            if streaming:
                await event_stream.push(StreamEvent(
                    event_type="turn_start",
                    turn=turns,
                    data={"max_turns": self.max_turns}
                ))

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Turn {turns}")
                print(f"{'='*50}")

            # Call LLM — streaming vs non-streaming
            response = None

            if streaming:
                # Streaming: yield text deltas, collect final response
                try:
                    async for resp in self.client.acreate_stream(
                        messages=messages,
                        system=self.system_prompt,
                        tools=self.tools,
                        max_tokens=min(self.client.max_output_tokens, 32000),
                    ):
                        # Check cancellation during streaming
                        if cancellation_event and cancellation_event.is_set():
                            break

                        if resp.is_delta:
                            for block in resp.content:
                                if isinstance(block, LLMTextBlock) and block.text:
                                    await event_stream.push(StreamEvent(
                                        event_type="text_delta",
                                        turn=turns,
                                        data={"text": block.text}
                                    ))
                        else:
                            response = resp
                except Exception as stream_err:
                    if self.verbose:
                        print(f"\n[Stream Error] {stream_err}")
                    if self._is_retryable_error(stream_err):
                        # Retry with exponential backoff (up to 3 attempts)
                        for attempt in range(1, 4):
                            delay = 2 ** attempt  # 2s, 4s, 8s
                            if self.verbose:
                                print(f"[Stream Retry] Attempt {attempt}/3 with non-streaming call after {delay}s...")
                            await asyncio.sleep(delay)
                            try:
                                response = await self.client.acreate(
                                    messages=messages,
                                    system=self.system_prompt,
                                    tools=self.tools,
                                    max_tokens=min(self.client.max_output_tokens, 32000),
                                )
                                break  # Success
                            except Exception as retry_err:
                                if self.verbose:
                                    print(f"[Stream Retry] Attempt {attempt}/3 failed: {retry_err}")
                                if attempt == 3 or not self._is_retryable_error(retry_err):
                                    break  # Give up
            else:
                # Non-streaming: single call with retry
                for attempt in range(4):  # 1 initial + 3 retries
                    try:
                        response = await self.client.acreate(
                            messages=messages,
                            system=self.system_prompt,
                            tools=self.tools,
                            max_tokens=min(self.client.max_output_tokens, 32000),
                        )
                        break
                    except Exception as call_err:
                        if attempt < 3 and self._is_retryable_error(call_err):
                            delay = 2 ** (attempt + 1)
                            if self.verbose:
                                print(f"\n[LLM Retry] Attempt {attempt+1}/3 failed: {call_err}")
                                print(f"[LLM Retry] Retrying after {delay}s...")
                            await asyncio.sleep(delay)
                        else:
                            if self.verbose:
                                print(f"\n[LLM Error] Non-retryable: {call_err}")
                            break

            # Check cancellation after LLM call
            if cancellation_event and cancellation_event.is_set():
                if self.verbose:
                    print("\n[Cancelled] Agent execution cancelled by user")
                break

            # Guard against missing response
            if response is None:
                error_msg = "LLM stream failed and retry was unsuccessful"
                if streaming:
                    await event_stream.push(StreamEvent(
                        event_type="error",
                        turn=turns,
                        data={"message": error_msg}
                    ))
                result = AgentResult(
                    success=False,
                    answer=error_msg,
                    steps=steps,
                    llm_calls=llm_calls,
                    total_turns=turns,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    error=error_msg,
                    skills_used=sorted(used_skills),
                    output_files=output_files,
                    final_messages=messages,
                )
                if streaming:
                    await event_stream.push(StreamEvent(
                        event_type="complete",
                        turn=turns,
                        data={
                            "success": False,
                            "answer": error_msg,
                            "total_turns": turns,
                            "total_input_tokens": total_input_tokens,
                            "total_output_tokens": total_output_tokens,
                            "skills_used": sorted(used_skills),
                            "output_files": output_files,
                        }
                    ))
                    await event_stream.close()
                return result

            # Record token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            last_input_tokens = input_tokens

            if self.verbose:
                print(f"Stop reason: {response.stop_reason}")
                print(f"Tokens: {input_tokens} in / {output_tokens} out")

            # Process response
            assistant_content = []
            tool_calls = []

            for block in response.content:
                if isinstance(block, LLMTextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                    if self.verbose:
                        print(f"Assistant: {block.text[:3000]}")
                        if len(block.text) > 3000:
                            print('...(truncated)')

                    steps.append(AgentStep(
                        role="assistant",
                        content=block.text,
                    ))

                elif isinstance(block, LLMToolCall):
                    tool_input = copy.deepcopy(block.input) if block.input else {}
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": tool_input,
                    })
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": tool_input,
                    })

                    if streaming:
                        await event_stream.push(StreamEvent(
                            event_type="tool_call",
                            turn=turns,
                            data={
                                "tool_name": block.name,
                                "tool_input": tool_input,
                            }
                        ))

            # Add assistant message
            messages.append({"role": "assistant", "content": assistant_content})

            # Record LLM call
            llm_calls.append(LLMCall(
                turn=turns,
                timestamp=datetime.now().isoformat(),
                model=self.model,
                request_messages=messages[:-1],  # Messages before this response
                response_content=assistant_content,
                stop_reason=response.stop_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ))

            # Handle max_tokens truncation — tool calls may be incomplete
            if response.stop_reason == "max_tokens" and tool_calls:
                if self.verbose:
                    print(f"\n[Warning] Response truncated (max_tokens). Discarding {len(tool_calls)} potentially incomplete tool call(s).")

                truncation_msg = (
                    "Your previous response was truncated because it exceeded the maximum output length. "
                    "The tool call(s) were incomplete and could not be executed. "
                    "Please try again with a shorter approach — for example, break the task into smaller steps "
                    "or generate less code at once."
                )
                tool_results = []
                for tool_call in tool_calls:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps({"error": truncation_msg}),
                        "is_error": True,
                    })
                    steps.append(AgentStep(
                        role="tool",
                        content=json.dumps({"error": "Response truncated (max_tokens) — tool call incomplete"}),
                        tool_name=tool_call["name"],
                        tool_input=tool_call["input"],
                    ))
                    if streaming:
                        await event_stream.push(StreamEvent(
                            event_type="tool_result",
                            turn=turns,
                            data={
                                "tool_name": tool_call["name"],
                                "tool_result": "Error: Response truncated (max_tokens). Tool call was incomplete.",
                                "tool_input": tool_call["input"],
                            }
                        ))
                messages.append({"role": "user", "content": tool_results})
                continue

            # If no tool calls, we're done — unless there's a steering message
            if response.stop_reason == "end_turn" and not tool_calls:
                # Check for steering message before finishing
                if event_stream and event_stream.has_injection():
                    steering_msg = event_stream.get_injection_nowait()
                    if steering_msg:
                        await event_stream.push(_make_steering_event(turns, steering_msg))
                        messages.append({
                            "role": "user",
                            "content": f"[User Steering Message]: {steering_msg}"
                        })
                        # Grant an extra turn so the LLM can process this steering
                        self.max_turns = max(self.max_turns, turns + 1)
                        continue  # Don't finish, loop back to LLM with steering message

                final_answer = response.text_content

                result = AgentResult(
                    success=True,
                    answer=final_answer,
                    steps=steps,
                    llm_calls=llm_calls,
                    total_turns=turns,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    skills_used=sorted(used_skills),
                    output_files=output_files,
                    final_messages=messages,
                )
                result.log_file = self._save_log(request, result)

                if streaming:
                    await event_stream.push(StreamEvent(
                        event_type="complete",
                        turn=turns,
                        data={
                            "success": True,
                            "answer": final_answer,
                            "total_turns": turns,
                            "total_input_tokens": total_input_tokens,
                            "total_output_tokens": total_output_tokens,
                            "skills_used": sorted(used_skills),
                            "output_files": output_files,
                            "final_messages": messages,
                        }
                    ))
                    await event_stream.close()

                return result

            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                # Check cancellation before each tool
                if cancellation_event and cancellation_event.is_set():
                    if self.verbose:
                        print("\n[Cancelled] Agent execution cancelled before tool execution")
                    break

                tool_name = tool_call["name"]

                if self.verbose:
                    print(f"\nTool: {tool_name}")
                    print(f"Input: {json.dumps(tool_call['input'], ensure_ascii=False)[:3000]}")

                # Special-case: ask_user ends the agent run and waits for user's next message
                if tool_name == "ask_user":
                    question = tool_call["input"].get("question", "")
                    ask_options = tool_call["input"].get("options")

                    # Append tool_result to messages so conversation history is coherent
                    tool_result = json.dumps({
                        "status": "waiting_for_user",
                        "question": question,
                        "options": ask_options,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": tool_result,
                    })
                    steps.append(AgentStep(
                        role="tool",
                        content=tool_result,
                        tool_name=tool_name,
                        tool_input=tool_call["input"],
                        tool_result=tool_result,
                    ))

                    if streaming and event_stream is not None:
                        # Streaming mode: emit ask_user event via SSE
                        prompt_id = str(uuid.uuid4())
                        await event_stream.push(StreamEvent(
                            event_type="ask_user",
                            turn=turns,
                            data={
                                "prompt_id": prompt_id,
                                "question": question,
                                "options": ask_options,
                            }
                        ))
                        await event_stream.push(StreamEvent(
                            event_type="tool_result",
                            turn=turns,
                            data={
                                "tool_name": tool_name,
                                "tool_input": tool_call["input"],
                                "tool_result": tool_result[:3000],
                            }
                        ))

                    # Add tool_results for any remaining tool_use blocks in the same
                    # assistant message so the conversation history stays valid (every
                    # tool_use must have a matching tool_result).
                    current_idx = tool_calls.index(tool_call)
                    for remaining in tool_calls[current_idx + 1:]:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": remaining["id"],
                            "content": json.dumps({"status": "skipped", "reason": "ask_user interrupted"}),
                        })

                    # Append tool_result to messages
                    messages.append({"role": "user", "content": tool_results})
                    # Append synthetic assistant message to maintain alternating roles in session
                    messages.append({
                        "role": "assistant",
                        "content": f"[Waiting for user response: {question}]",
                    })

                    # End the run successfully (both streaming and non-streaming)
                    final_answer = f"[Waiting for user response: {question}]"
                    result = AgentResult(
                        success=True,
                        answer=final_answer,
                        steps=steps,
                        llm_calls=llm_calls,
                        total_turns=turns,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        skills_used=sorted(used_skills),
                        output_files=output_files,
                        final_messages=messages,
                    )
                    result.log_file = self._save_log(request, result)

                    if streaming and event_stream is not None:
                        await event_stream.push(StreamEvent(
                            event_type="complete",
                            turn=turns,
                            data={
                                "success": True,
                                "answer": final_answer,
                                "total_turns": turns,
                                "total_input_tokens": total_input_tokens,
                                "total_output_tokens": total_output_tokens,
                                "skills_used": sorted(used_skills),
                                "output_files": output_files,
                                "final_messages": messages,
                            }
                        ))
                        await event_stream.close()
                    return result

                tool_result = await acall_tool(
                    tool_name,
                    tool_call["input"],
                    allowed_skills=self.allowed_skills,
                    tool_functions=self.tool_functions
                )

                # Track actually used skills
                if tool_name == "get_skill" and tool_call["input"].get("skill_name"):
                    used_skills.add(tool_call["input"]["skill_name"])

                if self.verbose:
                    print(f"Result: {tool_result[:3000]}")
                    if len(tool_result) > 3000:
                        print('...(truncated)')

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": tool_result,
                })

                steps.append(AgentStep(
                    role="tool",
                    content=tool_result,
                    tool_name=tool_name,
                    tool_input=tool_call["input"],
                    tool_result=tool_result,
                ))

                if streaming:
                    # Emit tool result event (truncate for SSE display)
                    await event_stream.push(StreamEvent(
                        event_type="tool_result",
                        turn=turns,
                        data={
                            "tool_name": tool_name,
                            "tool_input": tool_call["input"],
                            "tool_result": tool_result[:3000],
                        }
                    ))

                # Auto-detect output files from execute_code/bash/write
                if tool_name in ("execute_code", "bash", "write"):
                    try:
                        rd = json.loads(tool_result)
                        for nf in rd.get("new_files", []):
                            url = nf.get("download_url", "")
                            if url and url not in output_file_paths:
                                output_file_paths.add(url)
                                file_id = str(uuid.uuid4())
                                nf["file_id"] = file_id
                                output_files.append(nf)
                                if streaming:
                                    await event_stream.push(StreamEvent(
                                        event_type="output_file",
                                        turn=turns,
                                        data={
                                            "file_id": file_id,
                                            "filename": nf.get("filename"),
                                            "size": nf.get("size"),
                                            "content_type": nf.get("content_type"),
                                            "download_url": nf.get("download_url"),
                                        }
                                    ))
                    except (json.JSONDecodeError, TypeError):
                        pass

            # If cancelled during tool execution, break out
            if cancellation_event and cancellation_event.is_set():
                break

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

            # Check for steering message injection after tool results
            if event_stream and event_stream.has_injection():
                steering_msg = event_stream.get_injection_nowait()
                if steering_msg:
                    await event_stream.push(_make_steering_event(turns, steering_msg))
                    messages.append({
                        "role": "user",
                        "content": f"[User Steering Message]: {steering_msg}"
                    })
                    # Grant an extra turn so the LLM can process this steering
                    self.max_turns = max(self.max_turns, turns + 1)

            # Emit turn_complete checkpoint (all tool_use/tool_result pairs matched)
            if streaming:
                await event_stream.push(StreamEvent(
                    event_type="turn_complete",
                    turn=turns,
                    data={"messages_snapshot": messages.copy()}
                ))

        # Exited the loop — either max turns or cancellation

        # Determine if cancelled
        was_cancelled = cancellation_event and cancellation_event.is_set()

        if was_cancelled:
            final_answer = "Agent execution was cancelled."
            error_msg = "cancelled"
        else:
            # Max turns reached — give Claude one final turn to summarize progress
            if self.verbose:
                print(f"\n[Max Turns] Reached {self.max_turns} turns, requesting final summary...")

            messages.append({
                "role": "user",
                "content": (
                    f"You have reached the maximum number of turns ({self.max_turns}). "
                    "You cannot make any more tool calls. Please provide a final summary of "
                    "what you have accomplished so far and what remains to be done."
                ),
            })

            final_answer = "Max turns reached without completing the task."
            error_msg = "max_turns_exceeded"
            try:
                final_response = await self.client.acreate(
                    messages=messages,
                    system=self.system_prompt,
                    max_tokens=4096,
                )
                final_input = final_response.usage.input_tokens
                final_output = final_response.usage.output_tokens
                total_input_tokens += final_input
                total_output_tokens += final_output

                final_answer = final_response.text_content
                if final_answer:
                    steps.append(AgentStep(role="assistant", content=final_answer))
                    if streaming:
                        await event_stream.push(StreamEvent(
                            event_type="assistant",
                            turn=turns + 1,
                            data={"content": final_answer}
                        ))
            except Exception as e:
                if self.verbose:
                    print(f"[Max Turns] Final summary call failed: {e}")

        result = AgentResult(
            success=False,
            answer=final_answer,
            steps=steps,
            llm_calls=llm_calls,
            total_turns=turns,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            error=error_msg,
            skills_used=sorted(used_skills),
            output_files=output_files,
            final_messages=messages,
        )
        result.log_file = self._save_log(request, result)

        if streaming:
            await event_stream.push(StreamEvent(
                event_type="complete",
                turn=turns,
                data={
                    "success": False,
                    "answer": final_answer,
                    "total_turns": turns,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "error": error_msg,
                    "output_files": output_files,
                    "final_messages": messages,
                }
            ))
            await event_stream.close()

        return result

    def run_sync(
        self,
        request: str,
        conversation_history: Optional[List[Dict]] = None,
        image_contents: Optional[List[Dict]] = None,
    ) -> AgentResult:
        """Synchronous wrapper for background tasks (registry.py).

        Creates a new event loop and runs the async run() method.
        Must NOT be called from within an existing event loop.
        """
        return asyncio.run(self.run(request, conversation_history, image_contents))

    def cleanup(self) -> None:
        """
        Cleanup the agent's workspace.

        Call this when the agent request is complete to free resources.
        """
        if hasattr(self, 'workspace') and self.workspace:
            self.workspace.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup workspace."""
        self.cleanup()
        return False
