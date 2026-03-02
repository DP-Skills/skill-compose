"""Configuration management for Skill Composer."""
import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file into os.environ at startup
# Priority: config/.env (Docker volume, Settings API writes here) > ./.env (local dev fallback)
_config_env = Path(os.environ.get("CONFIG_DIR", "./config")) / ".env"
if _config_env.exists():
    load_dotenv(_config_env, override=True)
else:
    load_dotenv(override=True)

# Resolve env_file path for Pydantic Settings
_env_file = str(_config_env) if _config_env.exists() else ".env"


class Settings(BaseSettings):
    """Application settings"""
    model_config = SettingsConfigDict(env_file=_env_file, extra="ignore")

    # API Server
    host: str = "127.0.0.1"
    port: int = 62610
    debug: bool = False

    # LLM API Keys
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""  # For Gemini

    # Default model configuration
    default_model_provider: str = "kimi"
    default_model_name: str = "kimi-k2.5"

    # Legacy: claude_model for backward compatibility
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Agent configuration
    agent_max_turns: int = 60

    # Paths (can be overridden via environment variables for Docker)
    project_dir: str = "."
    skills_dir: str = ""  # SKILLS_DIR env var, defaults to custom_skills_dir if empty
    custom_skills_dir: str = "./skills"  # Fallback for skills_dir
    data_dir: str = "./data"  # DATA_DIR env var
    logs_dir: str = "./logs"  # LOGS_DIR env var
    upload_dir: str = "./uploads"  # UPLOADS_DIR env var
    config_dir: str = "./config"  # CONFIG_DIR env var
    backups_dir: str = "./backups"  # BACKUPS_DIR env var

    # File upload
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    # Code execution
    code_execution_timeout: int = 300  # seconds
    code_max_output_chars: int = 10000
    code_executor_type: str = "jupyter"  # "jupyter" or "simple"

    # Database (Phase 1: Skill Registry)
    # Can be overridden, but default uses data_dir
    database_url: str = ""
    database_echo: bool = False  # Log SQL statements

    # Authentication
    jwt_secret_key: str = ""  # Auto-generated from database_url if empty
    jwt_access_token_expire_hours: int = 24
    jwt_refresh_token_expire_days: int = 7
    auth_enabled: bool = False  # Set true to enable auth (login required)

    # Scheduler
    scheduler_enabled: bool = True
    scheduler_poll_interval: int = 30  # seconds between poll cycles

    # Channel adapters
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    telegram_bot_token: str = ""

    # Embedding (for memory vector search)
    embedding_provider: str = "openai"  # EMBEDDING_PROVIDER
    embedding_model: str = "text-embedding-3-small"  # EMBEDDING_MODEL
    embedding_dimensions: int = 1536  # EMBEDDING_DIMENSIONS
    embedding_api_key: str = ""  # EMBEDDING_API_KEY (fallback: OPENAI_API_KEY)
    embedding_base_url: str = ""  # EMBEDDING_BASE_URL (custom endpoints)

    # Memory
    memory_dir: str = "./memory"  # MEMORY_DIR

    # Meta skills (internal use only, not selectable by users)
    meta_skills: list[str] = ["skill-creator", "skill-updater", "skill-evolver", "skill-finder", "trace-qa", "skills-planner", "planning-with-files", "mcp-builder"]

    @property
    def effective_skills_dir(self) -> str:
        """Get effective skills directory (SKILLS_DIR or custom_skills_dir)"""
        return self.skills_dir if self.skills_dir else self.custom_skills_dir

    @property
    def effective_database_url(self) -> str:
        """Get effective database URL (DATABASE_URL or default PostgreSQL)"""
        if self.database_url:
            return self.database_url
        return "postgresql+asyncpg://skills:skills123@localhost:62620/skills_api"

    @property
    def effective_jwt_secret(self) -> str:
        """Get effective JWT secret.

        Priority:
        1. Explicit JWT_SECRET_KEY from env/config
        2. Auto-generated secret persisted to config/.env
        3. Random fallback (non-persistent, logs warning)
        """
        if self.jwt_secret_key:
            return self.jwt_secret_key
        # Try to load or generate a persistent secret
        return _get_or_create_jwt_secret(self.config_dir)

    @property
    def effective_config_path(self) -> str:
        """Get effective MCP config path"""
        return f"{self.config_dir}/mcp.json"


def _get_or_create_jwt_secret(config_dir: str) -> str:
    """Load or generate a persistent JWT secret in config/.env.

    Uses file locking to prevent race conditions when multiple workers start.
    """
    import secrets
    import fcntl

    env_path = Path(config_dir) / ".env"
    lock_path = Path(config_dir) / ".jwt_secret.lock"

    def _read_secret_from_env() -> str | None:
        if not env_path.exists():
            return None
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("JWT_SECRET_KEY=") and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return val
        except Exception:
            pass
        return None

    # Fast path: read without lock
    existing = _read_secret_from_env()
    if existing:
        return existing

    # Slow path: acquire lock, re-check, generate if needed
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                # Re-check after acquiring lock (another worker may have written)
                existing = _read_secret_from_env()
                if existing:
                    return existing

                new_secret = secrets.token_hex(32)
                if env_path.exists():
                    content = env_path.read_text(encoding="utf-8")
                    if "JWT_SECRET_KEY=" in content:
                        lines = content.splitlines()
                        new_lines = []
                        for line in lines:
                            if line.strip().startswith("JWT_SECRET_KEY="):
                                new_lines.append(f"JWT_SECRET_KEY={new_secret}")
                            else:
                                new_lines.append(line)
                        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    else:
                        with open(env_path, "a", encoding="utf-8") as f:
                            f.write(f"\nJWT_SECRET_KEY={new_secret}\n")
                else:
                    env_path.write_text(f"JWT_SECRET_KEY={new_secret}\n", encoding="utf-8")
                print(f"Auto-generated JWT secret key and saved to {env_path}")
                return new_secret
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: Could not persist JWT secret to {env_path}: {e}")
        # Last resort fallback
        return secrets.token_hex(32)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Module-level settings instance for convenience
settings = get_settings()


def _get_env_file_path() -> Path:
    """Get the .env file path (config/.env or ./.env)."""
    config_env = Path(os.environ.get("CONFIG_DIR", "./config")) / ".env"
    if config_env.exists():
        return config_env
    project_env = Path(".env")
    if project_env.exists():
        return project_env
    return config_env


def read_env_value(key: str) -> str:
    """Read a single key's value from the .env file.

    Always reads from disk so all uvicorn workers and callsites
    see the latest value written by the Settings API.
    Falls back to os.environ if not found on disk (e.g. in tests
    or when vars are injected via docker-compose environment).
    """
    env_path = _get_env_file_path()
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == key:
                        return v.strip()
        except Exception:
            pass
    # Fallback to os.environ (tests, docker-compose environment injection)
    return os.environ.get(key, "")


def read_env_all() -> dict[str, str]:
    """Read all key-value pairs from the .env file.

    Returns a dict of all env vars. Always reads from disk.
    """
    env_path = _get_env_file_path()
    if not env_path.exists():
        return {}
    result = {}
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


def get_search_dirs(project_dir: str = ".") -> list[Path]:
    """
    Get all searchable skill directories in priority order.

    Priority: custom > project .agent > global .agent > project .claude > global .claude
    """
    home = Path.home()
    project = Path(project_dir).resolve()
    settings = get_settings()

    dirs = [
        Path(settings.effective_skills_dir).resolve(),  # 0. Custom skills dir
        project / ".agent" / "skills",    # 1. Project universal
        home / ".agent" / "skills",        # 2. Global universal
        project / ".claude" / "skills",   # 3. Project claude
        home / ".claude" / "skills",       # 4. Global claude
    ]
    return dirs
