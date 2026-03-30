"""Auto-load credentials from Claude Code CLI, Codex CLI, and Snowflake.

Implements three credential strategies:
  1. Claude Code OAuth token from explicit env vars or an exported credentials file
     - Uses Authorization: Bearer header (NOT x-api-key)
     - Requires anthropic-beta: oauth-2025-04-20,claude-code-20250219
     - Supports $CLAUDE_CODE_OAUTH_TOKEN, $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR, and $ANTHROPIC_AUTH_TOKEN
     - Override path with $CLAUDE_CODE_CREDENTIALS_PATH
  2. Codex CLI token from ~/.codex/auth.json
     - Uses chatgpt.com/backend-api/codex/responses endpoint
     - Supports both legacy top-level tokens and current nested tokens shape
     - Override path with $CODEX_AUTH_PATH
  3. Snowflake credentials for Cortex REST API
     - Supports three token types: KEYPAIR_JWT, PROGRAMMATIC_ACCESS_TOKEN, OAUTH
     - All use Authorization: Bearer <token> + X-Snowflake-Authorization-Token-Type header
     - Key-pair JWT is generated from a PEM private key (PKCS#8); max lifetime 3600s
     - Env vars: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_PATH,
       SNOWFLAKE_PRIVATE_KEY_PASSPHRASE (optional), SNOWFLAKE_PAT_TOKEN, SNOWFLAKE_JWT_TOKEN
"""

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Required beta headers for Claude Code OAuth tokens
OAUTH_ANTHROPIC_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14"


def is_oauth_token(token: str) -> bool:
    """Check if a token is a Claude Code OAuth token (not a standard API key)."""
    return isinstance(token, str) and "sk-ant-oat" in token


@dataclass
class ClaudeCodeCredential:
    """Claude Code CLI OAuth credential."""

    access_token: str
    refresh_token: str = ""
    expires_at: int = 0
    source: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() * 1000 > self.expires_at - 60_000  # 1 min buffer


@dataclass
class CodexCliCredential:
    """Codex CLI credential."""

    access_token: str
    account_id: str = ""
    source: str = ""


def _resolve_credential_path(env_var: str, default_relative_path: str) -> Path:
    configured_path = os.getenv(env_var)
    if configured_path:
        return Path(configured_path).expanduser()
    return _home_dir() / default_relative_path


def _home_dir() -> Path:
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser()
    return Path.home()


def _load_json_file(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        logger.debug(f"{label} not found: {path}")
        return None
    if path.is_dir():
        logger.warning(f"{label} path is a directory, expected a file: {path}")
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {label}: {e}")
        return None


def _read_secret_from_file_descriptor(env_var: str) -> str | None:
    fd_value = os.getenv(env_var)
    if not fd_value:
        return None

    try:
        fd = int(fd_value)
    except ValueError:
        logger.warning(f"{env_var} must be an integer file descriptor, got: {fd_value}")
        return None

    try:
        secret = os.read(fd, 1024 * 1024).decode().strip()
    except OSError as e:
        logger.warning(f"Failed to read {env_var}: {e}")
        return None

    return secret or None


def _credential_from_direct_token(access_token: str, source: str) -> ClaudeCodeCredential | None:
    token = access_token.strip()
    if not token:
        return None
    return ClaudeCodeCredential(access_token=token, source=source)


def _iter_claude_code_credential_paths() -> list[Path]:
    paths: list[Path] = []
    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    if override_path:
        paths.append(Path(override_path).expanduser())

    default_path = _home_dir() / ".claude/.credentials.json"
    if not paths or paths[-1] != default_path:
        paths.append(default_path)

    return paths


def _extract_claude_code_credential(data: dict[str, Any], source: str) -> ClaudeCodeCredential | None:
    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken", "")
    if not access_token:
        logger.debug("Claude Code credentials container exists but no accessToken found")
        return None

    cred = ClaudeCodeCredential(
        access_token=access_token,
        refresh_token=oauth.get("refreshToken", ""),
        expires_at=oauth.get("expiresAt", 0),
        source=source,
    )

    if cred.is_expired:
        logger.warning("Claude Code OAuth token is expired. Run 'claude' to refresh.")
        return None

    return cred


def load_claude_code_credential() -> ClaudeCodeCredential | None:
    """Load OAuth credential from explicit Claude Code handoff sources.

    Lookup order:
      1. $CLAUDE_CODE_OAUTH_TOKEN or $ANTHROPIC_AUTH_TOKEN
      2. $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
      3. $CLAUDE_CODE_CREDENTIALS_PATH
      4. ~/.claude/.credentials.json

    Exported credentials files contain:
    {
      "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-...",
        "refreshToken": "sk-ant-ort01-...",
        "expiresAt": 1773430695128,
        "scopes": ["user:inference", ...],
        ...
      }
    }
    """
    direct_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if direct_token:
        cred = _credential_from_direct_token(direct_token, "claude-cli-env")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from environment")
        return cred

    fd_token = _read_secret_from_file_descriptor("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
    if fd_token:
        cred = _credential_from_direct_token(fd_token, "claude-cli-fd")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from file descriptor")
        return cred

    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    override_path_obj = Path(override_path).expanduser() if override_path else None
    for cred_path in _iter_claude_code_credential_paths():
        data = _load_json_file(cred_path, "Claude Code credentials")
        if data is None:
            continue
        cred = _extract_claude_code_credential(data, "claude-cli-file")
        if cred:
            source_label = "override path" if override_path_obj is not None and cred_path == override_path_obj else "plaintext file"
            logger.info(f"Loaded Claude Code OAuth credential from {source_label} (expires_at={cred.expires_at})")
            return cred

    return None


# ---------------------------------------------------------------------------
# Snowflake
# ---------------------------------------------------------------------------

#: Refresh JWT this many seconds before it expires to avoid mid-request failure.
_SNOWFLAKE_JWT_REFRESH_BUFFER = 300


@dataclass
class SnowflakeCredential:
    """Snowflake Cortex REST API credential.

    Covers all three token types accepted by the API:
      - KEYPAIR_JWT          — RS256 JWT generated from a PEM private key
      - PROGRAMMATIC_ACCESS_TOKEN — static PAT; no expiry, no auto-refresh
      - OAUTH                — pre-obtained OAuth Bearer token
    """

    account: str
    token: str
    token_type: str  # "KEYPAIR_JWT" | "PROGRAMMATIC_ACCESS_TOKEN" | "OAUTH"
    user: str = ""
    expires_at: float = 0.0  # Unix timestamp; 0 = unknown / non-expiring
    source: str = ""

    @property
    def is_expired(self) -> bool:
        """True if the token is within the refresh buffer window."""
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at - _SNOWFLAKE_JWT_REFRESH_BUFFER


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def generate_snowflake_jwt(
    account: str,
    user: str,
    private_key_path: str,
    passphrase: str = "",
    lifetime_seconds: int = 3600,
) -> tuple[str, float]:
    """Generate a Snowflake RS256 key-pair JWT.

    Uses only the ``cryptography`` package (already a transitive dependency via
    ``anthropic``).  No ``PyJWT`` required.

    Returns ``(token, expires_at)`` where ``expires_at`` is a Unix timestamp.

    Raises ``ValueError`` if the key file is missing or unreadable.
    Raises ``ImportError`` if ``cryptography`` is not installed.

    Account identifier formatting:
      Snowflake expects the *first segment* of the account identifier, uppercased.
      E.g. ``myorg-myaccount.us-east-1.aws`` → ``MYORG-MYACCOUNT``.
    """
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:
        raise ImportError(
            "The 'cryptography' package is required for Snowflake key-pair JWT generation. "
            "Install it with: pip install cryptography"
        ) from exc

    key_path = Path(private_key_path).expanduser()
    if not key_path.exists():
        raise ValueError(f"Snowflake private key not found: {key_path}")

    key_data = key_path.read_bytes()
    password = passphrase.encode() if passphrase else None

    try:
        private_key = serialization.load_pem_private_key(key_data, password=password, backend=default_backend())
    except Exception as exc:
        raise ValueError(f"Failed to load Snowflake private key from {key_path}: {exc}") from exc

    # Derive public key fingerprint: SHA256 of DER-encoded SubjectPublicKeyInfo, base64-encoded
    pub_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(pub_der).digest()).decode()

    # Snowflake requires the first segment of the account identifier, uppercased
    account_id = account.split(".")[0].upper()
    qualified_user = f"{account_id}.{user.upper()}"

    now = int(time.time())
    payload = {
        "iss": f"{qualified_user}.{fingerprint}",
        "sub": qualified_user,
        "iat": now,
        "exp": now + lifetime_seconds,
    }

    header = {"alg": "RS256", "typ": "JWT"}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()

    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    token = f"{header_b64}.{payload_b64}.{_b64url(signature)}"

    return token, float(now + lifetime_seconds)


def load_snowflake_credential() -> SnowflakeCredential | None:
    """Load a Snowflake credential from environment variables.

    Lookup order:
      1. ``SNOWFLAKE_PRIVATE_KEY_PATH`` + ``SNOWFLAKE_USER`` + ``SNOWFLAKE_ACCOUNT``
         → generates a fresh KEYPAIR_JWT (auto-refresh capable)
      2. ``SNOWFLAKE_PAT_TOKEN`` + ``SNOWFLAKE_ACCOUNT``
         → static Programmatic Access Token (no expiry)
      3. ``SNOWFLAKE_JWT_TOKEN`` + ``SNOWFLAKE_ACCOUNT``
         → pre-generated JWT (no auto-refresh; will expire in ~1 h)

    Returns ``None`` if no Snowflake credentials are found.
    """
    account = os.getenv("SNOWFLAKE_ACCOUNT", "").strip()
    if not account:
        return None

    # 1. Key-pair JWT (preferred — enables auto-refresh)
    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    user = os.getenv("SNOWFLAKE_USER", "").strip()
    if key_path and user:
        passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
        try:
            token, expires_at = generate_snowflake_jwt(account, user, key_path, passphrase)
            logger.info(f"Generated Snowflake KEYPAIR_JWT for {user}@{account}")
            return SnowflakeCredential(
                account=account,
                user=user,
                token=token,
                token_type="KEYPAIR_JWT",
                expires_at=expires_at,
                source="env-keypair",
            )
        except Exception as exc:
            logger.warning(f"Failed to generate Snowflake JWT: {exc}")

    # 2. Programmatic Access Token
    pat = os.getenv("SNOWFLAKE_PAT_TOKEN", "").strip()
    if pat:
        logger.info(f"Loaded Snowflake PAT for account {account}")
        return SnowflakeCredential(
            account=account,
            token=pat,
            token_type="PROGRAMMATIC_ACCESS_TOKEN",
            source="env-pat",
        )

    # 3. Pre-generated JWT (no auto-refresh)
    jwt_token = os.getenv("SNOWFLAKE_JWT_TOKEN", "").strip()
    if jwt_token:
        logger.warning(
            "Using pre-generated SNOWFLAKE_JWT_TOKEN — no auto-refresh. "
            "Token will expire in ~1 hour."
        )
        return SnowflakeCredential(
            account=account,
            token=jwt_token,
            token_type="KEYPAIR_JWT",
            source="env-jwt-pregenerated",
        )

    return None


def load_codex_cli_credential() -> CodexCliCredential | None:
    """Load credential from Codex CLI (~/.codex/auth.json)."""
    cred_path = _resolve_credential_path("CODEX_AUTH_PATH", ".codex/auth.json")
    data = _load_json_file(cred_path, "Codex CLI credentials")
    if data is None:
        return None
    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}

    access_token = data.get("access_token") or data.get("token") or tokens.get("access_token", "")
    account_id = data.get("account_id") or tokens.get("account_id", "")
    if not access_token:
        logger.debug("Codex CLI credentials file exists but no token found")
        return None

    logger.info("Loaded Codex CLI credential")
    return CodexCliCredential(
        access_token=access_token,
        account_id=account_id,
        source="codex-cli",
    )
