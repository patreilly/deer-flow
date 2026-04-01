"""Snowflake Cortex provider — Anthropic Messages API endpoint (Claude models only).

Connects to Snowflake's Anthropic-spec Cortex REST endpoint:
  POST https://<account>.snowflakecomputing.com/api/v2/cortex/v1/messages

This provider is for Claude models hosted on Snowflake Cortex. It inherits
prompt caching, thinking budget, and retry logic from ``ClaudeChatModel`` while
redirecting all requests to Snowflake instead of api.anthropic.com.

Supports the same three Snowflake auth methods as ``SnowflakeChatModel``:
  - Key-pair JWT  (KEYPAIR_JWT)            — auto-refreshes before expiry
  - Programmatic Access Token (PAT)         — static, no expiry
  - OAuth Bearer token                      — pre-obtained, treated like PAT

Note: Snowflake's ``/v1/messages`` endpoint requires ``anthropic-version: 2023-06-01``.
This header is injected automatically.

Config example — PAT (simplest):

  - name: snowflake-claude-sonnet
    display_name: Snowflake Cortex (Claude Sonnet 4.6)
    use: deerflow.models.snowflake_claude_provider:SnowflakeClaudeChatModel
    model: claude-sonnet-4-6
    snowflake_account: myorg-myaccount
    snowflake_pat_token: $SNOWFLAKE_PAT_TOKEN
    max_tokens: 16384
    supports_vision: true
    supports_thinking: true
    when_thinking_enabled:
      thinking:
        type: enabled

Config example — key-pair JWT:

  - name: snowflake-claude-haiku
    display_name: Snowflake Cortex (Claude Haiku 4.5)
    use: deerflow.models.snowflake_claude_provider:SnowflakeClaudeChatModel
    model: claude-haiku-4-5
    snowflake_account: myorg-myaccount
    snowflake_user: SERVICE_USER
    snowflake_private_key_path: $SNOWFLAKE_PRIVATE_KEY_PATH
    snowflake_private_key_passphrase: $SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
    max_tokens: 8192
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import PrivateAttr, SecretStr

from deerflow.models.claude_provider import ClaudeChatModel

logger = logging.getLogger(__name__)

_CORTEX_MESSAGES_PATH = "/api/v2/cortex"
_DEFAULT_JWT_LIFETIME = 3600  # Snowflake maximum


class SnowflakeClaudeChatModel(ClaudeChatModel):
    """ClaudeChatModel re-routed to Snowflake Cortex ``/api/v2/cortex/v1/messages``.

    Inherits prompt caching, thinking budget, retry logic, and OAuth billing
    handling from ``ClaudeChatModel``.  Snowflake-specific auth and endpoint
    configuration are layered on top in ``model_post_init``.

    Key differences from the standard ``ClaudeChatModel``:
    - ``base_url`` points at Snowflake, not api.anthropic.com
    - ``anthropic-version: 2023-06-01`` header is required and injected
    - ``X-Snowflake-Authorization-Token-Type`` header is injected
    - JWT tokens are auto-refreshed before expiry
    - OAuth billing header injection is disabled (not applicable on Snowflake)
    """

    # Snowflake-specific config fields
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_private_key_path: str = ""
    snowflake_private_key_passphrase: str = ""
    snowflake_pat_token: str = ""
    snowflake_token_lifetime_seconds: int = _DEFAULT_JWT_LIFETIME

    # Private runtime state
    _sf_account: str = PrivateAttr(default="")
    _sf_user: str = PrivateAttr(default="")
    _sf_key_path: str = PrivateAttr(default="")
    _sf_passphrase: str = PrivateAttr(default="")
    _sf_token_type: str = PrivateAttr(default="")
    _sf_token_expires_at: float = PrivateAttr(default=0.0)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Resolve Snowflake credentials and re-point the Anthropic client at Cortex."""
        from deerflow.models.credential_loader import (
            SnowflakeCredential,
            generate_snowflake_jwt,
            load_snowflake_credential,
        )

        account = (self.snowflake_account or os.getenv("SNOWFLAKE_ACCOUNT", "")).strip()
        if not account:
            raise ValueError(
                "snowflake_account is required for SnowflakeClaudeChatModel "
                "(or set the SNOWFLAKE_ACCOUNT environment variable)."
            )

        self._sf_account = account

        key_path = (self.snowflake_private_key_path or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")).strip()
        user = (self.snowflake_user or os.getenv("SNOWFLAKE_USER", "")).strip()
        passphrase = (self.snowflake_private_key_passphrase or os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")).strip()
        pat = (self.snowflake_pat_token or os.getenv("SNOWFLAKE_PAT_TOKEN", "")).strip()

        token: str = ""
        token_type: str = ""
        expires_at: float = 0.0

        if key_path and user:
            token, expires_at = generate_snowflake_jwt(
                account, user, key_path, passphrase, self.snowflake_token_lifetime_seconds
            )
            token_type = "KEYPAIR_JWT"
            self._sf_user = user
            self._sf_key_path = key_path
            self._sf_passphrase = passphrase
            logger.info(f"Generated Snowflake KEYPAIR_JWT for {user}@{account} (Claude endpoint)")
        elif pat:
            token = pat
            token_type = "PROGRAMMATIC_ACCESS_TOKEN"
            logger.info(f"Using Snowflake PAT for account {account} (Claude endpoint)")
        else:
            cred: SnowflakeCredential | None = load_snowflake_credential()
            if cred is None:
                raise ValueError(
                    "No Snowflake credentials found. Provide one of:\n"
                    "  - snowflake_private_key_path + snowflake_user (key-pair JWT)\n"
                    "  - snowflake_pat_token (Programmatic Access Token)\n"
                    "  - SNOWFLAKE_PAT_TOKEN, SNOWFLAKE_JWT_TOKEN env vars"
                )
            token = cred.token
            token_type = cred.token_type
            expires_at = cred.expires_at
            self._sf_user = cred.user

        self._sf_token_type = token_type
        self._sf_token_expires_at = expires_at

        # Set the Anthropic API key to our Snowflake token so the SDK sends
        # Authorization: Bearer <token> — identical to standard Claude auth.
        self.anthropic_api_key = SecretStr(token)

        # Re-point the Anthropic client at Snowflake's /v1/messages endpoint.
        # The Anthropic SDK respects base_url and appends the path automatically.
        snowflake_base_url = f"https://{account}.snowflakecomputing.com{_CORTEX_MESSAGES_PATH}"
        self.anthropic_api_url = snowflake_base_url

        # Required headers for Snowflake Cortex Messages endpoint.
        # Snowflake expects Authorization: Bearer, not the x-api-key that the
        # Anthropic SDK sends by default.
        self.default_headers = {
            **(self.default_headers or {}),
            "authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "X-Snowflake-Authorization-Token-Type": token_type,
        }

        # Disable ClaudeChatModel's OAuth billing header injection — it is
        # Anthropic-specific and would cause a 400 on Snowflake.
        self._is_oauth = False

        # Snowflake Cortex limits cache_control blocks to 4; ClaudeChatModel's
        # _apply_prompt_caching can generate 5–6 blocks (system + messages +
        # tools), causing 400 "A maximum of 4 blocks with cache_control may be
        # provided."  Disable prompt caching entirely for Snowflake — the same
        # approach already used for OAuth tokens in ClaudeChatModel.
        self.enable_prompt_caching = False

        super().model_post_init(__context)

    # ------------------------------------------------------------------
    # JWT refresh
    # ------------------------------------------------------------------

    def _refresh_token_if_needed(self) -> None:
        """Regenerate the JWT when it is within the refresh buffer window."""
        if self._sf_token_expires_at <= 0:
            return
        if time.time() < self._sf_token_expires_at:
            return
        if not (self._sf_key_path and self._sf_user):
            logger.warning(
                "Snowflake JWT has expired but key material is not available for refresh. "
                "Restart the server with valid SNOWFLAKE_PRIVATE_KEY_PATH to re-enable auto-refresh."
            )
            return

        from deerflow.models.credential_loader import generate_snowflake_jwt

        new_token, expires_at = generate_snowflake_jwt(
            self._sf_account,
            self._sf_user,
            self._sf_key_path,
            self._sf_passphrase,
            self.snowflake_token_lifetime_seconds,
        )
        self._sf_token_expires_at = expires_at
        self.anthropic_api_key = SecretStr(new_token)

        # Keep the Authorization: Bearer header in sync with the new token
        if isinstance(self.default_headers, dict):
            self.default_headers["authorization"] = f"Bearer {new_token}"

        # Patch live clients so the new token is used immediately
        for client in (getattr(self, "_client", None), getattr(self, "_async_client", None)):
            if client is not None and hasattr(client, "api_key"):
                client.api_key = new_token

        logger.info(f"Refreshed Snowflake JWT for {self._sf_user}@{self._sf_account} (Claude endpoint)")

    # ------------------------------------------------------------------
    # _generate / _agenerate — refresh token before each call
    # ------------------------------------------------------------------

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        self._refresh_token_if_needed()
        return super()._generate(messages, stop=stop, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        self._refresh_token_if_needed()
        return await super()._agenerate(messages, stop=stop, **kwargs)
