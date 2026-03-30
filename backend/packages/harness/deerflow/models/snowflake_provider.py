"""Snowflake Cortex provider — OpenAI-compatible chat completions endpoint.

Connects to Snowflake's OpenAI-compatible Cortex REST API:
  POST https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions

Supports all three Snowflake auth methods:
  - Key-pair JWT  (KEYPAIR_JWT)            — auto-refreshes before expiry
  - Programmatic Access Token (PAT)         — static, no expiry
  - OAuth Bearer token                      — pre-obtained, treated like PAT

Auth detection priority (config fields take precedence over env vars):
  1. ``snowflake_private_key_path`` + ``snowflake_user`` → KEYPAIR_JWT
  2. ``snowflake_pat_token``                             → PROGRAMMATIC_ACCESS_TOKEN
  3. Env var fallback via ``load_snowflake_credential()``

Config example — key-pair JWT:

  - name: snowflake-llama3
    display_name: Snowflake Cortex (Llama 3.3 70B)
    use: deerflow.models.snowflake_provider:SnowflakeChatModel
    model: snowflake-llama-3.3-70b
    snowflake_account: myorg-myaccount
    snowflake_user: SERVICE_USER
    snowflake_private_key_path: $SNOWFLAKE_PRIVATE_KEY_PATH
    snowflake_private_key_passphrase: $SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
    max_completion_tokens: 4096
    temperature: 0.7

Config example — Programmatic Access Token (PAT):

  - name: snowflake-gpt5
    display_name: Snowflake Cortex (GPT-5)
    use: deerflow.models.snowflake_provider:SnowflakeChatModel
    model: gpt-5
    snowflake_account: myorg-myaccount
    snowflake_pat_token: $SNOWFLAKE_PAT_TOKEN
    max_completion_tokens: 4096
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import PrivateAttr, SecretStr

from deerflow.models.patched_openai import PatchedChatOpenAI

logger = logging.getLogger(__name__)

_CORTEX_CHAT_PATH = "/api/v2/cortex/v1/"
_DEFAULT_JWT_LIFETIME = 3600  # Snowflake maximum


class SnowflakeChatModel(PatchedChatOpenAI):
    """PatchedChatOpenAI targeting Snowflake Cortex with automatic JWT refresh.

    The OpenAI client appends ``chat/completions`` to ``base_url``, which
    resolves to the correct Snowflake endpoint:
      https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions

    All three Snowflake auth methods are supported.  Key-pair JWT tokens are
    automatically regenerated before expiry on every ``_generate`` /
    ``_agenerate`` call.
    """

    # Snowflake-specific config fields
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_private_key_path: str = ""
    snowflake_private_key_passphrase: str = ""
    snowflake_pat_token: str = ""
    snowflake_token_lifetime_seconds: int = _DEFAULT_JWT_LIFETIME

    # Private runtime state — not serialised
    _sf_account: str = PrivateAttr(default="")
    _sf_user: str = PrivateAttr(default="")
    _sf_key_path: str = PrivateAttr(default="")
    _sf_passphrase: str = PrivateAttr(default="")
    _sf_token_type: str = PrivateAttr(default="")
    _sf_token_expires_at: float = PrivateAttr(default=0.0)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Resolve Snowflake credentials and configure the OpenAI-compat endpoint."""
        from deerflow.models.credential_loader import (
            SnowflakeCredential,
            generate_snowflake_jwt,
            load_snowflake_credential,
        )

        account = (self.snowflake_account or os.getenv("SNOWFLAKE_ACCOUNT", "")).strip()
        if not account:
            raise ValueError(
                "snowflake_account is required for SnowflakeChatModel "
                "(or set the SNOWFLAKE_ACCOUNT environment variable)."
            )

        self._sf_account = account

        # --- Resolve token and type ---
        token: str = ""
        token_type: str = ""
        expires_at: float = 0.0

        key_path = (self.snowflake_private_key_path or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")).strip()
        user = (self.snowflake_user or os.getenv("SNOWFLAKE_USER", "")).strip()
        passphrase = (self.snowflake_private_key_passphrase or os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")).strip()
        pat = (self.snowflake_pat_token or os.getenv("SNOWFLAKE_PAT_TOKEN", "")).strip()

        if key_path and user:
            token, expires_at = generate_snowflake_jwt(
                account, user, key_path, passphrase, self.snowflake_token_lifetime_seconds
            )
            token_type = "KEYPAIR_JWT"
            self._sf_user = user
            self._sf_key_path = key_path
            self._sf_passphrase = passphrase
            logger.info(f"Generated Snowflake KEYPAIR_JWT for {user}@{account}")
        elif pat:
            token = pat
            token_type = "PROGRAMMATIC_ACCESS_TOKEN"
            logger.info(f"Using Snowflake PAT for account {account}")
        else:
            # Fall back to env-var credential loader
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
            # Key-path not available from env-loaded pre-generated token, so no refresh.

        self._sf_token_type = token_type
        self._sf_token_expires_at = expires_at

        # Wire up ChatOpenAI base class fields
        self.openai_api_key = SecretStr(token)
        self.openai_api_base = f"https://{account}.snowflakecomputing.com{_CORTEX_CHAT_PATH}"
        self.default_headers = {
            **(self.default_headers or {}),
            "X-Snowflake-Authorization-Token-Type": token_type,
        }

        super().model_post_init(__context)

    # ------------------------------------------------------------------
    # JWT refresh
    # ------------------------------------------------------------------

    def _refresh_token_if_needed(self) -> None:
        """Regenerate the JWT when it is within the refresh buffer window.

        No-op for PAT / OAuth tokens (``_sf_token_expires_at == 0``).
        No-op when key material is unavailable (pre-generated token fallback).
        """
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
        self.openai_api_key = SecretStr(new_token)

        # Patch the live OpenAI client instances so in-flight and future requests use the new token
        for client in (getattr(self, "client", None), getattr(self, "async_client", None)):
            if client is not None and hasattr(client, "api_key"):
                client.api_key = new_token

        logger.info(f"Refreshed Snowflake JWT for {self._sf_user}@{self._sf_account}")

    # ------------------------------------------------------------------
    # _generate / _agenerate — refresh token before each call
    # ------------------------------------------------------------------

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        self._refresh_token_if_needed()
        return super()._generate(messages, stop=stop, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        self._refresh_token_if_needed()
        return await super()._agenerate(messages, stop=stop, **kwargs)
