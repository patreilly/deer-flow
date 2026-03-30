# Snowflake Cortex Integration — Research & Development Plan

## What the Docs Actually Say

Source: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api

### Endpoints

| Purpose | Path |
|---|---|
| OpenAI-compatible chat | `POST https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions` |
| Anthropic Messages API (Claude only) | `POST https://<account>.snowflakecomputing.com/api/v2/cortex/v1/messages` |

### Authentication

Three supported methods — all use `Authorization: Bearer <token>`:

| Method | Header value for `X-Snowflake-Authorization-Token-Type` | Notes |
|---|---|---|
| Programmatic Access Token (PAT) | `PROGRAMMATIC_ACCESS_TOKEN` | Static token, no expiry concern, simplest to set up |
| Key-pair JWT | `KEYPAIR_JWT` | RS256 JWT from PEM private key; max lifetime 3600s |
| OAuth | `OAUTH` | OAuth token flow |

The `X-Snowflake-Authorization-Token-Type` header is optional but recommended — Snowflake uses it to route auth correctly.

For the Messages API (Claude), also required: `anthropic-version: 2023-06-01`

### Request Format

- `/v1/chat/completions` — follows **OpenAI Chat Completions spec**
- `/v1/messages` — follows **Anthropic Messages API spec**
- `max_completion_tokens` replaces deprecated `max_tokens`
- Fine-tuned models referenced as `database.schema.model`

### Snowflake-Specific Parameters

- `reasoning.effort` and `reasoning.max_tokens` — Claude thinking (not standard OpenAI)
- `cache_control` with `ephemeral` type (5-min or 1-hour TTL) — Claude only

### Supported Models (as of docs fetch, March 2026)

| Family | Models |
|---|---|
| Claude | claude-sonnet-4-6, claude-opus-4-6, claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5, earlier |
| OpenAI | gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat, gpt-4.1 |
| Open source | llama3.1-8b, llama3.1-70b, llama3.1-405b, mistral-7b, mistral-large, mistral-large2, deepseek-r1 |
| Snowflake | snowflake-llama-3.3-70b |

Regional availability varies; cross-region inference available for some models.

### Rate Limits (examples)

| Model | Tokens/min | Requests/min |
|---|---|---|
| claude-sonnet-4-5 | 600K | 600 |
| gpt-5-nano | 5M | 5K |
| llama3.1-405b | 100K | 100 |

HTTP 429 on breach; sliding window counter.

---

## What My Original Plan Had Wrong

| Area | My assumption | Correct |
|---|---|---|
| Endpoint path | `/api/v2/cortex/inference:complete` | `/api/v2/cortex/v1/chat/completions` |
| Claude path | same as above | `/api/v2/cortex/v1/messages` |
| Auth methods | JWT key-pair only | PAT, JWT key-pair, OAuth |
| Auth hint header | not needed | `X-Snowflake-Authorization-Token-Type` |
| `max_tokens` field | supported | deprecated; use `max_completion_tokens` |
| `ChatOpenAI` usable? | No (wrong path) | **Yes** — base_url ending in `/v1/` works |

---

## Development Plan

### Architecture

Because `/api/v2/cortex/v1/chat/completions` is a proper OpenAI-compatible path, we can use
`PatchedChatOpenAI` as the base class (no custom httpx transport needed). The client appends
`chat/completions` to the `base_url`, producing the correct URL naturally.

For Claude models on Snowflake, a second subclass targets `/v1/messages` using the Anthropic spec.

### Files to Create / Modify

#### 1. `credential_loader.py` — add `SnowflakeCredential` + JWT generation

```python
@dataclass
class SnowflakeCredential:
    account: str
    user: str
    token: str
    token_type: str          # "KEYPAIR_JWT" | "OAUTH" | "PROGRAMMATIC_ACCESS_TOKEN"
    expires_at: float        # Unix timestamp; 0 = unknown (PAT/OAuth, no auto-refresh)
    source: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at - 300  # 5-min buffer


def generate_snowflake_jwt(account, user, private_key_path, passphrase="", lifetime_seconds=3600) -> tuple[str, float]:
    """RS256 JWT using only `cryptography` (already a transitive dep via anthropic)."""
    # Load PEM private key
    # Derive public key fingerprint: SHA256 of DER-encoded SubjectPublicKeyInfo, base64-encoded
    # Account identifier: first segment, uppercased (strip region/cloud suffix)
    # Qualified user: ACCOUNT.USER (both uppercase)
    # iss = ACCOUNT.USER.SHA256:<fingerprint>
    # sub = ACCOUNT.USER
    # Build JWT header+payload, sign with PKCS1v15+SHA256, return token and expiry timestamp


def load_snowflake_credential() -> SnowflakeCredential | None:
    """Load from env vars in priority order:
    1. SNOWFLAKE_JWT_TOKEN (pre-generated JWT, no auto-refresh)
    2. SNOWFLAKE_PAT_TOKEN (PAT, no expiry)
    3. SNOWFLAKE_PRIVATE_KEY_PATH + SNOWFLAKE_USER + SNOWFLAKE_ACCOUNT (generate JWT)
    """
```

#### 2. `snowflake_provider.py` — `SnowflakeChatModel` (OpenAI-compat path)

Extends `PatchedChatOpenAI`. Handles all three auth methods, auto-refreshes key-pair JWT.

```yaml
# config.yaml example — key-pair JWT
- name: snowflake-llama3
  display_name: Snowflake Cortex (Llama 3.3)
  use: deerflow.models.snowflake_provider:SnowflakeChatModel
  model: snowflake-llama-3.3-70b
  snowflake_account: myorg-myaccount          # or set SNOWFLAKE_ACCOUNT
  snowflake_user: SERVICE_USER                # or set SNOWFLAKE_USER
  snowflake_private_key_path: $SNOWFLAKE_PRIVATE_KEY_PATH
  snowflake_private_key_passphrase: $SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
  max_completion_tokens: 4096
  temperature: 0.7

# config.yaml example — PAT (simplest)
- name: snowflake-gpt5
  display_name: Snowflake Cortex (GPT-5)
  use: deerflow.models.snowflake_provider:SnowflakeChatModel
  model: gpt-5
  snowflake_account: myorg-myaccount
  snowflake_pat_token: $SNOWFLAKE_PAT_TOKEN
  max_completion_tokens: 4096
```

Key implementation points:
- `base_url = https://{account}.snowflakecomputing.com/api/v2/cortex/v1/`
- `default_headers` includes `X-Snowflake-Authorization-Token-Type`
- `_generate` / `_agenerate` call `_refresh_token_if_needed()` before delegating to super
- `_get_request_payload` rewrites `max_tokens` → `max_completion_tokens`

#### 3. `snowflake_claude_provider.py` — `SnowflakeClaudeChatModel` (Anthropic Messages path)

Extends `ClaudeChatModel`. Overrides `base_url` to point at Snowflake's `/v1/messages` endpoint.
Adds `anthropic-version: 2023-06-01` header. Supports `reasoning` and `cache_control`.

```yaml
# config.yaml example
- name: snowflake-claude-sonnet
  display_name: Snowflake Cortex (Claude Sonnet 4.6)
  use: deerflow.models.snowflake_claude_provider:SnowflakeClaudeChatModel
  model: claude-sonnet-4-6
  snowflake_account: myorg-myaccount
  snowflake_pat_token: $SNOWFLAKE_PAT_TOKEN
  max_completion_tokens: 16384
  supports_vision: true
  supports_thinking: true
```

#### 4. `config.yaml` — add commented examples for both providers

#### 5. `pyproject.toml` — no new dependencies needed

`cryptography` is already a transitive dependency via `anthropic`.

---

## Open Questions Before Implementation

1. **JWT account identifier format** — docs say "account identifier"; need to confirm whether to
   strip region suffix (e.g. `myorg-myaccount.us-east-1.aws` → `MYORG-MYACCOUNT`) or use full form.
   The JWT `iss`/`sub` claims require exact formatting or Snowflake rejects the token.

2. **PAT expiry** — docs describe PAT as a "static token" but don't specify if it has a TTL.
   Treat as non-expiring for now; surface a clear error on 401.

3. **`max_completion_tokens` vs `max_tokens`** — need to confirm whether Snowflake silently accepts
   `max_tokens` anyway or strictly rejects it. If accepted, no payload rewriting needed.

4. **OAuth flow** — not implementing initially; accept a pre-obtained OAuth Bearer token the same
   way as PAT. Full OAuth refresh flow is out of scope for v1.
