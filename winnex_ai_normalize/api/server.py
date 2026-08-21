"""
winnex-ai-normalize — OpenAI-compatible embedding API + secure provider CRUD.

Exposes the normalization service behind the OpenAI `/v1/embeddings`
contract, so any consumer (Liferay, Maestro, tracer, or a plain OpenAI
client) can get Madhava-ready vectors:

    POST /v1/embeddings  {"model": "...", "input": ["text", ...]}
        → {"data": [{"embedding": [..], "index": 0}], "model": "...", "usage": {...}}

Provider registration (secure, fail-closed, integration-friendly):
    GET    /v1/providers                       — list (masked)
    POST   /v1/providers  {name, base_url, model, api_key_env, priority}
                                               — register/update (admin key)
    DELETE /v1/providers/{name}                — remove (admin key)

  - The admin key comes from WINNEX_AI_NORMALIZE_ADMIN_KEY. If not set,
    the CRUD endpoints are DISABLED (fail-closed).
  - API keys are NEVER persisted: a provider references its key by an ENV
    var name (`api_key_env`); the actual key lives in the environment/KMS.
  - The same REST contract works for Liferay, the Maestro or any consumer.

License: Business Source License 1.1 (BSL 1.1)
"""
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from winnex_ai_normalize.core.embedding import get_embedding_service
from winnex_ai_normalize.core.config import load_config

app = FastAPI(
    title="winnex-ai-normalize",
    version="1.0.0",
    description="Input normalization for the Madhava engine (OpenAI-compatible embeddings + provider registry).",
)


# ---------------------------------------------------------------------------
# Provider CRUD (secure)
# ---------------------------------------------------------------------------
class ProviderIn(BaseModel):
    name: str
    type: str = "openai_compat"
    model: str = ""
    base_url: str = ""
    api_key: str = ""           # actual key (moved to env, never persisted)
    api_key_env: str = ""       # env var reference (preferred, no key in payload)
    dim: int = 0
    timeout: float = 20.0
    priority: int = 10
    enabled: bool = True


class ProviderUpdate(BaseModel):
    type: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    dim: Optional[int] = None
    timeout: Optional[float] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None


def _admin_required(authorization: str = Header(default="", alias="Authorization")):
    from winnex_ai_normalize.core.provider_registry import require_admin_key
    try:
        require_admin_key(authorization)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/v1/providers")
def list_providers(authorization: str = Header(default="", alias="Authorization")):
    """List registered providers (secrets masked)."""
    _admin_required(authorization)
    from winnex_ai_normalize.core.provider_registry import get_registry
    return {"providers": get_registry().list()}


@app.post("/v1/providers")
def upsert_provider(provider: ProviderIn,
                    authorization: str = Header(default="", alias="Authorization")):
    """Register or update an embedding provider (admin key required)."""
    _admin_required(authorization)
    from winnex_ai_normalize.core.provider_registry import get_registry
    try:
        cfg = get_registry().upsert(provider.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"status": "registered", "provider": cfg.name}


@app.delete("/v1/providers/{name}")
def delete_provider(name: str,
                    authorization: str = Header(default="", alias="Authorization")):
    """Remove a registered provider (admin key required)."""
    _admin_required(authorization)
    from winnex_ai_normalize.core.provider_registry import get_registry
    ok = get_registry().delete(name)
    if not ok:
        raise HTTPException(404, f"provider {name} not found")
    return {"status": "deleted", "provider": name}


class EmbeddingsRequest(BaseModel):
    model: str = ""
    input: List[str]


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    """OpenAI-compatible embeddings endpoint (Madhava-ready vectors)."""
    service = get_embedding_service()
    try:
        vecs = service.embed_texts(req.input)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    data = [
        {"embedding": vecs[i].tolist(), "index": i}
        for i in range(len(vecs))
    ]
    return {
        "object": "list",
        "data": data,
        "model": req.model or "winnex-ai-normalize",
        "usage": {"prompt_tokens": sum(len(t) // 4 for t in req.input),
                  "total_tokens": sum(len(t) // 4 for t in req.input)},
    }


@app.get("/v1/health")
@app.get("/v1/normalize/health")
def health():
    """Provider availability (fail loudly if none reachable)."""
    cfg = load_config()
    svc = get_embedding_service()
    status = svc.check_available()
    return {
        "status": "ok" if status.get("available") else "degraded",
        "service": "winnex-ai-normalize",
        "default_dim": cfg.default_dim,
        "default_provider": cfg.default_provider,
        "provider_order": cfg.provider_order,
        "providers": status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8102")))
