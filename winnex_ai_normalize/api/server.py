"""
winnex-ai-normalize — OpenAI-compatible embedding API.

Exposes the normalization service behind the OpenAI `/v1/embeddings`
contract, so any consumer (Liferay, Maestro, tracer, or a plain OpenAI
client) can get Madhava-ready vectors:

    POST /v1/embeddings  {"model": "...", "input": ["text", ...]}
        → {"data": [{"embedding": [..], "index": 0}], "model": "...", "usage": {...}}

Also exposes:
    GET  /v1/health            — provider availability
    GET  /v1/normalize/health  — alias (Liferay-friendly)

License: Business Source License 1.1 (BSL 1.1)
"""
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from winnex_ai_normalize.core.embedding import get_embedding_service
from winnex_ai_normalize.core.config import load_config

app = FastAPI(
    title="winnex-ai-normalize",
    version="1.0.0",
    description="Input normalization for the Madhava engine (OpenAI-compatible embeddings).",
)


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
