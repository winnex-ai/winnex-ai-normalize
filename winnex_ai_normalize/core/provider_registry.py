"""
winnex-ai-normalize — secure provider registry (CRUD + persistence).

A safe, integration-friendly way to register embedding providers. Design:

  - Persistence: a local JSON file (WINNEX_AI_NORMALIZE_PROVIDERS_FILE),
    read at startup and updated on write. Atomic write (temp + rename).
  - Secret safety: the registry NEVER stores API keys. A provider references
    its key by an ENV VAR name (`api_key_env`); the actual key lives in the
    environment / KMS. `to_dict()` masks everything.
  - Auth: the CRUD endpoints require an admin API key
    (WINNEX_AI_NORMALIZE_ADMIN_KEY). Without it, registration is refused.
  - Integration: the same REST contract works for Liferay, Maestro or any
    consumer — POST /v1/providers {name, base_url, model, api_key_env, ...}.

License: Business Source License 1.1 (BSL 1.1)
"""
import json
import logging
import os
import threading
from typing import Dict, List, Optional

from .config import ProviderConfig, NormalizeConfig

logger = logging.getLogger("winnex-ai-normalize.registry")


class ProviderRegistry:
    """Persistent, secure registry of embedding providers."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get(
            "WINNEX_AI_NORMALIZE_PROVIDERS_FILE",
            "/var/lib/winnex-ai-normalize/providers.json")
        self._lock = threading.Lock()
        self._providers: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._providers = data.get("providers", {})
            logger.info(f"registry: loaded {len(self._providers)} providers from {self.path}")
        except Exception as e:
            logger.error(f"registry: failed to load {self.path}: {e}")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"providers": self._providers}, f, indent=2)
        os.replace(tmp, self.path)   # atomic write

    # -- public API -----------------------------------------------------
    def upsert(self, provider: dict) -> ProviderConfig:
        """Add or update a provider. `api_key` (if present) is moved to the
        environment and replaced by an `api_key_env` reference — the key is
        NEVER persisted to disk."""
        name = provider.get("name", "").strip()
        if not name:
            raise ValueError("provider.name is required")
        # Secret handling: if the caller passes an actual api_key, store it in
        # the environment (never the registry file) and reference it by env name.
        api_key = provider.pop("api_key", None)
        api_key_env = provider.get("api_key_env", "")
        if api_key:
            env_name = api_key_env or f"WINNEX_PROVIDER_KEY_{name.upper()}"
            os.environ[env_name] = api_key
            provider["api_key_env"] = env_name
        cfg = ProviderConfig(name=name, **{
            k: v for k, v in provider.items() if k != "name" and v is not None
        })
        with self._lock:
            self._providers[name] = self._to_persistable(cfg)
            self._save()
        logger.info(f"registry: upserted provider '{name}'")
        return cfg

    def get(self, name: str) -> Optional[ProviderConfig]:
        with self._lock:
            p = self._providers.get(name)
        return ProviderConfig(name=name, **p) if p else None

    def list(self) -> List[dict]:
        """List providers, with secrets masked."""
        with self._lock:
            return [self._to_persistable(ProviderConfig(name=n, **p))
                    for n, p in self._providers.items()]

    def delete(self, name: str) -> bool:
        with self._lock:
            if name not in self._providers:
                return False
            del self._providers[name]
            self._save()
        logger.info(f"registry: deleted provider '{name}'")
        return True

    def to_config(self) -> NormalizeConfig:
        """Build a NormalizeConfig from the registry (for the service)."""
        cfg = NormalizeConfig()
        providers = []
        with self._lock:
            items = list(self._providers.items())
        for name, p in items:
            providers.append(ProviderConfig(name=name, **{
                k: v for k, v in p.items() if k != "name"}))
        if providers:
            cfg.providers = providers
            cfg.provider_order = [p.name for p in sorted(
                providers, key=lambda p: p.priority)]
        return cfg

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _to_persistable(cfg: ProviderConfig) -> dict:
        """ProviderConfig → dict for storage/display (api_key_env reference only)."""
        return {
            "type": cfg.type,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "api_key_env": cfg.api_key_env,
            "dim": cfg.dim,
            "timeout": cfg.timeout,
            "priority": cfg.priority,
            "enabled": cfg.enabled,
        }


# Singleton
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def require_admin_key(authorization: str) -> None:
    """Fail-closed admin check for the CRUD endpoints.

    The admin key comes from WINNEX_AI_NORMALIZE_ADMIN_KEY (env / KMS).
    If not configured, the endpoints are DISABLED (fail-closed) — safer than
    leaving provider registration open.
    """
    import hmac
    expected = os.environ.get("WINNEX_AI_NORMALIZE_ADMIN_KEY", "")
    if not expected:
        raise PermissionError(
            "Admin key not configured (WINNEX_AI_NORMALIZE_ADMIN_KEY). "
            "Provider registration is disabled (fail-closed).")
    token = authorization.replace("Bearer ", "").strip() if authorization else ""
    if not token or not hmac.compare_digest(token, expected):
        raise PermissionError("Invalid admin key")
