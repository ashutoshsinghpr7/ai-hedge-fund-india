"""Pickle-based cache for data provider responses.

Caches price data, financial metrics, line items, insider trades, and news
to avoid repeated API calls during analysis sessions.
"""

import json
import logging
import os
import pickle
import time
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600  # 1 hour


class DataCache:
    """Simple pickle-backed cache with TTL support."""

    def __init__(self, cache_dir: str = "./data/cache", ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self._locks: dict[str, Lock] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def _get_lock(self, key: str) -> Lock:
        if key not in self._locks:
            self._locks[key] = Lock()
        return self._locks[key]

    def _cache_path(self, prefix: str, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return os.path.join(self.cache_dir, f"{prefix}_{safe_key}.pkl")

    def _write_json_meta(self, path: str, meta: dict):
        meta_path = path.replace(".pkl", ".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def _read_json_meta(self, path: str) -> Optional[dict]:
        meta_path = path.replace(".pkl", ".json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _is_expired(self, meta: Optional[dict]) -> bool:
        if meta is None:
            return True
        if self.ttl_seconds <= 0:
            return False
        cached_at = meta.get("cached_at", 0)
        return (time.time() - cached_at) > self.ttl_seconds

    def get(self, prefix: str, key: str) -> Optional[dict]:
        path = self._cache_path(prefix, key)
        lock = self._get_lock(key)
        with lock:
            if not os.path.exists(path):
                return None
            meta = self._read_json_meta(path)
            if self._is_expired(meta):
                return None
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Failed to read cache %s: %s", key, e)
                return None

    def set(self, prefix: str, key: str, data: dict):
        path = self._cache_path(prefix, key)
        lock = self._get_lock(key)
        with lock:
            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                self._write_json_meta(path, {"cached_at": time.time(), "key": key, "prefix": prefix})
            except Exception as e:
                logger.warning("Failed to write cache %s: %s", key, e)

    def get_prices(self, key: str) -> Optional[list]:
        return self.get("prices", key)

    def set_prices(self, key: str, data: list):
        self.set("prices", key, data)

    def get_financial_metrics(self, key: str) -> Optional[list]:
        return self.get("metrics", key)

    def set_financial_metrics(self, key: str, data: list):
        self.set("metrics", key, data)

    def get_insider_trades(self, key: str) -> Optional[list]:
        return self.get("insiders", key)

    def set_insider_trades(self, key: str, data: list):
        self.set("insiders", key, data)

    def get_company_news(self, key: str) -> Optional[list]:
        return self.get("news", key)

    def set_company_news(self, key: str, data: list):
        self.set("news", key, data)


_cache_instance: Optional[DataCache] = None


def get_cache(cache_dir: str = "./data/cache") -> DataCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache(cache_dir)
    return _cache_instance
