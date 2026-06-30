"""Simple progress tracker for agent operations."""

import threading
from typing import Optional


class ProgressTracker:
    _instance: Optional["ProgressTracker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._started = False
        self._verbose = False

    @classmethod
    def get_instance(cls) -> "ProgressTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(self, verbose: bool = False):
        self._started = True
        self._verbose = verbose

    def stop(self):
        self._started = False

    def update_status(self, agent_id: str, ticker: Optional[str], message: str, analysis=None):
        if self._verbose:
            prefix = f"[{agent_id}]"
            if ticker:
                prefix += f" [{ticker}]"
            print(f"  {prefix} {message}")


progress = ProgressTracker.get_instance()
