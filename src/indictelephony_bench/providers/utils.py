"""Helpers shared by the HTTP-based providers: timing, retries, result shapes."""
import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from indictelephony_bench.config import get_settings
from indictelephony_bench.providers.base import SUCCESS, ProviderResult

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


class RetriableProviderError(Exception):
    """Raised when provider call should be retried."""


# httpx is only a dependency of the providers that call REST APIs directly.
# Akshara goes through the ConvoZen SDK, which uses requests, so importing httpx
# unconditionally here would break `pip install indictelephony-bench[akshara]`.
try:
    import httpx

    _TRANSIENT: tuple[type[BaseException], ...] = (
        RetriableProviderError, httpx.TimeoutException, httpx.NetworkError)
except ImportError:  # pragma: no cover - depends on which extras are installed
    _TRANSIENT = (RetriableProviderError,)


def success_result(prediction: str, start_time: float) -> ProviderResult:
    return {
        "prediction": prediction or "",
        "latency_ms": (time.time() - start_time) * 1000.0,
        "status": SUCCESS,
        "error": None,
    }


def failed_result(start_time: float, error: str) -> ProviderResult:
    return {
        "prediction": "",
        "latency_ms": (time.time() - start_time) * 1000.0,
        "status": "failed",
        "error": error,
    }


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


async def run_with_retry(func: Callable[[], Awaitable[T]]) -> T:
    settings = get_settings()
    max_attempts = max(1, settings.provider_retry_attempts)

    attempt = 0
    while True:
        attempt += 1
        try:
            return await func()
        except _TRANSIENT as exc:
            if attempt >= max_attempts:
                raise exc
            await asyncio.sleep(min(2 ** (attempt - 1), 8))


def wait_global_slot(lock_file: "Path", min_interval: float) -> None:
    """Block until `min_interval` seconds have passed since ANY process's last call.

    The benchmark runs one process per language, so an in-process throttle lets
    nine processes send nine times the configured rate. The last-call timestamp
    lives in a lock file shared by every process, guarded by an exclusive flock.
    Same design as the Smallest throttle. Blocking: call via asyncio.to_thread.
    """
    import fcntl
    import os
    import time

    if min_interval <= 0:
        return
    # Many processes may queue behind one slot; never give up before they all could run.
    deadline = time.time() + max(120.0, min_interval * 30)
    while True:
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.pread(fd, 64, 0).decode("utf-8", "ignore").strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.time()
            wait = min_interval - (now - last)
            if wait <= 0:
                os.ftruncate(fd, 0)
                os.pwrite(fd, f"{now:.6f}".encode(), 0)
                return
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        if time.time() > deadline:
            return
        time.sleep(min(wait, min_interval))
