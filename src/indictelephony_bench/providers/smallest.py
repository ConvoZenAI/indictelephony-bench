"""Smallest.ai Pulse over its public REST API.

Concurrency here is deliberately conservative and separately configurable: the
endpoint rate-limits harder than the others, and a run that trips the limit
produces failed rows, which are scored as deletions rather than retried forever.
"""
import asyncio
import fcntl
import os
import tempfile
import time
from pathlib import Path
from typing import TypeVar

import httpx

from indictelephony_bench.config import get_settings
from indictelephony_bench.providers.base import ASRProvider, ProviderResult, RowContext
from indictelephony_bench.providers.utils import (
    RetriableProviderError,
    failed_result,
    is_retryable_http_status,
    run_with_retry,
    success_result,
)

SMALLEST_STT_ENDPOINT = "https://api.smallest.ai/waves/v1/stt/"
DEFAULT_MULTILINGUAL_LANGUAGE = "multi-indic"
_SMALLEST_THROTTLE_LOCK = asyncio.Lock()
_LAST_SMALLEST_CALL_TS = 0.0

_THROTTLE_FILE = Path(tempfile.gettempdir()) / "smallest_stt_throttle.lock"


def _wait_global_slot(min_interval: float) -> None:
    """Block until `min_interval` has passed since any process's last call."""
    deadline = time.time() + 120.0
    while True:
        # os.open with O_RDWR|O_CREAT, never "a+": in append mode every write
        # goes to EOF regardless of seek(), so seek(0)+truncate()+write()
        # concatenated timestamps ("17875...0451787...904") and the next float()
        # raised, failing the row. os.pwrite writes at an explicit offset.
        fd = os.open(_THROTTLE_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.pread(fd, 64, 0).decode("utf-8", "ignore").strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0          # corrupt value: treat as "no recent call"
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


LANGUAGE_TAG_TO_SMALLEST = {
    "as": "as",
    "bn": "bn",
    "en": "en",
    "gu": "gu",
    "hi": "hi",
    "kn": "kn",
    "ml": "ml",
    "mr": "mr",
    "pa": "pa",
    "ta": "ta",
    "te": "te",
    "ur": "ur",
}


T = TypeVar("T")


def _chunked(items: list[T], size: int) -> list[list[T]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


class SmallestProvider(ASRProvider):
    provider_name = "smallest"

    @staticmethod
    def _effective_fallback_language() -> str:
        settings = get_settings()
        configured = (settings.smallest_fallback_language or "").strip()
        return configured or DEFAULT_MULTILINGUAL_LANGUAGE

    @staticmethod
    def _effective_model_name() -> str:
        settings = get_settings()
        model = (settings.smallest_model or "").strip()
        return model or "pulse"

    async def _throttle(self, min_interval_seconds: float) -> None:
        """Rate-limit across PROCESSES, not just within one.

        The in-process lock alone is not enough here: the benchmark runs one
        process per language, so ten processes each honouring a 2 s interval
        put a request on the wire every 0.2 s and the API answers 429. The
        timestamp of the last call is therefore kept in a small file that every
        process shares, guarded by an exclusive flock, so the interval means the
        same thing whether one language is running or ten.
        """
        if min_interval_seconds <= 0:
            return
        async with _SMALLEST_THROTTLE_LOCK:
            await asyncio.to_thread(_wait_global_slot, min_interval_seconds)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float:
        value = response.headers.get("Retry-After")
        if not value:
            return 0.0
        try:
            return max(0.0, float(value))
        except ValueError:
            return 0.0

    @staticmethod
    def _resolve_language(language: str | None) -> str:
        settings = get_settings()
        model = (settings.smallest_model or "").strip().lower()
        # Pulse Pro currently supports only English language code.
        if model == "pulse-pro":
            return "en"

        if not language:
            return SmallestProvider._effective_fallback_language()

        lang = language.strip().lower()
        if not lang:
            return SmallestProvider._effective_fallback_language()

        # Dataset tags like bn-en or en-mr are code-mix and should route to
        # a multilingual aggregator.
        if "-" in lang:
            return SmallestProvider._effective_fallback_language()

        return LANGUAGE_TAG_TO_SMALLEST.get(lang, SmallestProvider._effective_fallback_language())

    async def _request_transcript(
        self,
        client: httpx.AsyncClient,
        audio_path: str,
        language: str,
    ) -> str:
        settings = get_settings()
        path = Path(audio_path)
        audio_bytes = path.read_bytes()
        await self._throttle(max(0.0, float(settings.smallest_min_interval_seconds)))
        response = await client.post(
            SMALLEST_STT_ENDPOINT,
            params={
                "model": self._effective_model_name(),
                "language": language,
            },
            headers={
                "Authorization": f"Bearer {settings.smallest_api_key}",
                "Content-Type": "application/octet-stream",
            },
            content=audio_bytes,
        )
        if response.status_code == 429:
            # Honour Retry-After when present; otherwise back off a fixed amount
            # rather than returning instantly, which would spend the next retry
            # attempt on a request the server is still refusing.
            retry_after = self._retry_after_seconds(response)
            await asyncio.sleep(min(retry_after, 30.0) if retry_after > 0 else 5.0)
            raise RetriableProviderError("Retryable status 429")
        if is_retryable_http_status(response.status_code):
            raise RetriableProviderError(f"Retryable status {response.status_code}")
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("transcription", "")).strip()

    async def transcribe_batch(
        self,
        items: list[tuple[str, RowContext]],
    ) -> dict[str, ProviderResult]:
        settings = get_settings()
        now = time.time()
        if not settings.smallest_api_key:
            return {audio_path: failed_result(now, "Missing SMALLEST_API_KEY") for audio_path, _ in items}

        if not items:
            return {}

        per_item_start = {audio_path: time.time() for audio_path, _ in items}
        missing_paths = [audio_path for audio_path, _ in items if not Path(audio_path).exists()]
        if missing_paths:
            return {
                audio_path: failed_result(
                    per_item_start.get(audio_path, time.time()),
                    f"Audio file not found: {audio_path}",
                )
                for audio_path, _ in items
            }

        timeout = httpx.Timeout(settings.provider_timeout_seconds)
        max_parallel = max(
            1,
            min(
                int(settings.max_concurrency),
                int(settings.smallest_batch_size),
                int(settings.smallest_max_parallel_requests),
            ),
        )
        semaphore = asyncio.Semaphore(max_parallel)
        results: dict[str, ProviderResult] = {}

        async with httpx.AsyncClient(timeout=timeout) as client:

            async def _transcribe_one(audio_path: str, row: RowContext) -> tuple[str, ProviderResult]:
                start = per_item_start.get(audio_path, time.time())

                async def _call() -> str:
                    # Resolve from the BASE language, not the raw tag. The tag for
                    # a code-mixed row looks like "en-ta", which _resolve_language
                    # routes to multi-indic - and multi-indic returns Tamil audio
                    # transcribed in Devanagari. Sending "ta" instead returns
                    # correct Tamil script and a far better transcript. 51.7% of
                    # this corpus is code-mixed, so this was the majority path.
                    ctx = row or RowContext()
                    resolved_language = self._resolve_language(
                        ctx.base_language or ctx.language_tag
                    )
                    try:
                        return await self._request_transcript(client, audio_path, resolved_language)
                    except httpx.HTTPStatusError as exc:
                        fallback_language = self._effective_fallback_language()
                        if exc.response.status_code == 400 and resolved_language != fallback_language:
                            return await self._request_transcript(
                                client,
                                audio_path,
                                fallback_language,
                            )
                        raise

                async with semaphore:
                    try:
                        prediction = await run_with_retry(_call)
                        return audio_path, success_result(prediction, start)
                    except Exception as exc:
                        return audio_path, failed_result(start, str(exc))

            configured_batch_size = max(1, int(settings.smallest_batch_size))
            for batch in _chunked(items, configured_batch_size):
                payloads = await asyncio.gather(
                    *[_transcribe_one(audio_path, row) for audio_path, row in batch]
                )
                for audio_path, provider_result in payloads:
                    results[audio_path] = provider_result

        for audio_path, _ in items:
            if audio_path not in results:
                results[audio_path] = failed_result(
                    per_item_start.get(audio_path, time.time()),
                    "Missing Smallest batch result.",
                )

        return results

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        context: str | None = None,
        row: RowContext | None = None,
    ) -> ProviderResult:
        row = row or RowContext(language_tag=language or "")
        batch_results = await self.transcribe_batch([(audio_path, row)])
        return batch_results.get(audio_path, failed_result(time.time(), "Missing Smallest result."))
