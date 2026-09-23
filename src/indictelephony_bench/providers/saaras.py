"""Sarvam Saaras, batch or realtime.

The batch API is the default because it is what the published baseline used and
it survives rate limits better over tens of thousands of short files. Requests
carry the code-mix mode, since most of this corpus is code-mixed.
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path

import httpx
from sarvamai import SarvamAI

from indictelephony_bench.config import get_settings
from indictelephony_bench.providers.base import ASRProvider, ProviderResult, RowContext
from indictelephony_bench.providers.lang_codes import (
    resolve_saaras_language,
)
from indictelephony_bench.providers.utils import (
    RetriableProviderError,
    failed_result,
    run_with_retry,
    success_result,
    wait_global_slot,
)

_SAARAS_THROTTLE_LOCK = asyncio.Lock()
# Shared by every benchmark process (one per language), so SAARAS_MIN_INTERVAL_SECONDS
# is the rate for the whole run, not per language.
_SAARAS_THROTTLE_FILE = Path(tempfile.gettempdir()) / "saaras_stt_throttle.lock"
_LAST_SAARAS_CALL_TS = 0.0
_SAARAS_STATUS_ENDPOINT = "https://api.sarvam.ai/speech-to-text/job/v1/{job_id}/status"
_SAARAS_MAX_FILES_PER_BATCH = 20


class SaarasProvider(ASRProvider):
    provider_name = "saaras"

    def _extract_job_id(self, job: object) -> str:
        for attr in ("job_id", "id", "_job_id"):
            value = getattr(job, attr, None)
            if value:
                return str(value)
        getter = getattr(job, "get_job_id", None)
        if callable(getter):
            value = getter()
            if value:
                return str(value)
        return ""

    def _poll_job_status(self, settings: object, job_id: str) -> None:
        if not job_id:
            raise RuntimeError("Missing job_id for Saaras status polling.")

        poll_interval_seconds = max(
            0.005,
            float(getattr(settings, "saaras_status_poll_interval_ms", 5)) / 1000.0,
        )
        timeout_seconds = max(1.0, float(getattr(settings, "provider_timeout_seconds", 120.0)))
        url = _SAARAS_STATUS_ENDPOINT.format(job_id=job_id)
        headers = {"api-subscription-key": str(settings.saaras_api_key)}

        while True:
            response = httpx.get(url, headers=headers, timeout=timeout_seconds)
            if response.status_code in {429, 500, 502, 503}:
                raise RetriableProviderError(f"Saaras status poll retryable status {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            state = str(payload.get("job_state", "")).strip().lower()
            if state == "completed":
                return
            if state == "failed":
                error_message = str(payload.get("error_message", "")).strip()
                raise RuntimeError(f"Saaras batch job failed: {error_message or job_id}")
            time.sleep(poll_interval_seconds)

    def _read_batch_outputs(self, output_dir: Path, audio_paths: list[str]) -> dict[str, str]:
        transcripts: dict[str, str] = {}
        for audio_path in audio_paths:
            input_name = Path(audio_path).name
            json_file = output_dir / f"{input_name}.json"
            transcript = ""
            if json_file.exists():
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                transcript = str(payload.get("transcript", ""))
            transcripts[audio_path] = transcript
        return transcripts

    def _run_batch_sync(
        self,
        client: SarvamAI,
        audio_paths: list[str],
        settings: object,
        language_code: str,
    ) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # `model` is always passed: the SDK's create_job default is an older
            # model ('saarika:v2.5'), so omitting it would silently change systems.
            job = client.speech_to_text_job.create_job(
                model=settings.saaras_model,
                mode=settings.saaras_mode or "codemix",
                language_code=language_code,
                with_diarization=False,
            )
            try:
                job.upload_files(file_paths=audio_paths)
            except Exception as exc:
                raise RetriableProviderError(f"Saaras upload failed: {exc}") from exc

            # upload_files can return True while the server has not registered
            # the files, and start() then fails with "files list must not be
            # empty" - taking the whole batch with it. Observed on 10-file
            # batches, on a different language each run, so it is a race and not
            # a property of the audio. Retry the upload rather than lose 10 rows.
            # Retry start() on ANY failure, re-uploading first. Matching on the
            # "files list must not be empty" text was too narrow - the same race
            # surfaces with other messages, and each miss costs a whole batch
            # (20 rows). A start() that has already succeeded is not retried, so
            # an unconditional retry here cannot double-submit.
            started = False
            for attempt in range(5):
                try:
                    job.start()
                    started = True
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RetriableProviderError(f"Saaras start failed: {exc}") from exc
                    time.sleep(2.0 * (attempt + 1))
                    try:
                        job.upload_files(file_paths=audio_paths)
                    except Exception:
                        pass
            if not started:
                raise RetriableProviderError("Saaras job never started")
            job_id = self._extract_job_id(job)
            if job_id:
                self._poll_job_status(settings=settings, job_id=job_id)
            else:
                job.wait_until_complete()

            # The status endpoint reports COMPLETED slightly before the outputs
            # are fetchable, and download_outputs then 400s with "is not in
            # COMPLETED state. Current state: Running". Retry briefly rather
            # than failing a batch whose transcripts already exist.
            last_error: Exception | None = None
            for attempt in range(6):
                try:
                    job.download_outputs(output_dir=temp_dir)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if "completed state" not in str(exc).lower():
                        raise
                    time.sleep(2.0 * (attempt + 1))
            if last_error is not None:
                raise RetriableProviderError(str(last_error)) from last_error

            return self._read_batch_outputs(Path(temp_dir), audio_paths)

    async def _transcribe_sync_items(
        self,
        client: SarvamAI,
        items: list[tuple[str, RowContext]],
        per_item_start: dict[str, float],
        settings: object,
    ) -> dict[str, ProviderResult]:
        """One synchronous speech_to_text.transcribe call per file.

        Same model, mode and language routing as the batch path. Throttled by the
        same SAARAS_MIN_INTERVAL_SECONDS lock, so it is meant for smoke tests and
        re-running individual failed rows, not for a 26k-row run.
        """
        global _LAST_SAARAS_CALL_TS
        results: dict[str, ProviderResult] = {}
        for audio_path, row in items:
            row = row or RowContext()
            code = resolve_saaras_language(row.base_language, row.is_code_mix)

            def _sync(path: str = audio_path, language_code: str = code) -> str:
                try:
                    with open(path, "rb") as handle:
                        response = client.speech_to_text.transcribe(
                            file=handle,
                            model=settings.saaras_model,
                            mode=settings.saaras_mode or "codemix",
                            language_code=language_code,
                        )
                except Exception as exc:
                    text = str(exc).lower()
                    if "429" in text or "too many requests" in text or "rate_limit" in text or "timeout" in text:
                        raise RetriableProviderError(str(exc)) from exc
                    raise
                return str(getattr(response, "transcript", "") or "")

            async def _call(fn=_sync) -> str:
                global _LAST_SAARAS_CALL_TS
                async with _SAARAS_THROTTLE_LOCK:
                    await asyncio.to_thread(
                        wait_global_slot, _SAARAS_THROTTLE_FILE, float(settings.saaras_min_interval_seconds)
                    )
                    _LAST_SAARAS_CALL_TS = time.time()
                return await asyncio.to_thread(fn)

            start = per_item_start.get(audio_path, time.time())
            try:
                results[audio_path] = success_result(await run_with_retry(_call), start)
            except Exception as exc:  # noqa: BLE001 - recorded per row
                results[audio_path] = failed_result(start, str(exc))
        return results

    async def transcribe_batch(
        self,
        items: list[tuple[str, RowContext]],
    ) -> dict[str, ProviderResult]:
        settings = get_settings()
        now = time.time()
        if not settings.saaras_api_key:
            return {audio_path: failed_result(now, "Missing SAARAS_API_KEY") for audio_path, _ in items}

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

        # A job carries one language_code for every file in it, so rows must be
        # grouped by resolved code before batching. Code-mixed rows resolve to
        # `unknown` (auto/code-switch); monolingual rows get their xx-IN tag.
        by_language: dict[str, list[str]] = {}
        for audio_path, row in items:
            row = row or RowContext()
            code = resolve_saaras_language(row.base_language, row.is_code_mix)
            by_language.setdefault(code, []).append(audio_path)

        client = SarvamAI(api_subscription_key=settings.saaras_api_key)
        if str(settings.saaras_api_mode).strip().lower() == "sync":
            return await self._transcribe_sync_items(client, items, per_item_start, settings)
        configured_batch_size = max(1, int(settings.saaras_batch_size))
        batch_size = min(configured_batch_size, _SAARAS_MAX_FILES_PER_BATCH)
        results: dict[str, ProviderResult] = {}

        async def _run_one_batch(batch_paths: list[str], language_code: str) -> dict[str, str]:
            global _LAST_SAARAS_CALL_TS
            min_interval = max(0.0, float(settings.saaras_min_interval_seconds))
            async with _SAARAS_THROTTLE_LOCK:
                await asyncio.to_thread(wait_global_slot, _SAARAS_THROTTLE_FILE, min_interval)
                _LAST_SAARAS_CALL_TS = time.time()
            return await asyncio.to_thread(
                self._run_batch_sync,
                client,
                batch_paths,
                settings,
                language_code,
            )

        batches = [
            (language_code, paths[index : index + batch_size])
            for language_code, paths in by_language.items()
            for index in range(0, len(paths), batch_size)
        ]

        for language_code, batch_paths in batches:

            async def _call(
                batch_paths: list[str] = batch_paths,
                language_code: str = language_code,
            ) -> dict[str, str]:
                return await _run_one_batch(batch_paths=batch_paths, language_code=language_code)

            try:
                transcripts = await run_with_retry(_call)
                for audio_path in batch_paths:
                    results[audio_path] = success_result(
                        transcripts.get(audio_path, ""),
                        per_item_start.get(audio_path, time.time()),
                    )
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                if (
                    "429" in message
                    or "too many requests" in lowered
                    or "rate_limit_exceeded_error" in lowered
                    or "timeout" in lowered
                    # Both observed on a 100-row run and both transient: the same
                    # batch succeeds when retried in isolation.
                    or "files list must not be empty" in lowered
                    or "completed state" in lowered
                ):
                    error_message = f"Saaras batch retryable failure exhausted retries: {message}"
                else:
                    error_message = message
                for audio_path in batch_paths:
                    results[audio_path] = failed_result(
                        per_item_start.get(audio_path, time.time()),
                        error_message,
                    )

        for audio_path, _ in items:
            if audio_path not in results:
                results[audio_path] = failed_result(
                    per_item_start.get(audio_path, time.time()),
                    "Missing Saaras batch result.",
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
        return batch_results.get(audio_path, failed_result(time.time(), "Missing Saaras result."))
