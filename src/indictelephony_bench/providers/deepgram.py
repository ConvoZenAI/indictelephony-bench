"""Deepgram Nova-3 over its public REST API.

Nova-3 has no Malayalam, so `eligibility.py` drops those rows rather than
sending them; see `resolve_deepgram_language` for how the remaining rows choose
between the monolingual and multilingual modes.
"""
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from indictelephony_bench.config import get_settings
from indictelephony_bench.providers.base import ASRProvider, ProviderResult, RowContext
from indictelephony_bench.providers.lang_codes import (
    DEEPGRAM_MULTI,
    resolve_deepgram_language,
)
from indictelephony_bench.providers.utils import (
    RetriableProviderError,
    failed_result,
    is_retryable_http_status,
    run_with_retry,
    success_result,
)


class _UnsupportedLanguageCombination(Exception):
    """nova-3 rejected this mono language; retry the row on `multi`."""


class DeepgramProvider(ASRProvider):
    provider_name = "deepgram"

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        context: str | None = None,
        row: RowContext | None = None,
    ) -> ProviderResult:
        settings = get_settings()
        start = time.time()
        if not settings.deepgram_api_key:
            return failed_result(start, "Missing DEEPGRAM_API_KEY")

        path = Path(audio_path)
        if not path.exists():
            return failed_result(start, f"Audio file not found: {audio_path}")

        row = row or RowContext(language_tag=language or "")
        language_code = resolve_deepgram_language(row.base_language, row.is_code_mix)

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        audio_bytes = path.read_bytes()

        async def _request_transcript(client: httpx.AsyncClient, code: str) -> str:
            query_params = {
                "model": settings.deepgram_model,
                "language": code,
                "punctuate": "true",
                "smart_format": "true",
                "numerals": "true",
            }
            query = f"https://api.deepgram.com/v1/listen?{urlencode(query_params)}"

            response = await client.post(
                query,
                headers={
                    "Authorization": f"Token {settings.deepgram_api_key}",
                    "Content-Type": content_type,
                },
                content=audio_bytes,
            )
            if is_retryable_http_status(response.status_code):
                raise RetriableProviderError(f"Retryable status {response.status_code}")
            if (
                response.status_code == 400
                and "model/language/tier" in response.text
                and code != DEEPGRAM_MULTI
            ):
                # This model/tier does not offer that mono language. Fall back to
                # multi once rather than losing the row outright.
                raise _UnsupportedLanguageCombination(response.text[:200])
            response.raise_for_status()
            payload = response.json()
            return (
                payload.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )

        async def _call() -> str:
            async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
                try:
                    return await _request_transcript(client, language_code)
                except _UnsupportedLanguageCombination:
                    return await _request_transcript(client, DEEPGRAM_MULTI)

        try:
            prediction = await run_with_retry(_call)
            return success_result(prediction, start)
        except Exception as exc:
            return failed_result(start, str(exc))
