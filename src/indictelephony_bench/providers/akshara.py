"""Akshara, through the public ConvoZen SDK (`pip install convozen`).

SDK: https://pypi.org/project/convozen/

Authentication is an API key, read from `CONVOZEN_API_KEY` like every other
credential here - never from source. The model is `akshara-pro`, the SDK's
default; `AKSHARA_MODEL` overrides it. (The SDK also lists `akshara-base`, but
the service rejects that name.)

Three request choices are deliberate:

* **Language hints come from the utterance's tag**, so a code-mixed `en-hi` turn
  is sent as `["en", "hi"]`: the languages actually spoken, which is what the
  SDK's `lang_tags` asks for.
* **Keywords are never sent.** The SDK can bias recognition towards a keyword
  list, and this benchmark's references carry one per utterance. Passing them
  would hand the system the answer to the metric that scores it.
* **The file goes up as released**, 8 kHz mono. Resampling is the service's job,
  as it is for every other system here.

The SDK retries dropped connections and timeouts itself but not rate limits or
server errors, so those two - and only those - are retried here. A bad key
fails at once: retrying cannot fix it, and every row would wait to learn that.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from indictelephony_bench.config import get_settings
from indictelephony_bench.providers.base import ASRProvider, ProviderResult, RowContext
from indictelephony_bench.providers.utils import (
    RetriableProviderError,
    failed_result,
    run_with_retry,
    success_result,
)

# The languages the SDK accepts as hints, which are exactly the release's nine.
SDK_LANGUAGES = frozenset({"bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"})

# The SDK reports server failures as "API error <status>: <body>".
_SERVER_ERROR = re.compile(r"API error 5\d\d\b")


def language_hints(row: RowContext | None, language: str | None = None) -> list[str] | None:
    """The languages spoken in one utterance, in the form the SDK takes.

    Codes from the language tag first (`en-hi` -> ["en", "hi"]), then the base
    language if the tag did not already carry it. Anything the SDK would reject
    is dropped rather than failing the row, and an empty result sends no hint.
    """
    tag = (row.language_tag if row else "") or language or ""
    codes = [c for c in tag.lower().replace("_", "-").split("-") if c]
    base = (row.base_language if row else "").strip().lower()
    if base and base not in codes:
        codes.append(base)
    hints = [c for c in dict.fromkeys(codes) if c in SDK_LANGUAGES]
    return hints or None


class AksharaProvider(ASRProvider):
    provider_name = "akshara"

    def __init__(self) -> None:
        self._client = None

    def _stt(self):
        """One SDK client per provider, built on first use.

        The SDK validates the key and base URL when the client is built and
        raises ValueError if either is unusable; the caller reports that as a
        failed row instead of letting it end the run.
        """
        if self._client is None:
            import convozen

            settings = get_settings()
            kwargs = {"api_key": settings.convozen_api_key,
                      "timeout": int(settings.request_timeout_seconds)}
            if settings.convozen_base_url:
                kwargs["base_url"] = settings.convozen_base_url
            self._client = convozen.Client(**kwargs)
        return self._client.stt

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        context: str | None = None,
        row: RowContext | None = None,
    ) -> ProviderResult:
        from convozen import APIError, AuthenticationError, RateLimitError

        settings = get_settings()
        start = time.time()
        if not Path(audio_path).is_file():
            return failed_result(start, f"Audio file not found: {audio_path}")
        if not settings.convozen_api_key:
            return failed_result(start, "Akshara needs CONVOZEN_API_KEY; set it in .env or the environment.")

        try:
            stt = self._stt()
        except ValueError as exc:
            return failed_result(start, f"ConvoZen SDK configuration: {exc}")

        hints = language_hints(row, language)

        def _call() -> str:
            try:
                response = stt.transcribe(
                    audio_path,
                    model=settings.akshara_model,
                    lang_tags=hints,
                    # keywords deliberately omitted - see the module docstring
                )
            except RateLimitError as exc:
                raise RetriableProviderError(f"rate limited: {exc}") from exc
            except APIError as exc:
                # a server-side failure may pass; a 4xx means the request itself
                # is wrong and never will
                if _SERVER_ERROR.search(str(exc)):
                    raise RetriableProviderError(str(exc)) from exc
                raise
            return (response.text or "").strip()

        async def _attempt() -> str:
            return await asyncio.to_thread(_call)

        try:
            return success_result(await run_with_retry(_attempt), start)
        except AuthenticationError:
            return failed_result(start, "ConvoZen rejected CONVOZEN_API_KEY (401).")
        except Exception as exc:  # noqa: BLE001 - one row must not end the run
            return failed_result(start, str(exc))
