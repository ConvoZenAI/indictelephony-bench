"""ElevenLabs Scribe, through the official client.

Diarization and audio-event tagging are off: the benchmark scores one
single-speaker turn at a time, and event tags would appear in the transcript as
words nobody said.
"""
import asyncio
import time
from pathlib import Path

from elevenlabs.client import ElevenLabs

from indictelephony_bench.config import get_settings
from indictelephony_bench.providers.base import ASRProvider, ProviderResult, RowContext
from indictelephony_bench.providers.lang_codes import resolve_elevenlabs_language
from indictelephony_bench.providers.utils import (
    RetriableProviderError,
    failed_result,
    run_with_retry,
    success_result,
)


class ElevenLabsProvider(ASRProvider):
    provider_name = "elevenlabs"

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        context: str | None = None,
        row: RowContext | None = None,
    ) -> ProviderResult:
        settings = get_settings()
        start = time.time()
        if not settings.elevenlabs_api_key:
            return failed_result(start, "Missing ELEVENLABS_API_KEY")

        path = Path(audio_path)
        if not path.exists():
            return failed_result(start, f"Audio file not found: {audio_path}")

        row = row or RowContext(language_tag=language or "")
        language_code = resolve_elevenlabs_language(row.base_language, row.is_code_mix)

        client = ElevenLabs(api_key=settings.elevenlabs_api_key)

        def _sync_call() -> str:
            convert_kwargs: dict[str, object] = {
                "model_id": settings.elevenlabs_model_id,
                "diarize": False,
                "tag_audio_events": False,
            }
            # Omit the kwarg entirely for auto-detect: some SDK versions reject
            # language_code=None rather than treating it as unset.
            if language_code:
                convert_kwargs["language_code"] = language_code

            try:
                with path.open("rb") as audio_file:
                    result = client.speech_to_text.convert(
                        file=audio_file,
                        **convert_kwargs,
                    )
                text = getattr(result, "text", None)
                if text is None:
                    text = getattr(result, "transcript", None)
                if text is None and isinstance(result, dict):
                    text = result.get("text") or result.get("transcript")
                if text is None:
                    # Surface the shape instead of scoring str(result) as speech.
                    raise RuntimeError(
                        f"ElevenLabs response had no text field: {repr(result)[:200]}"
                    )
                return text
            except Exception as exc:
                message = str(exc)
                if "429" in message or "timeout" in message.lower():
                    raise RetriableProviderError(message) from exc
                raise

        async def _call() -> str:
            return await asyncio.to_thread(_sync_call)

        try:
            prediction = await run_with_retry(_call)
            return success_result(prediction, start)
        except Exception as exc:
            return failed_result(start, str(exc))
