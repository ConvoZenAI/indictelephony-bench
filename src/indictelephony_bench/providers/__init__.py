"""Calling a speech-to-text system, so a baseline can be re-run.

Scoring needs none of this. It is here because the vendors' transcripts are not
redistributable, so the published baselines are reproduced by running the
systems again rather than by re-scoring stored output.

Every provider returns the same `ProviderResult`, and a failure is a result with
a status other than `SUCCESS` rather than an exception, so one dead request cannot discard a
run of thousands. A failed row is scored as an empty hypothesis, never dropped.

Adding a system is one class and one registry line:

    from indictelephony_bench.providers import ASRProvider, register

    class MyProvider(ASRProvider):
        provider_name = "mine"
        async def transcribe(self, audio_path, language=None, context=None, row=None):
            ...
            return {"prediction": text, "latency_ms": ms, "status": SUCCESS, "error": None}

    register("mine", MyProvider)
"""
from indictelephony_bench.providers.base import SUCCESS, ASRProvider, ProviderResult, RowContext
from indictelephony_bench.providers.registry import PROVIDERS, get_provider, register

__all__ = ["SUCCESS", "ASRProvider", "ProviderResult", "RowContext",
           "PROVIDERS", "get_provider", "register"]
