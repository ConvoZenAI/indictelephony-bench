"""Settings, read from the environment or a local `.env`.

Only semantic WER and the optional inference runner need any of this; the four
deterministic metrics take no configuration at all, which is why they are the
part of the benchmark that always reproduces exactly.

No credential is ever read from source. A missing key raises where it is used,
with the variable name in the message, rather than silently scoring a run as if
the model had answered.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value, `#` comments, no interpolation.

    Deliberately not python-dotenv: one less dependency for a file format that
    is three lines of parsing, and the environment always wins over the file so
    a shell override behaves the way an operator expects.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Everything configurable, with defaults that match the published run."""

    # --- semantic WER -----------------------------------------------------
    # A judge proposal is accepted only if a deterministic check confirms it.
    # `min_similarity` is that check's floor for spelling variants, on a 0-100
    # scale; lowering it forgives more, including things that are not the same
    # word, so it is published with the results rather than tuned per system.
    semantic_min_similarity: int = 60
    # A number is accepted only when its value is confirmed. Turning this on
    # forgives number differences the check could not verify, which is exactly
    # the error a downstream system cannot absorb.
    semantic_allow_unverified_numbers: bool = False
    semantic_sample_rate: float = 1.0
    semantic_retry_attempts: int = 2

    judge_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    request_timeout_seconds: float = 120.0

    cache_dir: Path = Path("outputs/cache")

    # --- re-running a system (optional; scoring needs none of this) -------
    max_concurrency: int = 10
    provider_retry_attempts: int = 2

    # Akshara, through the public ConvoZen SDK. An empty base URL keeps the
    # SDK's own default endpoint.
    convozen_api_key: str = ""
    convozen_base_url: str = ""
    akshara_model: str = "akshara-pro"

    saaras_api_key: str = ""
    saaras_model: str = "saaras:v4"
    saaras_mode: str = "code-mix"
    saaras_api_mode: str = "batch"
    saaras_batch_size: int = 16
    saaras_min_interval_seconds: float = 0.0

    elevenlabs_api_key: str = ""
    elevenlabs_model_id: str = "scribe_v2"

    smallest_api_key: str = ""
    smallest_model: str = ""
    smallest_batch_size: int = 8
    smallest_fallback_language: str = ""
    smallest_max_parallel_requests: int = 4
    smallest_min_interval_seconds: float = 0.0

    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"

    @property
    def provider_timeout_seconds(self) -> float:
        """Alias kept because the provider modules read this name."""
        return self.request_timeout_seconds

    @property
    def judge_backend(self) -> str:
        """`gemini` or `openai`, taken from the model name, never rewritten.

        An earlier version silently rewrote any non-Gemini model to a Gemini one,
        so a configured model never took effect and nothing said so. An unusable
        model name now fails loudly at the call.
        """
        return "gemini" if "gemini" in self.judge_model.lower() else "openai"

    def require_judge_key(self) -> str:
        field = "GEMINI_API_KEY" if self.judge_backend == "gemini" else "OPENAI_API_KEY"
        key = self.gemini_api_key if self.judge_backend == "gemini" else self.openai_api_key
        if not key:
            raise RuntimeError(
                f"semantic WER needs {field}. Put it in .env or the environment; "
                f"the deterministic metrics (WER, CER, MER, KWER) need no key.")
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings for this process, read once.

    Cached, so a run cannot pick up a different configuration halfway through
    and produce a table scored two ways.
    """
    _load_dotenv(Path(".env"))
    env = os.environ.get
    return Settings(
        semantic_min_similarity=int(env("SEMANTIC_WER_MIN_SIMILARITY", "60")),
        semantic_allow_unverified_numbers=_flag("SEMANTIC_WER_ALLOW_UNVERIFIED_NUMBERS", False),
        semantic_sample_rate=float(env("SEMANTIC_WER_SAMPLE_RATE", "1.0")),
        semantic_retry_attempts=int(env("SEMANTIC_RETRY_ATTEMPTS", "2")),
        judge_model=env("SEMANTIC_WER_JUDGE_MODEL", "gemini-2.5-flash").strip(),
        gemini_api_key=env("GEMINI_API_KEY", "").strip(),
        openai_api_key=env("OPENAI_API_KEY", "").strip(),
        openai_base_url=env("OPENAI_BASE_URL", "").strip(),
        request_timeout_seconds=float(env("REQUEST_TIMEOUT_SECONDS", "120")),
        cache_dir=Path(env("CACHE_DIR", "outputs/cache")),
        max_concurrency=int(env("MAX_CONCURRENCY", "10")),
        provider_retry_attempts=int(env("PROVIDER_RETRY_ATTEMPTS", "2")),
        convozen_api_key=env("CONVOZEN_API_KEY", "").strip(),
        convozen_base_url=env("CONVOZEN_BASE_URL", "").strip(),
        akshara_model=env("AKSHARA_MODEL", "akshara-pro").strip(),
        saaras_api_key=env("SARVAM_API_KEY", env("SAARAS_API_KEY", "")).strip(),
        saaras_model=env("SAARAS_MODEL", "saaras:v4").strip(),
        saaras_mode=env("SAARAS_MODE", "code-mix").strip(),
        saaras_api_mode=env("SAARAS_API_MODE", "batch").strip(),
        saaras_batch_size=int(env("SAARAS_BATCH_SIZE", "16")),
        saaras_min_interval_seconds=float(env("SAARAS_MIN_INTERVAL_SECONDS", "0")),
        elevenlabs_api_key=env("ELEVENLABS_API_KEY", "").strip(),
        elevenlabs_model_id=env("ELEVENLABS_MODEL_ID", "scribe_v2").strip(),
        smallest_api_key=env("SMALLEST_API_KEY", "").strip(),
        smallest_model=env("SMALLEST_MODEL", "").strip(),
        smallest_batch_size=int(env("SMALLEST_BATCH_SIZE", "8")),
        smallest_fallback_language=env("SMALLEST_FALLBACK_LANGUAGE", "").strip(),
        smallest_max_parallel_requests=int(env("SMALLEST_MAX_PARALLEL_REQUESTS", "4")),
        smallest_min_interval_seconds=float(env("SMALLEST_MIN_INTERVAL_SECONDS", "0")),
        deepgram_api_key=env("DEEPGRAM_API_KEY", "").strip(),
        deepgram_model=env("DEEPGRAM_MODEL", "nova-3").strip(),
    )
