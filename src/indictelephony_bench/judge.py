"""The model call behind semantic WER, with a cache and a pluggable backend.

Semantic WER is the one metric here that asks a model anything, so this module
exists to keep that dependency small, honest and swappable:

* **Cached by content.** A cache key hashes the texts being compared, never the
  system that produced them, so two systems that made the same mistake pay for
  one judge call and, more importantly, are judged identically. The cache file
  is per model, because serving one model's answers to a run configured for
  another would be undetectable in the output.
* **Swappable.** Anything matching `JudgeBackend` can be registered, including a
  stub, so the metric can be exercised in tests and on machines with no key.
* **Failure is visible.** A call that will not succeed raises. A judge that
  cannot answer must never look like a judge that answered "no difference",
  because that silently turns a coverage gap into a score.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from indictelephony_bench.config import Settings, get_settings

log = logging.getLogger(__name__)

# Bump when a prompt or the accept/reject logic changes: cached answers from an
# older version then stop being read, instead of quietly mixing two definitions
# of the same metric in one table.
JUDGE_VERSION = "semantic_equiv_v2"


@runtime_checkable
class JudgeBackend(Protocol):
    """One structured-output call. Raise on failure; retries are handled here."""

    def __call__(self, prompt: str, schema: type, model: str, settings: Settings) -> dict[str, Any]:
        ...


_BACKENDS: dict[str, JudgeBackend] = {}


def register_backend(name: str, backend: JudgeBackend) -> None:
    """Add or replace a backend, e.g. a local model or a deterministic stub."""
    _BACKENDS[name] = backend


def get_backend(name: str) -> JudgeBackend:
    """Look up a registered backend, or raise naming the ones that exist."""
    if name not in _BACKENDS:
        raise KeyError(f"no judge backend {name!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


# --------------------------------------------------------------------------
# Built-in backends
# --------------------------------------------------------------------------
def _gemini(prompt: str, schema: type, model: str, settings: Settings) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.require_judge_key())
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,                      # the judge is a classifier, not a writer
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if parsed is not None and hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise ValueError("empty judge response")
    return schema.model_validate(json.loads(text)).model_dump()


def _openai(prompt: str, schema: type, model: str, settings: Settings) -> dict[str, Any]:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": settings.require_judge_key()}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    client = OpenAI(**kwargs)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format=schema,
        timeout=settings.request_timeout_seconds,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is not None:
        return parsed.model_dump()
    content = (completion.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("empty judge response")
    return schema.model_validate(json.loads(content)).model_dump()


register_backend("gemini", _gemini)
register_backend("openai", _openai)


# --------------------------------------------------------------------------
# Identity and sampling
# --------------------------------------------------------------------------
def identity(fields: Sequence[str]) -> str:
    """A cache key over the texts alone.

    Keyed on content, never on provider: the same (reference, hypothesis) pair
    from two systems must get one answer, or the metric would depend on whose
    output happened to be judged first.
    """
    payload = "\x1f".join(str(f or "") for f in fields)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def is_sampled_in(row_id: str, sample_rate: float) -> bool:
    """Deterministic per-row sampling, so a rerun judges the same subset.

    Random sampling would make two runs disagree for reasons that have nothing
    to do with the systems being compared.
    """
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    digest = hashlib.sha1(str(row_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF < sample_rate


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_path(name: str, model: str, settings: Settings | None = None) -> Path:
    """Where answers for this cache and this model live.

    The model is part of the filename. A shared file would serve one model's
    answers to a run configured for another, and nothing in the output would
    show it.
    """
    settings = settings or get_settings()
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in model.strip())
    path = settings.cache_dir / f"{name}__{slug}_{JUDGE_VERSION}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Read a JSONL cache, tolerating a torn final line from an interrupted run."""
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s: skipping unreadable cache line %d", path, number)
                continue
            key = record.get("key")
            if key:
                cached[key] = record.get("response", {})
    return cached


def _call_with_retry(backend: JudgeBackend, prompt: str, schema: type, model: str,
                     settings: Settings) -> dict[str, Any]:
    attempts = max(1, settings.semantic_retry_attempts)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return backend(prompt, schema, model, settings)
        except Exception as exc:  # noqa: BLE001 - surfaced as coverage loss, not a crash
            last = exc
            if attempt + 1 < attempts:
                # Under parallel load the usual failure is a rate limit, and an
                # immediate retry meets the same limit.
                time.sleep(min(60.0, 2.0 ** attempt + random.uniform(0, 1)))
    raise RuntimeError(f"judge failed after {attempts} attempts: {last}")


def run_cached_batch(items: Iterable[tuple[str, str]], schema: type, cache_name: str, *,
                     model: str | None = None, settings: Settings | None = None,
                     backend: JudgeBackend | str | None = None,
                     on_progress: Callable[[int, int], None] | None = None,
                     ) -> tuple[dict[str, dict[str, Any]], int]:
    """Run `(key, prompt)` pairs through the judge, reusing anything cached.

    Returns `(key -> response, failures)`. Failures are counted and reported
    rather than raised, so one unanswerable row does not discard a run of
    thousands; the caller decides what an unanswered row means, and semantic WER
    treats it as nothing forgiven.
    """
    settings = settings or get_settings()
    model = (model or settings.judge_model).strip()
    if backend is None:
        backend = get_backend(settings.judge_backend)
    elif isinstance(backend, str):
        backend = get_backend(backend)

    path = cache_path(cache_name, model, settings)
    responses = load_cache(path)
    pending = {key: prompt for key, prompt in items if key not in responses}
    if not pending:
        return responses, 0

    failures = 0
    with path.open("a", encoding="utf-8") as stream:
        for done, (key, prompt) in enumerate(pending.items(), start=1):
            try:
                response = _call_with_retry(backend, prompt, schema, model, settings)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.warning("judge failed for %s: %s", key, exc)
                continue
            responses[key] = response
            stream.write(json.dumps({"key": key, "model": model, "response": response},
                                    ensure_ascii=False) + "\n")
            stream.flush()          # a killed run keeps every answer already paid for
            if on_progress:
                on_progress(done, len(pending))
    return responses, failures
