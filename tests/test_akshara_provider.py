"""The Akshara provider, against a fake ConvoZen SDK: no key, no network.

What matters here is what the provider sends and how it treats each failure,
not the model's output, so the SDK is replaced by a recorder.
"""
import asyncio
import sys
import types

import pytest

from indictelephony_bench import config
from indictelephony_bench.providers.base import SUCCESS, RowContext


class _Err(Exception):
    pass


class FakeSDK:
    """Stands in for `convozen`: records calls, replays scripted outcomes."""

    ConvozenError = _Err

    class AuthenticationError(_Err):
        pass

    class RateLimitError(_Err):
        pass

    class APIError(_Err):
        pass

    def __init__(self):
        self.calls, self.outcomes, self.client_kwargs = [], [], None

    def Client(self, **kwargs):  # noqa: N802 - mirrors the SDK's name
        self.client_kwargs = kwargs
        sdk = self

        class _STT:
            def transcribe(self, audio, **kw):
                sdk.calls.append({"audio": audio, **kw})
                outcome = sdk.outcomes.pop(0) if sdk.outcomes else "hello"
                if isinstance(outcome, Exception):
                    raise outcome
                return types.SimpleNamespace(text=outcome, score=0.9)

        return types.SimpleNamespace(stt=_STT())


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    fake = FakeSDK()
    module = types.ModuleType("convozen")
    for name in ("Client", "AuthenticationError", "RateLimitError", "APIError", "ConvozenError"):
        setattr(module, name, getattr(fake, name))
    monkeypatch.setitem(sys.modules, "convozen", module)
    monkeypatch.setenv("CONVOZEN_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_RETRY_ATTEMPTS", "3")
    monkeypatch.chdir(tmp_path)                     # no stray .env
    config.get_settings.cache_clear()

    async def no_sleep(_):                          # retries without waiting
        return None
    from indictelephony_bench.providers import utils
    monkeypatch.setattr(utils.asyncio, "sleep", no_sleep)

    fake.audio = tmp_path / "u1.wav"
    fake.audio.write_bytes(b"RIFF....WAVE")
    yield fake
    config.get_settings.cache_clear()


def _run(fake, row=None, language=None):
    from indictelephony_bench.providers.akshara import AksharaProvider
    return asyncio.run(AksharaProvider().transcribe(str(fake.audio), language=language, row=row))


# --- what is sent ----------------------------------------------------------
@pytest.mark.parametrize("tag,base,want", [
    ("en-hi", "hi", ["en", "hi"]),       # code-mixed: both languages
    ("hi", "hi", ["hi"]),
    ("ta", "", ["ta"]),
    ("", "kn", ["kn"]),                  # no tag: fall back to the base language
    ("hi-mr", "mr", ["hi", "mr"]),
    ("en-xx", "en", ["en"]),             # a code the SDK rejects is dropped
    ("", "", None),                      # nothing known: send no hint at all
])
def test_language_hints(tag, base, want):
    from indictelephony_bench.providers.akshara import language_hints
    assert language_hints(RowContext(language_tag=tag, base_language=base)) == want


def test_request_carries_model_and_hints_but_never_keywords(sdk):
    result = _run(sdk, RowContext(language_tag="en-hi", base_language="hi", is_code_mix=True))
    call = sdk.calls[0]
    assert result["status"] == SUCCESS and result["prediction"] == "hello"
    assert call["model"] == "akshara-pro" and call["lang_tags"] == ["en", "hi"]
    # the references carry keywords; sending them would hand over the answer
    assert "keywords" not in call
    assert sdk.client_kwargs["api_key"] == "test-key"


def test_model_is_configurable(sdk, monkeypatch):
    monkeypatch.setenv("AKSHARA_MODEL", "akshara-test")
    config.get_settings.cache_clear()
    _run(sdk)
    assert sdk.calls[0]["model"] == "akshara-test"


# --- how each failure is treated -------------------------------------------
def test_missing_key_fails_without_calling_the_service(sdk, monkeypatch):
    monkeypatch.delenv("CONVOZEN_API_KEY")
    config.get_settings.cache_clear()
    result = _run(sdk)
    assert result["status"] != SUCCESS and "CONVOZEN_API_KEY" in result["error"]
    assert sdk.calls == []


def test_rate_limit_is_retried(sdk):
    sdk.outcomes = [FakeSDK.RateLimitError("Rate limit exceeded"), "second try"]
    result = _run(sdk)
    assert result["prediction"] == "second try" and len(sdk.calls) == 2


def test_server_error_is_retried(sdk):
    sdk.outcomes = [FakeSDK.APIError("API error 503: busy"), "recovered"]
    assert _run(sdk)["prediction"] == "recovered" and len(sdk.calls) == 2


def test_client_error_is_not_retried(sdk):
    sdk.outcomes = [FakeSDK.APIError("API error 400: bad audio")]
    result = _run(sdk)
    assert result["status"] != SUCCESS and len(sdk.calls) == 1


def test_bad_key_fails_at_once(sdk):
    sdk.outcomes = [FakeSDK.AuthenticationError("Invalid API key")]
    result = _run(sdk)
    assert "401" in result["error"] and len(sdk.calls) == 1


def test_retries_give_up_and_report(sdk):
    sdk.outcomes = [FakeSDK.RateLimitError("x")] * 5
    result = _run(sdk)
    assert result["status"] != SUCCESS and len(sdk.calls) == 3   # PROVIDER_RETRY_ATTEMPTS


def test_missing_audio_is_reported_not_raised(sdk):
    from indictelephony_bench.providers.akshara import AksharaProvider
    result = asyncio.run(AksharaProvider().transcribe("/no/such.wav"))
    assert result["status"] != SUCCESS and sdk.calls == []


# --- the status the runner and the loader rely on ---------------------------
def test_loader_prefers_the_status_providers_actually_write(tmp_path):
    from indictelephony_bench.data import load_predictions
    path = tmp_path / "p.csv"
    path.write_text(f"utterance_id,prediction,status\nu1,,failed\nu1,retried,{SUCCESS}\n")
    assert load_predictions(path)["u1"] == "retried"
