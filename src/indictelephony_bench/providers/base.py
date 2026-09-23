"""The provider interface every speech-to-text system implements.

One shared signature and one shared result shape, so the runner never special-
cases a vendor. A provider that cannot act on a parameter ignores it; that is
what a shared signature is for, and each one says so where it does.

A failure is a `ProviderResult` whose status is not `SUCCESS`, not an exception. One
dead request must not end a run of thousands, and the row still reaches the
scorer as an empty hypothesis rather than disappearing from the denominator.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

# The one status that means a transcript came back. The runner, the resume
# logic and the prediction loader all compare against this constant, so a
# provider and its readers can never disagree about what success is called.
SUCCESS = "success"


class ProviderResult(TypedDict):
    """What every provider returns. Any status other than `SUCCESS` means the row failed."""

    prediction: str
    latency_ms: float
    status: str
    error: str | None


@dataclass(frozen=True, slots=True)
class RowContext:
    """Per-row language information a provider needs to route its request.

    Providers differ in what they can act on: Deepgram picks a nova-3 mono/multi
    tag, ElevenLabs either passes a code or omits it for auto-detect, Saaras
    switches between auto and a fixed locale. All of them need the base language
    and whether the row is code-mixed, which is why this travels with the audio path.
    """

    language_tag: str = ""
    base_language: str = ""
    is_code_mix: bool = False

    @property
    def language(self) -> str | None:
        return self.language_tag or None


class ASRProvider(ABC):
    """One speech-to-text system. Implement `transcribe` and register the class."""

    provider_name: str

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        context: str | None = None,
        row: RowContext | None = None,
    ) -> ProviderResult:
        """
        Transcribe one audio file and return standardized provider output.

        `row` carries per-row routing info; providers that ignore language
        entirely may leave it unused. Falling back to `RowContext()` keeps a
        provider working when it is called without one (e.g. from a test).
        """
