"""Provider lookup, with imports deferred until a provider is actually used.

Each vendor needs its own SDK. Importing all of them at module load would make
`pip install indictelephony-bench` require every vendor's client library to
score a CSV, which is the opposite of what this package is for. So the registry
maps a name to a factory and the import happens on first use; a missing SDK then
names itself in the error, with the extra to install.
"""
from __future__ import annotations

from collections.abc import Callable

from indictelephony_bench.providers.base import ASRProvider

_FACTORIES: dict[str, Callable[[], ASRProvider]] = {}


def register(name: str, factory: Callable[[], ASRProvider]) -> None:
    """Add or replace a provider. A class is a valid factory."""
    _FACTORIES[name.lower().strip()] = factory


def _lazy(module: str, attr: str, extra: str) -> Callable[[], ASRProvider]:
    def factory() -> ASRProvider:
        import importlib

        try:
            mod = importlib.import_module(module)
        except ImportError as exc:  # the vendor SDK, not our code
            raise ImportError(
                f"{attr} needs an optional dependency: pip install "
                f"\'indictelephony-bench[{extra}]\'  ({exc})") from exc
        return getattr(mod, attr)()

    return factory


_BUILTIN = {
    "akshara": ("indictelephony_bench.providers.akshara", "AksharaProvider", "akshara"),
    "saaras": ("indictelephony_bench.providers.saaras", "SaarasProvider", "saaras"),
    "elevenlabs": ("indictelephony_bench.providers.elevenlabs", "ElevenLabsProvider", "elevenlabs"),
    "smallest": ("indictelephony_bench.providers.smallest", "SmallestProvider", "smallest"),
    "deepgram": ("indictelephony_bench.providers.deepgram", "DeepgramProvider", "deepgram"),
}
for _name, (_mod, _attr, _extra) in _BUILTIN.items():
    register(_name, _lazy(_mod, _attr, _extra))

PROVIDERS = tuple(sorted(_FACTORIES))


def get_provider(name: str) -> ASRProvider:
    """Build a provider by name, importing its SDK on first use."""
    key = (name or "").lower().strip()
    if key not in _FACTORIES:
        raise ValueError(f"unknown provider {name!r}; registered: {', '.join(sorted(_FACTORIES))}")
    return _FACTORIES[key]()
