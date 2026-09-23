"""IndicTelephony-Bench: scoring for code-mixed Indian telephone speech.

The benchmark is published as a dataset; this package is the scoring side of it,
so that a number in the paper or the article can be recomputed rather than taken
on trust. Everything here is deterministic except `metrics.semantic`, which is
documented separately because it calls a model.

    from indictelephony_bench import load_release, score

    release = load_release("Tamil")                 # from the Hugging Face hub
    result  = score(release, my_predictions)        # WER, CER, MER, KWER
"""
from indictelephony_bench.data import DATASET_ID, LANGUAGES, load_release
from indictelephony_bench.normalize import normalize
from indictelephony_bench.scoring import bootstrap_ci, score, score_by

__all__ = ["normalize", "load_release", "score", "score_by", "bootstrap_ci",
           "DATASET_ID", "LANGUAGES"]
__version__ = "1.0.1"
