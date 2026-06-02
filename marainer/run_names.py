"""Random, human-friendly run-name generation.

A training / evaluation run with an auto-generated name like ``train_funny_rabbit`` is far easier to
refer to in logs, MLflow, and conversation than a timestamp or a hash. :func:`generate_run_name`
produces a short ``adjective_noun`` slug (via the offline ``coolname`` library — no network) with an
optional prefix and/or postfix, and is the default ``run_name`` source for the marainer trainer /
evaluator.
"""

import coolname

DEFAULT_SEPARATOR = "_"


def generate_run_name(prefix: str = "", postfix: str = "", separator: str = DEFAULT_SEPARATOR) -> str:
    """Generate a random run name like ``funny_rabbit`` (optionally ``<prefix>_funny_rabbit_<postfix>``).

    Uses ``coolname`` to draw a two-word ``adjective_noun`` slug — offline, no network, a fresh name
    on every call. ``prefix`` and ``postfix`` are joined around it with ``separator`` and empty parts
    are skipped, so ``generate_run_name(prefix="train")`` yields e.g. ``train_funny_rabbit`` and
    ``generate_run_name()`` yields just ``funny_rabbit``.

    Args:
        prefix: Optional leading token (e.g. ``"train"`` / ``"evaluate"``). ``""`` (default) = none.
        postfix: Optional trailing token (e.g. a fold id or model tag). ``""`` (default) = none.
        separator: String joining prefix / slug / postfix and the two slug words. Defaults to ``"_"``.
    """
    slug = separator.join(coolname.generate(2))
    parts = [part for part in (prefix, slug, postfix) if part]
    return separator.join(parts)


__all__ = ["generate_run_name"]
