"""Random, human-friendly run-name generation.

A training / evaluation run with an auto-generated name like ``train_funny_rabbit`` is far easier to
refer to in logs, MLflow, and conversation than a timestamp or a hash. :func:`generate_run_name`
produces a short ``adjective_noun`` slug (via the offline ``coolname`` library — no network) with an
optional prefix and/or postfix, and is the default ``run_name`` source for the marainer trainer /
evaluator. :func:`run_prefix` supplies the prefix a runnable's ``task`` implies, so the two compose
into the one line every backend writes: ``generate_run_name(prefix=run_prefix(task))``.
"""

from typing import Literal

import coolname

DEFAULT_SEPARATOR = "_"

#: What a run name is prefixed with — three kinds of run, whatever the engine underneath.
#: A closed ``Literal`` rather than a bare ``str`` because the set is fixed and short, so a
#: caller (and a form/schema generator) can enumerate it.
RunPrefix = Literal["train", "evaluate", "predict"]

#: The task verbs that get their own prefix. Everything else — ``evaluate``, ``test``, and any
#: verb a future runnable adds — is an evaluation as far as a run NAME is concerned.
_PREFIX_BY_TASK = {"fit": "train", "predict": "predict"}


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


def run_prefix(task: str) -> RunPrefix:
    """The run-name prefix a runnable's ``task`` implies — ``train`` / ``evaluate`` / ``predict``.

    A runnable's task vocabulary is finer than a run name needs: ``evaluate`` and ``test`` are
    two entry points but one KIND of run, and a name that said ``test_funny_rabbit`` for one and
    ``evaluate_funny_rabbit`` for the other would split one concept across two prefixes in every
    tracking backend. So the mapping is deliberately many-to-one, and anything unrecognised —
    including a verb a future runnable adds — is an evaluation rather than an error: a run name
    is a label, and refusing to produce one is never the right answer to an unfamiliar verb.

    The task vocabulary itself belongs to the runnable (it is the ``@entrypoint`` set), which is
    why this takes a plain ``str`` and only the RETURN is a closed ``Literal``.

    Args:
        task: The runnable's task verb (``"fit"`` / ``"evaluate"`` / ``"test"`` / ``"predict"``).

    Returns:
        ``"train"`` for ``fit``, ``"predict"`` for ``predict``, ``"evaluate"`` otherwise.
    """
    return _PREFIX_BY_TASK.get(task, "evaluate")  # type: ignore[return-value]  # values are RunPrefix


__all__ = ["RunPrefix", "generate_run_name", "run_prefix"]
