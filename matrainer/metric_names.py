"""What a metric is CALLED, translated between the vocabularies eight engines use.

A run records ``val/accuracy``. The same run under Keras records ``val_accuracy``, under fastai
``accuracy`` (its recorder's metrics are the VALIDATION ones, unprefixed), under a
``transformers.Trainer`` ``eval_accuracy``, and under Lightning ``val/accuracy`` but with the loss
split into ``train/loss_step`` and ``train/loss_epoch``. Nothing renames these on the way out —
every :class:`~matrainer.tracking.base.Tracker` writes the key it is handed verbatim — so a
sweep across engines lands one quantity under four names and no tracker can plot them together.

This module is the translation layer, and it is deliberately **not** a one-way canonicaliser:

* :class:`MetricVocabulary` describes ONE engine's spelling — its separator, how it writes each
  split, what an unprefixed key means, and the base names it has its own word for.
* :data:`VOCABULARIES` holds one per framework, ``matrainer``'s being the canonical
  ``<split>/<name>`` shape that the torch and Lightning backends already emit and that
  :func:`~matrainer.torch.metrics.build_metric_collection` already produces via ``clone(prefix=)``.
* :class:`MetricNames` parses FROM the caller's vocabulary and renders INTO the configured one,
  so ``matrainer -> keras`` works exactly as well as ``keras -> matrainer``.

The split of duties across the two arguments is what makes ONE config key work for every backend:
``source`` is passed per CALL by the code doing the reporting (only it knows which engine produced
the names, and whether an unprefixed ``accuracy`` was measured on train or on validation), while
``target`` and ``renames`` come from CONFIG. A single top-level ``metric_names:`` key can therefore
broadcast into eight differently-sourced backends and mean the right thing in each.

**Why matrainer and not a training project.** Same argument as :mod:`matrainer.metric_slots`
beside it: this is the metrics vocabulary, which this package already owns, and a second task on
the same eight engines would otherwise copy it verbatim. Like that module it imports **nothing but
the standard library** (plus confluid's decorator), so the deliberately framework-free backends
(JAX, MLX, Keras, any hand-written loop) can use it. Unlike it, this one IS re-exported from the
package root: it sits on the path metrics take to a tracker, and a YAML has to be able to name it.

Example::

    names = MetricNames()                                   # the default: canonical output
    names({"val_loss": 0.3, "accuracy": 0.9}, source="keras")
    # {'val/loss': 0.3, 'train/accuracy': 0.9}
    names({"accuracy": 0.9}, source="fastai")               # fastai metrics are validation ones
    # {'val/accuracy': 0.9}
    MetricNames(target="keras")({"val/loss": 0.3})           # ...and back out again
    # {'val_loss': 0.3}
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional, Tuple, cast, get_args

from confluid import configurable, flow

__all__ = [
    "as_metric_names",
    "Framework",
    "Split",
    "FRAMEWORKS",
    "SPLITS",
    "AXIS_KEYS",
    "MetricVocabulary",
    "VOCABULARIES",
    "vocabulary",
    "MetricNames",
    "parse_metric_name",
    "render_metric_name",
]

#: The engines whose metric spelling this module knows, plus ``matrainer``'s own canonical one.
Framework = Literal[
    "matrainer",
    "torch",
    "lightning",
    "keras",
    "tensorflow",
    "jax",
    "mlx",
    "fastai",
    "huggingface",
]

#: The three splits a metric can belong to.
Split = Literal["train", "val", "test"]

#: Runtime tuples derived FROM the annotations — never a restated parallel list.
FRAMEWORKS: Tuple[str, ...] = get_args(Framework)
SPLITS: Tuple[str, ...] = get_args(Split)

#: Keys that are an AXIS rather than a measurement, and so belong to no split. Every engine that
#: reports them spells them the same way, and prefixing ``epoch`` with ``train/`` would invent a
#: second series for a number that is already the x-value of every other one.
AXIS_KEYS: Tuple[str, ...] = ("epoch", "step")

#: The canonical separator. A ``/`` is what TensorBoard groups on and what ClearML splits into
#: (title, series) — see ``clearml_backend.log_metrics`` — so the canonical shape is the one that
#: makes a tracker's own UI group train and val curves without being told to.
CANONICAL_SEPARATOR = "/"


class MetricVocabulary:
    """How ONE engine spells a metric name.

    Args:
        separator: Between the split and the base name (``"/"`` canonically, ``"_"`` for the
            engines that flatten).
        splits: Canonical split -> this engine's word for it. An empty string means the split is
            written with NO prefix at all (Keras' training metrics are bare ``loss`` /
            ``accuracy``), and a split this engine never reports is simply absent — rendering
            into it then falls back to the canonical word rather than dropping the split.
        default_split: Which split an unprefixed key belongs to. ``"train"`` for the engines whose
            validation metrics carry a prefix; ``"val"`` for fastai, whose ``Recorder`` columns
            after ``valid_loss`` ARE the validation metrics.
        aliases: This engine's word for a base name -> the canonical one, e.g. Keras'
            ``sparse_categorical_accuracy`` -> ``accuracy``. Many-to-one on purpose: ``compile()``
            picks one of four spellings from the loss and the target shape, and all four mean the
            same measurement. Parse-side ONLY, and deliberately not inverted for rendering: those
            four are compile-time artefacts rather than names anyone writes, so translating INTO
            Keras yields ``val_accuracy`` — this vocabulary decides how the SPLIT is spelled, and
            the base name stays the canonical one.
    """

    def __init__(
        self,
        separator: str = CANONICAL_SEPARATOR,
        splits: Optional[Mapping[str, str]] = None,
        default_split: str = "train",
        aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.separator = separator
        self.splits = dict(splits) if splits else {split: split for split in SPLITS}
        self.default_split = default_split
        self.aliases = dict(aliases) if aliases else {}

    # -- parsing ------------------------------------------------------------------------------
    def split_of(self, token: str) -> Optional[str]:
        """The canonical split this engine's ``token`` names, or ``None`` if it names no split."""
        for split, word in self.splits.items():
            if word and word == token:
                return split
        # A split this engine has no word of its own for is WRITTEN with the canonical word (see
        # `word_for`), so it must READ back as that split too — otherwise a name this module
        # rendered itself would not survive being parsed again (fastai has no test split, and
        # `test_loss` would come back as `val/test_loss`).
        return token if token in SPLITS and token not in self.splits else None

    def base_of(self, name: str) -> str:
        """The canonical base name for one this engine has its own word for."""
        return self.aliases.get(name, name)

    # -- rendering ----------------------------------------------------------------------------
    def word_for(self, split: str) -> str:
        """This engine's word for ``split`` — the canonical one when it has none of its own."""
        return self.splits.get(split, split)


#: Keras 3 and the two hand-written loops that follow its shape: bare training keys, ``val_``
#: prefixed validation ones. The aliases are the four names ``keras.Model.compile()`` resolves
#: ``"accuracy"`` into depending on the loss and the target shape — see
#: ``sonair.tensorflow_classifier._as_metric``, which documents the same contextual mapping.
_KERAS = MetricVocabulary(
    separator="_",
    splits={"train": "", "val": "val", "test": "test"},
    default_split="train",
    aliases={
        "sparse_categorical_accuracy": "accuracy",
        "categorical_accuracy": "accuracy",
        "binary_accuracy": "accuracy",
        "acc": "accuracy",
    },
)

#: The canonical shape, and what the torch / Lightning backends already emit.
_CANONICAL = MetricVocabulary(separator=CANONICAL_SEPARATOR)

#: One vocabulary per framework. ``jax`` / ``mlx`` share the Keras shape because that is the
#: spelling their ecosystems' own examples use, and ``tensorflow`` shares it because ``tf.keras``
#: IS Keras 3. Every value is looked up, never mutated.
VOCABULARIES: Dict[str, MetricVocabulary] = {
    "matrainer": _CANONICAL,
    "torch": _CANONICAL,
    "lightning": _CANONICAL,
    "keras": _KERAS,
    "tensorflow": _KERAS,
    "jax": _KERAS,
    "mlx": _KERAS,
    # fastai's Recorder writes `train_loss`, `valid_loss`, then the metrics — which it computes
    # on the VALIDATION split, unprefixed. An unprefixed key here is therefore a val metric, and
    # reading it as a training one is the mistake this table exists to prevent.
    "fastai": MetricVocabulary(
        separator="_",
        splits={"train": "train", "val": "valid"},
        default_split="val",
    ),
    # `transformers.Trainer` writes `train_*` on its training entries and `eval_*` on its
    # evaluation ones; a bare `loss` in `state.log_history` is the training loss.
    "huggingface": MetricVocabulary(
        separator="_",
        splits={"train": "train", "val": "eval", "test": "test"},
        default_split="train",
    ),
}


def vocabulary(framework: str) -> MetricVocabulary:
    """The :class:`MetricVocabulary` for ``framework``.

    An unknown name is an ERROR listing the known ones rather than a silent fall-back to the
    canonical vocabulary: falling back would rename nothing and look like the mapping simply had
    no work to do, which is indistinguishable from a typo in a config.
    """
    try:
        return VOCABULARIES[framework]
    except KeyError:
        raise ValueError(f"unknown framework {framework!r}; known: {', '.join(sorted(VOCABULARIES))}") from None


def parse_metric_name(name: str, source: str = "matrainer", split: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Split ``name`` into ``(split, base)`` as ``source`` spells it.

    ``split`` is ``None`` for an axis key (``epoch`` / ``step``), which belongs to no split and is
    returned unchanged. An explicit ``split`` argument overrides the vocabulary's ``default_split``
    for keys that carry no prefix of their own — that is how a caller reports a test pass with the
    same code that reports training.

    The canonical ``<split>/<base>`` form is recognised WHATEVER the source vocabulary, which is
    what makes the mapping idempotent: a backend that already names things canonically can declare
    its real source and still pass through unchanged instead of growing a second prefix.
    """
    if name in AXIS_KEYS:
        return None, name
    head, sep, rest = name.partition(CANONICAL_SEPARATOR)
    if sep and head in SPLITS:
        return head, rest
    vocab = vocabulary(source)
    # The vocabulary's own separator first, then `_` as a fallback for the ones that spell the
    # split with a `/`. A `LightningModule` is free to log `train_loss` beside `val/accuracy` —
    # nothing stops it — and reading the first as an unprefixed metric would record
    # `train/train_loss`, a column no other engine fills. `_` is only ever tried when no `/`
    # split was found, so `train/loss_step` keeps its base name intact.
    for separator in (vocab.separator, "_"):
        head, sep, rest = name.partition(separator)
        if sep:
            prefixed = vocab.split_of(head)
            if prefixed is not None:
                return prefixed, vocab.base_of(rest)
    return (split or vocab.default_split), vocab.base_of(name)


def render_metric_name(split: Optional[str], base: str, target: str = "matrainer") -> str:
    """The name ``target`` would write for ``(split, base)`` — the inverse of :func:`parse_metric_name`."""
    if split is None:
        return base
    vocab = vocabulary(target)
    word = vocab.word_for(split)
    return f"{word}{vocab.separator}{base}" if word else base


@configurable
class MetricNames:
    """Translate a mapping of metrics from one engine's spelling into another's.

    Wire it wherever metrics are reported and one top-level YAML key reaches every backend::

        metric_names: !class:matrainer.MetricNames        # the default: canonical everywhere
          renames: {top_k_accuracy: accuracy_top5}

        metric_names: !class:matrainer.MetricNames        # ...or flatten the lot to Keras spelling
          target: keras

    Zero-arg constructible and side-effect free, per the workspace construction rule — it is a
    pure config carrier, and every translation happens in :meth:`__call__`.

    Args:
        target: The vocabulary to render INTO. ``"matrainer"`` (default) is the canonical
            ``<split>/<name>``; naming an engine writes that engine's spelling instead.
        renames: Canonical base name -> the name to record, applied after the source's own
            aliases are resolved and before the target's spelling is applied — so it reads the
            same whatever the two vocabularies are. Use it for a metric this module cannot know
            about (``{top_k_accuracy: accuracy_top5}``), not to re-do the split prefix.
    """

    def __init__(self, target: Framework = "matrainer", renames: Optional[Mapping[str, str]] = None) -> None:
        self.target = target
        self.renames = dict(renames) if renames else {}

    def __call__(
        self,
        metrics: Mapping[str, float],
        source: Framework = "matrainer",
        split: Optional[Split] = None,
    ) -> Dict[str, float]:
        """``metrics`` re-keyed from ``source``'s spelling into :attr:`target`'s.

        Args:
            metrics: The engine's own ``{name: value}`` mapping.
            source: The engine that produced those names. The CALLER's to state — only it knows
                which loop wrote them.
            split: Which split unprefixed keys were measured on, when it is not the source
                vocabulary's default (a test pass reporting with the training code's names).
        """
        return {self.name(key, source=source, split=split): value for key, value in metrics.items()}

    def name(self, key: str, source: Framework = "matrainer", split: Optional[Split] = None) -> str:
        """One name translated — :meth:`__call__` for a single key."""
        parsed_split, base = parse_metric_name(key, source=source, split=split)
        return render_metric_name(parsed_split, self.renames.get(base, base), target=self.target)


def as_metric_names(value: Any = None) -> MetricNames:
    """A live :class:`MetricNames` from a slot that may hold one, a marker, or ``None``.

    The one line a runnable writes where it reports metrics, so eight backends do not each grow
    their own default-and-flow dance. Two things it absorbs: a `metric_names:` wired in YAML
    reaches a CONSTRUCTOR slot as a deferred ``Class`` marker (only the instance form ``!class:X()``
    builds eagerly), and an unwired slot must still translate — recording under the engine's own
    names is what this exists to stop, so the default is the canonical vocabulary rather than a
    passthrough.
    """
    return MetricNames() if value is None else cast(MetricNames, flow(value))
