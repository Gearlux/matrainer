"""Class-imbalance weighting for torch losses (``marainer.torch.balancing``).

The policy is three steps that belong together — capability check, weight
derivation, injection — so these pin each step AND the composed call, including
the degradations (a loss that takes no ``weight``, an empty target set) that
must leave training unweighted rather than raising.
"""

from typing import Any

import pytest
import torch
from confluid.fluid import Class as ConfluidClass
from torch import nn

from marainer.torch import apply_class_weights, inverse_frequency_weights, loss_accepts_weight

# --------------------------------------------------------------------------- #
# loss_accepts_weight
# --------------------------------------------------------------------------- #


def test_accepts_weight_for_a_weight_capable_loss() -> None:
    assert loss_accepts_weight(nn.CrossEntropyLoss()) is True


def test_does_not_accept_weight_for_a_loss_without_the_arg() -> None:
    assert loss_accepts_weight(nn.MSELoss()) is False


def test_accepts_weight_reads_a_deferred_fluid_without_building_it() -> None:
    """The check runs BEFORE the loss exists, so a trainer can skip the target walk."""
    assert loss_accepts_weight(ConfluidClass(nn.CrossEntropyLoss)) is True
    assert loss_accepts_weight(ConfluidClass(nn.MSELoss)) is False


def test_unresolvable_target_degrades_to_false() -> None:
    assert loss_accepts_weight(ConfluidClass("not.a.real.Loss")) is False


# --------------------------------------------------------------------------- #
# inverse_frequency_weights
# --------------------------------------------------------------------------- #


def test_uniform_distribution_weights_every_class_equally() -> None:
    weights = inverse_frequency_weights([0, 1, 2, 0, 1, 2], num_classes=3)
    assert weights is not None
    assert torch.allclose(weights, torch.ones(3))


def test_rare_classes_weigh_more_than_common_ones() -> None:
    weights = inverse_frequency_weights([0, 0, 0, 1], num_classes=2)
    assert weights is not None
    # w = total / (num_classes * count): 4/(2*3) and 4/(2*1)
    assert torch.allclose(weights, torch.tensor([2 / 3, 2.0]))
    assert weights[1] > weights[0]


def test_an_unobserved_class_gets_zero_not_infinity() -> None:
    weights = inverse_frequency_weights([0, 0], num_classes=3)
    assert weights is not None
    assert weights[1].item() == 0.0 and weights[2].item() == 0.0
    assert torch.isfinite(weights).all()


def test_out_of_range_ids_are_ignored_rather_than_raising() -> None:
    """A stray label must not abort a training run."""
    weights = inverse_frequency_weights([0, 1, 99, -1], num_classes=2)
    assert weights is not None
    assert torch.allclose(weights, torch.ones(2))


def test_no_observations_returns_none() -> None:
    """`None` distinguishes "no weights" from "all-zero weights"."""
    assert inverse_frequency_weights([], num_classes=3) is None
    assert inverse_frequency_weights([7, 8], num_classes=3) is None


def test_weights_are_float32() -> None:
    weights = inverse_frequency_weights([0, 1], num_classes=2)
    assert weights is not None and weights.dtype is torch.float32


# --------------------------------------------------------------------------- #
# apply_class_weights — the composed policy
# --------------------------------------------------------------------------- #


def test_apply_injects_weights_into_a_live_loss() -> None:
    loss = nn.CrossEntropyLoss()
    returned, weights = apply_class_weights(loss, [0, 0, 0, 1], num_classes=2)

    assert returned is loss  # a live module is mutated in place
    assert weights is not None
    assert returned.weight is not None
    assert torch.allclose(returned.weight, weights)


def test_apply_flows_a_fluid_and_returns_the_built_loss() -> None:
    """A Fluid cannot be mutated in place — hence the returned loss."""
    marker = ConfluidClass(nn.CrossEntropyLoss)
    returned, weights = apply_class_weights(marker, [0, 0, 0, 1], num_classes=2)

    assert isinstance(returned, nn.CrossEntropyLoss), "the Fluid must be flowed, not returned as a marker"
    assert weights is not None
    assert returned.weight is not None
    assert torch.allclose(returned.weight, weights)


def test_apply_leaves_a_weightless_loss_untouched() -> None:
    loss = nn.MSELoss()
    returned, weights = apply_class_weights(loss, [0, 0, 1], num_classes=2)

    assert returned is loss
    assert weights is None
    assert not hasattr(loss, "weight") or loss.weight is None


def test_apply_leaves_the_loss_untouched_when_nothing_was_observed() -> None:
    loss = nn.CrossEntropyLoss()
    returned, weights = apply_class_weights(loss, [], num_classes=3)

    assert returned is loss
    assert weights is None
    assert returned.weight is None


def test_apply_warns_when_the_loss_cannot_be_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diagnostic moves with the policy — a silent no-op would hide a config mistake."""
    from marainer.torch import balancing

    warnings: list = []
    monkeypatch.setattr(balancing.logger, "warning", lambda msg: warnings.append(msg))

    apply_class_weights(nn.MSELoss(), [0, 1], num_classes=2)

    assert any("no 'weight' arg" in str(w) for w in warnings), warnings


def test_apply_is_usable_as_the_caller_rebinding_idiom() -> None:
    """The documented call shape: rebind both the loss and the weights."""
    loss_fn: Any = ConfluidClass(nn.CrossEntropyLoss)
    loss_fn, class_weights = apply_class_weights(loss_fn, [0, 0, 1, 2], num_classes=3)

    assert class_weights is not None and class_weights.shape == (3,)
    out = loss_fn(torch.randn(4, 3), torch.tensor([0, 0, 1, 2]))
    assert torch.isfinite(out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
