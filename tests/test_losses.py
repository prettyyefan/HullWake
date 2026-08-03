import pytest

torch = pytest.importorskip("torch")

from hullwake.modeling.losses import (  # noqa: E402
    hull_wake_decorrelation_loss,
    wake_consistency_loss,
    wake_dominance_loss,
)


def test_consistency_is_zero_for_identical_predictions():
    logits = torch.tensor([[0.1, 1.2], [1.0, -0.2]])
    loss = wake_consistency_loss(logits, logits.clone(), torch.tensor([True, False]))
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_dominance_margin_penalizes_wake_dominance():
    hull = torch.tensor([[0.0, 0.2], [1.0, 0.0]])
    wake = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([1, 0])
    loss = wake_dominance_loss(
        hull, wake, labels, margin=0.2, lambda_negative=1.0
    )
    assert loss.item() > 0.0


def test_decorrelation_is_finite_for_constant_features():
    features = torch.ones(4, 8)
    loss = hull_wake_decorrelation_loss(features, features, epsilon=0.01)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
