import pytest

torch = pytest.importorskip("torch")

from hullwake.modeling.wake_extractor import ProposalAlignedWakeExtractor  # noqa: E402


def test_bidirectional_corridors_are_opposite_and_proposal_anchored():
    extractor = ProposalAlignedWakeExtractor(
        input_channels=256,
        embedding_dim=16,
        grid_size=8,
        length_max_ratio=3.0,
        width_max_ratio=1.0,
        fpn_strides=[4, 8, 16, 32],
        epsilon=0.01,
    )
    with torch.no_grad():
        extractor.orientation_head.weight.zero_()
        extractor.orientation_head.bias.zero_()
        extractor.length_head.weight.zero_()
        extractor.length_head.bias.zero_()
        extractor.width_head.weight.zero_()
        extractor.width_head.bias.zero_()
    box = torch.tensor([[10.0, 20.0, 30.0, 30.0]])
    hull = torch.zeros(1, 16)
    points, _, orientation, length, width = extractor._make_points(box, hull)
    assert torch.allclose(orientation, torch.tensor([0.0]))
    assert torch.allclose(length, torch.tensor([30.0]))
    assert torch.allclose(width, torch.tensor([10.0]))
    center = torch.tensor([20.0, 25.0])
    plus_mean_x = points[0, 0, -8:, 0].mean()
    minus_mean_x = points[0, 1, -8:, 0].mean()
    assert plus_mean_x < center[0]
    assert minus_mean_x > center[0]
    assert torch.allclose(
        points[0, 0, :8].mean(dim=0), center, atol=1e-5
    )
