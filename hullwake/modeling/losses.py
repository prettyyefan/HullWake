from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def foreground_logit(class_logits: torch.Tensor) -> torch.Tensor:
    """Collapse one or more foreground classes into a foreground-vs-background logit."""
    if class_logits.ndim != 2 or class_logits.shape[1] < 2:
        raise ValueError("class_logits must have shape [N, C] with C >= 2")
    return torch.logsumexp(class_logits[:, 1:], dim=1) - class_logits[:, 0]


def wake_consistency_loss(
    class_logits: torch.Tensor,
    attenuated_class_logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(positive_mask):
        return _zero(class_logits)
    log_p = F.log_softmax(class_logits[positive_mask], dim=-1)
    log_q = F.log_softmax(attenuated_class_logits[positive_mask], dim=-1)
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=-1).mean()


def wake_dominance_loss(
    hull_class_logits: torch.Tensor,
    wake_class_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    lambda_negative: float,
    negative_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    positive = labels > 0
    negative = labels == 0
    hull_score = foreground_logit(hull_class_logits)
    wake_score = foreground_logit(wake_class_logits)
    positive_loss = (
        F.relu(float(margin) + wake_score[positive] - hull_score[positive]).mean()
        if torch.any(positive)
        else _zero(wake_score)
    )
    if torch.any(negative):
        negative_terms = F.binary_cross_entropy_with_logits(
            wake_score[negative], torch.zeros_like(wake_score[negative]), reduction="none"
        )
        if negative_weights is not None:
            negative_terms = negative_terms * negative_weights[negative]
        negative_loss = negative_terms.mean()
    else:
        negative_loss = _zero(wake_score)
    return positive_loss + float(lambda_negative) * negative_loss


def hull_wake_decorrelation_loss(
    hull_features: torch.Tensor, wake_features: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if hull_features.shape != wake_features.shape:
        raise ValueError("hull_features and wake_features must have equal shapes")
    if hull_features.shape[0] < 2:
        return _zero(hull_features)
    hull_centered = hull_features - hull_features.mean(dim=0, keepdim=True)
    wake_centered = wake_features - wake_features.mean(dim=0, keepdim=True)
    numerator = torch.sum(hull_centered * wake_centered, dim=-1)
    denominator = (
        torch.linalg.vector_norm(hull_centered, dim=-1)
        * torch.linalg.vector_norm(wake_centered, dim=-1)
        + float(epsilon)
    )
    return torch.mean(torch.square(numerator / denominator))


def wake_response_loss(
    wake_logits: torch.Tensor,
    wake_targets: torch.Tensor,
    valid_proposals: torch.Tensor,
) -> torch.Tensor:
    if wake_logits.shape != wake_targets.shape:
        raise ValueError("wake_logits and wake_targets must have equal shapes")
    if valid_proposals.ndim != 1 or valid_proposals.shape[0] != wake_logits.shape[0]:
        raise ValueError("valid_proposals must have shape [N]")
    if not torch.any(valid_proposals):
        return _zero(wake_logits)
    return F.binary_cross_entropy_with_logits(
        wake_logits[valid_proposals], wake_targets[valid_proposals]
    )
