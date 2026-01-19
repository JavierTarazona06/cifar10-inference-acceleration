import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute distillation loss = alpha * KL + (1 - alpha) * CE.

    Args:
        student_logits: raw outputs from student (batch, num_classes)
        teacher_logits: raw outputs from teacher (batch, num_classes)
        targets: hard labels (batch,)
        alpha: weight for KL term (soft targets)
        temperature: softmax temperature T

    Returns:
        total_loss: scalar loss
        details: dict with kl_loss, ce_loss, temperature, alpha

    Example:
        total, parts = distillation_loss(s_logits, t_logits, labels, alpha=0.7, temperature=4.0)
        total.backward()
    """

    # Soft targets from teacher
    T = temperature
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    student_log_prob = F.log_softmax(student_logits / T, dim=1)

    kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)
    ce_loss = F.cross_entropy(student_logits, targets)
    total_loss = alpha * kl_loss + (1.0 - alpha) * ce_loss

    return total_loss, {
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "alpha": float(alpha),
        "temperature": float(T),
    }
