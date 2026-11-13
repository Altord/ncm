"""Small CPU checks for the optional codec distillation objective."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from training.reconstruction_loss import topk_teacher_distillation_loss


def main():
    logits = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 3., 4.]]], requires_grad=True)
    targets = torch.tensor([[3, 2, 3]])
    valid = torch.tensor([[True, True, False]])
    indices = torch.tensor([[[3, 2], [0, 2], [3, 2]]])
    probabilities = torch.tensor([[[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]])
    loss, count = topk_teacher_distillation_loss(logits, targets, valid, [(indices, probabilities)])
    expected = -(F.log_softmax(logits[0, 0], dim=-1)[3] * .75 + F.log_softmax(logits[0, 0], dim=-1)[2] * .25)
    assert count.item() == 1 and torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0 and logits.grad[0, 1:].abs().sum() == 0
    wrong_targets = torch.zeros_like(targets)
    empty_loss, count = topk_teacher_distillation_loss(logits, wrong_targets, torch.tensor([[True, False, False]]), [(indices, probabilities)])
    assert count.item() == 0 and empty_loss.item() == 0
    print('Codec distillation loss checks passed.')


if __name__ == '__main__':
    main()
