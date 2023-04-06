from typing import Optional
import torch
import torch.nn as nn
from fromage import utils


# def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
#     return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
#
#
# def contrastive_acc(logits: torch.Tensor, target: Optional[torch.Tensor] = None, topk=(1,)) -> torch.Tensor:
#     """
#     Args:
#       logits: (N, N) predictions.
#       target: (N, num_correct_answers) labels.
#     """
#     assert len(logits.shape) == 2, logits.shape
#     batch_size = logits.shape[0]
#
#     if target is None:
#         target = torch.arange(len(logits), device=logits.device)
#         return utils.accuracy(logits, target, -1, topk)
#     else:
#         assert len(target.shape) == 2, target.shape
#         with torch.no_grad():
#             maxk = max(topk)
#             if logits.shape[-1] < maxk:
#                 print(f"[WARNING] Less than {maxk} predictions available. Using {logits.shape[-1]} for topk.")
#             maxk = min(maxk, logits.shape[-1])
#
#             # Take topk along the last dimension.
#             _, pred = logits.topk(maxk, -1, True, True)  # (N, topk)
#             assert pred.shape == (batch_size, maxk)
#
#             target_expand = target[:, :, None].repeat(1, 1, maxk)  # (N, num_correct_answers, topk)
#             pred_expand = pred[:, None, :].repeat(1, target.shape[1], 1)  # (N, num_correct_answers, topk)
#             correct = pred_expand.eq(target_expand)  # (N, num_correct_answers, topk)
#             correct = torch.any(correct, dim=1)  # (N, topk)
#
#             res = []
#             for k in topk:
#                 any_k_correct = torch.clamp(correct[:, :k].sum(1), max=1)  # (N,)
#                 correct_k = any_k_correct.float().sum(0, keepdim=True)
#                 res.append(correct_k.mul_(100.0 / batch_size))
#             return res


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(2).sum(1).mean()


class RelaxedListMLE(nn.Module):
    def __init__(self, eps=1e-8, pad=-1):
        super(RelaxedListMLE, self).__init__()

        self.pad = pad
        self.eps = eps

    def forward(self, y_pred, y_true):
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == self.pad

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0
        return torch.mean(torch.sum(observation_loss, dim=1))
