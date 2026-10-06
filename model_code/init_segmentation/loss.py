import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss for class-imbalanced binary segmentation.

    Focal loss down-weights easy negatives, forcing the network to focus on
    hard/uncertain pixels — the main failure mode when plateauing near 0.1.

    Args:
        alpha:        Focal weighting factor for positive class (0.25 typical).
        gamma:        Focal modulating exponent.  Higher = more focus on hard
                      pixels.  Tune in [0.5, 5.0] via Optuna.
        dice_weight:  Weight of the Dice component.
        focal_weight: Weight of the Focal component.
        smooth:       Laplace smoothing for Dice denominator.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ---- Focal part ------------------------------------------------
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce
        focal_loss = focal.mean()

        # ---- Dice part -------------------------------------------------
        pred_sig = torch.sigmoid(y_pred)
        inter = (pred_sig * y_true).sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
            pred_sig.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
        )
        dice_loss = dice_loss.mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc


class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        smooth: float = 1.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            # Smooth labels towards 0.5: prevents overconfident BCE
            y_bce = y_true * (1.0 - self.label_smoothing) + self.label_smoothing * 0.5
        else:
            y_bce = y_true
        bce = F.binary_cross_entropy_with_logits(y_pred, y_bce)
        dice = self.dice(torch.sigmoid(y_pred), y_true)
        return self.bce_weight * bce + self.dice_weight * dice