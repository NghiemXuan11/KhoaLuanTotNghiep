import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BinaryCrossEntropy(nn.Module):
    def __init__(self, weight, ignore_lb=255, *args, **kwargs):
        super(BinaryCrossEntropy, self).__init__()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels)
        return loss

class InstanceLoss(nn.Module):
    def __init__(self, hm=2.0, sampling_size=2000):
        super(InstanceLoss, self).__init__()
        self.epsilon = 1e-9
        self.hm = hm
        self.sampling_size = sampling_size

    def forward(self, input, target):
        loss = torch.zeros(1, device=input.device)
        target = target.argmax(dim=1)

        batchsize, classes, height, width = input.size()
        p = F.softmax(input, dim=1)

        for i in range(batchsize):
            target_nonzero_pixels = torch.nonzero(target[i].view(-1), as_tuple=False).squeeze()
            if target_nonzero_pixels.numel() > 0:
                sampling_indices = torch.randint(0, target_nonzero_pixels.size(0), (self.sampling_size,), device=input.device)
                indices_to_keep = target_nonzero_pixels[sampling_indices]

                p_sampled = p[i][1:].view(classes - 1, -1)[:, indices_to_keep]
                tt_sampled = target[i].view(-1)[indices_to_keep]

                R = (tt_sampled.unsqueeze(1) == tt_sampled.unsqueeze(0)).float().to(input.device)
                diag = (1 - torch.eye(self.sampling_size, device=input.device))

                tik = p_sampled.unsqueeze(2).expand(-1, -1, self.sampling_size)
                tjk = p_sampled.unsqueeze(1).expand(-1, self.sampling_size, -1)

                dkl = torch.sum(tik * (torch.log(tik + self.epsilon) - torch.log(tjk + self.epsilon)), dim=0)
                loss += torch.sum(diag * ((R * dkl) + (1 - R) * torch.clamp(self.hm - dkl, min=0.0))) / (self.sampling_size * (self.sampling_size - 1))

            bg_mask = (target[i] == 0).float()
            bg_term = torch.sum(bg_mask * torch.log(p[i][0] + self.epsilon))
            fg_term = torch.sum((1 - bg_mask) * torch.log(torch.sum(p[i][1:], dim=0) + self.epsilon))
            loss += -(1 / (width * height)) * (fg_term + bg_term)

        loss = loss.mean()
        return loss / batchsize

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss cho bài toán segmentation đa lớp.

        Args:
            alpha (list or tensor, optional): Hệ số cân bằng giữa các lớp (len = num_classes). Nếu None, không dùng.
            gamma (float, optional): Hệ số điều chỉnh độ tập trung vào các điểm khó. Mặc định 2.0.
            reduction (str, optional): Cách tính tổng loss ['mean', 'sum', 'none']. Mặc định 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Tính Focal Loss cho multi-class segmentation.

        Args:
            logits (Tensor): Đầu ra của mô hình (chưa qua softmax), kích thước (N, C, H, W).

        Returns:
            Tensor: Giá trị loss.
        """
        # Đưa logits về dạng xác suất với softmax
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        # Tính cross-entropy loss cơ bản
        ce_loss = -targets * torch.log(probs + 1e-7)

        # Tính trọng số focal (1 - P_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Nếu có alpha, áp dụng trọng số cho từng lớp
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)  # Đưa alpha về device phù hợp
            focal_loss = self.alpha.view(1, -1, 1, 1) * focal_loss

        # Tính tổng loss theo reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
