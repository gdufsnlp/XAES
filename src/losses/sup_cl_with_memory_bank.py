from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

        self.device = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )

    def forward(self, features, labels, memory_features=None, memory_labels=None):

        if memory_features is None and memory_labels is None:
            labels = labels.contiguous().view(-1, 1)
            anchor_feature = features
            mask = torch.eq(labels, labels.T).float().to(self.device)
            contrast_feature = anchor_feature
            logits_mask = torch.ones_like(mask).to(self.device)
            self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
            logits_mask[:, :mask.size()[0]] = logits_mask[:, :mask.size()[0]].clone() * self_contrast_mask.to(self.device)
        elif memory_features is not None and memory_labels is not None:
            anchor_feature = features
            labels = labels.contiguous().view(-1, 1)
            memory_labels = memory_labels.contiguous().view(-1, 1)
            contrast_labels = torch.cat([labels, memory_labels])
            mask = torch.eq(labels, contrast_labels.T).float().to(self.device)
            contrast_feature = torch.cat([anchor_feature, memory_features]).detach()
            logits_mask = torch.ones_like(mask).to(self.device)
            self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
            logits_mask[:, :mask.size()[0]] = logits_mask[:, :mask.size()[0]].clone() * self_contrast_mask.to(self.device)

        # compute logits
        anchor_norm = torch.norm(anchor_feature, dim=1)
        contrast_norm = torch.norm(contrast_feature, dim=1)
        anchor_feature = anchor_feature / (anchor_norm.unsqueeze(1))
        contrast_feature = contrast_feature / (contrast_norm.unsqueeze(1))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask * logits_mask
        nonzero_index = torch.where(mask.sum(1) != 0)[0]
        if len(nonzero_index) == 0:
            return torch.tensor(0).float().to(self.device)
        # compute log_prob
        mask = mask[nonzero_index]
        logits_mask = logits_mask[nonzero_index]
        logits = logits[nonzero_index]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
