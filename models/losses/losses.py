import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanSquaredAngularLoss(torch.nn.Module):
    """
    Calculate mean squared angular loss between targets and predictions.
    """

    def __init__(self):
        super(MeanSquaredAngularLoss, self).__init__()
        self.cos_similarity = nn.CosineSimilarity()

    def forward(self, predicted, target):
        return self._mean_squared_angular_loss(predicted, target)

    def _mean_squared_angular_loss(self, predicted, target):

        # Calculate angular distance
        # NOTE: the gradient of acos is not defined at +1 and -1, but
        # cos_similarity returns values in the interval [-1,+1].Scale it down
        # just a bit to avoid nan.
        loss = torch.acos(self.cos_similarity(predicted, target) * 0.9999)

        # Calculate mean squared loss
        mean_squared_loss = torch.mean(torch.pow(loss, 2))

        return mean_squared_loss



if __name__ == "__main__":
    
    target = torch.tensor([
        [[[-1.]],[[0.]]],
        #[[[1., 1.]],[[0., 0.]]]
    ])
    pred = torch.tensor([
        [[[1.]],[[0.]]], 
        #[[[2., 2.]],[[0., 0.]]]
    ])

    msa = MeanSquaredAngularLoss()

    print(target.shape, pred.shape)

    print(msa(pred, target))
