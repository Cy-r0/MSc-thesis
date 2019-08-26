import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_squared_angular_loss(predicted, target):
    """
    Implement squared angular loss
    """

    # Normalise vectors and make them always slightly less than 1
    # to avoid the loss being exactly zero
    #predicted = torch.mul(F.normalize(predicted, dim=1), 0.99999)
    #target = torch.mul(F.normalize(target, dim=1), 0.99999)

    # Initialise cosine similarity
    # NOTE: it doesn't need the vectors to be normalised
    cos_distance = nn.CosineSimilarity()

    # Calculate angular distance
    loss = torch.div(torch.acos(cos_distance(predicted, target)), math.pi)
    
    
    print(cos_distance(predicted, target))
    print(torch.acos(cos_distance(predicted, target)))
    print(loss)

    # Calculate mean squared loss
    mean_squared_loss = torch.mean(torch.pow(loss, 2))

    return mean_squared_loss

    





if __name__ == "__main__":
    
    target = torch.tensor([
        [[[0., 0.]],[[1., 1.]]],
        #[[[1., 1.]],[[0., 0.]]]
    ])
    pred = torch.tensor([
        [[[1., 1.]],[[0., 0.]]], 
        #[[[2., 2.]],[[0., 0.]]]
    ])

    print(target.shape, pred.shape)

    print(mean_squared_angular_loss(pred, target))
