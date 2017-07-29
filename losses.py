import torch
from torch import nn

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        loss = torch.mean((input - target)*(input - target))
        return loss

class MSELoss_LLP(nn.Module):
### need to implement back-prop

    def __init__(self):
        super(MSELoss_LLP, self).__init__()

    def forward(self, input, target):
        predicted = torch.mean(input)
#        print 'predicted',predicted.data[0]
        expected = torch.mean(target)
#        print 'expected',expected.data[0]
#        loss = torch.mean(input)-torch.mean(target)
        loss = predicted - expected
        return torch.pow(loss,2)
