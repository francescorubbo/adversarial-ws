from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        init.normal(self.fc1.weight)
        init.constant(self.fc1.bias,0.)
        self.fc2 = nn.Linear(H, D_out)
        init.normal(self.fc2.weight)
        init.constant(self.fc2.bias,0.)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
