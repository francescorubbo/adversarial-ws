from dataprovider import getToys,getWeakToys,use_cuda
import numpy as np

means = [(18,26),(0.06,0.09),(0.23,0.28)]
stds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
samplesize = 20000
fractions = np.linspace(0.2,0.4,9)
#fractions = [0.4]
print fractions

n_epochs = 100
cp_delay = 50

D_in = 3
H = 30
D_out = 1

_, __, X_val, y_val, X_test, y_test,scaler = getToys(means,stds,samplesize,scale=True)

X_train, y_train = getWeakToys(means,stds,samplesize,fractions,scaler)

from models import Net
from torch import nn
from torch import optim
import torch

model = Net(D_in, H, D_out)

if use_cuda:
    model.cuda()
loss_fn = nn.MSELoss()

learning_rate = 9e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_loss = 1
checkpoint = {}
for t in range(n_epochs):

    permind = np.random.permutation(len(X_train))
    X_val = X_train[permind[0]]
    y_val = y_train[permind[0]][0]
    
    for ind in permind[1:]:
        X = X_train[ind]
        y = y_train[ind][0]
        y_pred = torch.mean(model(X))
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred_val = torch.mean(model(X_val))
    loss_val = loss_fn(y_pred_val, y_val)

    if t>cp_delay:
        is_best = loss_val < best_loss
        best_loss = min(loss_val,best_loss)
        if (is_best):
            checkpoint = {
                'epoch': t,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
                }

    print(t, 'train loss:',loss.data[0],'val loss:',loss_val.data[0])
#    print(t, 'train loss:',loss.data[0])

model.load_state_dict(checkpoint['state_dict'])
y_pred_test = model(X_test)

from pylab import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

bins = np.linspace(0,1.01,100)
hist(y_pred_test[y_test==1].data.numpy(),bins=bins,histtype='stepfilled')
hist(y_pred_test[y_test==0].data.numpy(),bins=bins,histtype='stepfilled')
show()
clf()

fpr,tpr,thres = roc_curve(y_test.data.numpy(), y_pred_test.data.numpy())
area =  auc(fpr, tpr)
print area
plot(fpr, tpr,label='AUC=%1.2f'%area)
legend()
show()
