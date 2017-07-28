from dataprovider import getToys,use_cuda

means = [(18,26),(0.06,0.09),(0.23,0.28)]
stds  = [(7,8),  (0.04,0.04),(0.05,0.04)]
samplesize = 20000

batch_size = 128
n_epochs = 200

X_train, y_train, X_val, y_val, X_test, y_test = getToys(means,stds,samplesize)

print X_train.size(),y_train.size()
print X_test.size(),y_test.size()

D_in = 3
H = 3
D_out = 1

from models import Net
from torch import nn
from torch import optim
from torch import randperm

model = Net(D_in, H, D_out)

if use_cuda:
    model.cuda()

loss_fn = nn.MSELoss()

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for t in range(n_epochs):

    permind = randperm(X_train.size()[0])
    X_batches = X_train[permind].split(batch_size)
    y_batches = y_train[permind].split(batch_size)

    for X,y in zip(X_batches,y_batches):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        y_pred_val = model(X_val)
        loss_val = loss_fn(y_pred_val, y_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(t, 'train loss:',loss.data[0],'val loss:',loss_val.data[0])

y_pred_test = model(X_test)

from pylab import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

bins = np.linspace(0,1.01,100)
hist(y_pred_test[y_test==1].data.numpy(),bins=bins,histtype='stepfilled')
hist(y_pred_test[y_test==0].data.numpy(),bins=bins,histtype='stepfilled')

fpr,tpr,thres = roc_curve(y_test.data.numpy(), y_pred_test.data.numpy())
area =  auc(fpr, tpr)
print area
#plot(fpr, tpr,label='AUC=%1.2f'%area)
#legend()
show()
