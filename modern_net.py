## Preprocessing
import tflearn.datasets.mnist as mnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CUDA = False

## Constants
NUM_EPOCH = 1000
BATCH_SIZE = 1024

## Data
train_x, train_y, test_x, test_y = mnist.load_data(one_hot=True)
train_x = train_x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

test_x = Variable(torch.Tensor(test_x).view(test_x.shape[0], 1, 28, 28))
test_y = Variable(torch.Tensor(np.argmax(test_y, axis=1)).view(test_y.shape[0])).long()

## Network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        if CUDA:
            self.conv1 = nn.Conv2d(1, 6, 5).cuda()
            self.conv2 = nn.Conv2d(6, 16, 5).cuda()
            self.fc1 = nn.Linear(16*4*4, 120).cuda()
            self.fc2 = nn.Linear(120, 84).cuda()
            self.fc3 = nn.Linear(84, 10).cuda()
        else:
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*4*4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


    def accuracy(self):
        result = self.forward(test_x)
        _, preds = result.max(1)
        return (preds == test_y).float().mean()


net = LeNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

## Training
for epoch in xrange(10000):
    batch = np.random.choice(train_x.shape[0], BATCH_SIZE)
    if CUDA: 
        batch_x = Variable(torch.Tensor(train_x[batch]).view(BATCH_SIZE, 1, 28, 28)).cuda()
        batch_y = Variable(torch.Tensor(np.argmax(train_y[batch], axis=1)).view(BATCH_SIZE)).long().cuda()
    else:
        batch_x = Variable(torch.Tensor(train_x[batch]).view(BATCH_SIZE, 1, 28, 28))
        batch_y = Variable(torch.Tensor(np.argmax(train_y[batch], axis=1)).view(BATCH_SIZE)).long()

    optimizer.zero_grad()
    
    result = net.forward(batch_x)
    loss = criterion(result, batch_y)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print "Loss:"
        print loss
        print "Accuracy:"
        print net.accuracy()
