import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

batch_size = 100
epochs = 5

train_dataset = datasets.KMNIST('./data', train=True, download=True, transform=transforms.ToTensor())

validation_dataset = datasets.KMNIST('./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print(model)

def train(epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data.item()))

def validate(loader, loss_vector, accuracy_vector):
    model.eval()
    loss, correct = 0, 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    loss /= len(loader)
    loss_vector.append(loss)
    accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)
    accuracy_vector.append(accuracy.item())
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(validation_loader.dataset), accuracy))

losst, acct = [], []
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    print('\n***Checking loss/acc at the end of the epoch***\nTrain: ')
    validate(train_loader, losst, acct)
    print('Validation: ')
    validate(validation_loader, lossv, accv)

# Homework: plot accuracy and loss for training and validation dataset, with epoch on the x-axis. (The point of this homework is to get used to performing simple experiments using PyTorch or other DL libraries.)
# Reference: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# Some of the code were taken from: https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
