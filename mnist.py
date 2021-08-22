# -*- coding: utf-8 -*-
"""
 @Time    : 2021/8/21 23:50
 @Author  : meehom
 @Email   : meehomliao@163.com
 @File    : mnist.py
 @Software: PyCharm
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import optim

'''BP in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
class Bp(nn.Module):
    def __init__(self):
        super(Bp, self).__init__()
        self.lr1 = nn.Linear(784, 128)
        self.lr2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.lr1(x)
        x = F.relu(x)
        x = self.lr2(x)
        x = F.softmax(x)
        return x
def train():
    learning_rate = 1e-3
    batch_size = 100
    epoches = 5
    correct_sum = 0
    bp = Bp()
    trans_img = transforms.Compose([transforms.ToTensor()])
    optimizer = optim.Adam(bp.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    for i in range(epoches):
        running_acc_train = 0
        for (img, label) in trainloader:
            running_loss = 0
            optimizer.zero_grad()
            input = img.reshape(100, 784)  # 784
            output = bp(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _,predict_train = torch.max(output, 1)
            correct_num = (predict_train == label).sum()
            running_acc_train += correct_num.item()
        running_acc = running_acc_train/len(trainset)
        print("epoch", i, "running-acc is :", running_acc)
    return bp
def test(bp):
    batch_size = 100
    trans_img = transforms.Compose([transforms.ToTensor()])
    testset = MNIST('./data', train=False, transform=trans_img, download=True)
    testloader = DataLoader(testset, batch_size, shuffle=False)
    running_acc_test = 0
    for (img, label) in testloader:
        input = img.reshape(100, 784)
        output = bp(input)
        _, predict_train = torch.max(output, 1)
        correct_num = (predict_train == label).sum()
        running_acc_test += correct_num.item()
    running_acc = running_acc_test / len(testset)
    print("test_acc is:", running_acc)

if __name__ == '__main__':
    bp = train()
    test(bp)