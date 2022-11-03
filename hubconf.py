from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

import matplotlib
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

from torchvision import datasets
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from torchmetrics import Precision, Recall, F1Score, Accuracy

device='cuda' if torch.cuda.is_available() else 'cpu'

class cs19b037AdvancedNN(nn.Module):
    def __init__(self, modules_list, num_classes, in_features):
        super().__init__()
        self.num_classes = num_classes
        self.linears = nn.ModuleList(modules_list)
        self.flatten = nn.Flatten()
        self.fc1=nn.Linear(in_features=in_features,out_features=self.num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)

        x = self.flatten(x)
        x=self.fc1(x)
        x = self.softmax(x)
        return x

def get_model(config_list,train_data_loader):
    modules_list = []
    for module in config_list:
        modules_list.append(nn.Conv2d(in_channels=module[0], out_channels=module[1], kernel_size=module[2], stride=module[3], padding=module[4]))
    in_features=None

    w = 0
    h = 0
    num_channels = 0
    num_classes = 0
    for (X, y) in train_data_loader:
        num_channels = X.shape[1]
        w = X.shape[2]
        h = X.shape[3]
        num_classes = max(num_classes,torch.max(y).item()-torch.min(y).item()+1)
        num_classes_g = num_classes

    X,_=train_data_loader.dataset[0]
    print(X.shape)

    for layer in modules_list:
        X=layer(X)
    in_features=X.shape[0]*X.shape[1]*X.shape[2]

    model2 = cs19b037AdvancedNN(modules_list=(modules_list),num_classes=num_classes,in_features=in_features).to(device)

    return model2,num_classes

def loss_fn(ypred, yground):
    x=-(yground * torch.log(ypred))
    x=torch.sum(x)

    return x

def get_lossfn_and_optimizer(mymodel):
    optimizer = torch.optim.SGD(model2.parameters(), lr=1e-3)

    return optimizer, loss_fn

def train(train_dataloader,model1,loss_fn1,optimizer1,num_classes,epochs=3):
    for epoch in range(epoch):
        model1.train()
        train_loss = 0
        correct = 0
        for i, (X,y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            ypred = model1(X)
            oh = torch.nn.functional.one_hot(y, num_classes=num_classes)
            loss = loss_fn1(ypred,oh)

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            train_loss += loss
            correct += (ypred.argmax(1) == y).type(torch.float).sum().item()
        print("Epoch", epoch, "accuracy:",
              correct/len(train_dataloader.dataset))

    return model1


def test(model1,test_data_loader,num_classes):
    size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)
    model1.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes=num_classes)
            pred = model1(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy = Accuracy().to(device)
    print('Accuracy :', accuracy(pred,y))
    precision = Precision(average = 'macro', num_classes = num_classes).to(device)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = num_classes).to(device)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = num_classes).to(device)
    print('f1_score :', f1_score(pred,y))
