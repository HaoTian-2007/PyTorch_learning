import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

#相比p10.py正确率没有明显提升

batch_size=64
transform=transforms.Compose([ #Compose用于把一系列操作变为管线
    transforms.ToTensor(), #W*H变成1*W*H
    transforms.Normalize((0.1307,),(0.3081),) #均值 标准差                      
])
train_dataset=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transform)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transform)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.conv3=torch.nn.Conv2d(20,40,kernel_size=3) #卷积核需要调小一点
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc1=torch.nn.Linear(40,30) #这里输入维度也改了
        self.fc2=torch.nn.Linear(30,20)
        self.fc3=torch.nn.Linear(20,10)
    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=F.relu(self.pooling(self.conv3(x)))
        x=x.view(batch_size,-1) #-1是自动计算
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)
model=Net()
#此处应有gpu迁移的代码
crtiterion=torch.nn.CrossEntropyLoss() #交叉熵
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0): #取出train_loader中的数据
        inputs,target=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=crtiterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            print(epoch,batch_idx,running_loss)
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad(): #不需要梯度
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _, predicted=torch.max(outputs.data,dim=1) #每一行中找值最大的，同时保存值和索引
            total+=labels.size(0) #label是N*1维，这一行加上的就是N
            correct+=(predicted==labels).sum().item() #把预测对的数量求和
    print(correct/total)

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()
    #非常简洁干净的main