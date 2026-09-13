import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

#多分类模型的两种简单表示
#y=np.array([1,0,0])
#z=np.array([0.2,0.1,-0.1])
#y_pred=np.exp(z)/np.exp(z).sum()
#loss=(-y*np.log(y_pred)).sum()
#print(loss)

#y=torch.LongTensor([0]) #必须使用LongTensor形
#z=torch.Tensor([0.2,0.1,-0.1])
#crtiterion=torch.nn.CrossEntropyLoss()
#loss=crtiterion(z,y)
#print(loss)


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
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)
    def forward(self,x):
        x=x.view(-1,784)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        return self.l5(x) #一层linear一层relu，最后一层不激活
model=Net()
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