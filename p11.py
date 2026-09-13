import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

#准确率稳定在了很高的水平，甚至达到了0.99，开心:)
#只修改了class，内含GoogLeNet和Residual Block两个模型，请自行删改标出来的代码前面的‘#’

class GoogLeNet(torch.nn.Module):
    def __init__(self,in_channels):
        super(GoogLeNet,self).__init__()
        self.branch_pool=torch.nn.Conv2d(in_channels,24,kernel_size=1) #池化通道的一维卷积
        self.branch1x1=torch.nn.Conv2d(in_channels,16,kernel_size=1) #1乘1通道
        self.branch5x5=torch.nn.Conv2d(16,24,kernel_size=5,padding=2) #5乘5通道
        self.branch3x3_1=torch.nn.Conv2d(16,24,kernel_size=3,padding=1) #3乘3通道
        self.branch3x3_2=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)
    def forward(self,x):
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1) #池化层
        branch_pool=self.branch_pool(branch_pool) #池化通道
        branch1x1=self.branch1x1(x) #1乘1通道
        branch5x5=self.branch5x5(self.branch1x1(x)) #5乘5通道
        branch3x3=self.branch3x3_2(self.branch3x3_1(self.branch1x1(x))) #3乘3通道
        outputs=[branch_pool,branch1x1,branch5x5,branch3x3]
        return torch.cat(outputs,dim=1) #沿[1]维度拼接，也就是C

class ResidualBlock(torch.nn.Module):
    def __init__ (self,channels):
        super(ResidualBlock,self).__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        y=self.conv1(x)
        return F.relu(x+y) #加完之后才可以池化

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        #self.conv2=torch.nn.Conv2d(88,20,kernel_size=5) #GoogLeNet
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5) #Residual
        self.incep1=GoogLeNet(in_channels=10)
        self.incep2=GoogLeNet(in_channels=20)
        self.rblock1=ResidualBlock(10)
        self.rblock2=ResidualBlock(20)
        self.mp=torch.nn.MaxPool2d(2)
        #self.fc=torch.nn.Linear(1408,10) #GoogLeNet
        self.fc=torch.nn.Linear(320,10) #Residual
    def forward(self,x):
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))
        #x=self.incep1(x)
        x=self.rblock1(x)
        x=F.relu(self.mp(self.conv2(x)))
        #x=self.incep2(x)
        x=self.rblock2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

batch_size=64
transform=transforms.Compose([ #Compose用于把一系列操作变为管线
    transforms.ToTensor(), #W*H变成1*W*H
    transforms.Normalize((0.1307,),(0.3081),) #均值 标准差                      
])
train_dataset=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transform)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transform)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)
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