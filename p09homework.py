import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from  torch.utils.data import TensorDataset

batch_size=64

train_df=pd.read_csv('trainp09.csv') #用pd读取文件
test_df=pd.read_csv('testp09.csv')
x_train=torch.from_numpy(train_df.iloc[:,1:94].values).float() #iloc为读取数据，第1列到第93列
label_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3,
             'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8} #字典
train_df['target'] = train_df['target'].map(label_map) #map是查询字典
y_train=torch.from_numpy(train_df.iloc[:,94].values).long() #要用long格式
train_dataset=TensorDataset(x_train,y_train)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
x_test=torch.from_numpy(test_df.iloc[:,1:94].values).float()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=torch.nn.Linear(93,512) #输入维度
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,9) #最终输出九个分类
    def forward(self,x):
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
    with torch.no_grad(): #不需要梯度
        batch = x_test[:48]              # 形状 (48, 93)，一定要是二维数组
        outputs = model(batch)            # 输出 (48, 9)
        _, predicted = torch.max(outputs.data, dim=1)
        print(predicted)

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
    test()
    #非常简洁干净的main