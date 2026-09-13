import torch
import numpy as np
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 

#本代码不能解决数据缺失

class TitanicDataset(Dataset): 
    def __init__(self,filepath): 
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32,skiprows=1,usecols=(1,10)) #跳过第一行，只考虑第一列第10列
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:, [1]]) #usecols中的第二个
        self.y_data=torch.from_numpy(xy[:, [0]]) #usecols中的第一个
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
    
dataset=TitanicDataset('trainp08.csv') 
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2) 

class Model(torch.nn.Module): 
    def __init__(self): 
        super(Model,self).__init__() 
        self.linear1=torch.nn.Linear(1,6) #输入为一维
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid() 
    def forward(self,x): #保持一个变量x
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x

model=Model() 

criterion=torch.nn.BCELoss(reduction='sum') 
optimizer=torch.optim.SGD(model.parameters(),lr=0.001) 

if __name__=='__main__': 
    for epoch in range(100):
        for i,data in enumerate(train_loader):
            inputs,label=data #可以自动被转换成tensor
            y_pred=model(inputs)
            loss=criterion(y_pred,label)
            print(epoch,i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"\n训练完成！")
 
class TitanicTestDataset(Dataset): #测试集
    def __init__(self,filepath): 
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32,skiprows=1,usecols=(0,9)) #0是瞎写的，因为usecols必须要二维
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:, [1]]) 
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
dataset=TitanicTestDataset('testp08.csv')
test_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2) 

if __name__=='__main__': 
    label_data = np.loadtxt('gender_submissionp08.csv', delimiter=',', dtype=np.float32, skiprows=1)
    true_label = label_data[:, 1] #标签文件的第二列
    predictions = []
    for inputs, idx in test_loader:
        outputs = model(inputs)

    diff = np.array(predictions) - true_label
    mse = (diff ** 2).mean()
    print(f"\nMSE: {mse:.4f}")
