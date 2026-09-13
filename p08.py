import torch
import numpy as np
from torch.utils.data import Dataset #抽象类，不能被实例化
from torch.utils.data import DataLoader #可以实例化

class DiabetesDataset(Dataset): #继承父类
    def __init__(self,filepath): #数据集太大不能全部读取的，用文件或列表
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1]) #所有行：所有列：最后一列不要 
        self.y_data=torch.from_numpy(xy[:, [-1]]) #所有行：最后一列
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
    
dataset=DiabetesDataset('diabetes.csv.gz') #文件路径在实例化的时候输入
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2) #数据集对象，容量，打乱，线程数

class Model(torch.nn.Module): 
    def __init__(self): 
        super(Model,self).__init__() 
        self.linear1=torch.nn.Linear(8,6) #这里改了，矩阵要改成相应维度（输入维度，输出维度）
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid() #和之前的sigmoid不是一个意思，这里表示模块，无需参数
        #self.activate=torch.nn.ReLU() #其他激活函数
    def forward(self,x): #保持一个变量x
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x
    #无需写backward

model=Model() 

criterion=torch.nn.BCELoss(reduction='sum') 
optimizer=torch.optim.SGD(model.parameters(),lr=0.02) 

if __name__=='__main__': #需要封装到if里面才不会报错
    for epoch in range(100):
        for i,data in enumerate(train_loader):
            inputs,label=data #可以自动被转换成tensor
            y_pred=model(inputs)
            loss=criterion(y_pred,label)
            print(epoch,i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


