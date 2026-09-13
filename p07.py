import torch
import torch.nn.functional as F #函数
import numpy as np #绘图用
import matplotlib.pyplot as plt

xy=np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32) #下载文件（文件名，分隔符，最后一个不变）
x_data=torch.from_numpy(xy[:,:-1]) #所有行：所有列：最后一列不要 
y_data=torch.from_numpy(xy[:, [-1]]) #所有行：最后一列

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

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step() 

print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())
x_test=torch.Tensor([4.0])
y_test=model(x_test)
print("y_pred=",y_test.data)

x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view((200,1))
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability')
plt.grid()
plt.show()