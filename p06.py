import torch
import torch.nn.functional as F #函数
import numpy as np #绘图用
import matplotlib.pyplot as plt

x_data=torch.Tensor([[1.0],[2.0],[3.0]]) 
y_data=torch.Tensor([[0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module): 
    def __init__(self): 
        super(LogisticRegressionModel,self).__init__() 
        self.linear=torch.nn.Linear(1,1) 
    def forward(self,x): 
        y_pred=F.sigmoid(self.linear(x)) #这里改了，在原来基础上加logisit
        return y_pred
    #无需写backward

model=LogisticRegressionModel() 

criterion=torch.nn.BCELoss(reduction='sum') #这里也改了
optimizer=torch.optim.SGD(model.parameters(),lr=0.02) 

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss)
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