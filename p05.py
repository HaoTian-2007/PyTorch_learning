import torch

x_data=torch.Tensor([[1.0],[2.0],[3.0]]) #注意双层列表
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module): #所有模型类必须继承自Module
    def __init__(self): #构造函数
        super(LinearModel,self).__init__() #调用父类
        self.linear=torch.nn.Linear(1,1) #构造对象，(输入维度，输出维度，用不用b)
    def forward(self,x): #前馈，名字不能改，因为有别的函数也叫forw，本质上是重写
        y_pred=self.linear(x) #可调用对象
        return y_pred
    #无需写backward

model=LinearModel() #实例化

criterion=torch.nn.MSELoss(size_average=False) #损失函数，是否求均值
optimizer=torch.optim.SGD(model.parameters(),lr=0.01) #优化器，检查model的所有成员权重并加和，学习率

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss)
    optimizer.zero_grad() #梯度归零
    loss.backward()
    optimizer.step() #梯度更新

print("w=",model.linear.weight.item()) #weight是矩阵，显示数值用item
print("b=",model.linear.bias.item())
x_test=torch.Tensor([4.0])
y_test=model(x_test)
print("y_pred=",y_test.data)