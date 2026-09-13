import torch

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel (torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Linear=torch.nn.Linear(1,1)
    def forward (self,x):
        y_pred=self.Linear(x)
        return y_pred

model=LinearModel()
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #学习率需要自己调一下
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w=",model.Linear.weight.item())
print("b=",model.Linear.bias.item())
x_test=torch.Tensor([4.0])
y_test=model(x_test)
print("y=",y_test.data)
