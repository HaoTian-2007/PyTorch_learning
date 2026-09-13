import torch

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.Tensor([1.0]) #一维,记录梯度
w.requires_grad=True #需要手动开启

def forward(x):
    return x*w
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
print("predict before training",4,forward(4).item()) #打印里面调用函数用item

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward() #自动保存到w里面,同时清空l
        print("\tgrad:",x,y,w.grad.item()) #grad是tensor类型,直接调用会建立计算图,要调用data数值
        w.data-=0.01*w.grad.data
        w.grad.data.zero_() #每次迭代都要显式清零
    print("progress:",epoch,l.item())
print("predict after training",4,forward(4).item())