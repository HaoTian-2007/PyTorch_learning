import torch

input=[3,4,6,5,7,
       2,4,6,8,2,
       1,6,7,8,4,
       9,7,4,6,2,
       3,7,5,4,1]
input=torch.Tensor(input).view(1,1,5,5) #view改变形状，变成张量，BCWH
conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,
                           padding=1,
                           #stride=2,
                           bias=False) #向外扩展一圈，bias为偏置量
kernel=torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3) #输入 输出 W H
conv_layer.weight.data=kernel.data #赋值给卷积核

#池化
#maxpooling_layer=torch.nn.MaxPool2d(kernel_size=2) #2*2的池化层,stride默认为2
#output=maxpooling_layer(input)

output=conv_layer(input)
print(output)