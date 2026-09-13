import torch

batch_size=1
seq_len=3 #几列
input_size=4
hidden_size=2
num_layers=1 #几行（几层）

cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size) #单指一个cell
#cell=torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers) #一堆cell和流程组成的完整网络

dataset=torch.randn(seq_len,batch_size,input_size)

hidden=torch.zeros(batch_size,hidden_size) #使用RNNCell
#hidden=torch.zeros(num_layers,batch_size,hidden_size) #使用RNN

for idx,input in enumerate(dataset): #使用RNNCell要手动写循环
    print(idx,input.shape)
    hidden=cell(input,hidden)
    print(hidden.shape,hidden)

#out,hidden=cell(dataset,hidden) #RNN内含循环
#print(out.shape,out,hidden.shape,hidden)