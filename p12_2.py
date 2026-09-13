import torch

batch_size=1
input_size=4
hidden_size=4

idx2char=['e','h','l','o'] #字典
x_data=[1,0,2,2,3]
y_data=[3,1,2,3,2]
one_hot_lookup=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
x_one_hot=[one_hot_lookup[x] for x in x_data] #独热向量，从上一行里找到符合自己数字的向量，取出

inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size) #RNN要seq_len
labels=torch.LongTensor(y_data).view(-1,1)

class RNNModel(torch.nn.Module):
    def __init__(self,input_size,batch_size,hidden_size): #RNN版需要加上num_layers
        super(RNNModel,self).__init__()
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.cell=torch.nn.RNNCell(input_size=self.input_size,hidden_size=self.hidden_size)
    def forward(self,input,hidden):
        hidden=self.cell(input,hidden)
        return hidden
        #RNN版：
        #hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        #out,_=self.rnn(input,hidden)
        #return out.view(-1,self.hidden_size)
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)
net=RNNModel(input_size,batch_size,hidden_size)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.1)

for epoch in range(15):

    #RNNCell版
    loss=0
    optimizer.zero_grad()
    hidden=net.init_hidden()
    for input,label in zip(inputs,labels):
        hidden=net(input,hidden)
        loss+=criterion(hidden,label)
        _,idx=hidden.max(dim=1)
        print(idx2char[idx.item()])
    loss.backward()
    optimizer.step()
    print(epoch+1,loss.item())

    #RNN版
    #optimizer.zero_grad()
    #outputs=net(inputs)
    #loss=criterion(outputs,labels)
    #loss.backward()
    #optimizer.step()
    #_,idx=outputs.max(dim=1)
    #idx=idx.data.numpy()
    #print(' '.join([idx2char[x] for x in idx]))
    #print(epoch+1,loss.item())