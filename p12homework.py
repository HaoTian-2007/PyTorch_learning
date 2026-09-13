import torch

#改动很小，loss变大

batch_size=1
input_size=4
hidden_size=8
num_class=4
num_layers=2
seq_len=5
embedding_size=10

idx2char=['e','h','l','o'] 
x_data=[[1,0,2,2,3]] 
y_data=[3,1,2,3,2]

inputs=torch.LongTensor(x_data)
labels=torch.LongTensor(y_data) 

class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super(EmbeddingModel,self).__init__()
        self.emb=torch.nn.Embedding(input_size,embedding_size) 
        self.gru=torch.nn.GRU(input_size=embedding_size,hidden_size=hidden_size,
                              num_layers=num_layers,batch_first=True) #第一处修改
        self.fc=torch.nn.Linear(hidden_size,num_class)
    def forward(self,x):
        hidden=torch.zeros(num_layers,x.size(0),hidden_size)
        x=self.emb(x)
        x,_=self.gru(x,hidden) #第二处修改
        x=self.fc(x)
        return x.view(-1,num_class)

net=EmbeddingModel()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.05)

for epoch in range(15): 
    optimizer.zero_grad()
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    _,idx=outputs.max(dim=1)
    idx=idx.data.numpy()
    print(' '.join([idx2char[x]for x in idx]))
    print(epoch+1,loss.item())
