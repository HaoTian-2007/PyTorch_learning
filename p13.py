import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import csv, time
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

#正确率先快速上升，后缓慢下降，最大值约为84%

HIDDEN_SIZE=100
BATCH_SIZE=256
N_LAYER=2
N_EPOCHS=100
N_CHARS=128

class NameDataset(Dataset):
    def __init__(self,is_train_set=True):
        filename='trainp13.csv'if is_train_set else'testp13.csv'
        with open(filename,'r',encoding='utf-8')as f: 
            reader=csv.reader(f)
            rows=list(reader)
        self.names=[row[0] for row in rows] #遍历rows这个列表中的每一个row，取第一个值
        self.len=len(self.names)
        self.countries=[row[1] for row in rows]
        self.country_list=list(sorted(set(self.countries))) #set:把列表去掉重复元素并变为集合
        self.country_dict=self.getCountryDict() #字典
        self.country_num=len(self.country_list)
    def __getitem__(self,index):
        return self.names[index],self.country_dict[self.countries[index]] #返回自己的名字、自己的国家在字典里的标号
    def __len__(self):
        return self.len
    def getCountryDict(self):
        country_dict=dict()
        for idx,country_name in enumerate(self.country_list,0):
            country_dict[country_name]=idx #写入字典
        return country_dict
    def idx2country(self,index):
        return self.country_list[index]
    def getCountriesNum(self):
        return self.country_num

trainset=NameDataset(is_train_set=True)
trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
testset=NameDataset(is_train_set=False)
testloader=DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)
N_COUNTRY=trainset.getCountriesNum()

class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True): 
        super(RNNClassifier,self).__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.n_directions=2 if bidirectional else 1 #双向/单向
        self.embedding=torch.nn.Embedding(input_size,hidden_size)
        self.gru=torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)
        self.fc=torch.nn.Linear(hidden_size*self.n_directions,output_size)
    def _init_hidden(self,batch_size):
        hidden=torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size) #如果是双向RNN，是两个矩阵拼接，需要×2
        return hidden
    def forward(self,input,seq_lengths):
        input=input.t() #做转置
        batch_size=input.size(1)
        hidden=self._init_hidden(batch_size) #初始化hidden
        embedding=self.embedding(input)
        gru_input=pack_padded_sequence(embedding,seq_lengths,batch_first=False) #将所有非零的输入拼接打包（取所有输入的第一个元素，再取所有的第二个元素。。。），前提是输入要按照序列长度降序排序
        output,hidden=self.gru(gru_input,hidden)
        if self.n_directions==2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat=hidden[-1]
        fc_output=self.fc(hidden_cat)
        return fc_output

def name2list(name):
    arr=[ord(c) for c in name] #储存每个字母的ASCII
    return arr,len(arr)

def make_tensor(names,countries):
    sequences_and_lengths=[name2list(name)for name in names] #获取每个输入的内容和长度
    name_sequences=[sl[0] for sl in sequences_and_lengths]
    seq_lengths=torch.LongTensor([sl[1]for sl in sequences_and_lengths])
    countries=torch.LongTensor(countries)
    seq_tensor=torch.zeros(len(name_sequences),seq_lengths.max()).long() #构建一个全零张量，再把存在的值贴上去
    for idx,(seq,seq_len) in enumerate(zip(name_sequences,seq_lengths),0):
        seq_tensor[idx,:seq_len]=torch.LongTensor(seq)
    seq_lengths,perm_idx=seq_lengths.sort(dim=0,descending=True) #排序的时候会保存idx
    seq_tensor=seq_tensor[perm_idx]
    countries=countries[perm_idx]
    return seq_tensor,seq_lengths,countries

def trainModel(start,epoch):
    total_loss=0
    for i,(names,countries) in enumerate(trainloader,1):
        inputs,seq_lengths,target=make_tensor(names,countries)
        output=classifier(inputs,seq_lengths)
        loss=criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if i%10==0:
            print(time_since(start),epoch,i*len(inputs)/len(trainset),total_loss/i*len(inputs))
    return total_loss

def time_since(since):
    now = time.time()
    s = now - since
    m = s // 60
    s = s % 60
    return f"{int(m)}m {int(s)}s"

def testModel():
    correct=0
    total=len(testset)
    with torch.no_grad():
        for i ,(names,countries) in enumerate(testloader,1):
            inputs,seq_lengths,target=make_tensor(names,countries)
            output=classifier(inputs,seq_lengths)
            pred=output.max(dim=1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item() #将target转化为pred形状，和pred逐个元素比较是否1相等，eq的返回值是布尔张量
            print(correct/total*100)
        return correct/total

if __name__=='__main__':
    classifier=RNNClassifier(N_CHARS,HIDDEN_SIZE,N_COUNTRY,N_LAYER)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(classifier.parameters(),lr=0.001)

    start=time.time() #距离开始时间过去了多久
    print(N_EPOCHS)
    acc_list=[]
    for epoch in range(1,N_EPOCHS+1):
        trainModel(start,epoch)
        acc=testModel()
        acc_list.append(acc)