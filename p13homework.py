import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import csv, time
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

#部分变量名延续了p13未做修改，训练时间长达45分钟，平均loss在0.05

HIDDEN_SIZE=100
BATCH_SIZE=256
N_LAYER=2
N_EPOCHS=20
N_CLASS=5 #种类数

class SentimentDataset(Dataset):
    def __init__(self,is_train_set=True):
        filename='trainp13homework.tsv'if is_train_set else'testp13homework.tsv'
        with open(filename,'r',encoding='utf-8')as f: 
            reader=csv.reader(f,delimiter='\t')
            rows=list(reader)
        self.phrases=[row[2] for row in rows[1:] if row[2].strip()] #跳过表头,取出非空短语
        self.len=len(self.phrases)
        self.is_train_set=is_train_set
        if(is_train_set):
            self.sentiments=[row[3] for row in rows[1:] if row[2].strip()] #取出打分
        word_list=[]
        for phrase in self.phrases:
            word_list.extend(phrase.split()) #短语按单词变成列表
        self.vocab=list(sorted(set(word_list))) #先变成集合再排序
        self.word2idx={word:idx+1 for idx,word in enumerate(self.vocab)} #往右一位
        self.vocab_size=len(self.vocab)+1 #因为往右了一位
        self.sentiment_list=['0','1','2','3','4']
        self.sentiment_dict={s:int(s) for s in self.sentiment_list} #给字符和数字建立字典
        self.sentiment_num=5
    def __getitem__(self, index):
        if self.is_train_set:
            return self.phrases[index],self.sentiment_dict[self.sentiments[index]] #返回短语和int评分
        else:
            return self.phrases[index],0 #0占位
    def __len__(self):
        return self.len
    def getVocabSize(self):
        return self.vocab_size

trainset=SentimentDataset(is_train_set=True)
trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
testset=SentimentDataset(is_train_set=False)
testloader=DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)

class RNNClassifier(torch.nn.Module): #这里没有变化
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

def phrase2list(phrase,word2idx): #把短语转成数字
    words=phrase.split() #短语切成单词
    arr=[word2idx.get(c,0) for c in words] #在词典里查询对应数字
    return arr,len(arr)

def make_tensor(names,countries,word2idx):
    sequences_and_lengths=[phrase2list(name,word2idx)for name in names] #获取数字列表和长度
    name_sequences=[sl[0] for sl in sequences_and_lengths]
    seq_lengths=torch.LongTensor([max(sl[1],1)for sl in sequences_and_lengths]) #避免只有标点
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
        inputs,seq_lengths,target=make_tensor(names,countries,trainset.word2idx) #这里需要把word2idx传进去
        output=classifier(inputs,seq_lengths)
        loss=criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if i%10==0:
            print(time_since(start),epoch,i*len(inputs)/len(trainset),total_loss/i) #这里用total_loss/i做平均损失
    return total_loss

def time_since(since):
    now = time.time()
    s = now - since
    m = s // 60
    s = s % 60
    return f"{int(m)}m {int(s)}s"

def testModel():
    predictions=[]
    with torch.no_grad():
        for i ,(names,fake_labels) in enumerate(testloader,1): #用fake_labels占位
            inputs,seq_lengths,target=make_tensor(names,fake_labels,trainset.word2idx)
            output=classifier(inputs,seq_lengths)
            pred=output.max(dim=1,keepdim=True)[1]
            predictions.append(pred)
        predictions=torch.cat(predictions)
        return predictions

if __name__=='__main__':
    classifier=RNNClassifier(trainset.getVocabSize(),HIDDEN_SIZE,N_CLASS,N_LAYER) #调用vocab实际大小
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(classifier.parameters(),lr=0.001)

    start=time.time() #距离开始时间过去了多久
    print(N_EPOCHS)
    acc_list=[]
    for epoch in range(1,N_EPOCHS+1):
        trainModel(start,epoch)
    acc=testModel()
    print(acc[:20]) #预测结果的前20个