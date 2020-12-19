import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import exp
from torch.autograd import Variable
dtype = torch.FloatTensor

train = open('./train.txt', encoding='utf8')
valid = open('./valid.txt', encoding='utf8')

train_sentences = []
valid_sentences = []
count = 0
for line in train:
    train_sentences.append(line.rstrip())
for line in valid:
    valid_sentences.append(line.rstrip())

train_word_list = "".join(train_sentences).split()
train_word_list = list(set(train_word_list))
train_word_dict = {w: i for i, w in enumerate(train_word_list)}
train_number_dict = {i: w for i, w in enumerate(train_word_list)}
n_class = len(train_word_list)

valid_word_list = "".join(valid_sentences).split()
valid_word_list = list(set(valid_word_list))
#valid_word_dict = {w:i for i,w in enumerate(valid_word_list)}
#valid_number_dict = {i:w for i,w in enumerate(valid_word_list)}
print("word_dict done")

n_step = 3
batch_size = 1000
batch_size_valid = 1000
n_hidden = 50
eye = np.eye(n_class)

def get_sentences(sentences):
    new_sentences = []
    for sen in sentences:
        sen = sen.split()
        for i in range(len(sen) - n_step):
            new_sentences.append(sen[i:i + n_step + 1])
    return new_sentences

def make_batch_randn(sentences, batch_size):
    sentences = get_sentences(sentences)
    index = np.random.choice(len(sentences), batch_size, replace=False)
    input_batch = []
    target_batch = []
    for i in index:
        sen = sentences[i]
        input = [train_word_dict[n] for n in sen[:-1]]
        input_batch.append(eye[input])
        target_batch.append(train_word_dict[sen[-1]])
    return input_batch, target_batch

def make_batch_all(sentences, batch_size, i):
    input_batch = []
    target_batch = []
    input = []
    length = len(sentences)
    for j in range(batch_size):
        if i*batch_size + j < length:
            sen = sentences[i * batch_size + j]
        else:
            break
        for n in sen[:-1]:
            if n in train_word_dict:
                input.append(train_word_dict[n])
            else:
                input.append(train_word_dict['<unk>'])
        if sen[-1] in train_word_dict:
            target = train_word_dict[sen[-1]]
        else:
            target = train_word_dict['<unk>']
        input_batch.append(eye[input])
        target_batch.append(target)
        input = []
    return input_batch,target_batch

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN,self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype)) #n_hidden,n_class
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))          #n_class

    def forward(self, hidden, X):
        X = X.transpose(0, 1)  # n_step,batch_size,n_class
        output, hidden = self.rnn(X, hidden)
        output = output[-1]  # batch_size,n_hidden
        y = self.b + torch.mm(output, self.W)  # batch_size,n_class
        return y



model = TextRNN()
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5,
                                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                                  cooldown=0, min_lr=0.00001, eps=1e-08)
#optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)

#input_batch_valid, target_batch_valid = make_batch(valid_sentences)
print("input_batch_valid done")



def cal_valid_ppl():
    loss_avg = 0
    sentences = get_sentences(valid_sentences)
    length = len(sentences)/batch_size_valid
    batch_sum = 0
    for i in range(int(length)):
        input, target = make_batch_all(sentences, batch_size_valid, i)
        input = Variable(torch.Tensor(input)).cuda()
        #arget = target_batch_valid[i*batch_size_valid:(i+1)*batch_size_valid]
        target = Variable(torch.LongTensor(target)).cuda()
        hidden = Variable(torch.zeros(1, batch_size_valid, n_hidden)).cuda()
        output = model(hidden, input)
        loss = criterion(output, target)
        loss_avg += float(loss)*batch_size_valid
        batch_sum += batch_size_valid
        if i == int(length) - 1 and i < length - 1:
            #input = input_batch_valid[(i + 1) * batch_size_valid:]
            input, target = make_batch_all(sentences, batch_size_valid, i + 1)
            input = Variable(torch.Tensor(input)).cuda()
            #target = target_batch_valid[(i + 1) * batch_size_valid:]
            target = Variable(torch.LongTensor(target)).cuda()
            hidden = Variable(torch.zeros(1, list(input.size())[0], n_hidden)).cuda()

            output = model(hidden, input)
            loss = criterion(output, target)
            loss_avg += float(loss) * (list(input.size())[0])
            batch_sum += list(input.size())[0]

    loss_avg = loss_avg / batch_sum
    ppl = exp(loss_avg)
    print(type(ppl))
    print('loss = ', '{:.6f}'.format(loss_avg), 'ppl =', '{:.6f}'.format(ppl))
    return ppl,loss_avg