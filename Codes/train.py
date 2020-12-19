import RNN
import numpy as np
import torch
from torch.autograd import Variable
from math import ceil

ppl_list = []
loss_list = []
loss_list_valid = []

print("start training……")
sentences = RNN.get_sentences(RNN.train_sentences)
length = ceil(len(sentences)/RNN.batch_size)
for epoch in range(500):

    for i in range(length):

        RNN.optimizer.zero_grad()
        input_batch, target_batch = RNN.make_batch_all(sentences, RNN.batch_size, i)
        input_batch = Variable(torch.Tensor(input_batch)).cuda()
        target_batch = Variable(torch.LongTensor(target_batch)).cuda()
        if list(input_batch.size())[0] < RNN.batch_size:
            hidden = Variable(torch.zeros(1, list(input_batch.size())[0], RNN.n_hidden)).cuda()
        else:
            hidden = Variable(torch.zeros(1, RNN.batch_size, RNN.n_hidden)).cuda()
        output = RNN.model(hidden, input_batch)
        loss = RNN.criterion(output, target_batch)
        loss_list.append(float(loss))
        np.array(loss_list)
        np.save('loss_train.npy', loss_list)
        loss.backward()
        RNN.optimizer.step()
        #if (epoch + 1) % 1 == 0 and i == length - 1:
        print("batch = ", '%03d' % (i + 1), "loss = ", '{:.6f}'.format(loss))
        if (i+1) % 100 == 0:
            print("epoch = ", '%03d' % (epoch + 1))
            ppl, loss_avg = RNN.cal_valid_ppl()
            RNN.scheduler.step(int(loss_avg))
            ppl_list.append(ppl)
            loss_list_valid.append(loss_avg)
            np.array(loss_list_valid)
            np.save('loss_valid.npy', loss_list_valid)
            np.array(ppl_list)
            np.save('ppl.npy', ppl_list)
