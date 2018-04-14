import os
import json
import time
import random

from pprint import pprint, pformat
import logging
log = logging.getLogger('main')
log.setLevel(logging.INFO)

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import itertools

from functools import partial
from collections import namedtuple, defaultdict
from nltk import word_tokenize

PAD=0

RawSample = namedtuple('RawSample', ['id','sentence_id',  'sentence', 'sentiment'])
Sample = namedtuple('Sample', ['id','sentence_id',  'tokens', 'sentiment'])

SELF_NAME = os.path.basename(__file__).replace('.py', '')
class Config:
    split_ratio = 0.90
    input_vocab_size = 30000
    hidden_dim = 200
    embed_dim = 200
    batch_size = 128
    cuda = True
    tqdm = True
    flush = False


def build_sentimentnet_sample(raw_sample):
    labels = ['negative', 'somewhat negative', 'neutral', 'sometwhat positive', 'positive']
    sentence = word_tokenize(raw_sample.sentence.strip(' \n\t').lower())
    return Sample(raw_sample.id, raw_sample.sentence_id.lower(),
                  sentence,
                  labels[int(raw_sample.sentiment.strip(' \n\t'))])

def prep_samples_for_sentimentnet(dataset, samples_from_each_class=1000000):
    ret = []
    vocabulary = defaultdict(int)
    labels = defaultdict(int)
    class_sample_counter = defaultdict(int)
    dataset = {s.id:s for s in dataset}    
    for i, (sid, sample) in enumerate(tqdm(dataset.items())):
        try:
            sample = build_sentimentnet_sample(sample)
            if class_sample_counter[sample.sentiment] > samples_from_each_class:
                continue

            class_sample_counter[sample.sentiment] += 1
            for token in sample.tokens:
                vocabulary[token] += 1
            labels[sample.sentiment] += 1
            ret.append(sample)
        except KeyboardInterrupt:
            return
        except:
            log.exception('at id: {}, sid: {}'.format(i, sid))

    return ret, vocabulary, labels



"""
# Batching utils   
"""
import numpy as np
def seq_maxlen(seqs):
    return max([len(seq) for seq in seqs])

def pad_seq(seqs, maxlen=0, PAD=PAD):
    def pad_seq_(seq):
        return seq[:maxlen] + [PAD]*(maxlen-len(seq))

    if len(seqs) == 0:
        return seqs
    
    if type(seqs[0]) == type([]):
        maxlen = maxlen if maxlen else seq_maxlen(seqs)
        seqs = [ pad_seq_(seq) for seq in seqs ]
    else:
        seqs = pad_seq_(seqs)
        
    return seqs


def batchop(datapoints, WORD2INDEX, LABEL2INDEX, *args, **kwargs):
    indices = [d.id for d in datapoints]
    seq = []
    label = []
    for d in datapoints:
        for w in d.tokens:
            if w in WORD2INDEX:
                seq.append(WORD2INDEX[w])
            else:
                seq.append(WORD2INDEX['UNK'])
                
        label.append(LABEL2INDEX[d.sentiment])

    seq = pad_seq(seq)
    return  np.array(seq), np.array(label)

class Model(nn.Module):
    def __init__(self, Config, input_vocab_size, output_vocab_size):
        super(Model, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = Config.hidden_dim
        self.embed_dim = Config.embed_dim

        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)
        self.encode = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True)
        self.classify = nn.Linear(2*self.hidden_dim, self.output_vocab_size)
        
        if Config.cuda:
            self.cuda()
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(2, batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, seq):
        seq      = Variable(torch.LongTensor(seq))

        if seq.dim() == 1: seq = seq.unsqueeze(0)
        if Config().cuda: 
            seq = seq.cuda()
            
        batch_size, seq_size = seq.size()
        seq_emb = F.tanh(self.embed(seq))
        seq_emb = seq_emb.transpose(1, 0)
        pad_mask = (seq > 0).float()
        
        states, cell_state = self.encode(seq_emb)

        logits = self.classify(states[-1])
        
        return F.log_softmax(logits, dim=-1)
        
    
import sys
if __name__ == '__main__':
    dataset = open('train.tsv', 'r').readlines()[1:]
    for i,s in enumerate(dataset):
        dataset[i] = RawSample(*s.split('\t'))

    print('raw dataset size: {}'.format(len(dataset)))
    labelled_samples, vocabulary, labels = prep_samples_for_sentimentnet(dataset)
            
    LABELS = sorted(list(labels.keys()))
    LABEL2INDEX = { w:i for i,w in enumerate(LABELS) }
    
    loss_weight = [1 - float(labels[i])/float(sum(labels.values())) for i in LABELS]
    log.info('loss_weight: {}'.format(pformat(loss_weight)))

    VOCAB = ['PAD', 'UNK', 'EOS'] + list(vocabulary.keys())
    WORD2INDEX = { w:i for i,w in enumerate(VOCAB) }
    if sys.argv[1] == 'train':
        pivot = int( Config().split_ratio * len(labelled_samples) )
        random.shuffle(labelled_samples)
        train_set, test_set = labelled_samples[:pivot], labelled_samples[pivot:]
        
        train_set = sorted(train_set, key=lambda x: -len(x.tokens))
        test_set  = sorted(test_set, key=lambda x: -len(x.tokens))

        model = Model(Config, len(VOCAB), len(LABELS))
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.1)

        batch_size = Config.batch_size
        try:
            for epoch in range(1000):
                num_batch = len(train_set)//batch_size
                for index in tqdm(range(num_batch)):
                    s, e = batch_size * index, batch_size * (index + 1)
                    input_, target = batchop(train_set[s:e], WORD2INDEX, LABEL2INDEX)
                    target = Variable(torch.LongTensor(target))
                    if Config.cuda:
                        target = target.cuda()

                    output = model(input_)
                    loss = loss_function(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('epoch: {} loss: {}'.format(epoch, loss.data[0]))
                if epoch and not epoch % 10:
                    test_loss = 0
                    accuracy  = 0
                    num_batch = len(test_set)//batch_size                
                    for index in tqdm(range(num_batch)):
                        s, e = batch_size * index, batch_size * (index + 1)
                        input_, target = batchop(test_set[s:e], WORD2INDEX, LABEL2INDEX)
                        target = Variable(torch.LongTensor(target))
                        if Config.cuda:
                            target = target.cuda()

                            output = model(input_)
                        test_loss += loss_function(output, target)
                        accuracy += (output.data.max(1)[1] == target.data).float().sum()/batch_size
                    print('epoch: {} -- test_loss: {} -- accuracy: {}'.format(epoch, test_loss.data[0]/num_batch, accuracy/num_batch))
        except KeyboardInterrupt:
            torch.save(model.state_dict(), '{}.{}'.format(SELF_NAME, 'pth'))
        
    if sys.argv[1] == 'predict':
        model =  Model(Config(), len(VOCAB),  len(LABELS))
        if Config().cuda:  model = model.cuda()
        model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, 'pth')))


        while True:
            start_time = time.time()
            strings = input('>>> ')
            s = []
            for w in word_tokenize(strings.lower()):
                if w in WORD2INDEX:
                    s.append(WORD2INDEX[w])
                else:
                    s.append(WORD2INDEX['UNK'])
                
            output = model(s)
            output = output.data.max(dim=-1)[1].cpu().numpy()
            label = LABELS[output[0]]
            print(label)
            
            duration = time.time() - start_time
            print(duration)
