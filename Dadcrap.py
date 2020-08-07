import os
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import csv
import pandas as pd
def select(column,condition,tf,df_data):
    if tf=='==':
        return df_data[df_data[column]==condition]
    if tf=='!=':
        return df_data[df_data[column]!=condition]
data = []
words = []
z = 0
sen_num = 0
data='c:/users/milind/documents/project/data/all_data.tsv'
with open(data) as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
    for i in tsvreader:
        words.append(i)
while z < len(words):
    if words[z] != []:
        words[z].insert(0, sen_num)
        data.append(words[z])

    elif words[z] == []:
        sen_num += 1
    z += 1
Data_frame = pd.DataFrame(data,
                          columns=['Sentence#', 'Word', 'Label'])
i=0
corpus=[]
sep=' '
while i<Data_frame['Sentence#'].max():
    sentence=[]
    z=1
    for word in select('Sentence#',i,'==',Data_frame)['Word']:
        sentence.append(word)
    sentence=sep.join(sentence)
    corpus.append(sentence)
    i+=1
print("Done")
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens
tokenized_corpus = tokenize_corpus(corpus)
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
vocabulary_size = len(vocabulary)
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))
idx_pairs = np.array(idx_pairs)  # it will be useful to have this as numpy array
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x
embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 101
learning_rate = 0.001
for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1, -1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data
        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 1   == 0:
        print(f'Loss at epo {epo}: {loss_val / len(idx_pairs)}')
torch.save(W2,'c:/users/milind/documents/project/step4/tensor.txt')
i2w=open('c:/users/milind/documents/project/step4/idx2word.txt','w')
i2w.write(str(idx2word))
i2w.close