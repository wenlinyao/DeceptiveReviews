import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cPickle
import random
from multiprocessing import Process
from keras.preprocessing import sequence
import time

class BasicRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args, vocab_size, pretrained=None):
        super(BasicRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size

        self.drop = nn.Dropout(self.args.dropout)
        self.encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)
        self.rnn = nn.LSTM(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers, dropout = self.args.dropout)
        self.linear = nn.Linear(self.args.metafeature_dim, self.args.metafeature_dim)
        #self.decoder = nn.Linear(self.args.rnn_size * 2 + self.args.metafeature_dim, 2)
        self.decoder = nn.Linear(self.args.rnn_size, 2)
        self.softmax = nn.Softmax()

        start = time.clock()
        self.init_weights(pretrained=pretrained)
        print("Initialized LSTM model")

    def forward(self, input, input_advertisig, metafeature, hidden):
        emb = self.drop(self.encoder(input))
        #emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)

        emb2 = self.drop(self.encoder(input_advertisig))
        output_advertising, hidden_advertising = self.rnn(emb2, hidden)

        #output = self.drop(output)

        if(self.args.aggregation=='mean'):
            output = torch.mean(output, 0)
            output_advertising = torch.mean(output_advertising, 0)
        elif(self.args.aggregation=='last'):
            last_idx = Variable(torch.LongTensor([output.size()[0] - 1]))
            last_idx_advertising = Variable(torch.LongTensor([output_advertising.size()[0] - 1]))
            output = torch.index_select(output, 0, last_idx)
            output_advertising = torch.index_select(output_advertising, 0, last_idx_advertising)

        output = torch.squeeze(output, 0)
        output_advertising = torch.squeeze(output_advertising, 0)
        output_metafeature = self.linear(Variable(torch.transpose(metafeature.data, 0, 1)))
        #decoded = self.decoder(torch.cat((output, output_advertising, output_metafeature), 1))

        decoded = self.decoder(output)

        prob = self.softmax(decoded)
        return prob, hidden

    def init_weights(self, pretrained):
        initrange = 0.1
        #if(pretrained is not None):
        print("Setting Pretrained Embeddings")
        pretrained = pretrained.astype(np.float32)
        #pretrained = torch.from_numpy(pretrained)
        self.encoder.weight.data.copy_(torch.from_numpy(pretrained))
        self.encoder.weight.requires_grad = self.args.trainable
        #else:
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_size)),
            Variable(torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_size)))

    
