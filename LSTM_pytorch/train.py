from __future__ import division
import argparse
from sklearn.metrics import accuracy_score
import json
import gzip
from models import BasicRNN
import cPickle as pickle
from datetime import datetime
import os
import random
import numpy as np
import time
import sys
import math
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from keras.preprocessing import sequence
from collections import Counter
import time
from multiprocessing import Process

def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    return x.data.type(torch.DoubleTensor).numpy()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    # What's this?
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)



class Experiment:
    def __init__(self, dst_folder, valid_test_cat_idxList, args):
        self.dst_folder = dst_folder
        self.valid_test_cat_idxList = valid_test_cat_idxList
        self.args = args
        rand_seed = 1
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        with open(self.dst_folder + "env.pkl",'r') as f:
            self.env = pickle.load(f)

        self.train_set = self.env['train']
        self.test_set_set = self.env['test']
        self.dev_set = self.env['dev']

        if(self.args.toy == True):
            print("Using toy mode...")
            random.shuffle(self.train_set)
            self.train_set = self.train_set[:500]
            for i in range(len(self.valid_test_cat_idxList)):
                random.shuffle(self.test_set_set[i])
                self.test_set_set[i] = self.test_set_set[i][:500]

        #for t in self.train_set:
        #    print t["polarity"],

        if(self.args.dev==0):
            self.train_set = self.train_set + self.dev_set

        print "creating model..."

        self.mdl = BasicRNN(self.args, len(self.env["word_index"]), pretrained = self.env["glove"])

    def select_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.mdl.parameters())

        if(self.args.opt=='Adam'):
            self.optimizer =  optim.Adam(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='RMS'):
            self.optimizer =  optim.RMSprop(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='SGD'):
            self.optimizer =  optim.SGD(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='Adagrad'):
            self.optimizer =  optim.Adagrad(parameters, lr=self.args.learn_rate)
        elif(self.args.opt=='Adadelta'):
            self.optimizer =  optim.Adadelta(parameters, lr=self.args.learn_rate)
        """
        if(self.args.opt=='Adam'):
            self.optimizer =  optim.Adam(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='RMS'):
            self.optimizer =  optim.RMSprop(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='SGD'):
            self.optimizer =  optim.SGD(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='Adagrad'):
            self.optimizer =  optim.Adagrad(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='Adadelta'):
            self.optimizer =  optim.Adadelta(self.mdl.parameters(), lr=self.args.learn_rate)
        """

    def evaluate(self, x):
        ''' Evaluates normal RNN model
        '''
        all_probs = []
        all_preds = []
        all_targets = []
        for instance in x:
            all_targets.append(instance["polarity"])
        print "Evaluate on test set..."
        all_tests = []
        for i in range(len(x)):
            sentence, sentence_advertising, metafeature, targets, actual_batch_size = self.make_batch(x[i:i+1], -1, evaluation=True)
            all_tests.append([sentence, sentence_advertising, metafeature, targets, actual_batch_size])

        for i in tqdm(range(len(x))):
            [sentence, sentence_advertising, metafeature, targets, actual_batch_size] = all_tests[i]
            hidden = self.mdl.init_hidden(actual_batch_size)
            output, hidden = self.mdl(sentence, sentence_advertising, metafeature, hidden)
            all_probs.append(output)
            all_preds.append(np.argmax(tensor_to_numpy(output)))
        print "len(all_targets):", len(all_targets), "len(all_preds):", len(all_preds)
        confusion_matrix = {}
        matches = 0
        for i in range(len(all_targets)):
            if all_targets[i] == all_preds[i]:
                matches += 1
            string = str(all_targets[i]) + " --> " + str(all_preds[i])
            if string in confusion_matrix:
                confusion_matrix[string] += 1
            else:
                confusion_matrix[string] = 1
        print "accuracy:", float(matches) / float(len(all_targets))
        print "confusion_matrix[target --> pred]:", confusion_matrix
        return all_probs, all_preds
        

    def pad_to_batch_max(self, x, max_len = 100):
        #lengths = [len(y) for y in x]
        #max_len = np.max(lengths)
        padded_tokens = sequence.pad_sequences(x, maxlen=max_len)
        #print "padded_tokens:", padded_tokens
        return torch.LongTensor(padded_tokens.tolist()).transpose(0,1)

    def make_batch(self, x, i, evaluation=False):
        ''' -1 to take all
        '''
        if(i>=0):
            batch = x[int(i * self.args.batch_size):int(i * self.args.batch_size)+self.args.batch_size]
        else:
            batch = x
        if(len(batch)==0):
            return None, None, None, None, self.args.batch_size

        sentence = self.pad_to_batch_max([x['tokenized_txt'] for x in batch], 100)
        sentence_advertising = self.pad_to_batch_max([x['advertising_words'] for x in batch], 10)
        metafeature = torch.FloatTensor([x['metafeature'] for x in batch]).transpose(0, 1)
        targets = torch.LongTensor(np.array([x['polarity'] for x in batch], dtype=np.int32).tolist())

        actual_batch_size = sentence.size(1)
        
        sentence = Variable(sentence, volatile=evaluation)
        sentence_advertising = Variable(sentence_advertising, volatile=evaluation)
        metafeature = Variable(metafeature, volatile=evaluation)
        

        targets = Variable(targets, volatile=evaluation)

        return sentence, sentence_advertising, metafeature, targets, actual_batch_size

    def train_batch(self, i):
        ''' Trains a regular RNN model
        '''
        #print self.make_batch(self.train_set, i)
        sentence, sentence_advertising, metafeature, targets, actual_batch_size = self.make_batch(self.train_set, i)

        
        if(sentence is None):
            return None
        
        hidden = self.mdl.init_hidden(actual_batch_size)
        #hidden = repackage_hidden(hidden)
        
        
        self.mdl.zero_grad()

        output, hidden = self.mdl(sentence, sentence_advertising, metafeature, hidden)
        
        #print "output:", output
        #print "targets:", targets
        loss = self.criterion(output, targets)
        loss.backward()

        nn.utils.clip_grad_norm(self.mdl.parameters(), self.args.clip)
        self.optimizer.step()

        return loss.data[0]

    def train(self):
        print("Starting training...")
        self.criterion = nn.CrossEntropyLoss()
        print(self.args)
        total_loss = 0
        num_batches = int(len(self.train_set) / self.args.batch_size) + 1
        print "len(self.train_set)", len(self.train_set)
        print "num_batches:", num_batches
        self.select_optimizer()
        for epoch in range(1, self.args.epochs+1):
            print "epoch: ", epoch
            t0 = time.clock()
            random.shuffle(self.train_set)
            print("========================================================================")
            losses = []
            actual_batch = self.args.batch_size
            for i in tqdm(range(num_batches)):
                loss = self.train_batch(i)
                if(loss is None):
                    continue    
                losses.append(loss)
            t1 = time.clock()
            print("[Epoch {}] Train Loss={} T={}s".format(epoch, np.mean(losses),t1-t0 ))
            if(0 < epoch and epoch < self.args.epochs and epoch % self.args.eval == 0):
                for i in range(len(self.valid_test_cat_idxList)):
                    all_probs, all_preds = self.evaluate(self.test_set_set[i])
    def test(self):
        #if epoch == self.args.epochs:
        for idx in range(len(self.valid_test_cat_idxList)):
            all_probs, all_preds = self.evaluate(self.test_set_set[idx])
            output_file = open(self.dst_folder + "thread" + str(self.valid_test_cat_idxList[idx]) + "/LSTM_true_and_pred_value", "w")
            for i, instance in enumerate(self.test_set_set[idx]):
                #print list(all_probs[i].data[0])
                output_file.write(str(instance["polarity"]) + "\t" + str(all_preds[i]) + "\t" + str(list(all_probs[i].data[0])) + "\n")
            output_file.close()
        



#if __name__ == '__main__':
def LSTM_main(current_folder, valid_test_cat_idxList, args):
    exp = Experiment(current_folder, valid_test_cat_idxList, args)
    exp.train()
    exp.test()

    
