# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import cPickle
import random
from multiprocessing import Process



def get_idx_from_sent(sent, word_idx_map, max_l=100):
    #Transforms sentence into a list of indices. Pad with zeroes.
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x

def make_idx_data_cv(revs, word_idx_map, max_l=100):
    #Transforms sentences into a 2-d matrix.
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l)   
        sent.append(rev["y"])
        
        #if rev["split"]==cv:            
        #    test.append(sent)        
        #else:  
        #    train.append(sent)   
        
        if rev["split"]== 1:
            train.append(sent)
            
        elif rev["split"]== 0:
            test.append(sent)
            
    #train = np.array(train,dtype="int")
    #test = np.array(test,dtype="int")

    random.shuffle(train)


    trainX, trainY = [], []
    testX, testY = [], []

    for item in train:
        trainX.append(item[:-1])
        trainY.append(item[-1])
    for item in test:
        testX.append(item[:-1])
        testY.append(item[-1])

    return trainX, trainY, testX, testY

def get_onehot_index(array):
    for i in range(0, len(array)):
        if array[i] == 1.0:
            return i
    return None

def LSTM(dst_folder, idx, epoch_num):
    random.seed(1)
    print ("loading data...",)
    x1 = cPickle.load(open( dst_folder + "thread" + idx + "/mr.p", "rb"))
    revs, W, W2, word_idx_map, vocab = x1[0], x1[1], x1[2], x1[3], x1[4]
    print ("data loaded!")

    i = 0
    trainX, trainY, testX, testY = make_idx_data_cv(revs, word_idx_map, max_l=100)


    # IMDB Dataset loading
    #train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
    #                                valid_portion=0.1)

    #trainX, trainY = train
    #testX, testY = test

    #print (testX)
    #raw_input("continue?")
    #print (testY)
    #raw_input("continue?")

    # Data preprocessing
    # Sequence padding

    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    #print (testY[0])
    testY = to_categorical(testY, nb_classes=2)
    #print (testY[0])
    #raw_input("continue?")

    W = tf.constant(W, name="W")
    print (W.get_shape())
    raw_input("continue?")

    # Network building
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=30000, output_dim=128, weights_init= W, trainable=False, name="EmbeddingLayer")
    net = tflearn.lstm(net, 128, dropout=0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch = epoch_num, validation_set=(testX, testY), show_metric=True,
              batch_size= 32)
    predictY = model.predict(testX)

    output = open(dst_folder + "thread" + idx + "/LSTM_true_and_pred_value", 'w')

    for i, value in enumerate(predictY):
        output.write(str(get_onehot_index(testY[i]))+ '\t')
        if value[0] > value[1]:
            output.write('0\t')
        else:
            output.write('1\t')
        output.write(str(value) + '\n')
    output.close()


def LSTM_main(dst_folder, parallel_num, epoch_num):
    processV = []
    for i in range(0, parallel_num):
        processV.append(Process(target = LSTM, args = (dst_folder, str(i), epoch_num, )))
    
    for i in range(0, parallel_num):
        processV[i].start()
        
    for i in range(0, parallel_num):
        processV[i].join()
