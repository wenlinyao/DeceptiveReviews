"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
from theano import tensor as printing
import re
import warnings
import sys
import time
from multiprocessing import Process
from conv_net_classes import *
warnings.filterwarnings("ignore")   




#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(dst_folder, idx, datasets,
                   U,
                   img_w=300, 
                   filter_hs=[3, 4, 5],
                   hidden_units=[100,3], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=100, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                ):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    print U
    print len(U)
    print len(U[0])
    #raw_input("continue?")

    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    #t_img_h = len(test_dataset[0][0])-1
    #print "img_h, t_img_h: ", img_h, ', ', t_img_h
    
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))# feature_maps = 100; filter_h = 3, 4, 5; filter_w = 300
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    index = T.lscalar()
    x = T.matrix('x')   # data, presented as rasterized images
    y = T.ivector('y') # labels, presented as 1D vector of [int] labels
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    print "datasets[0].shape[0]: ", datasets[0].shape[0]
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    print "datasets[1].shape[0]: ", datasets[1].shape[0]

    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[1].shape[0] % batch_size
        test_set = np.random.permutation(datasets[1])   
        extra_data = test_set[:extra_data_num]
        test_new_data=np.append(datasets[1],extra_data,axis=0)
    else:
        test_new_data = datasets[1]
    print "test_new_data.shape[0]: ", test_new_data.shape[0]
    
    n_test_batches = test_new_data.shape[0]/batch_size
    
    new_data = np.random.permutation(new_data)
    #ratio = float(train_val_boundary) / float(new_data.shape[0])
    #print ratio
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #n_train_batches = int(np.round(n_batches*ratio))
    
    
    #divide train set into train/val sets 
    #test_set_x = datasets[1][:,:img_h] 
    #test_set_y = np.asarray(datasets[1][:,-1],"int32")
    test_y = np.asarray(datasets[1][:,-1],"int32")
    
    test_set = test_new_data[:n_test_batches * batch_size, :]
    
    train_set = new_data[:n_train_batches*batch_size,:]
    #train_set = train_new_data[:n_train_batches*batch_size,:]
    
    
    val_set = new_data[n_train_batches*batch_size:,:]
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    test_set_x, test_set_y = shared_dataset((test_set[:,:img_h], test_set[:, -1]))
    
    n_val_batches = n_batches - n_train_batches
    #n_val_batches = t_n_batches
    
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
                                
                                
    #val_output_sigmoid = theano.function([index], classifier.predict_p(val_set_x[index * batch_size: (index + 1) * batch_size]))

    
    #compile theano functions to get train/val/test errors
    #classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    real_test_model = theano.function ( [index], classifier.errors(y), 
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                 y: test_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)      
                                
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)     
    
    print "\n*****************************"
    print train_set_y.eval()
    
    #print type(test_set_y)
    test_pred_layers = []
    test_size = test_set_x.shape[0].eval()
    
    print "test_size = test_set_x.shape[0]: ", test_size
    print "x.flatten(): ", x.flatten()
    print "img_h: ", img_h
    print "Words.shape[0]: ", Words.shape[0].eval()
    print "Words.shape[1]: ", Words.shape[1].eval()
    # x.flatten(): A copy of the input array, flattened to one dimension.
    #test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1])) # change test_size to batch_size
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, batch_size) # change test_size to batch_size
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    print "test_layer1_input: ", test_layer1_input
    test_y_pred = classifier.predict(test_layer1_input)
    test_y_pred_p = classifier.predict_p(test_layer1_input)



    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)   
    
    #test_model_all_3 = theano.function([x], test_y_pred, allow_input_downcast=True)
    #test_model_final_layer = theano.function([x], test_y_pred_p, allow_input_downcast=True)
            
    test_model_all_3 = theano.function([index], test_y_pred, 
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True)
                        
    test_model_final_layer = theano.function([index], test_y_pred_p,
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size]},
                         allow_input_downcast=True)
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
    #while (epoch < 1):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        

        print(' epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            #test_loss = test_model_all(test_set_x,test_set_y)        
            #test_perf = 1- test_loss
            print "n_test_batches: ", n_test_batches
            test_losses = [real_test_model(i) for i in xrange(n_test_batches)]
            #test_losses = [real_test_model(0)]
            test_perf = 1 - np.mean(test_losses)
            
            print (' test perf: %.2f %%' %  (test_perf*100.))

    start_time = time.time()
    print "outputting test result..."
    output_test = open(dst_folder + "thread" + idx + "/CNN_true_and_pred_value", 'w')
    if best_val_perf != 0:
        
        for i in xrange(n_test_batches):
            test_y_pred_cur = test_model_all_3(i)
            test_y_final_layer = test_model_final_layer(i)
            #print "i: ", i
            if i != n_test_batches - 1:
                for j in range(0, batch_size):
                    #print j, ' ',
                    #output_test.write( str(test_set_y[i * batch_size + j ].eval()) + '\t' + str(test_y_pred_cur[j]) + '\t' + str(test_y_final_layer[j]) + '\n' )
                    output_test.write(str(test_y[i * batch_size + j]) + '\t' + str(test_y_pred_cur[j]) + '\t' + str(test_y_final_layer[j]) + '\n' )
                    #output_test.write(  '\t' + str(test_y_final_layer[j]) + '\n' )


            else:
                for j in range(0, datasets[1].shape[0] % batch_size):
                    #print j, ' ',
                    #output_test.write( str(test_set_y[i * batch_size + j ].eval()) + '\t' + str(test_y_pred_cur[j]) + '\t' + str(test_y_final_layer[j]) + '\n' )
                    output_test.write(str(test_y[i * batch_size + j]) + '\t' + str(test_y_pred_cur[j]) + '\t' + str(test_y_final_layer[j]) + '\n' )
                    #output_test.write(  '\t' + str(test_y_final_layer[j]) + '\n' )
        output_test.close()
        return test_perf
    else:
        output_test.close()
        return 0

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    if x >= max_l+2*pad:
        x = x[:max_l + pad]

    while len(x) < max_l+2*pad:
        x.append(0)

    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        """
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
        """
        if rev["split"]== 0:            
            test.append(sent)        
        elif rev["split"]== 1:  
            train.append(sent)   

    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")

    return [train, test]     
  
   
#if __name__=="__main__":
def CNN(dst_folder, idx, argv1, argv2):
    print "loading data...",
    
    x1 = cPickle.load(open( dst_folder + "thread" + idx + "/mr.p", "rb"))

    revs, W, W2, word_idx_map, vocab = x1[0], x1[1], x1[2], x1[3], x1[4]
    
    print "data loaded!"
    #mode= sys.argv[1]
    #word_vectors = sys.argv[2]    
    mode= argv1
    word_vectors = argv2
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("../CNN/conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,1)
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=100,k=300, filter_h=5)
        
        perf = train_conv_net(dst_folder, idx, datasets,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3, 4],
                              conv_non_linear="relu",
                              hidden_units=[100,2], 
                              shuffle_batch=True, 
                              n_epochs=25, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size= 50,
                              dropout_rate=[0.5],
                            
                        
                            )
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)  
    print str(np.mean(results))





def conv_net_sentence_main(dst_folder, parallel_num):
    processV = []
    for i in range(0, parallel_num):
        processV.append(Process(target = CNN, args = (dst_folder, str(i), "-static", "-word2vec", )))
    
    for i in range(0, parallel_num):
        processV[i].start()
        
    for i in range(0, parallel_num):
        processV[i].join()