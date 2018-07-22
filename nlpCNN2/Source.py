import tensorflow as tf
from LoadFromFile import *
import sys
import argparse
from numpy import iterable
import time
import os
import multiprocessing
import threading

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
configOpt = [int(x) for x in open('config.txt').read().split(' ')]
x_train, y_train, x_test, y_test = LoadData(1, configOpt[0], configOpt[1], configOpt[2], .8)
MAX_DOCUMENT_LENGTH = configOpt[0]
EMBEDDING_SIZE = configOpt[1]
SENTENCE_SIZE = configOpt[2]

MAX_LABEL = 2

MultiCellEnable = True
CNNLayer2Enable = False



class NeuralModel: 
    _UNITS = 0
    _N_FILTERS = 0
    _WINDOW_SIZE = 0
    _POOLING_WINDOW = 0
    _POOLING_STRIDE = 0
    _EPOCH = 0
    _BATCH_SIZE = 0 
    
    number_cnn_layers = 3
    number_rnn_layers = 1
    
    _LEARNING_RATE = .001
    
    _DROPOUT = .8   
    _train = True
    
    def __init__(self,u,f,w,pw,ps,e,bs):
        self._UNITS = u
        self._N_FILTERS = f
        self._WINDOW_SIZE = w
        self._POOLING_WINDOW = pw
        self._POOLING_STRIDE = ps
        self._EPOCH = e
        self._BATCH_SIZE = bs
    
    def model(self, inpt, LstmSeq):
        with tf.device('/cpu:0'):
            inpt = tf.reshape(inpt,[-1,32,300,1])
        
        def cnn_layer(filters,windowsize):
            sublayer1 = tf.layers.conv2d(inpt, filters=filters, kernel_size=[windowsize, EMBEDDING_SIZE], padding='SAME', activation=tf.nn.relu)
            sublayer1 = tf.layers.max_pooling2d(sublayer1, pool_size=self._POOLING_WINDOW, strides=1, padding='SAME')        
            sublayer1 = tf.transpose(sublayer1, [0,1,3,2])
            
            if(CNNLayer2Enable):
                print("IN", sublayer1.get_shape().as_list())
                sublayer2 = tf.layers.conv2d(sublayer1, filters=filters, kernel_size=[windowsize, self._N_FILTERS], padding='SAME', activation= tf.nn.relu)
                sublayer2 = tf.layers.max_pooling2d(sublayer2, pool_size=1, strides=1, padding='SAME')
                rmax = tf.reduce_max(sublayer2,2)
                print("OT", rmax.get_shape().as_list())
                
                return rmax
            else:               
                return tf.reduce_max(sublayer1,3)
                         
        with tf.variable_scope('CNN_LAYER_1'):
            cnn_layers = list()
            for i in range(0, self.number_cnn_layers):
                cnn_output = cnn_layer(self._N_FILTERS, self._WINDOW_SIZE-i)
                cnn_layers.append(cnn_output)
            
            #print(cnn_layers[0].get_shape().as_list())
            cnn_o = tf.concat(cnn_layers, axis = 2)
            #if(CNNLayer2Enable):
            #    cnn_o = tf.squeeze(cnn_o, axis=[1])
            cnn_output = cnn_o        
            
            print("CNNO", cnn_o.get_shape().as_list())
        
        with tf.device('/cpu:0'):
            num_filter_total = self._N_FILTERS * 3
            cnn_output = tf.nn.dropout(cnn_output, self._DROPOUT)
            
            cnnshape = cnn_output.get_shape().as_list()[1:]
            Rinpt = tf.reshape(cnn_output,[-1, np.prod(cnnshape),1])
        
        def cell(size):
            layer = tf.contrib.rnn.LSTMCell(size)
            return tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob = 0.9)
        
        with tf.variable_scope('RNN_LAYER_1'):
            cells = None
            if(MultiCellEnable):
                cells = tf.contrib.rnn.MultiRNNCell([cell(self._UNITS) for _ in range(self.number_rnn_layers)])  
            else:
                cells = cell(self._UNITS)
            (rnn_output, state) = tf.nn.dynamic_rnn(cell=cells, inputs=Rinpt, dtype=tf.float32, sequence_length=LstmSeq)        
        
        
        with tf.device('/cpu:0'):
            rnnshape = rnn_output.get_shape().as_list()[1:]
            rnn_output = tf.reshape(rnn_output,[-1,np.prod(rnnshape)])

            norm = tf.layers.batch_normalization(rnn_output,training=self._train)
            output = tf.layers.dense(norm, MAX_LABEL, activation=tf.nn.tanh)  
            
        return output
   
    def main(self):
        from datetime import timedelta
        t0 = time.time()
        logString = str(self._UNITS) + " " + str(self._N_FILTERS) + " " + str(self._WINDOW_SIZE) + " " + str(self._POOLING_WINDOW) + " " + str(self._POOLING_STRIDE) + " " + str(self._EPOCH) + " " + str(self._BATCH_SIZE)                     
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        LstmSeq = tf.placeholder('float')
        
        prediction = self.model(x, LstmSeq)
        cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=y) )
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        config = tf.ConfigProto()
        epoch_loss = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            #train
            for epoch in range(self._EPOCH):
                epoch_loss = 0
                i = 0
                while i < len(x_train):
                    start = i
                    end = i + self._BATCH_SIZE
                    batch_x = np.array(x_train[start:end])
                    batch_y = np.array(y_train[start:end])
                    lstmseq = np.array([SENTENCE_SIZE for _ in range(len(batch_x))])
                    if(len(batch_x) == self._BATCH_SIZE):
                        _,c = sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y, LstmSeq:lstmseq})
                        epoch_loss += c
                    i+=self._BATCH_SIZE
                    
            self._train = False
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))            
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            lstmseq = np.array([SENTENCE_SIZE for _ in range(len(x_test))])         
            writeout = logString + " " +  str(accuracy.eval({x:x_test, y:y_test, LstmSeq:lstmseq}))
            
            print(writeout, epoch_loss, end=" # ")
            
        elapsed = time.time()-t0
        print(str(timedelta(seconds=elapsed)))
        
        #sys.stdout.flush()
            

def TestModel():    
    #model = NeuralModel(200,256,3,7,8,10,64)
    model = NeuralModel(100,100,3,7,8,10,64)
    
    opt = model.main()
    

def poolExperiment():
    import itertools
    
    UnitSize = [200]
    filterSizes = [256]
    windowSizes = [1,2,3,4]
    poolingWindows = [8, 10, 12, 16, 18]
    poolingStrides = [12,10, 16, 18]
    epochs = [10]
    batch_sizes = [128]
    
    iterables = [UnitSize,filterSizes,windowSizes,poolingWindows,poolingStrides,epochs,batch_sizes]
    config = itertools.product(*iterables)
        
    models = list()
    models.append(NeuralModel(100,100,3,8,8,60,64))
    for u,f,w,pw,ps,e,bs in config:
        models.append(NeuralModel(u,f,w,pw,ps,e,bs))
        
    for model in models:
        t = multiprocessing.Process(target=model.main)
        t.start()
        t.join()
        
                
if(__name__ == '__main__'):
    TestModel()
    #poolExperiment()


    
