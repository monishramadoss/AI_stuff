import tensorflow as tf
from LoadFromFile import *

import sys
import argparse
from numpy import iterable

MAX_DOCUMENT_LENGTH = 10188
EMBEDDING_SIZE = 300
MAX_LABEL = 64

configOpt = [int(x) for x in open('config.txt').read().split(' ')]

x_train, y_train, x_test, y_test = LoadData(configOpt[0], configOpt[1], configOpt[2], .8)
logfile = open("ExpLog.txt","w+")
logfile.close()

class NeuralModel: 
    FinishFlag = False
    _N_FILTERS = 0
    _WINDOW_SIZE = 0
    _POOLING_WINDOW = 0
    _POOLING_STRIDE = 0
    _LEARNING_RATE = .001
    _EPOCH = 10
    _BATCH_SIZE = 128
    
    def __init__(self,f,w,pw,ps):
        self._N_FILTERS = f
        self._WINDOW_SIZE = w
        self._POOLING_WINDOW = pw
        self._POOLING_STRIDE = ps
    
    def cnn_model(self, features, labels, mode):
        words_vectors = features['sents']
        #word_vectors = tf.expand_dims(word_vectors, 3)
        #print(words_vectors.get_shape().as_list())
        with tf.variable_scope('CNN_LAYER_1'):
            conv1 = tf.layers.conv2d(words_vectors, filters=self._N_FILTERS, kernel_size=[self._WINDOW_SIZE, EMBEDDING_SIZE], padding='VALID', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=self._POOLING_WINDOW, strides=self._POOLING_STRIDE, padding='SAME')
            pool1 = tf.transpose(pool1,[0,1,3,2])
        
        with tf.variable_scope('CNN_LAYER_2'):
            conv2 = tf.layers.conv2d(pool1, filters=2*self._N_FILTERS, kernel_size=[self._WINDOW_SIZE, self._N_FILTERS], padding='VALID', activation= tf.nn.relu)
            rmax = tf.reduce_max(conv2,1)
            pool2 = tf.squeeze(rmax, axis=[1])
        #print(pool2.get_shape().as_list())
        pool2 = tf.layers.batch_normalization(pool2,training=mode==tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(pool2, MAX_LABEL, activation=tf.nn.tanh)
        
        predicted_classes = tf.argmax(logits, 1)
        if(mode == tf.estimator.ModeKeys.PREDICT):
            return tf.estimator.EstimatorSpec(mode=mode, predictions={'class': predicted_classes,'prob': tf.nn.softmax(logits)})
        
        #loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels) )
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        if(mode == tf.estimator.ModeKeys.TRAIN):
            optimizer=tf.train.AdamOptimizer(learning_rate=self._LEARNING_RATE)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        
        eval_metric_ops = {'accuracy' : tf.metrics.accuracy(labels=labels, predictions=predicted_classes)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
   
    def main(self, unusedarg):
        classifier = tf.estimator.Estimator(model_fn=self.cnn_model, model_dir='./checkpoint/')
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sents' : x_train},
            y=y_train,
            batch_size=self._BATCH_SIZE,
            num_epochs=self._EPOCH,
            shuffle=True
            )
        
        classifier.train(input_fn=train_input_fn,steps=100)
        
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sents': x_test},
            y=y_test,
            num_epochs=1,
            shuffle=False
            )
        
        logString = str(self._N_FILTERS) + " " + str(self._WINDOW_SIZE) + " " + str(self._POOLING_WINDOW) + " " + str(self._POOLING_STRIDE)
        
        logfile = open("ExpLog.txt","a")
        scores = classifier.evaluate(input_fn=test_input_fn)
        logfile.write(logString + " " +  str(scores['accuracy']) + "\n")
        self.FinishFlag = True
        logfile.close()

def poolExperiment():
    import itertools
    import time
    import multiprocessing
    import threading
    filterSizes = [32,50,64,100,128,256,512,1024]
    windowSizes = [x for x in range(1,5)]
    poolingWindows = [x for x in range(1,31)]
    poolingStrides = [x for x in range(1,31)]
    
    iterables = [filterSizes,windowSizes,poolingWindows,poolingStrides]
    config = itertools.product(*iterables)
        
    models = list()
        
    for f,w,pw,ps in config:
        models.append(NeuralModel(f,w,pw,ps))
    
    finishflag = models[0].FinishFlag
    counter = 0
    

    def threadTask(model):
        model.main("")
        finishflag = model.FinishFlag
        
    while True:
        if(not finishflag):
            #t = threading.Thread(target=threadTask, args=(models[counter],))
            t = multiprocessing.Process(target=threadTask, args=(models[counter],))
            counter += 1
            t.start()
            t.join()
    
    '''
    for model in models:
        try:
            model.main("HJEL")
        except:
            logfile.write("E\n")
    ''' 
if(__name__ == '__main__'):
   poolExperiment()


    
