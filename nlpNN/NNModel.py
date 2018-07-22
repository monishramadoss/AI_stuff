import os

import tensorflow as tf
import numpy
from PreProcess import *
import time
import multiprocessing



cores = multiprocessing.cpu_count()+5

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['MKL_NUM_THREADS'] = str(cores)
os.environ['GOTO_NUM_THREADS'] = str(cores)
os.environ['OMP_NUM_THREADS'] = str(cores)
os.environ['openmp'] = 'True'


n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000
n_nodes_hl5 = 1000
n_nodes_hl6 = 1000

train_x,train_y,n_classes,classes = PreProcess.create_feature_set_and_labels('.\t_data.csv')
temp = list()
batch_size = 5
dimx = len(train_x[0])
x = tf.placeholder("float",[None, dimx])
y = tf.placeholder("float")

def neural_network_model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([dimx,n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_nodes_hl5])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}
    hidden_6_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5,n_nodes_hl6])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl6]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl6,n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    
    l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)

    l6 = tf.add(tf.matmul(l5,hidden_6_layer['weights']), hidden_6_layer['biases'])
    l6 = tf.nn.relu(l6)
    

    output = tf.add(tf.matmul(l6,output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x):
    print("Starting training")
    prediction = neural_network_model(x)
  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 30
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=cores)) as sess:

        sess.run(tf.global_variables_initializer())
        #train
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = numpy.array(train_x[start:end])
                batch_y = numpy.array(train_y[start:end])
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch ',epoch + 1,'completed out of ', hm_epochs, 'loss ',epoch_loss)
            
       
        saver.save(sess=sess,save_path="./model/model_variables.ckpt")
        sess.close()

def prediction():
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    inpt = "I need someone to pick up my package"
    dta = PreProcess.fromFeatureSet(inpt)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,save_path="./model/model_variables.ckpt")
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:[dta]}),1))
        print(result)
        print(classes)
        print(classes[result[0]])
        sess.close()


if(__name__ == '__main__'):
    start = time.time()
    train_neural_network(x)
    end = time.time()
    #prediction()
