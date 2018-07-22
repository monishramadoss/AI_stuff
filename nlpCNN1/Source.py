#@title Default title text
import numpy as np
import tensorflow as tf


t_dataLength = 10188
sentenceLength = 64
VectorLength = 300
batch_size = 10
ValidatePercent = 0.8
hm_epochs = 5
keep_rate = 0.8
num_Filter_1 = 32
num_Filter_2 = 64


print("Loading Data...")
#X_ = np.array()


with open("x.txt") as f:
    X = np.array(f.read().split(' ')[:-1],copy=False,dtype=np.float16)  
    X = np.reshape(X,[t_dataLength,sentenceLength,VectorLength])
    
    print(len(X),len(X[10187]),len(X[10187][63]))
    print("X_FILE READ DONE")
 

Y_ = list() 
with open("y.txt") as f:
    for labeldata in f.readlines():
        label = labeldata.split()
        Y_.append(label)
        
print(len(Y_),len(Y_[0]))
print('Y_FILE READ DONE')
print('LoadingNP_array...\n')

Y = np.array(Y_,dtype=np.int16)

n_classes = len(Y[0])

train_x = X[:int(len(X) * ValidatePercent) ]
train_y = Y[:int(len(Y) * ValidatePercent)]
test_x = X[int(len(X) * ValidatePercent):]
test_y = Y[int(len(Y) * ValidatePercent):]

#data = np.reshape(train_x, [-1,64,300,1])


x = tf.placeholder('float')
y = tf.placeholder('float')



keep_prob = tf.placeholder(tf.float32)

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

def convolutional_neural_network(x):
    words = 1
    
    weights = {'W_conv1':tf.Variable(tf.random_normal([1,300,1,num_Filter_1])),
               'W_conv2':tf.Variable(tf.random_normal([1,300,num_Filter_1,num_Filter_2])),
               'W_fc':tf.Variable(tf.random_normal([sentenceLength * VectorLength * num_Filter_2,64])),
               'out':tf.Variable(tf.random_normal([64, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([num_Filter_1])),
               'b_conv2':tf.Variable(tf.random_normal([num_Filter_2])),
               'b_fc':tf.Variable(tf.random_normal([64])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, sentenceLength, VectorLength, 1])

    conv1 = tf.nn.relu(conv(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool(conv1)
    
    conv2 = tf.nn.relu(conv(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool(conv2)

    fc = tf.reshape(conv2,[-1, sentenceLength*VectorLength*num_Filter_2])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #train
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch ', epoch + 1, ' completed out of ' ,  hm_epochs ,  ' loss ' , epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ' + str(accuracy.eval({x:test_x, y:test_y})))


train_neural_network(x)


