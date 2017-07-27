import tensorflow as tf
import numpy as np
import pickle

print('Importing data ...')
train_x,train_y,test_x,test_y = pickle.load(open('feature_set_emptySpaces.pickle', 'rb'))
print('Done importing data.')

flat_img_length = 32768

n_classes = 20
batch_size = 128

x = tf.placeholder('float', shape=[None, flat_img_length])
y = tf.placeholder('float', shape=[None, n_classes])

keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    # For reference: tf.truncated_normal([conv_width, conv_height, num_of_inputs, num_of_outputs])
    weights = {'W_conv1':tf.Variable(tf.truncated_normal([5,5,1,10])),
               'W_conv2':tf.Variable(tf.truncated_normal([5,5,10,20])),
               'W_conv3':tf.Variable(tf.truncated_normal([5,5,20,30])),
               #[(imgHeight/2)/2 * (imgWidth/2)/2 * depth of previous layer]
               'W_fc1':tf.Variable(tf.truncated_normal([16*32*30,30])),
               'W_fc2':tf.Variable(tf.truncated_normal([30,20])),
               'W_fc3':tf.Variable(tf.truncated_normal([20,10])),
               'out':tf.Variable(tf.truncated_normal([10, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.truncated_normal([10])),
               'b_conv2':tf.Variable(tf.truncated_normal([20])),
               'b_conv3':tf.Variable(tf.truncated_normal([30])),
               'b_fc1':tf.Variable(tf.truncated_normal([30])),
               'b_fc2':tf.Variable(tf.truncated_normal([20])),
               'b_fc3':tf.Variable(tf.truncated_normal([10])),
               'out':tf.Variable(tf.truncated_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 128, 256, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    fc1 = tf.reshape(conv3,[-1, 16*32*30])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])
    fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])
    fc2 = tf.nn.dropout(fc2, keep_prob)
    fc3 = tf.nn.relu(tf.matmul(fc2, weights['W_fc3'])+biases['b_fc3'])
    fc3 = tf.nn.dropout(fc3, keep_prob)

    output = tf.matmul(fc3, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    hm_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            batch_start = 0
            for _ in range(int(len(train_x)/batch_size)+1):
                batch_end = batch_start + batch_size
                epoch_x = train_x[batch_start:batch_end]
                epoch_y = train_y[batch_start:batch_end]
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob:.5})
                epoch_loss += c
                batch_start += batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        print('Accuracy:',accuracy.eval({x:test_x, y:test_y, keep_prob:1.0}))

train_neural_network(x)
