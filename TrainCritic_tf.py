import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print("Loading MNIST")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess=tf.InteractiveSession()

#declare input placeholders to which to upload data
nClasses=10
tfX=tf.placeholder(dtype=tf.float32,shape=[None,784]);
tfY=tf.placeholder(dtype=tf.float32,shape=[None,nClasses]);

#build model
layer=tf.layers.dense(tfX,64,activation=tf.nn.selu)
layer=tf.layers.dense(layer,10)
loss=tf.losses.softmax_cross_entropy(tfY,layer)
optimizer=tf.train.AdamOptimizer()
optimize=optimizer.minimize(loss)

#initialize variables
tf.global_variables_initializer().run(session=sess)

#optimize
for _ in range(1000):
    nMinibatch=64
    xs, ys = mnist.train.next_batch(nMinibatch)
    [temp,currLoss]=sess.run([optimize,loss],feed_dict={tfX:xs,tfY:ys})
    print("Loss {}".format(currLoss))

