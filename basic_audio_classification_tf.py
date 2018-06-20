import numpy as np
from audio_loader import load_audio
from audio_tools import count_convolutions
import tensorflow as tf

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

nClasses=10

(x_train, y_train), (x_test, y_test) = load_audio("speech_commands", nClasses)

kernel_size = 5
sess=tf.InteractiveSession()

convolution_layers = count_convolutions(x_train.shape, kernel_size)
print("{} convolution layers".format(convolution_layers))

#declare input placeholders to which to upload data
tfX=tf.placeholder(dtype=tf.float32,shape=[None,x_train.shape[1],1])
tfY=tf.placeholder(dtype=tf.float32,shape=[None,nClasses])

#build model
layer=tf.layers.conv1d(tfX,16,kernel_size,strides=2,activation=tf.nn.selu)
for i in range(convolution_layers):
    layer=tf.layers.conv1d(layer,32,kernel_size,strides=2,activation=tf.nn.selu)
layer=tf.layers.flatten(layer)
layer=tf.layers.dropout(layer,0.5)
layer=tf.layers.dense(layer,32,activation=tf.nn.selu)
layer=tf.layers.dropout(layer,0.5)


layer=tf.layers.dense(layer,nClasses)

loss=tf.losses.softmax_cross_entropy(tfY,layer)
optimizer=tf.train.AdamOptimizer()
optimize=optimizer.minimize(loss)

#initialize variables
tf.global_variables_initializer().run(session=sess)

g = tf.get_default_graph()

# And can inspect everything inside of it
for op in g.get_operations():
    print(op.name)

#optimize
for _ in range(1000):
    nMinibatch=64
    xs, ys = next_batch(nMinibatch,x_train,y_train)
    [temp,currLoss]=sess.run([optimize,loss],feed_dict={tfX:xs,tfY:ys})
    print("Loss {}".format(currLoss))