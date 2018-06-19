import numpy as np
from audio_loader import load_audio
import tensorflow as tf

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

nClasses=10

(x_train, y_train), (x_test, y_test) = load_audio("speech_commands", nClasses, reshape=False)

sess=tf.InteractiveSession()

#declare input placeholders to which to upload data
tfX=tf.placeholder(dtype=tf.float32,shape=[None,x_train.shape[1]])
tfY=tf.placeholder(dtype=tf.float32,shape=[None,nClasses])

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
    xs, ys = next_batch(nMinibatch,x_train,y_train)
    [temp,currLoss]=sess.run([optimize,loss],feed_dict={tfX:xs,tfY:ys})
    print("Loss {}".format(currLoss))