import tensorflow as tf
import numpy as np

# Create 1000 phony x, y, z data points, z = 2x^2 - 3y^2 + 1 + noise
n = 1000
x_data = np.random.uniform(-2, 2, n)
y_data = np.random.uniform(-2, 2, n)
z_data = 2 * x_data**2 - 3 * y_data**2 + 1

z_in = np.reshape(z_data, (-1, 1))

sess = tf.InteractiveSession()

L = tf.placeholder(tf.float32, [None, 2], name='L')

D_h = 10

W_in = tf.Variable(tf.truncated_normal([2, D_h], stddev=0.1), name='W_in')
b_in = tf.Variable(tf.zeros([1, D_h]), name='b_in')
M = tf.tanh(tf.matmul(L, W_in) + b_in)

W_out = tf.Variable(tf.truncated_normal([D_h, 1], stddev=0.1), name='W_out')
b_out = tf.Variable(tf.zeros([1, 1]), name='b_out')
z = tf.matmul(M, W_out) + b_out

z_ = tf.placeholder(tf.float32, [None, 1], name='z_')

loss = tf.nn.l2_loss(z - z_)
optimizer = tf.train.AdamOptimizer(5e-2)
train = optimizer.minimize(loss)

feed = [x_data, y_data]
L_in = [list(i) for i in zip(*feed)]

sess.run(tf.global_variables_initializer())

d = {L:L_in, z_:z_in}
for i in range(1000):
  sess.run(train, feed_dict = d)
  if i % 20 == 0:
    print("%s - loss: %s"%(i, loss.eval(feed_dict = d)))
