import tensorflow as tf
import numpy as np


# Create 1000 phony x, y data points, y = 2x + 0.5 + noise
n = 1000
x_data = np.linspace(-2, 2, n)
y_data = x_data * 2 + 0.5 + np.random.uniform(-0.5, 0.5, n)

x_in = np.reshape(x_data, (-1, 1))
y_in = np.reshape(y_data, (-1, 1))

# Try to find values for W and b that compute y_data = W * x_data + b
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable([-0.1], tf.float32)
b = tf.Variable([0.1], tf.float32)
y_ = W * x + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# Launch the graph and initialize the variables.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
d = {x:x_in, y:y_in}

# Fit the line (Learns best fit is W: 2, b: 0.5)
for step in range(100):
  sess.run(train, feed_dict=d)
  if step % 10 == 0:
    print("%s - W: %s b: %s loss: %s"%(step, sess.run(W, feed_dict=d), sess.run(b, feed_dict=d), sess.run(loss, feed_dict=d)))
