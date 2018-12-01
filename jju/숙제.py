import tensorflow as tf
import numpy as np
import sklearn
tf.set_random_seed(777)
x_data = []
y_data = []

for i in range(1,96):
    for i2 in range(5):
        x_data.append(np.float(i+i2))
# print(x_data)
x_data = np.array(x_data)
x_data = x_data.reshape(-1,5)
print(x_data)

for i in range(6,101):
    y_data.append(np.float(i))
y_data = np.array(y_data)
y_data = y_data.reshape(-1,1)

print(y_data)

X = tf.placeholder(tf.float32,[None,5])
Y = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.random_normal([5,20]))
b = tf.Variable(tf.random_normal([20]))

W2 = tf.Variable(tf.random_normal([20,10]))
b2 = tf.Variable(tf.random_normal([10]))

W3 = tf.Variable(tf.random_normal([10,1]))
b3 = tf.Variable(tf.random_normal([1]))


hy1 = tf.nn.relu(tf.matmul(X,W)+b)
hy2 = tf.nn.relu(tf.matmul(hy1,W2)+b2)
hypothesis = tf.sigmoid(tf.matmul(hy2,W3)+b3)




sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         # summary 실행 및 파일 기록
#         sess.run([train], feed_dict={X: x_data, Y: y_data})
#
#         if step % 1000 == 0:
#             print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    # print(h, c, a)





# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate=1e-100).minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# for step in range(10001):
#     sess.run([train], feed_dict={X: x_data, Y: y_data})
#
#     if step % 1000 == 0:
#         print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
# print(h, c, a)