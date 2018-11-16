# coding:utf-8
import tensorflow as tf
from numpy.random import RandomState  # 用于生成模拟数据集


batch_size = 8  # 训练数据的大小

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数, clip_by_value将数值限定在1e-10和1.0之间, 高于1.0的值会被替换成1.0
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  # 反向传播算法

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]  # x1 + x2 < 1 为正样本, 用1表示, 0表示负样本

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print sess.run(w1)
    print sess.run(w2)

    STEPS = 5000  # 训练轮数
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 交叉熵, 交叉熵越小, 预测结果与真实结果误差越小
            # 给定两个概率分布p和q, H(p, q) = - Sigma(p(x) * log(q(x))), p代表正确答案, q代表预测值
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_:Y})
            print "After {} training step(s), cross entropy on all data is {}".format(i, total_cross_entropy)

    print sess.run(w1)
    print sess.run(w2)
