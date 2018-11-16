# coding:utf-8
import os


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100  # 一个训练batch中训练数据个数, 数字越小训练过程越接近随机梯度下降, 数字越大训练过程越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
# 模型保存路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # L2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 对所有可训练变量定义滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 当分类问题只有一个正确答案时, 可以使用sparse_softmax_cross_entropy_with_logits函数加速交叉熵的计算
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    # minimize函数自动更新global_step并最小化loss, 返回更新后的参数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 通过反向传播更新神经网络参数, 同时更新每一个参数的滑动平均值
    # 控制依赖
    with tf.control_dependencies([train_step, variable_averages_op]):
        # 确保train_step, variable_averages_op被执行, 该部分在这之后执行
        train_op = tf.no_op(name='train')  # 空操作, 什么也不做

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 10 == 0:
                print "After {} training step(s), loss on training batch is {}".format(step, loss_value)
                # global_step参数会让保存模型的文件名末尾加上训练轮数, 如"model.ckpt-1000"表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
