# coding:utf-8
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 100
TRAINING_STEPS = 30000

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist.ckpt"

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True
        )
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('../log', tf.get_default_graph())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            if i % 1000 == 0:
                # 配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录信息的proto
                run_metadata = tf.RunMetadata()
                # 将配置信息和proto传入运行的过程, 记录运行时每一个节点的时间、空间开销信息
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys}, options=run_options, run_metadata=run_metadata)
                # 将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata, "step{:03d}".format(i))

                print "After {} training step(s), loss on training batch is {}".format(step, loss_value)
                # global_step参数会让保存模型的文件名末尾加上训练轮数, 如"model.ckpt-1000"表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
        writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
