# coding:utf-8
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import mnist_inference
import mnist_train


EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.validation.num_examples, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        reshape_x = np.reshape(mnist.validation.images, (mnist.validation.num_examples, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshape_x, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, False, None)  # 测试时不关注正则化损失
        # 计算预测结果正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 变量重命名, 直接获取对应变量的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获得模型保存时迭代轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print "After {} training step(s), validation accuracy = {}".format(global_step, accuracy_score)

                else:
                    print "No checkpoint file found"
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
