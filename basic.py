# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# matrix1 = tf.constant([[3., 3.]])  # 1X2矩阵
# matrix2 = tf.constant([[2.], [2.]])  # 2X1矩阵
#
# product = tf.matmul(matrix1, matrix2)  # 矩阵乘法
#
# sess = tf.Session()
#
# result = sess.run(product)
# print result
#
# sess.close()


# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable("v", initializer=tf.zeros_initializer, shape=[1])
#
# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable("v", initializer=tf.ones_initializer, shape=[1])
#
# with tf.Session(graph=g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("", reuse=True):
#         print sess.run(tf.get_variable("v"))
#
# with tf.Session(graph=g2) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("", reuse=True):
#         print sess.run(tf.get_variable("v"))


# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
#
# x = tf.constant([[0.7, 0.9]])
#
# # 前向传播算法
# a = tf.matmul(x, w1)  # 矩阵乘法
# y = tf.matmul(a, w2)
#
# sess = tf.Session()
# sess.run(w1.initializer)  # 初始化w1
# sess.run(w2.initializer)  # 初始化w2
# print sess.run(y)
# sess.close()


# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
#
# x = tf.placeholder(tf.float32, shape=(1, 2), name="input")  # 存放输入数据的地方
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
#
# sess = tf.Session()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
#
# print sess.run(y, feed_dict={x: [[0.7, 0.9]]})


# def get_weight(shape, lam):
#     var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
#     tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lam)(var))  # 将L2正则化损失加入集合losses
#     return var
#
# x = tf.placeholder(tf.float32, shape=(None, 2))
# y_ = tf.placeholder(tf.float32, shape=(None, 1))
# batch_size = 8
#
# layer_dimension = [2, 10, 10, 10, 1]  # 定义每一层网络中的节点个数
# n_layers = len(layer_dimension)  # 神经网络层数
#
# cur_layer = x  # 维护前向传播时当前层节点, 开始时是输入层
# in_dimension = layer_dimension[0]  # 当前层节点个数
#
# for i in range(1, n_layers):
#     out_dimension = layer_dimension[i]  # 下一层节点个数
#     weight = get_weight([in_dimension, out_dimension], 0.001)  # 计算当前层权重并将L2正则化损失加入计算图上的集合
#     bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
#     cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)  # ReLU激活函数
#     in_dimension = layer_dimension[i]
#
# mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))  # 计算均方误差
# tf.add_to_collection('losses', mse_loss)  # 将均方误差值加入集合losses
#
# loss = tf.add_n(tf.get_collection('losses'))  # 最终误差=均方误差+L2正则化误差
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     print sess.run(loss, feed_dict={x: [[0.0, 0.0], [1.0, 1.0]], y_: [[0], [1]]})


# mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
#
# print "Training data size: {}".format(mnist.train.num_examples)
# print "Validating data size: {}".format(mnist.validation.num_examples)
# print "Testing data size: {}".format(mnist.test.num_examples)
# print "Example training data: {}".format(mnist.train.images[0])
# print "Example training data label: {}".format(mnist.train.labels[0])
#
# batch_size = 100
# xs, ys = mnist.train.next_batch(batch_size)
#
# print "X shape: {}".format(xs.shape)
# print "Y shape: {}".format(ys.shape)


# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# result = v1 + v2
#
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess, "./model/test.ckpt")


# saver = tf.train.import_meta_graph("./model/test.ckpt.meta")
#
# with tf.Session() as sess:
#     saver.restore(sess, "./model/test.ckpt")
#     print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))


# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
#
# result = v1 + v2
#
# saver = tf.train.Saver()
# saver.export_meta_graph("./model/test.ckpt.meta.json", as_text=True)


# 前两个维度表示过滤器的尺寸, 第三个维度表示当前层的深度, 第四个维度表示过滤器的深度
filter_weight = tf.get_variable('weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

# 卷积层前向传播算法, 第一个参数为当前层节点矩阵, 这个矩阵是一个四维矩阵, 第一维对应一个输入batch, 如: input[0, :, :, :]表示第一张图片,
# 第二个参数为卷积层的权重, 第三个参数为不同维度上的步长, 因为卷积层的步长只对矩阵的长和宽有效, 所以第一维和最后一位一定是1,
# 最后一个参数为填充方式, SAME表示全0填充, VALID表示不填充
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.bias_add提供一个方便的函数给每一个节点加上偏置项, 这里不能直接使用加法, 因为矩阵上不同位置上的节点都需要加上同样的偏置项
bias = tf.nn.bias_add(conv, biases)

actived_conv = tf.nn.relu(bias)

# 最大池化层前向传播算法, ksize为过滤器尺寸, 池化层过滤器不能跨不同输入样例或节点矩阵深度, 所以第一维和最后一位必须是1
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')