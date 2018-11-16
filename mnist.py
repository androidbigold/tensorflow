# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# MNIST数据集相关常数
INPUT_NODE = 784  # 输入层节点数, 对于MNIST数据集等于图片像素
OUTPUT_NODE = 10  # 输出层节点数, 在MNIST数据集中需要区分0-9这10个数字

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数, 这里只有一个隐藏层, 该层有500个节点
BATCH_SIZE = 100  # 一个训练batch中训练数据个数, 数字越小训练过程越接近随机梯度下降, 数字越大训练过程越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 没有提供滑动平均类时, 直接使用参数当前取值
    if avg_class is None:
        # 计算隐藏层前向传播结果, 使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 计算输出层前向传播结果, 因为在计算损失函数时会一并计算softmax函数, 所以这里不需要加入激活函数
        # 因为预测时使用的是不同类别对应节点输出值的相对大小, 有没有softmax层对最后分类结果的计算没有影响
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数下神经网络前向传播的结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量, 这个变量不需要计算滑动平均值, 所有指定这个变量为不可训练的变量
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数变量, 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数变量上使用滑动平均
    # tf.trainable_variables返回图上集合GraphKeys.TRAINABLE_VARIABLES中的元素, 这个集合的元素就是所有没有指定trainable=False的参数
    variable_average_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数(用预测结果表达正确的标签), argmax用于选出最可能的分类结果
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)

    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失, 一般只计算神经网络边上权重的正则化损失, 而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    # 使用train.GradientDescentOptimizer优化算法来优化损失函数, minimize用于最小化loss并使global_step自增
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 训练神经网络模型时, 每过一遍数据既需要通过反向传播来更新神经网络中的参数, 又要更新每一个参数的滑动平均值
    # train_op = tf.group(train_step, variables_averages_op) 与下述写法等价
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 计算模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据, 一般在神经网络训练过程中通过验证数据来大致判断停止的条件和评判训练效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 准备测试数据, 在真实应用中, 这部分数据在训练时是不可见的, 这个数据只是作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据和测试数据上的结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print "After {} training step(s), validation accuracy using average model is {}, test accuracy using average model is {}".format(i, validate_acc, test_acc)

            # 产生这一轮使用的一个batch的训练数据, 并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束之后, 在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print "After {} training step(s), test accuracy using average model is {}".format(TRAINING_STEPS, test_acc)

def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
