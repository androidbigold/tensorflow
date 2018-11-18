# coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30  # LSTM中隐藏节点个数
NUM_LAYERS = 2  # LSTM层数
TIMESTEPS = 10  # 循环神经网络截断长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 10000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔

def generate_data(seq):
    X = []
    y = []
    # 序列第i项和后面的TIMESTEPS-1项合在一起作为输入, 第i+TIMESTEPS项作为输出。即用sin函数前面的TIMESTEPS个点的信息, 预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y):
    # 使用多层LSTM结构
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
    x_ = tf.unpack(X, axis=1)

    # RNN网络前向传播
    output, _ = tf.nn.rnn(cell, x_, dtype=tf.float32)
    # 本问题中只关注最后一个时刻的输出结果, 该结果为下一时刻的预测值
    output = output[-1]

    # 对LSTM网络的输出再加一层全连接层并计算损失(默认为平均平方差损失)
    prediction, loss = learn.models.linear_regrssion(output, y)

    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer="Adagrad", learning_rate=0.1)

    return prediction, loss, train_op

regressor = learn.Estimator(model_fn=lstm_model)

# 用正弦函数生成训练数据和测试数据
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 调用fit函数训练模型
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

predicted = [[pred] for pred in regressor.predict(test_X)]

rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print "Mean Square Error is: {}".format(rmse[0])

fig = plt.figure()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.savefig('sin.png')
