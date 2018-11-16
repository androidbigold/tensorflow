# coding:utf-8
import tensorflow as tf

# 定义神经网络结构相关参数
INPUT_NODE = 784  # 输入层节点数, 对于MNIST数据集等于图片像素
OUTPUT_NODE = 10  # 输出层节点数, 在MNIST数据集中需要区分0-9这10个数字
LAYER1_NODE = 500  # 隐藏层节点数, 这里只有一个隐藏层, 该层有500个节点

# 通过tf.get)variable函数来获取变量, 训练时会创建和谐变量, 在测试时通过保存的模型加载这些变量的取值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))  # 将当前权重的正则化损失加入集合losses
    return weights

def inference(input_tensor, regularizer):
    # 第一层神经网络的命名空间
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 第二层神经网络的命名空间
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
