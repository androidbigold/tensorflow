# coding:utf-8
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


# Inception-v3模型瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称
BOOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 模型文件目录
MODEL_DIR = '../../model/inception_dec_2015'

# 模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次, 所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中, 免去重复计算
CACHE_DIR = './tmp/bottleneck'

# 图片数据文件夹
INPUT_DATA = '../../DataSet/flower_photos'

# 验证数据百分比
VALIDATION_PERCENTAGE = 10

# 测试数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络相关参数
LEARENING_RATE = 0.01
STEPS = 4000
BATCH = 100


# 从数据文件夹中读取所有图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    result = {}  # 保存图片信息, key为类别, value为图片名称
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取子目录
    is_root_dir = True  # 得到的第一个目录是当前目录, 不用考虑
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']  # 有效的图片类型
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        label_name = dir_name.lower()  # 目录名为类别名称

        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分为训练集、测试集、验证数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {'dir': dir_name, 'training': training_images, 'testing': testing_images, 'validation': validation_images}
    return result

def get_image_path(image_lists, image_dir, label_name, index, category):
    """
    通过类别名称、所属数据集和图片编号获取一张图片的地址
    :param image_lists: 所有图片信息
    :param image_dir: 存放图片的根目录或特征向量文件存放根目录
    :param label_name: 类别名称
    :param index: 图片编号
    :param category: 所属数据集
    :return: 图片地址
    """
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 通过类别名称、所属数据集和图片编号获取经过处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

# 使用加载的训练好的模型处理一张图片, 得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 将当前图片作为输入计算瓶颈张量的值, 这个值就是这张图片新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    # 经过卷积神经网络处理的结果是一个四维数组, 需要压缩成一维数组(特征向量)
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 获取一张图片经过模型处理之后的特征向量, 该函数会先试图寻找已经计算且保存下来的特征向量, 如果找不到就先计算然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取图片对应的特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []  # 图像数据
    ground_truths = []  # 图像类别
    for _ in range(how_many):
        # 随机一个类别和图片编号加入当前训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truths = np.zeros(n_classes, dtype=np.float32)
        ground_truths[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truths)

    return bottlenecks, ground_truths

# 获取全部测试数据, 计算正确率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

def main(_):
    # 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取已经训练好的模型, 并返回数据输入及计算瓶颈层结果对应的张量
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入, 这个输入就是新的图片经过模型前向传播到达瓶颈层时的节点取值(特征提取)
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全连接层来解决新的图像分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARENING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.initializer_all_variables()
        sess.run(init)

        # 训练过程
        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # 在验证数据上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print 'Step {}: Validation accuracy on random sampled {} examples = {:.1f}'.format(i, BATCH, validation_accuracy * 100)

            # 在最后的测试集上测试正确率
            test_bottleneck, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
            test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottleneck, ground_truth_input: test_ground_truth})
            print 'Final test accuracy = {:.1f}'.format(test_accuracy * 100)


if __name__ == '__main__':
    tf.app.run()