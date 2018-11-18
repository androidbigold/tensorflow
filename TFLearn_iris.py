# coding:utf-8
import numpy as np
import tensorflow as  tf
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics

# 导入TFLearn
learn = tf.contrib.learn

# 自定义模型
def my_model(features, target):
    # 将预测目标转换为one-hot编码形式, 共有3个类别, 所以向量长度为3, 三个类别分别表示为(1, 0, 0), (0, 1, 0), (0, 0, 1)
    target = tf.one_hot(target, 3, 1, 0)

    # 定义模型以及其在给定数据上的损失函数, TFLearn通过logistic_regression封装了一个单层全连接神经网络
    logits, loss = learn.models.logistic_regression(features, target)

    # 创建模型的优化器, 并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss,  # 损失函数
        tf.contrib.framework.get_global_step(),  # 获取训练步数并在训练时更新
        optimizer='Adagrad',  # 定义优化器
        learning_rate=0.1  # 定义学习率
    )

    # 返回给定数据上的预测结果、损失值以及优化步骤
    return tf.argmax(logits, 1), loss, train_op

# 加载iris数据集, 并划分训练集和测试集
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# 对自定义模型进行封装
classifier = learn.Estimator(model_fn=my_model)

# 使用封装好的模型和训练数据执行100轮迭代
classifier.fit(x_train, y_train, steps=100)

# 使用训练好的模型进行预测
y_predicted = classifier.predict(x_test)

# 计算模型准确度
score = metrics.accuracy_score(y_test, list(y_predicted))
print "Accuracy: {:.2f}%".format(score * 100)
