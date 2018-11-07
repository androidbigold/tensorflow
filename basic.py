# coding:utf-8
import tensorflow as tf


matrix1 = tf.constant([[3., 3.]])  # 1X2矩阵
matrix2 = tf.constant([[2.], [2.]])  # 2X1矩阵

product = tf.matmul(matrix1, matrix2)  # 矩阵乘法

sess = tf.Session()

result = sess.run(product)
print result

sess.close()
