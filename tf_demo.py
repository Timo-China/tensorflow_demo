import tensorflow as tf

with tf.name_scope('graph') as scope:
    matrix1 = tf.constant([[3.,3.]], name = 'matrix1')
    matrix2 = tf.constant([[2.],[2.]], name = 'matrix2')
    product = tf.matmul(matrix1, matrix2, name='product')

session = tf.Session()

writer = tf.summary.FileWriter('logs/', session.graph);

init = tf.global_variables_initializer();

session.run(init)