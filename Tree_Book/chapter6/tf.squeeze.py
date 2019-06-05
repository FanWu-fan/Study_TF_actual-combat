import tensorflow as tf 
'''
从张量形状中移除大小为1的维度。

给定一个张量 input，该操作返回一个与已经移除的所有大小为1的维度具有相同类型的张量。
如果您不想删除所有大小为1的维度，则可以通过指定 axis 来删除特定的大小为1的维度。
'''
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t))  # [2, 3]

# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]