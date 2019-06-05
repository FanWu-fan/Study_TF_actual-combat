import tensorflow as tf 

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
# tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
# tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
'''
axis=0     代表在第0个维度拼接

axis=1     代表在第1个维度拼接 

对于一个二维矩阵，第0个维度代表最外层方括号所框下的子集，
第1个维度代表内部方括号所框下的子集。维度越高，括号越小
tf.concat()拼接的张量只会改变一个维度，其他维度是保存不变的。
比如两个shape为[2,3]的矩阵拼接，要么通过axis=0变成[4,3]，
要么通过axis=1变成[2,6]。改变的维度索引对应axis的值。

对于axis等于负数的情况
axis=-1表示倒数第一个维度，对于三维矩阵拼接来说，
axis=-1等价于axis=2。同理，axis=-2代表倒数第二个维度，
对于三维矩阵拼接来说，axis=-2等价于axis=1。

一般在维度非常高的情况下，我们想在最'高'的维度进行拼接，
一般就直接用countdown机制，直接axis=-1就搞定了。
'''

'''
tf中有两对方法比较容易混淆，涉及的是shape问题，在此做一些区分。
首先说明tf中tensor有两种shape，
分别为static (inferred) shape和dynamic (true) shape，
其中static shape用于构建图，由创建这个tensor的op推断（inferred）得来，故又称inferred shape。
如果该tensor的static shape未定义，则可用tf.shape()来获得其dynamic shape。
'''
x = tf.placeholder(tf.int32, shape=[4])
print (x.get_shape().as_list())
# ==> '(4,)'

'''
get_shape()返回了x的静态类型，4代指x是一个长度为4的向量。需要注意，get_shape()不需要放在session中即可运行。
与get_shape()不同，tf.shape()的示例代码如下：
'''
y, idx = tf.unique(x)
print(y.get_shape)
with tf.Session() as sess:
    print(sess.run(y,feed_dict={x:[1,1,1,2]}).shape)
    print(sess.run(y,feed_dict={x:[1,1,2,3]}).shape)
    

