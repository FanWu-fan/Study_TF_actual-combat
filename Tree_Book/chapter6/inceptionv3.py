import tensorflow as tf 
slim = tf.contrib.slim

#产生截断的正态分布
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0,stddev)

'''
下面定义函数 inception_v3_arg_scope,用来生成网络中经常用到的函数的默认参数，
比如卷积的激活函数，权重初始化方式，标准化器等。设置L2正则的wtight_decay默认值为0.00004，
标准差stddev默认值为0.1，参数 batch_norm_var_collection默认值为moving_vars.

使用slim.arg_scope,这是一个非常有用的工具，它可以给函数的参数自动赋予某些默认值，例如
with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_regularizer = slim.l2regularizer(weight_decay))
会对[slim.conv2d,slim.fully_connected]这两个函数的参数自动赋值，将参数weights_regularizer的值默认设置为slim.l2regularizer(weight_decay)
使用了 slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置。接下来嵌套一个 slim.arg_scope,对卷积层生成函数
slim.conv2d的几个参数赋予默认值，其权重初始化器weights_initializer设置为 trunc_normal(stddev)，激活函数设置为 ReLU,标准化
器设置为 slim.batch_norm,标准化器的参数设置为前面定义的 batch_norm_params，最后返回定义好的 scope
'''

# @slim.add_arg_scope
# def func1(a=0,b=0,c=0):
#   return a+b+c

# with slim.arg_scope([func1],a=10):
#     with slim.arg_scope([func1],b=20):
#         x = func1(c=30)
#         print(x)

#         x = func1(c=50)
#         print(x)

# def funs(fun,factor=20):
#     x = fun()
#     print(factor*x)

# @funs #等价于 funs(add(),factor=20)
# def add(a=10,b=20):
#     return (a+b)


def inception_v3_arg_scope(weight_decay=0.00004,
                            stddev = 0.1,
                            batch_norm_var_collection = 'moving_vars'):
    batch_norm_params ={
        'decay':0.997,
        'epslion':0.001,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
        'variables_collections':
        {
            'beta':None,
            'gamma':None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance':[batch_norm_var_collection],
        }
    }
    
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
        weight_decay=slim.l2regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params) as sc:
            return sc

def inception_v3_base(inputs,scope=None):

    end_points = {}
    with tf.variable_scope(name_or_scope = scope,default_name = 'InceptionV3',values=[input]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride = 1,padding='VALID'):
            #299*299*32
            net = slim.conv2d(inputs,32,[3,3],stride = 2, scope = 'Conv2d_1a_3x3') 
            #149*149*32
            net = slim.conv2d(net,32,[3,3],scope='Conv_2a_3x3') 
            #147*147*32
            net = slim.conv2d(net,64,[3,3],padding='SAME',scope ='Conv2d_2b_3x3')
            #147*147*64
            net = slim.max_pool2d(net,[3,3],stridet=2,scope='MAxPool_3a_3x3')
            #73*73*64
            net = slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1x1')
            #73*73*80
            net = slim.conv2d(net,192,[3,3],scope = 'Conv2d_4a_3x3')
            #71*71*192
            net = slim.max_pool2d(net,[3,3],stride=2,scpoe='MaxPool_5a_3x3')
            # 35*35*192

        '''
        接下来就是3个 Inception blocks,这三个各自分别由多个 Inception Module,
        第一个Inception blocks包含了3个结构类似的Inception Module，第一个名为Mixed_5b,
        '''
        # 第一个模块组的第一个模块——Mixed_5b
        # 17x17x768
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride = 1,padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, num_outputs=48, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    branch_1 = slim.conv2d(branch_1,64,[5,5],scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope="Conv2d_0a_1x1")
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope="Conv2d_0b_3x3")
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope="Conv2d_0c_3x3")
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,32,[1,1],scope="Conv2d_0b_1x1")
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            #第一个inception模块组的第2个 Inception module--Mixed_5c
            with tf.variable_scope("Mixed_5c"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=48, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=64, kernel_size=[5, 5], scope="Conv2d_0c_5x5")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0c_3x3")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
    
                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)
    
        # 第一个inception模块组的第3个 Inception module--Mixed_5d
            with tf.variable_scope("Mixed_5d"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=48, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=64, kernel_size=[5, 5], scope="Conv2d_0c_5x5")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_2 = slim.conv2d(batch_2, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0c_3x3")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
    
                net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)
            # 定义第二个Inception模块组,第1个Inception模块
            with tf.variable_scope("Mixed_6a"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net, num_outputs=384, kernel_size=[3,3],
                                        stride=2, padding="VALID",scope="Conv2d_1a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=64, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=96, kernel_size=[3, 3], scope="Conv2d_0b_3x3")
                    batch_1 = slim.conv2d(batch_1, num_outputs=96, kernel_size=[3, 3],
                                        stride=2, padding="VALID",scope="Conv2d_1a_1x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding="VALID",
                                            scope="MaxPool_1a_3x3")
                net = tf.concat([batch_0, batch_1, batch_2], 3)
            
            # 定义第二个Inception模块组,第一个Inception模块
            with tf.variable_scope("Mixed_6b"):
                with tf.variable_scope("Branch_0"):
                    batch_0 = slim.conv2d(net,num_outputs=192,kernel_size=[1,1],scope="Conv2d_0a_1x1")
                with tf.variable_scope("Branch_1"):
                    batch_1 = slim.conv2d(net, num_outputs=128, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_1 = slim.conv2d(batch_1, num_outputs=128, kernel_size=[1,7], scope="Conv2d_0b_1x7")
                    batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[7, 1],scope="Conv2d_0c_7x1")
                with tf.variable_scope("Branch_2"):
                    batch_2 = slim.conv2d(net, num_outputs=128, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=128, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=128, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                    batch_2 = slim.conv2d(batch_2, num_outputs=128, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                    batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
                with tf.variable_scope("Branch_3"):
                    batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                    batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
    
                net = tf.concat([batch_0, batch_1, batch_2,batch_3], 3)
            
            # 定义第二个Inception模块组,第三个Inception模块
        with tf.variable_scope("Mixed_6c"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(net,num_outputs=192,kernel_size=[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[1,7], scope="Conv2d_0b_1x7")
                batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[7, 1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
 
            net = tf.concat([batch_0, batch_1, batch_2,batch_3], 3)
 
        # 定义第二个Inception模块组,第四个Inception模块
        with tf.variable_scope("Mixed_6d"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(net,num_outputs=192,kernel_size=[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[1,7], scope="Conv2d_0b_1x7")
                batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[7, 1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
 
            net = tf.concat([batch_0, batch_1, batch_2,batch_3], 3)
 
        # 定义第二个Inception模块组,第五个Inception模块
        with tf.variable_scope("Mixed_6e"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(net,num_outputs=192,kernel_size=[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[1,7], scope="Conv2d_0b_1x7")
                batch_1 = slim.conv2d(batch_1, num_outputs=160, kernel_size=[7, 1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.conv2d(net, num_outputs=160, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0b_7x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[1, 7], scope="Conv2d_0c_1x7")
                batch_2 = slim.conv2d(batch_2, num_outputs=160, kernel_size=[7, 1], scope="Conv2d_0d_7x1")
                batch_2 = slim.conv2d(batch_2, num_outputs=192, kernel_size=[1, 7], scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                batch_3 = slim.avg_pool2d(net, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                batch_3 = slim.conv2d(batch_3, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0b_1x1")
 
            net = tf.concat([batch_0, batch_1, batch_2,batch_3], 3)
        end_points["Mixed_6e"] = net    #第二个模块组的最后一个Inception模块，将Mixed_6e存储于end_points中

        # 定义第三个Inception模块组,第一个Inception模块
        with tf.variable_scope("Mixed_7a"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_0 = slim.conv2d(net, num_outputs=320, kernel_size=[3, 3],stride=2,
                                      padding="VALID",scope="Conv2d_1a_3x3")
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[1,7], scope="Conv2d_0b_1x7")
                batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[7, 1],scope="Conv2d_0c_7x1")
                batch_1 = slim.conv2d(batch_1, num_outputs=192, kernel_size=[3, 3], stride=2,
                                      padding="VALID",scope="Conv2d_1a_3x3")
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID",
                                          scope="MaxPool_1a_3x3")
 
            net = tf.concat([batch_0, batch_1, batch_2], 3)
        
        # 定义第三个Inception模块组,第二个Inception模块
        with tf.variable_scope("Mixed_7b"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(net, num_outputs=320, kernel_size=[1, 1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(net, num_outputs=384, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_1 = tf.concat([
                    slim.conv2d(batch_1,num_outputs=384,kernel_size=[1,3],scope="Conv2d_0b_1x3"),
                    slim.conv2d(batch_1,num_outputs=384,kernel_size=[3,1],scope="Conv2d_0b_3x1")],axis=3)
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.conv2d(net,num_outputs=448,kernel_size=[1,1],scope="Conv2d_0a_1x1")
                batch_2 = slim.conv2d(batch_2,num_outputs=384,kernel_size=[3,3],scope="Conv2d_0b_3x3")
                batch_2 = tf.concat([
                    slim.conv2d(batch_2,num_outputs=384,kernel_size=[1,3],scope="Conv2d_0c_1x3"),
                    slim.conv2d(batch_2,num_outputs=384,kernel_size=[3,1],scope="Conv2d_0d_3x1")],axis=3)
            with tf.variable_scope("Branch_3"):
                batch_3 = slim.avg_pool2d(net,kernel_size=[3,3],scope="AvgPool_0a_3x3")
                batch_3 = slim.conv2d(batch_3,num_outputs=192,kernel_size=[1,1],scope="Conv2d_0b_1x1")
 
        net = tf.concat([batch_0, batch_1, batch_2,batch_3], 3)
 
        # 定义第三个Inception模块组,第三个Inception模块
        with tf.variable_scope("Mixed_7c"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(net, num_outputs=320, kernel_size=[1, 1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(net, num_outputs=384, kernel_size=[1, 1], scope="Conv2d_0a_1x1")
                batch_1 = tf.concat([
                    slim.conv2d(batch_1,num_outputs=384,kernel_size=[1,3],scope="Conv2d_0b_1x3"),
                    slim.conv2d(batch_1,num_outputs=384,kernel_size=[3,1],scope="Conv2d_0b_3x1")],axis=3)
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.conv2d(net,num_outputs=448,kernel_size=[1,1],scope="Conv2d_0a_1x1")
                batch_2 = slim.conv2d(batch_2,num_outputs=384,kernel_size=[3,3],scope="Conv2d_0b_3x3")
                batch_2 = tf.concat([
                    slim.conv2d(batch_2,num_outputs=384,kernel_size=[1,3],scope="Conv2d_0c_1x3"),
                    slim.conv2d(batch_2,num_outputs=384,kernel_size=[3,1],scope="Conv2d_0d_3x1")],axis=3)
            with tf.variable_scope("Branch_3"):
                batch_3 = slim.avg_pool2d(net,kernel_size=[3,3],scope="AvgPool_0a_3x3")
                batch_3 = slim.conv2d(batch_3,num_outputs=192,kernel_size=[1,1],scope="Conv2d_0b_1x1")
 
        net = tf.concat([batch_0, batch_1, batch_2,batch_3], 3)
 
    return net,end_points

def inception_v3(inputs,num_classes=1000,is_training=True,droupot_keep_prob = 0.8,
            prediction_fn = slim.softmax,spatial_squeeze = True,reuse = None,scope="InceptionV3"):
    '''
    InceptionV3整个网络的构建
    param :
    inputs -- 输入tensor
    num_classes -- 最后分类数目
    is_training -- 是否是训练过程
    droupot_keep_prob -- dropout保留节点比例
    prediction_fn -- 最后分类函数，默认为softmax
    patial_squeeze -- 是否对输出去除维度为1的维度
    reuse -- 是否对网络和Variable重复使用
    scope -- 函数默认参数环境
    return:
    logits -- 最后输出结果
    end_points -- 包含辅助节点的重要节点字典表
    '''
    with tf.variable_scope(scope,"InceptionV3",[inputs,num_classes],
                        reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm,slim.dropout],
                        is_training = is_training):
        net,end_points = inception_v3_base(inputs,scope=scope)     #前面定义的整个卷积网络部分

        #辅助分类节点部分
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride = 1,padding = "SAME"):
            #通过end_points取到Mixed_6e
            aux_logits = end_points["Mixed_6e"]
            with tf.variable_scope("AuxLogits"):
                aux_logits = slim.avg_pool2d(aux_logits,kernel_size=[5,5],stride=3,
                                                padding="VALID",scope="Avgpool_1a_5x5")
                aux_logits = slim.conv2d(aux_logits,num_outputs=128,kernel_size=[1,1],scope="Conv2d_1b_1x1")
                aux_logits = slim.conv2d(aux_logits,num_outputs=768,kernel_size=[5,5],
                                            weights_initializer=trunc_normal(0.01),padding="VALID",
                                            scope="Conv2d_2a_5x5")
                aux_logits = slim.conv2d(aux_logits,num_outputs=num_classes,kernel_size=[1,1],
                                            activation_fn=None,normalizer_fn=None,
                                            weights_initializer=trunc_normal(0.001),scope="Conv2d_1b_1x1")
                #消除tensor中前两个维度为1的维度
                if spatial_squeeze:
                    aux_logits = tf.squeeze(aux_logits,axis=[1,2],name="SpatialSqueeze")

                end_points["AuxLogits"] = aux_logits    #将辅助节点分类的输出aux_logits存到end_points中

            #正常分类预测
            with tf.variable_scope("Logits"):
                net = slim.avg_pool2d(net,kernel_size=[8,8],padding="VALID",
                                        scope="Avgpool_1a_8x8")
                net = slim.dropout(net,keep_prob=droupot_keep_prob,scope="Dropout_1b")
                end_points["Logits"] = net

                logits = slim.conv2d(net,num_outputs=num_classes,kernel_size=[1,1],activation_fn=None,
                                        normalizer_fn=None,scope="Conv2d_1c_1x1")
                if spatial_squeeze:
                    logits = tf.squeeze(logits,axis=[1,2],name="SpatialSqueeze")
            
            end_points["Logits"] = logits
            end_points["Predictions"] = prediction_fn(logits,scope="Predictions")

    return logits,end_points





