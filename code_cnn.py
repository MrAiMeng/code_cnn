import tensorflow as tf
import  os


# 路径参数
tf.app.flags.DEFINE_string('tfrecords_file_path','E:\python\\tensortest\\tfrecords\\image_code5.tfrecords',
                           "tfrecords文件所在路径")
tf.app.flags.DEFINE_string('events_file_path','E:\python\\tensortest\events', "events文件所在路径")
tf.app.flags.DEFINE_string('check_point_file_path','E:\python\\tensortest\check_point\checkpoint',
                           "check_point文件所在路径")
tf.app.flags.DEFINE_string('check_point_file','E:\python\\tensortest\check_point\cnn_code_model1',
                           "check_point模型文件")

# 过滤器参数
tf.app.flags.DEFINE_integer('filter_height', 3, 'filter_height')
tf.app.flags.DEFINE_integer('filter_width', 3, 'filter_width')
tf.app.flags.DEFINE_integer('filter_channels', 3, 'filter_channels')
# tf.app.flags.DEFINE_integer('out_channels', 32, 'out_channels')
# 图片参数
tf.app.flags.DEFINE_integer('image_height', 24, 'image_height')
tf.app.flags.DEFINE_integer('image_width', 72, 'image_width')
# one_hot参数
tf.app.flags.DEFINE_integer('one_hot_num', 36, 'one_hot_num')
tf.app.flags.DEFINE_integer('code_shape', 4, 'code_shape')
# 池化参数
# tf.app.flags.DEFINE_integer('pool_image_height', 12, 'pool_image_height')
# tf.app.flags.DEFINE_integer('pool_image_width', 36, 'pool_image_width')

FLAGS = tf.app.flags.FLAGS

def read_tfrecords():
    # 1.构造文件队列
    file_list = [FLAGS.tfrecords_file_path]
    tfrecords_queue = tf.train.string_input_producer(file_list, shuffle=False)
    # 2.定义文件阅读器，阅读文件
    reader = tf.TFRecordReader()
    file_name, content = reader.read(tfrecords_queue)
    # 3.解析tfrecords文件
    tfrecords_content_dict = tf.parse_single_example(content, features={
        'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'code': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }, name='read_trecords')
    # 4.string类型tfrecords文件续解码
    image = tf.decode_raw(tfrecords_content_dict['image'], tf.uint8)  # 此处数据类型要与写入时一致
    code = tf.decode_raw(tfrecords_content_dict['code'], tf.int32)
    # 图片形状[500, 72, 24, 3]
    image_reshape = tf.reshape(image, [FLAGS.image_height, FLAGS.image_width, FLAGS.filter_channels])
    # 验证码形状[4]
    code_reshape = tf.reshape(code, [FLAGS.code_shape])
    # 5.文件批处理
    image_batch = tf.train.batch([image_reshape], batch_size=500, num_threads=1, capacity=500, name='image_batch')
    code_batch = tf.train.batch([code_reshape], batch_size=500, num_threads=1, capacity=500, name='code_batch')
    print(image_batch, code_batch)
    return image_batch, code_batch

# 创建权重变量
def weight_variable(shape,name):
    weight_value = tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    weight = tf.Variable(initial_value=weight_value, trainable=True, name=name)
    return weight

def bias_variable(shape, name):
    bias = tf.Variable(initial_value=tf.constant(0.0, shape=shape), trainable=True, name=name)
    return bias

def cnn(image_batch, code_batch):
    # 1.卷积过滤器
    # 1.1 创建随机过滤器值(张量or变量)
    filter_shape1 = [FLAGS.filter_height, FLAGS.filter_width, 3, 32]
    convo_filter1 = weight_variable(filter_shape1, 'convo_filter1')
    convo_bias1 = bias_variable([32], 'convo_bias1')

    # 1.2利用过滤器进行卷积计算[500, 24, 72, 3] ->[500, 24, 72, 32]
    cnn_feature1 = tf.nn.conv2d(tf.cast(image_batch, tf.float32), convo_filter1, strides=[1, 1, 1, 1], padding='SAME',
                               name='cnn') + convo_bias1

    # 2.激活函数[500, 24, 72, 32]->[500, 24, 72, 32]
    relu_value1 = tf.nn.relu(cnn_feature1, name='relu')

    # 3.池化层[500, 24, 72, 32]->[500, 12, 36, 32]
    pool_value1 = tf.nn.max_pool(relu_value1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                name='pool_value')

    # # 4.全连接层(一层卷积时的全连接）
    # # 4.1 全连接[500, 12, 36, 32]->[500, 12*36*32]
    # # [500, 12*36*32] * [12*36*32, 4*36] + [500, 4*36]
    # full_value = tf.reshape(pool_value, [-1, 12*36*32])
    #
    # full_shape = [12*36*32, 4*36]
    # full_parameter = weight_variable(full_shape)
    # full_bias = bias_variable([4*36])

    # 二层卷积
    # 1.卷积过滤器
    # 1.1 创建随机过滤器值(变量)
    filter_shape2 = [FLAGS.filter_height, FLAGS.filter_width, 32, 64]
    convo_filter2 = weight_variable(filter_shape2, 'convo_filter2')
    convo_bias2 = bias_variable([64], 'convo_bias2')

    # 1.2利用过滤器进行卷积计算[500, 12, 36, 32] ->[500, 12, 36, 64]
    cnn_feature2 = tf.nn.conv2d(pool_value1, convo_filter2, strides=[1, 1, 1, 1], padding='SAME',
                               name='cnn') + convo_bias2

    # 2.激活函数[500, 12, 36, 64]->[500, 12, 36, 64]
    relu_value2 = tf.nn.relu(cnn_feature2, name='relu')

    # 3.池化层[500, 12, 36, 64]->[500, 6, 18, 64]
    pool_value2 = tf.nn.max_pool(relu_value2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                name='pool_value')

    # 4.全连接层(二层卷积时的全连接）
    # 4.1 全连接[500, 6, 18, 64]->[500, 6*18*64]
    # [500, 6*18*64] * [6*18*64, 4*36] + [500, 4*36]
    full_value = tf.reshape(pool_value2, [-1, 6*18*64])
    full_shape = [6*18*64, 4*36]
    full_parameter = weight_variable(full_shape, 'full_parameter')
    full_bias = bias_variable([4*36], 'full_bias')

    predict_value = tf.matmul(full_value, full_parameter, name='predict_value') + full_bias


    # 首先将存入的code进行one_hot编码,[500, 4, 36]
    true_code = tf.one_hot(code_batch, depth=36, axis=2, on_value=1.0)
    # 4.2 SoftMax计算，交叉熵计算,二维数据(真实值[500, 4, 36]->[500, 4*36])
    soft_code = tf.reshape(true_code, [500, FLAGS.code_shape * FLAGS.one_hot_num])

    loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=soft_code, logits=predict_value, name='softmax')
    # 4.3 损失值列表平均值计算
    loss_mean = tf.reduce_mean(loss_list)
    # 4.4 损失下降api
    lose = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = lose.minimize(loss_mean)

    # 4.5 准确性计算(真实值与预测值比较）,三维数据的比较(预测值[500, 4*36]->[500, 4, 36])
    equal_list = tf.equal(tf.argmax(true_code, 2), tf.argmax(tf.reshape(predict_value,[500, FLAGS.code_shape,
                                                                                       FLAGS.one_hot_num]), 2))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 变量初始化
    init_op = tf.global_variables_initializer()

    # 可视化，写成events文件
    # 收集单变量
    tf.summary.scalar(name='accuracy', tensor=accuracy)
    tf.summary.scalar(name='accuracy', tensor=loss_mean)
    # 收集多维变量
    tf.summary.histogram(name='parameter', values=full_parameter)
    tf.summary.histogram(name='bias', values=full_bias)
    merge = tf.summary.merge_all()

    # 保存模型(var_list为要保存的变量）
    model = tf.train.Saver(var_list=[convo_filter1, convo_bias1, convo_filter2, convo_bias2, full_parameter, full_bias],
                           max_to_keep=5)

    with tf.Session() as sess:
        # 模型加载(先判断模型加载器所在文件夹是否存在模型文件，注：判断是否存在checkpoint文件，而不是判断是否存在某个模型名）
        if os.path.exists(FLAGS.check_point_file_path):
            print('+'*20)
            model.restore(sess, FLAGS.check_point_file)
        else:
            print('*'*20)
            # 初始化变量
            sess.run(init_op)
        # 储存events文件
        events_writer = tf.summary.FileWriter(FLAGS.events_file_path, graph=sess.graph)
        # 1.开启数据读取线程
        # 1.1定义线程协调器
        coord = tf.train.Coordinator()
        # 1.2 开启线程
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        # print(sess.run(code_batch))
        for i in range(5000):
            # 要运行训练op
            sess.run(train_op)
            print('第%d次学习，损失值为%f，准确率为%f'%(i, loss_mean.eval(), accuracy.eval()))
            summary = sess.run(merge)
            events_writer.add_summary(summary, i)
            # # 关闭写入器
            # events_writer.close()
            if (i+1)%200 == 0:
                print('-'*20)
                # 储存模型
                model.save(sess, FLAGS.check_point_file)
        # 线程回收
        coord.request_stop()
        coord.join(thread)


if __name__ == '__main__':
    # 读取tfrecords数据
    image_batch, code_batch = read_tfrecords()

    # 进行卷积神经网网络训练
    cnn(image_batch, code_batch)
