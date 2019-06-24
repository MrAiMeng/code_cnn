"""卷积神经网络实现验证码的识别"""
# 读取csv文件出错，后续有时间再解决
import tensorflow as tf
import os


# 定义命令行参数
tf.app.flags.DEFINE_string('image_path_name','E:\python\数据\code_images',"图片数据所在路径")
tf.app.flags.DEFINE_string('target_path_name','E:\python\\tensortest\\code_name1.csv',
                           "图片对应验证码数据所在路径")
tf.app.flags.DEFINE_string('tfrecords_path_name',
                           'E:\python\\tensortest\\tfrecords\\image_code5.tfrecords',
                           "tfrecords文件所在路径")
tf.app.flags.DEFINE_string('code_name',"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "one-hot编码所需元素")

FLAGS = tf.app.flags.FLAGS


# 数据准备，读取数据生成tfrecords文件

# # 将图片数名字写成scv文件
def get_image_name():
    file_list = os.listdir(FLAGS.image_path_name)
    print(file_list)
    # 构建与图片对应的目标文件
    file_name = [file.split('.')[0] for file in file_list]
    with open('E:\python\\tensortest\\code_name1.csv', 'w', encoding='utf8') as f:
        for file in file_name:
            print(file)
            print(type(file))
            # for srt_name in file:
            f.write(file)# 在excel中打开e会被特殊处理，但通过pycharm打开时不会
            f.write('\n')


# 读取图片数据，返回图片数据批
def get_images():
    # 1.构造文件队列
    # 1.1构造文件列表
    file_list = os.listdir(FLAGS.image_path_name)
    print(file_list)
    # 图片数据获取
    # 1.2文件名与路径组成数据列表
    data_list = [os.path.join(FLAGS.image_path_name, file) for file in file_list]
    print(data_list)
    # 1.3构造文件队列
    data_queue = tf.train.string_input_producer(data_list, shuffle=False)
    # 2.构造阅读器读取数据
    reader = tf.WholeFileReader()
    file_name, image = reader.read(data_queue)
    # 3.数据解码
    image = tf.image.decode_png(image)
    image = tf.reshape(image, [24,72,3])

    # 将图片数据改为uint8格式减少存储空间
    image = tf.cast(image, tf.uint8)

    # 4.数据批处理
    image_batch = tf.train.batch([image], 1000, num_threads=1, capacity=1000, name='image_batch')
    print(image_batch)
    return image_batch


# 读取csv对应验证码图片的验证码字符串，返回验证码字符串批
def get_codes():
    # 验证码数据获取
    # 1.构造文件队列
    data_list = [FLAGS.target_path_name]
    print(data_list)
    # 1.2构造文件队列
    data_queue = tf.train.string_input_producer(data_list, shuffle=False)
    # 2.构造阅读器读取数据
    reader = tf.TextLineReader()
    file_name, code = reader.read(data_queue)
    # 3.数据解码
    code = tf.decode_csv(code, record_defaults=[['None']])# 此处指定数据格式需用中括号嵌套每一个字段也要用一个中括号
    # print(code)
    # 4.数据批处理
    code_batch = tf.train.batch(code, batch_size=1000, num_threads=1, capacity=1000, name='data_batch')
    print(code_batch)
    return code_batch


# 将code转化为one-hot编码(重点）
# 训练前使用，存储Tfrecords文件时不用,也可以将数据one-hot编码后转化成tensor类型进行存储
def code_one_hot(code_value):
    # 1.构建字符索引{'0':'0', '1':'1',...,'10':'A',...}
    num_letter = dict(enumerate(list(FLAGS.code_name)))
    print(num_letter)

    # 2.键值对反转{'0':'0', '1':'1',...,'A':'10',...}
    letter_num = dict(zip(num_letter.values(), num_letter.keys()))
    print(letter_num)

    # 3.one_hot编码
    one_hot_batch = []
    code_value_list = code_value.eval()
    print(code_value_list)
    # 数据流过一次就无法重新使用,应该先将数据取出保存，然后利用取出的数据进行循环
    # 如（错误写法）：此时每次都是重新取一批数据，从1000个数据中取第i个
    # for i in range(1000):
    #    code = code_value.eval()[i].decode(utf8')
    #    for j in code:
    #       ....
    for value in code_value_list:
        code_list = []
        code = value.decode('utf8')
        print(code)
        for j in code:
            # print(j)
            code_list.append(letter_num[j])
        # 编码后数据格式:[11, 27, 31, 14]
        print(code_list)
        one_hot_batch.append(code_list)
    # 将one-hot编码后的数据转化为tensor类型
    # print(one_hot_batch)
    # 1行4列型数据应该写为[4],而不是[1,4]
    one_hot_batch_tensor = tf.constant(one_hot_batch, dtype=tf.int32, shape = [1000, 4], name = 'one_hot_batch_tensor')
    return one_hot_batch_tensor

# 5.将数据写成tfrecords文件
def write_tfrecords(image_batch, one_hot_batch):
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_path_name)
    # 注意批数据的使用
    image_value = image_batch.eval()
    one_hot_value = one_hot_batch.eval()
    print(one_hot_value)
    for i in range(1000):
        print(one_hot_value[i])
        # 将图片内容数据以及编码后标签数据都转化为字符串类型
        image_str = image_value[i].tostring()
        code_str = one_hot_value[i].tostring()

        # value为实际的数据，在使用前要用eval提取数据
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
            'code': tf.train.Feature(bytes_list=tf.train.BytesList(value=[code_str]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    # 每次取1000个数据，取10次
    # for i in range(10):
    #     get_image_name() # 将图片名写成csv文件
    image_batch = get_images()
    code_batch = get_codes()

    with tf.Session() as sess:
        # 1.开启数据读取线程
        # 1.1定义线程协调器
        coord = tf.train.Coordinator()
        # 1.2 开启线程
        thread = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(10):
            # 将code_batch进行编码
            one_hot_batch = code_one_hot(code_batch)
            write_tfrecords(image_batch, one_hot_batch)
            print('第%d组数据'%i)

        # 线程回收
        coord.request_stop()
        coord.join(thread)
