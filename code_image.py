# 安装pillow包，从PIL（Python Image Library）导入以下模块
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random


def getRandomColor():
    '''获取一个随机颜色(r,g,b)格式的'''
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)
    return (c1, c2, c3)


def getRandomStr():
    '''获取一个随机字符串，每个字符的颜色也是随机的'''
    random_num = str(random.randint(0, 9))
    # chr() 用一个范围在 range（256）内的（就是0～255）整数作参数，返回一个对应的字符。返回值是当前整数对应的 ASCII 字符
    # random_low_alpha = chr(random.randint(97, 122))
    random_upper_alpha = chr(random.randint(65, 90))
    random_char = random.choice([random_num, random_upper_alpha])
    return random_char

if __name__ == '__main__':
    # 获取一个font字体对象参数是ttf的字体文件的目录，以及字体的大小
    font = ImageFont.truetype("C:\Windows\Fonts\VINERITC.TTF", size=260)
    for j in range(10000):
        name = ''
        # 获取一个Image对象，参数分别是RGB模式。宽150，高30，随机颜色
        image = Image.new('RGB', (1500, 400))
        # 获取一个画笔对象，将图片对象传过去
        draw = ImageDraw.Draw(image)

        # 循环4次，获取4个随机字符串
        for i in range(4):
            random_char = getRandomStr()
            name += random_char
            # 在图片上一次写入得到的随机字符串,参数是：定位，字符串，颜色，字体
            draw.text((100 + i * 300, 0), random_char, getRandomColor(), font=font)
        print(name)
        # 保存到硬盘，名为test.png格式为png的图片
        image.save(open('D:\数据\image\{}.png'.format(name), 'wb'), 'png')