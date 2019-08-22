'''
车牌框的识别 剪切保存
'''
# 使用HyperLPR已经训练好了的分类器
import os

import cv2
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from pip._vendor.distlib._backport import shutil


def find_car_num_brod():
    watch_cascade = cv2.CascadeClassifier('./cascade.xml')
    image = cv2.imread('./car_image/su.jpg')
    cv2.imshow('image', image)
    print('111111111111')
    resize_h = 1000
    height = image.shape[0]
    scale = image.shape[1] / float(height)
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(image_gray, 1.2, minNeighbors=4, minSize=(36, 9), maxSize=(106 * 40, 59 * 40))

    print('检测到车牌数', len(watches))
    if len(watches) == 0:
        return False
    for (x, y, w, h) in watches:
        print(x, y, w, h)
        # 不太明白
        cv2.rectangle(image, (x - h, y), (x + w, y + h), (0, 0, 255), 1)
        cut_img = image[y + 5 : y - 5 + h, x + 8 : x - 8 + w]
        cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)
        cv2.imShow('rectangle', cut_gray)
        cv2.waitKey(0)

        cv2.imwrite('./num_for_car.jpg', cut_gray)
        im = Image.open('./num_for_car.jpg')
        size = 720, 180
        mmm = im.resize(size, Image.ANTIALIAS)
        mmm.save('./num_for_car.jpg', 'JPEG', quality=95)
        #break
    return True

# 分割图像
def find_end(start, white, black, arg, white_max, black_max, width):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m] if arg else white[m] > (0.95 * black_max if arg else 0.95 * white_max)):
            end = m
            break
    return end

'''
剪切后车牌的字符单个拆分保存处理
'''
def cut_car_num_for_chart():
    # 1.读取图像,并把图像转换为灰度图像并显示
    img = cv2.imread('./num_for_car.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imShow('gray', img_gray)

    # 2.将灰度图像二值化,设定阈值为100,转化后白底黑字--->目标黑底白字
    img_thre = img_gray
    # 灰点 白点 加粗
    # cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV, img_thre)
    # 二值化处理 自适应阈值 效果不理想
    # th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 高斯除燥 二值化处理
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('threshold', th3)
    cv2.imwrite()

    # src = cv2.imread("D:\PyCharm\Test213\py_car_num_tensor\wb_img.jpg")
    # height, width, channels = src.shape
    # print("width:%s,height:%s,channels:%s" % (width, height, channels))
    # for row in range(height):
    #     for list in range(width):
    #         for c in range(channels):
    #             pv = src[row, list, c]
    #             src[row, list, c] = 255 - pv
    # cv2.imshow("AfterDeal", src)
    # cv2.waitKey(0)
    #
    # # 3、保存黑白图片
    # cv2.imwrite('D:\PyCharm\Test213\py_car_num_tensor\wb_img.jpg', src)
    # img = cv2.imread("D:\PyCharm\Test213\py_car_num_tensor\wb_img.jpg")  # 读取图片
    # src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    # src_img = src_gray

    # 4.分割字符
    # 记录每一列的白色,黑色像素综合
    white = []
    black = []
    height = th3.shape[0]
    width = th3.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0
        t = 0
        for j in range(height):
            if th3[j][i] == 255:
                s += 1
            if th3[j][i] == 0:
                t += 1;
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        print('blackmax ---> ' + str(black_max) + '---whitemax ---> ' + str(white_max))
        # False 代表白底黑字;True代表黑底白字
        arg = False
        if black_max > white_max:
            arg = True

        n = 1
        start = 1
        end = 2
        temp = 1
        while n < width - 2:
            n += 1
            if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
                # 上面这些判断用来辨别是白底黑字还是黑底白字
                # 0.05这个参数请多调整,对应上面的0.95
                start = n
                end = find_end(start, white, black, arg, white_max, black_max, width)
                n = end
                # 车牌框检测分割 二值化处理后 可以看到明显的左右边框  毕竟用的是网络开放资源 所以车牌框定位角度真的不准，
                # 所以我在这里截取单个字符时做处理，就当亡羊补牢吧
                # 思路就是从左开始检测匹配字符，若宽度（end - start）小与20则认为是左侧白条 pass掉  继续向右识别，否则说明是
                # 省份简称，剪切，压缩 保存，还有一个当后五位有数字 1 时，他的宽度也是很窄的，所以就直接认为是数字 1 不需要再
                # 做预测了（不然很窄的 1 截切  压缩后宽度是被拉伸的），
                # shutil.copy()函数是当检测到这个所谓的 1 时，从样本库中拷贝一张 1 的图片给当前temp下标下的字符
                # 车牌左边白条移除
                if end - start > 5:
                    print('end - start' + str(end - start))
                    if temp == 1 and end - start < 20:
                        pass
                    elif temp > 3 and end - start < 20:
                        # 认为这个字符是数字1 copy 一个 32*40的 1 作为temp.bmp
                        shutil.copy(
                            # 111.bmp 是一张 1 的样本图片
                            os.path.join('./tf_car_license_dataset/train_images/training-set/1/', '111.bmp'),
                            os.path.join('./img_cut/', str(temp) + '.bmp')
                        )
                        pass
                    else:
                        cj = th3[1:height, start:end]
                        cv2.imwrite('./img_cut_not_3240/' + str(temp) + '.jpg', cj)
                        im = Image.open('./img_cut_not_3240/' + str(temp) + '.jpg')
                        size = 32, 40
                        mmm = im.resize(size, Image.ANTIALIAS)
                        mmm.save('./img_cut/' + str(temp) + '.bmp', quality=95)
                        cv2.imshow('裁剪后:', mmm)
                        # cv2.imwrite('./py_car_num_tensor/img_cut/' + str(temp) + '.bmp', cj)
                        temp = temp + 1
                        # cv2.waitKey(0)

'''
车牌号码 省份检测: 粤 [粤G.SB250]
'''
SIZE = 1280
WIDTH = 32
HEIGHT = 40
# NUM_CLASSES = 7
PROVINCES = {'京', '闽', '粤', '苏', '沪', '浙', '豫'}
nProvinceIndex = 0
time_begin = time.time()


def conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b_conv1)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


def province_test():
    province_graph = tf.Graph()
    with province_graph.as_default():
        with tf.Session(graph=province_graph) as sess_p:
            # 定义输入节点,对应于图片像素值矩阵集合和图片标签(即所代表的数字)
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver_p = tf.train.import_meta_graph('./train-saver/province/teacher/model.ckpt.meta')
            model_file = tf.train.latest_checkpoint('./train-saver/province/teacher')
            saver_p.restore(sess_p, model_file)

            # 第一层卷积
            W_conv1 = sess_p.graph.get_tensor_by_name('W_conv1:0')
            b_conv1 = sess_p.graph.get_tensor_by_name('b_conv1:0')
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')




def province_letter_test():
    pass


def last_5_num_test():
    pass


if __name__ == '__main__':
    if find_car_num_brod():
        cut_car_num_for_chart()
        first = province_test()
        sencond = province_letter_test()
        last = last_5_num_test()
        print(first, sencond, last)