#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
身份证文字+数字生成类

"""
import numpy as np
import freetype
import copy
import random
import cv2
import matplotlib.pyplot as plt
#from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image, ImageDraw
from skimage import color, io, data

class put_chinese_text(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        '''
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        '''
        # [32, 256, 3], (0, 0), text, 21, (255, 255, 255)
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender/64.0

        #descender = metrics.descender/64.0
        #height = metrics.height/64.0
        #linegap = height - ascender + descender
        ypos = int(ascender)

        #text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1]+ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        '''
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        '''
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6   # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale)*0x10000, int(0.2*0x10000),\
                                 int(0.0*0x10000), int(1.1*0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)
            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        '''
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        '''
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row*cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]


class gen_id_card(object):
    def __init__(self):
       #self.words = open('AllWords.txt', 'r').read().split(' ')
       self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
       self.char_set = self.number
       #self.char_set = self.words + self.number
       self.len = len(self.char_set)
       
       self.max_size = 4
       self.ft = put_chinese_text('fonts/OCR-B.ttf')
       
    #随机生成字串，长度固定/不固定
    #返回text,及对应的向量
    def random_text(self):
        text = ''
        vecs = np.zeros((self.max_size * self.len))
        size = random.randint(1, self.max_size)  #长度随机 在1-maxlength之间
        #size = self.max_size
        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
        return text,vecs


    
    #根据生成的text，生成image,返回标签和图片元素数据
    def gen_image(self):
        text,vec = self.random_text()
        img = np.zeros([32,256,3])  #图片大小32*256
        # r = random.randint(0, 255)
        # b = random.randint(0, 255)
        # g = random.randint(0, 255)
        # print('r', r)
        # print('b', b)
        # print('g', g)
        color_ = (0,0,255) # 彩色
        pos = (0, 0)
        text_size = 21
        # [32, 256, 3], (0, 0), text, 21, (255, 255, 255)
        image = self.ft.draw_text(img, pos, text, text_size, color_)
        # print('image.shape', image.shape)
        # plt.imshow(image)
        # plt.imshow(image[:,:,2])
        # plt.show()
        # print(image)
        return image[:,:,2],text,vec

    #单字转向量
    def char2vec(self, c):
        vec = np.zeros((self.len))
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec
        
    #向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if(vecs[i] == 1):
                text = text + self.char_set[i % self.len]
        return text


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    print(indices, 'indices')

    values = np.asarray(values, dtype=dtype)
    print(values, 'values')

    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    print(shape, 'shape')

    return indices, values, shape

def get_a_image():
    obj = gen_id_card()
    #(batch_size,256,32)
    inputs = np.zeros([1, 256, 32])
    codes = []

    #生成不定长度的字串
    #image, text, vec = obj.gen_image(True)
    # vec: one-hot类型
    image, text, vec = obj.gen_image()
    img = Image.fromarray(image, mode='RGBA')
    color_ = random_color(0, 128)  # , random.randint(220, 250)
    create_noise_dots(img, color_)
    create_noise_curve(img, color_)
    img = img.convert('L')
    image = img.convert('1')
    image = np.array(image)
    # clearNoise(img)

    #np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
    inputs[0, :] = np.transpose(image.reshape((32, 256)))
    print('text', text)
    codes.append(list(text))
    print(codes, '\ncodes')
    targets = [np.asarray(i) for i in codes]
    # target: text对应的列表的列表
    print(targets, '\ntargers')
    # sparse_targets: 元组
    sparse_targets = sparse_tuple_from(targets)
    print(sparse_targets, '\nsparse_targets')

    seq_len = np.ones(inputs.shape[0]) * 256
    return inputs, sparse_targets, seq_len, image

def get_next_batch(batch_size=2):
    obj = gen_id_card()
    #(batch_size,256,32)
    inputs = np.zeros([batch_size, 256,32])
    codes = []

    for i in range(batch_size):
        #生成不定长度的字串
        #image, text, vec = obj.gen_image(True)
        image, text, vec = obj.gen_image()
        ### 新添
        img = Image.fromarray(image, mode='RGBA')
        color_ = random_color(0, 128)  # , random.randint(220, 250)
        create_noise_dots(img, color_)
        create_noise_curve(img, color_)
        img = img.convert('L')
        image = img.convert('1')
        image = np.array(image)
        ### 截止
        #np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i,:] = np.transpose(image.reshape((32, 256)))
        print('text', text)
        codes.append(list(text))
    print(codes, '\ncodes')
    targets = [np.asarray(i) for i in codes]
    print(targets, '\ntargers')
    #print(targets)
    sparse_targets = sparse_tuple_from(targets)
    print(sparse_targets, '\nsparse_targets')
    #(batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * 256

    return inputs, sparse_targets, seq_len

# 增加噪音
def create_noise_dots(image, color, width=2, number=50):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    while number:
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        # draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
        draw.point((x1, y1), fill=color)
        number -= 1
    return image
def create_noise_curve(image, color):
    w, h = image.size
    x1 = random.randint(0, int(w / 5))
    x2 = random.randint(w - int(w / 5), w)
    y1 = random.randint(int(h / 5), h - int(h / 5))
    y2 = random.randint(y1, h - int(h / 5))
    points = [x1, y1, x2, y2]
    end = random.randint(160, 200)
    start = random.randint(0, 20)
    ImageDraw.Draw(image).arc(points, start, end, fill=color)
    return image

def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


# 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
# 该函数也可以改成RGB判断的,具体看需求如何
def getPixel(image, x, y, G, N):
    L = image.getpixel((x, y))
    if L > G:
        L = True
    else:
        L = False

    nearDots = 0
    if L == (image.getpixel((x - 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x - 1, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x, y + 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y - 1)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y)) > G):
        nearDots += 1
    if L == (image.getpixel((x + 1, y + 1)) > G):
        nearDots += 1

    if nearDots < N:
        return image.getpixel((x, y - 1))
    else:
        return None


def clearNoise(image, G = 0, N = 2, Z = 1):
    draw = ImageDraw.Draw(image)
    for i in range(0, Z):
        for x in range(1, image.size[0] - 1):
            for y in range(1, image.size[1] - 1):
                color = getPixel(image, x, y, G, N)
                if color != None:
                    draw.point((x, y), color)


if __name__ == '__main__':
    genObj = gen_id_card()
    image_data, label, vec = genObj.gen_image()
    # image_data = np.where(image_data==0, 255, random.randint(1,128))
    print('image_data', image_data)

    img = Image.fromarray(image_data, mode='RGBA')

    color_ = random_color(0, 128)#, random.randint(220, 250)
    create_noise_dots(img, color_)
    create_noise_curve(img, color_)
    print(type(img), img.size, img.mode)
    print('np.array(img).shap', np.array(img).shape)
    plt.imshow(img)
    plt.show()

    # img2 = color.gray2rgb(image_data)
    # print(type(img2), img2.shape)
    # # print(img2)
    # plt.imshow(img2)
    # plt.show()

    img = img.convert('L')
    print(type(img), img.size, img.mode, np.array(img).shape)
    plt.imshow(img)
    plt.show()
    # print(img)
    # clearNoise(img, 220, 7, 1)
    # plt.imshow(img)
    # plt.show()

    #整个图像呈现出明显的黑白效果的过程
    img = img.convert('1')
    plt.imshow(img)
    plt.show()
    print('np.array(img).shape', np.array(img).shape)

    clearNoise(img)
    plt.imshow(img)
    plt.show()

    # cv2.imshow('img', np.array(img))
    # print(image_data.shape)
    cv2.imshow('image', image_data)
    cv2.waitKey(0)
    plt.imshow(image_data)
    plt.show()

    # get_next_batch()
    '''
    line = '湖南省邵阳县'
    img = np.zeros([300,300,3])

    color_ = (255,255,255) # Green
    pos = (3, 3)
    text_size = 20

    #ft = put_chinese_text('fonts/msyhbd.ttf')
    ft = put_chinese_text('fonts/huawenxihei.ttf')
    no = put_chinese_text('fonts/OCR-B.ttf')
    image = ft.draw_text(img, pos, line, text_size, color_)
    image1 = no.draw_text(image, (50,50), '1232142153253215', 20, (255,255,255))

    cv2.imshow('ss', image)
    cv2.imshow('image1', image1)
    cv2.waitKey(0)
    '''


