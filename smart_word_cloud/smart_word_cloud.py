# coding=utf-8
# Author: Yewei Liu (Lewis) <liuyeweilewis@gmail.com> <2300012959@stu.pku.edu.cn>
#
# (c) 2025
# License: MIT

from __future__ import division

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
import re
import os
from PIL import Image, ImageDraw, ImageFont

# set relative path
FILE = os.path.dirname(__file__)
nltk_data_path = os.path.join(FILE, '../data')  
nltk.data.path.append(nltk_data_path)
FONT_PATH = os.environ.get('FONT_PATH', os.path.join(FILE, 'DroidSansMono.ttf'))

class MyCounter(Counter):
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return MyCounter({k: self[k] * scalar for k in self})
        return NotImplemented

class Canvas():
    def __init__(self, height, width, mask = None, margin = 2):
        self.height = height
        self.width = width
        self.margin = margin
        self.img_grey = Image.new("L", (width, height))
        self.draw = ImageDraw.Draw(self.img_grey)
        if mask is not None:
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1), axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)
    
    def draw_word(self, word, font, seed = 777):
        hits = 0
        res = []
        box_size = self.draw.textbbox((0, 0), word, font=font, anchor="lt")
        size_h, size_w = box_size[3] + self.margin, box_size[2] + self.margin
        for i in range(1, self.height - size_h):
            for j in range(1, self.width - size_w):
                area = self.integral[i, j] + self.integral[i + size_h, j + size_w] - self.integral[i + size_h, j] - self.integral[i, j + size_w]
                if area == 0:
                    hits += 1
                    res.append((i + 1, j + 1))
        if hits == 0:
            return False
        rng = np.random.default_rng(seed)
        tmp = rng.integers(0, hits, dtype = np.uint32)
        h, w = res[tmp]
        print(h, h + size_h, w, w + size_w)
        self.draw.text((w + self.margin // 2, h + self.margin // 2), word, fill='white', font=font)
        tmp_array = np.zeros((self.height, self.width))
        tmp_array[h: h + size_h, w: w + size_w] = 1
        tmp_array = np.cumsum(np.cumsum(tmp_array.astype(np.uint32), axis=1), axis=0)
        self.integral += tmp_array
        return True
    
    def save(self, path, name):
        self.img_grey.save(path + name + '.png')





class SmartWordCloudGenerator(object):
    def __init__(self, default_stopwords = True, nlp_improvement = True, font_path=None):
        
        # initialize
        self.counter = MyCounter()

        # nlp_improvement
        self.nlp_improvement =  nlp_improvement
        if self.nlp_improvement:
            self.lemmatizer = WordNetLemmatizer()
        
        # stopwords
        self.stopwords = set()
        if default_stopwords:
            self.stopwords = set(stopwords.words('english'))
            if self.nlp_improvement:
                self.stopwords = set([self.lemmatizer.lemmatize(word) for word in self.stopwords])

        # font
        if font_path is None:
            font_path = FONT_PATH
        self.font_path = font_path
    

    def _preprocess(self, text):
        text = text.lower()
        text = re.findall(r'\b[a-zA-Z\'-]+\b', text)
        text = [word for word in text if word not in self.stopwords]
        if self.nlp_improvement:
            text = [self.lemmatizer.lemmatize(word) for word in text]
        return text
        

    def add_text(self, text, frequency_weight=1, focus = None, focusing_radius = 10, focusing_func = lambda x: 1 - x):
        text = self._preprocess(text)
        word_count = MyCounter(text)
        self.counter += word_count * frequency_weight
        if focus is not None:
            if not isinstance(focus, dict):
                raise TypeError("focus should be a dict or None")
            for w in focus:
                word = w
                if self.nlp_improvement:
                    word = self.lemmatizer.lemmatize(word)
                indices = [i for i in range(len(text)) if text[i] == word]
                for indice in indices:
                    for idx in range(max(0, indice - focusing_radius), min(len(text), indice + focusing_radius + 1)):
                        self.counter += {text[idx]: focus[w] * focusing_func(abs(idx - indice) / focusing_radius)}

    def add_stopword(self, word):
        if self.nlp_improvement:
            word = self.lemmatizer.lemmatize(word)
        self.stopwords.add(word)
        self.counter.pop(word, None)


    def add_stopwords(self, words:list):
        for word in words:
            self.add_stopword(word)


    def generate(self, max_font_size = 100, min_font_size = 10, font_size_func = lambda x: x, width = 400, height = 400, max_words = 50, mask = None, margin = 6):

        canvas = Canvas(height, width, mask, margin)

        sorted_items = sorted(self.counter.items(), key=lambda item: item[1], reverse=True)
        self.counter = dict(sorted_items)
        first = next(iter(self.counter.items()))
        maxn = first[1]
        num = 0
        for word in self.counter:
            freq = self.counter[word] / maxn
            font_size = int(round(font_size_func(freq) * max_font_size))
            print(word, font_size)
            if(font_size < min_font_size):
                break
            font = ImageFont.truetype(self.font_path, font_size)
            res = canvas.draw_word(word, font, seed=num)
            print(res)
            if res == False:
                break
            num += 1
            if num >= max_words:
                break
        print(num)
        
        canvas.save("image/", "tmp")


if __name__ == '__main__':
    gen = SmartWordCloudGenerator()
    input_path = 'text/the little prince.txt'
    lines = None
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
    text = "".join(lines) 
    gen.add_text(text)
    print(gen.counter)
    gen.generate()
