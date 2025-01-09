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
import re
import os
from PIL import Image, ImageDraw, ImageFont

# set relative path
FILE = os.path.dirname(__file__)
nltk_data_path = os.path.join(FILE, 'data')  
nltk.data.path.append(nltk_data_path)
FONT_PATH = os.environ.get('FONT_PATH', os.path.join(FILE, 'DroidSansMono.ttf'))

class MyCounter(Counter):
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return MyCounter({k: self[k] * scalar for k in self})
        return NotImplemented

class Canvas():
    def __init__(self, height, width, mask = None, margin = 2, bounding_width = None, background = None):
        self.height = height
        self.width = width
        self.margin = margin
        if background is None:
            self.img = Image.new("RGB", (width, height))
        else:
            self.img = background
        self.draw = ImageDraw.Draw(self.img)
        self.bounding_width = bounding_width
        if mask is not None:
            if mask.shape != (height, width):
                print(f"Mask shape {mask.shape} doesn't match the picture shape ({height}, {width}).")
                raise NotImplementedError
            self.integral = np.cumsum(np.cumsum(mask, axis=1), axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)
    
    def draw_word(self, word, font, seed = 777, color_mode = 'contrast', color = None):
        hits = 0
        res = []
        box_size = self.draw.textbbox((0, 0), word, font=font, anchor="lt")
        size_h, size_w = box_size[3] - box_size[1] + self.margin, box_size[2] - box_size[0] + self.margin
        for i in range(1, self.height - size_h - 2):
            for j in range(1, self.width - size_w - 2):
                area = self.integral[i - 1, j - 1] + self.integral[i + size_h, j + size_w] - self.integral[i + size_h, j - 1] - self.integral[i - 1, j + size_w]
                if area == 0:
                    hits += 1
                    res.append((i, j))
        if hits == 0:
            return False
        rng = np.random.default_rng(seed)
        tmp = rng.integers(0, hits, dtype = np.uint32)
        h, w = res[tmp]
        if color_mode == 'random':
            rng = np.random.default_rng(seed)
            color = tuple(rng.integers(0, 256, size=(3), dtype=np.uint32))
        elif color_mode == 'contrast':
            region = np.asarray(self.img)[h: h + size_h + 1, w: w + size_w + 1, :]
            color = np.mean(region, axis=(0, 1))
            color = (round(255 - color[0]), round(255 - color[1]), round(255 - color[2]))
        elif color_mode == 'uniform':
            if color == None:
                print("Missing the color parameter.")
                raise NotImplementedError
            else:
                color = color
        else:
            raise NotImplementedError
        if self.bounding_width is not None:
            self.draw.rectangle([w, h, w + size_w, h + size_h], outline=color, width=self.bounding_width)
        self.draw.text((w + self.margin // 2, h + self.margin // 2), word, fill=color, font=font, anchor="lt")
        tmp_array = np.zeros((self.height, self.width))
        tmp_array[h: h + size_h + 1, w: w + size_w + 1] = 1
        tmp_array = np.cumsum(np.cumsum(tmp_array.astype(np.uint32), axis=1), axis=0)
        self.integral += tmp_array
        return True
    
    def save(self, path):
        self.img.save(path)





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

    def generate(self, max_font_size = 50, min_font_size = 20, font_size_func = lambda x: x ** 0.6, 
                 momentum = 0.9, width = 400, height = 400, max_words = 50, mask = None, 
                 margin = 6, bounding_width = None, seed = 114, color_mode = 'contrast',
                 color = None, print_res = True, save_path = "tmp.png", background_path = None,
                 ):

        background = None
        if background_path != None:
            background = Image.open(background_path)
            background = background.convert("RGB")
            background = background.resize((height, width))
        canvas = Canvas(height, width, mask, margin, bounding_width, background=background)

        sorted_items = sorted(self.counter.items(), key=lambda item: item[1], reverse=True)
        self.counter = dict(sorted_items)
        first = next(iter(self.counter.items()))
        maxn = first[1]
        num = 0
        m = 1
        for word in self.counter:
            freq = self.counter[word] / maxn
            m = momentum * m + (1 - momentum) * freq
            font_size = int(round(font_size_func(m) * max_font_size))
            if(font_size < min_font_size):
                if print_res:
                    print("Break because font size are too small.")
                break
            font = ImageFont.truetype(self.font_path, font_size)
            rng = np.random.default_rng(seed + num)
            tmp = rng.integers(0, 100000, dtype = np.uint32)
            res = canvas.draw_word(word, font, seed = tmp, color_mode = color_mode, color = color)
            if res == False:
                if print_res:
                    print("Break because no space to draw more words.")
                break
            num += 1
            if num >= max_words:
                if print_res:
                    print("Break because reach the limit of maximum word numbers.")
                break
        canvas.save(save_path)
        if print_res:
            print(f"Successfully generated a word cloud of {num:d} words ~~~")
            print(f"Picture saved to {save_path}")


