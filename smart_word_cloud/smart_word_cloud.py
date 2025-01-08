# coding=utf-8
# Author: Yewei Liu (Lewis) <liuyeweilewis@gmail.com> <2300012959@stu.pku.edu.cn>
#
# (c) 2025
# License: MIT

from __future__ import division

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
import re
import os

# set relative path
nltk_data_path = os.path.join(os.path.dirname(__file__), '../data')  
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

class MyCounter(Counter):
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return MyCounter({k: self[k] * scalar for k in self})
        return NotImplemented
        

class SmartWordCloudGenerator(object):
    def __init__(self, default_stopwords = True, nlp_improvement = True):
        
        self.counter = MyCounter()

        self.nlp_improvement =  nlp_improvement
        if self.nlp_improvement:
            self.lemmatizer = WordNetLemmatizer()
        
        self.stopwords = set()
        if default_stopwords:
            self.stopwords = set(stopwords.words('english'))
            if self.nlp_improvement:
                self.stopwords = set([self.lemmatizer.lemmatize(word) for word in self.stopwords])
    

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


    def generate(self):
        pass

if __name__ == '__main__':
    gen = SmartWordCloudGenerator()
    input_path = 'text/the little prince.txt'
    lines = None
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
    text = "".join(lines) 
    focus = {'flower':100}  
    gen.add_text(text, focus=focus)
    print(gen.counter)
