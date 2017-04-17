# coding: utf-8

import collections
import pandas as pd
import MeCab
import numpy

learning_data = pd.read_csv('learning_data.csv')
# Ubuntu
mecab = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
# Mac
# mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'dictionary'])


class DataSet(object):
    def __init__(self, texts, labels):
        self._texts = texts
        self._labels = labels

    @property
    def texts(self):
        return self._texts

    @property
    def labels(self):
        return self._labels


def read_data_sets(validation_size=4000, one_hot=False):
    word_dictionary = {}
    word_arrays = []
    input_data = []
    correct_data = []

    with open("word_dictionary.txt", "r") as file:
        for line in file:
            line = line.replace('\n', '')
            l = line.split(': ')
            word_dictionary[l[1]] = l[0]

    for line in learning_data['title']:
        mecab.parseToNode('')  # https://shogo82148.github.io/blog/2015/12/20/mecab-in-python3-final/
        text = mecab.parseToNode(line)
        word_array = []
        while text:
            if text.surface == '':
                text = text.next
                continue
            word_array.append(word_dictionary[text.surface])
            text = text.next
        word_arrays.append(word_array)

    max_count = 0
    for word_array in word_arrays:
        if max_count < len(word_array):
            max_count = len(word_array)

    for word_array in word_arrays:
        less_count = max_count - len(word_array)
        for _ in range(less_count):
            word_array.append(0)
        input_data.append(word_array)

    for line in learning_data['category']:
        correct_data.append(line)

    if one_hot:
        max_category_number = 0
        for line in learning_data['category']:
            if max_category_number < int(line):
                max_category_number = int(line)
        data = []
        for line in learning_data['category']:
            array = [0 for _ in range(max_category_number + 1)]
            array[int(line)] = 1
            data.append(array)
        correct_data = data

    validation_text = numpy.array(input_data[:validation_size])
    validation_labels = numpy.array(correct_data[:validation_size])
    train_text = numpy.array(input_data[validation_size:])
    train_labels = numpy.array(correct_data[validation_size:])

    train = DataSet(train_text, train_labels)
    validation = DataSet(validation_text, validation_labels)

    return Datasets(train=train, validation=validation, dictionary=word_dictionary)
