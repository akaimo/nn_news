# coding: utf-8

import collections
import pandas as pd
import MeCab

learning_data = pd.read_csv('learning_data.csv')
mecab = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd -Owakati')


Datasets = collections.namedtuple('Datasets', ['train', 'validation'])

class DataSet(object):
    def __init__(self, texts, labels):
        self._texts = texts
        self._labels = labels


def read_data_sets(validation_size=4000):
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

    validation_text = input_data[:validation_size]
    validation_labels = correct_data[:validation_size]
    train_text = input_data[validation_size:]
    train_labels = correct_data[validation_size:]

    train = DataSet(train_text, train_labels)
    validation = DataSet(validation_text, validation_labels)

    return Datasets(train=train, validation=validation)
