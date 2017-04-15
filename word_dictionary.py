# coding: utf-8

import pandas as pd
import MeCab


learning_data = pd.read_csv('learning_data.csv')
# Ubuntu
mecab = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
# Mac
# mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')

s = set()

for text in learning_data['title']:
    t = mecab.parseToNode(text)
    while t:
        if t.surface == '':
            t = t.next
            continue
        s.add(t.surface)
        t = t.next

f = open('word_dictionary.txt', 'w')
for i, text in enumerate(s):
    f.write(str(i) + ': ' + text + '\n')
