# -*- coding: utf-8 -*-

import os
import pandas as pd


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for f in files:
            yield os.path.join(root, f)


data = pd.DataFrame({
    'title': [],
    'category': []
})

for file in find_all_files('./raw_data'):
    path, ext = os.path.splitext(file)
    if ext != '.csv':
        continue

    with open(file, "r") as f:
        csv_data = pd.read_csv(file)
        data = pd.concat([data, csv_data[['title', 'category']]])

data['category'] = data['category'].replace({
    '政治': '0',
    '社会': '1',
    '国際': '2',
    'ビジネス': '3',
    '科学・文化': '4',
    '暮らし': '5',
    '気象・災害': '6',
    'スポーツ': '7'
})

data.to_csv('learning_data.csv', index=False)

print('Success learning_data.csv')
