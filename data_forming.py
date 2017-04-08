# -*- coding: utf-8 -*-

import os
import pandas as pd


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for f in files:
            yield os.path.join(root, f)


data = pd.DataFrame({'title': [], 'category': []})
for file in find_all_files('./raw_data'):
    path, ext = os.path.splitext(file)
    if ext != '.csv':
        continue

    with open(file, "r") as f:
        csv_data = pd.read_csv(file)
        data = pd.concat([data, csv_data[['title', 'category']]])

data.to_csv('sample.csv', index=False)
