#!/usr/bin/env python3.6
"""Merges csv files for each diagnosis provided in the original dataset into
one csv per train/valid dataset.

Usage:
  make_labels.py <data_dir>
  make_labels.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_dir>         Path to a directory where the data lives e.g. 'MRNet-v1.0'
"""

import sys
import pandas as pd
import numpy as np
from docopt import docopt


def load_csv(data_dir, data_type, condition):
    csv_path = f'{data_dir}/{data_type}-{condition}.csv'
    return pd.read_csv(csv_path,
                       header=None,
                       names=['case', condition],
                       dtype={'case': str, f'{condition}': np.int64})

def main(data_dir):
    train_abnormal_df = load_csv(data_dir, 'train', 'abnormal')
    train_acl_df = load_csv(data_dir, 'train', 'acl')
    train_meniscus_df = load_csv(data_dir, 'train', 'meniscus')

    train_df = pd.merge(train_abnormal_df, train_acl_df, on='case') \
                 .merge(train_meniscus_df, on='case')

    valid_abnormal_df = load_csv(data_dir, 'valid', 'abnormal')
    valid_acl_df = load_csv(data_dir, 'valid', 'acl')
    valid_meniscus_df = load_csv(data_dir, 'valid', 'meniscus')

    valid_df = pd.merge(valid_abnormal_df, valid_acl_df, on='case') \
                 .merge(valid_meniscus_df, on='case')

    train_df.to_csv(f'{data_dir}/train_labels.csv', index=False)
    valid_df.to_csv(f'{data_dir}/valid_labels.csv', index=False)

    print(f"Created 'train_labels.csv' and 'valid_labels.csv' in {data_dir}")


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_dir>'])
