#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np


def load_csv(csv_path):
    return pd.read_csv(csv_path,
                       header=None,
                       names=['case', 'abnormal'],
                       dtype={'case': str, 'abnormal': np.int64})


def main(data_dir, out_dir):
    train_abnormal_df = load_csv(f'{data_dir}/train-abnormal.csv')
    train_acl_df = load_csv(f'{data_dir}/train-acl.csv')
    train_meniscus_df = load_csv(f'{data_dir}/train-meniscus.csv')

    train_df = pd.merge(train_abnormal_df, train_acl_df, on='case') \
                 .merge(train_meniscus_df, on='case')

    valid_abnormal_df = load_csv(f'{data_dir}/valid-abnormal.csv')
    valid_acl_df = load_csv(f'{data_dir}/valid-acl.csv')
    valid_meniscus_df = load_csv(f'{data_dir}/valid-meniscus.csv')

    valid_df = pd.merge(valid_abnormal_df, valid_acl_df, on='case') \
                 .merge(valid_meniscus_df, on='case')

    train_df.to_csv(f'{out_dir}/train_labels.csv')
    valid_df.to_csv(f'{out_dir}/valid_labels.csv')

    print(f"Created 'train_labels.csv' and 'valid_labels.csv' in {out_dir}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 make_labels.py <data_dir> <out_dir>')
        print('e.g. python3 make_labels.py data/MRNet-v1.0 data/processed')
        exit(1)

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    main(data_dir, out_dir)
