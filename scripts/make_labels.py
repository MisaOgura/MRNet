#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np


def load_csv(data_type, condition):
    csv_path = f'{data_dir}/{data_type}-{condition}.csv'
    return pd.read_csv(csv_path,
                       header=None,
                       names=['case', condition],
                       dtype={'case': str, f'{condition}': np.int64})

def main(data_dir):
    train_abnormal_df = load_csv('train', 'abnormal')
    train_acl_df = load_csv('train', 'acl')
    train_meniscus_df = load_csv('train', 'meniscus')

    train_df = pd.merge(train_abnormal_df, train_acl_df, on='case') \
                 .merge(train_meniscus_df, on='case')

    valid_abnormal_df = load_csv('valid', 'abnormal')
    valid_acl_df = load_csv('valid', 'acl')
    valid_meniscus_df = load_csv('valid', 'meniscus')

    valid_df = pd.merge(valid_abnormal_df, valid_acl_df, on='case') \
                 .merge(valid_meniscus_df, on='case')

    train_df.to_csv(f'{data_dir}/train_labels.csv', index=False)
    valid_df.to_csv(f'{data_dir}/valid_labels.csv', index=False)

    print(f"Created 'train_labels.csv' and 'valid_labels.csv' in {data_dir}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 make_labels.py <data_dir>')
        print('e.g. python3 make_labels.py MRNet-v1.0')
        exit(1)

    data_dir = sys.argv[1]

    main(data_dir)
