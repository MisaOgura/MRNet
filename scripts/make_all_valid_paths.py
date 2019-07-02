#!/usr/bin/env python3.6
"""Creates a csv containing paths for the validation dataset - to be used as an
argument to src/predict.py.

Usage:
  make_all_valid_paths.py <data_dir> <output_dir>
  make_all_valid_paths.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_dir>         Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <output_dir>       Directory where paths are saved as a csv file (with no header)
                     e.g. 'out_dir'
"""

import os
import sys
import csv
from docopt import docopt


def main(data_dir, output_dir):
    num_cases = 120
    staring_case = 1130
    base_valid_path = f'{data_dir}/valid'
    planes = ['sagittal', 'coronal', 'axial']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/all_valid_paths.csv'

    if os.path.exists(output_file):
        os.rename(output_file, f'{output_file}.back')
        print(f'!! {output_file} already exists, renamed to {output_file}.bak')

    print(f'Generating a list of paths to validation dataset...')
    print(f'Paths will be saved as {output_file}')

    with open(output_file, 'w') as f:
        for i in range(num_cases):
            current_case = 1130 + i

            for plane in planes:
                case_path = f'{base_valid_path}/{plane}/{current_case}.npy'
                writer = csv.writer(f)
                writer.writerow([case_path])


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_dir>'],
         arguments['<output_dir>'])
