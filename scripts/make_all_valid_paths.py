#!/usr/bin/python3

import sys
import csv


def main(data_dir, out_file):
    num_cases = 120
    staring_case = 1130
    base_valid_path = f'{data_dir}/valid'
    planes = ['sagittal', 'coronal', 'axial']


    with open(out_file, 'w') as out_file:
        for i in range(num_cases):
            current_case = 1130 + i

            for plane in planes:
                case_path = f'{base_valid_path}/{plane}/{current_case}.npy'
                writer = csv.writer(out_file)
                writer.writerow([case_path])


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_file = sys.argv[2]

    main(data_dir, out_file)
