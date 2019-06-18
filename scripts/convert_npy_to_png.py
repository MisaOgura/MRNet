#!/usr/bin/python3

import sys
import os
import numpy as np

from PIL import Image


def main(data_dir, case, out_dir):
    planes = ['axial', 'coronal', 'sagittal']

    for plane in planes:
        npy_path = f'{data_dir}/{plane}/{case}.npy'
        series = np.load(npy_path)

        dest_dir = f'{out_dir}/{case}/{plane}'
        os.makedirs(dest_dir)

        for i, image in enumerate(series):
            rgb = Image.fromarray(image, 'L').convert('RGB')
            png_path = f'{dest_dir}/{i:03d}.png'
            rgb.save(png_path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python3 convert_npy_to_png.py <data_dir> <case> <out_dir>')
        print('e.g. python3 convert_npy_to_png.py data/MRNet-v1.0/train 0001 data/processed/train')

    data_dir = sys.argv[1]
    case = sys.argv[2]
    out_dir = sys.argv[3]

    main(data_dir, case, out_dir)
