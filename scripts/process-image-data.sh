#!/bin/bash

set -e

if [ $# -ne 3 ]; then
  echo 'Usage: ./process-image-data.sh <data_dir> <labels.csv> <out_dir>'
  echo 'e.g. ./process-image-data.sh data_dir/train train_labels.csv out_dir/train'
fi

DATA_DIR=$1
LABELS_CSV=$2
OUT_DIR=$3

# Make sure the output dir is clean

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

# Get the number of CPU cores

if (( $(nproc) == 0 )); then
  num_cpu=$(nproc)
else
  num_cpu=$(sysctl -n hw.ncpu)
fi

# Parallelise the conversion

awk -F "," 'NR > 1 { print $2 }' $LABELS_CSV                       \
  | awk -v data_dir=$DATA_DIR -v out_dir=$OUT_DIR '{
      print "python3 scripts/convert_npy_to_png.py", data_dir, $0, out_dir
    }'                                                             \
  | xargs -P $num_cpu -I {} bash -c "{}"
