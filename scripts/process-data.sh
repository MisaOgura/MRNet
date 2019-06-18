#!/bin/bash

set -e

if [ $# -ne 2 ]; then
  echo 'Usage: ./process-data.sh <data_dir> <out_dir>'
  echo 'e.g. ./process-data.sh data/MRNet-v1.0 data/processed'
fi

DATA_DIR=$1
OUT_DIR=$2

# Make sure the output dir is clean

echo "Cleaning the output directory '$OUT_DIR'..."

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

# Make labels

echo "Creating labels from csv files..."

python3 scripts/make_labels.py $DATA_DIR $OUT_DIR

# Process image data

echo "Converting .npy image data to .png files..."

find $OUT_DIR -name "*labels.csv"                               \
  | awk -F "/" -v data_dir=$DATA_DIR -v out_dir=$OUT_DIR '{
      csv_file=$0;
      dataset=$3; gsub("_labels.csv", "", dataset)
      print data_dir "/" dataset, csv_file, out_dir "/" dataset
    }'                                                          \
  | awk '{ print "./scripts/process-image-data.sh", $0}'        \
  | xargs -I {} bash -c "{}"

echo "Preprocessing finished."
