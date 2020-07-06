#!/bin/bash
set -e
index=7
while [ true ]
do
  if [ ${index} -gt 30 ]
  then
    echo "all processes completed"
    exit 0
  fi
  train_dir=augmented.${index}
  mask_csv=train_ship_segmentations_v2_aug.csv.${index}
  output_train_csv=train_ship_segmentations_train.csv.${index}
  output_valid_csv=train_ship_segmentations_valid.csv.${index}
  python airbus_unet34_keras-3-train-test-split.py --train-dir ${train_dir} --mask-csv ${mask_csv} --output-train-csv ${output_train_csv} --output-valid-csv ${output_valid_csv}
  index=$((index+1))
done
