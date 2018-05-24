#!/bin/bash
echo lrb_"$dim"_"$depth"_"$width"
# CUDA_VISIBLE_DEVICES=0 ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 1 --vsize $dim --opt 'adam' --lr 0.001 --arch mnistfc --depth $depth --width $width


