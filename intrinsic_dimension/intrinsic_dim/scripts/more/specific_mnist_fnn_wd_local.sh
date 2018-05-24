#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 resman -r fnn_mnist_lrb_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize 10 --opt 'sgd' --lr 0.1 --l2 0.0001 --arch mnistfc --depth 2 --width 100
# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000}
# 
for depth in {2,}
do
  for width in {400,}
  do
    for dim in {300,}
    do
      if [ "$dim" = 0 ]; then
        echo dir_"$dim"_"$depth"_"$width"
        CUDA_VISIBLE_DEVICES=0 python ./train.py ~/data/mnist/h5/train.h5 ~/data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'sgd' --lr 0.1 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width &
      else  
        echo lrb_"$dim"_"$depth"_"$width"
        CUDA_VISIBLE_DEVICES=0 python ./train.py ~/data/mnist/h5/train.h5 ~/data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'sgd' --lr 0.1 --l2 0.0001 --arch mnistfc --depth $depth --width $width &
      fi
    done
  done
done     