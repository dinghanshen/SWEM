#!/bin/bash
for dim in {10,50,100,300,500,1000,2000,3000,4000}
	do
		CUDA_VISIBLE_DEVICES=7 resman -r fnn_mnist_$dim -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --arch mnistfc
	done

