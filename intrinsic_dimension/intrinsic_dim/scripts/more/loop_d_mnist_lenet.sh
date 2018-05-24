#!/bin/bash
# for dim in {10,50,100,500,1000,2000,5000,10000,15000,20000}
# CUDA_VISIBLE_DEVICES=0 resman -r lenet_mnist_10 -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize 100 --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistlenet
# {0,100,200,210,225,250,260,275,290,300,310,325,350,375,380,390,400,410,420,425,450,475,500,550,600,700,800,900,1000}
# {1000,1500,2000,2500,3000,3500,4000,4500,5000,10000}
for dim in {1100,1200,1300,1400,1500,2000,2500,3000,3500,4000,4500,5000,10000,20000,30000,40000,44426}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r lenet_mnist_dir -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistlenet_dir &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r lenet_mnist_$dim -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistlenet --fastfoodproj &
	fi
done

