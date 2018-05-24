#!/bin/bash
# CUDA_VISIBLE_DEVICES=6 ipython ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet
# {0,10,50,100,500,750,1000,1250,1500,1750,1900,1950,2000,2050,2100,2250,2500,5000,10000,15000}

for dim in {10000,}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=6 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet_dir &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 10 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet --sparseproj&
	fi
done

