#!/bin/bash
# CUDA_VISIBLE_DEVICES=6 ipython ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet
# {0,10,50,100,500,750,1000,1250,1500,1750,1900,1950,2000,2050,2100,2250,2500,5000,10000,15000}
# for dim in {0,250,500,750,1000,1250,1500,1750,1900,1950,2000,2050,2100,2250,2500,3000,4000,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9750,10000,15000,20000,25000,30000,35000,40000,45000,50000}

for l2_w in {0,}
do 
	for depth in {2,}
	do
		for width in {200,}
		do
			for dim in {1000,}
			do
				if [ "$dim" = 0 ]; then
					echo dir_"$dim"
					CUDA_VISIBLE_DEVICES=6 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 10 --vsize $dim --opt 'adam' --lr 0.001 --l2 $l2_w  --arch cifarfc_dir --depth $depth --width $width
				else	
					echo lrb_"$dim"
					CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 10 --vsize $dim --opt 'adam' --lr 0.001 --l2 $l2_w  --arch cifarfc --depth $depth --width $width --fastfoodproj
				fi
			done
		done
	done
done
