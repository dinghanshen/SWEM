#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 resman -r fnn_cifar_dir -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize 100 --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc_dir --depth 2 --width 100
# CUDA_VISIBLE_DEVICES=0 resman -r fnn_cifar_lrb -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize 4000 --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc --depth 2 --width 100
for depth in {1,2,3,4,5}
do
	for width in {50,100,200,400}
	do
		for dim in {0,1000,2000,3000,4000,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8500,9000,10000,12500,15000}
		do
			if [ "$dim" = 0 ]; then
				echo dir_"$dim"_"$depth"_"$width"
				CUDA_VISIBLE_DEVICES=`nextcuda --delay 100` resman -r fnn_cifar_dir_"$dim"_"$depth"_"$width" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc_dir --depth $depth --width $width &
			else	
				echo lrb_"$dim"_"$depth"_"$width"
				CUDA_VISIBLE_DEVICES=`nextcuda --delay 100` resman -r fnn_cifar_lrb_"$dim"_"$depth"_"$width" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc --depth $depth --width $width &
			fi
		done
	done
done

