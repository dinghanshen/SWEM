#!/bin/bash
for dim in {500,750}
do
	if [ "$dim" = 0 ]; then
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/DenseNet_40 -r densenet_cifar_$dim -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 300 --vsize $dim --opt 'adam' --lr 0.001 --arch cifarDenseNet_dir &
	else
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/DenseNet_40 -r densenet_cifar_$dim -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 300 --vsize $dim --opt 'adam' --lr 0.001 --arch cifarDenseNet --fastfoodproj &
	fi
done


# for dim in {0,50,100,250,500,1000,5000}
# do
# 	if [ "$dim" = 0 ]; then
# 		CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/DenseNet -r densenet_cifar_$dim -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 300 --vsize $dim --opt 'adam' --lr 0.001 --arch cifarDenseNet_dir &
# 	else
# 		CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/DenseNet -r densenet_cifar_$dim -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 300 --vsize $dim --opt 'adam' --lr 0.001 --arch cifarDenseNet --fastfoodproj &
# 	fi
# done
