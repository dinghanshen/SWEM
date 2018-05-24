#!/bin/bash

# # test mnist MLP lenet model
# for dim in {100,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj
# 	fi
# done

for dim in {1750,2000,2250,2500}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_mnist_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_mnist_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj &
	fi
done

# # test cifar MLP lenet model
# for dim in {100,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj
# 	fi
# done


for dim in {17500,20000,25000}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_cifar_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/results_cifar_MLP_LeNet3 -r LeNet_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj &
	fi
done

# # test mnist untied lenet model
# for dim in {100,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet --fastfoodproj
# 	fi
# done


# # test cifar untied lenet model
# for dim in {3000,0}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet_dir
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=7 python ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet --fastfoodproj
# 	fi
# done


