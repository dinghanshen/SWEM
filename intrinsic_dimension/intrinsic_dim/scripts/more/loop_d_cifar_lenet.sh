#!/bin/bash
# CUDA_VISIBLE_DEVICES=6 ipython ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet
# {}

# for dim in {0,100,500,750,1000,1250,1500,1750,1900,1950,2000,2050,2100,2250,2400,2500,2600,2750,2900,3000,4000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,62006}


for dim in {13116,200,0}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/lenet_cifar_fastf -r lenet_cifar_lrb_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 1e-05 --arch cifarlenet_dir  --c1=10 --c2=16 --d1=20 --d2=10 &
	else	
		echo lrb_"$dim"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/lenet_cifar_fastf -r lenet_cifar_lrb_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 1e-05 --arch cifarlenet --c1=10 --c2=16 --d1=20 --d2=10 --fastfoodproj &
	fi
done

# for dim in {62006,}
# do
# 	if [ "$dim" = 0 ]; then
# 		echo dir_"$dim"
# 		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/lenet_cifar_fastf -r lenet_cifar_dir_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet_dir &
# 	else	
# 		echo lrb_"$dim"
# 		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -d results/lenet_cifar_fastf -r lenet_cifar_lrb_"$dim" -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet --fastfoodproj &
# 	fi
# done

