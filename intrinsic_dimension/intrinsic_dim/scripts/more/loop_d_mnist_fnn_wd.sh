#!/bin/bash
# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000}
# 
# for depth in {0,1,2,3,4,5}
# for width in {50,100,200,400}
# for dim in {0,10,50,100,200,300,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,850,900,1000,1250,1500}
# {1750,2000,2250,2500,3000,4000,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12500,15000,17500,20000,22500,25000}
for depth in {1,2,3,4,5}
do
	for width in {50,100,200,400}
	do
		for dim in {1750,2000,2250,2500,3000,4000,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000}
		do
			if [ "$dim" = 0 ]; then
				echo dir_"$dim"_"$depth"_"$width"
				CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r fnn_mnist_d_higher_1500/fnn_mnist_dir_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width &
			else	
				echo lrb_"$dim"_"$depth"_"$width"
				CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r fnn_mnist_d_higher_1500/fnn_mnist_lrb_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj &
			fi
		done
	done
done	



