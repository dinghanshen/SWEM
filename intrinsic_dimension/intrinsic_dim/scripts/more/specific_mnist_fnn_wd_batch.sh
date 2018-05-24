#!/bin/bash
# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000}
Width=(50 50 50 50 50 50 50 50 50 50 50 100 100 100 100 100 100 100 100 200 200 200 200 200 200 200 200 200 200 400 400 400 400 400 400 400 400 400 400)
Depth=(1 2 3 3 4 4 5 5 5 5 5 1 2 2 3 4 4 5 5 0 0 2 2 2 3 3 4 4 5 0 2 2 3 4 5 5 5 5 5)
Dim=(600 700 675 775 500 600 575 600 625 675 1250 750 600 1500 775 600 800 750 850 600 750 700 800 850 525 750 650 700 700 850 550 1000 725 1500 775 800 900 1000 1500)

for i in "${!Width[@]}"; do
	if [ ${Dim[i]} = 0 ]; then
		# echo $i
		echo dir_"${Dim[i]}"_"${Depth[i]}"_"${Width[i]}"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r fnn_mnist_dir_"${Dim[i]}"_"${Depth[i]}"_"${Width[i]}" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize ${Dim[i]} --opt 'sgd' --lr 0.1 --l2 0.0001 --arch mnistfc_dir --depth ${Depth[i]} --width ${Width[i]} &
	else	
		echo lrb_"${Dim[i]}"_"${Depth[i]}"_"${Width[i]}"
		CUDA_VISIBLE_DEVICES=`nextcuda --delay 60` resman -r fnn_mnist_lrb_"${Dim[i]}"_"${Depth[i]}"_"${Width[i]}" -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize ${Dim[i]} --opt 'sgd' --lr 0.1 --l2 0.0001 --arch mnistfc --depth ${Depth[i]} --width ${Width[i]} &
	fi
done			

