#!/bin/bash
# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {125,150,175,225,250,275,325,350,375,425,450,475,500} {0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,100,200,300,400}
# CUDA_VISIBLE_DEVICES=0 resman -r dir -d fnn_cartpole0 -- ipython -- ./train_rl.py train.h5 val_h5 --vsize 10 --opt 'adam' --lr 0.0001 --l2 0.0001 --arch fc_dir --depth 2 --width 20 --output fnn_cartpole0
# {125,150,175,225,250,275,325,350,375,425,450,475,500}
# {0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,125,150,175,225,250,275,325,350,375,425,450,475,500}
for depth in {2,}
do
	for width in {400,20}
	do
		for dim in {0,3,4,5,6,7,8,9,10,11,12,13,14,15,15,17,18,19,20}
		do
			if [ "$dim" = 0 ]; then
				echo lrb_"$dim"_"$depth"_"$width"
				CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r dir -d fnn_cartpole_"$dim"_"$depth"_"$width" -- ipython -- ./train_rl.py train.h5 val_h5 --vsize $dim --opt 'rmsprop' --lr 0.0001 --l2 0.0001 --arch fc_dir --depth $depth --width $width --output fnn_cartpole_"$dim"_"$depth"_"$width" &
			else	
				echo lrb_"$dim"_"$depth"_"$width"
				CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r dir -d fnn_cartpole_"$dim"_"$depth"_"$width" -- ipython -- ./train_rl.py train.h5 val_h5 --vsize $dim --opt 'rmsprop' --lr 0.0001 --l2 0.0001 --arch fc --depth $depth --width $width --output fnn_cartpole_"$dim"_"$depth"_"$width" &
			fi
		done
	done
done			

