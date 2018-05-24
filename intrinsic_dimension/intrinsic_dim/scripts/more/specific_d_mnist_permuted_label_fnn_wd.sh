#!/bin/bash
# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000}

# for depth in {1,2,3,4,5}
# for width in {50,100,200,400}
# for dim in {0,10,50,100,200,300,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,850,900,1000,1250,1500}
# for dim in {0,1000,10000,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000} 
# {0,1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000,210000,220000,230000,240000,250000,260000,270000,280000,290000,300000}
for depth in {2,}
do
	for width in {400,}
	do
		for dim in {20000,}
		do
			for lr in {0.0001,}
			do
				for lrep in {0,}
					do
					if [ "$dim" = 0 ]; then
						echo dir_"$dim"_"$depth"_"$width"
						CUDA_VISIBLE_DEVICES=0 python ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr $lr --lrep $lrep --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width --partial_data 0.1
					else	
						echo lrb_"$dim"_"$depth"_"$width"
						CUDA_VISIBLE_DEVICES=1 python ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr $lr --lrep $lrep --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj --partial_data 1.0
					fi
				done
			done
		done
	done
done	


# for depth in {2,}
# do
# 	for width in {400,}
# 	do
# 		for dim in {0,}
# 		do
# 			for lr in {0.001,0.0005,0.0001}
# 			do
# 				for lrep in {0,100,250}
# 					do
# 					if [ "$dim" = 0 ]; then
# 						echo dir_"$dim"_"$depth"_"$width"
# 						CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$lr"_"$lrep"_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr $lr --lrep $lrep --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width &
# 					else	
# 						echo lrb_"$dim"_"$depth"_"$width"
# 						CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$lr"_"$lrep"_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr $lr --lrep $lrep --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj&
# 					fi
# 				done
# 			done
# 		done
# 	done
# done		

