# test SWEM AG-news model
# for dim in {0,1,3,5,10,50,100,250,500,1000}
for dim in {0,1}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/swem -r swem_${dim} -- ./train.py dataset/mnist/train.h5 dataset/mnist/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch swem_dir --depth 0 --width 200 --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/swem -r swem_${dim} -- ./train.py dataset/mnist/train.h5 dataset/mnist/test.h5 -E 5 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch swem --depth 0 --width 200 --vsize $dim --sparseproj --skiptfevents"
	fi
done

exit 0