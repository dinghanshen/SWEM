#!/bin/bash

# test cifar ""lucky_LeNet"" model
for repeat in {1,2,3}
do
	for dim in {0,100,200,250,400,500,750,1000,1250,1500,1750,1900,1950,2000,2050,2100,2250,2400,2500,2600,2750,2900,3000,4000,5000,7500,10000,12000,13116}
	do
		if [ "$dim" = 0 ]; then
			# echo dir_"$dim"
			echo "cd `pwd`; resman -d results/chun/results_cifar_lucky_LeNet -r LeNet_${dim}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 1e-05 --arch cifarlenet_dir --c1=10 --c2=16 --d1=20 --d2=10"
		else	
			# echo lrb_"$dim"
			echo "cd `pwd`; resman -d results/chun/results_cifar_lucky_LeNet -r LeNet_${dim}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 1e-05 --arch cifarlenet --c1=10 --c2=16 --d1=20 --d2=10 --fastfoodproj"
		fi
	done
done


# test cifar MLP-lenet model
for dim in {4000,5000,7500,10000,15000,20000,35000,50000,75000,100000}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet3 -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet3 -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj --skiptfevents"
	fi
done

exit 0

########################################################################



# Re-do cifar MLP model ( fix the error in Flatten() when building models)
for repeat in {1,2,3}
do
	for dim in {0,1000,2000,3000,4000,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8500,9000,10000,12500,15000,20000,30000,40000,50000,60000,70000,79685}
	do
		for depth in {2,3}
		do
			for width in {400,}
			do
				if [ "$dim" = 0 ]; then
					# echo dir_"$dim"
					echo "cd `pwd`; resman -d results/chun/results_cifar_MLP2 -r MLP_${dim}_${depth}_${width}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc_dir --depth $depth --width $width"
				else	
					# echo lrb_"$dim"
					echo "cd `pwd`; resman -d results/chun/results_cifar_MLP2 -r MLP_${dim}_${depth}_${width}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc  --depth $depth --width $width --fastfoodproj"
				fi
			done
		done		
	done
done
exit 0

# Add more dimensions for convnet controls (MLP_LeNet)

# test mnist MLP-lenet model
for dim in {1750,2000,2250,2500}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_MLP_LeNet3 -r mnist_MLP_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_MLP_LeNet3 -r mnist_MLP_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj --skiptfevents"
	fi
done


# test cifar MLP-lenet model
for dim in {17500,20000,25000}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet3 -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet3 -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj --skiptfevents"
	fi
done

exit 0

########################################################################

# Add more dots to compare performance vs trainable variables

# test cifar lenet model
for repeat in {1,2,3}
do
	for dim in {0,100,500,750,1000,1250,1500,1750,1900,1950,2000,2050,2100,2250,2400,2500,2600,2750,2900,3000,4000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,62006}
	do
		if [ "$dim" = 0 ]; then
			# echo dir_"$dim"
			echo "cd `pwd`; resman -d results/chun/results_cifar_LeNet -r LeNet_${dim}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet_dir"
		else	
			# echo lrb_"$dim"
			echo "cd `pwd`; resman -d results/chun/results_cifar_LeNet -r LeNet_${dim}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarlenet --fastfoodproj"
		fi
	done
done

# test mnist lenet model
for repeat in {1,2,3}
do
	for dim in {0,10,50,75,100,150,200,225,250,260,275,290,300,310,325,350,375,380,390,400,410,420,425,450,475,500,550,600,1000,1100,1200,1300,1400,1500,2000,2500,3000,3500,4000,4500,5000,7500,10000,15000,20000,25000,30000,35000,40000,42000,44426}
	do
		if [ "$dim" = 0 ]; then
			# echo dir_"$dim"
			echo "cd `pwd`; resman -d results/chun/results_mnist_LeNet -r LeNet_${dim}_${repeat} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistlenet_dir"
		else	
			# echo lrb_"$dim"
			echo "cd `pwd`; resman -d results/chun/results_mnist_LeNet -r LeNet_${dim}_${repeat} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistlenet --fastfoodproj"
		fi
	done
done

# test cifar MLP model
for repeat in {1,2,3}
do
	for dim in {0,1000,2000,3000,4000,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8500,9000,10000,12500,15000,20000,30000,40000,50000,60000,70000,79685}
	do
		for depth in {2,3}
		do
			for width in {400,}
			do
				if [ "$dim" = 0 ]; then
					# echo dir_"$dim"
					echo "cd `pwd`; resman -d results/chun/results_cifar_MLP -r MLP_${dim}_${depth}_${width}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc_dir --depth $depth --width $width"
				else	
					# echo lrb_"$dim"
					echo "cd `pwd`; resman -d results/chun/results_cifar_MLP -r MLP_${dim}_${depth}_${width}_${repeat} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarfc  --depth $depth --width $width --fastfoodproj"
				fi
			done
		done		
	done
done
exit 0


########################################################################################
# Re-do convnet controls experiments
#
# test mnist untied-lenet model
for dim in {0,100,200,225,250,275,300,325,350,375,400,425,450,475,500,550,600,700,800,900,1000,1100,1200,1300,1400,1500}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_Untied_LeNet3 -r mnist_Untied_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet_dir --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_Untied_LeNet3 -r mnist_Untied_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet --fastfoodproj --skiptfevents"
	fi
done

# test cifar untied-lenet model
for dim in {0,1000,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,6750,7000,7500,8000,9000,10000,11000,12000,13000,14000,15000}
do
	if [ "$dim" = 0 ]; then
	        #echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_Untied_LeNet3 -r cifar_Untied_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet_dir --skiptfevents"
	else	
		#echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_Untied_LeNet3 -r cifar_Untied_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet --fastfoodproj --skiptfevents"
	fi
done


# test mnist MLP-lenet model
for dim in {0,100,200,225,250,275,300,325,350,375,400,425,450,475,500,550,600,700,800,900,1000,1100,1200,1300,1400,1500}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_MLP_LeNet3 -r mnist_MLP_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_MLP_LeNet3 -r mnist_MLP_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj --skiptfevents"
	fi
done

# test cifar MLP-lenet model
for dim in {0,1000,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,6750,7000,7500,8000,9000,10000,11000,12000,13000,14000,15000}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet3 -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir --skiptfevents"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet3 -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj --skiptfevents"
	fi
done

exit 0
########################################################################################





# 50% and 10% data samples for permuted label version of MNIST using Adam (lr=0.0001)
for pd in {0.5,0.1}
do
	for depth in {5,}
	do
		for width in {400,}
		do
			for dim in {0,1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000,210000,220000,230000,240000,250000,260000,270000,280000,290000,300000}
			do
				if [ "$dim" = 0 ]; then
					#echo dir_"$dim"_"$depth"_"$width"
					#CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width &
					echo "cd `pwd`; resman -d results/chun/results_pl_pd2 -r fnn_mnist_t500_pl_${dim}_${depth}_${width}_${pd} -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width --partial_data $pd" 
				else	
					#echo lrb_"$dim"_"$depth"_"$width"
					#CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj&
					echo "cd `pwd`; resman -d results/chun/results_pl_pd2 -r fnn_mnist_t500_pl_${dim}_${depth}_${width}_${pd} -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj --partial_data $pd"
				fi
			done
		done
	done
done
exit 0


# Additional dim for DQN on easy Cartpole setups: v2 and v3
for env_name in {'CartPole-v2','CartPole-v3'}
do
	for repeat in {1,2,3,4,5,1,2,3,4,5}
	do
		for width in {20,400}
		do
			for dim in {1,2,16,17,18,19}
			do
				if [ "$dim" = 0 ]; then
					echo "cd `pwd`; resman -d results/chun/rl_results/dqn_cps_easy -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/chun/rl_results/dqn_cps_easy/fnn_${env_name}_${dim}_${width}_${repeat}"
				else
					echo "cd `pwd`; resman -d results/chun/rl_results/dqn_cps_easy -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/chun/rl_results/dqn_cps_easy/fnn_${env_name}_${dim}_${width}_${repeat}"
				fi
			done
		done
	done
done				
exit 0


# test mnist untied-lenet model
for dim in {0,100,200,225,250,275,300,325,350,375,400,425,450,475,500,550,600,700,800,900,1000}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_Untied_LeNet -r mnist_Untied_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet_dir"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_Untied_LeNet -r mnist_Untied_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistUntiedlenet --fastfoodproj"
	fi
done

# test cifar untied-lenet model
for dim in {0,1000,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,6750,7000,7500,8000}
do
	if [ "$dim" = 0 ]; then
		echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_Untied_LeNet -r cifar_Untied_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet_dir"
	else	
		echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_Untied_LeNet -r cifar_Untied_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarUntiedlenet --fastfoodproj"
	fi
done


# test mnist MLP-lenet model
for dim in {0,100,200,225,250,275,300,325,350,375,400,425,450,475,500,550,600,700,800,900,1000}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_MLP_LeNet -r mnist_MLP_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet_dir"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_mnist_MLP_LeNet -r mnist_MLP_LeNet_${dim} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch mnistMLPlenet --fastfoodproj"
	fi
done

# test cifar MLP-lenet model
for dim in {0,1000,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,6750,7000,7500,8000}
do
	if [ "$dim" = 0 ]; then
		# echo dir_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet_dir"
	else	
		# echo lrb_"$dim"
		echo "cd `pwd`; resman -d results/chun/results_cifar_MLP_LeNet -r cifar_MLP_LeNet_${dim} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 -E 100 --vsize $dim --opt 'adam' --lr 0.001 --l2 0.0001 --arch cifarMLPlenet --fastfoodproj"
	fi
done

exit 0
########################################################################################

# 50% and 10% data samples for permuted label version of MNIST using Adam (lr=0.0001)
for pd in {0.5,0.1}
do
	for depth in {5,}
	do
		for width in {400,}
		do
			for dim in {0,1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000,210000,220000,230000,240000,250000,260000,270000,280000,290000,300000}
			do
				if [ "$dim" = 0 ]; then
					#echo dir_"$dim"_"$depth"_"$width"
					#CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width &
					echo "cd `pwd`; resman -d results/chun/results_pl_pd -r fnn_mnist_t500_pl_${dim}_${depth}_${width}_${pd} -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width --partial_data $pd" 
				else	
					#echo lrb_"$dim"_"$depth"_"$width"
					#CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj&
					echo "cd `pwd`; resman -d results/chun/results_pl_pd -r fnn_mnist_t500_pl_${dim}_${depth}_${width}_${pd} -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj --partial_data $pd"
				fi
			done
		done
	done
done
exit 0

# DQN on easy Cartpole setups: v2 and v3
for env_name in {'CartPole-v2','CartPole-v3'}
do
	for repeat in {1,2,3,4,5}
	do
		for width in {20,400}
		do
			for dim in {0,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,28,29,30}
			do
				if [ "$dim" = 0 ]; then
					echo "cd `pwd`; resman -d results/chun/rl_results/dqn_cps_easy -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/chun/rl_results/dqn_cps_easy/fnn_${env_name}_${dim}_${width}_${repeat}"
				else
					echo "cd `pwd`; resman -d results/chun/rl_results/dqn_cps_easy -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/chun/rl_results/dqn_cps_easy/fnn_${env_name}_${dim}_${width}_${repeat}"
				fi
			done
		done
	done
done				
exit 0


# Redo DQN on Cartpole-v0 for subspace 15-25
for env_name in {'CartPole-v0',}
do
	for repeat in {0,1,2,3,4,5,6,7,8,9}
	do
		for width in {20,400}
		do
			#for dim in {0,3}
			for dim in {15,16,17,18,19,20,21,22,23,24,25}
			do
				if [ "$dim" = 0 ]; then
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					echo "cd `pwd`; resman -d results/chun/rl_results/CP0_refine -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/chun/rl_results/fnn_${env_name}_${dim}_${width}_${repeat}"
				else	
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					echo "cd `pwd`; resman -d results/chun/rl_results/CP0_refine -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/chun/rl_results/fnn_${env_name}_${dim}_${width}_${repeat}"
				fi
			done
		done
	done
done				
exit 0



# Redo DQN on Cartpole-v1 for MAX_EPISODES = 10000 (Previous runs use 10)
for env_name in {'CartPole-v1',}
do
	for repeat in {1,2,3,4,5}
	do
		for width in {20,400}
		do
			#for dim in {0,3}
			for dim in {0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,125,150,175,225,250,275,325,350,375,425,450,475,500}
			do
				if [ "$dim" = 0 ]; then
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					echo "cd `pwd`; resman -d results/chun/rl_results -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/chun/rl_results/fnn_${env_name}_${dim}_${width}_${repeat}"
				else	
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					echo "cd `pwd`; resman -d results/chun/rl_results -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/chun/rl_results/fnn_${env_name}_${dim}_${width}_${repeat}"
				fi
			done
		done
	done
done				
exit 0


# DQN on Cartpole
for env_name in {'CartPole-v0','CartPole-v1'}
do
	for repeat in {1,2,3,4,5}
	do
		for width in {20,400}
		do
			#for dim in {0,3}
			for dim in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,125,150,175,225,250,275,325,350,375,425,450,475,500}
			do
				if [ "$dim" = 0 ]; then
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					echo "cd `pwd`; resman -d results/chun/rl_results -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/chun/rl_results/fnn_${env_name}_${dim}_${width}_${repeat}"
				else	
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					echo "cd `pwd`; resman -d results/chun/rl_results -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/chun/rl_results/fnn_${env_name}_${dim}_${width}_${repeat}"
				fi
			done
		done
	done
done				
exit 0


# Permuted label version of MNIST using Adam (lr=0.0001)
for depth in {1,2,3,4,5}
do
	for width in {50,100,200,400}
	do
		for dim in {0,1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000,210000,220000,230000,240000,250000,260000,270000,280000,290000,300000}
		do
			if [ "$dim" = 0 ]; then
				#echo dir_"$dim"_"$depth"_"$width"
				#CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width &
				echo "cd `pwd`; resman -d results/chun -r fnn_mnist_t500_pl_${dim}_${depth}_${width} -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --arch mnistfc_dir --depth $depth --width $width"
			else	
				#echo lrb_"$dim"_"$depth"_"$width"
				#CUDA_VISIBLE_DEVICES=`nextcuda --delay 10` resman -r fnn_mnist_t500_pl_"$dim"_"$depth"_"$width" -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj&
				echo "cd `pwd`; resman -d results/chun -r fnn_mnist_t500_pl_${dim}_${depth}_${width} -- ./train.py /data/mnist/h5/train_shuffled_labels_0.h5 /data/mnist/h5/test_shuffled_labels_0.h5 -E 500 --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --arch mnistfc --depth $depth --width $width --fastfoodproj"
			fi
		done
	done
done			

