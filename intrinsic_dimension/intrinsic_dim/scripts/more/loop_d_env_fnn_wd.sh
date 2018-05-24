#!/bin/bash
# for depth in {0,1,2,3,4,5}
# width in {10,50,100,200,400,600,800,1000}
# for dim in {125,150,175,225,250,275,325,350,375,425,450,475,500} {0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,100,200,300,400}
# CUDA_VISIBLE_DEVICES=0 resman -r dir -d fnn_cartpole0 -- ipython -- ./train_rl.py train.h5 val_h5 --vsize 10 --opt 'adam' --lr 0.0001 --l2 0.0001 --arch fc_dir --depth 2 --width 20 --output fnn_cartpole0
# {125,150,175,225,250,275,325,350,375,425,450,475,500}
# {0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,125,150,175,225,250,275,325,350,375,425,450,475,500}
# env_choices = ['CartPole-v0', 'CartPole-v1', 'CartPole-v2', 'MountainCar-v0','MountainCarContinuous-v0','Acrobot-v1','Pendulum-v0']


for env_name in {'CartPole-v2','CartPole-v3','CartPole-v4','CartPole-v5',}
do
	for repeat in {1,}
	do
		for width in {400,}
		do
			#for dim in {0,3}
			for dim in {0,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,28,29,30}
			do
				if [ "$dim" = 0 ]; then
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/dqn_results_cps/ -r fnn_${env_name}_${dim}_${width}_${repeat} ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/dqn_results_cps/fnn_${env_name}_${dim}_${width}_${repeat} &
				
					# CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/dqn_results_temp/ -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --width $width --output results/dqn_results_temp/fnn_${env_name}_${dim}_${width}_${repeat} 
				else	
					# echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
					# python ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/dqn_results_mc0/fnn_${env_name}_${dim}_${width}_${repeat}  
					CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` resman -d results/dqn_results_cps/ -r fnn_${env_name}_${dim}_${width}_${repeat} -- ./train_dqn_general.py --vsize $dim --opt adam --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --width $width --output results/dqn_results_cps/fnn_${env_name}_${dim}_${width}_${repeat} &
				fi
			done
		done
	done
done				
exit 0

# for env_name in {'CartPole-v1',}
# do
# 	for depth in {2,}
# 	do
# 		for width in {20,}
# 		do
# 			for dim in {0,100}
# 			do
# 				if [ "$dim" = 0 ]; then
# 					echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
# 					CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` python ./train_dqn_general.py --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc_dir --depth $depth --width $width --output rl_results/fnn_"$env_name"_"$dim"_"$depth"_"$width" & 
# 				else	
# 					echo lrb_"$env_name"_"$dim"_"$depth"_"$width"
# 					CUDA_VISIBLE_DEVICES=`nextcuda --delay 30` python ./train_dqn_general.py --vsize $dim --opt 'adam' --lr 0.0001 --l2 0.0001 --env_name $env_name --arch fc --depth $depth --width $width --output rl_results/fnn_"$env_name"_"$dim"_"$depth"_"$width" &
# 				fi
# 			done
# 		done
# 	done
# done				

