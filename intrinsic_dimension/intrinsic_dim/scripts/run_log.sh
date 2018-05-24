#! /bin/bash


#echo "just for reference"
#exit 0

# [Thu 2017-10-19  6:42:33 pm] Toy example
CUDA_VISIBLE_DEVICES=0 resman -d results/toy -r toy_direct -- ./train_toy.py --opt adam --lr .01 -E 1000
for dim in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 30 50 80 100 200 300 500; do
    CUDA_VISIBLE_DEVICES=0 resman -d results/toy -r toy_proj_d${dim} --  ./train_toy.py --vsize $dim --opt adam --lr .5 -E 1000 --denseproj
done
exit 0

# [Wed 2017-10-18 10:45:41 pm] Random direct LeNet for CIFAR
for ii in `seq -w 1000`; do
    echo "cd `pwd`; resman -d results/tiny_cifar_lenet_dir -r tiny_cifar_lenet_dir_${ii} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 --arch cifarlenet_dir `./rand_args.py --lenet --l2` -E 100 --opt 'adam' --lr 0.001"
done
exit 0
# [Wed 2017-10-18 10:45:14 pm] Random direct MLP for CIFAR
for ii in `seq -w 1000`; do
    echo "cd `pwd`; resman -d results/tiny_cifar_mlp_dir -r tiny_cifar_mlp_dir_${ii} -- ./train.py /data/cifar-10/h5/cn-b01c/cifar-10-train.h5 /data/cifar-10/h5/cn-b01c/cifar-10-test.h5 --arch cifarfc_dir `./rand_args.py --l2` -E 100 --opt 'adam' --lr 0.001"
done
exit 0
# [Wed 2017-10-18 10:11:06 pm] Random direct Lenet for MNIST
for ii in `seq -w 1000`; do
    echo "cd `pwd`; resman -d results/tiny_mnist_lenet_dir -r tiny_mnist_lenet_dir_${ii} -- ./train.py /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 --arch mnistlenet_dir `./rand_args.py --lenet` -E 100 --opt 'adam' --lr 0.001 --l2 0.0001"
done
exit 0

# [Wed 2017-10-18  4:16:35 pm] L2vsdim Regularization comparison
for seed in 10 11 12 13 14; do
    for dim in 200 300 500 800 1000 2000 3000 5000 8000 10000 20000 30000 50000 80000 100000 200000 300000; do
        for l2 in 0; do
            echo "cd `pwd`; resman -d results/l2vsdim -r mnist_sgd_fastf_dim${dim}_l2${l2}_s${seed} -- ./train.py --arch mnistfc --fastfoodproj /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 --depth 2 --width 400 --opt sgd --lr .1 --mom .9 --lrr .5 --lrep 16 --lrst 5 -E 120 --l2 $l2 --vsize $dim --seed $seed"
        done
    done
    for l2 in 0 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2; do
        echo "cd `pwd`; resman -d results/l2vsdim -r mnist_sgd_dir_l2${l2}_s${seed} -- ./train.py --arch mnistfc_dir /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 --depth 2 --width 400 --opt sgd --lr .1 --mom .9 --lrr .5 --lrep 16 --lrst 5 -E 120 --l2 $l2 --seed $seed"
    done
done
exit 0

# [Wed 2017-10-18  2:57:32 am] mnist_sgd_fastf
for dim in 500 1000 2000 5000 10000 20000 50000 100000 200000; do
    for l2 in 0 1e-7 1e-6 1e-5 1e-4; do
        for seed in 10 11 12; do
            #echo "cd `pwd`; resman -d results/mnist_sgd_fastf -r mnist_sgd_fastf_dim${dim}_l2${l2}_s${seed} -- ./train.py --arch mnistfc --fastfoodproj /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 --depth 2 --width 400 --opt sgd --lr .1 --mom .9 --lrr .5 --lrep 4 --lrst 5 -E 30 --l2 $l2 --vsize $dim --seed $seed"
            echo "cd `pwd`; resman -d results/mnist_sgd_fastf -r mnist_sgd_4E_fastf_dim${dim}_l2${l2}_s${seed} -- ./train.py --arch mnistfc --fastfoodproj /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 --depth 2 --width 400 --opt sgd --lr .1 --mom .9 --lrr .5 --lrep 16 --lrst 5 -E 120 --l2 $l2 --vsize $dim --seed $seed"
        done
    done
done
exit 0

# [Tue 2017-10-17  7:48:21 pm] mnist_sgd_dir
for seed in 10 11 12 13 14; do
    echo "cd `pwd`; resman -d results/mnist_sgd_dir -r mnist_sgd_dir_s${seed} -- ./train.py --arch mnistfc_dir /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 --depth 2 --width 400 --opt sgd --lr .1 --mom .9 --lrr .5 --lrep 4 --lrst 5 -E 30 --seed $seed"
done
exit 0

# [Tue 2017-10-17  9:24:30 pm] MNIST Equal FF Proj Adam
MNIST_DATA="/data/mnist/h5/train.h5 /data/mnist/h5/val.h5"
for dim in 500 1000 2000 5000 10000 20000 50000; do
    for l2 in 0 1e-7 1e-6 1e-5 1e-4; do
        for seed in 10; do
            #CUDA_VISIBLE_DEVICES=`nextcuda` resman -r mnist_fastf_l2${l2}_s${seed} -- ./train.py --arch mnistfc --fastfoodproj $MNIST_DATA --depth 2 --width 400 --opt adam --lr .001 --lrr .5 --lrep 4 --lrst 5 -E 30 --l2 $l2 --seed $seed &
            #CUDA_VISIBLE_DEVICES=`nextcuda -d 30` resman -r mnist_3L4E_fastf_l2${l2}_s${seed} -- ./train.py --arch mnistfc --fastfoodproj $MNIST_DATA --depth 2 --width 400 --opt adam --lr .003 --lrr .5 --lrep 16 --lrst 5 -E 120 --l2 $l2 --vsize $dim --seed $seed &
            echo "cd `pwd`; resman -d results/mnist_adam_proj -r mnist_3L4E_fastf_dim${dim}_l2${l2}_s${seed} -- ./train.py --arch mnistfc --fastfoodproj $MNIST_DATA --depth 2 --width 400 --opt adam --lr .003 --lrr .5 --lrep 16 --lrst 5 -E 120 --l2 $l2 --vsize $dim --seed $seed"
        done
    done
done
exit 0

# [Tue 2017-10-17  8:47:25 pm] MNIST Equal Direct
MNIST_DATA="/data/mnist/h5/train.h5 /data/mnist/h5/val.h5"
for l2 in 0 1e-7 1e-6 1e-5 1e-4; do
    for seed in 10 11 12 13 14; do 
        CUDA_VISIBLE_DEVICES=`nextcuda` resman -r mnist_dir_l2${l2}_s${seed} -- ./train.py --arch mnistfc_dir $MNIST_DATA --depth 2 --width 400 --opt adam --lr .001 --lrr .5 --lrep 4 --lrst 5 -E 30 --l2 $l2 --seed $seed &
    done
done
wait
exit 0

# [ Thu Oct 12 16:20:52 EDT 2017 ] SqueezeNet with Fastfood
IMGNET_DATA="/data/imagenet/h5/train.h5 /data/imagenet/h5/val.h5"
for vsize in 10000 100000 200000 500000; do
    CUDA_VISIBLE_DEVICES=`nextcuda` resman -r squeezenet_fastfood_$vsize -- ./train.py $IMGNET_DATA --arch squeeze --fastfoodproj --mb 256 --vsize $vsize -E 50 -L 0.01 --lrr 0.5 --lrep 7 &
done
exit 0

# [Tue 2017-10-10  3:28:35 pm] Random HP search for MNIST MLP
for ii in `seq -w 1000`; do
    CUDA_VISIBLE_DEVICES=`nextcuda -d 2000` resman -r tiny_mnist_mlp_sweep_${ii} -- ./train.py --arch mnistfc_dir /data/mnist/h5/train.h5 /data/mnist/h5/val.h5 `./rand_args.py` --opt adam --lr .001 --lrr .5 --lrep 4 --lrst 5 -E 30 &
done
wait
exit 0

# [Thu 2017-10-05  4:44:44 pm] Scaling tests at just over 100k parameters
MNIST_DATA="/data_local/mnist/h5/train.h5 /data_local/mnist/h5/val.h5"
# Direct
CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_direct -- ./train.py --arch mnistfc_dir $MNIST_DATA -E 3 --depth 3 --width 101 --output skip
for vsize in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
    CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_fastfood_$vsize -- ./train.py --arch mnistfc --fastfoodproj $MNIST_DATA -E 3 --depth 3 --width 101 --mb 256 --vsize $vsize --output skip
done
for vsize in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
    CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_sparse_$vsize -- ./train.py --arch mnistfc --sparseproj $MNIST_DATA -E 3 --depth 3 --width 101 --mb 256 --vsize $vsize --output skip
done
for vsize in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
    CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_dense_$vsize -- ./train.py --arch mnistfc --denseproj $MNIST_DATA -E 3 --depth 3 --width 101 --mb 256 --vsize $vsize --output skip
done

exit 0

# [Thu 2017-10-05  3:12:00 pm] Scaling tests at just over 1M parameters
MNIST_DATA="/data_local/mnist/h5/train.h5 /data_local/mnist/h5/val.h5"
# Direct
CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_direct -- ./train.py --arch mnistfc_dir $MNIST_DATA -E 3 --depth 3 --width 536 --output skip
for vsize in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
    # LRB Dense
    CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_dense_$vsize -- ./train.py --arch mnistfc --denseproj $MNIST_DATA -E 3 --depth 3 --width 536 --mb 256 --vsize $vsize --output skip
    # LRB Sparse
    CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_sparse_$vsize -- ./train.py --arch mnistfc --sparseproj $MNIST_DATA -E 3 --depth 3 --width 536 --mb 256 --vsize $vsize --output skip
    # LRB Fastfood
    CUDA_VISIBLE_DEVICES=0 resman -r mnist_timing_fastfood_$vsize -- ./train.py --arch mnistfc --fastfoodproj $MNIST_DATA -E 3 --depth 3 --width 536 --mb 256 --vsize $vsize --output skip
done


# [ Fri Oct  6 16:41:49 EDT 2017 ] AlexNet
CeDA_VISIBLE_DEVICES=5 ipython --pdb -- ./train.py /data_local/imagenet/h5/train.h5 /data_local/imagenet/h5/val.h5 --arch alexnet_dir
MNIST_DATA="/data_local/mnist/h5/train.h5 /data_local/mnist/h5/val.h5"


