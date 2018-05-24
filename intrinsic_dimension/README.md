## Measuring the Intrinsic Dimension of Objective Landscapes


This repository contains source code necessary to reproduce the results presented in the ICLR 2018 [paper](https://openreview.net/pdf?id=ryup8-WCW):

```
@inproceedings{li_id_2018_ICLR
  title={Measuring the Intrinsic Dimension of Objective Landscapes},
  author={Li, Chunyuan and Farkhoor, Heerad and Liu, Rosanne and Yosinski, Jason},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## Contents
There are four steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#Dependencies)
2. [Prepare datasets](#Prepare-datasets)
3. [Subspace training](#Subspace-training)
    1. [Subspace training on image classification tasks](#Subspace-training-on-image-classification-tasks)
    2. [Subspace training on reinforcement learning tasks](#Subspace-training-on-reinforcement-learning-tasks)
    3. [Subspace training of ImageNet classification in distributed GPUs](#Subspace-training-of-ImageNet-classification-in-distributed-GPUs)
4. [Collect and plot results](#Collect-and-plot-results)


## Dependencies

This code is based on Python 2.7, with the main dependencies: [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/)
 
 * To run experiments: tensorflow-gpu, keras, numpy, h5py, IPython, colorama, scikit-learn. See "requirements.txt"
 
## Prepare datasets

We consider the following datasets: MNIST (_Standard_, _Shuffled-Pixel_ and _Shuffled-Label_ versions), CIFAR-10, and ImageNet.

All data are prepared in hdf5 format, with `train.h5` and `test.h5` representing separate sets of training and test. Each `.h5` file has the same fields 'images' and 'labels'.

Datasets can be downloaded [here](https://drive.google.com/open?id=1Cjky0VL6hFGppPhw3w4S-U4Ns0DAKa5T) (zip version is 347 MB, and the full size is 1.5G).

Put the downloaded and unzipped data in any directory and supply the relative path to `*.h5` to python script when executing (see [Train models](#Train-models) for examples).

## Subspace training

We construct custom keras layers for the special projection from subspace to full parameter space. The custom random projection layers (layer objects starting with `RProj`) are in `./keras-ext/` and used in various `model_builder` files in `./intrinsic_dim/`. The main `train.py` script conducts the training loop with the following options taken as arguments:  

- The `first two arguments` specify the paths for training and validation sets (_hdf5_ file), respectively; these arguments are required
- `--vsize`: subspace dimension, i.e., number of trainable parameters in the low-dimensional space
- `--epochs`: shortened as `-E`, number of training epochs (type=int); default 5
- `--opt`: optimization method to be used: e.g. `adam` (tf.train.AdamOptimizer) or `sgd` (tf.train.MomentumOptimizer); default `sgd`
- `--lr`: learning rate; default=.001
- `--l2`, L2 regularization to apply to direct parameters (type=float); default=0.0,
- `--arch`, which architecture to use from `arch_choices` (type=str), default=arch_choices[0]. Example architecture choices for direct training include 'mnistfc_dir', 'cifarfc_dir', 'mnistlenet_dir', 'cifarlenet_dir'; Example architecture choices for subspace training include 'mnistfc', 'cifarfc', 'mnistlenet',  'cifarlenet'                   
- `--output`: directory to save network checkpoints, tfevent files, etc.
- `projection type`: one and only one of three methods has to be specified to generate the randhom projection matrix . {`--denseproj`, `--sparseproj`, `--fastfoodproj`}
- `--depth` and `--width`, Hyperparameters of the fully connected networks: the number and width of layers in FC networks; default: depth=2 and width=200
- `--minibatch`: shortened as `-mb`, batch size for training; default 128
- `--d_rate`, Dropout rate to apply to direct parameters (type=float); default=0.0,
- `--c1`, `--c2`, `--d1` and `--d2`: Hyperparameters of LeNet: number of channels in the first/second conv layer, and width in firse/second in the dense layer; default: c1=6, c2=16, d1=120, d2=84

For more options, please see [`standard_parser.py`](./intrinsic_dim/standard_parser.py) and [`train.py`](./intrinsic_dim/train.py).

**1. Subspace training on image classification tasks**

First, to run direct training in the full parameter space as the baseline, select an architecture with `_dir` and do not add projection type. For example, to train a MNIST MLP network 784-200-200-10 (full parameter size: 986,643):
```
python ./train.py path-to-mnist-data/train.h5 path-to-mnist-data/test.h5 
    -E 100 --opt adam --lr 0.001 --l2 1e-05 --arch mnistfc_dir --depth 2 --width 200
```

To train the same network, but in a subspace of 1000, with various projection methods:
```
python ./train.py path-to-mnist-data/train.h5 path-to-mnist-data/test.h5 
    -E 100 --opt adam --lr 0.001 --l2 1e-05 --arch mnistfc --depth 2 --width 200 
    --vsize 1000 -fastfoodproj

python ./train.py path-to-mnist-data/train.h5 path-to-mnist-data/test.h5 
    -E 100 --opt adam --lr 0.001 --l2 1e-05 --arch mnistfc --depth 2 --width 200 
    --vsize 1000 -denseproj

python ./train.py path-to-mnist-data/train.h5 path-to-mnist-data/test.h5 
    -E 100 --opt adam --lr 0.001 --l2 1e-05 --arch mnistfc --depth 2 --width 200 
    --vsize 1000 -sparseproj
```

To further explore the toy problem in Section 2, you can check out the file [`train_toy.py`](./intrinsic_dim/train_toy.py).


**2. Subspace training on reinforcement learning tasks**

For example, to train 2-layer FC with width 200 on CartPole and subspace dimension as 20,
```
python ./train_dqn_general.py --vsize 20 --opt adam --lr 0.0001 --l2 0.0001 
--env_name 'CartPole-v0' --arch fc --width 200 --output results/rl_results/fnn_cartpole
```


**3. Subspace training of ImageNet classification in distributed GPUs**

An easy adoption of the software package [horovod](https://github.com/uber/horovod) allows for distributed training on many GPUs, which is helpful for large scale tasks like ImageNet. See [`train_distributed.py`](./intrinsic_dim/train_distributed.py) for details and for an impression how little the incurred changes are from `train.py`.

Follow horovod documentations for MPI and NCLL setup. Once they are, the script is executed like this:
```
mpirun -np 4 ./train_distrbuted.py path-to-imagenet-data/train.h5 path-to-imagenet-data/test.h5 -E 100
```

## Collect and plot results

Once the networks are trained and the results are saved, we extracted key results using Python script. We scan the performnace across different subspace dimensions, find the intrinsic dimension and plot the results.

The results can be plotted using the included IPython notebook `plots/main_plots.ipynb`.
Start the IPython Notebook server:

```
$ cd plots
$ ipython notebook
```

Select the `main_plots.ipynb` notebook and execute the included
code. Note that without modification, we have copyed our extracted results into the notebook, and script will output figures in the paper. If you've run your own training and wish to plot results, you have to orgnize your results in the same format instead.

_Shortcut: to skip all the work and just see the results, take a look at [this notebook with cached plots](plots/main_plots.ipynb)._


## Questions?

Please drop us ([Chunyuan](http://chunyuan.li/), [Rosanne](http://users.eecs.northwestern.edu/~rll943/) or [Jason](http://yosinski.com/)) a line if you have any questions.

