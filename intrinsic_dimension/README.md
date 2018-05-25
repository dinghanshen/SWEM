## Measuring the Intrinsic Dimension of SWEMs


## Contents
There are four steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#Dependencies)
2. [Prepare datasets](#Prepare-datasets)
3. [Subspace training](#Subspace-training)
4. [Collect and plot results](#Collect-and-plot-results)


## Dependencies

This code is based on Python 2.7, with the main dependencies: [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/)
 
 * To run experiments: tensorflow-gpu, keras, numpy, h5py, IPython, colorama, scikit-learn. See "requirements.txt"
 
## Prepare datasets

We consider the following datasets: _AG News_  and _Yelp P._ datasets. Similar data format is described in the [main page](https://github.com/dinghanshen/SWEM).

Datasets can be downloaded [here](https://drive.google.com/open?id=1DB-jnR0fF0vsdCEJqpU3nCDASPfDxHCN) (the total size is around 800MB).

Put the downloaded and unzipped data in the folder `dataset` and supply the relative path to python script when executing (see [Train models](#Train-models) for examples).

## Subspace training

We construct custom keras layers for the special projection from subspace to full parameter space. The custom random projection layers (layer objects starting with `RProj`) are in `./keras-ext/` and used in various `model_builder` files in `./intrinsic_dim/`. The main `train.py` script conducts the training loop with the following options taken as arguments:  

- `--dataset` specify the dataset ('agnews' or 'yelp'); these arguments are required
- `--vsize`: subspace dimension, i.e., number of trainable parameters in the low-dimensional space
- `--epochs`: shortened as `-E`, number of training epochs (type=int); default 5
- `--opt`: optimization method to be used: e.g. `adam` (tf.train.AdamOptimizer) or `sgd` (tf.train.MomentumOptimizer); default `sgd`
- `--lr`: learning rate; default=.001
- `--l2`, L2 regularization to apply to direct parameters (type=float); default=0.0,
- `--arch`, which architecture to use from `arch_choices` (type=str), default=arch_choices[0]. Example architecture choices for direct training include 'mnistfc_dir', 'cifarfc_dir', 'mnistlenet_dir', 'cifarlenet_dir'; Example architecture choices for subspace training include 'mnistfc', 'cifarfc', 'mnistlenet',  'cifarlenet'                   
- `--output`: directory to save network checkpoints, tfevent files, etc.
- `projection type`: one and only one of three methods has to be specified to generate the randhom projection matrix . {`--denseproj`, `--sparseproj`, `--fastfoodproj`}
- `--depth` and `--width`, Hyperparameters of the fully connected networks after SWEM layer or CNN layer: the number and width of layers in FC networks; default: depth=1 and width=200
- `--minibatch`: shortened as `-mb`, batch size for training; default 128


For more options, please see [`standard_parser.py`](./intrinsic_dim/standard_parser.py) and [`train.py`](./intrinsic_dim/train.py).

**Subspace training on the AG News task**

First, to run direct training in the full parameter space as the baseline, select an architecture with `_dir` and do not add projection type. For example, to train a SWEM model (followed by one layer MLP, please set --depth to 0 if no MLP is needed):
```
python --dataset agnews -E 10 --opt adam --lr 0.001 --l2 1e-05 --arch swem_dir --depth 1 --width 200
```

To train the same network, but in a subspace of 100, with fastfood projection method:
```
python --dataset agnews -E 10 --opt adam --lr 0.001 --l2 1e-05 --arch swem --depth 1 --width 200 --vsize 100 --sparseproj
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

_Shortcut: to skip all the work and just see the results, take a look at [this notebook with cached plots](./intrinsic_dim/plots/misc_plots.ipynb).


## Questions?

Please drop us ([Chunyuan](http://chunyuan.li/) or [Dinghan](https://sites.google.com/view/dinghanshen/)) a line if you have any questions on the subspace training. For more information on implementing subspace training, please see its [github repo](https://github.com/uber-research/intrinsic-dimension).  

