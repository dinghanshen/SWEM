# SWEM (Simple Word-Embedding-based Models)

This repository contains source code necessary to reproduce the results presented in the following paper:
* [*Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms*](https://arxiv.org/abs/1805.09843) (ACL 2018)

This project is maintained by [Dinghan Shen](https://sites.google.com/view/dinghanshen). Feel free to contact dinghan.shen@duke.edu for any relevant issues.

## Prerequisite: 
* CUDA, cudnn
* Python 2.7
* Tensorflow (version >1.0). We used tensorflow 1.5.
* Run: `pip install -r requirements.txt` to install requirements

## Data: 
* For convenience, we provide pre-processed versions for the following datasets: DBpedia, SNLI, Yahoo. Data are prepared in pickle format, and each `.p` file has the same fields in the same order: 
	* `train_text`, `val_text`, `test_text`, `train_label`, `val_label`, `test_label`, `dictionary(wordtoix)`, `reverse dictionary(ixtoword)`

* These `.p` files can be downloaded from the links below. After downloading, you can put them into a `data` folder:

 	* Ontology classification: [DBpedia (591MB)](https://drive.google.com/open?id=1EBmMise0LQu0QpO7T4a32WMFuTxAb6T0)

 	* Natural language inference: [SNLI (101MB)](https://drive.google.com/open?id=1M13UswHThZYt-ARrHg6sN7Dlel-d6BB3)

 	* Topic categorization: [Yahoo (1.7GB)](https://drive.google.com/open?id=1Dorz_CWZkHHpojVS4K4YUEhhczVLQgRc)

## Run 
* Run: `python eval_dbpedia_emb.py` for ontology classification on the DBpedia dataset
* Run: `python eval_snli_emb.py` for natural language inference on the SNLI dataset
* Run: `python eval_yahoo_emb.py` for topic categorization on the Yahoo! Answer dataset

* Options: options can be made by changing `option` class in any of the above three files: 
- `opt.emb_size`: number of word embedding dimensions.
- `opt.drop_rate`: the keep rate of dropout layer.
- `opt.lr`: learning rate.
- `opt.batch_size`: number of batch size.
- `opt.H_dis`: the dimension of last hidden layer.

* On a K80 GPU machine, training roughly takes about 3 minutes each epoch and 5 epochs for Debpedia to converge, 50 seconds each epoch and 20 epochs for SNLI, and 4 minutes each epoch and 5 epochs for the Yahoo dataset.

## Subspace Training & Intrinsic Dimension
To measure the [*intrinsic dimension*](https://eng.uber.com/intrinsic-dimension/) of word-embedding-based text classification tasks, we compare SWEM and CNNs via subspace training in Section 5.1 of the paper. 

Please follow the instructions in folder [`intrinsic_dimension`](./intrinsic_dimension) to reproduce the results.

## Citation 
Please cite our ACL paper in your publications if it helps your research:

```latex
@inproceedings{Shen2018Baseline, 
title={Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms}, 
author={Shen, Dinghan and Wang, Guoyin and Wang, Wenlin and Renqiang Min, Martin and Su, Qinliang and Zhang, Yizhe and Li, Chunyuan and Henao, Ricardo and Carin, Lawrence}, 
booktitle={ACL}, 
year={2018} 
}
```

