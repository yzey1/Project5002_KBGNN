# Project5002_KBGNN

This is an implementation for paper [Kernel-based Substructure Exploration for Next POI Recommendation](https://arxiv.org/abs/2210.03969).

We use Ubuntu 20.04, Python 3.8, PyTorch 1.11, RTX 3080 and CUDA 11.3 in our experiments.

`preprocess.py` is used to preprocess the dataset. `dataset.py` is used to load the dataset.

`SeqGraph.py` and `GeoGraph.py` are used to construct the sequential and geographical components of the model, respectively. `misc.py` contains some utility small modules. `consistency.py` is used to calculate the consistency loss of the model.

`main.py` is the main file for training the model. `ablation_geo.py` and `ablation_seq.py` are used for the ablation study on the geographical and sequential components, respectively.

## Preparation

Download the dataset from [here](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) and unzip it at the root of the project. The directory structure should look like this:

```
.
├── dataset_tsmc2014
│   ├── dataset_TSMC2014_NYC.txt
│   ├── dataset_TSMC2014_readme.txt
│   ├── dataset_TSMC2014_TKY.txt
├── preprocess.py
...
```

Run the following command to preprocess the dataset:

```bash
python preprocess.py
```

It may take tens of minutes. After preprocessing, you should see a new directory `processed_data/raw/` and a subdirectory `nyc` or `tky`, depending on the dataset you set in `preprocess.py`.

Under `nyc` or `tky`, you should see the following files:

```
.
├── dist_graph.pkl
├── dist_on_graph.npy
├── test.pkl
├── train.pkl
├── val.pkl
```

`dist_graph.pkl` is the graph structure of the dataset, containing edges and neighbors; `dist_on_graph.npy` is the distance corresponding to the edges; `train.pkl`, `val.pkl`, and `test.pkl` are the training, validation, and test sets, respectively.

## Training

Run the following command to train the model with default hyperparameters and GPU:

```bash
python main.py --gpu 0
```

To check the available hyperparameters, run:

```bash
python main.py --help
```

Replace `main.py` with `ablation_geo.py` or `ablation_seq.py` to run the ablation study on the geographical and sequential components, respectively.
