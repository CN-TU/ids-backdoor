# EagerNet
Contact: Fares Meghdouri, Maximilian Bachl

This repository contains the code, the data, the plots and the tex files for our paper **EagerNet: Early Predictions of Neural Networks for Computationally Efficient Intrusion Detection** (2020 4th Cyber Security in Networking Conference (CSNet))

# Datasets
We use two Intrusion Detection Datasets: [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/).
The raw PCAP files are converted into CAIA format using [go-flows](https://github.com/CN-TU/go-flows) and data is afterwards cleaned and z-normalized.

Data is taken from [this](https://github.com/CN-TU/adversarial-recurrent-ids) repository. The instructions are also included there.


If you want to produce the preprocessed files yourself: 
* Follow the information on how to reproduce the preprocessed datasets in the [Datasets-preprocessing](https://github.com/CN-TU/Datasets-preprocessing) repository.

# Usage Examples
After unzipping the data, you have the following possibilities:

* Training a binary model with increasing loss weights
``` sh
./learn.py --dataroot CAIA_17.csv --function train_eager_stopping --method nn --lr 0.001 --batchSize 512 --eagerStoppingWeightingMethod "eager_linearly_increasing_weights" --nLayers 3 --layerSize 128

```

* Make predictions using a preloaded model, multiclass classification and only considering 10000 samples
``` sh
./learn.py --dataroot ../CAIA_15.csv --function predict_eager --method nn --net 'runs/Jul15_11-21-05_gpu_0_3/net_1606741938.pth' --batchSize 1 --layerSize 128 --nLayers 3 --multiclass --maxSize 10000
```

* Create confidence-accuracy figure from the paper for a pre-trained network

``` sh
./learn.py --dataroot ../CAIA_15.csv --function create_plot_eager --method nn --net 'runs/Jul14_11-32-30_gpu_0_3/net_1606741938.pth' --batchSize 1 --layerSize 128 --nLayers 3 --maxSize 10000
```

* Create the accuracy per layer and attack from the paper for a pre-trained model
``` sh
./learn.py --dataroot ../CAIA_15.csv --function create_plot_eager --method nn --net 'runs/Jul15_11-21-05_gpu_0_3/net_1606741938.pth' --multiclass --batchSize 1 --layerSize 128 --nLayers 3 --maxSize 10000
```

# Trained Models
Models in the [runs](runs) folder have been trained with the following configurations:

| Folder | Variant | Number of Layers | Neurons per Layer | Weights Distribution | Dataset | Used In Paper |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Jul14_11-20-26_gpu_0_3 | B  | 3  | 128  | Uniform |CICIDS2017|X|
| Jul14_11-25-05_gpu_0_3 | B  | 3  | 128  | Increasing |CICIDS2017|X|
| Jul14_11-28-08_gpu_0_3 | B  | 3  | 128  | Decreasing |CICIDS2017|X|
| Jul14_11-22-43_gpu_0_3 | B  | 1  | 128  | Uniform |CICIDS2017|X|
| Jul14_11-26-37_gpu_0_3 | B  | 1  | 128  | Increasing |CICIDS2017|X|
| Jul14_11-29-35_gpu_0_3 | B  | 1  | 128  | Decreasing |CICIDS2017|X|
| Jul15_11-15-43_gpu_0_3 | M  | 3  | 128  | Uniform |CICIDS2017||
|  / | M  | 3  | 128  | Increasing |CICIDS2017||
| Jul16_10-06-45_gpu_0_3 | M  | 3  | 128  | Decreasing |CICIDS2017||
| Jul15_11-18-58_gpu_0_3 | M  | 1  | 128  | Uniform |CICIDS2017||
| Jul16_10-04-09_gpu_0_3 | M  | 1  | 128  | Increasing |CICIDS2017||
| Jul17_09-58-11_gpu_0_3 | M  | 1  | 128  | Decreasing |CICIDS2017||
| Jul15_10-26-27_gpu_0_3 | B  | 3  | 128  | Uniform |UNSW-NB15|X|
| Jul15_10-30-44_gpu_0_3 | B  | 3  | 128  | Increasing |UNSW-NB15|X|
| Jul14_11-32-30_gpu_0_3 | B  | 3  | 128  | Decreasing |UNSW-NB15|X|
| Jul15_10-28-33_gpu_0_3 | B  | 1  | 128  | Uniform |UNSW-NB15|X|
| Jul15_10-32-55_gpu_0_3 | B  | 1  | 128  | Increasing |UNSW-NB15|X|
| Jul15_10-35-54_gpu_0_3 | B  | 1  | 128  | Decreasing |UNSW-NB15|X|
| Jul15_11-21-05_gpu_0_3 | M  | 3  | 128  | Uniform |UNSW-NB15||
| Jul16_09-56-00_gpu_0_3 | M  | 3  | 128  | Increasing |UNSW-NB15||
| Jul16_10-00-10_gpu_0_3 | M  | 3  | 128  | Decreasing |UNSW-NB15||
| Jul16_09-50-26_gpu_0_3 | M  | 1  | 128  | Uniform |UNSW-NB15||
| Jul16_09-53-29_gpu_0_3 | M  | 1  | 128  | Increasing |UNSW-NB15||
| Jul16_09-58-27_gpu_0_3 | M  | 1  | 128  | Decreasing |UNSW-NB15||
| Jul16_10-09-48_gpu_0_3 | B  | 10  | 64  | Increasing |CICIDS2017|X|
| Jul16_10-24-05_gpu_0_3 | B  | 10  | 64  | FNN  |CICIDS2017|X|
| Jul17_09-55-07_gpu_0_3 | M  | 10  | 64  | Increasing |CICIDS2017|X|
| Jul17_09-53-15_gpu_0_3 | M  | 10  | 64  | FNN  |CICIDS2017|X|
| Jul20_17-10-53_gpu_0_3 | B  | 10  | 64  | Increasing |UNSW-NB15|X|
| Jul20_17-11-13_gpu_0_3 | B  | 10  | 64  | FNN  |UNSW-NB15|X|
| Jul20_17-09-33_gpu_0_3 | M  | 10  | 64  | Increasing |UNSW-NB15|X|
| Jul20_17-09-53_gpu_0_3 | M  | 10  | 64  | FNN  |UNSW-NB15|X|
| Jul18_17-11-59_gpu_0_3 | B  | 10  | 64  | Min Loss |CICIDS2017||
| Jul19_11-49-25_gpu_0_3 | M  | 10  | 64  | Min Loss |CICIDS2017||
| Jul19_11-52-38_gpu_0_3 | B  | 10  | 64  | Min Loss |UNSW-NB15||
| Jul19_11-53-33_gpu_0_3 | M  | 10  | 64  | Min Loss |UNSW-NB15||
| Jul19_18-27-17_gpu_0_3 | B  | 10  | 32  | Min Loss + Decreasing |CICIDS2017||
| Jul22_15-45-52_gpu_0_3 | M  | 3  | 128  | Uniform |CICIDS2017|X|
| Jul22_15-49-44_gpu_0_3 | M  | 10  | 64  | Uniform |CICIDS2017|X|
| Jul22_15-50-59_gpu_0_3 | M  | 3  | 128  | Uniform |UNSW-NB15|X|
| Jul22_15-51-30_gpu_0_3 | M  | 10  | 64  | Uniform |UNSW-NB15|X|
| Jul09_14-57-35_gpu_0_3 |B|/| 512 | Uniform |CICIDS2017||
| Jul09_14-58-15_gpu_0_3 |B|/| 512 | Decreasing |CICIDS2017||
| Jul09_15-01-30_gpu_0_3 |B|/| 512 | Increasing |CICIDS2017||
| Jul09_19-09-12_gpu_0_3 |B|/| 512 | Logistic Reg.|CICIDS2017||
| Jul10_14-24-04_gpu_0_3 |B|/| 512 | Decreasing |UNSW-NB15||
| Jul10_14-28-38_gpu_0_3 |B|1| 512 | Decreasing |CICIDS2017||
| Jul10_14-54-16_gpu_0_3 |B|3| 128 | Decreasing |CICIDS2017||
| Jul10_15-09-38_gpu_0_3 |B|3| 128 | Decreasing |UNSW-NB15||

The name of the `.pth` model represents the number of samples reached: E.g. net_100000.pth means that 100,000 samples were used for training


