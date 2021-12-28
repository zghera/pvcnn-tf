# Point-Voxel CNN TensorFlow
This repo is a TensorFlow 2 Implementation of Point-Voxel CNN for Efficient 3D Deep Learning (see [arXiv paper](https://arxiv.org/abs/1907.03739), [MIT HAN Lab Repo](https://github.com/mit-han-lab/pvcnn)). 

Development is currently in progress!

# Prerequisites
This code is built on Google Colab (see [build.ipynb](build.ipynb)). The following libraries must be installed in the Colab environment:
- Python >= 3.7
- [TensorFlow](https://github.com/tensorflow/tensorflow) == 2.6.2
- [TensorFlow I/O](https://github.com/tensorflow/io) == 0.21
- [numpy](https://github.com/numpy/numpy)
- [tqdm](https://github.com/tqdm/tqdm)
- [plyfile](https://github.com/dranjan/python-plyfile)
- [h5py](https://github.com/h5py/h5py)

# Data Preparation
## S3DIS
I re-use the data pre-processing used in [PVCNN](https://github.com/mit-han-lab/pvcnn) (see [`data/s3dis/`](data/s3dis/prepare_data.py)). One should first download the [S3DIS dataset from here](http://buildingparser.stanford.edu/dataset.html), then run
```
python data/s3dis/prepare_data.py -d [path to unzipped dataset dir]
```

You can run [`s3dis_viz.py`](https://github.com/zghera/pvcnn-tf/blob/master/s3dis_viz.py) for a visualizaion of the dataset. Here is one example output with clutter, ceiling, floor, and wall points removed:
<p align="center"><img src="/assets/s3dis-data-pipeline-output.png" alt="s3dis-data-pipeline-output" width="600"/></p>

# Performance of Pretrained Models
This project is still a work in progress. Numerical instabilities while training have impeded full training of the model to obtain performance results. However, the approximate 4x reduction in loss and 35% mean IoU accuracy acheived in the first 2500 iterations of the first epoch (see Figure below) suggests that the model was in fact learning up until crashing (i.e. NaN tensors).

<p align="center"><img src="/assets/train-metrics-vs-epoch.png" alt="s3dis-data-pipeline-output" width="600"/></p>

For more details, please see the "Experiment Results and Dicussion" section of the [final paper](/assets/ECE_570_Final_Paper__Point_Voxel_CNN_for_Efficient_3D_Deep_Learning_in_TensorFlow.pdf) associated with this project.

# Evaluating Pretrained Models
In Progress.

# Training
In Progress.

# License 
This repository is released under the MIT license. This includes the [license](https://github.com/mit-han-lab/pvcnn/blob/master/LICENSE) from the original authors. See [LICENSE](https://github.com/zghera/pvcnn-tf/blob/master/LICENSE) for additional details.

# Acknowledgement
The following modules / code-segments were adapted from [the official PVCNN implementation](https://github.com/mit-han-lab/pvcnn) (MIT License):
* [Data pre-processing.](data/s3dis/prepare_data.py)
* [Experiment configuration.](utils/config.py)
* [Utility functions used in training (`get_configs` and `get_save_path`)](utils/common.py)
