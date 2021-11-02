# Point-Voxel CNN TensorFlow
This repo is a TensorFlow 2 Implementation of Point-Voxel CNN for Efficient 3D Deep Learning (see [arXiv paper](https://arxiv.org/abs/1907.03739), [MIT HAN Lab Repo](https://github.com/mit-han-lab/pvcnn)). Development is currently in progress!

# Prerequisites
TODO

# Data Preparation
## S3DIS
I re-use the data pre-processing used in [PVCNN](https://github.com/mit-han-lab/pvcnn) (see [`data/s3dis/`](data/s3dis/prepare_data.py)). One should first download the [S3DIS dataset from here](http://buildingparser.stanford.edu/dataset.html), then run
```
python data/s3dis/prepare_data.py -d [path to unzipped dataset dir]
```

You can run [`s3dis_viz.py`](https://github.com/zghera/pvcnn-tf/blob/master/s3dis_viz.py) for a visualizaion of the dataset. Here is one example output with clutter, ceiling, floor, and wall points removed:
<p align="center"><img src="/assets/s3dis-data-pipeline-output.png" alt="s3dis-data-pipeline-output" width="500"/></p>

# Performance of Pretrained Models
TODO

# Evaluating Pretrained Models
TODO

# Training
TODO

# License 
This repository is released under the MIT license. This includes the [license](https://github.com/mit-han-lab/pvcnn/blob/master/LICENSE) from the original authors. See [LICENSE](https://github.com/zghera/pvcnn-tf/blob/master/LICENSE) for additional details.

# Acknowledgement
The following modules / code-segments were adapted from [the official PVCNN implementation](https://github.com/mit-han-lab/pvcnn) (MIT License):
* [Data pre-processing.](https://github.com/zghera/pvcnn-tf/blob/master/data/s3dis/prepare_data.py)
* [Training and evaluation.](https://github.com/zghera/pvcnn-tf/blob/master/train.py)
* [Experiment configuration.](https://github.com/zghera/pvcnn-tf/blob/master/utils/config.py)
