Vision-odometry with moving object removal based on SfMLearner

## Contributions
  1. A geometry consistency loss, which makes the predicted depths to be globally scale consistent.
  2. A self-discovered mask, which detects moving objects and occlusions for boosting accuracy.
  3. Scale-consistent predictions, which can be used in the Monocular Visual SLAM system.



## Preamble
This codebase was developed and tested with python 3.6, Pytorch 1.0.1, and CUDA 10.0 on Ubuntu 16.04. It is based on [Clement Pinard's SfMLearner implementation](https://github.com/ClementPinard/SfmLearner-Pytorch).


## Prerequisite

```bash
pip3 install -r requirements.txt
```

## Datasets

See "scripts/run_prepare_data.sh".

    For KITTI Raw dataset, download the dataset using this script http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website.

    For KITTI Odometry dataset, download the dataset with color images.

Or you can download our pre-processed dataset from the following link

[kitti_256 (for kitti raw)](https://1drv.ms/u/s!AiV6XqkxJHE2g1zyXt4mCKNbpdiw?e=ZJAhIl) | [kitti_vo_256 (for kitti odom)](https://1drv.ms/u/s!AiV6XqkxJHE2k3YRX5Z8c_i7U5x7?e=ogw0c7) | [kitti_depth_test (eigen split)](https://1drv.ms/u/s!AiV6XqkxJHE2kz5Zy7jWZd2GyMR2?e=kBD4lb) | [kitti_vo_test (seqs 09-10)](https://1drv.ms/u/s!AiV6XqkxJHE2k0BSVZE-AJNvye9f?e=ztiSWp)


## Training

The "scripts" folder provides several examples for training and testing.

You can train the depth model on KITTI Raw by running
```bash
sh scripts/train_resnet18_depth_256.sh
```
or train the pose model on KITTI Odometry by running
```bash
sh scripts/train_resnet50_pose_256.sh
```
Then you can start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. 



## Evaluation

You can evaluate depth on Eigen's split by running
```bash
sh scripts/test_kitti_depth.sh
```
evaluate visual odometry by running
```bash
sh scripts/test_kitti_vo.sh
```
and visualize depth by running
```bash
sh scripts/run_inference.sh
```

## Pretrained Models

[Latest Models](https://1drv.ms/u/s!AiV6XqkxJHE2kxX_Gek5fEQvMGma?e=ZfrnbR)

To evaluate the [NeurIPS models](https://1drv.ms/u/s!AiV6XqkxJHE2kxSHVMYvo7DmGqNb?e=bg3tWg), please download the code from 'Release/NeurIPS version'.


## Depth Results 

#### KITTI raw dataset (Eigen's splits)

|   Models   | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|------------|---------|--------|-------|-----------|-------|-------|-------|
| resnet18   | 0.119   | 0.857  | 4.950 | 0.197     | 0.863 | 0.957 | 0.981 |
| resnet50   | 0.114   | 0.813  | 4.706 | 0.191     | 0.873 | 0.960 | 0.982 |


#### NYUv2 dataset (Original Video)

|   Models   | Abs Rel | Log10  | RMSE  | Acc.1 | Acc.2 | Acc.3 |
|------------|---------|--------|-------|-------|-------|-------|
| resnet18   | 0.159   | 0.068  | 0.608 | 0.772 | 0.939 | 0.982 |
| resnet50   | 0.157   | 0.067  | 0.593 | 0.780 | 0.940 | 0.984 |


#### NYUv2 dataset (Rectifed Images by [Unsupervised-Indoor-Depth](https://github.com/JiawangBian/Unsupervised-Indoor-Depth))

|   Models   | Abs Rel | Log10  | RMSE  | Acc.1 | Acc.2 | Acc.3 |
|------------|---------|--------|-------|-------|-------|-------|
| resnet18   | 0.143   | 0.060  | 0.538 | 0.812 | 0.951 | 0.986 |
| resnet50   | 0.142   | 0.060  | 0.529 | 0.813 | 0.952 | 0.987 |



## Visual Odometry Results on KITTI odometry dataset 

#### Network prediction (trained on 00-08)

|Metric               | Seq. 09 | Seq. 10 |
|---------------------|---------|---------|
|t_err (%)            | 7.31    | 7.79    |
|r_err (degree/100m)  | 3.05    | 4.90    | 

#### Pseudo-RGBD SLAM output (Integration of SC-Depth in ORB-SLAM2)

|Metric               | Seq. 09 | Seq. 10 |
|---------------------|---------|---------|
|t_err (%)            | 5.08    | 4.32    |
|r_err (degree/100m)  | 1.05    | 2.34    | 



    
 ## Related projects
 
 * [SfMLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) (CVPR 2017, our baseline framework.)

 * [Depth-VO-Feat](https://github.com/Huangying-Zhan/Depth-VO-Feat) (CVPR 2018, trained on stereo videos for depth and visual odometry)
 
 * [DF-VO](https://github.com/Huangying-Zhan/DF-VO) (ICRA 2020, use scale-consistent depth with optical flow for more accurate visual odometry)
 
 * [Kitti-Odom-Eval-Python](https://github.com/Huangying-Zhan/kitti-odom-eval) (python code for kitti odometry evaluation)
 
 * [Unsupervised-Indoor-Depth](https://github.com/JiawangBian/Unsupervised-Indoor-Depth) (Using SC-SfMLearner in NYUv2 dataset)
 
