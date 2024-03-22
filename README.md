# Deep Global Registration

## Introduction
This repository contains python scripts for training and testing [Deep Global Registration, CVPR 2020 Oral](https://node1.chrischoy.org/data/publications/dgr/DGR.pdf).
Deep Global Registration (DGR) proposes a differentiable framework for pairwise registration of real-world 3D scans. DGR consists of the following three modules:

- a 6-dimensional convolutional network for correspondence confidence prediction
- 用于对应置信度预测的 6 维卷积网络
- a differentiable Weighted Procrustes algorithm for closed-form pose estimation
- 用于闭合形式姿态估计的可微加权 Procrustes 算法
- a robust gradient-based SE(3) optimizer for pose refinement.
- 一个强大的基于梯度的 SE(3) 优化器，用于姿态细化

For more details, please check out

- [CVPR 2020 oral paper](https://node1.chrischoy.org/data/publications/dgr/DGR.pdf)
- [1min oral video](https://youtu.be/stzgn6DkozA)
- [Full CVPR oral presentation](https://youtu.be/Iy17wvo07BU)

[![1min oral](assets/1min-oral.png)](https://youtu.be/stzgn6DkozA)


## Quick Pipleine Visualization
| Indoor 3DMatch Registration | Outdoor KITTI Lidar Registration |
|:---------------------------:|:---------------------------:|
| ![](https://chrischoy.github.io/images/publication/dgr/text_100.gif) | ![](https://chrischoy.github.io/images/publication/dgr/kitti1_optimized.gif) |

## Related Works
Recent end-to-end frameworks combine feature learning and pose optimization. PointNetLK combines PointNet global features with an iterative pose optimization method. Wang et al. in Deep Closest Point train graph neural network features by backpropagating through pose optimization.

最近的端到端框架结合了特征学习和姿势优化。 PointNetLK 将 PointNet 全局特征与迭代姿态优化方法相结合。 王等人在 Deep Closest Point 中，通过姿势优化反向传播来训练图神经网络特征。

We further advance this line of work. In particular, our Weighted Procrustes method reduces the complexity of optimization from quadratic to linear and enables the use of dense correspondences for highly accurate registration of real-world scans.

我们进一步推进这方面的工作。 特别是，我们的加权 Procrustes 方法将优化的复杂性从二次优化降低到线性优化，并能够使用密集对应来实现真实世界扫描的高精度配准。

## Deep Global Registration
The first component is a 6-dimensional convolutional network that analyzes the geometry of 3D correspondences and estimates their accuracy. Please refer to [High-dim ConvNets, CVPR'20](https://github.com/chrischoy/HighDimConvNets) for more details.

第一个组件是 6 维卷积网络，用于分析 3D 对应的几何形状并估计其准确性。

The second component we develop is a differentiable Weighted Procrustes solver. The Procrustes method provides a closed-form solution for rigid registration in SE(3). A differentiable version of the Procrustes method used for end-to-end registration passes gradients through coordinates, which requires O(N^2) time and memory for N keypoints. Instead, the Weighted Procrustes method passes gradients through the weights associated with correspondences rather than correspondence coordinates.

我们开发的第二个组件是可微的加权 Procrustes 求解器。 Procrustes 方法为 SE(3) 中的刚性配准提供了封闭式解决方案。 用于端到端配准的 Procrustes 方法的可微分版本通过坐标传递梯度，这需要 O(N^2) 时间和 N 个关键点的内存。 相反，加权 Procrustes 方法通过与对应而不是对应坐标关联的权重传递梯度。

The computational complexity of the Weighted Procrustes method is linear to the number of correspondences, allowing the registration pipeline to use dense correspondence sets rather than sparse keypoints. This substantially increases registration accuracy.

加权 Procrustes 方法的计算复杂度与对应的数量成线性关系，允许配准管道使用密集的对应集而不是稀疏的关键点。 这大大提高了配准准确性。

Our third component is a robust optimization module that fine-tunes the alignment produced by the Weighted Procrustes solver and the failure detection module.

我们的第三个组件是一个强大的优化模块，可以微调加权 Procrustes 求解器和故障检测模块产生的对齐。

This optimization module minimizes a differentiable loss via gradient descent on the continuous SE(3) representation space. The optimization is fast since it does not require neighbor search in the inner loop such as ICP.

该优化模块通过连续 SE(3) 表示空间上的梯度下降最小化可微损失。 优化速度很快，因为它不需要像 ICP 这样的内循环中的邻居搜索。

## Configuration
Our network is built on the [MinkowskiEngine](https://github.com/StanfordVL/MinkowskiEngine) and the system requirements are:

- Ubuntu 14.04 or higher
- <b>CUDA 10.1.243 or higher</b>
- pytorch 1.5 or higher
- python 3.6 or higher
- GCC 7

You can install the MinkowskiEngine and the python requirements on your system with:

```shell
# Install MinkowskiEngine
sudo apt install libopenblas-dev g++-7
pip install torch
export CXX=g++-7; pip install -U MinkowskiEngine --install-option="--blas=openblas" -v

# Download and setup DeepGlobalRegistration
git clone https://github.com/chrischoy/DeepGlobalRegistration.git
cd DeepGlobalRegistration
pip install -r requirements.txt
```

## Demo
You may register your own data with relevant pretrained DGR models. 3DMatch is suitable for indoor RGB-D scans; KITTI is for outdoor LiDAR scans.

您可以用相关的预训练 DGR 模型配准自己的数据。 3DMatch适用于室内RGB-D扫描； **KITTI 用于室外 LiDAR 扫描**。

| Inlier Model | FCGF model  | Dataset | Voxel Size    | Feature Dimension | Performance                | Link   |
|:------------:|:-----------:|:-------:|:-------------:|:-----------------:|:--------------------------:|:------:|
| ResUNetBN2C  | ResUNetBN2C | 3DMatch | 5cm   (0.05)  | 32                | TE: 7.34cm, RE: 2.43deg    | [weights](http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-3dmatch-v0.05.pth) |
| ResUNetBN2C  | ResUNetBN2C | KITTI   | 30cm  (0.3)   | 32                | TE: 3.14cm, RE: 0.14deg    | [weights](http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-kitti-v0.3.pth) |


```shell
python demo.py
```

| Input PointClouds           | Output Prediction           |
|:---------------------------:|:---------------------------:|
| ![](assets/demo_inputs.png) | ![](assets/demo_outputs.png) |


## Experiments
| Comparison | Speed vs. Recall Pareto Frontier |
| -------  | --------------- |
| ![Comparison](assets/comparison-3dmatch.png) | ![Frontier](assets/frontier.png) |


## Training
The entire network depends on pretrained [FCGF models](https://github.com/chrischoy/FCGF#model-zoo). Please download corresponding models before training.

整个网络依赖于预训练的[FCGF模型](https://github.com/chrischoy/FCGF#model-zoo)。 请在训练前下载相应的模型。

| Model       | Normalized Feature  | Dataset | Voxel Size    | Feature Dimension |                  Link   |
|:-----------:|:-------------------:|:-------:|:-------------:|:-----------------:|:------:|
| ResUNetBN2C | True                | 3DMatch | 5cm   (0.05)  | 32                     | [download](https://node1.chrischoy.org/data/publications/fcgf/2019-08-16_19-21-47.pth) |
| ResUNetBN2C | True                | KITTI   | 30cm  (0.3)   | 32                 | [download](https://node1.chrischoy.org/data/publications/fcgf/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth) |


### 3DMatch
You may download preprocessed data and train via these commands:

您可以下载预处理数据并通过以下命令进行训练：
```shell
./scripts/download_3dmatch.sh /path/to/3dmatch
export THREED_MATCH_DIR=/path/to/3dmatch; FCGF_WEIGHTS=/path/to/fcgf_3dmatch.pth ./scripts/train_3dmatch.sh
```

### KITTI
Follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry train set. Then train with

按照[KITTI Odometry网站](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)上的说明下载KITTI里程计训练集。 然后训练
```shell
export KITTI_PATH=/path/to/kitti; FCGF_WEIGHTS=/path/to/fcgf_kitti.pth ./scripts/train_kitti.sh
```

## Testing
3DMatch test set is different from train set and is available at the [download section](http://3dmatch.cs.princeton.edu/) of the official website. You may download and decompress these scenes to a new folder.

3DMatch测试集与训练集不同，可以在官网的[下载部分](http://3dmatch.cs.princeton.edu/)获得。 您可以下载这些场景并将其解压到新文件夹中。

To evaluate trained model on 3DMatch or KITTI, you may use

要在 3DMatch 或 KITTI 上评估经过训练的模型，您可以使用
```shell
python -m scripts.test_3dmatch --threed_match_dir /path/to/3dmatch_test/ --weights /path/to/dgr_3dmatch.pth
```
and
```shell
python -m scripts.test_kitti --kitti_dir /path/to/kitti/ --weights /path/to/dgr_kitti.pth
```

## Generate figures
We also provide experimental results of 3DMatch comparisons in `results.npz`. To reproduce figures we presented in the paper, you may use

我们还在“results.npz”中提供了 3DMatch 比较的实验结果。 要重现我们在论文中提供的数字，您可以使用
```shell
python scripts/analyze_stats.py assets/results.npz
```

## Citing our work
Please cite the following papers if you use our code:

```latex
@inproceedings{choy2020deep,
  title={Deep Global Registration},
  author={Choy, Christopher and Dong, Wei and Koltun, Vladlen},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{choy2019fully,
  title = {Fully Convolutional Geometric Features},
  author = {Choy, Christopher and Park, Jaesik and Koltun, Vladlen},
  booktitle = {ICCV},
  year = {2019}
}

@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={CVPR},
  year={2019}
}
```

## Concurrent Works

There have a number of 3D registration works published concurrently.

- Gojcic et al., [Learning Multiview 3D Point Cloud Registration, CVPR'20](https://github.com/zgojcic/3D_multiview_reg)
- Wang et al., [PRNet: Self-Supervised Learning for Partial-to-Partial Registration, NeurIPS'19](https://github.com/WangYueFt/prnet)
- Yang et al., [TEASER: Fast and Certifiable Point Cloud Registration, arXiv'20](https://github.com/MIT-SPARK/TEASER-plusplus)
