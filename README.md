# Code of Paper Learning to Generate Parameters of ConvNets for Unseen Image Data

Official implementation for paper [Learning to Generate Parameters of ConvNets for Unseen Image Data](https://arxiv.org/abs/2310.11862).

## Abstract
_Typical Convolutional Neural Networks (ConvNets) depend heavily on large amounts of image data and resort to an iterative optimization algorithm (e.g., SGD or Adam) to learn network parameters, which makes training very time- and resource-intensive. In this paper, we propose a new training paradigm and formulate the parameter learning of ConvNets into a prediction task: considering that there exists correlations between image datasets and their corresponding optimal network parameters of a given ConvNet, we explore if we can learn a hyper-mapping between them to capture the relations, such that we can directly predict the parameters of the network for an image dataset never seen during the training phase. To do this, we put forward a new hypernetwork based model, called PudNet, which intends to learn a mapping between datasets and their corresponding network parameters, and then predicts parameters for unseen data with only a single forward  propagation. Moreover, our model benefits from a series of adaptive hyper recurrent units sharing weights to capture the dependencies of parameters among different network layers. Extensive experiments demonstrate that our proposed method achieves good efficacy for unseen image datasets on two kinds of settings: Intra-dataset prediction and Inter-dataset prediction. Our PudNet can also well scale up to large-scale datasets, e.g., ImageNet-1K. It takes 8967 GPU seconds to train ResNet-18 on the ImageNet-1K using GC from scratch and obtain a top-5 accuracy of 44.65%. However, our PudNet costs only 3.89 GPU seconds to predict the network parameters of ResNet-18 achieving comparable performance (44.92%), more than 2,300 times faster than the traditional training paradigm._


## Requirement
- python3.6.5
- Pytorch0.4.1
- numpy
- scipy
- sklearn

## Files 
    --Incremental MNIST: Incremental MNIST dataset,including a training file and 4 testing files, which named as ldx_t.mat, lux_t.mat, rdx_t.mat and rux_t.mat.
    --Draw.py: Draw the accuracy performance curve.
    --xxx:.

## Basic Usage  
    There are various parameters in the input structure paras:

    --alp : Percentage of Fisher Information accumulated during Backpropagation.
    --xx:xx.

## Quick Start
```
python PudNet.py 
```
Parameters/options can be tuned to get better results.

## Citation 
Please cite our work if you feel the paper or the code are helpful.

```
@article{wang2023learning,
  author={Shiye Wang and Kaituo Feng and Changsheng Li and Ye Yuan and Guoren Wang},
  title={Learning to Generate Parameters of ConvNets for Unseen Image Data},
  journal={CoRR},
  volume={abs/2310.11862},
  year={2023}
}
```

## Contact 
If there are any questions, please feel free to contact with the authors:  Shiye Wang (Shiye Wang@bit.edu.cn). Enjoy the code.