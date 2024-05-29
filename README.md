# Code of Paper Learning to Generate Parameters of ConvNets for Unseen Image Data

Official implementation for paper [Learning to Generate Parameters of ConvNets for Unseen Image Data](https://arxiv.org/abs/2310.11862).

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

## Contact 
If there are any questions, please feel free to contact with the authors:  Shiye Wang (zhoudw@lamda.nju.edu.cn) and Changsheng Li (yangy@lamda.nju.edu.cn). Enjoy the code.