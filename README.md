# InstaHide training on CIFAR-10 with PyTorch

## Overview
InstaHide[1] is a practical instance-hiding method for image data encryption in privacy-sensitive distributed deep learning.

InstaHide uses the Mixup[2] method with a one-time secret key consisting of a pixel-wise random sign-flipping mask and samples from the same training dataset (Inside-dataset InstaHide) or a large public dataset (Cross-dataset InstaHide). It can be easily plugged into an existing distributed learning pipeline, and is very efficient and incurs minor reduction on accuracy.

We also release a [challenge](https://github.com/Hazelsuko07/InstaHide_Challenge) to further investigate the security of InstaHide.


## Citation
If you use InstaHide or this codebase in your research, then please cite our paper:
```
@inproceedings{hsla20,
    title = {InstaHide: Instance-hiding Schemes for Private Distributed Learning},
    author = {Yangsibo Huang and Zhao Song and Kai Li and Sanjeev Arora},
    booktitle = {Internation Conference on Machine Learning (ICML)},
    year = {2020}
}
```

## How to run
### Install dependencies
- Create an Anaconda environment with Python3.6
```
conda create -n instahide python=3.6
```
- Run the following command to install dependencies
```
conda activate instahide
pip install -r requirements.txt
```
### Important script arguments
Training configurations:
- `model`: network architecture (default: 'ResNet18')
- `lr`: learning rate (default: 0.1)
- `batch-size`: batch size (default: 128)
- `decay`: weight decay (default: 1e-4)
- `no-augment`: turn off data augmentation 
  
InstaHide configurations:
- `klam`: the number of images got mixed in an instahide encryption, `k` in the paper (default: 4)
- `mode`: 'instahide' or 'mixup' (default: 'instahide')
- `upper`: the upper bound of any coefficient, `c1` in the paper (default: 0.65)
- `dom`: the lower bound of the sum of coefficients of two private images, `c2` in the paper (default: 0.3, *only for Cross-dataset InstaHide*)
  
### Inside-dataset InstaHide:
Inside-dataset Instahide mixes each training image with random images within the same private training dataset. 

For inside-dataset InstaHide training, run the following script:
```
python train_inside.py --mode instahide --klam 4 --data cifar10
```

### Cross-dataset InstaHide:
Cross-dataset Instahide, arguably more secure, involves mixing with random images from a large public dataset. In the paper, we use the unlabelled [ImageNet](http://image-net.org/download)[3] as the public dataset.

For cross-dataset InstaHide training, first, prepare and preprocess your public dataset, and save it in `PATH/TO/FILTERED_PUBLIC_DATA`. Then, run the following training script:

```
python train_cross.py --mode instahide --klam 6 --data cifar10 --pair --dom 0.3 --help_dir PATH/TO/FILTERED_PUBLIC_DATA
```

## Try InstaHide on new datasets or your own data?
You can easily customize your own dataloader to test InstaHide on more datasets (see the `train_inside.py` and `train_cross.py`, around the 'Prepare data' section).

You can also try new models by defining the network architectures under the `\model` folder.

## Questions
If you have any questions, please open an issue or contact yangsibo@princeton.edu.



## References:
[1] [**InstaHide: Instance-hiding Schemes for Private Distributed Learning**](http://arxiv.org/abs/2010.02772), *Yangsibo Huang, Zhao Song, Kai Li, Sanjeev Arora*, ICML 2020

[2] [**mixup: Beyond Empirical Risk Minimization**](https://arxiv.org/abs/1710.09412), *Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz*, ICLR 2018

[3] [**ImageNet: A Large-Scale Hierarchical Image Database.**](http://www.image-net.org/papers/imagenet_cvpr09.pdf), *Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei*, CVPR 2009