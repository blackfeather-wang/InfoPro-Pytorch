# Image Classification on ImageNet

## Requirements
- python 3.7.7
- pytorch 1.6.0
- torchvision 0.8.1


## Run

Train ResNet-101 on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python imagenet_DDP.py PATH_TO_DATASET --arch resnet --net resnet101 --batch-size 1024 --lr 0.4 --epochs 90 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --ixx_r 5 --ixy_r 0.75
```

Train ResNet-152 on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python imagenet_DDP.py PATH_TO_DATASET --arch resnet --net resnet152 --batch-size 1024 --lr 0.4 --epochs 90 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --ixx_r 5 --ixy_r 1
```

Train ResNeXt-101, 32×8d on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python imagenet_DDP.py PATH_TO_DATASET --arch resnet --net resnext101_32x8d --batch-size 1024 --lr 0.4 --epochs 90 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --ixx_r 5 --ixy_r 0.75
```

Evaluate pre-trained model on ImageNet

```
CUDA_VISIBLE_DEVICES=0,1 python imagenet_DDP.py PATH_TO_DATASET --net MODEL_NAME --pre-train PATH_TO_PRE_TRAINED_MODELS -e --batch-size 512 --lr 0.4 --epochs 90 --workers 32 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```


## Pre-trained Models

- Measured by Top-1 error.

|Model|Top-1 error|Top-5 error|Link|
|-----|------|-----|-----|
|ResNet-101 (InfoPro*, K=2) |21.85|5.89|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8dde90d313fa4d638b2d/?dl=1) / [Google Drive](https://drive.google.com/file/d/175qnppNdBma8erNT_g8J0b2dxUaJaqpx/view?usp=sharing)|
|ResNet-152 (InfoPro*, K=2) |21.45|5.84|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6be22414da83485eb0f5/?dl=1) / [Google Drive](https://drive.google.com/file/d/1dgU-sjx7vsNvjVbpmQeXJmx1DA0IBpb7/view?usp=sharing)|
|ResNeXt101, 32x8d (InfoPro*, K = 2) |20.35|5.28|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/090b0e56b7164d80b1af/?dl=1) / [Google Drive](https://drive.google.com/file/d/1aYwU-t3zlol_ubPIkDN4PZa0Qt8FXaXW/view?usp=sharing)|



## Results

<p align="center">
    <img src="../figs/imagenet.png" width= "900">
</p>
