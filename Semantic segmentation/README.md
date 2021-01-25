# Semantic Segmentation

## Get Started

- Our code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please install all the required packages and prepare datasets (at least Cityscapes) following their docs.

- We have changed the following code files on the basis of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

```
/mmsegmentation-master/mmseg/models/backbones/resnet.py
/mmsegmentation-master/mmseg/segmentors/encoder_decoder.py
/mmsegmentation-master/mmseg/apis/train.py
```

- We also add some customized configs in

```
/mmsegmentation-master/mmseg/configs
```

- If you hope to implement InfoPro for other models or datasets, please adapt all aforementioned files and configs carefully (and also follow the guidelines of changing models & datasets in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)).



## Run

- Train DeepLabV3 on Cityscapes with 512x1024 crop sizes, batch size = 8 (4 GPUs)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ./configs/deeplabv3/infopro_deeplabv3_r101-d8_512x1024_40k_cityscapes.py 4
```


- Train DeepLabV3 on Cityscapes with 512x1024 crop sizes, batch size = 12 (4 GPUs)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ./configs/deeplabv3/infopro_deeplabv3_r101-d8_512x1024_bs12_40k_cityscapes.py 4
```


- Train DeepLabV3 on Cityscapes with 640x1280 crop sizes, batch size = 8 (4 GPUs)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ./configs/deeplabv3/infopro_deeplabv3_r101-d8_640x1280_40k_cityscapes.py 4
```


- Evaluate pre-trained models (4 GPUs)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_test.sh ./configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes.py  PATH_TO_PRE_TRAINED_MODELS 4 --eval mIoU
```




## Pre-trained Models

- Measured by mean Intersection over Union (mIoU).

|Model|Single Scale (SS)|Multi Scale (MS)|MS + Flip|Link|
|-----|------|-----|-----|-----|
|E2E, 40k, bs=8, 512x1024 |79.12|79.81|80.02|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/3d8ecee6d6d949c395f7/?dl=1) / [Google Drive](https://drive.google.com/file/d/1_xz1uzzx4bgxcv-HOLglxKSLRoLIc7OK/view?usp=sharing)|
|E2E, 60k, bs=8, 512x1024 |79.32|79.95|80.07|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/77fef599522d47d7b6bb/?dl=1) / [Google Drive](https://drive.google.com/file/d/1MRkF9EfmO_PowHquJM07_Ws_TQLbSVWw/view?usp=sharing)|
|InfoPro* (K=2), 40k, bs=8, 512x1024 |79.37|80.53|80.54|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2d743fd5e93d4e85ad6d/?dl=1) / [Google Drive](https://drive.google.com/file/d/14MvWcEYbOxtCxkoP_BRrVo2MZjDeWyCm/view?usp=sharing)|
|InfoPro* (K=2), 40k, bs=12, 512x1024 |79.99|81.09|81.20|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/bd083bc41c614150af86/?dl=1) / [Google Drive](https://drive.google.com/file/d/1WGb9HgUzlMoHsNPxp-J47lv8Bj2Amojk/view?usp=sharing)|
|InfoPro* (K=2), 40k, bs=8, 640x1280 |80.25|81.33|81.42|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/70553a08b7f84a9dbac1/?dl=1) / [Google Drive](https://drive.google.com/file/d/1ePGmVqOHeGJG7TWjqcqGNVwkhCf-LFG4/view?usp=sharing)|



# Results

<p align="center">
    <img src="../figs/segment.png" width= "900">
</p>