_base_ = './infopro_deeplabv3_r50-d8_640x1280_40k_cityscapes.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             info_pro_param=dict(loss_recons=2, loss_task=1))
