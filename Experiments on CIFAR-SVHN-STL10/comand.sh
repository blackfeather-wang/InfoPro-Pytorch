
###### CIFAR-10 ######


## ResNet-32 + InfoPro ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 1    --ixx_2 0   --ixy_2 0   --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 6 --ixy_1 0    --ixx_2 1   --ixy_2 2   --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0    --ixx_2 0   --ixy_2 2   --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.05 --ixx_2 0.2 --ixy_2 0.5 --aux_net_config 1c2f


## ResNet-110 + InfoPro ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.5 --ixx_2 0   --ixy_2 0 --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 1 --ixy_1 0   --ixx_2 0   --ixy_2 1 --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 0   --ixy_2 1 --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 0.5 --ixy_2 1 --aux_net_config 1c2f


## ResNet-110 + InfoPro* ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.1 --ixx_2 0 --ixy_2 0   --aux_net_config 1c2f --balanced_memory
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 3  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 5 --ixy_2 0.1 --aux_net_config 1c2f --balanced_memory
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 5 --ixy_2 0.1 --aux_net_config 1c2f --balanced_memory



###### STL-10 ######


## ResNet-110 + InfoPro ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 10 --ixy_1 1   --ixx_2 0  --ixy_2 0   --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 20 --ixy_1 0.2 --ixx_2 5  --ixy_2 2   --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 20 --ixy_1 0.2 --ixx_2 5  --ixy_2 1   --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 20 --ixy_1 0.5 --ixx_2 10 --ixy_2 0.1 --aux_net_config 1c2f


## ResNet-110 + InfoPro* ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 20 --ixy_1 1   --ixx_2 0  --ixy_2 0 --aux_net_config 1c2f --balanced_memory
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 3  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 20 --ixy_1 0.5 --ixx_2 5  --ixy_2 2 --aux_net_config 1c2f --balanced_memory
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 20 --ixy_1 0.5 --ixx_2 10 --ixy_2 1 --aux_net_config 1c2f --balanced_memory



###### SVHN ######


## ResNet-110 + InfoPro ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 1 --ixy_1 0.1 --ixx_2 0   --ixy_2 0 --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 0.5 --ixy_2 1 --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 0.5 --ixy_2 2 --aux_net_config 1c2f
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.2 --ixx_2 0.5 --ixy_2 1 --aux_net_config 1c2f



