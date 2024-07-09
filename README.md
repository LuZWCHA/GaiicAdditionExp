# GAIIC2024 附加实验代码
这里是和检测模型/分类/配准相关的预实验和实验代码介绍文档，主要检测部分代码在这里：[MMdetection](https://github.com/LuZWCHA/mmdetection/tree/merge_main_yang?tab=readme-ov-file) \
TODO [该文档存会不断完善]

## 环境准备
请使用```Conda```的 requirements.txt 进行安装
``` bash
conda install --yes --file requirements.txt
```


## 概述
文档中主要进行三个方向的预实验和实验：检测模型+分类模型的两阶段任务，离线配准，抗噪（标签噪声）实验。

### 检测+分类
检测加分类为离线任务，我们裁剪了从检测模型中提取的FP（实际为背景预测为前景）图像（两个模态同时裁剪所以是配对的），和所有GT框对应的图像。组成了5+1（原始5类加上背景类）共6分类的裁剪图像数据集。\
> 该实验在单卡机器上训练所有只提供了单卡的训练代码。
#### 数据准备

##### 生成裁剪数据集
裁剪代码请参考 检测部分代码 位于：```tools/analysis_tools/draw_pred_and_gt_boxes.py``` 中的[裁剪部分代码](https://github.com/LuZWCHA/mmdetection/blob/merge_main_yang/tools/analysis_tools/draw_pred_and_gt_boxes.py) ```parse(path, image_roots, save_root, ...)```请参考example中的代码设置```func```参数为```crop_bbox```, 注意将输出目录变量[output_dir](https://github.com/LuZWCHA/mmdetection/blob/1455122ddff712a86179462d3dace097f3098b4a/tools/analysis_tools/draw_pred_and_gt_boxes.py#L162)设置为实际存在的输出目录。
##### 数据集划分
这里可以根据实际情况进行任意划分，我们在实验中按照比赛数据集在检测任务上的划分进行划分（保持原有划分的训练集为训练集，验证集为验证集）
##### 生成训练脚本能够接受的数据集描述文件
裁剪完成后，使用脚本```classify/create_train_file.py``` 创建多种分类的数据集文件。如何创建6分类和2分类的代码示例再文件中给出，请修改具体目录，和具体要创建的类别。最终会生成一个json文件，请将该文件放入任意目录准备训练使用。
#### 分类模型介绍
我们采用的分类模型是 **torch** 原生```vit-base```模型。为了兼容双光数据我们进行了简单修改，模型文件位于```classify/models/dual_stream_vit.py```。
#### 模型训练
训练我们使用了Monai的transform模块，可以根据需要进行修改，所有的增广pipeline位于 ```classify/dataset/transforms/transforms.py```。
3分类的训练脚本如下：
``` shell
python train.py \
    -train_data/train_c3.json \
    -val data/val_c3.json \
    -e ViT_B_Vehicle_c3 \
    --task_name ViT_B_Vehicle_c3 \
    --num_class 3 \
    --class_names "truck" "freight_car" "bg" \
    --num_epoch 100 \
    --num_worker 16 \
    --batch_size 128 \
    --save_interval 4 \
    --lr 1e-4 \
    --evaluate_first \
    --amp
```
所有实验中使用的训练脚本可以在```classify/scripts```目录下找到，请根据你的实际硬件情况更改配置。

#### 抗标签噪声框架预实验
比赛中的实际过程是想先进行分类模型的噪声标签实验再想办法迁移到检测上，因此我们在分类任务上先进行了预实验。
训练抗噪模型我们采用了同样的双流模型架构，仅仅替换训练过程，抗噪声框架见```classify/lnl```，
训练脚本如下所示(请不要更改除了batch-size和num-class和exp-name外的其他参数，除非你熟悉了LNL部分的代码和逻辑)：


- 如果要训练6分类代码不用做任何修改，如果要做2分类（truck和freight_car）请注释掉 ```classify/lnl/custom_dataset.py``` 最后的第二个 ```make_ds_dataset``` 方法（#LINE 121），使第一个 ```make_ds_dataset``` 生效（#LINE 109）。

``` bash
# export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
    # --resume classify/checkpoints/dual_stream_vit_ds_custom/truck_freightcar_noise/1/dual_stream_vit_ds_custom.pth.tar \

python lnl/main.py \
    --net dual_stream_vit \
    --dataset ds_custom2 \
    --epochs 50 \
    --batch-size 256 \
    --num-class 6 \
    --exp-name six_classes_noise \
    --ood-noise 0.0 \
    --id-noise 0.2 \
    --lr 1e-3 \
    --mixup \
    --warmup 5 \
    --proj-size 768 \
    --cont
```
### 离线配准
离线配准我们参考了许多弹性配准方案，最后发现分享给的点匹配应该是不错的方法。
最后的实验代码我们采用了 **LoFTR** + TPS的方案。
#### 思路
我们直接采用了LoFTR的预训练模型进行点配准。等待配准完成后对配对点进行重点采样后映射到控制点对上进行弹性配准，为了性能可以跟上检测模型推理，我们采用了torch加速的tps弹性形变进行插值。为了筛选出不容易错配的图像对，防止错误配准破坏原图信息，我们按照以下规则进行图像级筛选：
1. 筛选白昼图像
2. 筛选配准置信度较高的图像
3. 筛选的点对相对距离小于40像素，防止超大形变破坏原图结构
4. 筛选有效点对数量大于16个点的图像

配准代码见文件： ```registration/LoFTR-master/registration.py```

批量配准函数使用方法, 在```registration/LoFTR-master```下新建一个python脚本，运行以下内容，注意修改路径：
``` python
from .registration import do_registration, WEIGHT_PATH
input_dir = "projects/data/mmdet/gaiic/GAIIC2024/test"
output_dir = "projects/data/mmdet/gaiic/GAIIC2024/reg_test"
do_registration(WEIGHT_PATH, get_image_pairs(input_dir), )
```
以上脚本会将所有test中符合条件的图像进行配准，并按照原始目录树输出配准后的文件。

### 抗噪（标签噪声）实验
在分类任务中我们使用了一套标签噪声矫正框架，除此之外还使用了基于CleanLab这类模型置信度的清洗框架，CleanLab在比赛数据集的清洗代码在 ```cleanlab/clean_obj_det.py``` 中。

请更改```pred_mmdet```和```label_coco```变量为实际的```mmdetection```格式的模型预测结果和```coco```格式的标注结果，然后直接运行该文件：
``` python
    pred_mmdet = "mmdetection/0530_train_tta.pkl"
    label_coco = "projects/data/mmdet/gaiic/mix_all/merged_coco_new_vis_3cls.json"
```
结果保存在```data```目录下，```cleanlab/data/issue_idx.pkl```, ```cleanlab/data/labels.pkl```分别保存问题图片的下标和解析后的label文件，可以用这两个文件获得对应图片名。

## 实验结果
### 2阶段分类
1. 检测模型2分类ACC：0.79
2. 分类模型二分类ACC：0.7253
3. 抗噪声分类模型ACC：0.719
### CleanLab
cleanlab: 人工在测试数据上筛选100例图像进行二分类，由Cleanlab筛出的错误图像不足实际错误标注的30%
### 离线配准
(需要重新计算) 直接使用配准后的测试集进行测试性能下降，mAP 0.538->0.533