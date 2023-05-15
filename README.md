# MRU-Net
MRU-Net：Multi-branch expanded residual U-Net framework for thyroid ultrasound nodule image segmentation.

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10
* Windows
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── src: 搭建MRU-Net模型代码
  ├── train_utils: 训练、验证相关模块
  ├── augmentation.py: 数据增强相关模块
  └── train.py: 以单GPU为例进行训练
```

## DRIVE数据集下载地址：
* 官方地址： [https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation/tree/main/picture] (https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation/tree/main/picture)
