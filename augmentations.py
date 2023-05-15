# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np



class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            #version 1.0.3
            import albumentations as A

            T = [
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
                    A.GaussNoise(),    # 将高斯噪声应用于输入图像。
                ], p=0.2),   # 应用选定变换的概率
                A.OneOf([
                    A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
                    A.MedianBlur(blur_limit=3, p=0.01),    # 中值滤波
                    A.Blur(blur_limit=3, p=0.01),   # 使用随机大小的内核模糊输入图像。
                ], p=0.2),
                # 随机应用仿射变换：平移，缩放和旋转输入
                A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
                A.CLAHE(p=0.01),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T)

            print('albumentations: ' + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            print('albumentations: '+ f'{e}')

    def __call__(self, im, p=0.8):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']


        if random.random() > p:
            im = augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5)
        if random.random() > p:
            im = hist_equalize(im, clahe=True, bgr=True) 
        return im


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # no return needed


def hist_equalize(im, clahe=True, bgr=True):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB





