import cv2
import numpy as np
from PIL import Image
# 替换为你的一张 mask 图片路径
mask_path = 'data/polypdataset/train/masks copy/1.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
img = Image.open('data/polypdataset/train/masks/1.png')

values, counts = np.unique(mask, return_counts=True)
for v, c in zip(values, counts):
    print(f"像素值 {v}: 出现次数 {c}")
print(f"图像模式 (Mode): {img.mode}") 
print(f"数组形状 (Shape): {np.array(img).shape}")
print(f"Mask 的唯一像素值: {np.unique(mask)}")
print(f"Mask 的形状: {mask.shape}")