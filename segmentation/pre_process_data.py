import os
import cv2
import numpy as np
from glob import glob

# 填入你所有的 masks 路径（包括训练集和那5个测试集）
paths = [
    'data/polypdataset/test/CVC-300/masks',
    'data/polypdataset/test/CVC-ClinicDB/masks',
    'data/polypdataset/test/CVC-ColonDB/masks',
    'data/polypdataset/test/ETIS-LaribPolypDB/masks',
    'data/polypdataset/test/Kvasir/masks',
]

for folder in paths:
    if not os.path.exists(folder): continue
    print(f"正在清洗: {folder}")
    for img_p in glob(os.path.join(folder, '*.png')):
        mask = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) 
        cv2.imwrite(img_p, mask)
print("✨ 全部清洗完成！现在的图片就是标准的 labelTrainIds 格式了。")