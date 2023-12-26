from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import pandas as pd
import glob
import torch


# 定义数据类来读取文件
class MalwareDataset(Dataset):
    def __init__(self, file_path):
        self.root_path = file_path
        self.file_path = []
        self.y_data = []
        self.len = 0

        category = 0  # 每种样本的类别(0-24)
        for dir_name in os.listdir(self.root_path):
            for file_name in os.listdir(os.path.join(self.root_path, dir_name)):
                self.file_path.append(os.path.join(self.root_path, dir_name, file_name))  # 存储文件路径
                self.y_data.append(category)  # 存储类别
                self.len = self.len + 1
            category += 1
        self.transforms_data = transforms.Compose(
            # [transforms.RandomRotation(-45, 45)],
            [transforms.ToTensor()]
        )

    def __getitem__(self, index):
        data_path = self.file_path[index]
        image = cv2.imread(data_path)
        image = cv2.resize(image, (224, 224))
        image = self.transforms_data(image)

        return image, self.y_data[index]

    def __len__(self):
        return self.len