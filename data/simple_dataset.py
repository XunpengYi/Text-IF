from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

class SimpleDataSet(Dataset):
    def __init__(self, visible_path, infrared_path, phase="train", transform=None):
        self.phase = phase
        self.infrared_path = infrared_path
        self.visible_path = visible_path
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.infrared_path)

    def __getitem__(self, item):
        image_A_path = self.visible_path[item]
        image_B_path = self.infrared_path[item]
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')

        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B = self.transform(image_A, image_B)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        return images_A, images_B, name