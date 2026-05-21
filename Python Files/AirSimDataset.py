import os
import torch
from typing import Callable, Optional
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class AirSimData(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.imagesList = []
        self.annotationsList = []

        if (split != "train") and (split != "val") and (split != "test") and (split != "single"):
            print("Error: split must be 'train', 'val', single, or 'test'")
            exit(0)

        self.imagesPath = root + "rgb/" + split + "/"
        self.annotationsPath = root + "annotations/" + split + "/"

        imagesList = []
        annotationsList = []

        for imagePath in os.listdir(self.imagesPath):
            imagesList.append(imagePath)

        for annotationPath in os.listdir(self.annotationsPath):
            annotationsList.append(annotationPath)

        self.imagesList = imagesList
        self.annotationsList = annotationsList

    def __getitem__(self, index):
        image = Image.open(self.imagesPath + self.imagesList[index]).convert('RGB')
        target = Image.open(self.annotationsPath + self.annotationsList[index])

        #Remapping labels
        label_mapping = {
            0: 0,
            1: 0, # Sky
            2: 1, # Ground
            3: 2, # Building
            4: 3, # Tree
            5: 4  # Car/obstacle
        }

        target = np.array(target)

        remappedTarget = target.copy()

        for key, value in label_mapping.items():
            remappedTarget[target == key] = value

        remappedTarget = Image.fromarray(remappedTarget)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            remappedTarget = self.target_transform(remappedTarget)

        remappedTarget = torch.from_numpy(np.array(remappedTarget)).long()

        return (image, remappedTarget)

    def __len__(self):
        return len(self.imagesList)