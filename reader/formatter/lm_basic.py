import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
cv2.setNumThreads(0)
import os
import json
import random

class lm_basic:
    def __init__(self, config):

        self.normalize = config.getboolean("train", "normalize")
        self.v_flip = config.getboolean("train", "v_flip")
        self.h_flip = config.getboolean("train", "h_flip")
        self.jit = config.getboolean("train", "h_flip")
        self.rotate = config.getint("train", "rotate")
        self.image_size = config.getint("train", "image_size")
        self.image_root = config.get("data", "image_dataset")

        self.data_transforms = {"train": [],
                                "valid": [],
                                "test": []}

        if self.rotate != 0:
            self.data_transforms["train"].append(transforms.RandomRotation(self.rotate))
        if self.v_flip == True:
            self.data_transforms["train"].append(transforms.RandomVerticalFlip())
        if self.h_flip == True:
            self.data_transforms["train"].append(transforms.RandomHorizontalFlip())
        if self.jit == True:
            self.data_transforms["train"].append(transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5))


        self.data_transforms["train"].append(transforms.ToTensor())
        self.data_transforms["valid"].append(transforms.ToTensor())
        self.data_transforms["test"].append(transforms.ToTensor())

        if self.normalize == True:
            self.data_transforms["train"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]))     
            self.data_transforms["valid"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229 , 0.224, 0.225]))
            self.data_transforms["test"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229 , 0.224, 0.225]))
        self.data_transforms["train"] = transforms.Compose(self.data_transforms["train"])
        self.data_transforms["valid"] = transforms.Compose(self.data_transforms["valid"])
        self.data_transforms["test"] = transforms.Compose(self.data_transforms["test"])

    def process(self, data, mode = "train"):

        imgs = []
        label = []
        image_num = []


        for item in data:
            label.append(item['label'])
            image_num.append(item['id'])

            liver_img = cv2.imread(item["liver_path"])
            lesion_img = cv2.imread(item["lesion_path"])

            up = np.nonzero(liver_img[:,:,0])[0].min()
            down = np.nonzero(liver_img[:,:,0])[0].max()
            left = np.nonzero(liver_img[:,:,0])[1].min()
            right = np.nonzero(liver_img[:,:,0])[1].max()
            liver_img = liver_img[up:down,left:right]
            liver_img = cv2.resize(liver_img, (self.image_size, self.image_size))

            up = max(np.nonzero(lesion_img[:,:,0])[0].min()-10,0)
            down = np.nonzero(lesion_img[:,:,0])[0].max()
            left = max(np.nonzero(lesion_img[:,:,0])[1].min()-10,0)
            right = np.nonzero(lesion_img[:,:,0])[1].max()
            lesion_img = lesion_img[up:down,left:right]
            lesion_img = cv2.resize(lesion_img, (self.image_size, self.image_size))

            liver_img = Image.fromarray(liver_img)
            lesion_img = Image.fromarray(lesion_img)

            img = np.zeros((self.image_size, self.image_size,3),dtype=np.float32)
            img[:,:, 0:2] = liver_img[:,:,0:2]
            img[:,:, 2:] = lesion_img[:,:,1]
            img = self.data_transforms[mode](img)

            imgs.append(img)

        label = torch.from_numpy(np.array(label)).long()
        imgs = torch.stack(imgs, dim = 0)

        return {"liver_lesion_image": imgs, "label": label, "image_num":image_num,}
