import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
cv2.setNumThreads(0)
import os
import json
import random

class l_basic:
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
            self.data_transforms["train"].append(transforms.Normalize(mean=[0.485, 0.456, 0.40], 
                        std=[0.229, 0.224, 0.225]))     
            self.data_transforms["valid"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229 , 0.224, 0.225]))
            self.data_transforms["test"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                        std=[0.229 , 0.224, 0.225]))
        self.data_transforms["train"] = transforms.Compose(self.data_transforms["train"])
        self.data_transforms["valid"] = transforms.Compose(self.data_transforms["valid"])
        self.data_transforms["test"] = transforms.Compose(self.data_transforms["test"])

    def process(self, data, mode = "train"):

        out_image = []
        label = []
        image_num = []

        for item in data:
            label.append(item['label'])
            image_num.append(item['id'])

            img = cv2.imread(os.path.join(item["liver_path"]))

            up = np.nonzero(img[:,:,0])[0].min()
            down = np.nonzero(img[:,:,0])[0].max()
            left = np.nonzero(img[:,:,0])[1].min()
            right = np.nonzero(img[:,:,0])[1].max()
            img = img[up:down,left:right]
            img = cv2.resize(img, (self.image_size, self.image_size))


            img = Image.fromarray(img)
            img = self.data_transforms[mode](img)

            out_image.append(img)

        label = torch.from_numpy(np.array(label)).long()
        out_image = torch.stack(out_image, dim = 0)

        return {"image": out_image, "label": label, "image_num":image_num}
