import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
cv2.setNumThreads(0)
import os
import json
import random
from numpy import asarray
from reader.formatter.img_preprocessing import add_elastic_transform,add_gaussian_noise, add_uniform_noise

class seg_basic:
    def __init__(self, config):


        self.normalize = config.getboolean("train", "normalize")
        self.v_flip = config.getboolean("train", "v_flip")
        self.h_flip = config.getboolean("train", "h_flip")
        self.jit = config.getboolean("train", "jittter")
        self.rotate = config.getint("train", "rotate")
        self.image_size = config.getint("train", "image_size")
        self.image_root = config.get("data", "image_dataset")

        self.data_transforms = {"train": [],
                                "valid": [],
                                "test": []}

        if self.jit == True:
            self.data_transforms["train"].append(transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5))


        self.data_transforms["train"].append(transforms.ToTensor())
        self.data_transforms["valid"].append(transforms.ToTensor())
        self.data_transforms["test"].append(transforms.ToTensor())

        if self.normalize == True:
            self.data_transforms["train"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))     
            self.data_transforms["valid"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            self.data_transforms["test"].append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        #compose
        self.data_transforms["train"] = transforms.Compose(self.data_transforms["train"])
        self.data_transforms["valid"] = transforms.Compose(self.data_transforms["valid"])
        self.data_transforms["test"] = transforms.Compose(self.data_transforms["test"])

    def preprocess(self, pil_img, scale=1.0,normalize=False, is_label=False):

        mean= 0.0
        std = 0.0

        # img_nd = np.array(pil_img)
        img_nd = pil_img

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd
        if is_label:
            img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        #normalize
        if normalize:
            mean, std = img_trans.mean(), img_trans.std()
            img_trans = (img_trans - mean+0.485) / (std/0.229)

        return img_trans,mean,std
    
    def process(self, data, mode = "train"):

        out_image = []
        labels = []
        image_num = []

        for item in data:

            img = Image.open(os.path.join(self.image_root,item["image_path"]))

            points = json.load(open(item["label_path"]))
            points = np.array([points],dtype=np.int32)
            #height, width
            label = np.zeros([img.size[1],img.size[0]],dtype = np.uint8)
            cv2.fillPoly(label, points, 255)

            l_size = img.size[0]
            r_size = img.size[1]

            img = np.array(img)

            up = np.nonzero(img[:,:,0])[0].min()
            down = np.nonzero(img[:,:,0])[0].max()
            left = np.nonzero(img[:,:,0])[1].min()
            right = np.nonzero(img[:,:,0])[1].max()

            img = img[up:down,left:right]
            label = label[up:down,left:right]

            label = cv2.resize(label,(self.image_size,self.image_size))
            img = cv2.resize(img,(self.image_size,self.image_size))



            if mode == "train":
                img = add_gaussian_noise(img)


            label,_,_ = self.preprocess(label,normalize=False,is_label=True)

            label = label.astype(np.float)

            img = Image.fromarray(img)
            img = self.data_transforms[mode](img)

            labels.append(torch.from_numpy(label).type(torch.FloatTensor))
            out_image.append(img.type(torch.FloatTensor))

            image_num.append(item["image_path"])

        labels = torch.stack(labels, dim=0)
        out_image = torch.stack(out_image, dim = 0)

        return {"image": out_image, "label": labels}