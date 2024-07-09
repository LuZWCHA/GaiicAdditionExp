import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from mypath import Path
import os
import numpy as np


class Custom(Dataset):
    def __init__(self, data, labels, transform=None, transform_cont=None, cont=False, consistency=False):
        self.num_class = 100
        self.data, self.targets =  data, labels
        self.transform = transform
        self.transform_cont = transform_cont
        self.cont = cont
        self.consistency = consistency
                
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img)
        
        if self.transform is not None:
            img_t = self.transform(img)
            if self.consistency:
                img_ = self.transform(img)
            else:
                img_ = img_t
                            
        sample = {'image':img_t, 'image_':img_, 'target':target, 'index':index}
        if self.cont:
            sample['image2'] = self.transform_cont(img)            
        return sample

    def __len__(self):
        return len(self.data)

class DualSteamCustom(Dataset):
    def __init__(self, data, cname2id, transform=None, transform_cont=None, cont=False, consistency=False):
        
        with open(data) as f:
            data = json.load(f)
        
        self.root = data["dataset_root"]
        self.cname2id = cname2id
        self.num_class = len(cname2id)

        self.data, self.targets = data["data"], [self.cname2id[i["label"]] for i in data["data"]]
        self.transform = transform
        self.transform_cont = transform_cont
        self.cont = cont
        self.consistency = consistency
                
    def __getitem__(self, index):
        img_rgb, img_tir, target = self.data[index]["rgb"], self.data[index]["tir"], self.targets[index]
        img_rgb = Image.open(os.path.join(self.root, img_rgb))
        img_tir = Image.open(os.path.join(self.root, img_tir))
        
        if self.transform is not None:
            img_rgb_t = self.transform(img_rgb)
            img_tir_t = self.transform(img_tir)
            if self.consistency:
                img_rgb_ = self.transform(img_rgb)
                img_tir_ = self.transform(img_tir)
            else:
                img_rgb_ = img_rgb_t
                img_tir_ = img_tir_t
                            
        # sample = {'image_rgb':img_t, 'image_rgb_':img_, 'image_tir':img_tir_, 'image_tir_':img_, 'target':target, 'index':index}
        # sample = {'image_rgb':img_t, 'image_rgb_':img_, 'image_tir':img_t, 'image_tir_':img_, 'target':target, 'index':index}
        sample = {
            "image": torch.concat([img_rgb_t, img_tir_t], dim=0),
            "image_": torch.concat([img_rgb_, img_tir_], dim=0),
            'target':target, 'index':index
        }
        if self.cont:
            sample['image2'] = torch.concat([self.transform_cont(img_rgb), self.transform_cont(img_tir)], dim=0)
        return sample

    def __len__(self):
        return len(self.data)

def make_dataset(root=Path.db_root_dir('custom')):    
    #To edit. *_paths are list of absolute paths to images and *_labels are lists of ints with the respective labels    
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
    #Untested example with a structure in root/{train,val}/{class1,...}/xx.{jpeg,png}
    classes = os.listdir(os.path.join(root, 'train')).sort()
    for split in ['train', 'val']:
        for i, c in enumerate(classes):
            images = os.listdir(os.path.join(root, f'{split}/{c}'))
            for im in images:
                if im.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:continue
                if split == 'train':
                    train_paths.append(os.path.join(root, f'{split}/{c}', im))
                    train_labels.append(i)
                else:
                    val_paths.append(os.path.join(root, f'{split}/{c}', im))
                    val_labels.append(i)       

    train_labels = np.array(train_labels) #Might not be required
    val_labels = np.array(val_labels)
    return train_paths, train_labels, val_paths, val_labels, None, None

def make_ds_dataset():
    CLASS2ID = {
        "bg": 0,
        "car": 1,
        "truck": 2,
        "bus": 3,
        "van": 4,
        "freight_car": 5,
    }
    return *Path.db_root_dir('ds_custom'), {"truck": 0, "freight_car": 1}

def make_ds_dataset():
    CLASS2ID = {
        "bg": 0,
        "car": 1,
        "truck": 2,
        "bus": 3,
        "van": 4,
        "freight_car": 5,
    }
    return *Path.db_root_dir('ds_custom2'), CLASS2ID