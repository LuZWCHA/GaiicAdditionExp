import json
import os
from torchvision.datasets.folder import ImageFolder
import torch.utils.data as data
from  monai.data.dataset import Dataset
import random

CLASS2ID = {
        "bg": 0,
        "car": 1,
        "truck": 2,
        "bus": 3,
        "van": 4,
        "freight_car": 5,
    }
class DualStreamDataset(Dataset):
    
    def __init__(self, data_file, cname2id=CLASS2ID, transform=None, with_noise_label=False, noise_ratio=0.2):
        with open(data_file) as f:
            data = json.load(f)
        self.R = random.Random(0)
        self.data_root = data["dataset_root"]
        data = data["data"]
        results = []
        id_length = len(cname2id)

        for d in data:
            _id = cname2id[d["label"]]
            if with_noise_label and self.R.random() < noise_ratio:
                random_idx = self.R.randint(0, id_length - 2)
                all_ids = list(range(0, id_length))
                all_ids.remove(_id)
                _id = all_ids[random_idx]

            results.append(
                {"rgb": os.path.join(self.data_root, d["rgb"]), 
                 "tir": os.path.join(self.data_root, d["tir"]), 
                 "label": _id, "filename": d["rgb"]})

        super().__init__(results, transform)
        
    
    # def __getitem__(self, index):
        # dual_data = self.data[index]
        # label_name = dual_data["label"]
        # # print(label_name, self.c2id)
        # cid = self.c2id[label_name]
        # rgb_path = os.path.join(self.data_root, dual_data["rgb"])
        # tir_path = os.path.join(self.data_root, dual_data["tir"])
        # data = {"rgb": rgb_path, "tir": tir_path, "label": cid}
        # if self.transform is not None:
        #     data = self.transform(data)
        
    #     return data
    
    def __len__(self):
        return len(self.data)
    
    
class DualStreamDetJsonDataset(Dataset):
    
    def __init__(self, det_json, annotation_file, dataset_root, image_path, cname2id=CLASS2ID, transform=None):
        with open(det_json) as f:
            data = json.load(f)
        with open(annotation_file) as f:
            anno = json.load(f)
        self.dataset_root = dataset_root
        # self.R = random.Random(0)
        # self.data_root = data["dataset_root"]
        images = anno["images"]
        self.image_dict = {
            image["image_id"]:image for image in images
        }
        self.image_dir = os.path.join(dataset_root, image_path)

        self.data = data
        
        
        # results = []
        # id_length = len(cname2id)

        # for d in data:
        #     _id = cname2id[d["label"]]
        #     if with_noise_label and self.R.random() < noise_ratio:
        #         random_idx = self.R.randint(0, id_length - 2)
        #         all_ids = list(range(0, id_length))
        #         all_ids.remove(_id)
        #         _id = all_ids[random_idx]

        #     results.append(
        #         {"rgb": os.path.join(self.data_root, d["rgb"]), 
        #          "tir": os.path.join(self.data_root, d["tir"]), 
        #          "label": _id})

        # super().__init__(results, transform)
        
    
    def __getitem__(self, index):
        dual_data = self.data[index]
        label_name = dual_data["label"]
        # print(label_name, self.c2id)
        cid = self.c2id[label_name]
        rgb_path = os.path.join(self.data_root, dual_data["rgb"])
        tir_path = os.path.join(self.data_root, dual_data["tir"])
        data = {"rgb": rgb_path, "tir": tir_path, "label": cid}
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def __len__(self):
        return len(self.data)