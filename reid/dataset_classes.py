import os
import cv2
import torch
import faiss
import random
import warnings
import numpy as np
import tqdm.auto as tqdm
import albumentations as A
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import faiss.contrib.torch_utils
import sys
from IPython.display import display, Javascript
import json

sys.path.append('/Users/jasonmccutchan/Desktop/AI_BasketBall_Video_Analysis')
# from re_id.models.model import *
# from re_id.data.download_data import config
# from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from timeit import default_timer as timer

class DEMO_GALLERY(Dataset):
    def __init__(self, gallery_path, transform):
        super().__init__()
        self.gallery_path = gallery_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.gallery_path))

    def __getitem__(self, item):
        gallery_image_name = os.listdir(self.gallery_path)[item]
        gallery_label = gallery_image_name
        gallery_image = cv2.imread(os.path.join(self.gallery_path, gallery_image_name))
        gallery_image = cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=gallery_image)
            gallery_image = transformed['image']
            gallery_image = gallery_image.astype(np.float32)
            gallery_image = torch.from_numpy(gallery_image)
            gallery_image = torch.permute(gallery_image, (2,0,1))
        else:
            gallery_image = gallery_image.astype(np.float32)
            gallery_image /= 255.
            gallery_image = torch.from_numpy(gallery_image)
            gallery_image = torch.permute(gallery_image, (2,0,1))

        return gallery_image, gallery_label
    
class DEMO_QUERY(Dataset):
    def __init__(self, query_path, transform):
        super().__init__()
        self.query_path = query_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.query_path))

    def __getitem__(self, item):
        query_image_name = os.listdir(self.query_path)[item]
        query_label = query_image_name
        query_image = cv2.imread(os.path.join(self.query_path, query_image_name))
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=query_image)
            query_image = transformed['image']
            query_image = query_image.astype(np.float32)
            query_image = torch.from_numpy(query_image)
            query_image = torch.permute(query_image, (2,0,1))
        else:
            query_image = query_image.astype(np.float32)
            query_image /= 255.
            query_image = torch.from_numpy(query_image)
            query_image = torch.permute(query_image, (2,0,1))

        return query_image, query_label

class DEMO_TRAIN(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.path = path
        self.transform = transform
        self.people_list = sorted(os.listdir(path))
        
    def __len__(self):
        return len(self.people_list)

    def __getitem__(self, item):
        anchor_name = self.people_list[item]
        anchor_id = int(anchor_name[:5])
        anchor = cv2.imread(os.path.join(self.path, anchor_name))
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        
        positive_list = [filename for filename in self.people_list if filename.startswith(anchor_name[:5])]
        positive_name = random.choice(positive_list)
        while positive_name == anchor_name:
            positive_name = random.choice(positive_list)
        positive = cv2.imread(os.path.join(self.path, positive_name))
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)

        negative_name = random.choice(self.people_list) 
        negative_id = int(negative_name[:5])
        while negative_id == anchor_id:
            negative_name = random.choice(self.people_list) 
            negative_id = int(negative_name[:5])
        negative = cv2.imread(os.path.join(self.path, negative_name))
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        set_images = [anchor, positive, negative]

        if self.transform:
            for idx, i in enumerate(set_images):
                transformed = self.transform(image=i)
                set_images[idx] = transformed['image']
                set_images[idx] = set_images[idx].astype(np.float32)
                set_images[idx] = torch.from_numpy(set_images[idx])
                set_images[idx] = torch.permute(set_images[idx], (2,0,1))
                
        else:
            tf = A.Compose([A.Resize(224,224)])
            for idx, i in enumerate(set_images):
                transformed = tf(image=i)
                set_images[idx] = transformed['image']
                set_images[idx] = set_images[idx].astype(np.float32)
                set_images[idx] /= 255.
                set_images[idx] = torch.from_numpy(set_images[idx])
                set_images[idx] = torch.permute(set_images[idx], (2,0,1))
                

        return set_images[0], set_images[1], set_images[2], anchor_id