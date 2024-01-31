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
# from re_id.data.download_data import config
from collections import Counter
from dataset_classes import DEMO_GALLERY, DEMO_QUERY, DEMO_TRAIN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timeit import default_timer as timer

def create_device():
    # Select the GPU/CPU to use 
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

def get_picture_statistic(image_path):
    widths = []
    heights = []

    for img in os.listdir(image_path):
        im = cv2.imread(os.path.join(image_path, img), cv2.COLOR_BGR2RGB)
        widths.append(im.shape[1])
        heights.append(im.shape[0])

    avg_width = round(sum(widths)/len(widths),2)
    avg_height = round(sum(heights)/len(heights),2)
    max_width = max(widths)
    max_height = max(heights)

    return avg_width, avg_height, max_width, max_height

# Loss Functions
def triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = ((anchor-positive)**2).sum(1).sqrt()
    negative_distance = ((anchor-negative)**2).sum(1).sqrt()
    loss = torch.relu(margin + positive_distance - negative_distance)
    return loss.mean()

def quadruplet_loss(anchor, positive, negative1, negative2, margin1=2.0, margin2=1.0):
    squarred_distance_pos = (anchor - positive).pow(2).sum(1)
    squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
    squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)
    quadruplet_loss = F.relu(margin1 + squarred_distance_pos - squarred_distance_neg) + F.relu(margin2 + squarred_distance_pos - squarred_distance_neg_b)
    return quadruplet_loss.mean()

# Validation
def validation(model, query_loader, gallery_loader, gallery_path, embedding_dim, topk):
    device = create_device()
    faiss_index_index = None
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)
    else:
        faiss_index = faiss.IndexFlatL2(embedding_dim)

    model = model.to(device)
    model.eval()

    gallery_list = []
    query_list = []
    matched_list = []
    with torch.no_grad():
        for gallery, labels in gallery_loader:
            gallery = gallery.to(device)

            outputs = model(gallery).cpu().numpy()
            faiss.normalize_L2(outputs)
            faiss_index.add(outputs)
            for label in labels:
                gallery_list.append(label)

    with torch.no_grad():
        for query, label in query_loader:
            for i in label:
                query_list.append(i)
            query = query.to(device)

            outputs = model(query)
            _, I = faiss_index.search(outputs, topk)
            for x in I:
                tmp = [gallery_list[x[i]] for i in range(topk)]
                matched_list.append(tmp)

    def calculate_map(query_list, matched_list, gallery_path):
        total_query_gt = 0
        precision = 0
        count = 0
        AP = 0
        mAP = []

        for query_name, matched_name in zip(query_list, matched_list):
            for x in os.listdir(gallery_path):
                if query_name[:5] == x[:5]:
                    total_query_gt += 1

            tmp_total_query_gt = total_query_gt
            for _, i in enumerate(matched_name, start=1):
                count += 1
                if tmp_total_query_gt == 0:
                    break
                elif query_name[:5] == i[:5]:
                    precision += 1
                    tmp_total_query_gt -= 1
                else:
                    continue
                AP += (precision/count)
            mAP.append(AP/total_query_gt)

            AP = 0
            total_query_gt = 0
            precision = 0
            count = 0
            
        return sum(mAP)/len(mAP)
       
    return calculate_map(query_list, matched_list, gallery_path)

# Training Function

def train(model, epochs, criterion, optimizer, lr_scheduler, train_loader, query_loader, gallery_loader, gallery_path, embedding_dim, topk, scheduler, fp16):
    device = create_device()
    print(f"Start Training...")
    if fp16:
        print(f"Training with Mixed Precision...")
    print()
    best_mAP = 0
    changes = 0
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(device)

    for epoch in range(epochs):
        print('epochs:', epoch)
        model.train()
        for step, (anchor, positive, negative, _) in enumerate(train_loader):
            print('step:', step)
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            if fp16:
                with torch.cuda.amp.autocast():
                    anchor_features = model(anchor) 
                    positive_features = model(positive)
                    negative_features = model(negative)
                    loss = criterion(anchor_features, positive_features, negative_features)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                anchor_features = model(anchor) 
                positive_features = model(positive)
                negative_features = model(negative) 

                loss = criterion(anchor_features, positive_features, negative_features)
                loss.backward()
                optimizer.step()

            if (step+1) % 5 == 0:
                print(f"Epoch:[{epoch}/{epochs}] | Step:[{step+1}/{len(train_loader)}] | Loss:{loss.item():.4f}")
        if scheduler:
            lr_scheduler.step()

        mAP = validation(model=model, query_loader=query_loader, gallery_loader=gallery_loader, gallery_path=gallery_path, embedding_dim=embedding_dim, topk=topk)
        weight_path = "./model_weights" 
        model_name = 'mobilenetv3'
        if mAP >= best_mAP:
            print(f"Best mAP is achieved!")
            print("Saving Best and Latest Model...")
            torch.save(model.state_dict(), os.path.join(weight_path, f"{model_name}_best.pth"))
            changes = mAP - best_mAP
            best_mAP = mAP
        torch.save(model.state_dict(), os.path.join(weight_path, f"{model_name}_latest.pth"))
        print("All Model Checkpoints Saved!")
        print("----------------------------")
        print(f"Best mAP: {best_mAP:.4f}")
        if mAP >= best_mAP:
            print(f"Current mAP: {mAP:.4f} (+{(changes):.4f})")
        elif mAP < best_mAP:
            print(f"Current mAP: {mAP:.4f} (-{(best_mAP-mAP):.4f})")
        print()
    print("Training is finished!")

# Inference Helper Methods

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model Size: {:.2f}MB'.format(size_all_mb))

def inference(model, query_loader, gallery_loader, embedding_dim, topk):
    device = create_device()
    assert topk >= 1

    faiss_index_index = None
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        faiss_index = faiss.GpuIndexFlatIP(res, embedding_dim)
    else:
        faiss_index = faiss.IndexFlatL2(embedding_dim)
    model = model.to(device)
    model.eval()

    gallery_list = []
    query_list = []
    matched_list = []
    inference_time = []
    with torch.no_grad():
        for (gallery, labels) in gallery_loader:
            gallery = gallery.to(device)
            
            outputs = model(gallery).cpu().numpy()
            faiss.normalize_L2(outputs)
            faiss_index.add(outputs)
            for label in labels:
                gallery_list.append(label)
    
    with torch.no_grad():
        for query, label in query_loader:
            for i in label:
                query_list.append(i)
            query = query.to(device)

            inference1 = timer()
            outputs = model(query)
            inference2 = timer()
            # torch.cuda.synchronize() # only if you have cuda support
            
            inference_time.append(inference2-inference1)
            
            start1 = timer()
            _, I = faiss_index.search(outputs, topk)
            end1 = timer()
            print(f"Search Elapsed Time: {end1-start1:.5f} seconds")

            for x in I:
                tmp = [gallery_list[x[i]] for i in range(topk)]
                matched_list.append(tmp)

    print(f"Average Inference Time Elapsed: {sum(inference_time)/len(inference_time):.4f} seconds")
                    
    return query_list, matched_list

def show_inference(topk, query_list, matched_list, stop=0):
    _, ax = plt.subplots(1,1+topk, figsize=(10,5))
    for step, (query_name, matched_name) in enumerate(zip(query_list, matched_list)):
        query = cv2.imread(os.path.join(query_path, query_name))
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        ax[0].imshow(query)
        ax[0].set_title(f"Q: {query_name[:5]}")
        ax[0].axis('off')

        for i in range(topk):
            matched = cv2.cvtColor(cv2.imread(os.path.join(gallery_path, matched_name[i])), cv2.COLOR_BGR2RGB)
            ax[i+1].imshow(matched)
            ax[i+1].axis('off')
            if int(query_name[:5]) == int(matched_name[i][:5]):
                ax[i+1].set_title(f"M: {matched_name[i][:5]}", color='green')
            else:
                ax[i+1].set_title(f"M: {matched_name[i][:5]}", color='red')
            
        if step == stop:
            break

def hard_voting(matched_list):
    final = []
    tmp = []
    for list in matched_list:
        for j in list:
            tmp.append(int(j[:5]))
        count = Counter(tmp)
        final.append(count.most_common(1)[0][0])
        tmp = []

    return final



# Evaluation Helpers

def calculate_cmc(query_list, matched_list, topk):
    count = 0
    total = 0
    rank = []

    for x in range(1, topk+1):
        for (query_name, matched_name) in zip(query_list, matched_list):
            for gallery in matched_name[:x]:
                if query_name[:5] == gallery[:5]:
                    count += 1
                    break
            total += 1
        rank.append((count/total)*100)
        count, total = 0, 0
    return rank

def calculate_map(query_list, matched_list, gallery_path):
    print(f'Number of Queries: {len(query_list)}')
    total_query_gt = 0
    precision = 0
    count = 0
    AP = 0
    mAP = []
    
    for query_name, matched_name in zip(query_list, matched_list):
        for x in os.listdir(gallery_path):
            if query_name[:5] == x[:5]:
                total_query_gt += 1

        tmp_total_query_gt = total_query_gt
        for _, i in enumerate(matched_name, start=1):
            count += 1
            if tmp_total_query_gt == 0:
                break
            elif query_name[:5] == i[:5]:
                precision += 1
                tmp_total_query_gt -= 1
            else:
                continue
            AP += (precision/count)
        mAP.append(AP/total_query_gt)

        AP = 0
        total_query_gt = 0
        precision = 0
        count = 0
        
    print(f"mAP: {(sum(mAP)/len(mAP)):4f}")