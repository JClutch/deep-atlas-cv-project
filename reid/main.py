import os
# import cv2
import torch
# import faiss
import random
import warnings
import numpy as np
# import tqdm.auto as tqdm
import albumentations as A
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
import matplotlib.pyplot as plt
# import faiss.contrib.torch_utils
# import sys
# from IPython.display import display, Javascript
import json
from model import MobileNetV3
# from re_id.data.download_data import config
from collections import Counter
from helpers import get_picture_statistic, get_model_size, calculate_cmc, calculate_map, train, inference, show_inference, hard_voting
from dataset_classes import DEMO_GALLERY, DEMO_QUERY, DEMO_TRAIN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timeit import default_timer as timer


if __name__ == '__main__':
    print("All imports are successful")
    #TODO: Figure out this block
    warnings.filterwarnings("ignore")
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark_enabled = True

    # Select the GPU/CPU to use 
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #TODO: Figure out this block
    path = "./data_reid/reid_training" 
    avg_width, avg_height, max_width, max_height= get_picture_statistic(image_path=path)
    print(f"Average Width: {avg_width}")
    print(f"Average Height: {avg_height}")
    print(f"Max Width: {max_width}")
    print(f"Max Height: {max_height}")

    #TODO: Figure out this block - Demo 1 of Anchor, Positive, Negative

    dataset = DEMO_TRAIN(path=path, transform=None)
    anchor, positive, negative, gt = next(iter(dataset))
    print(f"Anchor ID: {gt}")

    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(torch.permute(anchor, (1,2,0)))
    # ax[1].imshow(torch.permute(positive, (1,2,0)))
    # ax[2].imshow(torch.permute(negative, (1,2,0)))

    # for idx, name in enumerate(["Anchor", "Positive", "Negative"]):
    #     ax[idx].set_title(name)
    #     ax[idx].axis('off')

    #TODO: Figure out this block - Demo of Anchor, Positive, Negative

    dataset = DEMO_TRAIN(path=path, transform=None)

    train_set = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=False, drop_last=False, num_workers=8)
    a, p, n, gt = next(iter(train_set))
    print(f"Anchor ID: {gt[0]}")

    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(torch.permute(a[0], (1,2,0)))
    # ax[1].imshow(torch.permute(p[0], (1,2,0)))
    # ax[2].imshow(torch.permute(n[0], (1,2,0)))

    # for idx, name in enumerate(["Anchor", "Positive", "Negative"]):
    #     ax[idx].set_title(name)
    #     ax[idx].axis('off')

    #TODO: Figure out this block - Start of Training

    model = MobileNetV3()
    model_name = 'mobilenetv3'
    model = model.to(device)

    # Hyper Params
    epochs = 1
    learning_rate = 0.0001
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    weight_path = "./model_weights" 
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch, verbose=True)
    # embedding_dim = model(torch.randn(1, 3, 224, 224)).shape[-1]
    embedding_dim = model(torch.randn(1, 3, 224, 224).to(device)).shape[-1]

    # Create a directory to save the weights
    if os.path.isdir(weight_path) == False:
        os.mkdir(weight_path)

    # Load and setting datasets
    tf = A.Compose([A.Resize(224,224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    train_path = "./data_reid/reid_training/" 
    gallery_path = "./data_reid/reid_test/gallery" 
    query_path = "./data_reid/reid_test/query" 

    train_dataset = DEMO_TRAIN(train_path, tf)

    # Define the percentage of the trainset to use
    percentage = 0.02

    num_samples = int(len(train_dataset) * percentage)
    selected_samples = random.sample(range(len(train_dataset)), num_samples)
    selected_train_dataset = torch.utils.data.Subset(train_dataset, selected_samples)

    query_dataset = DEMO_QUERY(query_path, tf)
    gallery_dataset = DEMO_GALLERY(gallery_path, tf)

    train_loader = DataLoader(selected_train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
    query_loader = DataLoader(query_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    total_gallery_images = len(os.listdir(gallery_path))
    print(f"Total Gallery Images: {total_gallery_images}")

    # Call training function 
    # train(model=model, epochs=1, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, train_loader=train_loader, 
    #     query_loader=query_loader, gallery_loader=gallery_loader, gallery_path=gallery_path, embedding_dim=embedding_dim, topk=total_gallery_images, 
    #     scheduler=False, fp16=False)

    model.load_state_dict(torch.load(os.path.join(weight_path, f"{model_name}_best.pth")))
    get_model_size(model)
    query_list, matched_list = inference(model=model, query_loader=query_loader, gallery_loader=gallery_loader, embedding_dim=embedding_dim, topk=total_gallery_images)

    # show_inference(topk=5, query_list=query_list, matched_list=matched_list, stop=1)

    # Calculate statistics

    hard_voting(matched_list=matched_list)

    rank = calculate_cmc(query_list=query_list, matched_list=matched_list, topk=20)

    for i in range(20):
        print(f"Rank {i+1}: {rank[i]:.1f}")

    x_label = [i for i in range(1, 20+1)]
    y_label = rank

    # plt.plot(x_label, y_label, label=f"{model_name}_{embedding_dim}", linestyle="--", marker='o')
    # plt.title("CMC Rank")
    # plt.xlabel("Rank (m)")
    # plt.ylabel("Rank-m Identification Rate (%)")
    # plt.xticks(range(1,21))
    # plt.legend()
    # plt.show()

    calculate_map(query_list, matched_list, gallery_path)


    # main()