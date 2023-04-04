## author: xin luo
## creat: 2022.4.3, modify: 2023.2.3
## des: model traing with the dset(traset or full dset)
## usage: python trainer.py 
## note: the user should set configure parameters in the scripts/config.py file.

import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
from glob import glob
from scripts import config
from dataloader.preprocess import read_normalize
from utils.metric import oa_binary, miou_binary
from model_seg.unet import unet
from model_seg.deeplabv3plus import deeplabv3plus
from model_seg.deeplabv3plus_mobilev2 import deeplabv3plus_mobilev2
from model_seg.hrnet import hrnet
from model_seg.surface_water.gmnet import gmnet
from dataloader.parallel_loader import threads_scene_dset
from dataloader.loader import patch_tensor_dset, scene_tensor_dset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(999)   # make the training replicable


'''------train step------'''
def train_step(model, loss_fn, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y.float())
    loss.backward()
    optimizer.step()
    ### accuracy evaluation
    miou = miou_binary(pred=pred, truth=y)
    oa = oa_binary(pred=pred, truth=y)
    return loss, miou, oa

'''------validation step------'''
def val_step(model, loss_fn, x, y):
    model.eval()    ### evaluation mode
    with torch.no_grad():
        pred = model(x)
        loss = loss_fn(pred, y.float())
    miou = miou_binary(pred=pred, truth=y)
    oa = oa_binary(pred=pred, truth=y)
    return loss, miou, oa

'''------ train loops ------'''
def train_loops(model, loss_fn, optimizer, tra_loader, val_loader, epoches, lr_scheduler=None):
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    tra_loss_loops, tra_oa_loops, tra_miou_loops = [], [], []
    val_loss_loops, val_oa_loops, val_miou_loops = [], [], []

    for epoch in range(epoches):
        start = time.time()
        tra_loss, val_loss = 0, 0
        tra_miou, val_miou = 0, 0
        tra_oa, val_oa = 0, 0

        '''----- 1. train the model -----'''
        for x_batch, y_batch in tra_loader:
            if isinstance(x_batch, list):   ## multiscale input
                x_batch, y_batch = [batch.to(device) for batch in x_batch], y_batch.to(device)
            else: 
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss, miou, oa = train_step(model=model, loss_fn=loss_fn, 
                                        optimizer=optimizer, x=x_batch, y=y_batch)
            tra_loss += loss.item()
            tra_miou += miou.item()
            tra_oa += oa.item()
        if lr_scheduler:
          lr_scheduler.step(tra_loss)         # if using ReduceLROnPlateau
          # lr_scheduler.step()          # if using StepLR scheduler.

        '''----- 2. validate the model -----'''
        for x_batch, y_batch in val_loader:
            if isinstance(x_batch, list):   ## multiscale input                
                x_batch, y_batch = [batch.to(device).to(dtype=torch.float32) for batch in x_batch], y_batch.to(device)    
            else:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss, miou, oa = val_step(model=model, loss_fn=loss_fn, x=x_batch, y=y_batch)
            val_loss += loss.item()
            val_miou += miou.item()
            val_oa += oa.item()

        '''------ 3. print accuracy ------'''
        tra_loss, val_loss = tra_loss/size_tra_loader, val_loss/size_val_loader
        tra_miou, val_miou = tra_miou/size_tra_loader, val_miou/size_val_loader
        tra_oa, val_oa = tra_oa/size_tra_loader, val_oa/size_val_loader
        tra_loss_loops.append(tra_loss), tra_oa_loops.append(tra_oa), tra_miou_loops.append(tra_miou)
        val_loss_loops.append(val_loss), val_oa_loops.append(val_oa), val_miou_loops.append(val_miou)

        format = 'Ep{}: Tra-> Loss:{:.3f},Oa:{:.3f},Miou:{:.3f}, Val-> Loss:{:.3f},Oa:{:.3f},Miou:{:.3f},Time:{:.1f}s'
        print(format.format(epoch+1, tra_loss, tra_oa, tra_miou, val_loss, val_oa, val_miou, time.time()-start))

    metrics = {'tra_loss':tra_loss_loops, 'tra_oa':tra_oa_loops, 'tra_miou':tra_miou_loops, 'val_loss': val_loss_loops, 'val_oa': val_oa_loops, 'val_miou': val_miou_loops}
    return metrics


if __name__ == '__main__':
    ### 1. model instantiation
    if config.model_name == 'unet':
      model = unet(num_bands=config.num_bands, num_classes=2).to(device)
    elif config.model_name == 'deeplabv3plus':
      model = deeplabv3plus(num_bands=config.num_bands, num_classes=2).to(device)
    elif config.model_name == 'deeplabv3plus_mobilev2':
      model = deeplabv3plus_mobilev2(num_bands=config.num_bands, num_classes=2).to(device) 
    elif config.model_name == 'hrnet':
      model = hrnet(num_bands=config.num_bands, num_classes=2).to(device)         
    elif config.model_name == 'gmnet':
      model = gmnet(num_bands=config.num_bands, num_classes=2,\
                          scale_high=config.patch_size[0], scale_mid=config.patch_size[1], scale_low=config.patch_size[2]).to(device)
    print('Model name:', config.model_name)

    ## Data paths 
    ### Training part of the dataset.
    paths_scene_tra, paths_truth_tra = config.paths_scene_tra, config.paths_truth_tra
    ### Validation part of the dataset (patch format)
    paths_patch_val = sorted(glob(config.dir_val_patch+'/*'))   ## validatation patches

    '''--------- 1. Data loading --------'''
    '''----- 1.1 training data loading (from scenes path) '''
    tra_scenes, tra_truths = read_normalize(paths_img = paths_scene_tra, \
                                                paths_truth = paths_truth_tra, max_bands = config.max_img, min_bands = config.min_img)

    ''' ----- 1.2. Training data loading and auto augmentation'''
    tra_dset = threads_scene_dset(scene_list = tra_scenes, \
                                  truth_list = tra_truths, 
                                  transforms=config.transforms_tra, 
                                  patch_size=config.patch_size,
                                  num_thread=config.num_thread_data_load)           ####  num_thread(30) patches per scene.

    print('size of training data:  ', tra_dset.__len__())

    ''' ----- 1.3. validation data loading (validation patches) ------ '''
    patch_list_val = [torch.load(path) for path in paths_patch_val]

    val_dset = patch_tensor_dset(patch_pair_list = patch_list_val)
    print('size of validation data:', val_dset.__len__())

    tra_loader = torch.utils.data.DataLoader(tra_dset, batch_size=config.batch_size_tra, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=config.batch_size_val)

    ''' -------- 2. Model loading and training strategy ------- '''
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                  mode='min', factor=0.6, patience=20)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)

    ''' -------- 3. Model training for loops ------- '''
    metrics = train_loops(model=model,  
                        loss_fn=config.loss_bce, 
                        optimizer=optimizer,  
                        tra_loader=tra_loader,  
                        val_loader=val_loader,  
                        epoches=config.num_epoch,  
                        lr_scheduler=lr_scheduler,
                        )

    ''' -------- 4. trained model and accuracy metric saving  ------- '''
    ## model saving

    torch.save(model.state_dict(), config.path_weights_save)
    print('Model weights are saved to --> ', config.path_weights_save)
    ## metrics saving
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(config.path_metrics_save, index=False, sep=',')
    print('Training metrics are saved to --> ', config.path_metrics_save)

