## author: xin luo
## create: 2021.9.15; modify: 2023.4.2
## des: parallel data (scene to patch) loading, this scripts is 

import torch
import threading as td
from queue import Queue
from dataloader.preprocess import crop_scales, crop
import numpy as np
import gc

### Crop scenes to multiple patches.
def job_scenes2patches(q, scene_list, truth_list, transforms, patch_size=[2048, 512, 256]):
    patch_list, ptruth_list = [],[]
    ## '''convert image to patches group'''
    zip_data = list(zip(scene_list, truth_list))
    for scene, truth in zip_data:
        if len(patch_size) == 1:
            patches_group, truth = crop(patch_size=patch_size)(scene, truth)
        elif len(patch_size) > 1:
            patches_group, truth = crop_scales(patch_size=patch_size)(scene, truth)
        for transform in transforms:
            patches_group, truth = transform(patches_group, truth)
        truth = torch.unsqueeze(truth, 0)
        patch_list.append(patches_group), ptruth_list.append(truth)
    q.put((patch_list, ptruth_list))


def threads_read(scene_list, truth_list, transforms, patch_size, num_thread=20):
    '''multi-thread reading training data
        cooperated with the job function
    '''
    patch_lists, ptruth_lists = [], []
    q = Queue()
    threads = [td.Thread(target=job_scenes2patches, args=(q, scene_list, \
                                    truth_list, transforms, patch_size)) for i in range(num_thread)]
    start = [t.start() for t in threads]
    join = [t.join() for t in threads]   ## waiting for all the sub-threads to complete, and then go ahead the following script (main thread).
    for i in range(num_thread):
        patch_list, ptruth_list = q.get()
        patch_lists += patch_list
        ptruth_lists += ptruth_list
    return patch_lists, ptruth_lists

class threads_scene_dset(torch.utils.data.Dataset):
    ''' des: dataset (patch and the truth) parallel reading from RAM memory
        input: 
            scene_list, list, consist of torch.tensor-based scene.
            truth_list: list, consist of torch.tensor-based scene truth.
            transforms: list, consist of image augmentation functions.
            patch_size: list, the size of the cropped patch.
            num_thread: number of threads

        '''
    def __init__(self, scene_list, truth_list, transforms, patch_size=[2048, 512, 256], num_thread=1):

        self.scene_list = scene_list
        self.truth_list = truth_list 
        self.patch_size = patch_size
        self.num_thread = num_thread
        self.patches_list, self.ptruth_list = threads_read(\
                                scene_list, truth_list, transforms, patch_size, num_thread)   ### initilize the data
        self.transforms = transforms

    def __getitem__(self, index): 
        '''load patches and truths'''
        patch = self.patches_list[index]
        truth = self.ptruth_list[index]
        ### update the dataset
        if index == len(self.patches_list)-1:           
            del self.patches_list, self.ptruth_list
            gc.collect()
            self.patches_list, self.ptruth_list = threads_read(\
                            self.scene_list, self.truth_list, self.transforms, self.patch_size, self.num_thread)
        return patch, truth

    def __len__(self):
        return len(self.patches_list)

