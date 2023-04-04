## author: xin luo
## create: 2021.9.9
## des: simple pre-processing for the dset data (image and truth pair).

import numpy as np
import random
import cv2
from utils.geotif_io import readTiff
import threading as td
from queue import Queue

class normalize:
    '''normalization with the given per-band max and min values'''
    def __init__(self, max_bands, min_bands):
        '''max, min: list, values corresponding to each band'''
        self.max, self.min = max_bands, min_bands      
    def __call__(self, image):
        image_nor = []
        if isinstance(self.max,int):            
            self.max = [self.max for i in range(image.shape[-1])]
            self.min = [self.min for i in range(image.shape[-1])]        
        for band in range(image.shape[-1]):
            band_nor = (image[:,:,band]-self.min[band])/(self.max[band]-self.min[band]+0.0001)
            image_nor.append(band_nor)
        image_nor = np.array(image_nor)
        image_nor = np.clip(image_nor, 0., 1.) 
        return image_nor

def read_normalize(paths_img, paths_truth, max_bands, min_bands):
    ''' des: data (s1 ascending, s1 descending and truth) reading 
             and preprocessing
        input: 
            ascend image, descend image and truth image paths
            max, min: the max and min values of each band.
        return:
            scenes list and truths list
    '''
    scene_list, truth_list = [],[]
    for i in range(len(paths_img)):
        ## --- data reading
        scene, _ = readTiff(paths_img[i])
        truth, _ = readTiff(paths_truth[i])
        ## --- data normalization 
        scene = normalize(max_bands=max_bands, min_bands=min_bands)(scene)
        scene[np.isnan(scene)]=0         # remove nan value
        scene_list.append(scene), truth_list.append(truth)
    return scene_list, truth_list

# def crop(image, truth, size=[256]):
#     ''' numpy-based
#         des: randomly crop corresponding to specific size
#         input image and truth are np.array
#         input patch_size: (size of the cropped patch, the height and width are the same)
#     '''
#     start_h = random.randint(0, truth.shape[0]-size[0])
#     start_w = random.randint(0, truth.shape[1]-size[1])
#     patch = image[:, start_h:start_h+size[0], start_w:start_w+size[1]]
#     ptruth = truth[start_h:start_h+size[0], start_w:start_w+size[1]]
#     return patch, ptruth

class crop:
    ''' numpy-based
        des: randomly crop corresponding to specific size
        input image and truth are np.array
        input size: (size of the height and width, the height and width are the same)
    '''
    def __init__(self, patch_size=[256]):
        self.patch_size = patch_size[0]
    def __call__(self, image, truth):
        start_h = random.randint(0, truth.shape[0] - self.patch_size)
        start_w = random.randint(0, truth.shape[1] - self.patch_size)
        patch = image[:, start_h:start_h+self.patch_size, start_w:start_w + self.patch_size]
        ptruth = truth[start_h:start_h+self.patch_size, start_w:start_w + self.patch_size]
        return patch, ptruth

class crop_scales:
    ''' numpy-based
        des: randomly crop multiple-scale patches (from high to low)
        input patch_size: tuple or list (high -> low)
        we design a multi-thread processsing for resizing
    '''
    def __init__(self, patch_size=[2048, 512, 256], threads=True):
        self.patch_size = patch_size
        self.threads = threads

    def job_resize(self, q, band):
        band_down = cv2.resize(src=band, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        q.put((band_down))

    def threads_resize(self, patch):
        patch_down = []
        q = Queue()
        threads = [td.Thread(target=self.job_resize, args=(q, patch[i])) for i in range(patch.shape[0])]
        start = [t.start() for t in threads]
        join = [t.join() for t in threads]
        for i in range(len(threads)):
            band_down = q.get()
            patch_down.append(band_down)
        patch_down = np.array(patch_down)
        return patch_down

    def __call__(self, image, truth):
        '''input image and turth are np.array'''        
        patches_group = []
        patch_high, ptruth_high = crop(patch_size=self.patch_size)(image, truth)  ## high scale
        patches_group.append(patch_high)
        for size in self.patch_size[1:]:
            start_offset = (self.patch_size[0]-size)//2
            patch_lower = patch_high[:, start_offset:start_offset+size, \
                                                start_offset:start_offset+size]
            patches_group.append(patch_lower)
        ptruth = ptruth_high[start_offset:start_offset + size, \
                                                start_offset:start_offset+size]        
        patches_group_down = []
        for patch in patches_group[:-1]:
            if self.threads:
                patch_down = self.threads_resize(patch)
            else:
                patch_down=[cv2.resize(patch[num], dsize=(self.patch_size[-1], self.patch_size[-1]), \
                                    interpolation=cv2.INTER_LINEAR) for num in range(patch.shape[0])]
            patches_group_down.append(np.array(patch_down))
        patches_group_down.append(patch_lower)

        return patches_group_down, ptruth

