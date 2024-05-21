# rsimg-segmentation-pytorch
We perform the more easily use of the semantic segmentation models on the remote sensing image with pytorch.  
<font color='darkorange'>**The user can perform the model training and validation by easily change the parameters in the scripts/config.py file.** </font>


## Dataset
The surface water dataset built based on Sentinel-2 can be used for the quick testing in this repository, and the surface water dataset can be accessed at: https://zenodo.org/record/5205674.


## Models to be achieved 
1) Unet (Simple)  
2) DeeplabV3Plus [[Paper]](https://arxiv.org/abs/1802.02611)
3) DeeplabV3Plus with MobileNetV2 backbone 
5) WatNet [[Paper]](https://www.sciencedirect.com/science/article/pii/S0303243421001793)
4) HRNet [[Paper]](https://arxiv.org/abs/1908.07919v2)
5) GMNet [[Paper]](https://www.tandfonline.com/doi/full/10.1080/17538947.2023.2198266)

## Features
1) Model training and validation Synchronously (scripts/trainer.py).  
2) Generate validation part from the whole dataset (notebooks/dset_val_patch.ipynb).   
3) Plot the metric figures (notebooks/metric_plot).   

