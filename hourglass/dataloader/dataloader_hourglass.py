import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from dataloader.dataset_hourglass import heatmap_dataset


def heatmap_Dataloader(params):
    norm_factor = params['data_normalize_factor']
    ds_dir = params['dataset_dir']
    rgb2gray = params['rgb2gray'] if 'rgb2gray' in params else False
    dataset = params['dataset']
    num_worker = 8
    resize = params['resize'] if 'resize' in params else False
    sigma = params['sigma']
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    ###### Training Set ######

    image_datasets['train'] = heatmap_dataset(ds_dir, sigma, setname='train',
                                              transform=None, norm_factor=norm_factor,
                                              rgb2gray=rgb2gray, resize=resize)

    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ##### Validation Set ######

    image_datasets['val'] = heatmap_dataset(ds_dir, sigma, setname='val', transform=None,
                                            norm_factor=norm_factor,
                                            rgb2gray=rgb2gray, resize=resize)
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'],
                                    num_workers=num_worker)
    dataset_sizes['val'] = {len(image_datasets['val'])}

    # print(dataset_sizes)

    return dataloaders, dataset_sizes
