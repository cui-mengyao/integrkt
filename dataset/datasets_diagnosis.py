import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset_diagnosis import BaseVolumeDataset
import openpyxl

class KidneyVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-54, 247)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

DATASET_DICT = {
    "kidney": KidneyVolumeDataset,
}

def load_data_volume(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
    target_class = None
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    label_path = os.path.join(path_prefix, split+'_malign.xlsx')
    # label_path = os.path.join(path_prefix, split+'_grading.xlsx')
    # label_path = os.path.join(path_prefix, split+'_staging.xlsx')
    # label_path = os.path.join(path_prefix, split+'_subtyping.xlsx')    
    workbook = openpyxl.load_workbook(label_path)
    sheet = workbook.active
    img_files = []
    seg_files = []
    cls_labels = []
    for row in sheet.iter_rows(min_row=1):
        image_dir = os.path.join(path_prefix,'image_box', row[0].value + '_V')
        image_names = os.listdir(image_dir)
        for image_name in image_names:
            tmp = image_name.split('_')[-1]
            seg_files.append(os.path.join(path_prefix,'instance_box', row[0].value, tmp))
            img_files.append(os.path.join(path_prefix,'image_box', row[0].value, image_name))
            cls_labels.append(row[1].value)

    dataset = DATASET_DICT[data](
        img_files,
        cls_labels,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader
