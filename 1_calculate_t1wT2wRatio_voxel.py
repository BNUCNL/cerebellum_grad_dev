#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 00:24:23 2020

@author: liuxingyu
"""

# %%
import nibabel as nib
import os
import subprocess
import numpy as np
import pandas as pd
import cb_tools 
from tqdm import tqdm

# %% ====================================================
# set path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev_v2'
dataset = 'HCP-D'  # ['HCP-D', 'HCP-Adult']
index = 't1wT2wRatio'
subwise_parcellation_dir = os.path.join(results_dir,'cb_parcellation_subwise','parcellation')

# set HCP data path
if dataset == 'HCP-Adult':
    data_dir = '/nfs/m1/hcp'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub':np.str})
elif dataset == 'HCP-D':
    data_dir = '/nfs/e1/HCPD/fmriresults01'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_dev'), header=0)
    sublist['Sub'] = sublist['Sub'] + '_V1_MR'
    
# %% ====================================================
# get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')

# fsl(suit) mask - groupwise
atlas_fsl_name = 'cb_fsl'
template_mni_2mm_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')
cb_mask_mni_2mm_path = os.path.join(atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25.nii')

# acapulco adult mask - individual
atlas_acapulco_adult_name = 'cb_acapulco_adult'
atlas_acapulco_pediatric_name = 'cb_acapulco_pediatric'

# %% ====================================================
# save path
dataset_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(dataset_dir) is False: os.makedirs(dataset_dir)

# individual data
save_dir_indi = os.path.join(dataset_dir, 'individual_voxel')
save_dir_2mmtemp = os.path.join(save_dir_indi, 't1w_t2w')
if os.path.exists(save_dir_2mmtemp) is False: os.makedirs(save_dir_2mmtemp)

# %% ====================================================
# resample to 2mm
with tqdm(total=len(sublist)) as pbar:
    for sub in sublist['Sub']:
        t1w_brain_2_path = os.path.join(save_dir_2mmtemp, f'T1w_restore_brain_2_{sub}.nii.gz')
        t2w_brain_2_path = os.path.join(save_dir_2mmtemp, f'T2w_restore_brain_2_{sub}.nii.gz')  

        t1w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T1w_restore_brain.nii.gz')
        t2w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T2w_restore_brain.nii.gz')

        if os.path.exists(t1w_brain_2_path) is False or os.path.exists(t2w_brain_2_path) is False:              
            flirt_t1w = f'flirt -in {t1w_brain_path} -ref {template_mni_2mm_path} -applyisoxfm 2 -out {t1w_brain_2_path}'
            flirt_t2w = f'flirt -in {t2w_brain_path} -ref {template_mni_2mm_path} -applyisoxfm 2 -out {t2w_brain_2_path}'           
            subprocess.call(flirt_t1w, shell=True)  
            subprocess.call(flirt_t2w, shell=True)

        pbar.update(1)
print('resample done')

# %% =================================================== 
# compute t1w/t2w
# without mask - original space
with tqdm(total=len(sublist)) as pbar:
    for sub in sublist['Sub']:
        # orig space
        data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')
        if os.path.exists(data_sub_path) is False:

            t1w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T1w_restore_brain.nii.gz')
            t2w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T2w_restore_brain.nii.gz')
            t1w = nib.load(t1w_brain_path).get_fdata() 
            t2w = nib.load(t2w_brain_path).get_fdata()
            
            ## -------------------------------------
            # # thr 1.5 IQR
            # t1w = cb_tools.thr_IQR(t1w, times=1.5)
            # t2w = cb_tools.thr_IQR(t2w, times=1.5)
            ratio = t1w / t2w  
            
            # # 2nd thr
            # ratio = cb_tools.thr_IQR(ratio, times=1.5)
            ## -------------------------------------

            ratio = np.nan_to_num(np.asarray(ratio))
            img_temp = nib.Nifti1Image(ratio, None, nib.load(t1w_brain_path).header)
            nib.save(img_temp, data_sub_path)

        pbar.update(1)

print('compute t1w/t2w - without mask - original space done')

# %% =================================================== 
# compute t1w/t2w
# without mask - 2mm
with tqdm(total=len(sublist)) as pbar:
    for sub in sublist['Sub']:
        # 2mm
        data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}_2mm.nii.gz')
        if os.path.exists(data_sub_path) is False:
            t1w_brain_path = os.path.join(save_dir_2mmtemp, f'T1w_restore_brain_2_{sub}.nii.gz')
            t2w_brain_path = os.path.join(save_dir_2mmtemp, f'T2w_restore_brain_2_{sub}.nii.gz') 
                    
            t1w = nib.load(t1w_brain_path).get_fdata() 
            t2w = nib.load(t2w_brain_path).get_fdata()
            
            ## -------------------------------------
            # # thr 1.5 IQR
            # t1w = cb_tools.thr_IQR(t1w, times=1.5)
            # t2w = cb_tools.thr_IQR(t2w, times=1.5)
            ratio = t1w / t2w  
            
            # # 2nd thr
            # ratio = cb_tools.thr_IQR(ratio, times=1.5)
            ## -------------------------------------

            ratio = np.nan_to_num(np.asarray(ratio))
            img_temp = nib.Nifti1Image(ratio, None, nib.load(t1w_brain_path).header)
            nib.save(img_temp, data_sub_path)

        pbar.update(1)

print('compute t1w/t2w - without mask - 2mm done')
