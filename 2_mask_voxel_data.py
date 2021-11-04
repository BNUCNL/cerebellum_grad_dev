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
dataset = 'HCP-Adult'  # ['HCP-D', 'HCP-Adult']
index = 't1wT2wRatio'  # ['t1wT2wRatio', 'fALFF']
subwise_parcellation_dir = os.path.join(results_dir,'cb_parcellation_subwise','parcellation')

fsl_mask = False
acapulco_adult = True
acapulco_pediatric = False

# get sublist
if dataset == 'HCP-Adult':
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub':np.str})
elif dataset == 'HCP-D':
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
atlas_acapulco_adult_name = 'cb_acapulco_adult-2mm'
atlas_acapulco_pediatric_name = 'cb_acapulco_pediatric-2mm'

# %% ====================================================
# save path
dataset_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(dataset_dir) is False: os.makedirs(dataset_dir)

# individual data
save_dir_indi = os.path.join(dataset_dir, 'individual_voxel')
save_dir_2mmtemp = os.path.join(save_dir_indi, 't1w_t2w')
if os.path.exists(save_dir_2mmtemp) is False: os.makedirs(save_dir_2mmtemp)

# group data
# with fsl(suit) mask - groupwise
save_path_fsl_voxel = os.path.join(dataset_dir, f'{index}_{atlas_fsl_name}_voxel.nii.gz')
save_path_fsl_voxel_sublist = os.path.join(dataset_dir, f'{index}_{atlas_fsl_name}_voxel_sub')
# with acapulco adult mask - individual
save_path_acapulco_adult_voxel = os.path.join(dataset_dir, f'{index}_{atlas_acapulco_adult_name}_voxel.nii.gz')
save_path_acapulco_adult_sublist = os.path.join(dataset_dir, f'{index}_{atlas_acapulco_adult_name}_voxel_sub')
# with acapulco adult mask - individual
save_path_acapulco_pediatric_voxel = os.path.join(dataset_dir, f'{index}_{atlas_acapulco_pediatric_name}_voxel.nii.gz')
save_path_acapulco_pediatric_sublist = os.path.join(dataset_dir, f'{index}_{atlas_acapulco_pediatric_name}_voxel_sub')

# %% =================================================== 
# mask data
# with fsl(suit) mask - groupwise 2mm space

if fsl_mask:
    cb_mask = nib.load(cb_mask_mni_2mm_path).get_fdata()
    cb_mask[cb_mask!=0] = 1

    value_voxel = []
    sub_valid = []
    with tqdm(total=len(sublist)) as pbar:
        for sub in sublist['Sub']:
            if index == 't1wT2wRatio':
                data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}_2mm.nii.gz')
            elif index == 'fALFF':
                data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')
            if os.path.exists(data_sub_path) is True:
                voxel_data = nib.load(data_sub_path).get_fdata()

                value_voxel.append(voxel_data * cb_mask)
                sub_valid.append(sub) 

            pbar.update(1)

    # save 
    pd.DataFrame(sub_valid).to_csv(save_path_fsl_voxel_sublist, index=False, header=False)

    value_voxel = np.nan_to_num(np.asarray(value_voxel))
    img = nib.Nifti1Image(value_voxel.transpose(1,2,3,0), None, nib.load(template_mni_2mm_path).header)
    nib.save(img, save_path_fsl_voxel)

    # # merg all voxel data
    # allsub2merge_2mm = ''
    # for sub in sub_valid:
    #     allsub2merge_2mm += f'{os.path.join(save_dir_indi, f"{index}_{sub}_2mm.nii.gz ")}'
    # subprocess.call(f'fslmerge -t {save_path_voxel_2mm} {allsub2merge_2mm}', shell=True)

    # # do masking
    # subprocess.call(f'fslmaths {save_path_voxel_2mm} -mas {cb_mask_mni_2mm_path} {save_path_fsl_voxel}', shell=True)
    # subprocess.call(f'cp {save_path_voxel_2mm_sublist} {save_path_fsl_voxel_sublist}', shell=True)

    print('mask data - with fsl(suit) mask - groupwise 2mm space done')

# %% =================================================== 
# mask data
# with acapulco adult mask - individual - 2mm space
if acapulco_adult:
    value_voxel = []
    sub_valid = []
    with tqdm(total=len(sublist)) as pbar:
        for sub in sublist['Sub']:

            pcl_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'tmc', 'seg_mni_2mm.nii.gz')
            # flirt to standard space
            if os.path.exists(pcl_path) is False:
                pcl_mni_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'tmc', 'parc', 'T1w_restore_brain_n4_mni_seg_post.nii.gz')
                if os.path.exists(pcl_mni_path) is True:
                    subprocess.call(f'flirt -in {pcl_mni_path} -ref {os.path.join(atlas_dir, "Atlas_ROIs.2.nii.gz")} -interp nearestneighbour -out {pcl_path}', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True) 
            
            if os.path.exists(pcl_path) is True: 
                if index == 't1wT2wRatio':
                    data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}_2mm.nii.gz')
                elif index == 'fALFF':
                    data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')
                voxel_data = nib.load(data_sub_path).get_fdata()

                # get atlas
                atlas_data = nib.load(os.path.join(pcl_path)).get_fdata()
                atlas_cb = cb_tools.atlas_load(atlas_acapulco_adult_name, atlas_dir, atlas_data=atlas_data)
                cb_mask = np.isin(atlas_data, atlas_cb.label_info['key'])

                value_voxel.append(voxel_data * cb_mask)
                sub_valid.append(sub) 

            pbar.update(1)

    # save 
    pd.DataFrame(sub_valid).to_csv(save_path_acapulco_adult_sublist, index=False, header=False)

    value_voxel = np.nan_to_num(np.asarray(value_voxel))
    img = nib.Nifti1Image(value_voxel.transpose(1,2,3,0), None, nib.load(template_mni_2mm_path).header)
    nib.save(img, save_path_acapulco_adult_voxel)

    print('mask data - with acapulco adult mask - individual done')

# %% =================================================== 
# mask data
# with acapulco pediatric mask - individual - 2mm space
if acapulco_pediatric:
    value_voxel = []
    sub_valid = []
    with tqdm(total=len(sublist)) as pbar:
        for sub in sublist['Sub']:

            pcl_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'kki', 'seg_mni_2mm.nii.gz')
            # flirt to standard space
            if os.path.exists(pcl_path) is False:
                pcl_mni_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'kki', 'parc', 'T1w_restore_brain_n4_mni_seg_post.nii.gz')
                if os.path.exists(pcl_mni_path) is True:
                    subprocess.call(f'flirt -in {pcl_mni_path} -ref {os.path.join(atlas_dir, "Atlas_ROIs.2.nii.gz")} -interp nearestneighbour -out {pcl_path}', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True) 
            
            if os.path.exists(pcl_path) is True: 
                if index == 't1wT2wRatio':
                    data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}_2mm.nii.gz')
                elif index == 'fALFF':
                    data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')
                voxel_data = nib.load(data_sub_path).get_fdata()

                # get atlas
                atlas_data = nib.load(os.path.join(pcl_path)).get_fdata()
                atlas_cb = cb_tools.atlas_load(atlas_acapulco_pediatric_name, atlas_dir, atlas_data=atlas_data)
                cb_mask = np.isin(atlas_data, atlas_cb.label_info['key'])

                value_voxel.append(voxel_data * cb_mask)
                sub_valid.append(sub) 

            pbar.update(1)

    # save 
    pd.DataFrame(sub_valid).to_csv(save_path_acapulco_pediatric_sublist, index=False, header=False)

    value_voxel = np.nan_to_num(np.asarray(value_voxel))
    img = nib.Nifti1Image(value_voxel.transpose(1,2,3,0), None, nib.load(template_mni_2mm_path).header)
    nib.save(img, save_path_acapulco_pediatric_voxel)

    print('mask data - with acapulco pediatric mask - individual done')

# %% ===================================================
# save median results for HCP-Adult

if dataset == 'HCP-Adult':

    def get_median(data, thr=0):
        mask = np.array(data==0).sum(-1) < (1-thr)*len(data) # drop voxel with the size smaller than thr of the total sub
        data_median_masked = np.nanmedian(data[mask], -1)
        data_median = np.zeros(mask.shape)
        data_median[mask] = data_median_masked 
        return data_median

    # fsl
    if fsl_mask:
        value_voxel = nib.load(save_path_fsl_voxel).get_fdata()
        value_median = get_median(value_voxel, thr=0.5)
        save_path_voxel_median = save_path_fsl_voxel.replace('.nii.gz', '_median.nii.gz')
        img = nib.Nifti1Image(value_median, None, nib.load(template_mni_2mm_path).header)
        nib.save(img, save_path_voxel_median)

    # acapulco adult
    if acapulco_adult:
        value_voxel = nib.load(save_path_acapulco_adult_voxel).get_fdata()
        value_median = get_median(value_voxel, thr=0.5)
        save_path_voxel_median = save_path_acapulco_adult_voxel.replace('.nii.gz', '_median.nii.gz')
        img = nib.Nifti1Image(value_median, None, nib.load(template_mni_2mm_path).header)
        nib.save(img, save_path_voxel_median)

    print('save median results done')
