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

# %%
# set path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev'
dataset = 'HCP-Adult'  # ['HCP-D', 'HCP-Adult']

temp_data_dir = f'/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev/temp/{dataset}/t1w_t2w'
if os.path.exists(temp_data_dir) is False: os.makedirs(temp_data_dir)

# set HCP data path
if dataset == 'HCP-Adult':
    data_dir = '/nfs/m1/hcp'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub':np.str})
elif dataset == 'HCP-D':
    data_dir = '/nfs/e1/HCPD/fmriresults01'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_dev'), header=0)
    sublist['Sub'] = sublist['Sub'] + '_V1_MR'
    
#%% get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')
cb_mask_mni_path = os.path.join(atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25.nii')
cb_mask_mni_onlylobues = os.path.join(atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25_onlylobues.nii.gz')
atlas_mni_2_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')

cb_mask = nib.load(cb_mask_mni_path).get_fdata()
cb_mask[cb_mask!=0] = 1
cb_mask = np.asarray(cb_mask,dtype=np.bool)

#%% 
index = 't1wT2wRatio'
atlas_cb_name = 'cb_anat_fsl'
atlas_cb = cb_tools.atlas_load(atlas_cb_name, atlas_dir)

save_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(save_dir) is False: os.makedirs(save_dir)

save_path_roi = os.path.join(save_dir, f'{index}_{index}.npy')
save_path_voxel = os.path.join(save_dir, f'{index}_cb_voxel.nii.gz')
save_path_sublist = os.path.join(save_dir, f'{index}_sub')

#%% resample to 2mm
for sub in sublist['Sub']:
    t1w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T1w_restore_brain.nii.gz')
    t2w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T2w_restore_brain.nii.gz')

    t1w_brain_2_path = os.path.join(temp_data_dir, f'T1w_restore_brain_2_{sub}.nii.gz')
    t2w_brain_2_path = os.path.join(temp_data_dir, f'T2w_restore_brain_2_{sub}.nii.gz')   

    if os.path.exists(t2w_brain_2_path) is False or os.path.exists(t1w_brain_2_path) is False:              
        flirt_t1w = f'flirt -in {t1w_brain_path} -ref {atlas_mni_2_path} -applyisoxfm 2 -out {t1w_brain_2_path}'
        flirt_t2w = f'flirt -in {t2w_brain_path} -ref {atlas_mni_2_path} -applyisoxfm 2 -out {t2w_brain_2_path}'           
        subprocess.call(flirt_t1w, shell=True)  
        subprocess.call(flirt_t2w, shell=True)
        
        print('sub ' + sub + 'resample done')

#%% compute t1w/t2w
value_voxel = []
value_roi = []
sub_valid = []

for sub in sublist['Sub']:  
    t1w_brain_path = os.path.join(temp_data_dir, f'T1w_restore_brain_2_{sub}.nii.gz')
    t2w_brain_path = os.path.join(temp_data_dir, f'T2w_restore_brain_2_{sub}.nii.gz')   
              
    t1w = nib.load(t1w_brain_path).get_fdata() 
    t2w = nib.load(t2w_brain_path).get_fdata()
    
    t1w = t1w * cb_mask
    t2w = t2w * cb_mask
    
    # thr 1.5 IQR
    t1w = cb_tools.thr_IQR(t1w, times=1.5)
    t2w = cb_tools.thr_IQR(t2w, times=1.5)
    ratio = t1w / t2w  
    
    # 2nd thr
    ratio = cb_tools.thr_IQR(ratio, times=1.5)
    
    value_voxel.append(ratio)
    value_roi.append(cb_tools.roiing_volume(atlas_cb.data, ratio, method='nanmean', key=atlas_cb.label_info['key'])[1])
    
    sub_valid.append(sub)        
    print('sub ' + sub + ' done')

#%% save mean results
# get mean map before set nan to zero
if dataset == 'HCP-Adult':
    value_voxel = np.asarray(value_voxel)
    value_mean = np.nanmean(value_voxel, 0)
    value_mean[np.isnan(value_voxel).sum(0) > 0.3*len(value_voxel)] = np.nan  # drop voxel with size less than 70% sub
    value_mean = np.nan_to_num(value_mean)
        
    save_path_voxel_mean = save_path_voxel.replace('.nii.gz', '_mean.nii.gz')
    img = nib.Nifti1Image(value_mean, None)
    nib.save(img, save_path_voxel_mean)
    subprocess.check_output(f'fslcpgeom {cb_mask_mni_path} {save_path_voxel_mean} -d', shell=True)
    subprocess.check_output('fslmaths {0}.nii.gz -mas {1} {0}_onlylobues.nii.gz'.format(
        save_path_voxel_mean.split('.')[0], cb_mask_mni_onlylobues), shell=True)
        
#%% save sub-wise results
value_roi = np.nan_to_num(np.asarray(value_roi))
value_voxel = np.nan_to_num(np.asarray(value_voxel))

# save 
np.save(save_path_roi, value_roi) 
pd.DataFrame(sub_valid).to_csv(save_path_sublist, index=False, header=False)

img = nib.Nifti1Image(value_voxel.transpose(1,2,3,0), None)
nib.save(img, save_path_voxel)
subprocess.call(f'fslcpgeom {cb_mask_mni_path} {save_path_voxel} -d', shell=True)
