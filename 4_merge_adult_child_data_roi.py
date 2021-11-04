#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 00:24:23 2020

@author: liuxingyu
"""

#%%
import os
import numpy as np
import pandas as pd
import cb_tools
import nibabel as nib

# %%
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev_v2'
merged_results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev_v2'
index = 't1wT2wRatio' # ['t1wT2wRatio', 'fALFF']
data_type = 'roi'  # ['roi', 'voxel']
atlas_cb_name = 'cb_acapulco_adult'  # ['cb_anat_fsl', 'cb_func_7nw', 'cb_acapulco_adult', 'cb_acapulco_adult-2mm', 'cb_acapulco_pediatric', 'cb_fsl']
dataset = ['HCP-D', 'HCP-Adult']  # ['HCP-D', 'HCP-Adult']

atlas_dir = os.path.join(os.getcwd(), 'atlas')

# %% 
if data_type == 'roi':

    #  merge datasets
    num_str_col = 6
    data_roi_ds = []
    for ds in dataset:
            
        dataset_dir = os.path.join(results_dir, index, ds)
        result_sub_path = os.path.join(dataset_dir, f'{index}_{atlas_cb_name}_roi_sub')
        
        # roi_data
        data_roi = np.load(os.path.join(dataset_dir, f'{index}_{atlas_cb_name}_roi.npy'))
        atlas = cb_tools.atlas_load(atlas_cb_name, atlas_dir)
        
        # sub_info  
        result_sub = pd.read_table(result_sub_path, header=None, sep='_', names=['Sub'], usecols=[0], dtype={'Sub':np.str})
        sub_info_path = os.path.join(results_dir, f'HCP{ds.split("-")[-1]}_preproc_subinfo')
        sub_info = pd.read_table(sub_info_path, sep='\t', dtype={'Sub':np.str})
        sub_info = result_sub.merge(sub_info, on='Sub', how='left')

        # convert to df
        data_roi_df = pd.DataFrame(data_roi, columns=atlas.label_info['name']).astype(np.float)
        data_roi_df = pd.concat([data_roi_df, sub_info], axis=1)
        
        data_roi_ds.append(data_roi_df)

    # save
    data_roi_all = pd.concat((data_roi_ds[i] for i in range(len(data_roi_ds))), axis=0)
    data_roi_all.to_csv(os.path.join(merged_results_dir, index, f'{index}_{atlas_cb_name}_roi.csv'), index=0)

# %%
# only for data in mni 2mm space
if data_type == 'voxel':

    # get cb mask
    template_mni_2mm_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')
    cb_mask_mni_path = os.path.join(atlas_dir,'Cerebellum-MNIfnirt-maxprob-thr25.nii')
    cb_mask = nib.load(cb_mask_mni_path).get_fdata()
    cb_mask = cb_mask.astype(np.bool)

    data_voxel_ds = []
    for ds in dataset:
        
        dataset_dir = os.path.join(results_dir, index, ds)
        result_sub_path = os.path.join(dataset_dir, f'{index}_{atlas_cb_name}_voxel_sub')

        # voxel data
        data_voxel = nib.load(os.path.join(dataset_dir, f'{index}_{atlas_cb_name}_voxel.nii.gz')).get_fdata()[cb_mask].T
        
        # sub_info  
        result_sub = pd.read_table(result_sub_path, header=None, sep='_', names=['Sub'], usecols=[0], dtype={'Sub':np.str})
        sub_info_path = os.path.join(results_dir, f'HCP{ds.split("-")[-1]}_preproc_subinfo')
        sub_info = pd.read_table(sub_info_path, sep='\t', dtype={'Sub':np.str})
        sub_info = result_sub.merge(sub_info, on='Sub', how='left')
        
        # convert to df
        data_voxel_df = pd.DataFrame(data_voxel)
        data_voxel_df = pd.concat([data_voxel_df, sub_info], axis=1)
        data_voxel_ds.append(data_voxel_df)
    
    # %% save
    data_voxel_all = pd.concat((data_voxel_ds[i] for i in range(len(data_voxel_ds))), axis=0)
    data_voxel_all.to_csv(os.path.join(merged_results_dir, index, f'{index}_{atlas_cb_name}_voxel.csv'), index=0)
# %%
