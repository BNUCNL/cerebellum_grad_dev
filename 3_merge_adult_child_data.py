#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 00:24:23 2020

@author: liuxingyu
"""

import nibabel as nib
import os
import numpy as np
import pandas as pd
import cb_tools

# %%
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev'
index = 'fALFF' # ['t1wT2wRatio', 'fALFF']

dataset = ['HCP-D', 'HCP-Adult']

# %% get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')
atlas_mni_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')
cb_mask_mni_path = os.path.join(atlas_dir,'Cerebellum-MNIfnirt-maxprob-thr25.nii')

cb_mask = nib.load(cb_mask_mni_path).get_fdata()
cb_mask[cb_mask!=0] = 1
cb_mask = np.asarray(cb_mask,dtype=np.bool)

# %% merge datasets
num_str_col = 6
data_voxel_ds = []
data_roi_ds = []
for ds in dataset:
        
    dataset_dir = os.path.join(results_dir, index, ds)
    result_sub_path = os.path.join(dataset_dir, '{0}_sub'.format(index))
    
    if index == 't1wT2wRatio' :
        atlas_cb_name = 'cb_anat_fsl'
        data_voxel = nib.load(os.path.join(dataset_dir, '{0}_cb_voxel.nii.gz'.format(index))).get_data()[cb_mask].T
        
    elif index == 'fALFF' :
        atlas_cb_name = 'cb_anat_cifti' 
        data_voxel = cb_tools.CiftiReader(os.path.join(dataset_dir, '{0}_cb_voxel.dscalar.nii'.format(index))).get_data()
        
    data_roi = np.load(os.path.join(dataset_dir, '{0}_{1}.npy'.format(index, atlas_cb_name)))
    atlas = cb_tools.atlas_load(atlas_cb_name, atlas_dir)
    
    # sub_info  
    result_sub = pd.read_table(result_sub_path, header=None, sep='_', names=['Sub'], usecols=[0], dtype={'Sub':np.str})
    sub_info_path = os.path.join(results_dir, 'HCP{0}_preproc_subinfo'.format(ds.split('-')[-1]))
    sub_info = pd.read_table(sub_info_path, sep='\t', dtype={'Sub':np.str})
    sub_info = result_sub.merge(sub_info, on='Sub', how='left')
    
    # only lobes
    data_roi = data_roi[:, :19]
    atlas.label_info = atlas.label_info[:19]
    
    # convert to df
    data_voxel_df = pd.DataFrame(data_voxel)
    data_roi_df = pd.DataFrame(data_roi, columns=atlas.label_info['name']).astype(np.float)
    
    # data_df = data_df.apply(lambda x: x.fillna(x.mean()), axis=0) 
    data_voxel_df = pd.concat([data_voxel_df, sub_info], axis=1)
    data_roi_df = pd.concat([data_roi_df, sub_info], axis=1)
    
    # retain lobule 1-9 for roi data
    col = [col for col in data_roi_df.columns[:-num_str_col] if col.split('_')[0] == 'X' ]
    data_roi_df.drop(columns=col, inplace=True)
    
    data_roi_ds.append(data_roi_df)
    data_voxel_ds.append(data_voxel_df)

# %% save
data_roi_all = pd.concat((data_roi_ds[i] for i in range(len(data_roi_ds))), axis=0)
data_voxel_all = pd.concat((data_voxel_ds[i] for i in range(len(data_voxel_ds))), axis=0)

data_roi_all.to_csv(os.path.join(results_dir, index, '{0}_{1}.csv'.format(index, atlas_cb_name)), index=0)
data_voxel_all.to_csv(os.path.join(results_dir, index, '{0}_cb_voxel.csv'.format(index)), index=0)
