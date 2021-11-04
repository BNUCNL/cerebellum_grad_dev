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
import pickle

# %%
# set path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev_v2'
dataset = 'HCP-D'  # ['HCP-D', 'HCP-Adult']
index = 't1wT2wRatio'  # ['t1wT2wRatio', 'fALFF']
atlas_cb_name = 'cb_acapulco_adult-vertical' # ['cb_anat_fsl', 'cb_func_7nw', 'cb_acapulco_adult', 'cb_acapulco_adult-2mm', 'cb_acapulco_pediatric', 'cb_acapulco_pediatric-2mm', 'cb_acapulco_adult-vertical', 'cb_acapulco_adult-2mm-vertical']
# {'t1wT2wRatio' : 'cb_acapulco_adult', 'fALFF' : 'cb_acapulco_adult-2mm'}

subwise_parcellation_dir = os.path.join(results_dir,'cb_parcellation_subwise','parcellation')

#%% save path
dataset_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(dataset_dir) is False: os.makedirs(dataset_dir)

save_path_roi = os.path.join(dataset_dir, f'{index}_{atlas_cb_name}_roi.npy')
save_path_roi_sublist = os.path.join(dataset_dir, f'{index}_{atlas_cb_name}_roi_sub')

#%%
def ver_slide_slice(atlas, ratio):
    hemi_direct = {'l': 1, 'r': -1} # neurimaging is inversed, left hemi is from middle to right
    ver_data = {}
    for _, row in atlas.label_info.loc[atlas.label_info['hemi']!='Vermis'].iterrows():
        row_atlas_data = atlas.data == row['key']
        row_vertical = row_atlas_data.sum((1,2))

        if row_vertical.max() == 0:
            ver_data[row['name']] = []
        else:
            mid_end_p = np.where(row_vertical != 0)[0][[0, -1]][::hemi_direct[row['hemi']]]

            row_ver_data = []
            for i in np.arange(mid_end_p[0], mid_end_p[1]+1, hemi_direct[row['hemi']]):
                ratio_ver = ratio[i][row_atlas_data[i,:,:]]
                row_ver_data.append([i, np.nanmean(ratio_ver), np.nanmedian(ratio_ver), 
                                    np.nanstd(ratio_ver), np.nonzero(ratio_ver)[0].shape[0]])
            ver_data[row['name']] = np.asarray(row_ver_data)
        
    return ver_data

#%% get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')

if atlas_cb_name in ['cb_anat_fsl', 'cb_func_7nw']:
    # get voxel data
    voxel_mask_name = 'cb_fsl'
    voxel_path = os.path.join(dataset_dir, f'{index}_{voxel_mask_name}_voxel.nii.gz')
    voxel_sublist = os.path.join(dataset_dir, f'{index}_{voxel_mask_name}_voxel_sub')

    value_voxel = nib.load(voxel_path).get_fdata()
    
    # get roi data
    atlas_cb = cb_tools.atlas_load(atlas_cb_name, atlas_dir)

    # roi summary
    value_roi = cb_tools.roiing_volume(atlas_cb.data, value_voxel, method='nanmedian', key=atlas_cb.label_info['key'])[1] 
    value_roi = value_roi.T   

    # sublist
    subprocess.call(f'cp {voxel_sublist} {save_path_roi_sublist}', shell=True)

elif ('acapulco' in atlas_cb_name) and ('2mm' in atlas_cb_name):
    # get voxel data
    voxel_mask_name = atlas_cb_name.split('-', -1)[0] + '-2mm'
    voxel_path = os.path.join(dataset_dir, f'{index}_{voxel_mask_name}_voxel.nii.gz')
    voxel_sublist = os.path.join(dataset_dir, f'{index}_{voxel_mask_name}_voxel_sub')
    sublist = pd.read_csv(voxel_sublist, header=None, names=['Sub'], dtype={'Sub':np.str})

    value_voxel = nib.load(voxel_path).get_fdata()
    
    # roi summary
    value_roi = []
    with tqdm(total=value_voxel.shape[-1]) as pbar:
        for i, sub in enumerate(sublist['Sub']):
            data_sub = value_voxel[:,:,:,i]

            # get atlas
            if 'adult' in atlas_cb_name:
                pcl_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'tmc', 'seg_mni_2mm.nii.gz') 
            elif 'pediatric' in atlas_cb_name:
                pcl_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'kki', 'seg_mni_2mm.nii.gz') 
            
            atlas_data = nib.load(os.path.join(pcl_path)).get_fdata()
            atlas_cb = cb_tools.atlas_load(voxel_mask_name, atlas_dir, atlas_data=atlas_data)

            if 'vertical' not in atlas_cb_name:
                value_roi.append(cb_tools.roiing_volume(atlas_cb.data, data_sub, method='nanmedian', key=atlas_cb.label_info['key'])[1])
            else:
                value_roi.append(ver_slide_slice(atlas_cb, data_sub))

            pbar.update(1) 
    
    subprocess.call(f'cp {voxel_sublist} {save_path_roi_sublist}', shell=True)

elif ('acapulco' in atlas_cb_name) and ('2mm' not in atlas_cb_name):
    # get sublist
    voxel_mask_name = atlas_cb_name.split('-')[0] + '-2mm'
    voxel_sublist = os.path.join(dataset_dir, f'{index}_{voxel_mask_name}_voxel_sub')
    sublist = pd.read_csv(voxel_sublist, header=None, names=['Sub'], dtype={'Sub':np.str})
    save_dir_indi = os.path.join(dataset_dir, 'individual_voxel')

    # roi summary
    value_roi = []
    sub_valid = []
    with tqdm(total=len(sublist)) as pbar:
        for sub in sublist['Sub']:  

            # get voxel data
            data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')
            data_sub = nib.load(data_sub_path).get_fdata()

            # get atlas
            if 'adult' in atlas_cb_name:
                pcl_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'tmc', 'T1w_restore_brain_n4_mni_seg_post_inverse.nii.gz')
            elif 'pediatric' in atlas_cb_name:
                pcl_path = os.path.join(subwise_parcellation_dir, dataset, sub.split('_')[0], 'kki', 'T1w_restore_brain_n4_mni_seg_post_inverse.nii.gz')
            
            atlas_data = nib.load(os.path.join(pcl_path)).get_fdata()
            atlas_cb = cb_tools.atlas_load(voxel_mask_name, atlas_dir, atlas_data=atlas_data)

            if 'vertical' not in atlas_cb_name:
                value_roi.append(cb_tools.roiing_volume(atlas_cb.data, data_sub, method='nanmedian', key=atlas_cb.label_info['key'])[1])
            else:
                value_roi.append(ver_slide_slice(atlas_cb, data_sub))

            sub_valid.append(sub)
            pbar.update(1)

    pd.DataFrame(sub_valid).to_csv(save_path_roi_sublist, index=False, header=False)
    
#%% save sub-wise results
if 'vertical' in atlas_cb_name:
    pickle.dump(value_roi, open(save_path_roi.replace('.npy', '.pckl'),'wb'))
else:
    value_roi = np.nan_to_num(np.asarray(value_roi))
    np.save(save_path_roi, value_roi)
