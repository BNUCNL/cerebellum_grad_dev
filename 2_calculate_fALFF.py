#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 00:24:23 2020

@author: liuxingyu
"""

# %%
import os
import subprocess
import numpy as np
import pandas as pd
import cb_tools
from scipy import signal

# %%
# set path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev'
dataset = 'HCP-D'  # ['HCP-D', 'HCP-Adult']

# set HCP data path
if dataset == 'HCP-Adult':
    data_dir = '/nfs/m1/hcp'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub':np.str})
    rf_runlist = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
elif dataset == 'HCP-D':
    data_dir = '/nfs/e1/HCPD/fmriresults01'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_dev'), header=0)
    sublist['Sub'] = sublist['Sub'] + '_V1_MR'
    rf_runlist = ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']

#%% get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')
cb_mask_fslr_path = os.path.join(atlas_dir,  'Cerebellum-MNIfnirt-maxprob-thr25.dscalar.nii')
cb_mask_mni_onlylobues = os.path.join(atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25_onlylobues.nii.gz')

cb_mask = cb_tools.CiftiReader(cb_mask_fslr_path).get_data()
cb_mask[cb_mask!=0] = 1
cb_mask = np.asarray(cb_mask,dtype=np.bool)

#%% 
index = 'fALFF'
atlas_cb_name = 'cb_anat_cifti'
atlas_cb = cb_tools.atlas_load(atlas_cb_name, atlas_dir)

save_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(save_dir) is False: os.makedirs(save_dir)

save_path_roi = os.path.join(save_dir, f'{index}_{atlas_cb_name}.npy')
save_path_voxel = os.path.join(save_dir, f'{index}_cb_voxel.dscalar.nii')
save_path_sublist = os.path.join(save_dir, f'{index}_sub')

# %%
def fALFF(data, fs):
    """

    Parameters
    ----------
        data: shape = [n_samples, n_features].
    """
    # remove linear trend
    data_detrend = signal.detrend(data, axis=-1)
    # convert to frequency domain        
    freqs, psd = signal.welch(data_detrend, fs=fs)
    # calculate fALFF
    falff = np.sum(psd[:, (freqs>0.01) * (freqs<0.08)], axis=-1) / np.sum(psd[:, freqs<0.5*fs], axis=-1)
    
    return falff
    
#%% compute fALFF
value_voxel = []
value_roi = []
sub_valid = []

for sub in sublist['Sub']:
    rf = [os.path.join(data_dir, sub, f'MNINonLinear/Results/{i}/{i}_Atlas_MSMAll_hp0_clean.dtseries.nii') for i in rf_runlist]       
    if np.asarray([os.path.exists(rf[i]) for i in range(len(rf))]).sum() != len(rf):
        pass
    else:
        value_voxel_run = []
        value_roi_run = []
        for rf_path in rf:   
            subprocess.call(f'wb_command -cifti-create-dense-from-template {cb_mask_fslr_path} {results_dir}/cbonly_temp.dtseries.nii -cifti {rf_path}', shell=True)          
            rf_data = cb_tools.CiftiReader(f'{results_dir}/cbonly_temp.dtseries.nii').get_data()

            falff = fALFF(rf_data.T, fs=1/0.72)          
            # thr 1.5 IQR
            falff = cb_tools.thr_IQR(falff, times=1.5)
            
            # first save voxelwise falff, then roi mean
            value_voxel_run.append(falff)      
            falff_roi = cb_tools.roiing_volume(atlas_cb.data, falff, method='nanmean', key=atlas_cb.label_info['key'])[1]
            value_roi_run.append(falff_roi)
            
        value_voxel.append(np.nanmean(np.asarray(value_voxel_run), 0))
        value_roi.append(np.nanmean(np.asarray(value_roi_run), 0))
        sub_valid.append(sub)
 
    print('sub ' + sub + ' done')

#%% save mean results
# get mean map before set nan to zero
if dataset == 'HCP-Adult':
    value_voxel = np.asarray(value_voxel)
    value_mean = np.nanmean(value_voxel, 0)
    value_mean[np.isnan(value_voxel).sum(0)>0.3*len(value_voxel)] = np.nan  # drop voxel with size less than 70% sub
    value_mean = np.nan_to_num(value_mean)
        
    brain_models = cb_tools.CiftiReader(cb_mask_fslr_path).brain_models()
    save_path_voxel_mean = save_path_voxel.replace('dscalar.nii', '_mean.dscalar.nii')
    cb_tools.save2cifti(save_path_voxel_mean, value_mean[None,...], brain_models, volume=cb_tools.CiftiReader(cb_mask_fslr_path).volume)

    subprocess.check_output('wb_command -cifti-separate {0} COLUMN -volume-all {1}'.format(save_path_voxel_mean, save_path_voxel_mean.replace('.dscalar.nii', '_cbonly.nii.gz')), shell=True)    
    subprocess.check_output('fslmaths {0}_cbonly.nii.gz -mas {1} {0}_cbonly_onlylobues.nii.gz'.format(save_path_voxel_mean.split('.')[0], cb_mask_mni_onlylobues), shell=True)

#%% save sub-wise results
value_roi = np.nan_to_num(np.asarray(value_roi))
value_voxel = np.nan_to_num(np.asarray(value_voxel))

# save 
np.save(save_path_roi, value_roi) 
pd.DataFrame(sub_valid).to_csv(save_path_sublist, index=False, header=False)

brain_models = cb_tools.CiftiReader(cb_mask_fslr_path).brain_models()
cb_tools.save2cifti(save_path_voxel, value_voxel, brain_models, volume=cb_tools.CiftiReader(cb_mask_fslr_path).volume)
subprocess.check_output('wb_command -cifti-separate {0} COLUMN -volume-all {1}'.format(save_path_voxel, save_path_voxel.replace('.dscalar.nii', '_cbonly.nii.gz')), shell=True)
