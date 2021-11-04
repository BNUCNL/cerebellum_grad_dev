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
from tqdm import tqdm
import nibabel as nib

# %%
# set path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev_v2'
dataset = 'HCP-D'  # ['HCP-D', 'HCP-Adult']
index = 'fALFF'
subwise_parcellation_dir = os.path.join(results_dir,'cb_parcellation_subwise','parcellation')

# set HCP data path
if dataset == 'HCP-Adult':
    data_dir = '/nfs/m1/hcp'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub':np.str})
    rf_runlist = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    rf_run_name = 'Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tr = 0.72
elif dataset == 'HCP-D':
    data_dir = '/nfs/e1/HCPD/fmriresults01'
    sublist = pd.read_csv(os.path.join(results_dir, 'sub_dev'), header=0)
    sublist['Sub'] = sublist['Sub'] + '_V1_MR'
    rf_runlist = ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']
    rf_run_name = 'Atlas_MSMAll_hp0_clean.dtseries.nii'
    tr = 0.8

#%% get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')

# fsl(suit) mask - groupwise
atlas_fsl_name = 'cb_fsl'
template_mni_2mm_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')
cb_mask_mni_2mm_path = os.path.join(atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25.nii')

# acapulco adult mask - individual
atlas_acapulco_adult_name = 'cb_acapulco_adult-2mm'
atlas_acapulco_pediatric_name = 'cb_acapulco_pediatric-2mm'

#%% 
# save path
dataset_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(dataset_dir) is False: os.makedirs(dataset_dir)

# individual data
save_dir_indi = os.path.join(dataset_dir, 'individual_voxel')
if os.path.exists(save_dir_indi) is False: os.makedirs(save_dir_indi)

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
    
#%% 
# =================================================== 
# compute fALFF
# without mask - original space
with tqdm(total=len(sublist)) as pbar:
    for sub in sublist['Sub']:
        
        data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')

        if os.path.exists(data_sub_path) is False:
            rf = [os.path.join(data_dir, sub, f'MNINonLinear/Results/{i}/{i}_{rf_run_name}') for i in rf_runlist]       
            if np.asarray([os.path.exists(rf[i]) for i in range(len(rf))]).sum() != len(rf):
                pass
            else:
                value_voxel_run = []
                for rf_path in rf:   
                    # subprocess.call(f'wb_command -cifti-create-dense-from-template {cb_mask_fslr_path} {results_dir}/cbonly_temp.dtseries.nii -cifti {rf_path}', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)          
                    subprocess.call(f'wb_command -cifti-separate {rf_path} COLUMN -volume-all {os.path.join(results_dir, "cbonly_temp.nii.gz")}', shell=True)    

                    rf_data = nib.load(os.path.join(results_dir, 'cbonly_temp.nii.gz')).get_fdata()
                    
                    # number of dim from 3 to 1
                    cb_mask = np.array(rf_data!=0).sum(-1)!=0
                    rf_data_cb = rf_data[cb_mask]

                    falff_cb = fALFF(rf_data_cb, fs=1/tr)
                    # # thr 1.5 IQR
                    # falff = cb_tools.thr_IQR(falff, times=1.5)

                    # number of dim from 1 to 3
                    falff = np.zeros(rf_data.shape[:-1])
                    falff[cb_mask] = falff_cb
                    value_voxel_run.append(falff) 

                value_voxel_run = np.nan_to_num(np.asarray(value_voxel_run))
                value_voxel = np.nanmean(value_voxel_run, 0)
                value_voxel[np.array(value_voxel_run==0).sum(0) > 0] = 0
                img_temp = nib.Nifti1Image(value_voxel, None, nib.load(cb_mask_mni_2mm_path).header)
                nib.save(img_temp, data_sub_path)

        pbar.update(1)

print('compute fALFF - without mask - original space done')

# %%
