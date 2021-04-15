
#%%
import nibabel as nib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import cb_tools
import heritability

# %%
# set results path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev'

# %%
# prepare roi data
data_t1wT2wRatio = pd.read_csv(os.path.join(results_dir, 't1wT2wRatio', 't1wT2wRatio_cb_anat_fsl.csv'))
data_falff = pd.read_csv(os.path.join(results_dir, 'fALFF', 'fALFF_cb_anat_cifti.csv'))
dem_herit = pd.read_csv(os.path.join(results_dir,'sub_heritabitily.csv'), dtype=np.str)

atlas_dir = os.path.join(os.getcwd(), 'atlas')
atlas = cb_tools.atlas_load('cb_anat_fsl', atlas_dir)

num_str_col = 6
palette_cb = sns.diverging_palette(230,230, l=80, center='dark', n=9)

# %%
def hemi_merging(x, num_str_col):
    x_l = x[[x.columns[i] for i in range(x.shape[-1]) if x.columns[i].split('_')[-1]=='l']]
    x_l.rename(columns=lambda x: x.split('_')[0], inplace=True)
    
    x_r = x[[x.columns[i] for i in range(x.shape[-1]) if x.columns[i].split('_')[-1]=='r']]   
    x_r.rename(columns=lambda x: x.split('_')[0], inplace=True)

    x_merged = pd.concat([pd.concat([x_l,x_r]).mean(level=0),x.iloc[:,-num_str_col:]], axis=1)
    
    return x_merged

data_t1wT2wRatio = hemi_merging(data_t1wT2wRatio, num_str_col)
data_falff = hemi_merging(data_falff, num_str_col)
lobues_name = atlas.label_info['lobule'][:18:2]

#%% heritabitily

def twins_select(data_herit, dem_herit, num_str_col):
    twins_in = np.c_[np.isin(dem_herit['twin1'].values, data_herit['Sub'].values),
                     np.isin(dem_herit['twin2'].values, data_herit['Sub'].values)]
    
    dem_herit = dem_herit.loc[twins_in.sum(-1) == 2]
    
    data_herit = data_herit.set_index('Sub',drop=False)

    col = data_herit.columns[:-num_str_col]
    # MZ
    mz1 = data_herit.loc[dem_herit.loc[dem_herit['zygosity']=='MZ','twin1'],col]
    mz2 = data_herit.loc[dem_herit.loc[dem_herit['zygosity']=='MZ','twin2'],col]
    mz = np.stack((mz1.values, mz2.values))
    
    # DZ
    dz1 = data_herit.loc[dem_herit.loc[dem_herit['zygosity']=='DZ','twin1'],col]
    dz2 = data_herit.loc[dem_herit.loc[dem_herit['zygosity']=='DZ','twin2'],col]
    dz = np.stack((dz1.values, dz2.values))
    
    return mz, dz

#%%
# calculate h2
mz_anat, dz_anat = twins_select(data_t1wT2wRatio, dem_herit, num_str_col)
mz_func, dz_func = twins_select(data_falff, dem_herit, num_str_col)

#%%
# plot
_, h2_anat = heritability.heritability(mz_anat, dz_anat, n_bootstrap=1000, confidence=95)
h2_anat_df = pd.DataFrame(h2_anat, columns=atlas.label_info['lobule'][:18:2].values)
h2_anat_df = h2_anat_df.stack().reset_index(-1, name='h2')

_, h2_func = heritability.heritability(mz_func, dz_func, n_bootstrap=1000, confidence=95)
h2_func_df = pd.DataFrame(h2_func, columns=atlas.label_info['lobule'][:18:2].values)
h2_func_df = h2_func_df.stack().reset_index(-1, name='h2')

h2_df = [h2_anat_df, h2_func_df]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3.3))
for i in range(2):
    sns.barplot(x='h2', y='level_1', ci=None, palette=palette_cb, data=h2_df[i], ax=axes.flatten()[i])
    axes.flatten()[i].set_xlim([0,0.7])
    
    axes[i].tick_params(colors='gray', which='both')
#
    [axes[i].spines[k].set_color('darkgray') for k in ['top','bottom','left','right']]
plt.tight_layout()

#%% grad
shape = polyfit_dataframing(gradient_magnitude(data_herit.iloc[:,:-num_str_col]), deg=2)
data_herit['a_ml'] = mytool.core.thr_IQR(shape['a'].values, times=3, series=False)
data_herit['p1_ml'] = mytool.core.thr_IQR(shape['p1'].values, times=3, series=False)

mz, dz = organize_herit_data(data_herit.iloc[:, -2:], dem_herit)
h2, _ = mytool.heritability.heritability(mz, dz, n_bootstrap=1000, confidence=95)

elif atlas_cb_name == 'cb_voxel':
    h2 = mytool.heritability.heritability(mz, dz, n_bootstrap=None, confidence=95)
    if index == 'myelin':
        # save nii
        img_data = np.zeros(cb_mask.shape)
        img_data[cb_mask] = h2
        save_path = os.path.join(index_dir, '{0}_cb_voxel_-_h2.nii.gz'.format(index))
        img = nib.Nifti1Image(img_data, None)
        nib.save(img, save_path)
        subprocess.call('fslcpgeom {0} {1} -d'.format(cb_mask_mni_path, save_path), shell=True)
        subprocess.call('flirt -in {0} -ref {1} -out {2}'.format(save_path, suit_path, save_path.split('.', 1)[0]+'_suit.nii.gz'), shell=True)
    #        subprocess.call('fslmaths {0}_suit.nii.gz -mas {1} {0}_suit_mask.nii.gz'.format(save_path.split('.', 1)[0], suit_path), shell=True)
        
    elif index == 'fALFF':
        # save nii
        brain_models = mytool.mri.CiftiReader(cb_mask_fslr_path).brain_models()   
        img_data = h2
        save_path = os.path.join(index_dir, '{0}_cb_voxel_-_h2.dscalar.nii'.format(index))
        mytool.mri.save2cifti(save_path, img_data[None,...], brain_models, volume=mytool.mri.CiftiReader(cb_mask_fslr_path).volume)
        subprocess.check_output('wb_command -cifti-separate {0} COLUMN -volume-all {1}_cbonly.nii.gz'.format(save_path, save_path.split('.')[0]), shell=True)
        subprocess.call('flirt -in {0}_cbonly.nii.gz -ref {1} -out {0}_cbonly_suit.nii.gz'.format(save_path.split('.')[0], suit_path), shell=True)


elif atlas_cc_name == 'cc_voxel':
    h2 = mytool.heritability.heritability(mz, dz, n_bootstrap=None, confidence=95)
    # save cifti
    brain_models = mytool.mri.CiftiReader(cc_mask_fslr_path).brain_models()        
    img_data = h2       
    save_path = os.path.join(index_dir, '{0}_-_cc_voxel_h2.dscalar.nii'.format(index))
    mytool.mri.save2cifti(save_path, img_data[None,...], brain_models)

elif atlas_cc_name == 'cc_msm':
    h2,_ = mytool.heritability.heritability(mz, dz, n_bootstrap=100, confidence=95)
    h2 = h2[0,:]
    # save cifti
    atlas = atlas_load('cc_msm')
    msm_a = np.zeros((1, atlas.data.shape[0]))
    for key in atlas.label_info['key'].astype(np.int):
        msm_a[0,atlas.data==key] = h2[key-1]    
    brain_models = mytool.mri.CiftiReader(cc_mask_fslr_path).brain_models()
    save_path = os.path.join(index_dir, '{0}_-_cc_msm_h2.dscalar.nii'.format(index))
    mytool.mri.save2cifti(save_path, msm_a, brain_models)
