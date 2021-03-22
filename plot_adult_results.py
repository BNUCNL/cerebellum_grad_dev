
#%%
import nibabel as nib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import copy
import cb_tools

# %%
# set results path
results_dir = 'Z://server_temp//cb_results'
sub_adult = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub':np.str})

# set atlas path
atlas_dir = os.path.join(os.getcwd(), 'atlas')
atlas_mni_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')
cb_mask_mni_path = os.path.join(atlas_dir,'Cerebellum-MNIfnirt-maxprob-thr25.nii')

# get cerebellum mask
cb_mask = nib.load(cb_mask_mni_path).get_fdata()
cb_mask[cb_mask!=0] = 1
cb_mask = np.asarray(cb_mask,dtype=np.bool)

# %%  data prepare
index = 'myelin'  # ['myelin', 'fALFF']
index_dir = os.path.join(results_dir, index)

# prepare voxel data
data_adult_voxel = nib.load(os.path.join(index_dir, 'HCP-Adult', '{0}_cb_voxel_-_thr_mean_onlylobues.nii.gz').format(index)).get_fdata()
data_adult_voxel *= cb_mask

# prepare roi data
atlas_cb = {'myelin' : 'cb_anat_fsl', 'fALFF' : 'cb_anat_cifti'}
atlas = cb_tools.atlas_load(atlas_cb[index], atlas_dir)
data_roi = pd.read_csv(os.path.join(index_dir, '{0}_{1}_-_thr.csv'.format(index, atlas_cb[index])))
data_adult_roi = data_roi.merge(sub_adult, on='Sub', how='inner')

# %% Fig 1/2 B
# violin plot of cb lobules
num_str_col = 6
data = copy.deepcopy(data_adult_roi)

data.loc[:,data.columns[:-num_str_col]] = cb_tools.thr_IQR(
    data.loc[:,data.columns[:-num_str_col]].values, times=1.5, series=True)  # remove outliers outside 1.5 IQR

data_stack = pd.melt(data, id_vars=data.columns[-num_str_col:], 
                     var_name=['lobule'], value_vars=data.columns[:-num_str_col], value_name=index)
data_stack = pd.concat([data_stack, data_stack['lobule'].str.split('_',expand=True).rename(columns={0:'roi',1:'hemi'})], axis=1)

# plot
row = 'roi'
value = index
row_order = atlas.label_info['lobule'][:18:2]
palette_cb = sns.diverging_palette(230,230, l=80, center='dark', n=len(atlas.label_info['lobule'][:18:2]))

fig, ax = plt.subplots(figsize=(3.5, 4.5))
sns.violinplot(x=value, y=row, palette=palette_cb, order=row_order, fliersize=1,
                    cut=0, bw=.2, whis=3, linewidth=1, data=data_stack, ax=ax)
plt.tight_layout()

# %% Fig 1/2 C
# 3d plot for cerebellum 
data = copy.deepcopy(data_adult_voxel) 
colormap = 'viridis'
clip_pct = [0, 1] # thr by percentile
vmin, vmax = 1.3, 1.8 # thr by absolute valus
fig = plt.figure()
   
location = cb_tools.select_surf(cb_mask * data, tolerance=0, linkage=[5,4,3,2,1])
hue_data = data[location[:,0], location[:,1], location[:,2]]
hue_data_thr = np.clip(hue_data, np.quantile(hue_data,clip_pct[0]), np.quantile(hue_data,clip_pct[1]))

ax = fig.add_subplot(111, projection='3d')   
s = ax.scatter(location[:,0], location[:,1], location[:,2], c=hue_data_thr, cmap=colormap, s=80, marker='o', alpha=1, vmin=vmin, vmax=vmax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

lim_range = np.max(np.asarray(np.where(data)).max(-1) - np.asarray(np.where(data)).min(-1))
ax.set_xlim(np.asarray(np.where(data))[0,:].min()-5, np.asarray(np.where(data))[0,:].min()+lim_range+5)
ax.set_ylim(np.asarray(np.where(data))[1,:].min()-5, np.asarray(np.where(data))[1,:].min()+lim_range+5)
ax.set_zlim(np.asarray(np.where(data))[2,:].min()-5, np.asarray(np.where(data))[2,:].min()+lim_range+5)

ax.grid(False)
plt.axis('off')
plt.colorbar(s)

