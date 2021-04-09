
#%%
import nibabel as nib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import cb_tools
from pygam import LinearGAM

# %%
# set results path
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev'
index = 'fALFF'  # ['t1wT2wRatio', 'fALFF']

#%% get cerebellum mask
atlas_dir = os.path.join(os.getcwd(), 'atlas')
atlas_mni_path = os.path.join(atlas_dir, 'MNI152_T1_2mm_brain.nii.gz')
cb_mask_mni_path = os.path.join(atlas_dir,'Cerebellum-MNIfnirt-maxprob-thr25.nii')

cb_mask = nib.load(cb_mask_mni_path).get_fdata()
cb_mask[cb_mask!=0] = 1
cb_mask = np.asarray(cb_mask,dtype=np.bool)

# %%  data prepare
index_dir = os.path.join(results_dir, index)
sub_adult = pd.read_csv(os.path.join(results_dir, 'sub_adult'), header=0, dtype={'Sub': np.str})
sub_dev = pd.read_csv(os.path.join(results_dir, 'sub_dev'), header=0, dtype={'Sub': np.str})
num_str_col = 6

# prepare roi data
atlas_cb = {'t1wT2wRatio' : 'cb_anat_fsl', 'fALFF' : 'cb_anat_cifti'}
atlas = cb_tools.atlas_load(atlas_cb[index], atlas_dir)
data_roi = pd.read_csv(os.path.join(index_dir, '{0}_{1}.csv'.format(index, atlas_cb[index])))
data_roi = data_roi.astype({'Sub': np.str})
data_adult_roi = data_roi.merge(sub_adult, on='Sub', how='inner')
data_dev_roi = data_roi.merge(sub_dev, on='Sub', how='inner')

# prepare voxel data
data_adult_voxel = nib.load(os.path.join(index_dir, 'HCP-Adult', '{0}_cb_voxel.nii.gz').format(index)).get_fdata()
data_adult_voxel *= cb_mask
data_dev_voxel = nib.load(os.path.join(index_dir, 'HCP-D', '{0}_cb_voxel.nii.gz').format(index)).get_fdata()
data_dev_voxel *= cb_mask

#
palette_cb = sns.diverging_palette(230,230, l=80, center='dark', n=len(atlas.label_info['lobule'][:18:2]))

# %% Fig 1/2 B
# violin plot of cb lobules
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

fig, ax = plt.subplots(figsize=(3, 3.5))
sns.violinplot(x=value, y=row, palette=palette_cb, order=row_order, fliersize=1,
                    cut=0, bw=.2, whis=3, linewidth=1, data=data_stack, ax=ax)

ax.tick_params(colors='gray', which='both')
[ax.spines[k].set_color('darkgray') for k in ['top','bottom','left','right']]
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

# %% roi results
def hemi_merging(x, num_str_col):
    x_l = x[[x.columns[i] for i in range(x.shape[-1]) if x.columns[i].split('_')[-1]=='l']]
    x_l.rename(columns=lambda x: x.split('_')[0], inplace=True)
    
    x_r = x[[x.columns[i] for i in range(x.shape[-1]) if x.columns[i].split('_')[-1]=='r']]   
    x_r.rename(columns=lambda x: x.split('_')[0], inplace=True)

    x_merged = pd.concat([pd.concat([x_l,x_r]).mean(level=0),x.iloc[:,-num_str_col:]], axis=1)
    
    return x_merged

dev = hemi_merging(data_dev_roi, num_str_col)
adult = hemi_merging(data_adult_roi, num_str_col)
lobues_name = atlas.label_info['lobule'][:18:2]

# %% Fig 1/2 D
# average gradient in children and adults groups
dev_mean = np.nanmean(cb_tools.thr_IQR(dev.loc[:,dev.columns[:-num_str_col]].values, times=1.5, series=True), 0)
adult_mean = np.nanmean(cb_tools.thr_IQR(adult.loc[:,adult.columns[:-num_str_col]].values, times=1.5, series=True), 0)

plt.plot(lobues_name, adult_mean, c='seagreen', label='adult')
plt.bar(lobues_name, adult_mean, color='seagreen', alpha=0.8)
plt.plot(lobues_name, dev_mean, c='palegreen', ls='--', label='child')
plt.bar(lobues_name, dev_mean, color='palegreen', alpha=0.5)
plt.ylim([1.5,2.3])
plt.legend()

# %% Fig 1/2 E, F
# dev trojetory of curvature and extreme point

def gradient_magnitude(x):
    deg = 2
    polyfit = np.asarray([np.polyfit(np.arange(x.shape[-1]) - (x.shape[-1]-1)/2, 
                                     x.iloc[i, :].to_numpy(dtype=float), deg=deg) for i in range(x.shape[0])])
    polyfit_df = pd.DataFrame(polyfit, columns=['a','b','c'])
    polyfit_df['p1'] = -polyfit_df['b']/(2*polyfit_df['a'])
    
    return polyfit_df

# get curvature and extreme point
deg = 2
data = copy.deepcopy(dev)
x = 'Age in months'
y = ['a', 'p1']

shape = gradient_magnitude(data.iloc[:,:-num_str_col])
shape = pd.concat((shape, data.iloc[:,-num_str_col:]), axis=1)

data = copy.deepcopy(shape)
data[data.columns[:-num_str_col]] = cb_tools.thr_IQR(data[data.columns[:-num_str_col]].values, times=3, series=True) # remove outliers
data.dropna(inplace=True)
data_g = data.groupby(['Age in years']).mean().loc[:, data.columns[:-num_str_col]]

# plot dev trajactory
_, axes = plt.subplots(nrows=1, ncols=2, figsize=[8,3.3])  
plt.subplots_adjust(bottom=0.2, wspace=0.3)
for i, yi in enumerate(y):

    sns.scatterplot(data[x]/12, data[yi], s=10, color='mediumaquamarine', ax=axes[i])
    gam = LinearGAM(n_splines=5).gridsearch(data[x].values[...,None]/12, data[yi].values)
    xx = gam.generate_X_grid(term=0, n=500)
        
    axes[i].plot(xx, gam.predict(xx), '--', color='seagreen')
    axes[i].plot(xx, gam.prediction_intervals(xx, width=.95), color='mediumaquamarine', ls='--', alpha=0.5)
    axes[i].scatter(data_g.index, data_g[yi], c='seagreen', s=15, marker='D')
    
    axes[i].set_xticks(np.arange(8, 23, 2))
    axes[i].set_xlim([7,23])
    axes[i].tick_params(colors='gray', which='both')
    [axes[i].spines[k].set_color('darkgray') for k in ['top','bottom','left','right']]
    
    if yi == 'a':
        axes[i].set_yticks(np.arange(0.02, 0.08, 0.02))
    elif yi == 'p1':
        axes[i].set_yticks([-1,0,1])
        axes[i].set_yticklabels(['CrusI', 'CrusII', 'VIIb'])
        axes[i].set_ylim([-1,1])    
        axes[i].invert_yaxis()

# %% Fig 1/2 G, H
# dev trojetory for each lobule
order = 1

data = copy.deepcopy(dev)
data[data.columns[:-num_str_col]] = cb_tools.thr_IQR(data[data.columns[:-num_str_col]].values, times=1.5, series=True)
data_g = data.groupby(['Age in years']).mean().loc[:, data.columns[:-num_str_col]]

sns.set_palette(palette_cb)

_, ax = plt.subplots(figsize=[4,3])   
[sns.regplot(data['Age in months'] / 12, data[i], order=order, scatter=False, line_kws={'lw':1}) for i in data.columns[:-num_str_col]]
[sns.scatterplot(data_g.index, data_g[i], s=10, marker='D') for i in data_g.columns]

ax.set_xlim([7,23])
ax.set_xticks(np.arange(8, 23, 2))
ax.set_yticks(np.arange(1.6, 2.3, 0.2))
ax.tick_params(colors='gray', which='both')
[ax.spines[k].set_color('darkgray') for k in ['top','bottom','left','right']]
plt.tight_layout()

# %% Fig 1/2 H & I
# k 
data = copy.deepcopy(dev)
data[data.columns[:-num_str_col]] = cb_tools.thr_IQR(data[data.columns[:-num_str_col]].values.T, times=1.5, series=True).T
nan_voxel = np.isnan(data[data.columns[:-num_str_col]]).sum(0)>len(data)*0.5
data.loc[:, np.r_[nan_voxel, np.zeros(num_str_col).astype(np.bool)]] = 0

x = 'Age in months'
y = data.columns[:-num_str_col]

order = 1
polyfit = np.asarray([np.polyfit(data.loc[~np.isnan(data[y_i]), x] / 12, data.loc[~np.isnan(data[y_i]), y_i], deg=order) for y_i in y])
polyfit[np.isnan(data[data.columns[:-num_str_col]].values).sum(0)>0.3*(len(data)-num_str_col)] = np.nan
polyfit_df = pd.DataFrame(polyfit, columns=['a','b'])

k = polyfit_df['a']
plt.bar(lobues_name, k, color=palette_cb)

# hist map for each lobula
if atlas_cb_name == 'cb_voxel':
    if index == 'myelin':
        # save nii
        polyfit_img = np.zeros(cb_mask.shape)
        polyfit_img[cb_mask] = polyfit_df['a'].values
        save_path = os.path.join(index_dir, '{0}_cb_voxel_a_deg{1}.nii.gz'.format(index, order))
        img = nib.Nifti1Image(polyfit_img, None)
        nib.save(img, save_path)
        subprocess.call('fslcpgeom {0} {1} -d'.format(cb_mask_mni_path, save_path), shell=True)
        subprocess.call('flirt -in {0} -ref {1} -out {2}'.format(save_path, suit_path, save_path.split('.', 1)[0]+'_suit.nii.gz'), shell=True)
    #        subprocess.call('fslmaths {0}_suit.nii.gz -mas {1} {0}_suit_mask.nii.gz'.format(save_path.split('.', 1)[0], suit_path), shell=True)
        
    elif index == 'fALFF':
        # save nii
        brain_models = mytool.mri.CiftiReader(cb_mask_fslr_path).brain_models()   
        polyfit_img = polyfit_df['a'].values
        save_path = os.path.join(index_dir, '{0}_cb_voxel_a_deg{1}.dscalar.nii'.format(index, order))
        mytool.mri.save2cifti(save_path, polyfit_img[None,...], brain_models, volume=mytool.mri.CiftiReader(cb_mask_fslr_path).volume)
        subprocess.check_output('wb_command -cifti-separate {0} COLUMN -volume-all {1}_cbonly.nii.gz'.format(save_path, save_path.split('.')[0]), shell=True)
        subprocess.call('flirt -in {0}_cbonly.nii.gz -ref {1} -out {0}_cbonly_suit.nii.gz'.format(save_path.split('.')[0], suit_path), shell=True)
