
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

# %%
# prepare roi data
data_t1wT2wRatio = pd.read_csv(os.path.join(results_dir, 't1wT2wRatio', 't1wT2wRatio_cb_anat_fsl.csv'))
data_falff = pd.read_csv(os.path.join(results_dir, 'fALFF', 'fALFF_cb_anat_cifti.csv'))

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

# %%
def inner_sub(dataframe_list):
    sub = set(dataframe_list[0]['Sub'].tolist())
    for dataframe in dataframe_list[1:]:
        sub = sub & set(dataframe['Sub'].tolist())
    return list(sub)       
    
#sub = inner_sub([data_ml, data_alff, sub_adult])
sub = inner_sub([data_t1wT2wRatio.query('`Age in years` < 20'), data_falff.query('`Age in years` < 20')])
sub_df = pd.DataFrame(sub, columns=['Sub'])
data_t1wT2wRatio_coup = data_t1wT2wRatio.merge(sub_df, on='Sub', how='inner')
data_falff_coup = data_falff.merge(sub_df, on='Sub', how='inner')

# %%
def isfc(data1, data2):
    from scipy.spatial.distance import cdist

    """Cal functional connectivity between data1 and data2.

    Parameters
    ----------
        data1: used to calculate functional connectivity,
            shape = [n_samples1, n_features].
        data2: used to calculate functional connectivity,
            shape = [n_samples2, n_features].

    Returns
    -------
        isfc: functional connectivity map of data1 and data2,
            shape = [n_samples1, n_samples2].

    Notes
    -----
        1. data1 and data2 should both be 2-dimensional.
        2. n_features should be the same in data1 and data2.
    """

    corr = np.nan_to_num(1 - cdist(data1, data2, metric='correlation'))
    return corr

# %% plot Fig 3A
dev_coup = isfc(data_t1wT2wRatio_coup.iloc[:,:-num_str_col].values.T,
                data_falff_coup.iloc[:,:-num_str_col].values.T)
# isc
isc_df = pd.DataFrame(np.c_[np.diag(dev_coup) , data_falff_coup.columns[:-num_str_col]], columns=['coup', 'roi']) 
fig, ax = plt.subplots(figsize=(3.5, 4.5))
sns.barplot(x='coup', y='roi', palette=palette_cb, linewidth=1, data=isc_df, ax=ax)

ax.tick_params(colors='gray', which='both')
[ax.spines[k].set_color('darkgray') for k in ['top','bottom','left','right']]
plt.tight_layout()

# %%
# isfc
r2 = dev_coup ** 2

fig, axes = plt.subplots(nrows=2,ncols=np.int(np.ceil(r2.shape[0]/2)), 
                         sharey=True, subplot_kw={'projection':'polar'})
angles = [i / r2.shape[0] *2*np.pi + 2*np.pi/r2.shape[0]*0.5 for i in range(r2.shape[0])] 
angles += angles[:1]

ax = axes.flat[0]

for i, ax in enumerate(axes.flat[:r2.shape[0]]):
    if i <= r2.shape[0] //2:
        i = r2.shape[0] //2 -i 

    ax.bar(angles, np.r_[r2[i,:], r2[i,0]],color=palette_cb.as_hex())
    ax.set_xticks(angles)
    ax.set_xticklabels(np.r_[data_falff_coup.columns[:-num_str_col].values, np.asarray(data_falff_coup.columns[0])],
                              fontsize=10)
    ax.set_yticks(np.arange(0, 0.09, 0.04))
    ax.set_ylim([0, 0.08])
    ax.tick_params(axis='y', colors='gray', labelsize=10)
    ax.grid(color='lightgray')
    ax.set_title(data_falff_coup.columns[i], fontweight='bold')

plt.tight_layout()
