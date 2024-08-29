#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:12:50 2020

@author: liuxingyu
"""

import nibabel as nib
import os
import numpy as np
import pandas as pd
from nibabel.cifti2 import cifti2
import copy
from scipy import stats

#%%
class CiftiReader(object):
    """ copy from freeroi by Xiayu CHEN """

    def __init__(self, file_path):
        self.full_data = cifti2.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    @property
    def brain_structures(self):
        return [_.brain_structure for _ in self.header.get_index_map(1).brain_models]

    @property
    def label_info(self):
        """
        Get label information from label tables
        Return:
        ------
        label_info[list]:
            Each element is a dict about corresponding map's label information.
            Each dict's content is shown as below:
                key[list]: a list of integers which are data values of the map
                label[list]: a list of label names
                rgba[ndarray]: shape=(n_label, 4)
                    The four elements in the second dimension are
                    red, green, blue, and alpha color components for label (between 0 and 1).
        """
        label_info = []
        for named_map in self.header.get_index_map(0).named_maps:
            label_dict = {'key': [], 'label': [], 'rgba': []}
            for k, v in named_map.label_table.items():
                label_dict['key'].append(k)
                label_dict['label'].append(v.label)
                label_dict['rgba'].append(v.rgba)
            label_dict['rgba'] = np.asarray(label_dict['rgba'])
            label_info.append(label_dict)

        return label_info

    @property
    def volume(self):
        return self.header.get_index_map(1).volume

    def brain_models(self, structures=None):
        """
        get brain model from cifti file
        Parameter:
        ---------
        structures: list of str
            Each structure corresponds to a brain model.
            If None, get all brain models.
        Return:
        ------
            brain_models: list of Cifti2BrainModel
        """
        brain_models = list(self.header.get_index_map(1).brain_models)
        if structures is not None:
            if not isinstance(structures, list):
                raise TypeError("The parameter 'structures' must be a list")
            brain_models = [brain_models[self.brain_structures.index(s)] for s in structures]
        return brain_models

    def map_names(self, rows=None):
        """
        get map names
        Parameters:
        ----------
        rows: sequence of integer
            Specify which map names should be got.
            If None, get all map names
        Return:
        ------
        map_names: list of str
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                map_names = [named_map.map_name for named_map in named_maps]
            else:
                map_names = [named_maps[i].map_name for i in rows]
        else:
            map_names = []
        return map_names

    def label_tables(self, rows=None):
        """
        get label tables
        Parameters:
        ----------
        rows: sequence of integer
            Specify which label tables should be got.
            If None, get all label tables.
        Return:
        ------
        label_tables: list of Cifti2LableTable
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                label_tables = [named_map.label_table for named_map in named_maps]
            else:
                label_tables = [named_maps[i].label_table for i in rows]
        else:
            label_tables = []
        return label_tables

    def get_data(self, structure=None, zeroize=False):
        """
        get data from cifti file
        Parameters:
        ----------
        structure: str
            One structure corresponds to one brain model.
            specify which brain structure's data should be extracted
            If None, get all structures, meanwhile ignore parameter 'zeroize'.
        zeroize: bool
            If true, get data after filling zeros for the missing vertices/voxels.
        Return:
        ------
        data: numpy array
            If zeroize doesn't take effect, the data's shape is (map_num, index_num).
            If zeroize takes effect and brain model type is SURFACE, the data's shape is (map_num, vertex_num).
            If zeroize takes effect and brain model type is VOXELS, the data's shape is (map_num, i_max, j_max, k_max).
        map_shape: tuple
            the shape of the map.
            If brain model type is SURFACE, the shape is (vertex_num,).
            If brain model type is VOXELS, the shape is (i_max, j_max, k_max).
            Only returned when 'structure' is not None and zeroize is False.
        index2v: list
            index2v[cifti_data_index] == map_vertex/map_voxel
            Only returned when 'structure' is not None and zeroize is False.
        """

        _data = np.array(self.full_data.get_data())
        if structure is not None:
            brain_model = self.brain_models([structure])[0]
            offset = brain_model.index_offset
            count = brain_model.index_count

            if zeroize:
                if brain_model.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                    n_vtx = brain_model.surface_number_of_vertices
                    data = np.zeros((_data.shape[0], n_vtx), _data.dtype)
                    data[:, list(brain_model.vertex_indices)] = _data[:, offset:offset+count]
                elif brain_model.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                    # This function have not been verified visually.
                    vol_shape = self.header.get_index_map(1).volume.volume_dimensions
                    data_shape = (_data.shape[0],) + vol_shape
                    data_ijk = np.array(list(brain_model.voxel_indices_ijk))
                    data = np.zeros(data_shape, _data.dtype)
                    data[:, data_ijk[:, 0], data_ijk[:, 1], data_ijk[:, 2]] = _data[:, offset:offset+count]
                else:
                    raise RuntimeError("The function can't support the brain model: {}".format(brain_model.model_type))
                return data
            else:
                if brain_model.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                    map_shape = (brain_model.surface_number_of_vertices,)
                    index2v = list(brain_model.vertex_indices)
                elif brain_model.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                    # This function have not been verified visually.
                    map_shape = self.header.get_index_map(1).volume.volume_dimensions
                    index2v = list(brain_model.voxel_indices_ijk)
                else:
                    raise RuntimeError("The function can't support the brain model: {}".format(brain_model.model_type))
                return _data[:, offset:offset+count], map_shape, index2v
        else:
            return _data


def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
    """ copy from freeroi by Xiayu CHEN
    Save data as a cifti file
    If you just want to simply save pure data without extra information,
    you can just supply the first three parameters.
    NOTE!!!!!!
        The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
        Nibabel-2.3.0 can support for Cifti2Image indeed.
        And the header will be regard as Nifti2Header when loading cifti file by nibabel earlier than 2.3.0.
    Parameters:
    ----------
    file_path: str
        the output filename
    data: numpy array
        An array with shape (maps, values), each row is a map.
    brain_models: sequence of Cifti2BrainModel
        Each brain model is a specification of a part of the data.
        We can always get them from another cifti file header.
    map_names: sequence of str
        The sequence's indices correspond to data's row indices and label_tables.
        And its elements are maps' names.
    volume: Cifti2Volume
        The volume contains some information about subcortical voxels,
        such as volume dimensions and transformation matrix.
        If your data doesn't contain any subcortical voxel, set the parameter as None.
    label_tables: sequence of Cifti2LableTable
        Cifti2LableTable is a mapper to map label number to Cifti2Label.
        Cifti2Lable is a specification of the label, including rgba, label name and label number.
        If your data is a label data, it would be useful.
    """
    if file_path.endswith('.dlabel.nii'):
        assert label_tables is not None
        idx_type0 = 'CIFTI_INDEX_TYPE_LABELS'
    elif file_path.endswith('.dscalar.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SCALARS'
    else:
        raise TypeError('Unsupported File Format')

    if map_names is None:
        map_names = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

    if label_tables is None:
        label_tables = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

    # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
    # and this index_map always contains some scalar information, such as named_maps.
    # We can get label_table and map_name and metadata from named_map.
    mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0)
    for mn, lbt in zip(map_names, label_tables):
        named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
        mat_idx_map0.append(named_map)

    # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
    # and this index_map always contains some brain_structure information, such as brain_models and volume.
    mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    for bm in brain_models:
        mat_idx_map1.append(bm)
    if volume is not None:
        mat_idx_map1.append(volume)

    matrix = cifti2.Cifti2Matrix()
    matrix.append(mat_idx_map0)
    matrix.append(mat_idx_map1)
    header = cifti2.Cifti2Header(matrix)
    img = cifti2.Cifti2Image(data, header)
    cifti2.save(img, file_path)

#%%
class Atlas:
    """ roi atlas
    
    attributes:
        data:
        
        label_info:[DataFrame]
    """

    def __init__(self, data, label_info=None):
             
        self.data = data
        self.label_info = label_info
        
        # if data is not None:
        #     label = np.asarray(np.unique(self.data), dtype=np.int)
        #     self.label =  np.asarray(label[label!=0])
            
        #     self.label_num = np.size(self.label)

        #     roi_size = [np.sum(~np.isnan(self.data[self.data==i])) for i 
        #                 in self.label]
        #     self.roi_size = roi_size

        
#%% altas load
def atlas_load(atlas_name, atlas_dir, atlas_data=None):

    if atlas_name in ['cb_anat_fsl', 'cb_anat_cifti']:
        if atlas_name == 'cb_anat_fsl':
            atlas_data = nib.load(os.path.join(
                    atlas_dir,'Cerebellum-MNIfnirt-maxprob-thr25.nii')).get_fdata()
        
        elif atlas_name == 'cb_anat_cifti':
            atlas_data = CiftiReader(os.path.join(
                        atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25.dscalar.nii')).get_data()[0,:]
        
        atlas_info = pd.read_table(os.path.join(atlas_dir, 'Cerebellum-MNIfnirt.txt'), 
                                            sep='\t', header=None)
        atlas_info.columns = ['key','name','lobule','hemi', 'landmark']
        atlas_info['landmark'] = atlas_info['landmark'].apply(lambda x: np.array(eval(x)))
        atlas = Atlas(data=atlas_data, label_info=atlas_info)
    
    elif atlas_name in ['cb_acapulco_adult', 'cb_acapulco_adult-2mm']:
          
        atlas_info = pd.read_table(os.path.join(atlas_dir, 'acapulco_adult.txt'), 
                                            sep='\t', header=None)
        atlas_info.columns = ['key','name','lobule','hemi']
        atlas_info = atlas_info.astype({'key': np.int})
        atlas = Atlas(data=atlas_data, label_info=atlas_info)       

    elif atlas_name in ['cb_acapulco_pediatric', 'cb_acapulco_pediatric-2mm']:
          
        atlas_info = pd.read_table(os.path.join(atlas_dir, 'acapulco_pediatric.txt'), 
                                            sep='\t', header=None)
        atlas_info.columns = ['key','name','lobule','hemi']
        atlas_info = atlas_info.astype({'key': np.int})
        atlas = Atlas(data=atlas_data, label_info=atlas_info)    

    elif atlas_name == 'cb_anat_7t':
        atlas_data = nib.load(os.path.join(
                atlas_dir,'CHROMA_lobules_cortex_map.nii.gz')).get_fdata()
        atlas_info = pd.read_table(os.path.join(atlas_dir, 'CHROMA.txt'), 
                                           sep='\t', header=None)
        atlas_info.columns = ['key','name','lobule','hemi']
        atlas_info = atlas_info.astype({'key': np.int})
        atlas = Atlas(data=atlas_data, label_info=atlas_info)              

    elif atlas_name == 'cc_msm':
        atlas_data = CiftiReader(os.path.join(
            atlas_dir, 'MMP_mpmLR32k.dlabel.nii'))
        atlas_info = pd.read_table(os.path.join(atlas_dir, 'MMP_mpmLR32k.txt'), 
                                           sep='\t', header=0)
        color = atlas_info['color'].str.split().values
        atlas_info['color'] = [np.asarray(color[i], dtype=np.int) for i in range(len(color))]
        atlas = Atlas(data=atlas_data.get_data()[0,:], label_info=atlas_info[['key','name','hemi','roi','color', 'rs_network']])

    elif atlas_name in ['cb_func_7nw', 'cb_func_7nw_cifti']:

        if atlas_name == 'cb_func_7nw':
            atlas_data = nib.load(os.path.join(
                atlas_dir, 'Buckner_7Networks_MNIwholebrain.nii.gz')).get_fdata()
        elif atlas_name == 'cb_func_7nw_cifti':
            atlas_data = CiftiReader(os.path.join(
                atlas_dir, 'Buckner_7Networks_MNIwholebrain.32k_fs_LR.dscalar.nii')).get_data()

        atlas_info = pd.read_table(os.path.join(atlas_dir, 'Buckner_7Networks.txt'), 
                                           sep='\t', header=None, skiprows=1)
        atlas_info.columns = ['key','name', 'color', 'unknown']
        color = atlas_info['color'].str.split().values
        atlas_info['color'] = [np.asarray(color[i], dtype=np.int) for i in range(len(color))]
        atlas = Atlas(data=atlas_data, label_info=atlas_info[['key','name','color']])

    return atlas

#%% only render surface for visualization
def select_surf(data, tolerance=0, linkage=None):
    """
    get surface location for 3d mni data to reduce resource consumption for rendering.

    Parameters
    ----------
    data : 3d-array
        mni data
    tolerance : int
        the number of surface neighboring voxels to retain
    likage ： list
        retain one voxel, if the number of 1-degree adjacent voxels that was not 0 is 
        larger than the specified linkage. Ineractively performed.
        
    Returns
    -------
    locations: 2d-array
        [n_voxels, 3]
    """
    def linkage_select(data, loc, linkage):
        linkage_index = np.zeros(loc.shape[0])
        for i, loc_i in enumerate(list(loc)):
            if (data != 0)[
                    loc_i[0]-1:loc_i[0]+2, 
                    loc_i[1]-1:loc_i[1]+2, 
                    loc_i[2]-1:loc_i[2]+2].sum() > linkage:
                linkage_index[i] = 1
        return linkage_index
        
    loc = np.asarray(np.where(data)).T
    loc_surf_index = np.zeros(loc.shape[0])
                
    for i, loc_i in enumerate(list(loc)):
        samexy_z = np.where(data[loc_i[0],loc_i[1],:])[0]
        samexz_y = np.where(data[loc_i[0],:,loc_i[2]])[0]
        sameyz_x = np.where(data[:,loc_i[1],loc_i[2]])[0]
        
        if (loc_i[0] >= sameyz_x.max()-tolerance or loc_i[0] <= sameyz_x.min()+tolerance or 
            loc_i[1] >= samexz_y.max()-tolerance or loc_i[1] <= samexz_y.min()+tolerance or
            loc_i[2] >= samexy_z.max()-tolerance or loc_i[2] <= samexy_z.min()+tolerance):
            loc_surf_index[i] = 1
        
    if linkage is not None:
        linkage_index = np.ones(loc.shape[0])
        data_link = copy.deepcopy(data)
        for i in linkage:
            remain = ~(linkage_index.astype(np.bool))
            data_link[loc[remain, 0], loc[remain, 1], loc[remain, 2]] = 0
            linkage_index *= linkage_select(data_link, loc, linkage=i)
        loc_surf_index *= linkage_index
        
    return loc[loc_surf_index.astype(np.bool)]


#%% threshold by n times IQR
def thr_IQR(x, times=3, series=False, exclude_zero=True):
    # if series is True, the last axis should be series
    
    if series is False:
        x = x[...,None]

    if exclude_zero is True:
        qu = np.asarray([np.nanquantile(x[..., i][x[..., i]!=0], 0.75) for i in range(x.shape[-1])])
        ql = np.asarray([np.nanquantile(x[..., i][x[..., i]!=0], 0.25) for i in range(x.shape[-1])])
    else:
        qu = np.asarray([np.nanquantile(x[..., i], 0.75) for i in range(x.shape[-1])])
        ql = np.asarray([np.nanquantile(x[..., i], 0.25) for i in range(x.shape[-1])])      
        
    x_post = copy.deepcopy(x)
    x_post[x_post > (qu + times*(qu-ql))] = np.nan
    x_post[x_post < (ql - times*(qu-ql))] = np.nan
    
    if series is False:
        return x_post[...,0]
    else:
        return x_post
    
#%%
def roiing_volume(roi_annot, data, method='nanmean', key=None):

    if key is not None:
        roi_key = key
    else:
        roi_key = np.asarray(np.unique(roi_annot), dtype=np.int)
        roi_key = roi_key[roi_key != 0]
    
    roi_data = []
    
    for i in roi_key:
        # ignore nan
        if method == 'nanmean':
            roi_data.append(np.nanmean(data[roi_annot==i], 0))
        elif method == 'nanmedian':
            roi_data.append(np.nanmedian(data[roi_annot==i], 0))  
        elif method == 'nanstd':
            roi_data.append(np.nanstd(data[roi_annot==i], 0))        
        elif method == 'nanmax':
            roi_data.append(np.nanmax(data[roi_annot==i], 0))
        elif method == 'nanmin':
            roi_data.append(np.nanmin(data[roi_annot==i], 0))
        elif method == 'nansize':
            roi_data.append(np.sum(~np.isnan(data[roi_annot==i])))
    
    roi_data = np.asarray(roi_data)
    return roi_key, roi_data


#%% heritability

def get_twins_id(src_file, trg_file=None):
    """
    Get twins ID according to 'ZygosityGT' and pair the twins according to
    'Family_ID' from HCP restricted information.
    Parameters
    ----------
    src_file : str
        HCP restricted information file (CSV format)
    trg_file : str
        If is not None, save twins ID information to a file (CSV format)
    Returns
    -------
    df_out : DataFrame
        twins ID information
    """
    assert src_file.endswith('.csv')
    zygosity = ('MZ', 'DZ')
    df_in = pd.read_csv(src_file)

    df_out = {'twin1': [], 'twin2': [], 'zygosity': [], 'familyID': []}
    for zyg in zygosity:
        df_zyg = df_in[df_in['ZygosityGT'] == zyg]
        family_ids = sorted(set(df_zyg['Family_ID']))
        for fam_id in family_ids:
            subjs = df_zyg['Subject'][df_zyg['Family_ID'] == fam_id]
            subjs = subjs.reset_index(drop=True)
            assert len(subjs) == 2
            df_out['twin1'].append(subjs[0])
            df_out['twin2'].append(subjs[1])
            df_out['zygosity'].append(zyg)
            df_out['familyID'].append(fam_id)
    df_out = pd.DataFrame(df_out)

    if trg_file is not None:
        assert trg_file.endswith('.csv')
        df_out.to_csv(trg_file, index=False)

    return df_out


def count_twins_id(data):
    """
    Count the number of MZ or DZ pairs
    Parameters
    ----------
    data : DataFrame | str
        twins ID information
        If is str, it's a CSV file of twins ID information.
    """
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, str):
        data = pd.read_csv(data)
    else:
        raise TypeError('The input data must be a DataFrame or str!')

    zygosity = ('MZ', 'DZ')
    for zyg in zygosity:
        df_zyg = data[data['zygosity'] == zyg]
        print(f'The number of {zyg}:', len(df_zyg))


def filter_twins_id(data, limit_set, trg_file=None):
    """
    The twin pair will be removed as long as anyone of it is not in limit set
    Parameters
    ----------
    data : DataFrame | str
        twins ID information
        If is str, it's a CSV file of twins ID information.
    limit_set : collection
        a collection of subject IDs
    trg_file : str, default None
        If is not None, save filtered twins ID to a file (CSV format)
    Returns
    -------
    data : DataFrame
        filtered twins ID information
    """
    if isinstance(data, pd.DataFrame):
        data = data.copy()
    elif isinstance(data, str):
        data = pd.read_csv(data)
    else:
        raise TypeError('The input data must be a DataFrame or str!')

    # filter twins ID
    for idx in data.index:
        if (data['twin1'][idx] not in limit_set or
            data['twin2'][idx] not in limit_set):
            data.drop(index=idx, inplace=True)

    if trg_file is not None:
        assert trg_file.endswith('.csv')
        data.to_csv(trg_file, index=False)

    return data


def icc(x, n_permutation=None):
    '''
    Calculate intraclass correlation between two squences.
    Parameters
    ----------
    x : array-like, 2D or 3D
        2D shape: [2, n_sample]
        3D shape: [2, n_sample, n_features]
    n_permutation : positive integer
        If is not None, do permutation with n_permutation times.

    Returns
    -------
    r : float
        intraclass correlation

    References
    ----------
    https://github.com/noahbenson/hcp-lines/blob/master/notebooks/hcp-lines.ipynb
    '''

    assert x.shape[0] == 2

    if n_permutation is not None:
        r = icc(x)
        
        n = x.shape[1]
        rs = []
        for i in range(n_permutation):
            pair_idx = np.arange(n)
            np.random.shuffle(pair_idx)
            x_i = copy.deepcopy(x)
            x_i[0,:] = x[0, pair_idx]
            rs.append(icc(x_i))
        
        rs = np.asarray(rs)
        
        return r, rs

# =============================================================================
#     # ICC Class 3    
#     mu_t = np.nanmean(x, (0, 1))  
#     mu_b = np.nanmean(x, axis=0)
#     mu_w = np.nanmean(x, axis=1)
#     
#     ms_b = np.nansum(((mu_b - mu_t)**2) * 2, 0) / (x.shape[1] - 1)
#     ms_e = (np.nansum((x - mu_t)**2, (0,1)) - 
#             np.nansum((mu_b - mu_t)**2, (0)) * 2 - 
#             np.nansum((mu_w - mu_t)**2, (0)) * x.shape[1]) / (x.shape[1] - 1)
#     
#     r = (ms_b - ms_e) / (ms_b + ms_e)
# =============================================================================
    
    # ICC Class 1
    mu_b = np.nanmean(x, axis=0)
    ms_e = np.nansum((x - mu_b)**2, (0,1)) / x.shape[1]
    ms_b = np.nanvar(mu_b, axis=0, ddof=1)*2
    r = (ms_b - ms_e) / (ms_b + ms_e)
    
    return r


def heritability(mz, dz, n_permutation=None):
    '''
    heritability(mz, dz) yields Falconer's heritability index, h^2.
    Parameters
    ----------
    mz, dz: array-like, 2D or 3D
        2D shape: [2, n_sample]
        23 shape: [2, n_sample, n_features]
    n_permutation : positive integer
        If is not None, do permutation with n_permutation times.
    confidence : a list of number between 0 and 100
        It is used when n_bootstrap is not None.
        It determines the single-tail confidence boundary of the bootstrap. For example,
        [95] indicates the confidance boundary to be 95-percentile values.
    Returns
    -------
    h2 : float
        heritability
    percentile : float
        the percentile of h2 in the distribution of all permutations.
        Only returned when n_bootstrap is not None.
        
    References
    ----------
    https://github.com/noahbenson/hcp-lines/blob/master/notebooks/hcp-lines.ipynb
    '''

    if n_permutation is None:
        r_mz = icc(mz)
        r_dz = icc(dz)
        h2 = 2 * (r_mz - r_dz)
        return h2
    
    else:
        r_mz, rs_mz = icc(mz, n_permutation=n_permutation)
        r_dz, rs_dz = icc(dz, n_permutation=n_permutation)
        h2 = 2 * (r_mz - r_dz)
        h2s = 2 * (rs_mz - rs_dz)

        percentile = [stats.percentileofscore(h2s[:,i], h2[i]) for i in range(len(h2))]
        return h2, percentile

def isfc(data1, data2=None):
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
    if data2 is None:
        data2 = data1

    corr = np.nan_to_num(1 - cdist(data1, data2, metric='correlation'))
    return corr


# %%
def isc(data1, data2=None):

    """calculate inter-subject correlation along the determined axis.

    Parameters
    ----------

        data1: used to calculate functional connectivity,
            shape = [n_samples, n_features].
        data2: used to calculate functional connectivity,
            shape = [n_samples, n_features].

    Returns
    -------
        isc: point-to-point functional connectivity list of
            data1 and data2, shape = [n_samples, ].

    Notes
    -----
        1. data1 and data2 should both be 2-dimensional.
        2. [n_samples, n_features] should be the same in data1 and data2.

    """

    if data2 is None:
        data2 = data1
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)

    z_data1 = np.nan_to_num(stats.zscore(data1, axis=-1))
    z_data2 = np.nan_to_num(stats.zscore(data2, axis=-1))
    corr = np.sum(z_data1*z_data2, axis=-1)/(np.size(data1, -1))

    return corr
