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
        
        label = np.asarray(np.unique(self.data), dtype=np.int)
        self.label =  np.asarray(label[label!=0])
        
        self.label_num = np.size(self.label)

        roi_size = [np.sum(~np.isnan(self.data[self.data==i])) for i 
                    in self.label]
        self.roi_size = roi_size

        
#%% altas load
def atlas_load(atlas_name, atlas_dir):
    
    if atlas_name == 'cb_anat_fsl':
        atlas_data = nib.load(os.path.join(
                atlas_dir,'Cerebellum-MNIfnirt-maxprob-thr25.nii')).get_fdata()

    elif atlas_name == 'cb_anat_cifti':
        atlas_data = CiftiReader(os.path.join(
                 atlas_dir, 'Cerebellum-MNIfnirt-maxprob-thr25.dscalar.nii')).get_fdata()[0,:]
   
    atlas_info = pd.read_table(os.path.join(atlas_dir, 'Cerebellum-MNIfnirt.txt'), 
                                       sep='\t', header=None)
    atlas_info.columns = ['key','name','lobule','hemi', 'landmark']
    atlas_info['landmark'] = atlas_info['landmark'].apply(lambda x: np.array(eval(x)))
    atlas = Atlas(data=atlas_data, label_info=atlas_info)

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