addpath /usr/local/neurosoft/matlab_tools/spm12
addpath /usr/local/neurosoft/matlab_tools/spm12/compat
addpath /usr/local/neurosoft/matlab_tools/spm12/toolbox/DARTEL
addpath /usr/local/neurosoft/matlab_tools/spm12/toolbox/suit

%%
results_dir = '/nfs/s2/userhome/liuxingyu/workingdir/cerebellum_grad_dev';

%% t1wt2wratio - adult
map_file = fullfile(results_dir, 't1wT2wRatio/HCP-Adult/t1wT2wRatio_cb_voxel_mean.nii.gz');
Data=suit_map2surf(map_file, 'space', 'FSL');

fig = figure;
range = [1.5, 2.2];
suit_plotflatmap(Data,'cmap',viridis, 'cscale', range); 
colormap viridis;
colorbar();
caxis(range); 

%% t1wt2wratio - pct
map_file = fullfile(results_dir, 't1wT2wRatio/t1wT2wRatio_cb_voxel_dev_pct.nii.gz');
Data=suit_map2surf(map_file, 'space', 'FSL');

fig = figure;
range = [-0.005, 0.01];
suit_plotflatmap(Data,'cmap',viridis, 'cscale', range); 
colormap viridis;
colorbar();
caxis(range); 

%% fALFF - adult
map_file = fullfile(results_dir, 'fALFF/HCP-Adult/fALFF_cb_voxel_mean_cbonly.nii.gz');
Data=suit_map2surf(map_file, 'space', 'FSL');

fig = figure;
range = [0.102,0.13];
suit_plotflatmap(Data,'cmap',viridis, 'cscale', range); 
colormap viridis;
colorbar();
caxis(range); 

%% fALFF - pct
map_file = fullfile(results_dir, 'fALFF/fALFF_cb_voxel_dev_pct.nii.gz');
Data=suit_map2surf(map_file, 'space', 'FSL');

fig = figure;
range = [-0.01, 0];
suit_plotflatmap(Data,'cmap',viridis, 'cscale', range); 
colormap viridis;
colorbar();
caxis(range); 


