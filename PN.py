#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:45:43 2022

@author: Toor
"""

import heartpy as hpy
import glob 
import pandas as pd 
from scipy.io import loadmat
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

#%%

prefix = "/Users/Toor/Desktop/大学/大学课程/senior thesis/PPG_ACC_dataset/"
not_working = []  # store the list of PPG data that could not be preprocessed 

def load(category, threshold):
    temp = []
    disregard = []
    fSample = 400 
    for name in glob.glob(prefix + 'S[1-9]/'+category+'[1-9]_ppg.mat'):
        try:
            s = loadmat(name)["PPG"][:,1]
            s = hpy.remove_baseline_wander(s, fSample)
            s = hpy.filter_signal(s, cutoff=0.3, sample_rate=fSample, order=2, 
                                  filtertype='highpass')
            s = hpy.filter_signal(s, cutoff=10, sample_rate=fSample, order=2, 
                                  filtertype='lowpass')
            working_data, measures = hpy.process(s, fSample)
            if measures["bpm"] < threshold:
                temp.append(measures)
            else:
                disregard.append(name)
        except:
            not_working.append(name)
    return temp, disregard
#%%
rest,rest_d = load("rest", 130)
rest_mean = dict(pd.DataFrame(rest).mean())
print(rest_mean["bpm"])

#%%
squat, squat_d = load("squat", 200)
squat_mean =  dict(pd.DataFrame(squat).mean())
print(squat_mean["bpm"])
#%%
step, step_d = load("step",200)
step_mean = dict(pd.DataFrame(step).mean())
print(step_mean["bpm"])

#%%
# getting the mean values for dict.values 
combined_mean = []
combined_mean.append(list(rest_mean.values()))
combined_mean.append(list(squat_mean.values()))
combined_mean.append(list(step_mean.values()))

# extract dict.values into lists
rest_all = [list(i.values()) for i in rest]
squat_all = [list(i.values()) for i in squat]
step_all = [list(i.values()) for i in step]
combined_all = np.vstack((np.array(rest_all), np.array(squat_all),np.array(step_all)))

#%% 
"""
2D and 3D MDS 
"""
# 2D MDS of all points 
model2d=MDS()
fig, ax = plt.subplots()

rest_2d = model2d.fit_transform(rest_all)
for elem in rest_2d: 
    ax.scatter(elem[0],elem[1], c = '#8CD2FF')

squat_2d = model2d.fit_transform(squat_all)
for elem in squat_2d: 
    ax.scatter(elem[0],elem[1], c = '#FFD98C')

step_2d = model2d.fit_transform(step_all)
for elem in step_2d:
    ax.scatter(elem[0],elem[1], c = '#AD8CFF')

leg = ax.legend(['rest','squat','step'])
leg.legendHandles[0].set_color('#8CD2FF')
leg.legendHandles[1].set_color('#FFD98C')
leg.legendHandles[2].set_color('#AD8CFF')

plt.show()

#%% 
# 3D MDS for all points 
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

model3d=MDS(n_components=3)

rest_3d = model3d.fit_transform(rest_all)
for elem in rest_3d: 
    ax.scatter(elem[0],elem[1], c = '#8CD2FF')

squat_3d = model3d.fit_transform(squat_all)
for elem in squat_3d: 
    ax.scatter(elem[0],elem[1], c = '#FFD98C')

step_3d = model3d.fit_transform(step_all)
for elem in step_3d:
    ax.scatter(elem[0],elem[1], c = '#AD8CFF')

leg = ax.legend(['rest','squat','step'])
leg.legendHandles[0].set_color('#8CD2FF')
leg.legendHandles[1].set_color('#FFD98C')
leg.legendHandles[2].set_color('#AD8CFF')

plt.show()

#%%
# 2D MDS of mean 
model2d=MDS()
comb_2d = model2d.fit_transform(combined_mean)

fig, ax = plt.subplots()
ax.scatter(comb_2d[0][0],comb_2d[0][1], c = '#8CD2FF', label = 'rest')
ax.scatter(comb_2d[1][0],comb_2d[1][1], c = '#FFD98C', label = 'squat')
ax.scatter(comb_2d[2][0],comb_2d[2][1], c = '#AD8CFF', label = 'step')
ax.legend()
plt.show()
#%%
# 3D MDS of mean 
model3d=MDS(n_components=3)
comb_3d = model3d.fit_transform(combined_mean)

fig = plt.figure()
ax = plt.axes(projection='3d')
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

ax.scatter(comb_3d[0][0],comb_3d[0][1],comb_3d[0][2], c = '#8CD2FF', label = 'rest')
ax.scatter(comb_3d[1][0],comb_3d[1][1],comb_3d[1][2], c = '#FFD98C', label = 'squat')
ax.scatter(comb_3d[2][0],comb_3d[2][1],comb_3d[2][2], c = '#AD8CFF', label = 'step')
ax.legend()
plt.show()

#%% 
"""
TSNE 
"""
import warnings 
warnings.filterwarnings('ignore')

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize = (10,7))
perplexity = [i for i in range(2,25,4)]  # since the perplexity cannot exceed n_examples 
fig.suptitle('TSNE with diff perplexity')


def tsne_plot(perp, ax):
    tsne = TSNE(n_components = 2, perplexity=perp,init = 'pca', learning_rate='auto')
    # rest
    rest_tsne = tsne.fit_transform(np.array(rest_all)) 
    for elem in rest_tsne: 
        ax.scatter(elem[0],elem[1], c = '#8CD2FF')
    # squat
    squat_tsne = tsne.fit_transform(np.array(squat_all))
    for elem in squat_tsne: 
        ax.scatter(elem[0],elem[1], c = '#FFD98C')   
    # step
    step_tsne = tsne.fit_transform(np.array(step_all))
    for elem in step_tsne:
        ax.scatter(elem[0],elem[1], c = '#AD8CFF')
    # set legend
    leg = ax.legend(['rest','squat','step'])
    leg.legendHandles[0].set_color('#8CD2FF')
    leg.legendHandles[1].set_color('#FFD98C')
    leg.legendHandles[2].set_color('#AD8CFF')
    ax.title.set_text(perp)

axs = [ax1,ax2,ax3,ax4,ax5,ax6]
for i in range(len(perplexity)):
    tsne_plot(perplexity[i],axs[i])

plt.show()

#%%
# if not divide them first and make them combined at first and transform
# split them after fitting 

import warnings 
warnings.filterwarnings('ignore')

fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize = (15,10))
perplexity = [i for i in range(5,85,8)]  # since the perplexity cannot exceed n_examples 
fig.suptitle('TSNE with diff perplexity')

def tsne_plot(perp, ax):
    tsne = TSNE(n_components = 2, perplexity=perp, init = 'pca', learning_rate='auto')
    comb_tsne = tsne.fit_transform(combined_all)
    # rest 
    for elem in comb_tsne[:25]: 
        ax.scatter(elem[0],elem[1], c = '#8CD2FF')
    # squat
    for elem in comb_tsne[25:25+29]: 
        ax.scatter(elem[0],elem[1], c = '#FFD98C')   
    # step
    for elem in comb_tsne[25+29:]:
        ax.scatter(elem[0],elem[1], c = '#AD8CFF')
    # set legend
    leg = ax.legend(['rest','squat','step'])
    leg.legendHandles[0].set_color('#8CD2FF')
    leg.legendHandles[1].set_color('#FFD98C')
    leg.legendHandles[2].set_color('#AD8CFF')
    ax.title.set_text(perp)

axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]
for i in range(len(perplexity)):
    tsne_plot(perplexity[i],axs[i])

plt.show()
#%% 
"""
UMAP
"""
from umap import UMAP

fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5,figsize = (15,10))
neigh = [i for i in range(5,85,8)]  # since the perplexity cannot exceed n_examples 
fig.suptitle('UMAP with diff perplexity')

def umap_plot(n, ax):
    umap_2d = UMAP(n_components=2,n_neighbors = n)
    comb_umap = umap_2d.fit_transform(combined_all)
    # rest 
    for elem in comb_umap[:25]: 
        ax.scatter(elem[0],elem[1], c = '#8CD2FF')
    # squat
    for elem in comb_umap[25:25+29]: 
        ax.scatter(elem[0],elem[1], c = '#FFD98C')   
    # step
    for elem in comb_umap[25+29:]:
        ax.scatter(elem[0],elem[1], c = '#AD8CFF')
    # set legend
    leg = ax.legend(['rest','squat','step'])
    leg.legendHandles[0].set_color('#8CD2FF')
    leg.legendHandles[1].set_color('#FFD98C')
    leg.legendHandles[2].set_color('#AD8CFF')
    ax.title.set_text(n)

axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]
for i in range(len(neigh)):
    umap_plot(neigh[i],axs[i])

plt.show()

#%%

# train classifier 
    # simple classifier like KNN 
    # random forest 
    # and then to use CNN to see if its better 
    
"""
Parallel coordinates
"""
from pandas.plotting import parallel_coordinates
# change original data into arrays with catogories added to the end 
rest_cat = np.full((25,1),'rest')
rest_with_cat = np.concatenate((np.array(rest_all),rest_cat),axis=1)

squat_cat = np.full((29,1),'squat')
squat_with_cat = np.concatenate((np.array(squat_all),squat_cat),axis=1)

step_cat = np.full((30,1),'step')
step_with_cat = np.concatenate((np.array(step_all),step_cat),axis=1)

comb_with_cat = np.vstack((rest_with_cat,squat_with_cat,step_with_cat))

key = list(rest[0].keys())
key.append('category')
comb_df = pd.DataFrame(comb_with_cat,columns = key)

parallel_coordinates(comb_df, 'category', color=('#8CD2FF', '#FFD98C', '#AD8CFF'))
