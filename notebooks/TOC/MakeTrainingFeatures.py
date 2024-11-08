#!/usr/bin/env python
# coding: utf-8

# ## Continental shelves

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from os import walk
import pickle
from pathlib import Path
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.colors as colors

from os import walk  
import torch.nn.functional as F
import sys

module_path = Path().resolve().parents[1] / "src" / "preprocessing"

# Append the directory to sys.path
sys.path.append(str(module_path))


from makeTrainingFeatures import *


#your current directory
import os
currdir = os.getcwd()
#os.chdir(currdir)


# In[29]:


# Define file paths
data_path = Path().resolve().parents[1] / 'data' / 'raw' / 'labels'
toc_labels_file = "toc_continentalshelves.csv"
lee_features_path = Path().resolve().parents[1] / 'data' / 'raw' / 'features' / 'FeaturesPhrampusLee_TOCSedRate_updated'
feature_selection_path = Path().resolve().parents[1] / 'data' / 'interim' / 'selectedFeatureLists' / 'selectedfeatures_v_men.txt'
save_path = Path().resolve().parents[1] / 'data' / 'interim' / 'inputfeatures' / 'SedTOCFeaturesnoNAN_TOC_DO_men_firsthalf'


# In[30]:


# Read data
df = pd.read_csv(Path(data_path / toc_labels_file))
df = df[["Latitude", "Longitude", "TOC [%]"]]
df = df[df["Longitude"] > -180]
df = df.dropna(how="any")


# In[31]:


from sklearn.model_selection import train_test_split


# In[8]:


df_1_3, df_2_3 = train_test_split(df, test_size=2/3, random_state=42)


# In[10]:


df_infogain_DO_first_half = pd.read_csv("preprocessed/infogain_DO_1_3_first_half.csv")


# In[13]:


selected_columns = ['Latitude', 'Longitude', 'TOC [%]' ]
df_infogain_DO_first_half = df_infogain_DO_first_half[selected_columns]


# In[12]:


df_2_3


# In[14]:


df_infogain_DO_first_half


# In[19]:


df_infogain_DO_first_half_2_3 = pd.concat([df_infogain_DO_first_half, df_2_3])
print("firsssttttttttttttttttttttttttttt")

# In[20]:


df_infogain_DO_first_half_2_3


# In[22]:


df


# In[21]:


# Plot TOC on map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plot_toc_on_map(df, world)

# Plot TOC on map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plot_toc_on_map(df_infogain_DO_first_half_2_3, world)


# In[23]:


# Select features
selected_files = select_features(lee_features_path, feature_selection_path)


# In[24]:


# Process features
features, mean, std, lat_labels, lon_labels = process_features(lee_features_path, selected_files, df_infogain_DO_first_half_2_3)


# In[25]:


lat_labels = np.array( df_infogain_DO_first_half_2_3["Latitude"])
lon_labels = np.array( df_infogain_DO_first_half_2_3["Longitude"])
np.save(save_path / "numpy_lat", lat_labels)
np.save(save_path / "numpy_lon", lon_labels)


# In[26]:


df_infogain_DO_first_half_2_3.shape


# In[ ]:


df_1_3.shape


# In[ ]:


# Do PCA of features


# In[27]:


# Save processed data
save_data(features, df_infogain_DO_first_half_2_3["TOC [%]"].values, mean, std, lat_labels, lon_labels, save_path)  


