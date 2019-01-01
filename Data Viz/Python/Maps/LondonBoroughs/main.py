import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

map_df = gpd.read_file('London_Borough_Excluding_MHW.shp')
df = pd.read_csv('london-borough-profiles.csv', header=0, encoding='ISO-8859-1')

#column filtering, we are interested in
df = df[['Area name',
         'Happiness score 2011-14 (out of 10)',
         'Anxiety score 2011-14 (out of 10)',
         'Population density (per hectare) 2015',
         'Mortality rate from causes considered preventable']]

#rename, since ugly names
score = df.rename(index=str, columns={
    "Area name" : 'borough',
    "Happiness score 2011-14 (out of 10)": "happiness",
    "Anxiety score 2011-14 (out of 10)": "anxiety",
    "Population density (per hectare) 2015": "pop_density_per_hectare",
    "Mortality rate from causes considered preventable": 'mortality'
})
