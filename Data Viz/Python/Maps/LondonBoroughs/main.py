import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

map_df = gpd.read_file('London_Borough_Excluding_MHW.shp')
df = pd.read_csv('london-borough-profiles.csv')