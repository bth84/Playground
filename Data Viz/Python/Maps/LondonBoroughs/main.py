import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

map_df = gpd.read_file('data/London_Borough_Excluding_MHW.shp')
df = pd.read_csv('data/london-borough-profiles.csv', header=0, encoding='ISO-8859-1')

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

merged = map_df.set_index('NAME').join(score.set_index('borough'))

#set a variable that will call whatever column we want to visualise on the map
variable = 'pop_density_per_hectare'


def plot_by(feature):
    #set the range for the chloropleth
    vmin, vmax = 120, 220

    #create figure and axes for MPL
    fig, ax = plt.subplots(1, figsize=(10,6))

    #create map
    merged.plot(column=feature, cmap='Blues', linewidth=.8, ax=ax, edgecolor='.8')

    #___prettify___

    #axis remove
    ax.axis('off')

    #set title
    ax.set_title('Preventable death rate in London',
                 fontdict={
                     'fontsize' : '25',
                     'fontweight' : '3'
                 })

    #Annotation for data source
    ax.annotate('Source: London Datastore, 2014',
                xy=(.1, .08),
                xycoords='figure fraction',
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=10,
                color='#555555')

    #colorbar as legend
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)

    plt.show()

#plot_by(variable)