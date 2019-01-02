import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import imageio

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
df = pd.read_csv('data/MPS_Borough_Level_Crime_Historic.csv')
filtered = df['Major Category'] == 'Violence Against the Person'
violence = df[filtered]

df3 = violence.groupby('Borough').sum()

melted = pd.melt(violence, id_vars=['Borough', 'Major Category', 'Minor Category'])

df2 = melted.groupby('Borough').sum()

crime = melted.pivot_table(values='value', index=['Borough', 'Major Category'], columns='variable', aggfunc=np.sum)
crime.columns = crime.columns.get_level_values(0)

merged1 = map_df.set_index('NAME').join(df3)
merged1 = merged1.reindex(merged1.index.rename('Borough'))
merged1.fillna(0, inplace=True)

output_path = 'charts/maps'
i = 0

list_of_years = ['200807', '200907', '201007', '201107', '201207', '201307', '201407', '201507', '201607']
vmin, vmax = 200, 1200
images = []

for year in list_of_years:
    fig = merged1.plot(
        column=year,
        cmap='Purples',
        figsize=(10,10),
        linewidth=0.8,
        edgecolor='0.8',
        legend=True,
        vmin=vmin,
        vmax=vmax,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )

    fig.axis('off')
    fig.set_title('Violent Crimes in London',
                  fontdict={
                      'fontsize': '25',
                      'fontweight' : '3'
                  })

    only_year = year[:4]
    fig.annotate(
        only_year,
        xy=(.1, .225),
        xycoords='figure fraction',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=35
    )

    filepath = os.path.join(output_path, only_year+'_violence.png')
    chart = fig.get_figure()
    chart.savefig(filepath, dpi=300)
    images.append(imageio.imread(filepath))

imageio.mimsave('charts/gifs/london_crime.gif', images, duration=.5)