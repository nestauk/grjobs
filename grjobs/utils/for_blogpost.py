# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: grjobs
#     language: python
#     name: grjobs
# ---

# %%
import collections

from ojd_daps.dqa.data_getters import get_db_job_ads

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import matplotlib.font_manager as fm

# %%
#get fake green jobs

fake_green = list(get_db_job_ads(limit = 10000, return_features=True))

# %% [markdown]
# **GREEN LOCATIONS MAP**

# %%
import numpy as np 

green_locations = [fake_green[i]['features']['location'] for i, green in enumerate(fake_green) if 'location' in fake_green[i]['features'].keys()]
shapefile = '/Users/india.kerlenesta/Projects/ojo/grjobs/inputs/shapefiles/NUTS_RG_01M_2021_4326_LEVL_2.shp'

nuts = gpd.read_file(shapefile)
nuts_uk = nuts[nuts['CNTR_CODE'] == 'UK'].reset_index(drop = True)
green_nuts2 = pd.DataFrame(pd.DataFrame(green_locations).groupby('nuts_2_code')['nuts_2_code'].count())
green_nuts2['code'] = green_nuts2.index
green_nuts2 = green_nuts2.reset_index(drop = True)
green_nuts2['log'] = np.log(green_nuts2.nuts_2_code)
green_nuts2 = green_nuts2.rename(columns = {'nuts_2_code': 'count',
                   'code': 'NUTS_ID'})

final_nuts2 = nuts_uk.merge(green_nuts2, on = 'NUTS_ID')

# set the value column that will be visualised
variable = 'log'
# set the range for the choropleth values
vmin, vmax = 0, 50
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(30, 10))
# remove the axis
ax.axis('off')
# add a title and annotation
font_directory = '/Users/india.kerlenesta/Library/Fonts/Averta_Standard/AvertaStd-Bold.ttf'
prop = fm.FontProperties(fname=font_directory)
ax.set_title('green jobs by location', fontproperties = prop, fontsize= 22, color='#F67E00')


ax.annotate('Source: Wikipedia - https://en.wikipedia.org/wiki/Provinces_of_Indonesia', xy=(0.6, .05), xycoords='figure fraction', fontsize=12, color='#555555')
# Create colorbar legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm.set_array([]) # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it

final_nuts2.plot(column=variable, cmap='viridis', linewidth=0.2, ax=ax, edgecolor='1')

# %% [markdown]
# **GREEN JOBS BY SIC CODE**

# %% [markdown]
# **GREEN JOBS BY SOC CODE**

# %%
green_soccodes = [fake_green[i]['features']['soc'] for i, green in enumerate(fake_green) if 'soc' in fake_green[i]['features'].keys()]
green_soc_titles = ' '.join([soccode['soc_title'] for soccode in green_soccodes])

# %%
import collections 

soc_counts = collections.Counter([soccode['soc_title'] for soccode in green_soccodes])
soc_counts.most_common()

# %% [markdown]
# stacked graph

# %%
font_directory = '/Users/india.kerlenesta/Library/Fonts/Averta_Standard/AvertaStd-Bold.ttf'
prop = fm.FontProperties(fname=font_directory)


# %% [markdown]
# word cloud

# %%
font_directory = '/Users/india.kerlenesta/Library/Fonts/Averta_Standard/AvertaStd-Bold.ttf'
prop = fm.FontProperties(fname=font_directory)

def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

# Import package
from wordcloud import WordCloud, STOPWORDS
text = ' '.join([soccode['soc_title'] for soccode in green_soccodes])
# Generate word cloud
wordcloud = WordCloud(width = 3000, 
                      height = 2000, 
                      random_state=1,
                      font_path = font_directory,
                      background_color='white', 
                      colormap='viridis', 
                      collocations=False, 
                      stopwords = STOPWORDS).generate(green_soc_titles)
# Plot
plot_cloud(wordcloud)

# %% [markdown]
# **GREEN JOBS BY MOST COMMON TITLES**

# %% [markdown]
# word cloud

# %% [markdown]
# **GREEN JOBS vs. ALL JOBS SALARY**

# %%
fake_green

# %%
