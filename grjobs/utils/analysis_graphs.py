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
#     display_name: green_jobs
#     language: python
#     name: green_jobs
# ---

# %%
import collections
import sys
sys.path.insert(0, '/Users/india.kerlenesta/Projects/ojo/ojd_daps/')

from ojd_daps.dqa.data_getters import get_db_job_ads
from ojd_daps.dqa.data_getters import get_locations

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from descartes import PolygonPatch

# %%
#get fake green jobs

fake_green = list(get_db_job_ads(limit = 1000, return_features=True))

# %%
fake_green

# %%
#get green job locations

green_jobs = [job for job in get_db_job_ads(job_board='reed', limit = 10000) if job['description'] != '[]']

job_locations = list(get_locations('reed', level='nuts_2'))
for job in job_locations:
    job['id'] = job.pop('job_id')
    
labelled_job_locations = collections.defaultdict(dict)
for ads in (green_jobs, job_locations):
    for ad in ads:
        labelled_job_locations[ad['id']].update(ad)

green_job_locations = [job for job in list(labelled_job_locations.values()) if len(job) > 2 and 'nuts_2_code' in job.keys()]

# %%
#merge and plot green job locations
import numpy as np 

shapefile = '/Users/india.kerlenesta/Projects/ojo/greenjobs/greenjobs/ref-nuts-2021-01m.shp /NUTS_RG_01M_2021_4326_LEVL_2.shp'

nuts = gpd.read_file(shapefile)
nuts_uk = nuts[nuts['CNTR_CODE'] == 'UK'].reset_index(drop = True)
green_nuts2 = pd.DataFrame(pd.DataFrame(green_job_locations).groupby('nuts_2_code')['nuts_2_code'].count())
green_nuts2['code'] = green_nuts2.index
green_nuts2 = green_nuts2.reset_index(drop = True)
green_nuts2['log'] = np.log(green_nuts2.nuts_2_code)
green_nuts2 = green_nuts2.rename(columns = {'nuts_2_code': 'count',
                   'code': 'NUTS_ID'})

final_nuts2 = nuts_uk.merge(green_nuts2, on = 'NUTS_ID')

# %%
nuts_uk.shape
final_nuts2.shape
final_nuts2.head()

# %%
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

# %%
