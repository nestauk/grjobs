# File: analysis/plot_analysis.py

"""Module for plotting graphs based off analysis on labelled job ads in s3. 
"""
# ---------------------------------------------------------------------------------
import random
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#
from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs import get_yaml_config, Path, PROJECT_DIR
from grjobs.analysis.run_analysis import (
    cluster_job_titles,
    label_job_title_clusters,
    get_green_ids
    )
# ---------------------------------------------------------------------------------
#load relevant config file
analysis_params = get_yaml_config(Path(str(PROJECT_DIR) + "/grjobs/config/analysis_config.yaml"))

#load relevant asset path 
prop = fm.FontProperties(fname=str(PROJECT_DIR) + analysis_params["FONT_PATH"])

def plot_job_title_clusters(labelled_jobs, random_cluster_no) -> plt:
    """
     Plots cluster dataframe as a scatter plot.  

     Takes as input labelled jobs. Randomly picks x different cluster names and plots them
     on a scatter plot. 
         
    Returns:
        A labelled scatter plot of job title clusters and their associated cluster label. 

    """ 
    embedding_df = cluster_job_titles(labelled_jobs)
    labelled_embedding_df = label_job_title_clusters(embedding_df)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    clustered = labelled_embedding_df.loc[labelled_embedding_df.labels != -1, :]
    
    x = clustered.x.tolist()
    y = clustered.y.tolist()
    
    random_cluster_names = random.sample(list(set(clustered['cluster_name'].astype(str).tolist())), random_cluster_no)

    cluster_labels = []
    x_labels = []
    y_labels = []

    for cluster_name in random_cluster_names:
        x_label = clustered[clustered['cluster_name'].astype(str) == cluster_name]['x'].tolist()
        y_label = clustered[clustered['cluster_name'].astype(str) == cluster_name]['y'].tolist()
        cluster_labels.append(cluster_name.replace('[', '').replace(']', '').replace("'", ''))
        x_labels.append(random.choice(x_label))
        y_labels.append(random.choice(y_label))
    
    plt.scatter(x, 
            y, 
            c=clustered.labels, 
            s=1, 
            cmap='hsv_r')

    for i, txt in enumerate(cluster_labels):
        plt.annotate(txt, 
                 (x_labels[i], 
                  y_labels[i]),
                fontsize=15,
                fontproperties=prop,
                color='black')

    plt.axis('off')
    ax.set_title('Job Title Clusters in Green Industries', 
             fontproperties=prop,
            size=25,
            color="black")
    color = 'viridis'

    return plt

def plot_green_locations(labelled_jobs) -> plt:
    """
     Plots percentage of job adverts in green industries nuts 2 code. 
         
    Returns:
        An ordered bar chart of the percentage of 'green' jobs by UK nuts 2 code.

    """ 
    green_job_locations = []
    for job in labelled_jobs:
        if 'location' in job['features'].keys():
            green_job_locations.append(job['features']['location']['nuts_2_code'])

    all_locations = []
    green_ids = get_green_ids(labelled_jobs)
    for job in get_db_job_ads(limit = None, return_features = True):
        if job['id'] not in green_ids and 'location' in job['features'].keys():
            all_locations.append(job['features']['location']['nuts_2_code'])

    green_locations_df = pd.DataFrame(Counter(green_job_locations), index = ['green_count']).T
    green_locations_df['NUTS2_CODE'] = green_locations_df.index
    all_locations_df =  pd.DataFrame(Counter(all_locations), index = ['all_count']).T
    all_locations_df['NUTS2_CODE'] = all_locations_df.index

    locations_df = green_locations_df.merge(all_locations_df, on = 'NUTS2_CODE')
    locations_df['percentage'] = (locations_df['green_count']/locations_df['all_count'])*100

    top = locations_df.sort_values(by='percentage', ascending=False)[:5].sort_values('percentage', ascending=True)
    
    #plot
    fig, ax = plt.subplots(figsize=(9.2, 5))
    plt.barh(top['NUTS2_CODE'], top['percentage'], color="blue")
    plt.xlabel('percentage of green roles (%)', fontproperties=prop, size = 16)
    plt.xticks(fontproperties=prop, size = 14)
    plt.ylabel('NUTS2 code',fontproperties=prop, size=16)
    plt.yticks(fontproperties=prop, size = 14)

    plt.title('Online Job Adverts in Green Industries (%) by NUTS2 code', fontproperties=prop, size=20)
    plt.tight_layout()

    return plt

if __name__ == '__main__':
    labelled_jobs = get_labelled_jobs('green_jobs_output')
    plot_job_title_clusters(labelled_jobs, 15)
    plot_green_locations(labelled_jobs)