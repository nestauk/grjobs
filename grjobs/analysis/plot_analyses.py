# File: analysis/plot_analyses.py

"""Module for plotting graphs based off analysis on labelled job ads in s3. 
"""
# ---------------------------------------------------------------------------------
import random
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from adjustText import adjust_text

#
from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs import get_yaml_config, Path, PROJECT_DIR
from grjobs.analysis.run_analyses import (
    cluster_job_titles,
    label_job_title_clusters,
    get_job_ids,
    get_recent_job_ads,
    get_labelled_jobs,
)

# ---------------------------------------------------------------------------------
# load relevant config file
analysis_params = get_yaml_config(
    Path(str(PROJECT_DIR) + "/grjobs/config/analysis_config.yaml")
)

# load relevant asset path
prop = fm.FontProperties(fname=str(PROJECT_DIR) + analysis_params["FONT_PATH"])


def plot_job_title_clusters(labelled_jobs, random_cluster_no):
    """
     Plots cluster dataframe as a scatter plot.  

     Takes as input labelled jobs. Randomly picks x different cluster names and plots them
     on a scatter plot. 
    """
    embedding_df = cluster_job_titles(labelled_jobs)
    labelled_embedding_df = label_job_title_clusters(embedding_df)

    fig, ax = plt.subplots(figsize=(20, 10))
    clustered = labelled_embedding_df.loc[labelled_embedding_df.labels]

    x = clustered.x.tolist()
    y = clustered.y.tolist()

    clustered["cluster_name"] = [name[0] for name in clustered["cluster_name"]]
    random_cluster_names = random.sample(
        list(set(clustered["cluster_name"])), random_cluster_no
    )

    cluster_labels = []
    x_labels = []
    y_labels = []

    for cluster_name in random_cluster_names:
        x_label = clustered[clustered["cluster_name"].astype(str) == cluster_name][
            "x"
        ].tolist()
        y_label = clustered[clustered["cluster_name"].astype(str) == cluster_name][
            "y"
        ].tolist()
        cluster_labels.append(cluster_name)
        x_labels.append(random.choice(x_label))
        y_labels.append(random.choice(y_label))

    plt.scatter(x, y, c=clustered.labels, s=1, cmap="hsv_r")

    texts = []
    for cluster, x, y in zip(cluster_labels, x_labels, y_labels):
        texts.append(plt.text(x, y, cluster, fontproperties=prop, size=20))

    plt.axis("off")
    ax.set_title(
        "Job Title Clusters in Green Industries",
        fontproperties=prop,
        size=25,
        color="black",
    )
    adjust_text(
        texts, force_points=1, arrowprops=dict(arrowstyle="->", color="r", lw=0.5)
    )

def plot_green_locations(labelled_jobs):
    """
     Plots percentage of job adverts in green industries nuts 2 code. 
    """
    green_job_locations = []
    for job in labelled_jobs:
        if "location" in job["features"].keys():
            green_job_locations.append(job["features"]["location"]["nuts_2_name"])

    all_locations = []
    green_ids = get_job_ids(labelled_jobs)
    for job in get_recent_job_ads():

        if job["id"] not in green_ids and "location" in job["features"].keys():
            all_locations.append(job["features"]["location"]["nuts_2_name"])

    green_locations_df = pd.DataFrame(
        Counter(green_job_locations), index=["green_count"]
    ).T
    green_locations_df["NUTS2_CODE"] = green_locations_df.index
    all_locations_df = pd.DataFrame(Counter(all_locations), index=["all_count"]).T
    all_locations_df["NUTS2_CODE"] = all_locations_df.index

    locations_df = green_locations_df.merge(all_locations_df, on="NUTS2_CODE")
    locations_df["percentage"] = (
        locations_df["green_count"] / locations_df["all_count"]
    ) * 100

    top = locations_df.sort_values(by="percentage", ascending=False)[:10].sort_values(
        "percentage", ascending=True
    )

    # plot
    fig, ax = plt.subplots(figsize=(9.2, 5))
    plt.barh(top["NUTS2_CODE"], top["percentage"], color="blue")
    plt.xlabel('% of job vacancies in green industries', fontproperties=prop, size=16)
    plt.xticks(fontproperties=prop, size=14)
    plt.yticks(fontproperties=prop, size=14)

    plt.title(
        "Online Job Vacancies \n in Green Industries by Region",
        fontproperties=prop,
        size=20,
    )
    plt.tight_layout()

if __name__ == "__main__":
    green_labelled_jobs = get_labelled_jobs("green_jobs_output")
    plot_job_title_clusters(green_labelled_jobs, 15)
    #plot_green_locations(green_labelled_jobs)