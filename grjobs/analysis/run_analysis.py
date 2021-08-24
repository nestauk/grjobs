# File: analysis/run_analysis.py

"""Module for running analysis on labelled job ads in s3. 
"""
# ---------------------------------------------------------------------------------
import pickle
import pandas as pd
import statistics 
import umap
import hdbscan
import numpy as np

from ipywidgets import IntProgress
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#
from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs import get_yaml_config, Path, PROJECT_DIR
from grjobs.pipeline.create_labelled_data import (
    load_pkl_from_s3
    )
# ---------------------------------------------------------------------------------
#load relevant config file 
analysis_params = get_yaml_config(Path(str(PROJECT_DIR) + "/grjobs/config/analysis_config.yaml"))

def get_labelled_jobs(prefix="green_jobs_output"):
    """loads labelled data as pickle from s3"""
    return load_pkl_from_s3(prefix)

def get_green_ids(labelled_jobs):
    """gets green ids"""
    return [job['id'] for job in labelled_jobs]

def get_transformer():
    """loads sentence transformer"""
    return SentenceTransformer(analysis_params['embedding_model'])

def cluster_job_titles(labelled_jobs) -> pd.DataFrame():
    """Clusters unique job titles for jobs labelled green.
    
    Encodes unique job titles using a sentence transformer. Reduces dimensionality of embeddings using UMAP to
    two dimensions. Clusters dimensionality-reduced job title embeddings using HBDSCAN.
    
    Returns:
        A cluster dataframe which includes dimensionality reduced values, the cluster label, the 
        associated job title and the probability the job title belongs to the cluster.
        
    """
    green_job_titles = list(set([job['job_title_raw'].lower() for job in labelled_jobs]))
    model = get_transformer()
    # embed unique job titles
    job_title_embeddings = model.encode(green_job_titles)
    
    # reduce dim
    umap_embeddings = umap.UMAP(n_neighbors=analysis_params['n_neighbors'], 
                                n_components = analysis_params['components'],
                                metric = analysis_params['umap_metric']).fit_transform(job_title_embeddings)
    
    #cluster reduced dim
    cluster = hdbscan.HDBSCAN(min_cluster_size=analysis_params['cluster_size'],
                      metric=analysis_params['hdbscan_metric'],                      
                      cluster_selection_method=analysis_params['method']).fit(umap_embeddings)
    
    #create df
    embedding_df = pd.DataFrame(umap_embeddings, columns = ['x', 'y'])
    embedding_df['labels'] = cluster.labels_
    embedding_df['sentence'] = green_job_titles
    embedding_df['membership_probability'] = cluster.probabilities_
    
    return embedding_df

def get_tfidf_top_features(sents,n_top=1):
    """Gets the top tfidf feature for sentences.
    
    Vectorises sentences using a tfidf vectoriser and returns the top n terms.
    
    Returns:
        top n terms based on tfidf score for documents. 
    """
    vectoriser = TfidfVectorizer(max_df=analysis_params['max_df'], 
                                 stop_words='english',
                                ngram_range = (2,2))
    tfidf = vectoriser.fit_transform(sents)                             
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(vectoriser.get_feature_names())
    return tfidf_feature_names[importance[:n_top]]

def label_job_title_clusters(job_embedding_df, n_top=1) -> pd.DataFrame():
    """Labels job title clusters based on top tfidf features per cluster.
    
    Subsets embedding df to only include clustered job titles with cluster membership
    probability above a threshold.  
    
    Returns:
        A cluster dataframe that also includes the cluster label associated with the cluster.
        
    """                         
    job_embedding_df = job_embedding_df[(job_embedding_df['labels'] != -1) & 
                                (job_embedding_df['membership_probability'] > analysis_params['membership_prob'])].reset_index(drop = True)
    cluster_names = []
    for clust in list(set(job_embedding_df['labels'].tolist())):
        sents = job_embedding_df[job_embedding_df['labels'] == clust]['sentence'].tolist()
        if len(sents) > 0:
            tfidf_labels = get_tfidf_top_features(sents, n_top)
            cluster_names.append((clust, tfidf_labels))
    
    cluster_names_df = pd.DataFrame(cluster_names, columns = ['labels', 'cluster_name'])
                                 
    return job_embedding_df.merge(cluster_names_df,how='left', left_on='labels', right_on='labels')

def get_salaries(labelled_jobs) -> list:
    """Gets minimum annualised salaries and maximum annualised salaries of green and non green jobs. 

    Returns:
        two lists - the first list is the minimum annualised salary and the maximum annualised salary per green job
        and the second contains the minimum annualised salary and the maximum annualised salary per non green job.  
    """ 
    green_salaries = []
    for job in labelled_jobs:
        salary_range = [job['features']['salary']['min_annualised_salary'], job['features']['salary']['max_annualised_salary']]
        if 'salary' in job['features'].keys():
            green_salaries.append(salary_range)

    non_green_salaries = []
    green_ids = get_green_ids(labelled_jobs)
    for job in get_db_job_ads(limit = None, return_features = True):
        salary_range = [job['features']['salary']['min_annualised_salary'], job['features']['salary']['max_annualised_salary']]
        if job['id'] not in green_ids and 'salary' in job['features'].keys():
            non_green_salaries.append(salary_range)

    return green_salaries, non_green_salaries 

def calculate_median_salaries(labelled_jobs):
    """Calculates median minimum salary and median maximum salary of labelled jobs. 

    Returns:
        four values - median min salary for green jobs, median max salary for green jobs,  median min salary for non green jobs, 
        median max salary for non green jobs.
    """ 
    green_salaries, non_green_salaries = get_salaries(labelled_jobs)

    return (f"the minimum median green salary is {statistics.median([sal[0] for sal in green_salaries])}",
            f"the maximum median green salary is {statistics.median([sal[1] for sal in green_salaries])}",
            f"the minimum median non green salary is {statistics.median([sal[0] for sal in non_green_salaries])}",
            f"the maximum median non green salary is {statistics.median([sal[1] for sal in non_green_salaries])}")

if __name__ == '__main__':
    labelled_jobs = get_labelled_jobs('green_jobs_output')
    calculate_median_salaries(labelled_jobs)