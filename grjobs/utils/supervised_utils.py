"""
utils.keyword_expansion_utils
--------------
Module for classifying jobs as green. Author: India Kerle
"""

from keyword_expansion_utils import * 
from text_cleaning_utils import *
from helper_utils import *

import collections
import boto3
import json
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

import numpy as np
from functools import lru_cache
from xgboost import XGBClassifier

from ojd_daps.dqa.data_getters import get_db_job_ads

S3_PATH = 'labs/green-jobs/{}'
BUCKET_NAME = 'open-jobs-lake'

@lru_cache(maxsize=None)
def load_from_s3(filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH.format(filename))
    return obj["Body"].read().decode()
   
def load_json_from_s3(prefix):
    """Save data as json from S3"""
    return json.loads(load_from_s3(f"{prefix}.json"))

def label_jobs(limit):
    '''takes as input limit the number of jobs to retrieve from database.

    :return: list of job ad dictionaries with training labels.'''

    training = load_json_from_s3('final_training_set')
    job_ads = [job for job in get_db_job_ads(job_board='reed', limit = limit) if job['description'] != '[]']

    labelled_jobs = collections.defaultdict(dict)
    for ads in (training, job_ads):
        for ad in ads:
            labelled_jobs[ad['id']].update(ad)

    labelled_jobs = list(labelled_jobs.values())

    return labelled_jobs

def clean_labelled_jobs(labelled_jobs):
    '''takes as input labelled jobs.

    :return: list of job ad dictionaries with cleaned text.'''

    labelled_clean_jobs = []
    for job in labelled_jobs:
    	job['clean_description'] = clean_text(job['job_title_raw'] + ' ' + job['description'])
    	labelled_clean_jobs.append(job)
    
    return labelled_clean_jobs

def add_green_counts(clean_labelled, cleaned_queries):
    
    '''input: a dictionary of labelled jobs and list of 'green' terms.
    return: a dictionary of labelled jobs with a 'green' count feature. A 
    green count feature refers to the number of any 'green' terms normalised by
    word length per job ad.''' 
        
    green_counts_feature = []
    for job in clean_labelled:
        job_toks = word_tokenize(job['clean_description'])
        for green in cleaned_queries:
            if len(green.split(' ')) > 1:
                green_counts = job['clean_description'].count(green)
                normalised_counts = green_counts/len(job_toks)
                green_counts_feature.append((job['id'], normalised_counts))
            if len(green.split(' ')) == 1:
                green_toks_count = job_toks.count(green)
                normalised_tok_counts = green_toks_count/len(job_toks)
                green_counts_feature.append((job['id'], normalised_tok_counts))

    keys = set(k for k, _ in green_counts_feature)
    green_totals = {unique_key: sum(v for k, v in green_counts_feature if k == unique_key) for unique_key in keys}

    with_green_counts = []
    for job_ad in clean_labelled:
        for green_key, green_total in green_totals.items():
            if str(job_ad['id']) == str(green_key):
                job_ad['green_count'] = green_total
                with_green_counts.append(job_ad)
    
    return with_green_counts


def tfidf_vectorise(labelled_clean_jobs): #this needs to be scaled up 
    """
    This function vectorises job descriptions and stacks the normalised
    green count feature. 

    :return: stacked job description and green count matrices
    """
    texts = [clean['clean_description'] for clean in labelled_clean_jobs]
    green_counts = np.array([job['green_count'] for job in labelled_clean_jobs])

    vectorizer = TfidfVectorizer(min_df = 0.05, max_df = 0.60)
    features = vectorizer.fit_transform(texts).toarray()
    
    stacked_features = np.hstack((features, green_counts[:,None]))

    labelled_transform = []
    for job, vec in zip(labelled_clean_jobs, stacked_features):
        job['vec'] = vec
        labelled_transform.append(job)

    return labelled_transform
    
def oversample_training_ads(labelled_transform):
    """
    This function uses SMOTE to oversample labelled 'green' jobs 
    in an effort to address the imbalanced nature of the task.
    
    :return: a list of training matrices and training labels.
    """
    X_train = [job['vec'] for job in labelled_transform if 'label' in job.keys()]
    y_train = [job['label'] for job in labelled_transform if 'label' in job.keys()]
    
    oversample = SMOTE()
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    
    return X_over, y_over

def classify_jobs(X_train, y_train, labelled_transform):
    """
    This function trains an xgboost classifier on oversampled job matrices
    and predicts whether a job is 'green' or not.
    :return: a list of predictions 
    """
    
    not_labeled_features = np.array([job['vec'] for job in labelled_transform if 'label' not in job.keys()])
    
    xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
    xgb.fit(np.array(X_train), y_train)
    predict = xgb.predict(not_labeled_features)
    
    return predict

def get_green_jobs(limit, pretrained_model_path):
    """
    this function acts as a 'wrapper' function to return a dictionary
    of jobs that are classified as 'green'.

    :return: a list of dictionaries of 'green' jobs
    """

    labelled_jobs = label_jobs(limit)
    labelled_clean_jobs = clean_labelled_jobs(labelled_jobs)
    cleaned_queries = green_words_postprocess(pretrained_model_path)
    with_green_counts = add_green_counts(labelled_clean_jobs, cleaned_queries)
    labelled_transform = tfidf_vectorise(with_green_counts)
    X_train, y_train = oversample_training_ads(labelled_transform)
    predict = classify_jobs(X_train, y_train, labelled_transform)

    to_predict = [job for job in labelled_transform if 'label' not in job.keys()]
    
    predict_jobs = []
    for job, label in zip(to_predict, predict):
        job['label'] = label
        predict_jobs.append(job)

    green_jobs = [job for job in predict_jobs if job['label'] == 'green']

    return [i for n, i in enumerate(green_jobs) if i not in green_jobs[n + 1:]]
