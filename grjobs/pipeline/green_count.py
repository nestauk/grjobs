# File: pipeline/green_count.py

"""Module for adding 'green' count feature to jobs description data.
"""
# ---------------------------------------------------------------------------------
import boto3
import json
import pickle
import collections

from functools import lru_cache
from nltk.tokenize import word_tokenize

#from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs.utils.text_cleaning_utils import clean_text

# ---------------------------------------------------------------------------------

S3_PATH = "labs/green-jobs/{}"
BUCKET_NAME = "open-jobs-lake"
# get model ouputs path
pretrained_model_path = str(PROJECT_DIR) + grjobs_config["MODEL_OUTPUT_PATH"]

@lru_cache(maxsize=None)
def load_from_s3(filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH.format(filename))
    return obj["Body"].read().decode()

def load_json_from_s3(prefix="final_training_set"):
    """Save data as json from S3"""
    return json.loads(load_from_s3(f"{prefix}.json"))

def load_pkl_from_s3(prefix="green_jobs_output"): 
    """Save data as pickle from S3"""
    s3 = boto3.resource("s3")
    return pickle.loads(s3.Bucket(BUCKET_NAME).Object(S3_PATH.format(f"{prefix}.pkl")).get()['Body'].read())

def green_count(text, green_words):
    """Counts number of any green terms or phrases per
    cleaned job description text.

    If green phrase is greater than one, count number of
    times whole phrase appears in description.
    If green phrase is one word, split text and count
    number of times word appears in description.
    Normalise count by length of job description and
    sum normalised counts.

    Returns:
        A green count 'score'. For example:

        0.3
    """

    text_tokenised = word_tokenize(text)
    text_length = len(text_tokenised)

    green_counts = []
    for green_word in green_words:
        if len(green_word.split(" ")) > 1:
            green_counts.append(text.count(green_word) / text_length)
        elif len(green_word.split(" ")) == 1:
            green_counts.append(text_tokenised.count(green_word) / text_length)

    return sum(green_counts)
