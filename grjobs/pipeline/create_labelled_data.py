# File: pipeline/create_training_data.py

"""Module for adding labels, cleaning text and
adding 'green' count feature to jobs description data.

  Typical usage example:

  labelled_data = create_labelled_data('final_training_set', None, green_words)

"""
# ---------------------------------------------------------------------------------
import boto3
import json
import pickle
import collections

from functools import lru_cache
from nltk.tokenize import word_tokenize

from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs.utils.text_cleaning_utils import clean_text

# ---------------------------------------------------------------------------------

S3_PATH = "labs/green-jobs/{}"
BUCKET_NAME = "open-jobs-lake"


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


def create_labelled_data(prefix, limit, green_words):
    """clean job descriptions, add green counts and labels
    to job descriptions.

    Returns:
        A list of dictionaries of job advertisements
        with cleaned text, green counts and training labels (where labelled).

        For example:

         {'__version__': '21.05.12.134_batch',
        'id': 41547521,
        'data_source': 'Reed',
        'created': 1607644800000,
        'url': None,
        's3_location': 'reed-41547521_test-False.txt',
        'job_title_raw': 'Porta Cabin Cleaner',
        'job_location_raw': 'Biggleswade, Bedfordshire',
        'job_salary_raw': '10.1300-10.7900',
        'company_raw': 'Berry Recruitment',
        'contract_type_raw': 'Temporary',
        'description': "[ We are looking for a cleaner for a...
        'label': 'not_green',
        'clean_description': 'cleaner two week contract...
        'green_count': 0.0}
    """

    training_data = load_json_from_s3(prefix)
    job_ads = [job for job in get_db_job_ads(limit=limit) if job["description"] != "[]"]

    labelled_jobs = collections.defaultdict(dict)
    for ads in (training_data, job_ads):
        for ad in ads:
            ad["clean_description"] = clean_text(
                ad["job_title_raw"] + " " + ad["description"]
            )
            ad["green_count"] = green_count(ad["clean_description"], green_words)
            labelled_jobs[ad["id"]].update(ad)

    return list(labelled_jobs.values())
