# Identifying Jobs in EGSS

## Identifying Jobs in EGSS: Introduction

The aim of this project is to identify jobs in green industries within the [Open Jobs Observatory](https://github.com/nestauk/ojd_daps) job adverts database, in partnership with the Department of Education.

In parallel with government intervention to stimulate the green economy, we have developed one of the first open methodologies for automatically identifying job advertisements in green industries. This effort has come during a time of busy policy action: the UK Government has recently committed to creating and supporting millions of jobs in green industries by 2030, in a new Ten Point Plan for a Green Industrial Revolution report. They have also created a Green Jobs Taskforce to facilitate this goal. 

We chose to operationalise one official definition of jobs in green industries: the United Nations System of Environmental Accounting’s Environmental Goods and Services Sector (EGSS). The EGSS is made up of areas of the economy engaged in producing goods and services for environmental protection purposes, as well as those engaged in conserving and maintaining natural resources. There are 17 different UK specific activities associated with EGSS, including (but not limited to): wastewater management, forest management, environmental consulting and in-house business activities that include waste and recycling. Our methodology identifies both critical roles (e.g. a renewable energy engineer) and general roles (e.g. an accountant for a green energy company) within these sectors.

If you're interested in learning more about our results, [click here](https://www.nesta.org.uk/project-updates/green-jobs-results-OJO/). If you're interested in reading about the methodology, [click here](https://www.nesta.org.uk/project-updates/green-jobs-methodology-OJO/).

## Identifying Jobs in EGSS: Methodology

At the highest level, we took a supervised machine learning approach to identifying jobs in green industries. This meant that we manually labelled jobs as either green or not green according to the EGSS definition and trained a classifier to label unseen jobs as belonging to either of those categories. Please see below a diagram of the methodology:

<img width="1437" alt="methodology" src="https://user-images.githubusercontent.com/46863334/133442923-ce6d14e4-2103-4f54-a0d8-87285a7dc860.png">

The current methodology results in a weighted F1 score of: **94%**.

If you're interested in reading more about the methodology itself, [click here](https://www.nesta.org.uk/project-updates/green-jobs-methodology-OJO/).

## Running the Jobs in EGSS Pipeline

## [EXTERNAL] Applying the model

To apply the current model in the repo to job adverts outside the [Open Jobs Observatory (OJO) database](https://github.com/nestauk/ojd_daps), you will first need to a) clone the repository then b) create a grjobs virtual environment. 

To clone the repository:

git clone git@github.com:nestauk/grjobs.git

Then, you can create a virutal environment and install the relevant requirements by writing the following commands in your terminal:


`conda create --name grjobs` - to create the grjobs virtual environment

`conda activate grjobs` - to activate the grjobs virtual environment

`pip install -r requirements.txt` 
`pip install -e .` - to install the relevant modules

`conda install -c conda-forge py-xgboost` - to install xgboost

Then (in your activated grjobs environment), run the following:

```
from grjobs.pipeline.green_classifier import load_model
model = load_model('best_model')
model.predict([{'job_title_raw': 'job title', 'description': 'description'}])
```

Where the model takes a list of dictionaries of job adverts as input. 

**NOTE:** the job advert will need to be structured identically to jobs in the database i.e. the job advert must be a dictionary with keys `job_title_raw` and `description` containing the job title and job description as values. Also note that the model was originally built to identify jobs within the [Open Jobs Observatory (OJO) database](https://github.com/nestauk/ojd_daps) and so may not generalise as well to all job adverts. 

## [INTERNAL] Training/applying the model

To clone the repository:

`git clone git@github.com:nestauk/grjobs.git`

Assumed Python version: `python==3.8`

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt` and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Configure conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

Then (in your activated grjobs environment):

`conda install -c conda-forge py-xgboost` - to install xgboost

`conda install -c conda-forge hdbscan` - to install hdbscan for job title clustering analysis

To access the ojd_daps codebase and job ads data from the database, you will need to clone the ojd_daps repo by following instructions [here](https://github.com/nestauk/ojd_daps#for-contributors). Make sure you have run `export PYTHONPATH=$PWD` at the repository's root to access the codebase. You will need to either be on Nesta HQ's wifi or have your VPN turned on to access data from the database.

To train the model with model parameters in the `base.yaml` config file, run the following metaflow command (in your activated `grjobs` environment!):

`python grjobs/pipeline/train_flow.py run`

This will output a `'best_model.pkl'` that you can then load and apply to the job ads data.

You can run the trained model on data from the OJO database by running:

`python grjobs/pipeline/green_classifier_flow.py run`

This will apply the model to 100 job ads within the job ads database and assign an associated class (green or not_green) per job.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>

