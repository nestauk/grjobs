# Green Jobs

## Identifying Green Jobs 

The aim of this project is to identify green jobs within the [Open Jobs Observatory](https://github.com/nestauk/ojd_daps) job adverts database as part of a case study deliverable for the Department of Education. 

This repo contains the methodology for doing so. At a high level, the methodology is as follows:

1) Preprocess the text to: 
	a) Remove punctuation
	b) Detect sentences
	c) Lemmatise terms
	d) lowercase terms
	e) remove numbers
	f) remove stopwords

2) Generate bespoke normalised green count feature based on keyword expansion   
3) Create tfidf vectors and stack bespoke feature 
4) Oversample labelled training embeddings to address class imbalance using a Synthetic Minority Oversampling Technique (SMOTE)
5) Train a gradient boosted decision tree algorithm to predict whether jobs are green or not_green 

The final output is a saved `.json` file of job IDs and their associated class: green or not_green.

The current methodology results in a weighted F1 score of: **94%**. 

## Running the Green Jobs Pipeline

To clone the repository: 

```git clone git@github.com:nestauk/grjobs.git``` 

Assumed Python version: ```python==3.8```

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt` and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS


Please checkout an existing branch (for example, the branch for the PR you are reviewing), or checkout a new branch (which must conform to our naming convention). If you have already made changes to a branch, you should commit or stash these. Then (from the repo base):

``` pip install -e .``` - to upload the necessary requirements to run the script

```conda install -c anaconda py-xgboost``` - to install mac OS, anaconda-compatible xgboost (see known issue <a target="_blank" href="https://github.com/dmlc/xgboost/issues/1446">here</a>)

Then, download the pretrained w2v model:

```wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -P path/to/inputs/pretrained_models```

To train the model with model parameters in the ```base.yaml``` config file, run the following metaflow command (in your activated `grjobs` environment!):

```python path/to/grjobs/pipeline/train_flow.py run```

This will output a ```'best_model.pkl'``` that you can then load and apply to the job ads data.

Alternatively, you can run the already saved, trained model on data from the database by running:

```python path/to/grjobs/pipeline/green_classifier_flow.py run```

This will apply the model to 100 job ads and output a ```.json``` dictionary with job ids and their associated class. For example:

```{'41547517': 'not_green', '41547520': 'not_green', '41547521': 'not_green'...}```

Do also make sure you have followed the instructions from the [ojd_daps](https://github.com/nestauk/ojd_daps#for-contributors) repo so you can access job ads data from the database. 

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
