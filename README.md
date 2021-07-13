# Green Jobs

## Identifying Green Jobs 

The aim of this project is to identify green jobs within the [Open Jobs Observatory](https://github.com/nestauk/ojd_daps) job adverts database as part of a case study deliverable for the Department of Education. 

This repo contains the current methodology for doing so. At a high level, the revised, supervised methodology is as follows:

1) Preprocess the text to: 
	a) Remove punctuation
	b) Detect sentences
	c) Lemmatise terms
	d) lowercase terms
	e) remove numbers
	f) remove stopwords

2) Generate bespoke normalised green count feature based on keyword expansion   
3) Create tfidf vectors and stack bespoke feature 
4) Oversample labelled training embeddings to address class imbalance using a Synthetic Minority Oversampling Technique
5) Train a gradient boosted decision tree algorithm to predict whether jobs are green or not_green      
The final output is a dictionary of jobs that have been classified as 'green' following the above methodology. 

The current methodology results in a weighted F1 score of: **94%**. 

To see methodology on labelled data, see ```supervised``` in utils. 

## Running the Green Jobs Pipeline

Assumed Python version: ```python==3.8```

To clone the repository:

```git clone git@github.com:nestauk/grjobs.git``` 

Please checkout an existing branch (for example, the branch for the PR you are reviewing), or checkout a new branch (which must conform to our naming convention). If you have already made changes to a branch, you should commit or stash these. Then (from the repo base):

```pip install -U -r requirements.txt``` - to upload the necessary requirements to run the script

```conda install -c anaconda py-xgboost``` - to install mac OS, anaconda-compatible xgboost (see known issue <a target="_blank" href="https://github.com/dmlc/xgboost/issues/1446">here</a>)

To import word2vec's pretrained model, run:

```wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"```

## To Dos

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
