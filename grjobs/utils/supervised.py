# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: green_jobs
#     language: python
#     name: green_jobs
# ---

# +
import sys
sys.path.insert(0, '/Users/india.kerlenesta/Projects/ojo/ojd_daps/')

from supervised_utils import *
from keyword_expansion_utils import *
from text_cleaning_utils import *
from helper_utils import *
# -

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# +
# 1. get and clean labelled data

labelled = load_json_from_s3('final_training_set')
clean_labelled = clean_labelled_jobs(labelled)

# +
# 2. generate bespoke feature 
#2.0 get expanded green terms list

pretrained_model_path = '/Users/india.kerlenesta/Projects/ojo/GoogleNews-vectors-negative300.bin.gz'
cleaned_queries = green_words_postprocess(pretrained_model_path)

# +
# 2. generate bespoke feature 
#2.1 add normalised green count 

with_green = add_green_counts(clean_labelled, cleaned_queries)

# +
# 3. convert to stacked tfidf

labelled_transform = tfidf_vectorise(with_green)

# +
# 4. Split labelled data

features = [job['vec'] for job in labelled_transform]
labels = [job['label'] for job in labelled_transform]

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    labels, 
                                                    test_size=0.1)

# +
# 5. oversampleget_green_jobs

oversample = SMOTE()
X_over, y_over = oversample.fit_resample(X_train, y_train)

# +
#6. classify

xgb = XGBClassifier(max_depth= 7, min_child_weight= 1)
xgb.fit(np.array(X_over), y_over)
predict = xgb.predict(np.array(X_test))

# +
#7. print classification report

target_names = ['green', 'not_green']
print(classification_report(y_test, predict, target_names=target_names))
# -

#0. test code
pretrained_model_path = '/Users/india.kerlenesta/Projects/ojo/GoogleNews-vectors-negative300.bin.gz'
get_green_jobs(1000, pretrained_model_path)
