# %% [markdown]
# File: pipeline/green_classifier_flow.py

# %%
"""A flow for identifiying job ad descriptions as 'green' or not.

Typical usage example:

    python grjobs/pipeline/green_classifier_flow.py run

"""
# ---------------------------------------------------------------------------------
import json
import pickle
import datetime
# %%
from metaflow import FlowSpec, step, batch, retry

# %%
from grjobs import get_yaml_config, Path, PROJECT_DIR
from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs.pipeline.green_classifier import load_model
# ---------------------------------------------------------------------------------
# load config file
grjobs_config = get_yaml_config(Path(str(PROJECT_DIR) + "/grjobs/config/base.yaml"))

class GreenFlow(FlowSpec):
    
    @step
    def start(self):
        self.model = load_model('best_model')
        print('loaded model!')
        self.next(self.apply_model)

    @step
    def apply_model(self):
        jobs = [job for job in get_db_job_ads(limit = 100, return_features = True) if job['description'] != '[]']
        y_pred = self.model.predict(jobs)
        print('100 jobs classified!')
        self.next(self.end)

    @step
    def end(self):
        print('finished running!')

# %%
if __name__ == '__main__':
    GreenFlow()
