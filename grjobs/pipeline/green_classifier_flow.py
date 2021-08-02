# File: pipeline/green_classifier_flow.py

"""A flow for identifiying job ad descriptions as 'green' or not.

Typical usage example:

    python grjobs/pipeline/green_classifier_flow.py run

"""
# ---------------------------------------------------------------------------------
import json

from metaflow import FlowSpec, step

from ojd_daps.dqa.data_getters import get_db_job_ads
from grjobs.pipeline.green_classifier import load_model
# ---------------------------------------------------------------------------------
# load config file
grjobs_config = get_yaml_config(Path(str(PROJECT_DIR) + "/grjobs/config/base.yaml"))

# get outputs path
directory_path = str(PROJECT_DIR) + grjobs_config['PRED_OUTPUT_PATH']

class GreenFlow(FlowSpec):
    
    @step
    def start(self):
        self.model = load_model('best_model')
        print('loaded model!')
        self.next(self.apply_model)
 
    @step
    def apply_model(self):
        jobs = [job for job in get_db_job_ads(limit = 100) if job['description'] != '[]']
        y_pred = self.model.predict(jobs)
        print('predicted if jobs are green!')

        green_jobs = dict()
        for job, label in zip(jobs, y_pred):
            green_jobs[job['id']] = label

        print(green_jobs)

        with open(directory_path + 'green_jobs_output.json', 'w') as f:
            json.dump(green_jobs, f)

        self.next(self.end)

    @step
    def end(self):
        print('finished running!')

if __name__ == '__main__':
    GreenFlow()