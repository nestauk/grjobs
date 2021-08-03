# File: pipeline/train_flow.py

"""A flow for training the model to predict job ad descriptions as 'green' or not.

Typical usage example:
    
    python grjobs/pipeline/train_flow.py run
    
"""
# ---------------------------------------------------------------------------------
from metaflow import FlowSpec, step

from grjobs.pipeline.green_classifier import GreenClassifier

from grjobs.getters.keywords import get_expanded_green_words
from grjobs.pipeline.create_labelled_data import create_labelled_data, load_from_s3, load_json_from_s3
# ---------------------------------------------------------------------------------

class TrainGreenFlow(FlowSpec):
    
    @step
    def start(self):
        self.labelled_data = load_json_from_s3('final_training_set')
        self.model = GreenClassifier() 
        self.next(self.split_data)

    @step
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.split_data(
            self.labelled_data, 0.1, verbose=True)
        print('split training data!')
        self.next(self.fit_model)

    @step
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        print('made predictions!')
        self.next(self.evaluate)

    @step
    def evaluate(self):
        green_class_results = self.model.evaluate(self.y_test, self.predictions, verbose = True)
        print('evaluated model!')
        self.next(self.save)

    @step
    def save(self):
        self.model.save_model('best_model')
        print('saved model!')
        self.next(self.end)

    @step
    def end(self):
        print('finished running!')

if __name__ == '__main__':
    TrainGreenFlow()