# File: pipeline/train_flow.py

"""A flow for training the model to predict job ad descriptions as 'green' or not.

Typical usage example:
    
    python grjobs/pipeline/train_flow.py run
    
"""
# ---------------------------------------------------------------------------------
from metaflow import FlowSpec, step

from grjobs.pipeline.green_classifier import GreenClassifier

from grjobs.getters.keywords import get_expanded_green_words
from grjobs.pipeline.create_labelled_data import create_labelled_data
# ---------------------------------------------------------------------------------

class TrainGreenFlow(FlowSpec):
    
    @step
    def start(self):
        self.green_words = get_expanded_green_words()
        self.labelled_data = create_labelled_data('final_training_set', None, self.green_words)
        self.model = GreenClassifier() 
        self.next(self.split_data)

    @step
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.split_data(
            self.labelled_data, 0.1, verbose=True)
        self.next(self.fit_model)

    @step
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        green_class_results = self.model.evaluate(self.y_test, self.predictions, verbose = True)
        self.next(self.save)

    def save(self):
        self.model.save_model('best_model')
        self.next(self.end)

    @step
    def end(self):
        print('finished running!')

if __name__ == '__main__':
    TrainGreenFlow()