from supervised_utils import *
from ojd_daps.dqa.data_getters import get_locations

##0. get Green Jobs 

pretrained_path = '/Users/india.kerlenesta/Projects/ojo/GoogleNews-vectors-negative300.bin.gz'
training_path = 'training_set.csv'
green = get_green_jobs(training_path, pretrained_path, None)

##1. Green jobs by location

def get_green_locations(green_jobs):
	'''input: dictionary of jobs labelled as green.
	   return: dictionary of green jobs with associated nuts 2 codes.'''
    
    job_locations = list(get_locations('reed', level = 'nuts_2'))
    
	green_locations = []
	for job in job_locations:
	    for green in green_jobs:
	        if job['job_id'] == green['id']:
	        	green['location'] = job['nuts_2_code']
	            green_locations.append(green)
    
    return green_locations 

##2. Green jobs by SOC

##3. Green jobs by SIC