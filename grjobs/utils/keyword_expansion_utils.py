"""
utils.keyword_expansion_utils
--------------
Module for creating and expanding green keyword list. Author: India Kerle
"""

from text_cleaning_utils import *
from helper_utils import *

import gensim

def get_preliminary_green_words():
    ''':return: list of preliminary green words based on EGSS activity descriptions.'''
    
    preliminary_green_wordslist = ['waste disposal', 'sewage treatment', 'waste processing', 'recycling',
     'salvage wrecks', 'metal processing', 'water treatment', 'water reuse', 'renewable energy', 'renewable heat energy',
     'biofuels', 'heat pump', 'wastewater treatment', 'reforestation', 'afforestation', 'protected forests',
     'organic agriculture', 'thermal insulation production', 'vibration insulation production', 'environmental protection education',
     'conservation education', 'environmental policy', 'environmental public administration', 'environmental charities',
     'fauna flora preservation', 'energy installation', 'renewable energy technology', 'environmental advice', 'environmental engineering',
     'low emission vehicles', 'carbon capture', 'environmental industrial equipment']
    
    return preliminary_green_wordslist

def get_more_green_words():
	''':return: list of more green words.'''

	more_green_words = ['sustainability', 'green', 'esg', 'climate change', 'climate emergency', 'energy sustainability', 'offshore wind', 'environmental charity']

	return more_green_words

def get_w2v_model(pretrained_model_path):
    '''':return: gensim's pretrained word2vec model used to expand initial green words list.''' 

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model_path, binary = True) 
    
    return w2v_model

def get_expanded_green_words(pretrained_model_path):
    ''':return: expanded query list of green words.'''

    preliminary_green_words = get_preliminary_green_words()
    w2v_model = get_w2v_model(pretrained_model_path)

    expanded_queries = [[pair[0] for pair in w2v_model.most_similar(word.split(), topn = 10)] for word in preliminary_green_words]
    all_green_words = flatten_list(expanded_queries) + preliminary_green_words
    
    return all_green_words

def green_words_postprocess(pretrained_model_path):
    '''takes as input pretrained model path and postprocesses list, including removing hardcoded bad green words,
    lowering, removing symbols and adding hardcoded good green words. Returns cleaned, expanded list of green words.'''

# Hardcoded list of bad keywords based off of manual review. keyword is considered bad if it is: 1) too generic or 2) contextually unrelated 
    
    green_words_list = get_expanded_green_words(pretrained_model_path)
    
    bad_green_words = ['joseph schofer professor', 'reheats', 'pump', 'treament', 'car', 'engineering', 
                       'nonprofit', 're forestation', 'processing', 'wreck', 'robert schwaneberg covers',
                       'enviornmental', 'loving amoeba', 'carine weiss', 'technologies', 'co', 'lisa michals covers',
                       'addition exponent evaluates', 'dpfs', 'sage advice', 'processed', 'mont cuet', 'organics',
                       'pol icy', 'health', 'voc abatement', 'charitable organizations', 'waste msw', 'metal',
                       'systems crestron streamlines', 'project webfoot', 'matt krupnick covers', 'conventional combustion engines',
                       'health effects unesaki', 'retailing wholesaling', 'police', 'adminstration', 'technology', 'organic',
                       'charity', 'ethanol', 'melissa mc  ever covers', 'engineer', 'unroaded', 'envi ronmental', 'charity groundwork',
                       'warden dominick de  rose', 'policies', '###g co km', 'wrecked', 'agri', 'g co km', 'emmission', 
                       'nonprofits', 'tyrone jue', 'clergymen commemorated', 'fossil fuels', 'pubic', 'ed ucation', 'bedroom hathaway',
                       'environ mental', 'padrini proceeds', 'electromechanical assemblies', 'arahuay lone industry', 'charitable causes',
                       'airtightness', 'remove saucepan', 'ag', 'sage adivce', 'farming', 'congressman hulshof', 'capturing',
                       'polices', 'scott travis covers', 'pumps', 'sunken vessels', 'brownfield cleanups', 'potable', 'cars', 
                       'mnemo alheo complete', 'educational', 'chippy olver', 'policy', 'metals', 'eduction', 'melissa mcever covers', 
                       'elastomeric seals', 'warden dominick derose', 'tubular skylights', 'noncarbon', 'treatement', 'government', 'fossil fuel', 
                       'animal welfare', 'dehumidify', 'pumping', 'cwellyn', 'yokohama isogo area', 'educa tion', 'charitable', 
                       'public', 'co  ', 'electricity', 'gama neguma', 'dispose', 'forest', 'enviromental', 'habitat']
    

    cleaned_queries = list(set([green.lower().replace('_', ' ') for green in green_words_list]))
    cleaned_queries = [green for green in cleaned_queries if green not in bad_green_words] 
    cleaned_queries = flatten_list(cleaned_queries) + get_more_green_words()

    return cleaned_queries