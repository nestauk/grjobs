"""
utils.helper_utils
--------------
Module for helper functions. Author: India Kerle
"""

def flatten_list(list_of_lists):
    '''takes as input lists of lists and returns one flattened list.'''

    flatten = [item for sublist in list_of_lists for item in sublist]

    return flatten