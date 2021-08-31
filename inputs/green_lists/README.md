# Data Inputs

This inputs folder contains three `.txt` files relevant to the keyword expansion step of the methodology:

1. initial_green_list.txt
2. general_green_list.txt
3. bad_green_words.txt

## `initial_green_list.txt`

The `initial_green_list.txt` contains **manually generated key words and phrases** that map onto the 17 EGSS activities described by the ONS. You can learn more about the EGSS [here](https://www.ons.gov.uk/economy/environmentalaccounts/bulletins/ukenvironmentalaccounts/2010to2015) and investigate the ONS methodological annex that describes the activities [here](https://www.ons.gov.uk/economy/environmentalaccounts/datasets/ukenvironmentalgoodsandservicessectoregssmethodologyannex).

## `general_green_list.txt`

The `general_green_list.txt` contains **key words and phrases** that are considered 'general' green terms such as 'sustainability' or 'green'.

## `bad_green_words.txt`

The `bad_green_words.txt` contains **words and phrases** that are considered 'bad' green words. This list is used during the post processing of keyword expansion. Words and phrases generated via keyword expansion are considered bad if the term is 1) too generic or 2) contextually unrelated.
