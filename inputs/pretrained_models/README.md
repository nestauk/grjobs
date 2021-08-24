# Data Inputs

This inputs folder should contain the word2vec model needed for the keyword expansion step of the methodology. 

To download the pretrained w2v model:

```wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -P path/to/inputs/pretrained_models```

where `path/to` refers to wherever you have cloned the repository to the ```inputs/pretrained_models```.