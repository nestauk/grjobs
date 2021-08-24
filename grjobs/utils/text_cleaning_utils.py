# File: utils/text_cleaning_utils.py

"""Module for preprocessing and postprocessing of online job vacancy
   text data.

  Author: Karlis Kanders - modified by India Kerle

  Typical usage example:

  for text in list_of_job_description_texts:
    clean_text(text)

"""
# ---------------------------------------------------------------------------------

import pandas as pd
import string
from string import digits
from toolz import pipe
import re

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()

### Compiling regex patterns as they might get used many times over ###

# Hardcoded rules for dealing with punctuation marks and other custom symbols
punctuation_replacement_rules = {
    # old patterns: replacement pattern
    "[\u2022,\u2023,\u25E6,\u2043,\u2219]": ",",  # Convert bullet points to commas
    r"[-/:\\]": " ",  # Convert colon, hyphens and forward and backward slashes to spaces
    r"[^a-zA-Z0-9,.; #(++)]": "",  # Preserve spaces, commas, full stops, semicollons for discerning noun chunks
}

# Patterns for cleaning punctuation, for clean_punctuation()
compiled_punct_patterns = [re.compile(p) for p in punctuation_replacement_rules.keys()]
punct_replacement = list(punctuation_replacement_rules.values())

# Pattern for fixing a missing space between enumerations, for split_sentences()
compiled_missing_space_pattern = re.compile("([a-z])([A-Z])([a-z])")

compiled_nonalphabet_nonnumeric_pattern = re.compile(r"([^a-zA-Z0-9 #(++)+])")
compiled_padded_punctuation_pattern = re.compile(r"( )([^a-zA-Z0-9 #(++)+])")

### Components of the text preprocessing pipeline ###


def WordNetLemmatizer():
    nltk.download("wordnet")
    return nltk.WordNetLemmatizer()


def lemmatise(term):
    """Apply the NLTK WN Lemmatizer to the term"""
    lem = WordNetLemmatizer()
    return lem.lemmatize(term)


def clean_punctuation(text):
    """Replaces punctuation according to the predefined patterns"""
    for j, pattern in enumerate(compiled_punct_patterns):
        text = pattern.sub(punct_replacement[j], text)
    return text


def remove_punctuation(text):
    """Remove punctuation marks and replace with spaces (to facilitate lemmatisation)"""
    text = compiled_nonalphabet_nonnumeric_pattern.sub(r" ", text)
    return text


def pad_punctuation(text):
    """Pad punctuation marks with spaces (to facilitate lemmatisation)"""
    text = compiled_nonalphabet_nonnumeric_pattern.sub(r" \1 ", text)
    return text


def unpad_punctuation(text):
    """Remove spaces preceding punctuation marks"""
    text = compiled_padded_punctuation_pattern.sub(r"\2", text)
    return text


def detect_sentences(text):
    """
    Splits a word written in camel-case into separate sentences. This fixes a case
    when the last word of a sentence in not seperated from the capitalised word of
    the next sentence. This tends to occur with enumerations.
    For example, the string "skillsBe" will be converted to "skills. Be"
    Note that the present solution doesn't catch all such cases (e.g. "UKSkills")
    Reference: https://stackoverflow.com/questions/1097901/regular-expression-split-string-by-capital-letter-but-ignore-tla
    """
    text = compiled_missing_space_pattern.sub(r"\1. \2\3", text)
    return text


def lowercase(text):
    """Converts all text to lowercase"""
    return text.lower()


def lemmatize_paragraph(text):
    """
    Lemmatizes each word in a paragraph.
    Note that this function has to be included in a processing pipeline as, on
    its own, it does not deal with punctuation marks or capital letters.
    """
    text = " ".join([lemmatizer.lemmatize(token) for token in text.split(" ")])
    return text


def remove_punct(text):
    """
    removes punctuation
    """
    punct = string.punctuation
    translator = str.maketrans("", "", punct)
    no_punct = text.translate(translator)

    return no_punct


def remove_stopwords(text):
    """Removes stopwords"""
    stopws = stopwords.words("english")

    text = " ".join([token for token in text.split(" ") if token not in stopws])

    return text


def remove_job_stopwords(text):
    """removes job-related stopwords."""
    job_stops = [
        "recruit",
        "role",
        "cv",
        "currently",
        "skill",
        "website",
        "apply",
        "please",
        "background",
        "desirable",
        "someone",
        "salary",
        "work",
        "career",
        "job",
        "hour",
        "responsibility",
        "data",
        "now",
        "experience",
        "candidate",
        "application",
        "looking",
        "seeking",
        "hourly",
        "hour",
        "recruitment",
        "opportunity",
        "part",
        "exciting",
        "graduate",
        "consultant",
    ]
    text = " ".join([token for token in text.split(" ") if token not in job_stops])

    return text


def remove_digits(text):
    """takes as input a string and returns string stripped of digits."""

    nums = str.maketrans("", "", digits)
    no_digits = text.translate(nums)

    return no_digits


def clean_up(text):
    """Removes extra spaces between words"""
    text = " ".join(text)
    return text


def word_tokenize(text):
    """tokenises text to the sentence level"""
    words = tokenize.word_tokenize(text)
    return words


def clean_text(text):
    """
    Pipeline for preprocessing online job vacancy and skills-related text.
    """
    return pipe(
        text,
        detect_sentences,
        lowercase,
        clean_punctuation,
        pad_punctuation,
        lemmatize_paragraph,
        remove_stopwords,
        remove_job_stopwords,
        remove_digits,
        unpad_punctuation,
        remove_punct,
        word_tokenize,
        clean_up,
    )
