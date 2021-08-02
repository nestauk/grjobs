# File: getters/keywords.py

"""Module for creating and expanding green keyword list.

  Typical usage example:

  green_words = get_expanded_green_words()

"""
# ---------------------------------------------------------------------------------

import glob
import gensim

from grjobs import get_yaml_config, Path, PROJECT_DIR

# ---------------------------------------------------------------------------------

# Load config file
grjobs_config = get_yaml_config(Path(str(PROJECT_DIR) + "/grjobs/config/base.yaml"))

# get green lists path
green_list_path = str(PROJECT_DIR) + grjobs_config["GREEN_LIST_PATH"]

# get pretrained model path
pretrained_model_path = str(PROJECT_DIR) + grjobs_config["PRETRAINED_PATH"]


def get_expanded_green_words() -> list:
    """Generates list of green words via keyword expansion.

    Retrieves three .txt files: initial green words list,
    general green words list and bad green words list.
    Expands initial green words list using a word2vec model
    and post processes list by lowering, replacing symbols and
    removing terms based on being present in bad green words list.

    Returns:
        A list of green words based off of initial EGSS list, keyword
        expansion and general green words list. For
        example:

        [waste management,
        carbon capture,
        sustainability,
        ...]
    """

    all_green_lists = [
        open(file).read().split("\n") for file in glob.glob(green_list_path + "*.txt")
    ]

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
        pretrained_model_path, binary=True
    )

    expanded_queries = []
    for green_word in all_green_lists[1]:
        expanded_green_words = [
            similar_term[0]
            for similar_term in w2v_model.most_similar(
                green_word.split(), topn=grjobs_config["similar_words"]
            )
        ]
        for expanded_green_word in expanded_green_words:
            expanded_queries.append(expanded_green_word.lower().replace("_", " "))

    all_green_words = list(
        set(
            [
                green_word
                for green_word in expanded_queries
                if green_word not in all_green_lists[0]
            ]
            + all_green_lists[1]
            + all_green_lists[2]
        )
    )

    return all_green_words
