import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import time

from utils import *


def match_name(name, country, list_names, list_key, list_country, min_score=0):
    # -1 score incase we don't get any matches
    max_score = -1
    key = -1
    # Returning empty name for no match as well
    max_name = ""
    # Iternating over all names in the other
    for name2, key2, country2 in zip(list_names, list_key, list_country):
        #Finding fuzzy match score
        # score = fuzz.ratio(name, name2)
        # score = fuzz.partial_ratio(name, name2)
        # score = fuzz.token_sort_ratio(name, name2)
        score = fuzz.token_set_ratio(name, name2)
        # Checking if we are above our threshold and have a better score
        if (score > min_score) & (score > max_score) & (country == country2):
            max_name = name2
            key = key2
            max_score = score
            if max_score == 100:
                break
    return (max_name, max_score, key)


dict_df = load_data()
# List for dicts for easy dataframe creation
dict_list = []

# iterating over our players without salaries found above
list_names = dict_df['label_sample']['p2.hotel_name']
list_key = dict_df['label_sample']['p2.key']
list_country = dict_df['label_sample']['p2.country_code']

start_time = time.time()
for name, key, country in zip(dict_df['label_sample']['p1.hotel_name'], dict_df['label_sample']['p1.key'], dict_df['label_sample']['p1.country_code']):

    # Use our method to find best match, we can set a threshold here
    match = match_name(name, country, list_names, list_key, list_country, 80)

    # New dict for storing data
    dict_ = {}
    dict_.update({"p1.key": key})
    dict_.update({"p2.key": match[2]})
    dict_.update({"p1.hotel_name": name})
    dict_.update({"p2.hotel_name": match[0]})
    dict_.update({"score": match[1]})
    dict_list.append(dict_)

print("--- %s seconds to find the closes string ---" % (time.time() - start_time))
merge_table = pd.DataFrame(dict_list)
print_results_stats(merge_table, dict_df['label_sample'])

