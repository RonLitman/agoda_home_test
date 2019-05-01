import pandas as pd
import numpy as np
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import time
from scipy import spatial
from sklearn.metrics.pairwise import linear_kernel

from utils import *


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def match_name(names, tf_idf_matrix, name, country, list_names, list_key, list_country, min_score=0.0):
    # -1 score incase we don't get any matches
    max_score = -1
    key = -1
    # Returning empty name for no match as well
    max_name = ""
    # Iternating over all names in the other
    for name2, key2, country2 in zip(list_names, list_key, list_country):

        score = linear_kernel(tf_idf_matrix[names.index[names == name]], tf_idf_matrix[names.index[names == name2]]).flatten()[0] * 100
        # Checking if we are above our threshold and have a better score
        if (score > min_score) & (score > max_score) & (country == country2):
            max_name = name2
            key = key2
            max_score = score
            if max_score == 100:
                break
    return (max_name, max_score, key)


dict_df = load_data()


names = dict_df['label_sample']['p1.hotel_name'].reset_index(drop=True)
names = names.append(dict_df['label_sample']['p2.hotel_name']).reset_index(drop=True)

duplicated_name = names[names.duplicated()].copy()
duplicated_name = pd.DataFrame(duplicated_name, columns=['left_side']).reset_index(drop=True).drop_duplicates()
duplicated_name['right_side'] = duplicated_name['left_side']
duplicated_name['score'] = 1

names = names.drop_duplicates().reset_index(drop=True)

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(names)

# List for dicts for easy dataframe creation
dict_list = []

# iterating over our players without salaries found above
list_names = dict_df['label_sample']['p2.hotel_name']
list_key = dict_df['label_sample']['p2.key']
list_country = dict_df['label_sample']['p2.country_code']

start_time = time.time()
for name, key, country in zip(dict_df['label_sample']['p1.hotel_name'], dict_df['label_sample']['p1.key'], dict_df['label_sample']['p1.country_code']):

    # Use our method to find best match, we can set a threshold here
    match = match_name(names, tf_idf_matrix, name, country, list_names, list_key, list_country, 0.8)

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
merge_table.to_csv('test.csv')




