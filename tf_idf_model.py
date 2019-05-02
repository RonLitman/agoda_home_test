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
from sklearn.neighbors import KNeighborsClassifier

from utils import *


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def match_name(names, tf_idf_matrix, name, country, list_names, list_key, list_country, min_score=0.0):
    max_score = -1
    key = -1
    max_name = ""

    cosine_similarities = linear_kernel(tf_idf_matrix[names.index[names == name]], tf_idf_matrix).flatten() * 100
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]

    for index in related_docs_indices:
        score = cosine_similarities[index]
        name2 = names[index]
        if name2 in list_names.unique():
            key2 = list_key[list_names == name2].values[0]
            country2 = list_country[list_names == name2].values[0]
            if (score > min_score) & (country == country2):
                max_name = name2
                key = key2
                max_score = score
                break

    return max_name, max_score, key


dict_df = load_data()


names = dict_df['p1']['p1.hotel_name'].reset_index(drop=True)
names = names.append(dict_df['p2']['p2.hotel_name']).reset_index(drop=True)

duplicated_name = names[names.duplicated()].copy()
duplicated_name = pd.DataFrame(duplicated_name, columns=['left_side']).reset_index(drop=True).drop_duplicates()
duplicated_name['right_side'] = duplicated_name['left_side']
duplicated_name['score'] = 1

names = names.drop_duplicates().reset_index(drop=True)

start_time = time.time()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(names)

dict_list = []

list_names = dict_df['p2']['p2.hotel_name']
list_key = dict_df['p2']['p2.key']
list_country = dict_df['p2']['p2.country_code']


for name, key, country in zip(dict_df['p1']['p1.hotel_name'], dict_df['p1']['p1.key'], dict_df['p1']['p1.country_code']):

    match = match_name(names, tf_idf_matrix, name, country, list_names, list_key, list_country, 20)

    dict_ = {}
    dict_.update({"p1.key": key})
    dict_.update({"p2.key": match[2]})
    dict_.update({"p1.hotel_name": name})
    dict_.update({"p2.hotel_name": match[0]})
    dict_.update({"score": match[1]})
    dict_list.append(dict_)

print("--- %s seconds to find the closes string ---" % (time.time() - start_time))
merge_table = pd.DataFrame(dict_list)
# print_results_stats(merge_table, dict_df['label_sample'])
merge_table.to_csv('test.csv')




