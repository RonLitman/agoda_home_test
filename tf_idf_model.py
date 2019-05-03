import pandas as pd
import numpy as np
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics.pairwise import linear_kernel


from utils import *


def ngrams(string, n=3):
    '''
    define the ngram for the DF-IDF
    :param string:
    :param n: lenght of the gram
    :return: list of the n-grams
    '''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def match_name(names, tf_idf_matrix, name, country, list_names, list_key, list_country, min_score=0.0):
    '''
    get the closest name by cosine distance from the tf_idf natrix
    :param names: key for names of the tf_idf_matrix; pd.Series
    :param tf_idf_matrix: tf_idf matrix
    :param name: currant name from list number 1; str
    :param country: country of name from list number 1; str
    :param list_names: names from list number 2; list
    :param list_key: keys from list number 2; list
    :param list_country: countries from list number 2; list
    :param min_score: the threshold for making a prediction
    :return: max_name, max_score, key; list
    '''
    max_score = -1
    key = -1
    max_name = ""

    # we can use a linear kernel since we want the dot product
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

# load the data set
dict_df = load_data()

# add to the hotel name the city name and the address
dict_df['p1'] = dict_df['p1'].fillna(' ')
dict_df['p2'] = dict_df['p2'].fillna(' ')
dict_df['p1']['p1.hotel_name'] = dict_df['p1']['p1.hotel_name'] + '   ' + dict_df['p1']['p1.city_name'] + '   ' + dict_df['p1']['p1.hotel_address']
dict_df['p2']['p2.hotel_name'] = dict_df['p2']['p2.hotel_name'] + '   ' + dict_df['p2']['p2.city_name'] + '   ' + dict_df['p2']['p2.hotel_address']

# get a unique list of all the hotel + city name + address
names = dict_df['p1']['p1.hotel_name'].reset_index(drop=True)
names = names.append(dict_df['p2']['p2.hotel_name']).reset_index(drop=True)

# drop duplicates
names = names.drop_duplicates().reset_index(drop=True)

# fit TF-IDF
start_time = time.time()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(names)

dict_list = []

list_names = dict_df['p2']['p2.hotel_name']
list_key = dict_df['p2']['p2.key']
list_country = dict_df['p2']['p2.country_code']

# run over both lists and find the pairs
for name, key, country in zip(dict_df['p1']['p1.hotel_name'], dict_df['p1']['p1.key'], dict_df['p1']['p1.country_code']):

    match = match_name(names, tf_idf_matrix, name, country, list_names, list_key, list_country, 0.0)

    dict_ = {}
    dict_.update({"p1.key": key})
    dict_.update({"p2.key": match[2]})
    dict_.update({"p1.hotel_name": name})
    dict_.update({"p2.hotel_name": match[0]})
    dict_.update({"score": match[1]})
    dict_list.append(dict_)

print("--- %s seconds to find the closes string ---" % (time.time() - start_time))
merge_table = pd.DataFrame(dict_list)

# get stats on the result - can be done only when running on the examples
# print_results_stats(merge_table, dict_df['label_sample'])

# plot the number of predicted pair by the threshold
plot_n_preds_per_thres(merge_table)
merge_table.to_csv('test_sub.csv')




