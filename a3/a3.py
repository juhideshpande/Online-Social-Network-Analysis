
# coding: utf-8

# In[1]:


from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


# In[2]:


def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


# In[4]:


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


# In[5]:


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    tokens=[]
    for u in movies['genres']:
        tokens.append(tokenize_string(u))
    movies['tokens']=tokens
    return movies


# In[62]:


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    begin = 0
    words_added = []
    int_list = []
    result_stored = []
    words = defaultdict(lambda:0)
    for u in range(len(movies)):
        words_added.append(movies['tokens'][u])

    for ur in words_added:
        for d in ur:
            int_list.append(d)

    values_sum = Counter()
    for jj in list(movies['tokens']):
        values_sum.update(set(jj))

    for var in sorted(set(int_list)):
        words[var] = begin
        begin = begin + 1

    tot_movies = len(movies)
    range_stored = range(len(movies))

    for sk in range_stored:
        temp = Counter()
        temp.update(movies['tokens'][sk])
        answer = sorted(movies['tokens'][sk])
        intermediate= defaultdict(lambda: 0)
        jw = [] 
        jx = [] 
        jy = [] 
        jz = []
        for v in range(len(answer)):
            if answer[v] not in jz:
                jz.append(answer[v])
                xyz = temp[answer[v]]
                temp_maxm = max(temp.values())
                num_division = (xyz / temp_maxm)
                denom_log = math.log(( tot_movies / values_sum[answer[v]]),10)
                intermediate[answer[v]] = num_division * denom_log
                jw.append(intermediate[answer[v]])
                jy.append(0)
                jx.append(words[answer[v]])
        result_stored.append(csr_matrix((jw, (jy, jx)), shape=(1, len(words))))
    movies['features'] = pd.Series(result_stored, index=movies.index)


    return tuple((movies, words))
      


# In[58]:


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


# In[59]:


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    dot_num=np.dot(a.toarray(),b.toarray().T)
    star_denom=(np.linalg.norm(a.toarray()))*(np.linalg.norm(b.toarray()))
    return dot_num[0][0]/star_denom
#         dot_num=np.dot(a,b)
#         star_denom=(np.linalg.norm(a))*(np.linalg.norm(b))
#         return dot_num[0][0]/star_denom 


# In[60]:


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ans_list= []
    for s1, k in ratings_test.iterrows():
        g = ratings_train[ratings_train.userId == k['userId']]
        kg = movies[movies.movieId == k['movieId']].iloc[0]

        kx, ky, kz = [], [], []
        for s2, j in g.iterrows():
            train_movie = movies[movies.movieId == j['movieId']].iloc[0]

            ans = cosine_sim(kg['features'], train_movie['features'])
            kz.append(j['rating'])
            if ans > 0:
                kx.append(ans)
                ky.append(ans * j['rating'])

        ans_list.append(np.mean(kz) if len(kx) == 0 else np.mean(ky) / np.mean(kx))
    return np.array(ans_list)


# In[64]:


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()
def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()

