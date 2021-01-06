import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from os import path

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class Reco_colab:
    def __init__(self):
        # Load Datasets

        if path.exists("pearsons.pickle") and path.exists("n_ratings.pickle"):
            with open('pearsons.pickle', 'rb') as f:
                self.item_corr_matrix = pickle.load(f)       

            with open('n_ratings.pickle', 'rb') as f:
                self.n_ratings = pickle.load(f)       
                    
        else:
            data = pd.read_csv("movies.csv")
            ratings = pd.read_csv("ratings.csv")
            data = data.merge(ratings)
            data = data[['movieId','title','userId','rating']]
            data['title'] = data['title'].apply(lambda x: x.split('(')[0])
            self.n_ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
            self.n_ratings['total ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())
            self.n_ratings.rename(columns = {'rating': 'mean ratings'}, inplace=True)
            self.n_ratings.sort_values('total ratings', ascending=False)
            util_mat = data.pivot_table(index = 'userId', columns = 'title', values = 'rating')
            item_util_matrix = util_mat.apply(lambda col : col.fillna(col.mean()), axis=0)
            self.item_corr_matrix = item_util_matrix.corr()
            with open('pearsons.pickle', 'wb') as f:
                pickle.dump(self.item_corr_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open('n_ratings.pickle', 'wb') as f:
                pickle.dump(self.n_ratings, f, protocol=pickle.HIGHEST_PROTOCOL)



    def get_recommended(self,item, min_ratings = 100):
        movie_corr = self.item_corr_matrix[item]
        movie_corr = movie_corr.sort_values(ascending=False)
        movies_similar_to_item = pd.DataFrame(data=movie_corr.values, columns=['Correlation'], index = movie_corr.index)
        movies_similar_to_item = movies_similar_to_item.join(self.n_ratings['total ratings'])
        movies_similar_to_item = movies_similar_to_item[1:]
        res =  movies_similar_to_item[movies_similar_to_item['total ratings'] > min_ratings ].sort_values(ascending=False,by=['Correlation']).head(10)
        res = res.reset_index().values.tolist()['title']
        return res                                                                           