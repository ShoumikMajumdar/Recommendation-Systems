# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:46:11 2021

@author: Shoum
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class Reco:
    def __init__(self):
        self.data = pd.read_csv("movies_metadata.csv")
        self.data['genres'] = self.data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        
        self.df = self.data[['adult','genres','title','overview','vote_average','vote_count']]
        self.df = self.df.iloc[:20000,:]
        self.df['overview'] = self.df['overview'].fillna('')
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['overview'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix,self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index = self.df['title'])

    def cold_start(self, title,percentile=0.95):
        vote_avg = self.df[self.df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_avg.mean()
        
        vote_ct = self.df[self.df['vote_count'].notnull()]['vote_count'].astype('int')
        m = vote_ct.quantile(percentile)
        
        qualified = self.df[(self.df['vote_count'] >= m) & (self.df['vote_average'].notnull()) & (self.df['vote_count'].notnull())][['title','vote_count', 'vote_average']]
        
        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(10)
        
        qualified = qualified.values.tolist()
        return qualified

    def get_reco(self,title):
        try:
            idx = self.indices[title]
        except KeyError:
            res = self.cold_start(title)
            return res
        sim_score = list(enumerate(self.cosine_sim[idx]))
        sim_score = sorted(sim_score,key = lambda x : x[1], reverse = True)
        sim_score = sim_score[1:11]
        movie_indices = [i[0] for i in sim_score]
        res =  self.df['title'].iloc[movie_indices]
        res = res.values.tolist()
        return res

    