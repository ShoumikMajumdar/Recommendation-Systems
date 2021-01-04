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


def get_reco(df,title,indices, cosine_sim):
    idx = indices[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score,key = lambda x : x[1], reverse = True)
    sim_score = sim_score[1:11]
    movie_indices = [i[0] for i in sim_score]
    return df['title'].iloc[movie_indices]



def main():
    data = pd.read_csv("movies_metadata.csv")
    data.shape
    data['genres'] = data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    df = data[['adult','genres','title','overview','vote_average','vote_count']]
    df = df.iloc[:20000,:]
    df['overview'] = df['overview'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    tfidf_matrix.shape
    cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
    
    indices = pd.Series(df.index, index = df['title'])
    
    res = get_reco(df,"Some movie",indices,cosine_sim)
    

if __name__=='__main__':
    main()