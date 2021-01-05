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
import os.path
from os import path


class Reco_content:
    def __init__(self):

        if path.exists("df.pickle"):
            with open('df.pickle', 'rb') as f:
                self.df = pickle.load(f)

        else:
            self.data = pd.read_csv("movies_metadata.csv")
            self.data = self.data.drop([19730,29503,35587])
            self.data['genres'] = self.data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
            self.df = self.data[['adult','genres','title','overview','vote_average','vote_count']]
            self.df = self.df.iloc[:20000,:]
            self.df['overview'] = self.df['overview'].fillna('')
            with open('df.pickle', 'wb') as f:
                pickle.dump(self.df, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Plot

        if path.exists("plot_sim.pickle"):
            print("TFIDF plot vecotor loaded from disk")
            with open('plot_sim.pickle', 'rb') as f:
                self.plot_sim = pickle.load(f)
        else:
            print("Creating TFIDF plot vector")
            self.tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['overview'])
            self.plot_sim= linear_kernel(self.tfidf_matrix,self.tfidf_matrix)
            with open('plot_sim.pickle', 'wb') as f:
                pickle.dump(self.plot_sim, f, protocol=pickle.HIGHEST_PROTOCOL)

        # # Genre
        if path.exists("merged.pickle"):
            with open('merged.pickle', 'rb') as f:
                self.merged = pickle.load(f)

        else:
            self.credits = pd.read_csv("credits.csv")
            self.keywords = pd.read_csv("keywords.csv")

            self.keywords['id'] = self.keywords['id'].astype('int')
            self.credits['id'] = self.credits['id'].astype('int')
            self.data['id'] = self.data['id'].astype('int')
            self.merged = self.data.merge(self.keywords,on='id')
            self.merged = self.merged.merge(self.credits,on='id')
            
            features = ['cast', 'crew', 'keywords']
            for feature in features:
                self.merged[feature] = self.merged[feature].apply(literal_eval)
        
            features = ['cast', 'keywords']
            for feature in features:
                self.merged[feature] = self.merged[feature].apply(self.get_list)

            self.merged['director'] = self.merged['crew'].apply(self.get_director)
            self.merged['Screenplay'] = self.merged['crew'].apply(self.get_screenplay)
            self.merged = self.merged[['genres','title','keywords','cast','director','Screenplay']]

            features = ['genres','keywords','cast','director','Screenplay']
            for feature in features:
                self.merged[feature] = self.merged[feature].apply(self.clean_data)

            self.merged['metadata'] = self.merged.apply(self.soup,axis = 1)

            self.merged = self.merged.iloc[:20000,:]
            with open('merged.pickle', 'wb') as f:
                pickle.dump(self.merged, f, protocol=pickle.HIGHEST_PROTOCOL)   


        if path.exists("crew_sim.pickle"):
            print("TFIDF Crew vector loaded from disk")
            with open('crew_sim.pickle', 'rb') as f:
                self.crew_sim = pickle.load(f)
        else:
            print("Creating TFIDF crew vector")
            self.count = CountVectorizer(stop_words='english')
            self.count_matrix = self.count.fit_transform(self.merged['metadata'])
            self.crew_sim = linear_kernel(self.count_matrix,self.count_matrix)
            with open('crew_sim.pickle', 'wb') as f:
                pickle.dump(self.crew_sim, f, protocol=pickle.HIGHEST_PROTOCOL)

               
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

    #Get crew list
    def get_list(self,x):
        if isinstance(x,list):
            words = []
            for i in x:
                words.append(i['name'])
            
            if len(words)>3:
                words = words[:3]
                
            return words
        return[]

    #get director list
    def get_director(self,x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    #get screenplay list
    def get_screenplay(self,x):
        for i in x:
            if i['job'] == 'Screenplay':
                return i['name']
        
        return np.nan

    #clean data to create soup
    def clean_data(self,x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    def soup(self,x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['Screenplay'])+ ' ' + ' '.join(x['genres'])
    

    """
    Get recommendations based on keywords in the plot
    """
    def get_reco_plot(self,title):
        indices = pd.Series(self.df.index, index = self.df['title'])
        try:
            idx = indices[title]
        except KeyError:
            res = self.cold_start(title)
            return res
        sim_score = list(enumerate(self.plot_sim[idx]))
        sim_score = sorted(sim_score,key = lambda x : x[1], reverse = True)
        sim_score = sim_score[1:11]
        movie_indices = [i[0] for i in sim_score]
        res =  self.df['title'].iloc[movie_indices]
        res = res.values.tolist()
        return res

    """
    Get recommendations based on genre, cast, director and screenwriter
    """

    def get_reco_crew(self,title):
        indices = pd.Series(self.merged.index , index = self.merged['title'])
        try:
            ind = indices[title]
        except KeyError:
            res = self.cold_start(title)
            return res
        sim_scores = list(enumerate(self.crew_sim[ind]))
        sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in  sim_scores]
        res = self.merged['title'].iloc[movie_indices]
        res = res.values.tolist()
        return res



   