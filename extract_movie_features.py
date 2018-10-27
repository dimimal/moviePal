import pandas as pd
import numpy as np

movies = pd.read_csv("data-set.csv")
genres_dict = {'Action': 0, 'Adventure':1, 'Animation':2, 'Children\'s':3, 'Comedy':4, 'Crime':5, 'Documentary':6, 
				'Drama':7, 'Fantasy':8, 'Film-Noir':9, 'Horror':10, 'Musical':11, 'Mystery':12, 'Romance':13, 'Sci-Fi':14, 
				'Thriller':15, 'War':16, 'Western':17}

genres = movies['Genres'].as_matrix()
gen_mat = np.zeros((genres.shape[0],18),dtype=int)

i = 0
for genre in genres:
	genre = genre.split('|')
	for g in genre:
		gen_mat[i,genres_dict[g]] = 1
	i += 1

G = pd.DataFrame(gen_mat)
G.to_csv('movie_feats_mat.csv',index=False)