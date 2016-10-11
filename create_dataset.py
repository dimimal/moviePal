# Takes input from: movies.csv, links.csv, ratings.csv, ratings.list located in the path directory
#
# Outputs: rating_mat.csv, indicator_mat.csv, data-set.csv in the current directory

import numpy as np
import pandas as pd
import re

path = "C:\Users\Alex\Documents\University\Python\Data\Movie_Lens_Latest"
#####################################################################
# Create Movie Data Set
# imdb links of the form :http://www.imdb.com/title/tt"imdbId"/

movies = pd.read_csv(path + '/movies.csv')
links = pd.read_csv(path + '/links.csv')
links = links.drop(['tmdbId','movieId'], axis=1)
movies = pd.concat([movies, links], axis=1)

# movie ids differ from movie index so we need to dictionaries to get from one value to the other
# dictionary with movie indx -> movie id values
movie_id_dict = movies['movieId'].to_dict()
# invert to dictionary with movie id -> movie indx values
rev_movie_id_dict = {v: k for k, v in movie_id_dict.items()}

######################################################################
## Create user and rating Data-set
num_users = 668
num_movies = 10330

ratings = pd.read_csv(path + '/ratings.csv')
ratings = ratings.drop('timestamp', axis=1)

# create ratings and indicator matrices
# ratings mat(# movies * # users) with values R(i,j) the rating of movie i by user j
# indicator mat(# movies * # users) with values I(i,j) 1 if movie i was rated by user j

rating_mat = np.zeros((num_movies,num_users),dtype=float)
indicator_mat = np.zeros((num_movies,num_users),dtype=int)

for i in range(1,num_users):
	user_ratings = ratings[ratings['userId']==i]
	for j in user_ratings.index:
		movie_id = user_ratings.loc[j,'movieId']
		usr_rating = user_ratings.loc[j,'rating']
		rating_mat[int(rev_movie_id_dict[movie_id]),int(i-1)] = usr_rating
		indicator_mat[int(rev_movie_id_dict[movie_id]),int(i-1)] = 1

# Save the two matrices to be used for collaborative filtering
R = pd.DataFrame(rating_mat) 
I = pd.DataFrame(indicator_mat)

R.to_csv('rating_mat.csv',index=False)
I.to_csv('indicator_mat.csv',index=False)

#######################################################################
## Add user movie rating
movie_ratings = []
# get the median value of all ratings for a movie
for i in range(0,(num_movies-1)):
	movie = ratings[ratings['movieId']==movie_id_dict[i]]
	avg_rating = movie['rating'].median()
	movie_ratings.append(avg_rating)#

movies['user rating'] = movie_ratings


#######################################################################
# Get IMDB rating
ratings_file = open(path + '/ratings.list', 'r')
imdb_ratings = ratings_file.read()
imdb_rating_list = []

# Look into the imdb ratings file for the movies in our data set and 
# get the ratings
print len(movies.index)
for i in movies.index:
	print i
	# get movie titles
	title = movies.loc[i,'title'].split(',')[0]
	title = title.split('(')[0].rstrip()
	print title
	# create a regex to get the ratings
	regex = '(\d\.\d)  \S*\s?'+title
	imdb_rating = re.findall(regex,imdb_ratings)
	# some movies may not have an imdb rating
	if len(imdb_rating) > 0:
		imdb_rating_list.append(imdb_rating[0])
	else:
		imdb_rating_list.append(0.0)
	print imdb_rating_list[-1]	

imdb_rating = pd.Series(imdb_rating_list)

#######################################################################
# Output to .csv
imdb_rating.to_csv('imdb_rating.csv', index=False)
movies.to_csv('data-set.csv', index=False)	

