import numpy as np
import pandas as pd

path = "/home/dimitris/GitProjects/moviePal/Movie Data"

# ======================== Loads user, movie data and ratings from csv =======================
# Returns three DataFrames with user, movie data and ratings
# movies: num movies x 3 dataframe
# users: num users x 4 dataframe
# ratings: num ratings x 3 dataframe
def read_data():
	# read file with training raitings
	ratings = pd.read_csv(path + "/ratings.csv", header=None)
	# UserID::MovieID::Rating::Timestamp
	ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
	# drop the useless column
	ratings = ratings.drop('Timestamp', axis=1)
	ratings = ratings.sort_values(by='MovieID')

	# read file with user info
	users = pd.read_csv(path + "/users.csv", header=None)
	#UserID::Gender::Age::Occupation::Zip-code
	users.columns = ['UserID','Gender','Age','Occupation','Zip-code']
	users.drop('Zip-code',axis=1)
	users = users.sort_values(by='UserID')

	# read file with movie data
	# MovieID::Title::Genres
	movies = pd.read_csv(path + "/movies.csv")
	return movies, users, ratings

def create_user_feats():
	# check for male or female
	m_f = {'M':1,'F':0}
	gend = []
	for g in users['Gender'].as_matrix():
		gend.append(m_f[g])
	users['feat 1'] = gend
	# check age
	age = []
	for a in users['Age'].as_matrix():
		age.append(a/(56.0 - 18.0))
	users['feat 2'] = age
	# check occupation
	occ = []
	for o in users['Occupation'].as_matrix():
		occ.append((o + 1)/21.0)
	users['feat 3'] = occ
	print users

# ========================== Creates the matrices for collaborative filtering =======================
# Returns the R and I data frames with the ratings
# R: num movies x num users data frame. R(i,j) is the rating that user j has given tp movie i
# I: num movies x num users data frame. I(i,j) is one if user j has rated movie i else zero.
def create_ratings_matrix(ratings,rev_movie_id_dict,num_movies,num_users):
	ratings_mat = np.zeros((num_movies,num_users),dtype=float)
	indicators_mat = np.zeros((num_movies,num_users),dtype=int)

	# go through the ratings
	for i in range(1,num_users):
		user_ratings = ratings[ratings['UserID']==i]
		for j in user_ratings.index:
			movie_id = user_ratings.loc[j,'MovieID']
			usr_rating = user_ratings.loc[j,'Rating']
			ratings_mat[int(rev_movie_id_dict[movie_id]),int(i-1)] = float(usr_rating)
			indicators_mat[int(rev_movie_id_dict[movie_id]),int(i-1)] = 1

	R = pd.DataFrame(ratings_mat)
	I = pd.DataFrame(indicators_mat)

	R.to_csv('rating_mat.csv',index=False)
	I.to_csv('indicator_mat.csv',index=False)

	return R, I

# ====================== Add average user rating for movies =============================
def avg_user_rating(ratings,movies,movie_id_dict,num_movies):
	movie_ratings = []
	# get the median value of all ratings for a movie
	for i in range(0,(num_movies)):
		movie = ratings[ratings['MovieID']==movie_id_dict[i]]
		avg_rating = movie['Rating'].median()
		movie_ratings.append(avg_rating)#

	movies['user rating'] = movie_ratings
	return movies

# ======================================== Main =========================================
# Load the data
print "Loading data..."
movies, users, ratings = read_data()
num_movies = movies.shape[0]
num_users = users.shape[0]
# movie ids differ from movie index so we need to dictionaries to get from one value to the other
# dictionary with movie indx -> movie id values
movie_id_dict = movies['MovieID'].to_dict()
# invert to dictionary with movie id -> movie indx values
rev_movie_id_dict = {v: k for k, v in movie_id_dict.items()}
# Create the matrices
print "Creating the matrices..."
ratings_mat, indicators_mat = create_ratings_matrix(ratings,rev_movie_id_dict,num_movies,num_users)
print ratings_mat.shape, indicators_mat.shape
# Add avg user rating for movies
print "Rating..."
movies = avg_user_rating(ratings,movies,movie_id_dict,num_movies)
# Output to .csv
print "Writing output..."
movies.to_csv('data-set.csv', index=False)
