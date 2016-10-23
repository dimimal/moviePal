# Author: Alex Gidiotis
# gidiotisAlex@outlook.com.gr

# We use the Alternating Least Squares collaborative filtering algorithm to train a movie recommender system.
# The dataset we currently use is the Movie_Lens_1M dataset.
# The model learns the user and movie parameters in order to make good predictions for different users.

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
import pickle

# ======================================= Load data and split into training and test sets =====================================================================
# Loads the data and returns all the data-structures we are going to use.
# Returns: movies: a data-frame with all the movie ids and titles
#		   movie_id_dict, rev_movie_id_dict: dictionaries we use to match movie ids to indexes
#		   n_users, n_movies: the dimensionality of the data
#		   train_data, test_data: 75%-25% split of the initial dataset
#		   ratings_mat, test_mat: the training and test data in matrix format. n_users x n_movies and R(i,j) is the rating that user i gave to movie j.
#		   indicators_mat, test_indicators_mat: matrices we use for training. n_users x n_movies and I(i,j) is 1 if user i has rated movie j.

def load_data(path):
	header = ['user_id', 'item_id', 'rating', 'timestamp']
	ratings = pd.read_csv(path + '/ratings.csv', sep=',', names=header)
	movies = pd.read_csv(path + '/movies.csv')

	# Create a movie indx -> id dictionary.
	movie_id_dict = movies['MovieID'].to_dict()
	# Invert dictionary movie id -> movie indx
	rev_movie_id_dict = {v: k for k, v in movie_id_dict.items()}

	# The data dimensions.
	n_users = ratings.user_id.unique().shape[0]
	n_movies = movies.shape[0]

	print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies)

	print "Num users:",n_users,"Num movies:",n_movies

	# Split the data into training set (75%) and test set(25%).
	train_data, test_data = cv.train_test_split(ratings,test_size=0.25)

	train_data = pd.DataFrame(train_data)

	test_data = pd.DataFrame(test_data)

	print "Creating training and test matrices..."
	# Create training and test matrix
	# Ratings matrices are n_users x n_movies and R(i,j) is the rating that user i gave to movie j.
	ratings_mat = np.zeros((n_users, n_movies))
	for line in train_data.itertuples():
		movie_idx = rev_movie_id_dict[line[2]]
		ratings_mat[line[1]-1, movie_idx] = line[3]  

	test_mat = np.zeros((n_users, n_movies))
	for line in test_data.itertuples():
		movie_idx = rev_movie_id_dict[line[2]]
		test_mat[line[1]-1, movie_idx] = line[3]

	# Indicators matrix for training data: n_users x n_movies and I(i,j) is 1 if user i has rated movie j.
	indicators_mat = ratings_mat.copy()
	indicators_mat[indicators_mat > 0] = 1
	indicators_mat[indicators_mat == 0] = 0

	# Indicators matrix for test data: n_users x n_movies and I(i,j) is 1 if user i has rated movie j.
	test_indicators_mat = test_mat.copy()
	test_indicators_mat[test_indicators_mat > 0] = 1
	test_indicators_mat[test_indicators_mat == 0] = 0
	return movies, movie_id_dict, rev_movie_id_dict, n_users, n_movies,train_data, test_data, ratings_mat, test_mat, indicators_mat, test_indicators_mat

# =============================== Implement the RMSE cost function to optimize ===============================================================================
# Computes the root mean squared error for a set of parameters.
def rmse(indicators_mat,ratings_mat,movie_feats,user_feats):
	rmse = np.sqrt(np.sum((indicators_mat * (ratings_mat - np.dot(user_feats.T,movie_feats)))**2)/len(ratings_mat[ratings_mat > 0]))
	return rmse

# ========================================================= Main ==============================================================================================
path = "C:\Users\Alex\Documents\University\Python\Data\Movie_Lens_1M"
movies, movie_id_dict, rev_movie_id_dict, n_users, n_movies,train_data, test_data, ratings_mat, test_mat, indicators_mat, test_indicators_mat = load_data(path)

# Initialize model and training parameters.
lmbda = 0.1 # Regularisation weight
k = 20 # Dimensionality of latent feature space.
m, n = ratings_mat.shape # Number of users and movies for the training set.
n_epochs = 15 # Number of epochs

print "Creating parameters to minimize..."
user_feats = 3 * np.random.rand(k,m) # Latent user feature matrix
movie_feats = 3 * np.random.rand(k,n) # Latent movie feature matrix
# Initially set movie features to the avg movie rating.
movie_feats[0,:] = ratings_mat[ratings_mat != 0].mean(axis=0) # Avg. rating for each movie
E = np.eye(k) # (k x k)-dimensional idendity matrix

# Lists we may use to plot the training curves.
train_errors = []
test_errors = []

# ========================================================= Learn the Model ==================================================================================
print "Start learning..."
# Repeat until convergence.
for epoch in range(n_epochs):
	# Fix movie feats and estimate user feats
	print "Estimating user feats..."
	for i, Ii in enumerate(indicators_mat):
		nui = np.count_nonzero(Ii) # Number of items user i has rated
		if (nui == 0): nui = 1 # Be aware of zero counts!
	
		# Least squares solution
		Ai = np.dot(movie_feats, np.dot(np.diag(Ii), movie_feats.T)) + lmbda * nui * E
		Vi = np.dot(movie_feats, np.dot(np.diag(Ii), ratings_mat[i].T))
		user_feats[:,i] = np.linalg.solve(Ai,Vi)
	
	print "Estimating movie feats..."    
	# Fix user feats and estimate movie feats
	for j, Ij in enumerate(indicators_mat.T):
		nmj = np.count_nonzero(Ij) # Number of users that rated item j
		if (nmj == 0): nmj = 1 # Be aware of zero counts!
		
		# Least squares solution
		Aj = np.dot(user_feats, np.dot(np.diag(Ij), user_feats.T)) + lmbda * nmj * E
		Vj = np.dot(user_feats, np.dot(np.diag(Ij), ratings_mat[:,j]))
		movie_feats[:,j] = np.linalg.solve(Aj,Vj)
	
	# Estimate the training and test error for each epoch.
	train_rmse = rmse(indicators_mat,ratings_mat,movie_feats,user_feats)
	test_rmse = rmse(test_indicators_mat,test_mat,movie_feats,user_feats)
	train_errors.append(train_rmse)
	test_errors.append(test_rmse)
	
	print "[Epoch %d/%d] train error: %f, test error: %f" \
	%(epoch+1, n_epochs, train_rmse, test_rmse)
	
print "Algorithm converged"

# ==================================================== Prepare to Make Predictions =============================================================
# Calculate prediction matrix R_hat (low-rank approximation for ratings_mat)
R_hat = pd.DataFrame(np.dot(user_feats.T,movie_feats))
ratings_mat = pd.DataFrame(ratings_mat)

# Compare true ratings of user 17 with predictions
ratings = pd.DataFrame(data=ratings_mat.loc[16,ratings_mat.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16,ratings_mat.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']

print ratings

predictions = R_hat.loc[16,ratings_mat.loc[16,:] == 0] # Predictions for movies that the user 17 hasn't rated yet
top5 = predictions.sort_values(ascending=False).head(n=5)
recommendations = pd.DataFrame(data=top5)
recommendations.columns = ['Predicted Rating']

print recommendations

# =========================================== Save output Model for later use ========================================================================
# We need to save the following: ratings_mat, R_hat, user_feats, movie_feats, ratings
print "writing output..."
movie_params_file = open("movie_parameters.pkl", 'wb')
pickle.dump(movie_feats, movie_params_file)
movie_params_file.close()

user_params_file = open("user_parameters.pkl", 'wb')
pickle.dump(user_feats, user_params_file)
user_params_file.close()

ratings_mat_file = open("ratings_mat.pkl", 'wb')
pickle.dump(ratings_mat, ratings_mat_file)
ratings_mat_file.close()

R_hat_file = open("R_hat.pkl", 'wb')
pickle.dump(R_hat, R_hat_file)
R_hat_file.close()

ratings.to_csv("Output_ratings.csv", index=False)
