# Author: Alex Gidiotis 
# gidiotisAlex@outlook.com.gr

#====================================== COLABORATIVE FILTERING ALGORITHM ===========================================
# We estimate the user and movie parameters with collaborative filtering from the ratings each user gave each movie.
# Our objective is to minimize the squared error of prediction.
# The trained model will estimate the ratings that each user would give to the movies he hasn't already rated.

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.optimize import minimize
import pickle
import os

#============================= Load all the data we are going to use ==============================================
# Returns: ratings_mat: a num_movies x num_users matrix where R[i,j] is the rating that user j gave to the movie i.
#		   indicators: a num_movies x num_users matrix where I[i,j] is one if user j has rated movie i.
#		   movie_params: a num_movies x num_movie_feats matrix where X[i] is the feature vector of movie i.
#		   num_movie_feats, num_feats, num_movies, num_users
def load_data():
	
	# #_movies * #_users
	ratings_mat = pd.read_csv("rating_mat.csv").as_matrix()
	indicators_mat = pd.read_csv("indicator_mat.csv").as_matrix().astype(bool)
	num_movies = ratings_mat.shape[0]
	num_users = ratings_mat.shape[1] 
	#=============================================================================================================
	# Modified for more feats.
	#movie_params = pd.read_csv("movie_feats_mat.csv").as_matrix()
	num_feats = 20
	movie_params = np.random.rand(num_movies,num_feats)

	# content based features
	num_movie_feats = movie_params.shape[1]
	#num_feats = num_movie_feats
	#============================================================================================================
	print num_movies, num_users, num_feats
	return ratings_mat, indicators_mat, movie_params, num_movie_feats, num_feats, num_movies, num_users

#==================================== Cost and Gradients Computation =============================================

def cost_function(params, ratings_mat, indicators_mat, num_users, num_movies, num_features, Lambda):
	# Returns: the cost and gradient for the minimization objective.
 
	# Unfold the U and W matrices from params
	movie_params = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
	user_params = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()

	cost = 0
	movie_params_grad = np.zeros(movie_params.shape)
	user_params_grad = np.zeros(user_params.shape)

	# Notes: movie_params - num_movies  x num_features matrix of movie features
	#        user_params - num_users  x num_features matrix of user features
	#        ratings_mat - num_movies x num_users matrix of user ratings of movies
	#        indicators_mat - num_movies x num_users matrix, where indicators_mat(i, j) = 1 if the
	#            i-th movie was rated by the j-th user
	#        movie_params_grad - num_movies x num_features matrix, containing the
	#                 partial derivatives w.r.t. to each element of movie_params
	#        user_params_grad - num_users x num_features matrix, containing the
	#                     partial derivatives w.r.t. to each element of user_params
	#		 Lambda: is the regularization factor.
	# ============================================================================================================
	# Theta[j].T.dot(X[i]) is the prediction for the movie i by the user j.
	# J[X,Theta] = (1/2)*sum(sum((squared error)^2)) is the cost function.

	squared_error = (ratings_mat - np.dot(movie_params,user_params.T))**2
	cost = (sum(sum(indicators_mat*squared_error)))/2

	movie_params_grad = np.dot((np.dot(user_params,movie_params.transpose()) 
				- ratings_mat.transpose()).transpose()*indicators_mat,user_params)

	user_params_grad = np.dot(((np.dot(user_params,movie_params.transpose())
				 - ratings_mat.transpose()).transpose()*indicators_mat).transpose(),movie_params)

	# Add regularization if Lambda is greater than zero.
	cost = cost + (Lambda*sum(sum(movie_params**2)))/2 + (Lambda*sum(sum(user_params**2)))/2
	print cost

	movie_params_grad = movie_params_grad + Lambda * movie_params
	user_params_grad = user_params_grad + Lambda * user_params

	# The conjugate gradients optimization requires the returned values flattened in one vector.
	grad = np.hstack((movie_params_grad.T.flatten(),user_params_grad.T.flatten()))

	return cost, grad
#========================================= Gradient Checking =====================================================
# This part is used for debugging the gradient and cost functions.

from computeNumericalGradient import computeNumericalGradient

def checkCostFunction(Lambda=0):
	"""Creates a collaborative filering problem
	to check your cost function and gradients, it will output the
	analytical gradients produced by your code and the numerical gradients
	(computed using computeNumericalGradient). These two gradient
	computations should result in very similar values.
	"""

	## Create small problem
	X_t = np.random.rand(4, 3)
	Theta_t = np.random.rand(5, 3)

	# Zap out most entries
	Y = X_t.dot(Theta_t.T)
	Y[np.where(np.random.random_sample(Y.shape) > 0.5, True, False)] = 0
	R = np.zeros(Y.shape)
	R[np.where(Y != 0, True, False)] = 1

	## Run Gradient Checking
	X = np.random.random_sample(X_t.shape)
	Theta = np.random.random_sample(Theta_t.shape)
	num_users = Y.shape[1]
	num_movies = Y.shape[0]
	num_features = Theta_t.shape[1]

   # Unroll parameters
	params = np.hstack((X.T.flatten(), Theta.T.flatten()))

	costFunc = lambda t: cost_function(t, Y, R, num_users, num_movies, num_features, Lambda)

	def costFunc_w(t):
		Jgrad = costFunc(t)
		return Jgrad

	numgrad = computeNumericalGradient(costFunc_w, params)

	cost, grad = cost_function(params, Y, R, num_users, num_movies, num_features, Lambda)


	print np.column_stack((numgrad, grad))

	print 'The above two columns you get should be very similar.\n' \
			 '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n'

	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

	print 'If your backpropagation implementation is correct, then\n ' \
		  'the relative difference will be small (less than 1e-9). \n' \
		  '\nRelative Difference: %g\n' % diff

#============================================= Mean Normalization ================================================

def normalize_ratings(ratings_mat, indicators_mat):
	# Normalized ratings_mat so that each movie has a rating of 0 on average. 
	# Returns: the mean rating in Ymean and the normalized ratings matrix in Ynorm.

	m, n = ratings_mat.shape
	Ymean = np.zeros(m)
	Ynorm = np.zeros(ratings_mat.shape)
	# Subtract the mean of all non zero values from every row (movie)
	for i in range(m):
		idx = (indicators_mat[i,:]==1).nonzero()[0]
		if len(idx):
			Ymean[i] = np.mean(ratings_mat[i, idx])
			Ynorm[i, idx] = ratings_mat[i, idx] - Ymean[i]
		else:
			Ymean[i] = 0.0
			Ynorm[i, idx] = 0.0
	return Ynorm, Ymean

#=============================================== MAIN ============================================================
# The main function that goes through the training process.
print "Loading ratings data..."
ratings_mat, indicators_mat, movie_params, num_movie_feats, num_feats, num_movies, num_users = load_data()

#=================================== create parameters ================================================
# Create the user and movie parameters to estimate.
user_params = np.random.rand(num_users,num_feats)
params = np.hstack((movie_params.T.flatten(), user_params.T.flatten())) 

#===================================== Evaluate cost function ====================================================
# We check if the implementation of gradiet and cost is correct.
J, grad = cost_function(np.hstack((movie_params.T.flatten(), user_params.T.flatten())), ratings_mat, indicators_mat, num_users, num_movies,
			   num_feats, 0)

print 'Checking Gradients (without regularization) ...'
checkCostFunction()
print 'Checking Gradients (with regularization) ...'
checkCostFunction(1)

#========================================== Load Movies Data Set ===================================================
# Movies data set and dictionary used to make predictions.
movies = pd.read_csv('data-set.csv')
movie_title_dict = movies['Title'].to_dict()

#============================================ Normalize feats ======================================================
# Mean normalization of ratings.
ratings_mat_norm, ratings_mat_mean = normalize_ratings(ratings_mat, indicators_mat)
print "Start learning..."

#============================================ Learn the Model ======================================================
# We are going to use conjugate gradients optimization.
# Due to scaling issues with large data sets we are going to train on mini batches of 604 users.
# Train 10 epochs.
batch_size = 604
for j in range(100):
	# Go through all 10 mini-batches of users.
	for i in range(0,num_users,batch_size):
		print "Batch %d to %d users" %(i,(i+batch_size))
		# Mini-batch parameters
		num_users_batch = batch_size
		user_params_batch = user_params[i:(i + batch_size),:]

		ratings_mat_norm_batch = ratings_mat_norm[:,i:(i+batch_size)]
		indicators_mat_batch = indicators_mat[:,i:(i+batch_size)]

		# stack parameters in a vector.
		initial_parameters = np.hstack((movie_params.T.flatten(), user_params_batch.T.flatten()))
		# regularize by this value
		reg = 0.1

		# Cost and gradient functions for the optimization.
		costFunc = lambda p: cost_function(p, ratings_mat_norm_batch, indicators_mat_batch, 
					num_users_batch, num_movies, num_feats, reg)[0]
		gradFunc = lambda p: cost_function(p, ratings_mat_norm_batch, indicators_mat_batch, 
					 num_users_batch, num_movies, num_feats, reg)[1]

		# Use the conjugate gradients optimization.
		result = minimize(costFunc, initial_parameters, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 1})
		theta = result.x

		# Unfold returned values
		movie_params = theta[:num_movies*num_feats].reshape(num_movies, num_feats)
		user_params_batch = theta[num_movies*num_feats:].reshape(num_users_batch, num_feats)

		# Save the user batch parameters.
		if i > 0:
			user_params[i:(i + batch_size),:] = user_params_batch
		else:
			print "First mini batch finished"
			user_params[i:(i + batch_size),:] = user_params_batch
		cost = result.fun
	# Print rmse to monitor overall progress.
	print "epoch:",j,'rmse:',np.sqrt(np.sum((indicators_mat * 
						(ratings_mat - (np.dot(movie_params,user_params.T)+ratings_mat_mean.reshape((ratings_mat_mean.shape[0],1)))))**2)
						/len(ratings_mat[ratings_mat > 0]))

print 'Recommender system learning completed.'

#==================================== Save values in order to use our model later. ==============================================

movie_params_file = open("movie_parameters.pkl", 'wb')
pickle.dump(movie_params, movie_params_file)
movie_params_file.close()

user_params_file = open("user_parameters.pkl", 'wb')
pickle.dump(user_params, user_params_file)
user_params_file.close()

ratings_mat_mean_file = open("ratings_mat_mean.pkl", 'wb')
pickle.dump(ratings_mat_mean, ratings_mat_mean_file)
ratings_mat_mean_file.close()