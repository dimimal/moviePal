import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pickle
import os

# cost function to minimize
# args: movie_params: # movies * # feats
#		user_params: # users * # feats
#		ratings_mat: # movies * # users (R(i,j) is the rating user j has given to movie i)
#		indicators_mat: # movies * # users (I(i,j)=1 if movie i was rated by user j)
#		reg: regularization factor > 0 in order to regularize


# returns: reg_cost: the squared error cost regularized by reg
# 		   grad: the two gradients in a matrix
def cost_function(params, movie_params, ratings_mat, indicators_mat, num_users, num_movies, num_features, reg):

	"""returns the cost and gradient for the
	"""
	# Unfold the movie_params and user_params matrices from params
	user_params = np.array(params).reshape(num_features, num_users).T.copy()

	cost = 0
	user_params_grad = np.zeros(user_params.shape)

	squared_error = (np.dot(user_params,movie_params.transpose()) - ratings_mat.transpose())**2
	cost = (sum(sum(indicators_mat*squared_error.transpose())))/2

	user_params_grad = np.dot(((np.dot(user_params,movie_params.transpose())
				 - ratings_mat.transpose()).transpose()*indicators_mat).transpose(),movie_params)

	# add regularization if reg > 0
	cost = cost + (reg/2)*sum(sum(user_params**2))

	user_params_grad = user_params_grad + reg * user_params

	grad = user_params_grad.T.flatten()
	print cost
	return cost, grad

# normalize ratings matrix
# args: ratings_mat: matrix to be normalized
# 		indicators_matrix: matrix with the indicators
# returns: ratings_mat_norm: the normalized matrix
#		   ratings_mat_mean: a vector with the row means

def normalize_ratings(ratings_mat, indicators_mat):
	"""normalized ratings_mat so that each movie has a rating of 0 on average,
	and returns the mean rating in ratings_mat_mean.
	"""
	m, n = ratings_mat.shape
	ratings_mat_mean = np.zeros(m)
	ratings_mat_norm = np.zeros(ratings_mat.shape)

	# subtract the mean of all non-zero values for each row
	for i in range(m):
		idx = (indicators_mat[i,:]==1).nonzero()[0]
		if len(idx):
			ratings_mat_mean[i] = np.mean(ratings_mat[i, idx])
			ratings_mat_norm[i, idx] = ratings_mat[i, idx] - ratings_mat_mean[i]
		else:
			ratings_mat_mean[i] = 0.0
			ratings_mat_norm[i,idx] = 0.0
	return ratings_mat_norm, ratings_mat_mean
#=================================================================================================
# This part is used for debugging
# Checks numerical and gradients for consistency

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
	params = np.hstack((Theta.T.flatten()))

	costFunc = lambda t: cost_function(t, movie_params, Y, R, num_users, num_movies, num_features, Lambda)

	def costFunc_w(t):
		Jgrad = costFunc(t)
		return Jgrad

	numgrad = computeNumericalGradient(costFunc_w, params)

	cost, grad = cost_function(params, movie_params, Y, R, num_users, num_movies, num_features, Lambda)


	print np.column_stack((numgrad, grad))

	print 'The above two columns you get should be very similar.\n' \
			 '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n'

	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

	print 'If your backpropagation implementation is correct, then\n ' \
		  'the relative difference will be small (less than 1e-9). \n' \
		  '\nRelative Difference: %g\n' % diff

#=============================================== MAIN ================================================
print "Loading ratings data..."
# #_movies * #_users
ratings_mat = pd.read_csv("rating_mat.csv").as_matrix()
indicators_mat = pd.read_csv("indicator_mat.csv").as_matrix().astype(bool)
movie_params = pd.read_csv("movie_feats_mat.csv").as_matrix()

# content based features
num_movie_feats = movie_params.shape[1]
num_feats = num_movie_feats
num_movies = ratings_mat.shape[0]
num_users = ratings_mat.shape[1]
print num_movies, num_users

#=================================== create parameters to minimize ===================================
user_params = np.random.rand(num_users,num_feats)

params = user_params.T.flatten()

print ratings_mat.shape, indicators_mat.shape, movie_params.shape, user_params.shape


#======================================= Estimate initial cost =======================================
cost, grad = cost_function(params, movie_params, ratings_mat, indicators_mat, num_users, num_movies, num_feats, 0)

print "Initial cost is %f" %cost

#checkCostFunction()
#========================================== Add some ratings =========================================
movies = pd.read_csv('data-set.csv')
movie_title_dict = movies['Title'].to_dict()
#  Initialize my ratings
my_ratings = np.zeros(num_movies)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[109] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
#my_ratings[1060] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
#my_ratings[71] = 3
my_ratings[1188] = 5
#my_ratings[] = 4
my_ratings[293] = 5
#my_ratings[65] = 3
my_ratings[2260] = 5
#my_ratings[] = 4
my_ratings[1238] = 5
my_ratings[2890] = 5

print 'New user ratings:'
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print 'Rated %d for %s\n' % (my_ratings[i], movie_title_dict[i])

#  Add our own ratings to the data matrix
ratings_mat = np.column_stack((my_ratings, ratings_mat))
indicators_mat = np.column_stack((my_ratings, indicators_mat)).astype(bool)

num_users = ratings_mat.shape[1]

# use previous parameters if available.
#if os.path.exists("movie_parameters.pkl"):
#	print "Loading previous model..."
#	movie_params = np.load("movie_parameters.pkl")
#	user_params = np.load("user_parameters.pkl")
#	print movie_params.shape, user_params.shape
## else randomly initialize
#else:
print "Random initialization."

user_params = np.random.rand(num_users,num_feats)

#================================== Normalize feats ==================================
ratings_mat_norm, ratings_mat_mean = normalize_ratings(ratings_mat, indicators_mat)
print "Start learning..."
#================================== Learn the training parameters ==================================
# We are going to use conjugate gradients optimization.
# Due to scaling issues with large data sets we are going to train on mini batches of 604 users.
# Train 10 epochs.
batch_size = 604
for j in range(10):
	for i in range(0,(num_users - batch_size),batch_size):
		print "Batch %d to %d users" %(i,(i+batch_size))
		# Batch parameters
		num_users_batch = batch_size
		user_params_batch = user_params[i:(i + batch_size),:]

		ratings_mat_norm_batch = ratings_mat_norm[:,i:(i+batch_size)]
		indicators_mat_batch = indicators_mat[:,i:(i+batch_size)]

		# stack parameters in a matrix
		initial_parameters = user_params_batch.T.flatten()
		# regularize by this value
		reg = 1.5

		# Cost and gradient functions for the optimization.
		costFunc = lambda p: cost_function(p, movie_params, ratings_mat_norm_batch, indicators_mat_batch,
					num_users_batch, num_movies, num_feats, reg)[0]
		gradFunc = lambda p: cost_function(p, movie_params, ratings_mat_norm_batch, indicators_mat_batch,
					 num_users_batch, num_movies, num_feats, reg)[1]

		result = minimize(costFunc, initial_parameters, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 100})
		theta = result.x

		# unfold returned values
		user_params_batch = theta.reshape(num_users_batch, num_feats)
		if i > 0:
			user_params[i:(i + batch_size),:] = user_params_batch

		else:
			print "First mini batch finished"
			user_params[i:(i + batch_size),:] = user_params_batch
		cost = result.fun

print 'Recommender system learning completed.'

#===================== Save values in order to use our model later =====================
movie_params_file = open("movie_parameters.pkl", 'wb')
pickle.dump(movie_params, movie_params_file)
movie_params_file.close()

user_params_file = open("user_parameters.pkl", 'wb')
pickle.dump(user_params, user_params_file)
user_params_file.close()

ratings_mat_mean_file = open("ratings_mat_mean.pkl", 'wb')
pickle.dump(ratings_mat_mean, ratings_mat_mean_file)
ratings_mat_mean_file.close()

#================================= Predict for user 1 =================================

p = movie_params.dot(user_params.T)
my_predictions = p[:, 0] + ratings_mat_mean

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
post = pre[pre[:,1].argsort()[::-1]]
r = post[:,1]
ix = post[:,0]

print '\nTop recommendations for you:'
for i in range(10):
	j = int(ix[i])
	#print movies[movies['movieId'] == movie_id]
	print 'Predicting rating %.1f for movie %s\n' % (my_predictions[j], movie_title_dict[j])

