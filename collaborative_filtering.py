import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pickle

# cost function to minimize
# args: movie_params: # movies * # feats
#		user_params: # users * # feats
#		ratings_mat: # movies * # users (R(i,j) is the rating user j has given to movie i)
#		indicators_mat: # movies * # users (I(i,j)=1 if movie i was rated by user j)
#		reg: regularization factor > 0 in order to regularize

# returns: reg_cost: the squared error cost regularized by reg
# 		   grad: the two gradients in a matrix
def cost_function(params, ratings_mat, indicators_mat, num_users, num_movies, num_features, reg):

	"""returns the cost and gradient for the
	"""
	# Unfold the movie_params and user_params matrices from params
	movie_params = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
	user_params = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()

	cost = 0
	movie_params_grad = np.zeros(movie_params.shape)
	user_params_grad = np.zeros(user_params.shape)

	squared_error = (np.dot(user_params,movie_params.transpose()) - ratings_mat.transpose())**2
	cost = (sum(sum(indicators_mat*squared_error.transpose())))/2

	movie_params_grad = np.dot((np.dot(user_params,movie_params.transpose()) 
				- ratings_mat.transpose()).transpose()*indicators_mat,user_params)

	user_params_grad = np.dot(((np.dot(user_params,movie_params.transpose())
				 - ratings_mat.transpose()).transpose()*indicators_mat).transpose(),movie_params)

	# add regularization if reg > 0
	cost = cost + (reg/2)*sum(sum(movie_params**2)) + (reg/2)*sum(sum(user_params**2))

	movie_params_grad = movie_params_grad + reg * movie_params
	user_params_grad = user_params_grad + reg * user_params

	grad = np.hstack((movie_params_grad.T.flatten(),user_params_grad.T.flatten()))
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
	for i in range(n):
		idx = (indicators_mat[i,:]==1).nonzero()[0]
		if len(idx):
			ratings_mat_mean[i] = np.mean(ratings_mat[i, idx])
			ratings_mat_norm[i, idx] = ratings_mat[i, idx] - ratings_mat_mean[i]
		else:
			ratings_mat_mean[i] = 0.0
			ratings_mat_norm[i,idx] = 0.0

	return ratings_mat_norm, ratings_mat_mean
################################################################################################
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

###############################################################################################
print "Loading ratings data..."
ratings_mat = pd.read_csv("rating_mat.csv").as_matrix()
indicators_mat = pd.read_csv("indicator_mat.csv").as_matrix()

# content based features
num_feats = 10
num_movies = 10330
num_users = 668 

########################## create parameters to minimize ##############################
movie_params = np.random.rand(num_movies,num_feats)
user_params = np.random.rand(num_users,num_feats)

params = np.hstack((movie_params.T.flatten(), user_params.T.flatten()))

print ratings_mat.shape, indicators_mat.shape, movie_params.shape, user_params.shape


########################## Estimate initial cost ######################################
cost, grad = cost_function(params, ratings_mat, indicators_mat, num_users, num_movies, num_feats,1.5)

print "Initial cost is %f" %cost

checkCostFunction()

################################# Normalize feats #####################################
ratings_mat_norm, ratings_mat_mean = normalize_ratings(ratings_mat, indicators_mat)

########################## Learn the training parameters ##############################
# use conjugate gradients optimization

print "Start learning..."
# stack parameters in a matrix
initial_parameters = np.hstack((movie_params.T.flatten(), user_params.T.flatten()))

# regularize by 1.5
reg = 1.5

costFunc = lambda p: cost_function(p, ratings_mat_norm, indicators_mat, num_users, num_movies, num_feats, reg)[0]
gradFunc = lambda p: cost_function(p, ratings_mat_norm, indicators_mat, num_users, num_movies, num_feats, reg)[1]

result = minimize(costFunc, initial_parameters, method='CG', jac=gradFunc, options={'disp': True, 'maxiter': 1000.0})
theta = result.x
cost = result.fun

# unfold returned values
movie_params = theta[:num_movies*num_feats].reshape(num_movies, num_feats)
user_params = theta[num_movies*num_feats:].reshape(num_users, num_feats)

print 'Recommender system learning completed.'

###################### Save values in order to use our model later ####################
movie_params_file = open("movie_parameters.pkl", 'wb')
pickle.dump(movie_params, movie_params_file)
movie_params_file.close()

user_params_file = open("user_parameters.pkl", 'wb')
pickle.dump(user_params, user_params_file)
user_params_file.close()

ratings_mat_mean_file = open("ratings_mat_mean.pkl", 'wb')
pickle.dump(ratings_mat_mean, ratings_mat_mean_file)
ratings_mat_mean_file.close()

######################## Predict for user 1 #############################################

p = movie_params.dot(user_params.T)
my_predictions = p[:, 0] + ratings_mat_mean

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
post = pre[pre[:,1].argsort()[::-1]]
r = post[:,1]
ix = post[:,0]

movies = pd.read_csv('data-set.csv')
movie_title_dict = movies['title'].to_dict()

print '\nTop recommendations for you:'
for i in range(10):
	j = int(ix[i])
	#print movies[movies['movieId'] == movie_id]
	print 'Predicting rating %.1f for movie %s\n' % (my_predictions[j], movie_title_dict[j])

