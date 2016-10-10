import pandas as pd
import numpy as np

# cost function to minimize
# args: movie_params: # movies * # feats
#		user_params: # users * # feats
#		ratings_mat: # movies * # users (R(i,j) is the rating user j has given to movie i)
#		indicators_mat: # movies * # users (I(i,j)=1 if movie i was rated by user j)
#		reg: regularization factor > 0 in order to regularize

# returns: reg_cost: the squared error cost regularized by reg
# 		   movie_grad: # movies * # feats 
#		   user_grad: # users * # feats
def cost_function(movie_params, user_params, ratings_mat, indicators_mat,reg):

	squared_error = (np.dot(user_params,movie_params.transpose()) - ratings_mat.transpose())**2
	cost = (sum(sum(indicators_mat*squared_error.transpose())))/2

	movie_grad = np.dot((np.dot(user_params,movie_params.transpose()) 
				- ratings_mat.transpose()).transpose()*indicators_mat,user_params)

	user_grad = np.dot(((np.dot(user_params,movie_params.transpose())
				 - ratings_mat.transpose()).transpose()*indicators_mat).transpose(),movie_params)

	# add regularization if reg > 0
	cost_reg = cost + (reg/2)*sum(sum(movie_params**2)) * (reg/2)*sum(sum(user_params**2))

	movie_grad_reg = movie_grad + reg * movie_params
	user_grad_reg = user_grad + reg * user_params

	return cost_reg, movie_grad_reg, user_grad_reg

print "Loading ratings data..."
ratings_mat = pd.read_csv("rating_mat.csv").as_matrix()
indicators_mat = pd.read_csv("indicator_mat.csv").as_matrix()

# content based features
num_feats = 100
num_movies = 10330
num_users = 668 

# create parameters to minimize
movie_params = np.random.rand(num_movies,num_feats)
user_params = np.random.rand(num_users,num_feats)

print ratings_mat.shape, indicators_mat.shape, movie_params.shape, user_params.shape

cost, movie_grad, user_grad = cost_function(movie_params, user_params, 
									ratings_mat, indicators_mat,1.5)

print "Initial cost is %f" %cost