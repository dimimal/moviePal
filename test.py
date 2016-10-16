import pandas as pd
import numpy as np

m_p = np.load("movie_parameters.pkl")
u_p = np.load("user_parameters.pkl")
df = pd.read_csv("data-set.csv")
r_m = np.load("ratings_mat_mean.pkl")

user = 0
p = m_p.dot(u_p.T)
my_predictions = p[:, (user + 1)] + r_m

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
post = pre[pre[:,1].argsort()[::-1]]
r = post[:,1]
ix = post[:,0]


movie_title_dict = df['Title'].to_dict()

print '\nTop recommendations for you:'
for i in range(10):
	j = int(ix[i])
	#print movies[movies['movieId'] == movie_id]
	print 'Predicting rating %.1f for movie %s\n' % (my_predictions[j], movie_title_dict[j])

ratings = pd.read_csv("rating_mat.csv").as_matrix()	

print '\nOriginal ratings provided:'
my_ratings = ratings[:, user]
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print 'Rated %d for %s\n' % (my_ratings[i], movie_title_dict[i])