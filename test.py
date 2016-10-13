import pandas as pd
import numpy as np

m_p = np.load("movie_parameters.pkl")
u_p = np.load("user_parameters.pkl")
df = pd.read_csv("data-set.csv")
r_m = np.load("ratings_mat_mean.pkl")

p = m_p.dot(u_p.T)
my_predictions = p[:, 0] + r_m

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
post = pre[pre[:,1].argsort()[::-1]]
r = post[:,1]
ix = post[:,0]


movie_title_dict = df['title'].to_dict()

print '\nTop recommendations for you:'
for i in range(50):
	j = int(ix[i])
	#print movies[movies['movieId'] == movie_id]
	print 'Predicting rating %.1f for movie %s\n' % (my_predictions[j], movie_title_dict[j])