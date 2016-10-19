import pandas as pd
import numpy as np

m_p = np.load("movie_parameters.pkl")
u_p = np.load("user_parameters.pkl")
df = pd.read_csv("data-set.csv")
r_m = np.load("ratings_mat_mean.pkl")
ratings = pd.read_csv("rating_mat.csv").as_matrix()	
indicators = pd.read_csv("indicator_mat.csv").as_matrix().astype(bool)

user = 1
p = m_p.dot(u_p.T)

my_predictions = p[:, (user )] + r_m

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])

post = pre[pre[:,1].argsort()[::-1]]

r = post[:,1]
ix = post[:,0]

movie_title_dict = df['Title'].to_dict()

print '\nTop recommendations for you:'
for i in range(10):
	j = int(ix[i])
	#print movies[movies['movieId'] == movie_id]
	movie = df[df['Title'] == movie_title_dict[j]]
	print 'Predicting rating %.1f for movie %s %s\n' % (my_predictions[j], movie_title_dict[j], movie['Genres'])
raw_input("Press key")
print '\nOriginal ratings provided:'
my_ratings = ratings[:, user]
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
    	movie = df[df['Title'] == movie_title_dict[i]]
        print 'Rated %d for %s %s\n' % (my_ratings[i], movie_title_dict[i], movie['Genres'])
         