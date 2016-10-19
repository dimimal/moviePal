import pandas as pd
import numpy as np
import re

m_p = np.load("movie_parameters.pkl")
u_p = np.load("user_parameters.pkl")
df = pd.read_csv("data-set.csv")
r_m = np.load("ratings_mat_mean.pkl")
#ratings = pd.read_csv("rating_mat.csv").as_matrix()	
#indicators = pd.read_csv("indicator_mat.csv").as_matrix().astype(bool)
user_prefs = pd.read_csv("user_preferences.csv")
movie_genres_dict = df['Genres'].to_dict()

user = int(raw_input("Enter user: "))
genre = raw_input("Enter genre: ")
user_preference = user_prefs.iloc[user].values
print user_preference

p = m_p.dot(u_p.T)

my_predictions = p[:, (user )] + r_m

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])

for pr in range(len(pre)):
	ix = pre[pr,0]
	genres = movie_genres_dict[ix].split('|')
	pref = 0.0
	for g in user_preference:
		pref += 1.0
		if g in genres:
			pre[pr,1] += 1.0/(pref + 1.0) 
			break

post = pre[pre[:,1].argsort()[::-1]]

r = post[:,1]
ix = post[:,0]

movie_title_dict = df['Title'].to_dict()

if genre == "":
	count = 50
else:
	count = 100
print '\nTop recommendations for you:'
for i in range(count):
	j = int(ix[i])
	#print movies[movies['movieId'] == movie_id]
	movie = df[df['Title'] == movie_title_dict[j]]
	genres = movie['Genres'].values[0]
	genres_regex =  "("+ genre + ")"
	if re.search(genres_regex,genres):
		print 'Predicting rating %.1f for movie %s %s\n' % (r[i], movie_title_dict[j], movie['Genres'].values[0])
#raw_input("Press key")
#print '\nOriginal ratings provided:'
#my_ratings = ratings[:, user]
#for i in range(len(my_ratings)):
#    if my_ratings[i] > 0:
#    	movie = df[df['Title'] == movie_title_dict[i]]
#        print 'Rated %d for %s %s\n' % (my_ratings[i], movie_title_dict[i], movie['Genres'].values[0])
         