import pandas as pd
import numpy as np
import re

def rmse(indicators,ratings,m_p,u_p):
    return np.sqrt(np.sum((indicators * (ratings - (np.dot(m_p.T,u_p))))**2)/len(ratings[ratings > 0]))

print "Loading parameters..."
m_p = np.load("movie_parameters.pkl")
u_p = np.load("user_parameters.pkl")
df = pd.read_csv("data-set.csv")

ratings = pd.read_csv("rating_mat.csv").as_matrix()	
indicators = pd.read_csv("indicator_mat.csv").as_matrix().astype(bool)
user_prefs = pd.read_csv("user_preferences.csv")
movie_genres_dict = df['Genres'].to_dict()

print "Computing rmse..."
print m_p.shape, u_p.shape
train_rmse = rmse(indicators,ratings,m_p,u_p)
print train_rmse


user = int(raw_input("Enter user: "))
genre = raw_input("Enter genre: ")
user_preference = user_prefs.iloc[user].values
print user_preference

p = np.dot(m_p.T,u_p)

my_predictions = p[:, (user )]

pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
'''
for pr in range(len(pre)):
	ix = pre[pr,0]
	genres = movie_genres_dict[ix].split('|')
	pref = 0.0
	for g in user_preference:
		pref += 1.0
		if g in genres:
			pre[pr,1] += 1.0/(pref + 2.0) 
			break
'''
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

raw_input("Press key")
'''
print '\nOriginal ratings provided:'
my_ratings = ratings[:, user]
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
    	movie = df[df['Title'] == movie_title_dict[i]]
        print 'Rated %d for %s %s\n' % (my_ratings[i], movie_title_dict[i], movie['Genres'].values[0])
'''