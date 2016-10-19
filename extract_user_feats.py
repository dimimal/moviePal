import pandas as pd
import numpy as np
import operator

ratings = pd.read_csv("rating_mat.csv").as_matrix()
df = pd.read_csv("data-set.csv")
num_users = ratings.shape[1]
user_preferences = pd.DataFrame()
movie_genres_dict = df['Genres'].to_dict()

for i in range(num_users):
	usr_ratings = ratings[:,i]
	rated_movies = usr_ratings.nonzero()
	user_gen = {}

	for j in rated_movies[0]:
		movie_rating = int(ratings[j,i])
		genres = movie_genres_dict[j].split('|')
		for genre in genres:
			user_gen[genre] = user_gen.get(genre,0) + movie_rating

	sorted_user_pref = sorted(user_gen.items(), key=operator.itemgetter(1), reverse=True)
	if len(sorted_user_pref) > 2:
		first_pref, second_pref, third_pref = sorted_user_pref[0][0],sorted_user_pref[1][0],sorted_user_pref[2][0]
	elif len(sorted_user_pref) == 2:
		first_pref, second_pref, third_pref = sorted_user_pref[0][0],sorted_user_pref[1][0],None
	elif len(sorted_user_pref) == 1:
		first_pref, second_pref, third_pref = sorted_user_pref[0][0],None,None
	else:
		first_pref, second_pref, third_pref = None,None,None

	user_vector = [first_pref, second_pref, third_pref]
	user_preferences = user_preferences.append([user_vector], ignore_index=True)

user_preferences.columns = ['1st preference', '2nd preference', '3rd preference']
print user_preferences

user_preferences.to_csv('user_preferences.csv', index=False)
