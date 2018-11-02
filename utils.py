# Import packages
import os
import pandas as pd

"""
DIR_PATH = os.path.dirname(__file__)
DATASET_FOLDER = "ml-1m"
"""

# Specify User's Age and Occupation Column
AGES = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"}

# Tag Ids
GENRES = {
            'Void': 0,
            'Action': 1,
            'Adventure': 2,
            'Animation': 3,
            'Children\'s': 4,
            'Comedy': 5,
            'Crime': 6,
            'Documentary': 7,
            'Drama': 8,
            'Fantasy': 9,
            'Film-Noir': 10,
            'Horror': 11,
            'Musical': 12,
            'Mystery': 13,
            'Romance': 14,
            'Sci-Fi': 15,
            'Thriller': 16,
            'Western': 17,
            'War': 18}

OCCUPATIONS = {
        0: "other or not specified",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"}


def load_data(path):
    """load_data: Loads the data from the dat files
    args:
        path: The directory which contains the necessary
        folders.

    returns: The dataframe with the data
             ratings = [user_id, movie_id, rating, emb_id, emb_]
    """

    # Read the Ratings File
    ratings = pd.read_csv(
            os.path.join(path, "ml-1m", "ratings.dat"),
            sep='::',
            engine='python',
            encoding='latin-1',
            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Load Users file
    users = pd.read_csv(
            os.path.join(path, "ml-1m", "users.dat"),
            sep="::",
            engine='python',
            encoding='latin-1',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])

    # Read the Movies File
    movies = pd.read_csv(
        os.path.join(path, "ml-1m", "movies.dat"),
        sep='::',
        engine='python',
        encoding='latin-1',
        names=['movie_id', 'title', 'genres'])

    movie_genres = movies['genres'].str.split('|', n=5, expand=True).fillna('Void').applymap(lambda x: GENRES[x]).copy()
    print(type(movie_genres.values))
    movies['genre_ids'] = pd.DataFrame(movie_genres.values.sum(axis=0))
    # movies['genre_ids'] = movies['genres']
    # n_users = len(ratings['user_id'].unique())
    # n_movies = len(ratings['movie_id'].unique())
    users['gender'] = users['gender'].apply(lambda x: 0 if x == 'F' else 1)
    # Process ratings dataframe for Keras Deep Learning model
    # Add user_emb_id column whose values == user_id - 1
    ratings['user_emb_id'] = ratings['user_id'] - 1

    # Add movie_emb_id column whose values == movie_id - 1
    ratings['movie_emb_id'] = ratings['movie_id'] - 1
    # ratings = ratings.drop(['timestamp'], axis=1).fillna(0)
    """
    movies = pd.read_csv(
            os.path.join('ml-1m', 'movies.dat'),
            sep="::",
            header=None,
            names=['movieId', 'title', 'genres'],
            engine='python')
    """

    print(len(ratings), 'ratings loaded')

    return ratings, users, movies


if __name__ == "__main__":
    pass
