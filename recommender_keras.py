import os
import sys
import argparse

from time import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from keras.optimizers import Adam
import keras
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from utils import load_data


# Parser for the arguments
def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-test', help='flag for Testing')

    return parser.parse_args()


def test_model():
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    filename = os.path.join(os.path.dirname(__file__), "test_v2.csv")
    test_csv = pd.read_csv(filename)
    print(test_csv.columns)
    return test_csv


def main(args):

        # Get the currrent directory path
    current_path = os.path.dirname(__file__)
    # Load data
    # ratings = load_data(current_path)

    """
    ratings = pd.read_csv(
            os.path.join(current_path, 'ml-1m', 'ratings.dat'),
            sep="::",
            header=None,
            names=['userId', 'movieId', 'rating', 'timestamp'],
            engine='python')

    ratings = ratings.drop(['timestamp'], axis=1).fillna(0).copy()

    movies = pd.read_csv(
            os.path.join('ml-1m', 'movies.dat'),
            sep="::",
            header=None,
            names=['movieId', 'title', 'genres'],
            engine='python')

    # Extract the various genres into columns
    movie_genres = movies['genres'].str.split('|', n=3, expand=True)

    tags = movies['movieId'].to_frame().merge(
            movie_genres,
            left_on='movieId',
            right_index=True).fillna(0)

    # Change the column names for easy access
    tags.rename(
            columns={
                list(tags)[1]: 'tag_1',
                list(tags)[2]: 'tag_2',
                list(tags)[3]: 'tag_3',
                list(tags)[4]: 'tag_4'}, inplace=True)
    tags = tags.drop(['tag_4'], axis=1)

    # Change data types to categorical
    tags['tag_1'] = tags['tag_1'].replace(tag_ids).astype('int32')
    tags['tag_2'] = tags['tag_2'].replace(tag_ids).astype('int32')
    tags['tag_3'] = tags['tag_3'].replace(tag_ids).astype('int32')
    """
    # Drop genres
    # movies = movies.drop(['genres'], axis=1)

    # data = pd.concat([movies, ratings], axis=1)
    # data = pd.concat([movies, ratings], join_axes=1)
    # Merge dataframes
    # data = pd.merge(movies, ratings, on='movieId')
    ratings, users, movies = load_data(current_path)

    print(ratings.head())
    print(ratings.columns)
    #sys.exit(-1)
    print(users.head())
    print(movies.head())
    """
    ratings_data = pd.concat([ratings['user_emb_id'], ratings['movie_emb_id'], ratings['rating']]).values
    users_data = pd.concat([users['user_emb_id'], ratings['movie_emb_id'], ratings['rating']]).values

    data.userId = data.userId.astype('category').cat.codes.values
    data.movieId = data.movieId.astype('category').cat.codes.values

    """
    Users = ratings['user_emb_id']
    Movies = ratings['movie_emb_id']
    Ratings = ratings['rating']
    # n_tags = len(tags.columns[5:].unique())
    # Split train test
    data = pd.concat([Users, Movies, Ratings], axis=1, keys=['user_emb_id', 'movie_emb_id', 'rating'])
    data['user_emb_id'] = data['user_emb_id'].astype('category').cat.codes.values
    data['movie_emb_id'] = data['movie_emb_id'].astype('category').cat.codes.values
    data['rating'] = data['rating'].astype('category').cat.codes.values

    y = np.zeros((data.rating.shape[0], 5))
    y[np.arange(data.rating.shape[0]), data.rating - 1] = 1

    #X_train, X_test = train_test_split(data, train_size=0.99)
    X_train = data
    # X_test = pd.read_csv(os.path.join(current_path, "test_v2.csv"), header=0)
    n_latent_factors = 32
    print(data.head())
    n_users = len(data['user_emb_id'].unique())
    n_movies = len(data['movie_emb_id'].unique())
    # sys.exit(-1)
    # Init movie Inputs
    user_input = keras.layers.Input(shape=(1,),name='user_input',dtype='int64')
    user_embedding = keras.layers.Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
    user_vec = keras.layers.Dropout(0.5)(user_vec)

    movie_input = keras.layers.Input(shape=(1,),name='movie_input',dtype='int64')
    movie_embedding = keras.layers.Embedding(n_movies, n_latent_factors, name='movie_embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)

    sim = keras.layers.concatenate([user_vec, movie_vec], name='concat', axis=1)

    nn_inp = keras.layers.Dense(128, activation='relu')(sim)
    nn_inp = keras.layers.Dropout(0.5)(nn_inp)
    nn_inp = keras.layers.BatchNormalization()(nn_inp)
    nn_inp = keras.layers.Dense(128, activation='relu')(nn_inp)
    nn_inp = keras.layers.Dropout(0.4)(nn_inp)
    nn_inp = keras.layers.BatchNormalization()(nn_inp)
    # nn_inp = keras.layers.Dense(1,activation='relu')(nn_inp)
    nn_inp = keras.layers.Dense(128, activation='relu')(nn_inp)

    # Try softmax
    nn_inp = keras.layers.Dense(5, activation='softmax')(nn_inp)

    model = keras.models.Model([user_input, movie_input], nn_inp)
    model.summary()

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy')

    if os.path.exists(os.path.join(current_path, 'model.h5')):
        model.load_weights(os.path.join(current_path, 'model.h5'))
        X_test = test_model()
        y_hat = model.predict([X_test.user, X_test.movie])

        # print(MSE(X_test.rating, y_hat))
        # X_tast.rename(columns={'userId': 'ID'})
        results = pd.DataFrame({'rating': y_hat[:, 0]}, dtype=np.float32)
        pd.concat([X_test['ID'], results], axis=1).to_csv('output_predictions.csv', index=False)
        print("Predictions Done")

    # Instantiate Callbacks
    checkPointPath = os.path.join(current_path, "weights", "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5")
    tensorboard = TensorBoard(
            log_dir=os.path.join(current_path, "logs/{}".format(time())))
    checkpoint = ModelCheckpoint(
            checkPointPath,
            monitor='val_loss',
            save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=7, mode='auto')

    # Fit
    # model.fit([X_train.userId, X_train.movieId, X_train[['tag_1', 'tag_2',  'tag_3']]], X_train.rating)
    model.fit(
            [X_train.user_emb_id, X_train.movie_emb_id],
            y,
            callbacks=[tensorboard, checkpoint, earlyStopping],
            validation_split=0.1,
            epochs=20,
            batch_size=128,
            verbose=1)

    # Save model
    model.save(os.path.join(current_path, 'model.h5'))

    # Test model
    # y_hat = np.round(model.predict([X_train.userId, X_train.movieId, X_train[['tag_1', 'tag_2',  'tag_3']]]), 0)
    """
    y_hat = model.predict([X_test.user_emb_id, X_test.movie_emb_id])
    print(MSE(X_test.rating, y_hat))
    """
    # print(MSE(X_test.rating, model.predict([X_train.userId, X_train.movieId, X_train[['tag_1', 'tag_2',  'tag_3']]])))

if __name__ == "__main__":
    args = argument_parser()
    main(args)
