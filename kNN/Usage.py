import pandas as pd
import Preprocessing
import ItemKNN
import UserKNN
import cPickle as pickle
import Evaluation
import recommender

# rating_df = pd.read_csv('data_2/ratings.dat')
# rating_df.columns = ['userId', 'movieId', 'rating', 'time']
# rating_df.drop('time', axis=1, inplace=True)
# train_df, test_df = Preprocessing.train_test_split(rate_df=rating_df)
#
# train_df.to_csv('data_2/train.csv', header=True, index=False)
# test_df.to_csv('data_2/test.csv', header=True, index=False)

# -----------------------------------------------------------------------------------------------
# Parameters
k = 15
at_top_n = 10

# MovieLens Datasets
dataset = '100K'
# dataset = '1M'

# -----------------------------------------------------------------------------------------------

# User kNN
if dataset == '100K':
    user_knn_model = UserKNN.UserKNN(_k=k, rating_file_path='data_2/u1.base', has_header=False, n_jobs=8)
elif dataset == '1M':
    user_knn_model = UserKNN.UserKNN(_k=k, rating_file_path='data_2/train.csv', has_header=True, n_jobs=8)
user_knn_model.compute_similarity_matrix(1, k, False, False, True)

# predicted_rating = user_knn_model.predict(1, 33)

# Ranking prediction User kNN
user_recs = user_knn_model.recommend_all(top_n=at_top_n, rec_to_file_name='data_2/recs.pkl')

# -----------------------------------------------------------------------------------------------

# Item kNN
# if dataset == '100K':
#     item_knn_model = ItemKNN.ItemKNN(_k=k, rating_file_path='data_2/u1.base', has_header=False, n_jobs=8)
# elif dataset == '1M':
#     item_knn_model = ItemKNN.ItemKNN(_k=k, rating_file_path='data_2/train.csv', has_header=True, n_jobs=8)
# item_knn_model.compute_similarity_matrix(1, k, False, False, True)

# pickle.dump(item_similar, open('data/item_similar.pkl', 'wb'))
# item_similar = pickle.load(open('data/item_similar.pkl', 'rb'))

# print item_similar[item_knn_model.movieId_to_idx[920]]

# Rating prediction
# predicted_rating = item_knn_model.predict(1, 33, item_similar)

# Ranking prediction Item kNN
# user_recs = item_knn_model.recommend_all(top_n=at_top_n, rec_to_file_name='data_2/recs.pkl')

# -----------------------------------------------------------------------------------------------


# print predicted_rating


# Evaluation
print 'Evaluation...'
if dataset == '100K':
    train_df = pd.read_csv('data_2/u1.base', header=-1)
    train_df.columns = ['userId', 'movieId', 'rating', 'time']
    train_df.drop('time', axis=1, inplace=True)

    test_df = pd.read_csv('data_2/u1.test', header=-1)
    test_df.columns = ['userId', 'movieId', 'rating', 'time']
    test_df.drop('time', axis=1, inplace=True)
elif dataset == '1M':
    train_df = pd.read_csv('data_2/train.csv', header=0)
    train_df.columns = ['userId', 'movieId', 'rating']

    test_df = pd.read_csv('data_2/test.csv', header=0)
    test_df.columns = ['userId', 'movieId', 'rating']

# Ranking (Precision-Recall)
prec, recall = Evaluation.precision_recall(user_recs, train_df.values.tolist(), test_df.values.tolist(), at_top_n)
print 'Precision = %.4f\nRecall = %.4f' % (prec, recall)


# Rating Prediction
preds, actuals = recommender.predict_test(user_knn_model, test_df.values.tolist())
rmse = Evaluation.rmse(preds, actuals)
print 'User kNN rmse', rmse

# preds, actuals = recommender.predict_test(item_knn_model, test_df.values.tolist())
# rmse = Evaluation.rmse(preds, actuals)
# print 'Item kNN rmse', rmse

import numpy as np
print 'mean predictions', np.mean(preds)
print 'mean actual', np.mean(actuals)





