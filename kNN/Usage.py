import pandas as pd
import Preprocessing
import ItemKNN
import UserKNN
import cPickle as pickle
import Evaluation

# rating_df = pd.read_csv('data/ratings.dat')
# rating_df.columns = ['userId', 'movieId', 'rating', 'time']
# rating_df.drop('time', axis=1, inplace=True)
# train_df, test_df = Preprocessing.train_test_split(rate_df=rating_df)
#
# train_df.to_csv('data/train.csv', header=True, index=False)
# test_df.to_csv('data/test.csv', header=True, index=False)

k = 15
at_top_n = 10

# Item kNN
item_knn_obj = ItemKNN.ItemKNN(_k=k , rating_file_path='data_2/u1.base', has_header=False, n_jobs=1)
item_similar = item_knn_obj.compute_similarity_matrix(1, k, False, False, True)
# pickle.dump(item_similar, open('data/item_similar.pkl', 'wb'))

# User kNN
# user_knn = UserKNN.UserKNN(_k=k, rating_file_path='data_2/u1.base', has_header=False, n_jobs=8)
# user_similar = user_knn.compute_similarity_matrix(1, k, False, False, True)

# item_similar = pickle.load(open('data/item_similar.pkl', 'rb'))

# print item_similar[item_knn_obj.movieId_to_idx[920]]

# Rating prediction
# predicted_rating = item_knn_obj.predict(1, 914, item_similar) # actual is 3
# print predicted_rating

# Ranking prediction Item kNN
user_recs = item_knn_obj.recommend_all(top_n=at_top_n, item_similar=item_similar, rec_to_file_name='data_2/recs.pkl')

# Ranking prediction User kNN
# user_recs = user_knn.recommend_all(top_n=at_top_n, user_similar=user_similar, rec_to_file_name='data_2/recs.pkl')

# Evaluation Precision-Recall
train_df = pd.read_csv('data_2/u1.base', header=-1)
train_df.columns = ['userId', 'movieId', 'rating', 'time']
train_df.drop('time', axis=1, inplace=True)

test_df = pd.read_csv('data_2/u1.test', header=-1)
test_df.columns = ['userId', 'movieId', 'rating', 'time']
test_df.drop('time', axis=1, inplace=True)

prec, recall = Evaluation.precision_recall(user_recs, train_df.values.tolist(), test_df.values.tolist(), at_top_n)
print 'Precision = %.4f\nRecall = %.4f' % (prec, recall)
