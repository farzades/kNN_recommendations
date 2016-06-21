import pandas as pd
import numpy as np
import math

def precision_recall(recs, train, test, at_top_n):
    """
    :param recs: recommendations as dictionary with users as key and item list as value
    :param train: list of lists, each inner list should contain [user_ID, item_ID, rating]
    :param test: list of lists, each inner list should contain [user_ID, item_ID, rating]
    :param at_top_n: Precision@top_n and Recall@top_n
    :return: precision, recall
    """
    tdf = pd.DataFrame(test, columns=['userId', 'movieId', 'rating'])

    train_test_df = pd.DataFrame(train+test, columns=['userId', 'movieId', 'rating'])
    precision_sum = 0
    recall_sum = 0
    hit_sum = 0
    test_count = 0
    for user in recs.keys():
        # this is for positive rates only, for now we ignore this
        # test_items = tdf[(tdf['userId'] == user) & (tdf['rating'] >= 2.5)]['movieId'].values.tolist()

        # test_items = tdf[tdf['userId'] == user]['movieId'].values.tolist()
        test_items = tdf[tdf['userId'] == user]
        avg_ratins = train_test_df[train_test_df['userId'] == user]['rating'].mean()
        test_items = test_items[test_items['rating'] >= 0]['movieId'].values.tolist()

        # if all the test cases are negative rated items then skip over this negative bastard.
        if len(test_items) == 0:
            # print 'Negative user !!! ', user
            continue

        test_count += 1

        hit_count = len(set.intersection(set(test_items), set(recs[user])))
        # print '-' * 50
        # print 'user', user
        # print 'hit count', hit_count
        # print test_items
        # print recs[user]
        # if hit_count > 0:
        #     print '#'*200

        hit_sum += hit_count
        precision = float(hit_count) / at_top_n
        recall = float(hit_count) / len(test_items)

        precision_sum += precision
        recall_sum += recall

    # print 'average hits for all users = %f' % (hit_sum / float(len(recs.keys())))
    # print 'average hits for all users = %f' % (hit_sum / float(num_users_test))
    # print 'test_count', test_count
    precision_total = precision_sum / float(test_count)
    recall_total = recall_sum / float(test_count)
    # print 'precision %.4f' % precision_total
    # print 'recall %.4f' % recall_total

    return precision_total, recall_total


def rmse(preds, actuals):
    """
    Root Mean Square Error
    preds and actuals should be paired that means the index of preds and actuals are the same for each (u,i)
    :param preds: predicted ratings
    :param actuals: actual ratings
    :return: rmse
    """
    print ("Computing Root Mean Squared Error...")

    if len(preds) != len(actuals):
        raise Exception('The Length of predictions and actuals are not the same !!')

    errors = []
    for i in range(len(preds)):
        errors.append((preds[i] - actuals[i])**2)

    rmse_val = math.sqrt(np.mean(errors))
    return rmse_val


def mae(preds, actuals):
    """
    Mean Absolute Error
    preds and actuals should be paired that means the index of preds and actuals are the same for each (u,i)
    :param preds: predicted ratings
    :param actuals: actual ratings
    :return:
    """
    print ("Computing Mean Absolute Error...")

    if len(preds) != len(actuals):
        raise Exception('The Length of predictions and actuals are not the same !!')

    errors = []
    for i in range(len(preds)):
        errors.append(math.fabs(preds[i] - actuals[i]))

    mae_val = np.mean(errors)
    return mae_val


def coverage_error(preds, actuals):
    pass
    # coverage = (float(n) - float(self.cov_error)) / float(n)
