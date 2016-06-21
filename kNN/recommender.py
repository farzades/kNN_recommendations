from joblib import Parallel, delayed

def predict_test(model, test):
    """
    Predict all the ratings based on (user,item,rating) in test.
    :param model: model for prediction
    :param test: list of lists for each triplet (user,item,rating)
    :return: predictions, actual_ratings
    """

    # movie_recs_par = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0, max_nbytes="100M") \
    #     (delayed(recommend_one)(movie_index, top_n, self.item_similars,
    #                             self.user_num,
    #                             self.item_user_matrix,
    #                             self.k)
    #      for movie_index in range(self.movies_num))
    #

    preds = []
    actuals = []
    for [user_id, item_id, r] in test:
        preds.append(model.predict(user_id, item_id))
        actuals.append(r)

    return preds ,actuals
