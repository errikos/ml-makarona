"""Simple user-mean model."""
import itertools


class UserMean(object):
    """Class modelling the user-mean recommendation model."""
    def __init__(self):
        self.user_means = {}

    def update(self, user, mean):
        self.user_means[user] = mean

    def get(self, user):
        return self.user_means.get(user)


def train(training_data):
    data = sorted(map(lambda row: (row[0], row[2]), training_data), key=lambda t: t[0])
    model = UserMean()
    for user, ratings in itertools.groupby(data, key=lambda t: t[0]):
        ratings = list(map(lambda t: t[1], ratings))
        model.update(user, sum(ratings) / len(ratings))
    return model


def predict(model, prediction_set):
    pairs = filter(lambda row: (row[0], row[1]), prediction_set)
    return map(lambda p: (p[0], p[1], model.get(p[0])), pairs)
