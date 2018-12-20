"""Simple item-mean model."""
import itertools


class ItemMean(object):
    """Class modelling the item-mean recommendation model."""
    def __init__(self):
        self.item_means = {}

    def update(self, item, mean):
        self.item_means[item] = mean

    def get(self, item):
        return self.item_means.get(item)


def train(training_data):
    data = sorted(map(lambda row: (row[1], row[2]), training_data), key=lambda t: t[0])
    model = ItemMean()
    for item, ratings in itertools.groupby(data, key=lambda t: t[0]):
        ratings = list(map(lambda t: t[1], ratings))
        model.update(item, sum(ratings) / len(ratings))
    return model


def predict(model, prediction_set):
    pairs = filter(lambda row: (row[0], row[1]), prediction_set)
    return map(lambda p: (p[0], p[1], model.get(p[1])), pairs)
