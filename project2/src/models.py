"""
Models module.
"""


class Model(object):
    """Base class for a recommendation model."""

    def __init__(self, **kwargs):
        pass

    def fit(self):
        pass


class SparkALS(Model):
    pass


class SurpriseModel(Model):
    """Surprise-based recommendation model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def fit(self):
        pass


class CoClustering(SurpriseModel):
    pass


class ItemBased(SurpriseModel):
    pass


class SVD(SurpriseModel):
    pass


class SVDpp(SurpriseModel):
    pass


class UserBased(SurpriseModel):
    pass


__models__ = [SparkALS, CoClustering, ItemBased, SVD, SVDpp, UserBased]
