"""
A collection of helper functions, shared among the other scripts.
"""
import numpy as np
import pandas as pd


def raw_line(line):
    """Parse one line of the dataset into a (user, item, rating) triplet of strings."""
    id_, rating = line.strip().split(',')
    user, item = map(lambda x: x[1:], id_.split('_'))
    return user, item, rating


def typed_line(line):
    """Parse one line of the dataset into a typed (user, item, rating) triplet."""
    user, item, rating = raw_line(line)
    return int(user), int(item), float(rating)


def read_lines(path, header=True):
    """Open the dataset file and return its lines (raw)."""
    with open(path, 'r') as f:
        if not header:
            f.readline()  # skip header
        return f.readlines()


def normalize(input_path, output_path):
    with open(input_path, 'r') as f:
        f.readline()  # skip header
        triplets = map(raw_line, f.readlines())
    with open(output_path, 'w+') as f:
        f.write('user,item,rating\n')  # write header
        f.writelines('{u},{i},{r}\n'.format(u=str(int(u)-1), i=str(int(i)-1), r=float(r)) for u, i, r in triplets)


def denormalize(input_path, output_path):
    with open(input_path, 'r') as f:
        f.readline()  # skip header
        triplets = [line.strip().split(',') for line in f.readlines()]
    with open(output_path, 'w+') as f:
        f.write('Id,Prediction\n')
        f.writelines('r{u}_c{i},{r}\n'.format(u=str(int(u)+1), i=str(int(i)+1), r=int(r)) for u, i, r in triplets)


def write_normalized(output_path, data):
    with open(output_path, 'w+') as f:
        f.write('user,item,rating\n')
        f.writelines('{u},{i},{r}\n'.format(u=u, i=i, r=r) for u, i, r in data)


def read_to_df(path):
    """Read the dataset into a pandas DataFrame, one rating triplet per row."""
    return pd.DataFrame.from_records(map(typed_line, read_lines(path, header=False)),
                                     columns=['user', 'item', 'rating'])


def read_to_np(path):
    """Read a normalized dataset into a numpy array. Rows are users, columns are items, values are ratings."""
    data = [(int(user), int(item), float(rating))
            for user, item, rating in map(lambda r: r.split(','), read_lines(path, header=False))]
    shape = max(set(t[0] for t in data))+1, max(set(t[1] for t in data))+1  # get data shape (rows, columns)
    ratings = np.zeros(shape)
    for user, item, rating in data:  # fill array with data
        ratings[user, item] = rating
    return ratings


def tuples_to_df(ts):
    """Convert a collection of (user, item, rating) tuples into a pandas DataFrame."""
    return pd.DataFrame.from_records(ts, columns=['user', 'item', 'rating'])


def split_normalized_data(path, ratio, seed=None):
    """Split the dataset into training/testing parts based on the provided training ratio.

    Return two lists of (user, item, rating) triplets: the training and testing sets, respectively.
    """
    ratings = read_to_np(path)
    np.random.seed(seed)
    nz_users, nz_items = np.nonzero(ratings)  # get indices of non-zero ratings

    def split_user_items(u):
        items, = np.nonzero(ratings[u])
        selected = np.random.choice(items, size=int(len(items) * ratio), replace=False)
        residual = set(items) - set(selected)
        return selected, residual

    training, testing = [], []
    for user in np.unique(nz_users):
        selected, residual = split_user_items(user)
        for item in selected:
            training.append((user, item, ratings[user, item]))
        for item in residual:
            testing.append((user, item, ratings[user, item]))

    key = lambda x: (x[1], x[0])
    return sorted(training, key=key), sorted(testing, key=key)


def clip(n, low=1.0, high=5.0):
    return min(max(n, low), high)
