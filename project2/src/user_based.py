import numpy as np
import scipy.sparse as sp
import math
import itertools
import csv

from functools import reduce
from helpers import read_txt, load_data, preprocess_data

def cache_similarities(path, sim_dense):

	with open(path, 'w') as csvfile:
		fieldnames = ['User1', 'User2', 'Similarity']
		writer = csv.DictWriter(csvfile, delimiter=",", \
								fieldnames=fieldnames, lineterminator = '\n')
		writer.writeheader()
		for row in range(sim_dense.shape[0]):
			for col in range(sim_dense.shape[1]):
				u1 = str(row)
				u2 = str(col)
				sim = str(sim_dense[row, col])
				
				writer.writerow({'User1': u1, 'User2': u2, 'Similarity': sim})


def parse_similarities(path, dimension):
	'''
		path: Path of the file where similarities are stored
		dimension:  Number of rows and columns the 
					constructed lil matrix should have

		returns: lil matrix of similarities
	'''

	def deal_line(line):
		u1, u2, sim = line.split(',')
		return int(u1), int(u2), double(sim)

	data = [deal_line(line) for line in read_txt(path)]

	similarities = sp.lil_matrix((dimension, dimension))
	for u1, u2, sim in data:
		similarities[u1, u2] = sim

	return similarities


def calculate_user_bias(r_user, item):

	mean_user_ratings = r_user.sum() / r_user.nonzero()[0].shape[0]
	user_bias = r_user[item, 0] - mean_user_ratings

	return user_bias


def pearson_corr(r_x, r_y, common_items, mean_x_ratings, mean_y_ratings):

	numerator = 0
	denominator1 = 0
	denominator2 = 0

	for i in common_items:
		bias_x = r_x[i, 0] - mean_x_ratings
		bias_y = r_y[i, 0] - mean_y_ratings
		
		numerator += bias_x * bias_y
		denominator1 += bias_x * bias_x
		denominator2 += bias_y * bias_y

	denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

	return numerator / denominator


def cosine_sim(r_x, r_y, common_items):

	numerator = 0
	denominator1 = 0
	denominator2 = 0

	for i in common_items:
		numerator += r_x[i, 0] * r_y[i, 0]
		denominator1 += r_x[i, 0] * r_x[i, 0]
		denominator2 += r_y[i, 0] * r_y[i, 0]

	denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

	return numerator / denominator


def find_common_items(user1_items, user2_items, use_intersection=True):

	if use_intersection == False:

		zipped = itertools.zip_longest(user1_items, user2_items, fillvalue=-1)

		for i, j in zipped:
			if i != j:
				return set()

		return set(user1_items)

	return set(user1_items).intersection(set(user2_items))


# TODO Remove? 
# def calculate_user_similarities(ratings, calculate_all=False, \
# 								cosine=False, cache=False):

# 	user_items = []
# 	for user in range(ratings.shape[1]):
# 		user_items.append(ratings[:, user].nonzero()[0])

# 	sim = sp.lil_matrix((ratings.shape[1], ratings.shape[1]))
# 	for user1 in range(ratings.shape[1]):

# 		print("user1: ", user1)

# 		r_x = ratings[:, user1]
# 		mean_x_ratings = r_x.sum() / r_x.nonzero()[0].shape[0]

# 		for user2 in range(user1 + 1, ratings.shape[1]):

# 			r_y = ratings[:, user2]
# 			mean_y_ratings = r_y.sum() / r_y.nonzero()[0].shape[0]

# 			common_items = find_common_items(user_items[user1], \
# 											user_items[user2], calculate_all)

# 			if common_items != set():

# 				if cosine:
# 					corr = cosine_sim(r_x, r_y, common_items)
# 				else:
# 					corr = pearson_corr(r_x, r_y, common_items, \
# 										mean_x_ratings, mean_y_ratings)
# 				sim[user1, user2] = corr
# 				sim[user2, user1] = corr

# 	return sim


def calculate_user_similarities(ratings, cache=False, cosine=False, \
								use_intersection=True):

	def user_similarities(ratings, cosine, use_intersection, \
						  writer=None, csvfile=None):

		user_items = []
		for user in range(ratings.shape[1]):
			user_items.append(ratings[:, user].nonzero()[0])

		if writer is None:
			sim = sp.lil_matrix((ratings.shape[1], ratings.shape[1]))

		for user1 in range(ratings.shape[1]):

			print("user1: ", user1)

			r_x = ratings[:, user1]
			mean_x_ratings = r_x.sum() / r_x.nonzero()[0].shape[0]

			for user2 in range(user1 + 1, ratings.shape[1]):

				r_y = ratings[:, user2]
				mean_y_ratings = r_y.sum() / r_y.nonzero()[0].shape[0]

				common_items = find_common_items(user_items[user1], \
												 user_items[user2], \
												 use_intersection)

				if common_items != set():

					if cosine:
						corr = cosine_sim(r_x, r_y, common_items)
					else:
						corr = pearson_corr(r_x, r_y, common_items, \
											mean_x_ratings, mean_y_ratings)

					if writer is None:	
						sim[user1, user2] = corr
						sim[user2, user1] = corr
					else:
						writer.writerow({'User1': user1, 'User2': user2, \
										 'Similarity': corr})
						writer.writerow({'User1': user2, 'User2': user1, \
										 'Similarity': corr})
						csvfile.flush()

		if writer is None:
			return sim

	if cache:

		if use_intersection:
			path = "../data/pearson_sim_intersection.csv"
		else:
			path = "../data/pearson_sim.csv"

		csvfile = open(path, 'w')
		fieldnames = ['User1', 'User2', 'Similarity']
		writer = csv.DictWriter(csvfile, delimiter=",", \
								fieldnames=fieldnames, lineterminator = '\n')
		writer.writeheader()

		user_similarities(ratings, cosine, use_intersection, writer, csvfile)
	else:
		return user_similarities(ratings, cosine, use_intersection)


def calculate_rating(ratings, sim, item, user):

	numerator = 0
	denominator = 0

	for neighbour in ratings.shape[1]:
		if ratings[item, neighbour] != 0:

			bias_y = calculate_user_bias(ratings[:, neighbour], item)
			numerator += sim[user, neighbour] * bias_y
			denominator += abs(sim[user, neighbour])


	r_x = ratings[:, user]
	mean_x_ratings = r_x.sum() / r_x.nonzero()[0].shape[0]

	return mean_x_ratings + numerator / denominator

if __name__ == "__main__":
	path_dataset = "../data/train.csv"
	ratings = load_data(path_dataset)

	# print("Calculating user similarities...", end="", flush=True)
	print("Calculating user similarities...")
	calculate_user_similarities(ratings, cache=True)
	print("Done")

	# print("Caching similarities in file pearson_similarities.csv...",\
	# 		 end="", flush=True)
	# cache_similarities("../data/pearson_similarities.csv", sim.todense())
	# print("Done")

	# print("Calculating missing ratings...", end="", flush=True)
	# print(calculate_rating(ratings, sim, 36, 0))
	# print("Done")