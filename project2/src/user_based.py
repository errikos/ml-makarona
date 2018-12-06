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
		return int(u1), int(u2), float(sim)

	data = [deal_line(line) for line in read_txt(path)[1:]]

	similarities = np.zeros((dimension, dimension))
	for u1, u2, sim in data:
		similarities[u1, u2] = sim

	return similarities


def calculate_user_similarities(ratings_dense, user_mean_ratings, \
								cache=False, cosine=False, \
								use_intersection=True, \
								min_common_percentage=0):

	def pearson_corr(ratings_dense, user_mean_rating, common_items, \
					 user1, user2):

		numerator = 0
		denominator1 = 0
		denominator2 = 0

		for item in common_items:

			bias_x = ratings_dense[item, user1] - user_mean_ratings[user1]
			bias_y = ratings_dense[item, user2] - user_mean_ratings[user2]
			
			numerator += bias_x * bias_y
			denominator1 += bias_x * bias_x
			denominator2 += bias_y * bias_y

		denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

		return numerator / denominator


	def cosine_sim(ratings_dense, common_items, user1, user2):

		numerator = 0
		denominator1 = 0
		denominator2 = 0

		for item in common_items:

			r_item_user1 = ratings_dense[item, user1]
			r_item_user2 = ratings_dense[item, user2]

			numerator += r_item_user1 * r_item_user2
			denominator1 += r_item_user1 * r_item_user1
			denominator2 += r_item_user2 * r_item_user2

		denominator = math.sqrt(denominator1) * math.sqrt(denominator2)

		return numerator / denominator


	def find_common_items(user1_items, user2_items, use_intersection=True):

		if use_intersection == False:

			zipped = itertools.zip_longest(user1_items, user2_items, \
										   fillvalue=-1)

			for i, j in zipped:
				if i != j:
					return set()

			return set(user1_items)

		return set(user1_items).intersection(set(user2_items))


	def user_similarities(ratings_dense, user_mean_ratings, cosine, \
					  	  use_intersection, writer=None, csvfile=None, \
					  	  min_common_percentage=0):

		num_users = ratings_dense.shape[1]

		user_items = []
		for user in range(num_users):
			user_items.append(np.nonzero(ratings_dense[:, user])[0])

		if writer is None:
			similarities = np.zeros((num_users, num_users))

		# TODO Remove?
		found_neighbour = []
		for user in range(num_users):
			found_neighbour.append(False)

		for user1 in range(num_users):

			print("user1: ", user1)

			for user2 in range(user1 + 1, num_users):

				common_items = find_common_items(user_items[user1], \
												 user_items[user2], \
												 use_intersection)

				min_items = min(len(user_items[user1]), \
								len(user_items[user2]))

				# If the two users don't have common items, or their
				# common items are not at least as many as some threshold,
				# we don't calculate their similarity. 
				if common_items and \
				   len(common_items) >= min_items * min_common_percentage:

					found_neighbour[user1] = True
					found_neighbour[user2] = True

					if cosine:
						corr = cosine_sim(ratings_dense, common_items, \
										  user1, user2)
					else:
						corr = pearson_corr(ratings_dense, user_mean_ratings, \
											common_items, user1, user2)

					if writer is None:	
						similarities[user1, user2] = corr
						similarities[user2, user1] = corr
					else:
						writer.writerow({'User1': user1, 'User2': user2, \
										 'Similarity': corr})
						writer.writerow({'User1': user2, 'User2': user1, \
										 'Similarity': corr})
						# csvfile.flush()

			# TODO Remove?
			if found_neighbour[user1] == False:
				print("User " , user1 , " without any neighbours!")

		if writer is None:
			return similarities

	if cache:

		if use_intersection:
			path = "../data/pearson_sim_intersection_0.3.csv"
		else:
			path = "../data/pearson_sim.csv"

		csvfile = open(path, 'w')
		fieldnames = ['User1', 'User2', 'Similarity']
		writer = csv.DictWriter(csvfile, delimiter=",", \
								fieldnames=fieldnames, lineterminator = '\n')
		writer.writeheader()

		user_similarities(ratings_dense, user_mean_ratings, cosine, \
						  use_intersection, writer, csvfile, \
						  min_common_percentage)
	else:
		return user_similarities(ratings_dense, user_mean_ratings, \
								 cosine, use_intersection, \
								 min_common_percentage)


def create_submission(sub_path, ratings_dense, similarities, \
					  user_mean_ratings):

	''' Loads the sample submission file in order to know which ratings need 
		to be written to csv, and returns the final submission file.
	'''
	def calculate_rating(num_users, ratings_dense, similarities, \
					 	 user_mean_ratings, item, user):

		numerator = 0
		denominator = 0

		for neighbour in range(num_users):
			if neighbour != user and ratings_dense[item, neighbour] != 0:

				bias_y = ratings_dense[item, neighbour] - \
						 user_mean_ratings[neighbour]

				numerator += similarities[user, neighbour] * bias_y
				denominator += abs(similarities[user, neighbour])

		return user_mean_ratings[user] + numerator / denominator


	path_sample_sub = "../data/submission.csv"
	sample_sub_data = load_data(path_sample_sub, sparse_matrix=False)

	# rows, cols = sample_sub_data.nonzero()
	rows, cols = np.nonzero(sample_sub_data)
	zp = list(zip(rows, cols))
	zp.sort(key = lambda tup: tup[1])

	with open(sub_path, 'w') as csvfile:

		fieldnames = ['Id', 'Prediction']
		writer = csv.DictWriter(csvfile, delimiter=",", \
							fieldnames=fieldnames, lineterminator = '\n')
		writer.writeheader()

		num_users = ratings_dense.shape[1]

		counter = 0 #TODO Remove
		for row, col in zp:

			# TODO Remove
			counter += 1
			print(counter)

			r = "r" + str(row + 1)
			c = "c" + str(col + 1)
			val = int(round(calculate_rating(num_users, ratings_dense, \
											 similarities, user_mean_ratings, \
						    				 row, col)))

			if val > 5:
				val = 5
			elif val < 1:
				val = 1

			writer.writerow({'Id': r + "_" + c, 'Prediction': val})
			# csvfile.flush()


if __name__ == "__main__":

	ratings_path = "../data/train.csv"
	ratings_dense = load_data(ratings_path, sparse_matrix=False)
	num_users = ratings_dense.shape[1]

	# Calculate the mean of each user's ratings and store it
	print("Prepping...", end="", flush=True)
	user_mean_ratings = []
	for user in range(num_users):
		user_ratings = ratings_dense[:, user]
		user_mean_ratings.append(np.sum(user_ratings) / \
								 len(np.nonzero(user_ratings)[0]))
	print("Done")

	# Calculate user similarities
	similarities_path = "../data/pearson_sim_intersection_0.3.csv"
	# print("Calculating user similarities...", end="", flush=True)
	# print("Calculating user similarities...")
	# calculate_user_similarities(ratings_dense, user_mean_ratings, \
	# 							cache=True, min_common_percentage=0.3)
	# print("Done")

	print("Extracting cached user similarities...", end="", flush=True)
	similarities = parse_similarities(similarities_path, num_users)
	print("Done")

	submission_path = "../submissions/sub_user_based_inter_0.3.csv"
	# print("Calculating missing ratings...", end="", flush=True)
	print("Calculating missing ratings...")
	create_submission(submission_path, ratings_dense, similarities, \
					  user_mean_ratings)
	print("Done")