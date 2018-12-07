import sys
import numpy as np
import scipy.sparse as sp
import math
import itertools
import csv

from functools import reduce
from helpers import read_txt, load_data, split_data

def parse_similarities(path, dimension):
	'''
		path: Path of the file where similarities are stored
		dimension:  Number of rows and columns the 
					constructed array should have

		returns: numpy array of similarities
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
								similarities_path, cache=False, \
								cosine=False, use_intersection=True, \
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

		if denominator == 0:
			return 1
		else:
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

		# TODO Check for division with 0

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

		print("Min common percentage:", min_common_percentage)

		num_users = ratings_dense.shape[1]

		user_items = []
		for user in range(num_users):
			user_items.append(np.nonzero(ratings_dense[:, user])[0])

		similarities = np.zeros((num_users, num_users))

		for user1 in range(num_users):

			if user1 % 50 == 0:
				print("Progress: %d/1000" % user1)

			for user2 in range(user1 + 1, num_users):

				common_items = find_common_items(user_items[user1], \
												 user_items[user2], \
												 use_intersection)

				min_items = min(len(user_items[user1]), \
								len(user_items[user2]))

				if common_items:

					# If a user's neighbour has fewer common items than
					# some threshold, we reduce his significance. 
					n = len(common_items)
					threshold = min_items * min_common_percentage

					if  n < threshold:
						significance = n / threshold
					else:
						significance = 1

					if cosine:
						corr = cosine_sim(ratings_dense, common_items, \
										  user1, user2)
					else:
						corr = pearson_corr(ratings_dense, user_mean_ratings, \
											common_items, user1, user2)

					corr *= significance
	
					similarities[user1, user2] = corr
					similarities[user2, user1] = corr
					if writer is not None:
						writer.writerow({'User1': user1, 'User2': user2, \
										 'Similarity': corr})
						writer.writerow({'User1': user2, 'User2': user1, \
										 'Similarity': corr})

		return similarities

	writer = None
	csvfile = None
	if cache:

		csvfile = open(similarities_path, 'w')
		fieldnames = ['User1', 'User2', 'Similarity']
		writer = csv.DictWriter(csvfile, delimiter=",", \
								fieldnames=fieldnames, lineterminator = '\n')
		writer.writeheader()

	return user_similarities(ratings_dense, user_mean_ratings, cosine, \
							 use_intersection, writer, csvfile, \
					  		 min_common_percentage)


def predict_ratings(train_ratings, test_ratings, \
					similarities, user_mean_ratings, create_submission=False, \
					submission_path=None):

	''' Loads the sample submission file in order to know which ratings need 
		to be written to csv, and returns the final submission file.
	'''
	def calculate_rating(num_users, ratings, similarities, \
					 	 user_mean_ratings, item, user):

		# Determine N top neighbours for the given user
		N = 50  #TODO Pass as argument
		have_rated_item = np.nonzero(ratings[item, :])[0]

		top_N_similarities = []
		for neighbour in have_rated_item:

			top_N_similarities.append((neighbour, \
									   abs(similarities[user, neighbour])))


		top_N_similarities.sort(key = lambda tup: tup[1], reverse=True)
		top_N_similarities = top_N_similarities[:N]
		
		# Calculate rating
		numerator = 0
		denominator = 0
		for neighbour, sim in top_N_similarities:

			bias_y = ratings[item, neighbour] - user_mean_ratings[neighbour]
			numerator += similarities[user, neighbour] * bias_y
			denominator += sim

		return user_mean_ratings[user] + numerator / denominator


	rows, cols = np.nonzero(test_ratings)
	zp = list(zip(rows, cols))
	zp.sort(key = lambda tup: tup[1])

	if create_submission:
		csvfile = open(submission_path, 'w')

		fieldnames = ['Id', 'Prediction']
		writer = csv.DictWriter(csvfile, delimiter=",", \
							fieldnames=fieldnames, lineterminator = '\n')
		writer.writeheader()
	else:
		predicted_ratings = np.zeros((test_ratings.shape))

	num_users = train_ratings.shape[1]

	counter = 0
	for row, col in zp:

		counter += 1
		if counter % 1000 == 0:
			print("Progress: %d/%d" % (counter, len(rows)))

		val = int(round(calculate_rating(num_users, train_ratings, \
										 similarities, user_mean_ratings, \
					    				 row, col)))
		if val > 5:
			val = 5
		elif val < 1:
			val = 1

		if create_submission:
			r = "r" + str(row + 1)
			c = "c" + str(col + 1)
			writer.writerow({'Id': r + "_" + c, 'Prediction': val})
		else:
			predicted_ratings[row, col] = val

	if create_submission:
		return None
	else:
		return predicted_ratings


def rmse(predicted_ratings, true_ratings):
	""" 
		Compute the loss (RMSE) of the prediction of nonzero elements.
	"""
	rows, cols = np.nonzero(true_ratings)
	nz = list(zip(rows, cols))

	mse = 0
	for row, col in nz:
		mse += (true_ratings[row, col] - predicted_ratings[row, col]) ** 2

	return math.sqrt(1.0 * mse / len(rows))


if __name__ == "__main__":

	similarities_path = "../data/pearson_sim_test_mode_intersection.csv"
	# submission_path = "../submissions/sub_user_based_inter_0.1.csv"
	use_cached_sim = True  #TODO Handle as sys argument
	cache = False
	test_mode = False
	min_common_percentage = 0

	if len(sys.argv) == 3:
		if sys.argv[1] == "--test" or sys.argv[2] == "--test":
			print("Running on test mode.")
			test_mode = True
			submission_path = ""
		if sys.argv[1] == "--cache" or sys.argv[2] == "--cache":
			cache = True
		
	if len(sys.argv) == 2:
		if sys.argv[1] == "--test":
			print("Running on test mode.")
			test_mode = True
			submission_path = ""
		elif sys.argv[1] == "--cache":
			cache = True

	# Load ratings
	ratings_path = "../data/train.csv"
	ratings = load_data(ratings_path, sparse_matrix=False)

	# Split ratings into a training and a test set, which, if we are not
	# in test mode, is the set of real missing ratings that need to be
	# submitted.
	if test_mode:

		train_ratings, test_ratings = split_data(ratings)
	else:
		train_ratings = ratings
		
	num_users = train_ratings.shape[1]

	# Calculate the mean of each user's ratings and store it
	print("Prepping...", end="", flush=True)
	user_mean_ratings = []
	for user in range(num_users):
		user_ratings = train_ratings[:, user]
		user_mean_ratings.append(np.sum(user_ratings) / \
								 len(np.nonzero(user_ratings)[0]))
	print("Done")

	if use_cached_sim:
		# Extract cached user similarities
		print("Extracting cached user similarities...", end="", flush=True)
		similarities = parse_similarities(similarities_path, num_users)
		print("Done")
	else:
		# Calculate user similarities
		print("Calculating user similarities...")
		similarities = calculate_user_similarities(train_ratings, \
							user_mean_ratings, similarities_path, \
							cache=cache, \
							min_common_percentage=min_common_percentage)

		print("Done")

	# TODO Remove! Get only top neighbours for each user
	# top_similarities = np.zeros(similarities.shape)
	# for user in range(similarities.shape[0]):
	# 	nnz_indices = list(np.nonzero(similarities[user, :])[0])
	# 	nnz_similarities = list(similarities[user, nnz_indices])

	# 	zipped = list(zip(nnz_indices, nnz_similarities))
	# 	zipped.sort(key = lambda tup: abs(tup[1]), reverse=True)

	# 	top_neighbours = zipped[:500]
	# 	for index, sim in top_neighbours:
	# 		top_similarities[user, index] = sim
	# similarities = top_similarities

	# Predict ratings
	if test_mode:
		print("Calculating missing ratings...")
		predicted_ratings = predict_ratings(train_ratings, test_ratings, \
											similarities, user_mean_ratings)
		print("Done")

		# Check test error
		print("Error: ", rmse(predicted_ratings, test_ratings))
		
	else:
		print("Producing submission file...")
		sample_submission_path = "../data/submission.csv"
		test_ratings = load_data(sample_submission_path, sparse_matrix=False)

		predict_ratings(train_ratings, test_ratings, \
						similarities, user_mean_ratings, \
						create_submission=True, \
						submission_path=submission_path)

		print("Done")

