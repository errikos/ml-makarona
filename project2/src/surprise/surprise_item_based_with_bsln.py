import numpy as np
import csv
import sys

from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

from helpers import load_data, calculate_rmse

sim_options = {'name': 'pearson_baseline', 'user_based': False}
bsl_options = {'method': 'als', 'n_epochs': 100, 'reg_u': 0.09, \
			   'reg_i': 0.09}

# Load ratings
ratings_path = "./data/train_clean.csv"
# ratings_path = "./data/surprise_item_based_bsln_top50_full_enhanced_clean.csv"
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
ratings = Dataset.load_from_file(ratings_path, reader)

# Test parameters
test_size = 0.1
seed = 50

def tune():

	print("Tuning...")

	# Sample random training set and test set.
	train_ratings, test_ratings = train_test_split(ratings,\
								  				   test_size=test_size, \
								  				   random_state=seed)

	best_rmse = 100
	for K in range(10, 100, 10):

		# Build KNN item based model.
		algorithm = KNNBaseline(k=K, sim_options=sim_options)

		# Train the algorithm on the training set, and predict ratings 
		# for the test set.
		algorithm.fit(train_ratings)
		predictions = algorithm.test(test_ratings)

		# Then compute RMSE
		print("K:", K)
		rmse = accuracy.rmse(predictions)
		if rmse < best_rmse:
			best_rmse = rmse
			best_param = K

	print("Best K:", best_param, " with rmse:", best_rmse)


def test_mine(K=50):

	print("Testing...")

	# Load ratings
	ratings_path = "./data/988_training.csv"
	reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
	ratings = Dataset.load_from_file(ratings_path, reader)

	# Retrieve the trainset.
	train_ratings = ratings.build_full_trainset()

	# Build KNN item based model and train it.
	algorithm = KNNBaseline(k=K, sim_options=sim_options)
	algorithm.fit(train_ratings)

	print("Calculating RMSE...")

	print("RMSE:", calculate_rmse(algorithm, "./data/988_test.csv"))


def test(K=50):

	print("Testing...")

	# Build KNN item based model.
	algorithm = KNNBaseline(k=K, sim_options=sim_options)


	# Sample random training set and test set.
	train_ratings, test_ratings = train_test_split(ratings, \
						 						   test_size=test_size, \
						 						   random_state=seed)

	# Train the algorithm on the training set, and predict ratings 
	# for the test set.
	algorithm.fit(train_ratings)
	predictions = algorithm.test(test_ratings)

	# Then compute RMSE
	accuracy.rmse(predictions)
	

def test_crossval(cv=3, K=50):

	print("Cross validating...")

	# Build KNN item based model.
	algorithm = KNNBaseline(k=K, sim_options=sim_options)

	# Run 2-fold cross-validation and print results
	cross_validate(algorithm, ratings, \
					measures=['RMSE'], cv=cv, verbose=True)


def produce_enhanced(K=50):

	print("Producing enhanced training set...")

	# TODO Perhaps use surprise class Trainset's all_items method 
	# TODO (and outer to inner id conversion)
	# Load original ratings into numpy array
	orig_ratings = load_data("../../data/train.csv", sparse_matrix=False)

	# Retrieve the trainset.
	train_ratings = ratings.build_full_trainset()

	# Build KNN item based model.
	algorithm = KNNBaseline(k=K, sim_options=sim_options)
	algorithm.fit(train_ratings)

	# Store enhanced training set
	submission_path = "./data/surprise_item_based_bsln_top" + \
						str(K) +"_full_enhanced_clean.csv"
	csvfile = open(submission_path, 'w')

	fieldnames = ['User', 'Item', 'Rating']
	writer = csv.DictWriter(csvfile, delimiter=",", \
						fieldnames=fieldnames, lineterminator = '\n')
	writer.writeheader()

	counter = 0
	for row in range(10000):
		for col in range(1000):
			
			uid = str(row)
			iid = str(col)

			# Careful not to overwrite original ratings
			if orig_ratings[row, col] == 0:
				counter += 1
				if counter % 1000 == 0:
					print("Progress: %d/10,000,000" % counter)

				val = int(round(algorithm.predict(uid, iid)[3]))
				if val > 5:
					val = 5
				elif val < 1:
					val = 1
				
				writer.writerow({'User': uid, 'Item': iid, 'Rating': val})
				csvfile.flush()
			else:
				writer.writerow({'User': uid, 'Item': iid, \
								 'Rating': orig_ratings[row, col]})
				csvfile.flush()


def submit(K=50):

	print("Creating submission...")

	# Retrieve the trainset.
	train_ratings = ratings.build_full_trainset()

	# Build KNN item based model and train it.
	algorithm = KNNBaseline(k=K, sim_options=sim_options)
	algorithm.fit(train_ratings)

	# Get submission file format
	print("Producing submission file...")
	sample_submission_path = "../../data/submission.csv"
	test_ratings = load_data(sample_submission_path, sparse_matrix=False)

	rows, cols = np.nonzero(test_ratings)
	zp = list(zip(rows, cols))
	zp.sort(key = lambda tup: tup[1])

	# Create submission file
	submission_path = "./submissions/surprise_item_based_bsln_top" + \
						str(K) +"_full_enhanced.csv"
	csvfile = open(submission_path, 'w')

	fieldnames = ['Id', 'Prediction']
	writer = csv.DictWriter(csvfile, delimiter=",", \
						fieldnames=fieldnames, lineterminator = '\n')
	writer.writeheader()

	counter = 0
	for row, col in zp:

		counter += 1
		if counter % 1000 == 0:
			print("Progress: %d/%d" % (counter, len(rows)))

		uid = str(row)
		iid = str(col)
		val = int(round(algorithm.predict(uid, iid)[3]))

		if val > 5:
			val = 5
		elif val < 1:
			val = 1
		
		r = "r" + str(row + 1)
		c = "c" + str(col + 1)
		writer.writerow({'Id': r + "_" + c, 'Prediction': val})


if __name__ == '__main__':

	if len(sys.argv) == 2:
		if sys.argv[1] == '--tune':
			tune()
		elif sys.argv[1] == '--test':
			test()
		elif sys.argv[1] == '--crossval':
			test_crossval()
		elif sys.argv[1] == '--submit':
			submit()
		else:
			test()
	else:
		test()