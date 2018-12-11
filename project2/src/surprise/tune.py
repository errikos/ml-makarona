from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from helpers import load_data, calculate_rmse

# NOTE: use GridSearchCV for a genrealized method (?)
def tune_K_crossval(sim_options, ratings, cv=5):
    #Evaluating RMSE of algorithm KNNBaseline on 5 split(s).
#
#                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
#RMSE (testset)    0.9877  0.9919  0.9922  0.9905  0.9898  0.9904  0.0016
#Fit time          21.59   21.99   21.99   22.00   22.08   21.93   0.17
#Test time         54.72   54.65   54.47   54.44   54.55   54.57   0.11
#{'test_rmse': array([ 0.98768632,  0.99190453,  0.99217937,  0.99052025,  0.98983562]), 'fit_time': (21.588273286819458, 21.987202644348145, 21.988197088241577, 21.996177434921265, 22.079952001571655), 'test_time': (54.7196900844574, 54.649856090545654, 54.47136926651001, 54.44243550300598, 54.546159505844116)}
    print("Tuning via cross validation...")

    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(ratings,\
                                                   test_size=0.1, \
                                                   random_state=50)

    best_rmse = 100
    for K in range(40, 130, 10):

        # Build KNN item based model.
        algorithm = KNNBaseline(k=K, sim_options=sim_options)
        res = cross_validate(algorithm, ratings, \
                    measures=['RMSE'], cv=cv, verbose=True)
        print("K:", K, "  ", res)
        # Train the algorithm on the training set, and predict ratings 
        # for the test set.
        # algorithm.fit(train_ratings)
        # predictions = algorithm.test(test_ratings)

        # # Then compute RMSE
        # print("K:", K)
        # rmse = accuracy.rmse(predictions)
        # if rmse < best_rmse:
        #     best_rmse = rmse
        #     best_param = K

    # print("Best K:", best_param, " with rmse:", best_rmse)