def tune_crossval(cv=4):

    print("Tuning via cross validation...")

    # Build KNN item based model.
    algorithm = KNNBaseline(k=K, sim_options=sim_options)


    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(ratings, \
                                  # train_size=0.02, test_size=0.02)
                                  test_size=0.2)

    # Train the algorithm on the training set, and predict ratings 
    # for the test set.
    algorithm.fit(train_ratings)
    predictions = algorithm.test(test_ratings)

    # Then compute RMSE
    accuracy.rmse(predictions)