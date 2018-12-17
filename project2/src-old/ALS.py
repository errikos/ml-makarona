import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt


from helpers import load_data, preprocess_data
from helpers import build_index_groups, create_submission
from helpers import calculate_mse
from helpers import split_data
from plots import plot_raw_data



path_dataset = "../data/train.csv"
ratings = load_data(path_dataset, sparse_matrix=False)
# print(ratings[:10,:10].todense())
# vait = [1,2,4,7,8]
# vaus = [2,4,6,8,9]
# #rat = ratings[vait][:,vaus]
# rat = ratings[vait]
# print(rat.todense()[:10,:10])
# rat = rat[:,vaus]
# print(rat.todense()[:10,:10])
# print(ratings[vait,vaus].shape)
# print(ratings[vait,vaus].todense())


# ----------------------------------------------------------------------------------------




num_items_per_user, num_users_per_item = plot_raw_data(ratings)

print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))


# ----------------------------------------------------------------------------------------


train, test = split_data(ratings, p_test=0.1)


# ----------------------------------------------------------------------------------------

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
        
    num_item, num_user = train.shape
    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    # return a vector of how many movies each user has rated
    item_nnz = np.count_nonzero(train, axis=1)
    item_sum = train.sum(axis=1)
    rint(len(item_nnz), len(item_sum))
    for ind in range(num_item):
        # each column of item_features is the mean. why for the first line only
        # it's like it says: for k=0 we have an initialization
        item_features[0, ind] = item_sum[ind] / item_nnz[ind]
    return user_features, item_features

# ----------------------------------------------------------------------------------------

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - round(user_info.T.dot(item_info))) ** 2
    return np.sqrt(1.0 * mse / len(nz))


# ----------------------------------------------------------------------------------------


def matrix_factorization_SGD(train, test):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 10   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    num_epochs = 20     # number of full passes through the train set
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

#matrix_factorization_SGD(train, test) 

# ----------------------------------------------------------------------------------------

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix: Z.
    how it works: For each user  go and update this user's predictions (Z)"""
    
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))
    # Z
    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        # "n"xK
        M = item_features[:, items]
        #print(M.shape)
        
        # update column row of user features
        # Kx1
        V = M @ train[items, user]
        #print(V.shape)
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        #print("AAAA" + str(X.shape))
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


# ----------------------------------------------------------------------------------------




def ALS(train, test, K, lambda_u, lambda_i):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = K   # K in the lecture notes
    lambda_user = lambda_u #
    lambda_item = lambda_i
    stop_criterion = 1e-5
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user,nnz_users_per_item=np.count_nonzero(train,axis=0),np.count_nonzero(train, axis=1)
    
    # group the indices by row or column index
    #print(train.shape)
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    #print(len(nz_train))
    #print(nz_item_userindices[:10])
    #print(nz_user_itemindices[:10])

    # run ALS
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)
    print("test RMSE after running ALS({K} {lamu} {lami}): {v}.".format(K=K, 
                                                        lamu=lambda_u, lami=lambda_i,
                                                        v=rmse))

    return rmse, user_features, item_features
    # Create Submission
    


# kappas = range(9,12,1)
# lambda_u = range(90,91,1)
# lambda_i = range(90,91,1)
# # kappas = range(5,6,1)
# # lambda_u = range(1,2,1)
# # lambda_i = range(1,2,1)
# min_err = 100
# best_config = (0,0,0)
# print("\nTuning lambdas")
# # tune lambdas (separately from K. not the best but faster)
# for u,i in zip(lambda_u, lambda_i):
#   test_err, W, Z = ALS(train, test, 10, u/1000, i/1000)    
#   if test_err < min_err:
#       min_err = test_err
#       best_config = (10, u, i)

# # tune K
# print("\nTuning K")
# min_err = 100
# best_W_Z = (None, None)
# for k in kappas:
#   test_err, W, Z = ALS(train, test, k, best_config[1]/1000, best_config[2]/1000)
#   if test_err < min_err:
#       min_err = test_err
#       best_config = (k, best_config[1], best_config[2])
#       best_W_Z = (W,Z)

kappas = range(8,13,1)
lambda_u = range(80,105,1)
lambda_i = range(80,105,1)

min_err = 10000
best_config = (10,90,90)
print("\nTuning lambda_u")
# tune lamda_u (separately from K. not the best but faster)
for u in lambda_u:
    test_err, W, Z = ALS(train, test, 10, u/1000, best_config[2]/1000)    
    if test_err < min_err:
        min_err = test_err
        best_config = (best_config[0], u, best_config[2])

print("\nTuning lambda_i")
for i in lambda_i:
    test_err, W, Z = ALS(train, test, 10, best_config[1]/1000, i/1000)    
    if test_err < min_err:
        min_err = test_err
        best_config = (best_config[0], best_config[1], i)

# tune K
print("\nTuning K")
min_err = 10000
best_W_Z = (None, None)
for k in kappas:
    test_err, W, Z = ALS(train, test, k, best_config[1]/1000, best_config[2]/1000)
    if test_err < min_err:
        min_err = test_err
        best_config = (k, best_config[1], best_config[2])
        best_W_Z = (W,Z)
print("Best parameters are: K={k}, lambda_u={u}, lambda_i={i}. Test error={er}".format(
                                k=best_config[0], u=best_config[1], i=best_config[2],
                                er=min_err))

X = best_W_Z[1].transpose().dot(best_W_Z[0])
create_submission("../data/ULTI.csv", X)