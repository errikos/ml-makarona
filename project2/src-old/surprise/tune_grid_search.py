from surprise import *
from surprise.model_selection import GridSearchCV
import os


def tune_grid_search(data,
                     algorithm,
                     param_grid,
                     save_file,
                     measures=('rmse', ),
                     cv=2,
                     n_jobs=1,
                     pre_dispatch=2):

    print("Tuning via grid search and " + str(cv) +
          "-fold cross validation...")

    gs = GridSearchCV(
        algorithm,
        param_grid,
        measures=measures,
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch)

    gs.fit(data)

    # Print best parameters and RMSE:
    print("Printing best parameters found...")

    path = "./results/"
    os.makedirs(path, exist_ok=True)

    f = open(path + save_file, "a")
    f.write("Best RMSE:" + str(gs.best_score['rmse']) + "\n")
    f.write("Params:" + str(gs.best_params['rmse']) + "\n")
    f.write("\n")

    f.close()
