# EPFL Machine Learning 2018-2019
## Project 2: Recommender System


### TLDR;

In order to reproduce the CrowdAI submission, download the intermediate files from the Google Drive
[link](https://drive.google.com/drive/folders/1xK0RSuqCuR9OmThDE2pSX0BdfFxMY094),
put them in the `src` directory and run:

    python3 run.py --testing=intermediate-data/testing.csv \
                    --testing-predictions=intermediate-data/predictions_testing \
                    --submission-predictions=intermediate-data/predictions_submission \
                    --output=final_submission.csv


The CrowdAI submission ID is 25307 from user "ergys".                    


### Introduction

The project consists of several standalone Python scripts, each of which can be run interactively
from the command line, as well as invoked via the [`click`](https://click.palletsprojects.com)
context (please see `all.py` for an example).

The name of a script indicates it's functionality. More precisely:
  * `tools.py`: contains various tools for dataset manipulation.
  * `tune.py`: contains commands to tune the hyper-parameters of each model.
  * `train.py`: contains commands to train each mdoel.
  * `blend.py`: blends the models together, based on their predictions.
  * `all.py`: executes the whole pipeline (training on the 90% training and predicting for the
    10% testing, training on the 100% training and predicting for the submission and, finally,
    blending the results based on the testing dataset RMSE) - this script takes a LOT of time
    to run!
  
In order to obtain more information on the commands and the expected parameters of each command,
please invoke them by passing the `--help` option (either directly to a script or to a command).

There is also a bash script (`all.bash`) which executes exactly the same commands as `all.py`. 


### Setting up the Anaconda environment

For your convenience, we include an Anaconda environment definition file, from which you can
create an Anaconda environment containing all the dependencies for our code.

Please note that pyspark (about 200MB) is a dependency, since we use the ALS implementation from
the Spark MLlib. That being said, since Spark is not yet compatible with Java versions larger than 1.8,
you need to have Java version 1.8 in your path, or set the `JAVA_HOME` env. variable to a JRE 1.8
installation.

You can create the environment by running:

    conda env create -f epfml-ml-makarona.yml
    
and then activate it with:

    source activate epfml-ml-makarona  # Linux/macOS
    activate epfml-ml-makarona         # Windows
    
For more information, please see: https://conda.io/docs/user-guide/tasks/manage-environments.html


### Data files

While all scripts accept the various input and out file paths as parameters, the `all.py` and
`all.bash` scripts expect the `train.csv` and `submission.csv` datasets to be under `data`.

However, `all.py` takes a long time to run, so you will probably want to run `run.py` with the already
produced (intermediate) files from the trained models. The format of the intermediate files is slightly
different from the given dataset files. The differences are:
  * schema: "normalized" to (user, item, rating).
  * indexing: user and item IDs are 0-indexed instead of 1-indexed.
  
We name the intermediate files "normalized", meaning that their format is normalized as described above.
In order to easily convert to/from normalized file format, one can use the `tools.py` script.

### How to run

In order to reproduce the results in CrowID, you have two options:
  * Run the whole pipeline via `all.py` or `all.bash`, which is very slow
    (it may take more that 1 day),
    as it has to train 12 models twice (as described above and in the report).
  * Run `blend.py`, providing it with the intermediate files paths.
  
You can download the intermediate files from [here](https://drive.google.com/open?id=1aMuw3N0EMJsNhXxn-1V9yyjwPcQGCpi6).

Assuming that you have downloaded the intermediate data in `/home/user/Downloads`, you may
run the following two commands in order to produce the submission CSV file for CrowdAI.

    python3 run.py --testing=/home/user/Downloads/intermediate-data/testing.csv \
                    --testing-predictions=/home/user/Downloads/intermediate-data/predictions_testing \
                    --submission-predictions=/home/user/Downloads/intermediate-data/predictions_submission \
                    --output=final_submission.csv
    
As always, you can see more information on the accepted options with the `--help` parameter.

Please note that `blend.py` reads CSV files from two directories: `intermediate/predictions_testing`
and `intermediate/predictions_submission`. The files corresponding to the same model must be identically
named, in order for the blending to be successful. The script will give you a chance to check for this
before starting. If using the files provided by us, there should be no problem.
