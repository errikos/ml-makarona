import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import csv

from helpers import load_data, preprocess_data

path_dataset = "../data/submission.csv"
res_path = "../data/our_submission.csv"
ratings = load_data(path_dataset)
sub = ratings[:17,:17]
print(sub.todense())
# vait = [1,2,4,7,8]
# vaus = [2,4,6,8,9]
#rat = ratings[vait][:,vaus]
# rat = ratings[vait]
# print(rat.todense()[:10,:10])
# rat = rat[:,vaus]
# print(rat.todense()[:10,:10])

# print(ratings[vait,vaus].shape)

# print(ratings[vait,vaus].todense())
print("NONZERO")
predictions = ratings
rows, cols = predictions.nonzero()
print(rows)
print(cols)
zp = list(zip(rows, cols))
zp.sort(key=lambda tup: tup[1])
# print(zp)

cnt = 0
for row, col in zp:
    r = "r" + str(row+1)
    c = "c" + str(col+1)
    print( r + "_" + c + "," + str(predictions[row,col]))
    cnt+=1
    if(cnt>30):
        break
# for entry in ratings[:10]:
#     print(entry)

with open(res_path, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for row, col in zp:
            r = "r" + str(row+1)
            c = "c" + str(col+1)
            writer.writerow({'Id': r+"_"+c, 'Prediction': int(predictions[row,col])})