import numpy as np
import parsers

#x = np.array([   [[1],[2]]   ,  [[3],[4]]   ,   [[5],[6]]    ])
#print(x.shape)

'''
arr = np.arange(9).reshape((3, 3))
print(arr)
np.random.shuffle(arr)
print(arr)
'''
'''
x = np.arange(9.).reshape(3, 3)
print(x)
tmp, irel = np.where( x > 4 )
print(np.unique(tmp))
'''

#lambdas = np.logspace(-5, 0, 2)
#print (lambdas)


#print(np.append(lambdas,3))

x = np.array([[3, 4], [8, 9]])
print(parsers.build_poly(x,3))

zeroethPower = np.ones((x.shape[0], 1))