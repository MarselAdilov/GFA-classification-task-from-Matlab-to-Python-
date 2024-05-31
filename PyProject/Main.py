import scipy
import pandas as pd
import numpy as np
import time
from ReLU import ReLU_new
from Conv import Conv
from Pool import Pool
from trainConv import trainConv
from pprint import pprint

# Load 'Primary_data'
print('>>> Loading \'Primary_data\' - \'D_Max\' - \'D\'')
Primary_data = pd.read_csv('data.csv')
nanFilter = Primary_data.isna().any(axis=1).to_numpy() # None/nan filter
print('\tPrimary_data.shape = ', Primary_data.shape)
print('\tnanFilter.shape = ', nanFilter.shape)
print('\tnanFilter = ', nanFilter)
D_Max = Primary_data[~nanFilter]
D = np.reshape(np.array(D_Max, dtype=float), (1, -1))
print('\tD.shape = ', D.shape)
print('\tD = ', D)

# Load 'Data'
print('>>> Loading \'Data\' - \'X\'')
Data = scipy.io.loadmat('Data.mat')
X = Data['Data']
X = X[:,:,1:]  # removing first column of zeros
X = X[:,:,~nanFilter] # filtering the same way as D
print('\tX.shape = ', X.shape)
print('\tX = ', X)

# Initialize parameters
print('>>> Initialize parameters')
CC = [2, 2, 20]  # Cross correlation architecture 4*3*20
hidden_layers = [10, 10, 10]  # 3 layers
epoch = 2

print('>>> trainConv()')
start_time = time.time()
WC_Dmax, net_Dmax, tr_Dmax = trainConv(X[:, :, :], D[0, :], hidden_layers, CC, epoch)
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time)
