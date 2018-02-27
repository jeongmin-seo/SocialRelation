#################################################################
## this code for changing Depth value file format to matlab file
## first need depth raw text file
## this code get only depth image value
## created by 2017.06.29
#################################################################

##load package
import scipy as sp
import numpy as np
import scipy.io as sio

## setting parameter
file_path = "/home/user/Workspace/relative_depth/src/experiment/"
file_format = ".txt"

for M in range(16,260):
    print(M)
    file_ind = str(M)
    save_file_name = file_ind + "_DepthGT.mat"

    full = file_path + file_ind + file_format
    temp = np.ones(1)

    f = open(full, 'r')
    for i in range(1,19):
        line = f.readline()
        if i == 18:
            temp = line
    f.close()

    temp = temp.split(' ')
    r = len(temp)

    for i in range(0, r):
        temp[i] = float(temp[i])

    a={}
    a['value'] = temp

    sio.savemat(save_file_name,a)