# Script for training one iteration of DGD for learned 3D photoacoustic
# imaging. This is meant to be called from Matlab, where evaluation of the
# gradient is done with the k-wave toolbox.
#
# This is accompanying code for: Hauptmann et al., Model based learning for 
# accelerated, limited-view 3D photoacoustic tomography, 
# https://arxiv.org/abs/1708.09832
#
# written by Andreas Hauptmann, January 2018
# ==============================================================================


#import Load_PAT3D_fast_16beam_eval as loadPAT
from Operators import Load_PAT2D_data as loadData
import matplotlib.pyplot as plt

import h5py
import numpy as np

#dataSetTest   = '../saveData/GradTest_Beam16_Iter1.mat'
#dataPAT = loadPAT.read_data_sets(dataSetTest)


MEASDATA = '../data/testDataSet.mat'
measData=loadData.read_data_sets(MEASDATA)


''' ----> LOAD MATRIX <---- '''
matrixName = '../matrices/threshSingleMatrix4Py.mat'
fData = h5py.File(matrixName,'r')
inData = fData.get('A')  
rows = inData.shape[0]
cols = inData.shape[1]
print(rows, cols)
forwMat = np.matrix(inData)
    

#Some consistency tests of projections
sampleIdx=80
testData=measData.test.dataTru[sampleIdx,:,:]
testForw=measData.test.dataApr[sampleIdx,:,:]
testImag=measData.test.imagTru[sampleIdx,:,:]


''' ----> APPLY MATRIX <---- '''
imagVec=np.reshape(testImag,[64*64])
testForw=np.reshape(np.matmul(forwMat,imagVec),[64,64])

plt.figure()
plt.imshow(testForw)

plt.figure()
plt.imshow(testData)


''' ----> TEST ADJOINT <---- '''
dataVec=np.reshape(np.array(testForw),[64*64])
testAdjoint = np.reshape(np.matmul(np.transpose(forwMat),dataVec),[64,64])

plt.figure()
plt.imshow(testAdjoint)


