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
import Load_PAT2D_data as loadData
import matplotlib.pyplot as plt

import fastPAT


#dataSetTest   = '../saveData/GradTest_Beam16_Iter1.mat'
#dataPAT = loadPAT.read_data_sets(dataSetTest)


MEASDATA   = '../data/testDataSet.mat'
measData=loadData.read_data_sets(MEASDATA)


KGRID_BACK  = '../data/kgrid_small.mat'
kgridBack=fastPAT.kgrid(KGRID_BACK)

KGRID_FORW  = '../data/kgrid_smallForw.mat'
kgridForw=fastPAT.kgrid(KGRID_FORW)


angThresh=60
kspaceMethod=fastPAT.fastPAT(kgridBack,kgridForw,angThresh)

#Some consistency tests of projections
sampleIdx=80
testData=measData.test.dataTru[sampleIdx,:,:]
testForw=measData.test.dataApr[sampleIdx,:,:]
testImag=measData.test.imagTru[sampleIdx,:,:]

plt.figure()
plt.imshow(testData)

plt.figure()
plt.imshow(testForw)

plt.figure()
plt.imshow(testImag)

res=kspaceMethod.kspace_backward(testData)
plt.figure()
plt.imshow(res)

res2=kspaceMethod.kspace_forward(testImag)
plt.figure()
plt.imshow(res2)

res3=kspaceMethod.kspace_backward(testForw)
plt.figure()
plt.imshow(res3)


#res4=kspaceMethod.kspace_forward(res3)
#plt.figure()
#plt.imshow(res4)
#
#
#res5=kspaceMethod.kspace_backward(res4)
#plt.figure()
#plt.imshow(res5)