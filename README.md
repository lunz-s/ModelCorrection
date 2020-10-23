# On learned operator correction in inverse problems

This repository contains the code used in the paper [On learned operator correction in inverse problems](https://arxiv.org/abs/2005.07069).
We investigate correcting for modelling errors in Photoacoustic Tomography (PAT) by training a neural network as operator correction. 

### Running the code
The files training_*.py contain the code to train the correction for all corrections introduced in the paper. In order to run the code locally, it is required to 
change the paths for savepoints, operator models and data to the appropriate paths on your device. The Evaluation notebook contains routines to evaluate the trained model and to create the figures displayed in the paper.

### Data, Operators and Savepoints
The data and operators can be downloaded [here](https://figshare.com/s/2e1ccd5319d5728683d6) and the weights of the trained network for the recursive forward-adjoint correction for both ball and vessel data, as introced in the paper, can be found [here](https://figshare.com/s/01bfbb4c69a885e0c5c3).
