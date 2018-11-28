# TOPOLOGICAL EXPLORATION OF NEURONAL NETWORK DYNAMICS
by Jean-Baptiste BARDIN, Gard SPREEMAN and Kathryn HESS. 


This repository contains the python code necessary to perform the analysis of the original paper.
The code will reauire you installed python 3.5 and the package brian2, PySpike and Scikit-learn
The persistent homology was computed with Ripser.


## 1) Simulation fo the Neuronal network
Execute brunel_network.py:
the script will simulate the network with differente input and g parameter values. 
Each simmulations is repeated 10 time with different random seeds.
The script will save the population firing rate and the individuals spike trains in a .npy file.


## 2) Processing before computing Persistent Homology: Computing similarity metrics
Execute neuron_similarity_metrics.py:
python neuron_similarity_metrics.py -i [input] -g [g]
parameters -i and -g are necessary to load the spiketrains of the required simulation.
You need to run the simulation of the network for this set of parameter before exectuing this script.

This script will output three txt files, one for each neuron similarity metrics.
This files are readily usable with Ripser to compute persistent homology.

## 3) Computing Persistent Homology.
This is perform with RIPSER: link
Please refer to their documentation to execute the analysis.

## 4) Processing before Machine Learning, feature extraction.
Execute brunel_features.py:
This script will take the results from the persistent homology computation done by ripser
and compute the feature that will be used during the machine learning procedure.

## 5) Machine learning
Execute brunel_classification.py
execute the machine learning procedure



