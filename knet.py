from scipy.spatial import distance
import numpy as np
import utils4knets
import APKnet
import CSKnet
from sklearn.metrics import pairwise_distances

'''
Brief K-nets description. 
K-networks: Exemplar based clustering algorithm. It can be operated as a deterministic or stochastic process. 
The basic K-nets parameter is k an integer resolution parameter. The smaller the value of K-nets the larger the number 
of clusters extracted. K-nets operation is composed of two sequential phases. In the first (Construction/Selection), an 
initial partition is determined by selecting a number of exemplars from the overall dataset while in the second 
(Assignment) the samples are assigned to their nearest exemplar or we can have a number of iterations similar to k-means 
until convergence.

Input Parameters:
Required: 
X:  Input data that can be in the form of 1) Data Matrix (NxD, N: Number of samples, D: Number of features, 
    2) Similarity Matrix (NxN), 3) NNs and DNNs Lists (both of dimensionality NxK, with K being the K NNs of every 
    sample)
k: resolution parameter (1<=k<~=N/2, with N being the number of samples)

Optional:
c: The number of requested clusters. 
sims: if sims=1, it is indicated that a similarity matrix is given as input (default: 0).

Output Values:
 
'''


# Ioannis A. Maraziotis
# (c) 2021

# Initialize the parameters for Single-layer K-nets into a K-net structure (kns) that currently is a dictionary
# and set any user defined values. KNS is utilized to transfer the values of the parameters in the various Knet functions.
def handle_input_arguments(k, **in_args):
    # kns = {'k': 1, 'exact': [], 'rsv': 1, 'metric': 'euclidean'}
    kns = {'k': 1}
    kns['c'] = 0
    kns['metric'] = 'euclidean'
    kns['iters'] = 20
    kns['rsv'] = 1  # Random sampling value (rsv=1: utilize all data for CSPhase)
    kns['sims'] = 0  # Similarities passed as data source instead of data matrix
    kns['info'] = 0  # Print operational msgs (default: 0 - do not print)
    kns['min_max'] = 1  # minimize or maximize clustering criterion
    kns['dthres'] = 10000  # threshold that if exceeded the single-layer Knets is not activated
    kns['ecs'] = 1000  # Exemplars Components Size: The exemplars are divided into components of size ecs
    kns['scs'] = 1000  # Samples Components Size: The samples in a dataset are divided into components of size scs
    kns[
        'ENE'] = 50  # Number of Exemplar's Nearest Exemplars that will be considered for the new assignment of the samples
    kns[
        'ENM'] = 50  # Number of Exemplar's Nearest cluster Members that will be considered for detecting the new exemplar

    if isinstance(k, dict):
        struct = k
    else:
        struct = in_args
        kns['k'] = k

    # Set in Knet Structure the user defined paramaters
    for key, value in struct.items():
        kns[key] = value

    # print(kns)
    return kns


"""input: X, kns. 
X: is the data type . It can be one of the followings:
(1). Data matrix, (2): Similarity Matrix, (3): Nearest Neighbors and corresponding distances sets/arrays.
kns: is a structure (i.e. dictionary) with the parametes of the model. 

"""


def initialize_knet(X, kns):

    # Xtype is the input type. It can be: (1) Data Matrix, (2) Similarity
    # matrix, (3): Nearest Neighbors and corresponding distances sets/arrays.
    Xtype = 1
    if kns['sims'] == 1:
        Xtype = 2
        n = np.shape(X)[0]
    elif isinstance(X, list):
        Xtype = 3
        NNs = X[0]
        DNNs = X[1]
        n = np.shape(NNs)[0]
    else:
        n = np.shape(X)[0]
    
    kns['Xtype'] = Xtype


    # Minimize or maximize the clustering criterion of the most central points
    # to have maximum or minimum distances....
    if kns['min_max'] == 1:
        sorting_option = 'ascend'
    else:
        sorting_option = 'descend'


    # If a random sampling value has been provided, randomly
    # select a fraction (CSinds) of the input dataset for the Construction /
    # Selection Phase. In this version rsv is activated if the input is a data
    # matrix.

    # Number of samples (samples4CS) that will be considered during the CSPhase
    if kns['rsv'] != 1:
        if kns['rsv'] > 1:  # If rsv is in the form of samples number
            samples4CS = np.random.randint(n, size=kns['rsv'])
        else:  # If rsv is in the form of percentage of the total samples
            samples4CS = np.random.randint(n, size=int(kns['rsv'] * n))
    else:
            samples4CS = np.arange(n)  # The whole dataset will be considered for CSPhase

    kns['samples4CS'] = samples4CS

    if isinstance(X, list):
        # The samples that are part of the NNs lists have to be extracted as well
        NNs = NNs[samples4CS, :]
        DNNs = DNNs[samples4CS, :]
        Similarities = []
    else:
        if np.shape(samples4CS)[0] < kns['dthres']:
            if kns['sims'] == 0:
                # Similarities = distance.cdist(X[samples4CS, :], X[samples4CS, :],
                #                               kns['metric'])  # Calculate Similarity Matrix
                Similarities = pairwise_distances(X[samples4CS, :], metric=kns['metric'])
            else:
                Similarities = X[np.ix_(samples4CS, samples4CS)]

            if kns['min_max'] == 1:
                NNs = np.argsort(Similarities)
                DNNs = np.take_along_axis(Similarities, NNs, axis=1)
            elif kns['min_max'] == 2:
                NNs = np.argsort(-Similarities)  # The minus isng is for sorting reasons
                DNNs = np.take_along_axis(Similarities, NNs, axis=1)
        else:
            Similarities = []
            NNs = DNNs = []

    return kns, Similarities, NNs, DNNs
   

"""
knet: Implements a single layer K-Net. The algorithm has 2 modes SMODE (sparseMODE) and DMODE (denseMode). 
In DMODE a full similarity matrix is build and provided as input in the CSPhase

Input Parameters:
X: it can be: 
    data matrix of the form NxD where N: number of samples, D: number of Features
    similarity matrix NxN, N: number of samples
    A tuple in the form (NNs, NDists), NNs: matrix of the form NxD, N number of samples K: nearest neighbors and NDists: The Distance from the neighbors

sims: X is a similarity matrix (sims=1) or a data matrix (sims=0, default).
min_max: Maximize (min_max=2) or minimize (min_max = 1, default) the partitioning criterion.
rsv: Random Sampling Value, the number or percentage of samples utilized during CSPhase. 
     It can be an integer 2 <= rsv <= N, N: number of samples, or 
     a real number 0<rsv<1 indicating the percentage of samples, 
     the default value is 1 (i.e. all samples)
dthres: The number of samples  

"""


def train(X, k, **inp_args):
    # Knet structure (kns) is a dictionary with the parameters of the K-network.

    kns = handle_input_arguments(k, **inp_args)
    kns, CS_Similarities, NNs, DNNs = initialize_knet(X, kns)

    if len(CS_Similarities) == 0 and len(NNs) == 0:
        print('Number of data samples larger than allowed. Utilize rsv, or increase dthres or utilize Sparse Similarity Matrix.')
        labels = []
        kns = []
    else:
        # Activate Construction/Selection Phase to Extract preExemplars
        exemplars = CSKnet.CSPhase_SMODE(NNs, DNNs, kns)

        if isinstance(X, list):
            labels = APKnet.Aphase_SSM(NNs, DNNs, kns['samples4CS'][exemplars], kns)
        else:
            if kns['sims'] == 1:  # if no sampling requested for CSPhase
                # labels = APKnet.Aphase_SMODE(Similarities, exemplars, kns)  # Assignment Phase Similarities Mode
                labels = APKnet.Aphase_SMODE(X, kns['samples4CS'][exemplars],kns)  # Assignment Phase Similarities Mode
            else:
                labels = APKnet.Aphase_DMODE(X, kns['samples4CS'][exemplars], kns)  # Assignment Phase Data Mode

    return np.int32(labels), kns
