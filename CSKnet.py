# Author: Ioannis Maraziotis <imaraziotis@gmail.com>
#
# License: BSD 3 clause
import scipy.io as sio
from scipy.spatial import distance
import numpy as np
import utils4knets

# *********************************************
#      Construction/Selection Knet Phases
# *********************************************

"""
CSPhase_SMODE (Construction & Selection Phase / Similarity Mode): Constructs Pre-Clusters and Selects the most compact
ones. The number of the corresponding/selected Pre-Exemplars formulates the final number of clusters.

Input Parameters:
Similarities: This is the data input and it can have the forms:
    1. Square Similarity Matrix of the form NxN, where N is the number of samples.
    2. A tuple composed of two vectors. The first one contains the K+k NNs of each sample, while the second the
       second the corresponding similarity values.
k: The clustering resolution parameter
c: Optional Parameter indicating the number of requested clusters.
min_max: Indicates whether K-Nets critetion is to be minimized or maximized (default min_max = 1)
"""

def CSPhase_SMODE(NNs, DNNs, kns):


    # n = np.shape(Similarities)[0]
    n = np.shape(NNs)[0]

    # Check if we have to minimize or maximize the criterion and if a data or similarity matrix has been provided as input
    # if kns['min_max'] == 1:
    #     sorted_dists_inds = np.argsort(Similarities, axis=1)
    # else:
    #     sorted_dists_inds = np.transpose(np.argsort(Similarities, axis=0)[::-1])

    scores = np.zeros(n)
    IDX = np.zeros(n)
    cur_exemplars_num = 0
    PCs = [None] * n
    exemplars = []

    Kinit = kns['k']
    # The basic loop of CSPhase K-Net. It originates from the input k value and decreases it until the requested number of
    # exemplars is reached.
    for k in range(Kinit, 0, -1):
        # Construction Phase
        for i in np.arange(n):
            sv = DNNs[i, :]
            si = NNs[i, :]

            if kns['min_max'] == 1:
                # cinds = np.nonzero(Similarities[i, sorted_dists_inds[i, :]] <= Similarities[i, sorted_dists_inds[i, k-1]])
                equal_distanced_members = np.nonzero(sv <= sv[k-1])
            else:
                # cinds = np.nonzero(Similarities[i, sorted_dists_inds[i, :]] >= Similarities[i, sorted_dists_inds[i, k-1]])
                equal_distanced_members = np.nonzero(sv >= sv[k-1])

            # scores[i] = np.sum(Similarities[i, sorted_dists_inds[i, cinds]]) / (k + 1)
            scores[i] = np.sum(sv[equal_distanced_members]) / (k + 1)
            # PCs[i] = sorted_dists_inds[i, cinds]  # PCs.append(sorted_dists_inds[i, cinds])
            PCs[i] = si[equal_distanced_members]  # PCs.append(sorted_dists_inds[i, cinds])

        if kns['min_max'] == 1:
            sorted_scores_inds = np.argsort(scores)
        else:
            sorted_scores_inds = np.argsort(scores)[::-1]

        # Selection Phase
        for i in np.arange(n):
            if np.sum(IDX[PCs[sorted_scores_inds[i]]]) == 0:
                cur_exemplars_num = cur_exemplars_num + 1
                IDX[PCs[sorted_scores_inds[i]]] = 1
                exemplars.append(sorted_scores_inds[i])  # exemplars[cur_exemplars_num] = 1#

        # Break the CSPhase if is:
        #   1. Standard mode (i.e. c=0) OR
        #   2. Exact mode (c>0) AND the current number of exemplars is larger than the requested number c.
        if cur_exemplars_num >= kns['c']:
            break

    Nex = len(exemplars)  # Number of exemplars

    # if the number of requested exemplars/clusters c is larger than the current exemplars number Nex, set
    if kns['c'] > Nex:
        c = Nex

    # Select the exemplars corresponding to the c most compact clusters.
    if kns['c'] != 0:
        exemplars = exemplars[0:kns['c']]

    return exemplars


def CSPhase_SMODE_prior(kns):


    # n = np.shape(Similarities)[0]
    n = np.shape(kns['NNs'])[0]

    # Check if we have to minimize or maximize the criterion and if a data or similarity matrix has been provided as input
    # if kns['min_max'] == 1:
    #     sorted_dists_inds = np.argsort(Similarities, axis=1)
    # else:
    #     sorted_dists_inds = np.transpose(np.argsort(Similarities, axis=0)[::-1])

    scores = np.zeros(n)
    IDX = np.zeros(n)
    cur_exemplars_num = 0
    PCs = [None] * n
    exemplars = []

    Kinit = kns['k']
    # The basic loop of CSPhase K-Net. It originates from the input k value and decreases it until the requested number of
    # exemplars is reached.
    for k in range(Kinit, 0, -1):
        # Construction Phase
        for i in np.arange(n):
            sv = kns['DNNs'][i, :]
            si = kns['NNs'][i, :]

            if kns['min_max'] == 1:
                # cinds = np.nonzero(Similarities[i, sorted_dists_inds[i, :]] <= Similarities[i, sorted_dists_inds[i, k-1]])
                equal_distanced_members = np.nonzero(sv <= sv[k-1])
            else:
                # cinds = np.nonzero(Similarities[i, sorted_dists_inds[i, :]] >= Similarities[i, sorted_dists_inds[i, k-1]])
                equal_distanced_members = np.nonzero(sv >= sv[k-1])

            # scores[i] = np.sum(Similarities[i, sorted_dists_inds[i, cinds]]) / (k + 1)
            scores[i] = np.sum(sv[equal_distanced_members]) / (k + 1)
            # PCs[i] = sorted_dists_inds[i, cinds]  # PCs.append(sorted_dists_inds[i, cinds])
            PCs[i] = si[equal_distanced_members]  # PCs.append(sorted_dists_inds[i, cinds])

        if kns['min_max'] == 1:
            sorted_scores_inds = np.argsort(scores)
        else:
            sorted_scores_inds = np.argsort(scores)[::-1]

        # Selection Phase
        for i in np.arange(n):
            if np.sum(IDX[PCs[sorted_scores_inds[i]]]) == 0:
                cur_exemplars_num = cur_exemplars_num + 1
                IDX[PCs[sorted_scores_inds[i]]] = 1
                exemplars.append(sorted_scores_inds[i])  # exemplars[cur_exemplars_num] = 1#

        # Break the CSPhase if is:
        #   1. Standard mode (i.e. c=0) OR
        #   2. Exact mode (c>0) AND the current number of exemplars is larger than the requested number c.
        if cur_exemplars_num >= kns['c']:
            break

    Nex = len(exemplars)  # Number of exemplars

    # if the number of requested exemplars/clusters c is larger than the current exemplars number Nex, set
    if kns['c'] > Nex:
        c = Nex

    # Select the exemplars corresponding to the c most compact clusters.
    if kns['c'] != 0:
        exemplars = exemplars[0:kns['c']]

    return exemplars


def CSPhase_SMODE0(Similarities, kns):

    n = np.shape(Similarities)[0]

    # Check if we have to minimize or maximize the criterion and if a data or similarity matrix has been provided as input
    if kns['min_max'] == 1:
        sorted_dists_inds = np.argsort(Similarities, axis=1)
    else:
        sorted_dists_inds = np.transpose(np.argsort(Similarities, axis=0)[::-1])

    scores = np.zeros(n)
    IDX = np.zeros(n)
    cur_exemplars_num = 0
    PCs = [None] * n
    exemplars = []

    Kinit = kns['k']
    # The basic loop of CSPhase K-Net. It originates from the input k value and decreases it until the requested number of
    # exemplars is reached.
    for k in range(Kinit, 0, -1):
        # Construction Phase
        for i in np.arange(n):
            if kns['min_max'] == 1:
                cinds = np.nonzero(Similarities[i, sorted_dists_inds[i, :]] <= Similarities[i, sorted_dists_inds[i, k-1]])
            else:
                cinds = np.nonzero(Similarities[i, sorted_dists_inds[i, :]] >= Similarities[i, sorted_dists_inds[i, k-1]])

            scores[i] = np.sum(Similarities[i, sorted_dists_inds[i, cinds]]) / (k + 1)
            PCs[i] = sorted_dists_inds[i, cinds]  # PCs.append(sorted_dists_inds[i, cinds])

        if kns['min_max'] == 1:
            sorted_scores_inds = np.argsort(scores)
        else:
            sorted_scores_inds = np.argsort(scores)[::-1]

        # Selection Phase
        for i in np.arange(n):
            if np.sum(IDX[PCs[sorted_scores_inds[i]]]) == 0:
                cur_exemplars_num = cur_exemplars_num + 1
                IDX[PCs[sorted_scores_inds[i]]] = 1
                exemplars.append(sorted_scores_inds[i])  # exemplars[cur_exemplars_num] = 1#

        # Break the CSPhase if is:
        #   1. Standard mode (i.e. c=0) OR
        #   2. Exact mode (c>0) AND the current number of exemplars is larger than the requested number c.
        if cur_exemplars_num >= kns['c']:
            break

    Nex = len(exemplars)  # Number of exemplars

    # if the number of requested exemplars/clusters c is larger than the current exemplars number Nex, set
    if kns['c'] > Nex:
        c = Nex

    # Select the exemplars corresponding to the c most compact clusters.
    if kns['c'] != 0:
        exemplars = exemplars[0:kns['c']]

    return exemplars
