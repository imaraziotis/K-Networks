# Author: IA Maraziotis <imaraziotis@gmail.com>
# 
# Licence: BSD 3 clause

# ann_knet: Aproximate K-Nets clustering. Recursive function.
# Operation: 1) Partitions subset of the dataset into a number of clusters, 2) Partition each one of the partititions
# into a number of clusters. Keep on Applying the aforemenionted process untill.

import numpy as np
from scipy.spatial import distance
import knet
import utils4knets as utils
from sklearn.metrics import pairwise_distances
# from numba import prange

def train(X, layers = 1, k=5, thres=2000, iters=0):
    prior = []
    exemplars = np.arange(np.shape(X)[0])
    for i in range(len(k)):
        prior.append(aknet(X[exemplars, :], k=k[i], thres=thres, iters=iters))
        exemplars = np.unique(prior[-1])

    return prior



# Given a dataset in X aknet applies K-nets on a random subset of the 
def aknet(data, k=3, thres=2000, iters = 0):

    # init_labels, tkns = SLKnet.sknet(data, k, rsv=thres, iters=0)
    init_labels, tkns = knet.train(data, k, rsv=thres, iters=iters)
    init_exemplars = np.unique(init_labels)
    labels = init_labels
    for i in init_exemplars:
        tinds = np.nonzero(init_labels == i)
        if np.shape(tinds)[1] > thres:
            tlabels = aknet(data[tinds, :])
            labels[tinds] = tinds[labels]
        elif np.shape(tinds)[0] > k*5:
            # tlabels = SLKnet.sknet(data[tinds, :], k)
            tlabels = knet.train(data[tinds, :], k)
            labels[tinds] = tinds[labels]
    return labels


def samples_in_HAP_partition(prior_sv, prior_si, clusters_indices, start_layer_index, end_layer_index=0):

    Exemplars = []
    for i in prior_sv:
        Exemplars.append(np.unique(i))

    cur_layer_index = start_layer_index
    while cur_layer_index != end_layer_index:
        all_samples = []
        for i in range(len(clusters_indices)):
        # for i in np.arange(num):
            left, right = utils.bsfreq(prior_sv[cur_layer_index], Exemplars[cur_layer_index][clusters_indices[i]])
            # inds = prior_sv[cur_layer_index]==Exemplars[cur_layer_index][clusters_indices[i]]
            all_samples.append(prior_si[cur_layer_index][np.arange(left,right)])

        clusters_indices = all_samples
        cur_layer_index -= 1

    # Finally detect the corresponding samples
    for cur_layer_index in np.arange(cur_layer_index, -1, 1):
        clusters_indices = Exemplars[cur_layer_index][clusters_indices]

    all_samples = clusters_indices
    return all_samples


def samplesIN_HKnet_Partition0(Exemplars, prior_sv, prior_si, clusters_indices, start_layer_index, end_layer_index=-1):
    cur_layer_index = start_layer_index
    while cur_layer_index != end_layer_index:
            all_samples = np.array((), dtype=np.int32)
            for i in range(len(clusters_indices)):
                left, right = utils.bsfreq(prior_sv[cur_layer_index], Exemplars[cur_layer_index][clusters_indices[i]])
                all_samples = np.r_[all_samples, prior_si[cur_layer_index][np.arange(left,right)]]
            clusters_indices = all_samples
            cur_layer_index -=1

    for cli in range(cur_layer_index, 1, -1):
        clusters_indices = Exemplars[cur_layer_index][clusters_indices]
    return all_samples

def get_ranges_arr(starts,ends):
    counts = ends - starts
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1],dtype=int)
    id_arr[0] = starts[0]
    id_arr[counts_csum[:-1]] = starts[1:] - ends[:-1] + 1
    return id_arr.cumsum()

    
def samplesIN_HKnet_Partition(Exemplars, prior_sv, prior_si, clusters_indices, start_layer_index, end_layer_index=-1):
    cur_layer_index = start_layer_index
    while cur_layer_index != end_layer_index:
            Lefts = np.zeros((len(clusters_indices)), dtype=np.int32)
            Rights = np.zeros((len(clusters_indices)), dtype=np.int32)
            for i in range(len(clusters_indices)):
                left, right = utils.bsfreq(prior_sv[cur_layer_index], Exemplars[cur_layer_index][clusters_indices[i]])
                Lefts[i] = left
                Rights[i] = right
            clusters_indices = prior_si[cur_layer_index][get_ranges_arr(Lefts, Rights)]
            cur_layer_index -=1

    for cli in range(cur_layer_index, 1, -1):
        clusters_indices = Exemplars[cur_layer_index][clusters_indices]
    return clusters_indices

# This functions detects and returns the k nearest Neighbors of a dataset X based on a number of pre-defined or pre-calculatered
# partitions in prior. 
# ENN: percentage or actual number of the nearest neighbors of each exemplar
def knets4anns(X, k, prior, enn, metric='euclidean'):
    K = k + 5  # The extra 5 number of NNs is for the case that the exact same distance fr

    if enn < 1:
        enn = np.int32(np.ceil(len(np.unique(prior[-1]))*enn))
    else:
        enn = enn

    Exemplars = []
    for i in prior:
        Exemplars.append(np.unique(i))

    prior_si = []
    prior_sv = []
    # Sort the clustering indices (layers partitions)
    for cur_prior in prior:
        si = np.argsort(cur_prior)
        prior_si.append(si)
        prior_sv.append(cur_prior[si])

    hierarchy_height = len(prior) - 1
    MVS = np.zeros((np.shape(X)[0], K))
    NNS = np.zeros((np.shape(X)[0], K))
    for i in range(np.shape(Exemplars[hierarchy_height])[0]):

        # source = samplesIN_HKnet_Partition(Exemplars, prior_sv, prior_si, [i], hierarchy_height)
        source = samplesIN_HKnet_Partition(Exemplars, prior_sv, prior_si, [i], hierarchy_height)
        tmp = X[Exemplars[hierarchy_height - 1][Exemplars[hierarchy_height][i]], :]
        tmp = tmp[np.newaxis, :]
        # dists = distance.cdist(tmp, X[Exemplars[hierarchy_height - 1], :])
        dists = pairwise_distances(tmp, X[Exemplars[hierarchy_height - 1], :])
        si = np.argsort(dists)
        texinds = np.unique(si[:, 0:enn])

        # target = samplesIN_HKnet_Partition(Exemplars, prior_sv, prior_si, texinds, hierarchy_height - 1)
        target = samplesIN_HKnet_Partition(Exemplars, prior_sv, prior_si, texinds, hierarchy_height - 1)
        cdists = pairwise_distances(X[source, :], X[target, :])

        tsi = np.argsort(cdists, axis=1)
        tsv = np.take_along_axis(cdists, tsi, axis=1)
        NNS[source, :] = target[tsi[:, 0:K]]
        MVS[source, :] = tsv[:, 0:K]

    return np.int32(NNS), MVS