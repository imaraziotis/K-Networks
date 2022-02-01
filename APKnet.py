# Author: Ioannis Maraziotis <i.maraziotis@gmail.com>
#
# License: BSD 3 clause

import scipy.io as sio
from scipy.spatial import distance
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import utils4knets
# import numba
# from numba import prange

# *******************************
#    Assignment Knet Phase
# *******************************
# The implementation of the Assignment phase for the 3 different types of data formats 
# (Similarity, Pattern and Sparse Similarity matrix) accepted as input by K-Nets.
"""
assign_DMODE: assignment of samples to Nearst exemplars based on Data matrix. 

The dataset accepts as input: 
1. the index vector of the current partition
2. The newExemplars that have already been detected in function denex

For every exemplar in current_partition: 
1. Detects the K Nearest New Exemplars (ENE)
2. Assigns the members of each cluster among the ENE 

"""
def assign_DMODE(data, current_partition, newExemplars, kns):
    priorExemplars = np.unique(current_partition)
    new_labels = np.zeros(np.shape(current_partition)[0], dtype=np.int32)

    if kns['ENE'] > 1:
        Exemplar_K_Nearest_Exemplars = kns['ENE']
    else:
        Exemplar_K_Nearest_Exemplars = int(np.ceil(np.shape(newExemplars)) * kns['ENE'])

    # sort current partition (use in binary search below)
    prior_sorted_inds = np.argsort(current_partition)
    sorted_prior = current_partition[prior_sorted_inds]
    adists = pairwise_distances(data[newExemplars, :], metric=kns['metric'])
    Acsi = np.argsort(adists, axis=1)
    for i in np.arange(np.shape(priorExemplars)[0]):  # For every exemplar in current partition
        # Find the members of a cluster in the current partition.
        left, right = utils4knets.bsfreq(sorted_prior, priorExemplars[i])
        # cluster_members = sorted_prior[np.arange(left, right)]
        cluster_members = prior_sorted_inds[np.arange(left, right)]

        # For the corresponding exemplar in the newExemplars list/vector/array find its K Nearest Exemplars from the
        # newExemplars list.
        tdists = distance.cdist(np.array([data[newExemplars[i], :]]), data[newExemplars, :], kns['metric'])
        tdists = adists[i, :]
        # tdists = pairwise_distances(np.array([data[newExemplars[i], :]]), data[newExemplars, :], kns['metric'])

        csi = np.argsort(tdists)
        # csi = Acsi[i, :]
        # pExNNs = csi[0, 0:Exemplar_K_Nearest_Exemplars]
        pExNNs = csi[:Exemplar_K_Nearest_Exemplars]

        # Assign each one of the current cluster members to its new exemplar from the newExemplars list.
        MV, idx = samples2exemplars_DMODE(data, newExemplars[pExNNs], cluster_members, kns)
        new_labels[cluster_members] = idx

    return new_labels


# This function assigns a 'set' of samples to their nearest Exemplars
# called by: denex
def min_mean_dists_mat(data, cluster_members, exemplar_NNs, kns):
    cluster_members_pairs = utils4knets.set2parts(np.shape(cluster_members)[0], kns['scs'])

    exemplar_NNs_pairs = utils4knets.set2parts(np.shape(exemplar_NNs)[0], kns['ecs'])

    AS = np.zeros((np.shape(cluster_members_pairs)[1], np.shape(exemplar_NNs)[0]))

    for i in np.arange(0, np.shape(exemplar_NNs_pairs)[1]):  # for every data inds set
        cur_exemplars_NNs_inds = np.arange(exemplar_NNs_pairs[0, i], exemplar_NNs_pairs[1, i])
        for j in np.arange(0, np.shape(cluster_members_pairs)[1]):  # for every exeplars inds set
            cur_cluster_members_inds = np.arange(cluster_members_pairs[0, j], cluster_members_pairs[1, j])
            AS[j, cur_exemplars_NNs_inds] = np.sum(pairwise_distances(data[cluster_members[cur_cluster_members_inds], :],
                                                                  data[exemplar_NNs[cur_exemplars_NNs_inds], :],
                                                                  kns['metric']), axis=0)

            # AS[j, cur_exemplars_NNs_inds] = np.sum(distance.cdist(data[cluster_members[cur_cluster_members_inds], :],
            #                                                       data[exemplar_NNs[cur_exemplars_NNs_inds], :],
            #                                                       kns['metric']), axis=0)

    pmm = np.sum(AS, axis=0) / np.shape(exemplar_NNs)[0]
    aMV = np.min(pmm)
    aMI = np.argmin(pmm)

    return aMV, aMI


"""
denex (Detect new exemplars): For every cluster in a partition: 1) detect its members, 2) From its members detect the
nearest members of the cluster's exemplar, 3) set as new exemplar the one that has the minimum mean distance from all members

Parameters:
data: matrix of the form NxD
labels: assignment of the samples into clusters based on current data partition.
ENM: Number of exemplar's nearest members to be considered 
"""


def denex(data, labels, kns):
    exemplars = np.unique(labels)
    Nex = np.shape(exemplars)[0]  # Number of exemplars
    nExemplars = np.zeros(Nex, dtype=np.int32)

    labels_sorted_inds = np.argsort(labels)
    sorted_labels = labels[labels_sorted_inds]
    
    for i in range(Nex):
        curK = kns['ENM']

        # Find the members of the current cluster utilizing binary search
        left, right = utils4knets.bsfreq(sorted_labels, exemplars[i])
        members = labels_sorted_inds[range(left, right)]
        Nmembers = np.shape(members)[0]  # Number of members

        if curK <= 1:  # If the number of members to be considered 4 new exemplar is given in percentage (i.e. < 1)
            curK = int(np.ceil(curK * Nmembers))
        if Nmembers < curK:  # if the number of nearest members to be considered is larger than the number of members.
            curK = Nmembers

        # Find current Exemplars nearest neighbors (i.e. from the members of the cluster)
        # Exemplars_NNs_Dists = distance.cdist(np.array([data[exemplars[i], :]]), data[members, :], kns['metric'])
        Exemplars_NNs_Dists = pairwise_distances(np.array([data[exemplars[i], :]]), data[members, :], kns['metric'])
        sort_inds = np.argsort(Exemplars_NNs_Dists)
        Exemplars_NNs = members[sort_inds[0, 0:curK]]
        # Exemplars_NNs = Exemplars_NNs[0, :]
        tmv, tmi = min_mean_dists_mat(data, members, Exemplars_NNs, kns)
        nExemplars[i] = Exemplars_NNs[tmi]
    return nExemplars


# For every cluster: find the member with the minimum distance from the others. Set this as the new cluster exemplar.
def detect_new_exemplars_SMODE(Similarities, the_labels, the_exemplars, kns):
    exemplars_num = np.shape(the_exemplars)[0]

    for i in range(exemplars_num):
        members_indices = np.asarray(np.nonzero(the_labels == the_exemplars[i])[0])
        tmp = Similarities[np.ix_(members_indices, members_indices)]
        if kns['min_max'] == 1:
            the_exemplars[i] = members_indices[np.argmin(np.mean(tmp, axis=0))]
        else:
            # the_exemplars[i] = members_indices[np.argmax(np.mean(tmp, axis=0))]
            the_exemplars[i] = members_indices[np.argmax(np.mean(tmp, axis=0))]

    return the_exemplars


# Given a similarity matrix and a set of exemplars detect the NNexemplar for every sample.
def samples2exemplars_SMODE(Similarities, the_exemplars, kns):
    if kns['min_max'] == 1:
        nexinds = np.argmin(Similarities[the_exemplars, :], axis=0)
    else:
        nexinds = np.argmax(Similarities[the_exemplars, :], axis=0)

    unq_exemplars_inds = np.unique(nexinds)
    the_labels = np.zeros(np.shape(Similarities)[0])

    for i in np.arange(np.shape(unq_exemplars_inds)[0]):
        cur_inds = np.where(nexinds == i)
        the_labels[cur_inds] = the_exemplars[i]

    return the_labels


"""
Given a dataset (data) assign each sample in a set of indices (data_inds) to its nearest exemplar (given a set of 
exemplars: exemplars_inds). If the number of exemplars and/or the number of data indices is larger than a threshold 
break the corresponding set(s) into a number of subsets to match the size of the threshold(s).   
"""


def samples2exemplars_DMODE(data, exemplars_inds, data_inds, kns):
    # break the samples and exemplars indices into sets according to the corresponding threshold values
    data_pairs = utils4knets.set2parts(np.shape(data_inds)[0], kns['scs'])
    exemplars_pairs = utils4knets.set2parts(np.shape(exemplars_inds)[0], kns['ecs'])

    aMV = np.zeros(np.shape(data_inds)[0])
    aMI = np.zeros(np.shape(data_inds)[0], dtype=np.int32)

    for i in np.arange(0, np.shape(data_pairs)[1]):  # for every data inds set
        cur_data_inds = np.arange(data_pairs[0, i], data_pairs[1, i])

        MV = float('inf') * np.ones(np.shape(cur_data_inds)[0])
        MI = np.zeros(np.shape(cur_data_inds)[0])

        for j in np.arange(0, np.shape(exemplars_pairs)[1]):  # for every exeplars inds set
            cur_exemplars_inds = np.arange(exemplars_pairs[0, j], exemplars_pairs[1, j])

            cdists = pairwise_distances(data[exemplars_inds[cur_exemplars_inds], :], data[data_inds[cur_data_inds], :],
                                    kns['metric'])

            # cdists = distance.cdist(data[exemplars_inds[cur_exemplars_inds], :], data[data_inds[cur_data_inds], :],
            #                         kns['metric'])
            mi = np.argmin(cdists, axis=0)
            mv = np.min(cdists, axis=0)  # <-- No need for this change using the line above
            replace_inds = mv < MV

            MV[replace_inds] = mv[replace_inds]
            MI[replace_inds] = cur_exemplars_inds[mi[replace_inds]]

        aMI[cur_data_inds] = MI
        aMV[cur_data_inds] = MV

    exemplars2inds = 1
    # print(np.unique(exemplars_inds))
    if exemplars2inds == 1:
        aMI = exemplars_inds[aMI]

    # aMI = aMI.astype(int)
    return aMV, aMI


# Assignment Phase under Similarity mode.
def Aphase_SMODE(Similarities, exemplars, kns):
    labels = samples2exemplars_SMODE(Similarities, exemplars, kns)
    prior = labels
    for t in range(kns['iters']):
        exemplars = detect_new_exemplars_SMODE(Similarities, labels, exemplars, kns)
        labels = samples2exemplars_SMODE(Similarities, exemplars, kns)

        # check for convergence
        if np.sum(labels - prior) == 0.:
            # print(t)
            break
        prior = labels

    return labels


# APhase Data Mode
def Aphase_DMODE(data, exemplars, kns):
    MV, labels = samples2exemplars_DMODE(data, np.array(exemplars), np.arange(0, np.shape(data)[0]), kns)
    prior = labels
    for t in range(kns['iters']):

        exemplars = denex(data, labels, kns)
        # MV, labels = samples2exemplars_DMODE(data, np.array(exemplars), np.arange(0, np.shape(data)[0]), 'euclidean')
        labels = assign_DMODE(data, labels, np.array(exemplars), kns)

        # check for convergence
        if np.sum(labels - prior) == 0.:
            # print(t)
            break
        prior = labels

    return labels


# This function based on a SSM detects the new Exemplars, based on the NNs of the current Exemplars
def detect_new_exemplars_SSM(NNs, DNNs, Labels, Exemplars, kns):
    Num_Of_Exemplars_NNs = 10

    nExemplars = Exemplars
    Labels = np.int32(Labels)
    labels_sorted_inds = np.argsort(Labels)
    sorted_labels = Labels[0, labels_sorted_inds]

    for i in np.arange(np.shape(Exemplars)[0]): # for every exemplar
        # inds = np.nonzero(Labels == Exemplars[i])[1]

        left, right = utils4knets.bsfreq(sorted_labels[0], Exemplars[i])
        inds = labels_sorted_inds[0, range(left, right)]

        # Find its nearest neighbors (their number is predefined)
        Ex_NNs = NNs[Exemplars[i], :]
        Ex_NNs = Ex_NNs[0:Num_Of_Exemplars_NNs]

        # For every neighbor of the exemplar find its distance for the rest of the members of the cluster.
        V = np.zeros((np.shape(Ex_NNs)[0]))
        for j in np.arange(np.shape(Ex_NNs)[0]):
            mask = np.isin(NNs[Ex_NNs[j], :], inds, assume_unique=True)
            tmp = np.nonzero(mask)
            indices=np.int32((np.shape(NNs)[1]-1)*np.ones((len(inds))))
            indices[:np.shape(tmp)[1]]=tmp[0]
            V[j] = np.mean(DNNs[Ex_NNs[j], indices])
            
        if kns['min_max'] == 1:
            tmi = np.argmin(V)
        else:
            tmi = np.argmax(V)

        nExemplars[i] = Ex_NNs[tmi]

    return nExemplars

# Assignment phase based on Sparse Similarity Matrix (SSM).
def Aphase_SSM(NNs, DNNs, exemplars, kns):
    Labels = SSM_Samples2Exemplars(NNs, exemplars, kns)
    onebeforelast = Labels
    # sexemplars = np.sort(exemplars)
    last = Labels
    ci = 0
    for iters in np.arange(kns['iters']):
        ci = ci + 1
        exemplars = detect_new_exemplars_SSM(NNs, DNNs, Labels, exemplars, kns)
        Labels = SSM_Samples2Exemplars(NNs, exemplars, kns)
        if (np.sum(Labels - last) == 0) or (ci > 0 and (np.sum(Labels - onebeforelast) == 0)):
            print('convergence iters:', iters)
            break
        tmp=np.nonzero(Labels-last)[1]
        print('Percentage changed: ', np.shape(tmp)[0]/np.shape(Labels)[1],np.shape(tmp)[0])
        last = Labels
        if ci==2:
            onebeforelast = Labels
            ci = 0
    kns['aiters'] = iters
    return Labels

# This function assigns samples to exemplars based on a sparse similarity matrix (in the form of two matrices, one
# with the NNs of every sample and one with the corresponding distances).
def SSM_Samples2Exemplars(NNs, Exemplars, kns):

    Labels = np.zeros((1, np.shape(NNs)[0]))

    s = 0
    out = []
    tidx = np.zeros((1, np.shape(NNs)[0]))
    s = 0
    for i in range(0, np.shape(NNs)[0]):

        # Find the Nearest Exemplars of the current sample
        qidx = np.zeros((np.shape(NNs)[0]))
        
        qidx[Exemplars] = 1
        qidx[NNs[i, :]] += 1

        # qidx[NNs[i, :]] = 1
        # qidx[Exemplars] += 1

        ia = np.nonzero(qidx[NNs[i, :]] == 2)
        cn = NNs[i, ia]

        if np.shape(ia)[1] != 0:
            if kns['min_max'] == 1:  # Minimization Criterion
                Labels[0, i] = cn[0, np.argmin(ia)]
            else:
                Labels[0, i] = cn[0, np.argmax(ia)]
        else:
            Labels[0, i] = -1

        # celems = np.isin(NNs[i,:], Exemplars, assume_unique=True)
        # s = np.shape(celems)
        # if s[0] != 0:
        #     if kns['min_max'] == 1:  # Minimization Criterion
        #         # Labels[0, i] = cn[0, np.argmin(ia)]
        #         Labels[0, i] = NNs[i, celems][0]
        #         # print(cn[0, np.argmin(ia)] - NNs[i,np.in1d(NNs[i,:], Exemplars)][0])
        #     else:
        #         Labels[0, i] = NNs[i, celems][-1]
        # else:
        #     Labels[0, i] = -1

    unassigned_inds = np.nonzero(Labels == -1)[1]
    # print(unassigned_inds)

    # To the samples that had not the exemplars in the list of their NNs assign the label of its Nearest Neighbor
    if np.shape(unassigned_inds)[0]  != 0:
        cur_Neighbor = 2
        flag = 1
        while flag:
            for i in range(np.shape(unassigned_inds)[0]):
                if Labels[0, NNs[unassigned_inds[i], cur_Neighbor]] != -1:
                    Labels[0, unassigned_inds[i]] = Labels[0, NNs[unassigned_inds[i], cur_Neighbor]]
            cur_Neighbor += 1
            unassigned_inds = np.nonzero(Labels == -1)[1]
            if np.shape(unassigned_inds)[0] == 0:
                flag = 0

    # print(np.unique(np.int32(Labels)))
    return Labels
