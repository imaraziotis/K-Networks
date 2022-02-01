<p align="center">
<img src="/images/K-Nets4.png" >
</p>
<p align="center">
K-Nets partitioning 100000 2d points into 100, 1000 and 10000 clusters.
</p>

## K-Networks (K-Nets)
K-Networks is a scalable exemplar-based clustering algorithm. Its operation is composed of three phases. In the first two, the number of exemplars is determined and we have an initial partition while the last is a fine tuning phase.


### Resolution: 
Clustering resolution (i.e. number of extracted clusters) in K-Nets is controlled through the integer parameter k indicating the k NNs of every sample. The smaller the value of k the larger the number of extracted clusters. <br><br>
There is however a supplemenary mode refered to as Exact Operational Mode (EOM) under which we can partiton a dataset into a prespecified number of clusters C. Through EOM we can provide as input to K-Nets, along the k value, another integer indicating the exact number of clusters in which we want to partition the dataset. In EOM it is beneficial, but not obligatory, the number of clusters extracted from K-Nets for the input k value to be near the number of requested clusters C [(see example notebook)](Knets_Artificial.ipynb). <br>

### Brief 
Based on the initial K-Nets partition (Construction/Selection) the number of iterations requested for convergence during the last (Assignment) phase is relatively small in most setups. The key K-Nets process in Construction is the detection of the k-NNs of every sample. The complexity introduced by this operation can be reduced, in cases of medium to small resolution, since K-Nets can succesfully conclude the Selection phase based on only a fraction of the total number of samples from Construction. While this operation converts to stochastic the otherwise deterministic K-Nets operation, the solution remains stable due to EOM and Assignment phase [(see example notebook)](Knets_Artificial.ipynb).

### Citation:
If you make use of this software please consider citing the following papers:<br>
1. IA Maraziots, S Perantonis, A. Dragomir, D. Thanos, "K-Nets: Clustering though Nearest Neighbors Networks", Pattern Recognition 2019<br> 
2. <br>

### Licence:
The code in this repositary is released under the following licence:
 BSD-3-Clause License