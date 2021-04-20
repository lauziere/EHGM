# *An Exact Hypergraph Matching Algorithm for Nuclear Identification in Emrbyonic C. elegans* 

This repository contains the Python implementation of *Exact HGM* applied to seam cell nuclei identification in embryonc *Caenorhabditis elegans*, as described in:

- Andrew Lauziere, Ryan Christensen, Hari Shroff, Radu Balan. [*An Exact Hypergraph Matching Algorithm for Nuclear Identification in Embryonic Caenorhabditis elegans*] (http://arxiv.org)

## Overview

The algorithm exactly solves arbitrarily complex point-set matching problems by phrasing the task as *hypergraph matching*. A hypergraph furthers the canonical graph to include *hyperedges*. Hyperedges allow for associating more than two vertices at a time. The increased expressive power allows for modeling more complex phenomena. The discrete optimization problem uses a degree *n* mutlilinear objective function under both binary constraints and one-to-one constraints.

The optimization problem is exactly solved via a branch and bound approach. Branches of size *k* are assigned at each step according to the selection rule *H*. The selection rule greedily steers the search according to lower degree hyperedge multiplicities. Deeper into the search the aggregation rule *I* accumululates higher degree hyperedge terms. Branches are pruned throughout the search to remove infeasible paths. 

## Nuclear Identification in Embryonic *Caenorhabditis elegans*

The search algorithm is applied to finding globally optimal nuclear identity assignments in embryonic *C. elegans*. The roundworm features either *n=20* or *n=22* seam cell nuclei. The seam cell nuclei can be used as fiducial markers, such that when correctly identified the nuclei recover the posture of the coiled embryo. Competing constraints in fluorescence microscopy force researchers to image sparsely in time in order to capture high spatial resolution images. Images taken at five minute intervals contain bright fluorescent elipsoids; each of which is the glowing nucleus in a seam cell. The center coordinates of the set of homogeneous nucleus are used as input with goal of identity assignment. 

One graphical model and three hypergraphical models are proposed to contextualize relationships among the nuclei. *QAP* models the task as graph matching. Edge lengths from pairs of local nuclei within the worm are used to predict nuclear identities. More complex hyerpgraphical models *Pairs* and *PF* use degree four and six multiplicity terms, allowing for more complex features better able to capture the worm's physiology. The *Full* model uses exclusively degree *n* terms, yielding the most computationally intense search. All models use *k=2*, selecting each pair of nuclei at a time from the tail to head. 

## Installation

The code requires Python 3.6 or higher. *ExactHGM* can be installed through 'pip':

  'pip install exacthgm'
  
## Use

The four aforementioned models (*QAP*, *Pairs*, *Full*, *PF*) are ready for use on seam cell identification. The file 'config.py' allows a user to set hyperparameters and choose which dataset in which to run the search. The sixteen datasets are freely available for evaluation. Runtime limits and an initial upper bound on the objective minimum can be set in the configuration. 

The file 'search.py' runs the search contingent on the settings in 'config.py'. Output folder structure is generated in the '.../Exact_HGM/Results/...' folder. The top predictions, corresponding costs, and runtime are saved for each sample. 

The file 'build_arrays.py' creates the relevant statistical estimates for the multivariate Gaussian distributions. These are included in the repository already. However, the process in which to use the annotated data is highlighted such that users can craft other hypergraphical models for seam cell identification, or follow the process on other datasets.  



