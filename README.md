# *An Exact Hypergraph Matching Algorithm for Nuclear Identification in Emrbyonic C. elegans* 

This repository contains the Python implementation of *Exact HGM* applied to seam cell nuclei identification in embryonc *C. elegans*, as described in:

- Andrew Lauziere, Ryan Christensen, Hari Shroff, Radu Balan. [*An Exact Hypergraph Matching Algorithm for Nuclear Identification in Embryonic Caenorhabditis elegans*] (http://arxiv.org)

## Overview

The algorithm exactly solves arbitrarily complex point-set matching problems by phrasing the task as *hypergraph matching*. A hypergraph furthers the canonical graph to include *hyperedges*. Hyperedges allow for associating more than two vertices at a time. The increased expressive power allows for modeling more complex phenomena. The discrete optimization problem uses a degree *n* mutlilinear objective function under both binary constraints and one-to-one constraints. Degree *2d* tensors <img src="https://latex.codecogs.com/gif.latex?\mathbf{Z}^{(d)}, d=1, 2, \dots, n" /> to store degree *d* feature dissimilarities between point-sets. 


