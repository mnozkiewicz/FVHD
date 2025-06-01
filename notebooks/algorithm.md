# FVHD algorithm

## Embedding Algorithms Overview

Our goal is to find a mapping from a high-dimensional space Y to a lower-dimensional space X, typically 2D, for purposes such as visualization or compression.

The core idea of the algorithm builds on techniques like Multidimensional Scaling (MDS) and t-SNE. These methods aim to preserve the structure of the original space by minimizing a cost function of the form $E(||Y − X||)$.

Here, $||·||$ denotes a measure of topological dissimilarity between the original space Y and the embedded space X. This dissimilarity is often expressed as the difference between dissimilarity matrices, where $D_{ij}$ represents the distance (in some chosen metric) between data points $i$ and $j$.

In t-SNE, for example, the dissimilarity has a probabilistic interpretation, using pairwise similarities computed from conditional probabilities.

## Problem with Computational Cost

However, this approach introduces significant computational complexity. For a dataset with $n$ examples, each of dimensionality $M$, the time complexity of computing the dissimilarity matrix is $O(n² · M)$.

Fortunately, this computation is only required once at the beginning. However, we must also compute a dissimilarity matrix for the low-dimensional representations. This matrix needs to be recomputed at every iteration of the optimization process, resulting in a time complexity of $O(a · n²)$, where $a$ is the number of optimization steps required for convergence.

The space complexity is also $O(n²)$ due to the size of the dissimilarity matrices. This becomes a major bottleneck for large datasets — for example, when $n = 10^6$, storing and processing such matrices becomes computationally impractical.

## Sparse graph solution

To address scalability challenges, we adopt a sparse graph approximation strategy. Rather than computing all pairwise distances, we build a k-nearest neighbors (k-NN) graph, where each point connects only to its k closest neighbors.

To preserve some global structure and prevent the optimization from becoming overly local, we augment this graph by adding a small number of random edges. These long-range connections help the algorithm capture broader relationships across the dataset.

We assign weight 0 to edges between nearest neighbors, and weight 1 to the randomly added edges.

The problem of finding the optimal embedding thus reduces to fidning the 2d representation of this augmented k-NN graph in a low-dimensional space. This approach reduces the time complexity from $O(a \cdot n^2)$ to $O(a \cdot n)$, where $a$ is the number of optimization iterations.

## Loss function

Our loss function is the measure of dissimilarity between spaces Y and X. This is the mean squared error between dissimilarity matrix of Y and X. The matrix is no longer squared because if represents the sparse graph. Given that we use binary distances in space Y and euclidean distance in space X, the loss function has the following form.

$$
\mathcal{L} = \frac{1}{|NN|} \sum_{(i,j) \in NN} d_{ij}^2 + c \cdot \frac{1}{|RN|} \sum_{(i,j) \in RN} (1 - d_{ij})^2
$$

Where:

- $d_{ij}$ is the distance between point $i$ and $j$ in embedding space,
- $NN$ is the set of nearest neighbors,
- $RN$ is the set of random neighbors,
- $c$ is a hyperparameter controlling the contribution of the random neighbor loss term.

## Optimizer Algorithm steps

### 1. Building the graph

The step of finding the nearest neighbors has the $O(n \cdot log(n) \cdot M)$ time complexity. Then the random neighbors are added.

The step of building the graph in determined by two hyperparameters:

- $nn$ - the number of nearest neighbors
- $rn$ - the number of random neighbors

### 2. Optimizing the loss

The aformentioned loss function is then optimized with respect to 2-d embeddings. This step is mostly defined by the optimizer used.


## Force directed algorithm steps

This is an iterative method. We start from some random embedding and in every step we refine the positions of the points based on the notion of attractive and repulsive forces. 

### 1. Building the graph

This step is similar to the geaph building step from the optimizer algorithm.

### 2. Position update step

First we compute difference between neighboring vectors and their euclidean distance.

**Diffs**
$$
\Delta \mathbf{x}_{ij} = \mathbf{x}_i - \mathbf{x}_j
$$

**Distances**
$$
d_{ij} = \|\Delta \mathbf{x}_{ij}\| = \sqrt{\sum_k (x_{i,k} - x_{j,k} + \varepsilon)^2 }
$$

Then we calculate the attractive and repulsive forces.


**Attractive forces**
$$
\mathbf{F}_{\text{NN}, ij} = 
\begin{cases}
\frac{1}{d_{ij} + \varepsilon} \cdot \Delta \mathbf{x}_{ij}, & \text{(if using mutual neighbors)} \\
\Delta \mathbf{x}_{ij}, & \text{(default)}
\end{cases}
$$

**Repulsive forces**
$$
\mathbf{F}_{\text{RN}, ij} = \left( \frac{d_{ij} - 1}{d_{ij} + \varepsilon} \right) \cdot \Delta \mathbf{x}_{ij}
$$

**Total forces acting on a particle (point in 2d)**
$$
\mathbf{F}_i = -\sum_j \mathbf{F}_{\text{NN}, ij} - c \cdot \sum_j \mathbf{F}_{\text{RN}, ij}
$$


At the end we update our position using momentum like optimizer.

$$
\Delta \mathbf{x}_i^{(t)} = a \cdot \Delta \mathbf{x}_i^{(t-1)} + b \cdot \mathbf{F}_i
$$

$$
\mathbf{x}_i^{(t)} = \mathbf{x}_i^{(t-1)} + \eta \cdot \Delta \mathbf{x}_i^{(t)}
$$

$a$, $b$, and $\eta$  are hyperparameters.

In the algorithm the learning rate can be dynamically adjusted based on the particle velocities.
