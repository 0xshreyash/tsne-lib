# t-SNE Interactive
This program allows the users to perform the t-Distributed Stochastic Neighbour Embedding algorithm. It allows for the creation of 
animation showcasing the progress of the application and also allows users to zoom-in to a region of graph produced in order to perform
t-SNE on the zoomed-in region (the utility of this is described in a following section).

## Introduction
t-SNE is based on the paper 

## Animation
This library allows you to create animations as t-SNE is going through its iterations.
![a relative link](./media/tsne.gif)

## Zooming-in
t-SNE is a stochastic method which means that if we are performing dimensionality reduction on a large number of points into a 2-D region
there might be scenarios wherein points look close by virtue of being far away from other points. But it might so happen that these points
are far away n-dimensional space and t-SNE might be able to display this distance between the points by applying the algorithm on just these
points without having a lot of other noise that pushes these points together during dimension reduction.

The difficulty in doing this is that when we use a library like sklearn it is easy to n-dimenional points to 2-dimensions and plot the 2-D
points on a graph. But when we want to apply t-SNE to a subsection of the graph an inverse function that maps from the points in 2-dimensions
to n-dimensions is not readily available. This program allows you to do this by simply zooming into a region of the graph that you want to re-perform
t-SNE on and selecting the desired perplexity of the t-SNE being applied.

