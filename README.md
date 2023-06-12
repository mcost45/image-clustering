# Image Clustering

**Configurable, unsupervised learning for determining optimal clusters within image datasets**

## Usage

1. Place images within the [data folder](data)
2. Set values within [config.py](config.py) as necessary
3. Run the main function, see the determined clusters within the [output folder](output)

## Process

Firstly, images are preprocessed, allowing for cropping, trimming, channel conversion and resizing, as per provided
configuration.

Feature vectors are then extracted by a pre-trained image-recognition CNN model (choice of VGG16, VGG19, or Xception) -
describing the underlying properties of each image.

Next, PCA is optionally applied to normalised feature vectors to reduce the data's dimensionality, whilst ensuring a
given percentage of explained variance. A lower explained variance value correlates with higher compression and faster
clustering performance, but more potential feature loss.

Clustering is then applied to the reduced feature data. Either k-means or DBSCAN may be used. DBSCAN may be preferred if
the number of expected clusters is not pre-determined, or clusters have uneven sizes, or unregular shapes. Both
clustering methods will test a range of given values to determine optimal clusters: a range of 'k' number of potential
clusters for k-means, or a range of epsilon values for DBSCAN.

The results for each test clustering are compared by determining their silhouette score. The clustering with the highest
score will be taken as the optimal result. In the case of DBSCAN, outlier images may optionally be excluded from the
silhouette score calculation.

Finally, the original images will be output to labelled folders for easy visualisation of the clustering results.

## Dependencies

- python - 3.10
- tensorflow - 2.10
- pillow - 9.4
- numpy - 1.23
- sklearn - 1.2