import os
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# INPUT IMAGE OPTIONS
# ----------------------------------------------------------------------------------------------------------------------
IMAGE_FOLDER_PATH = "data"
IMAGE_INPUT_TYPE = "jpg"
IMAGE_COLOUR_MODE = "RGB"
# Images will be resized to the following dimensions - 299 required for the xception model, 224 required for a VGG model
IMAGE_OUTPUT_WIDTH_PX = 224
IMAGE_OUTPUT_HEIGHT_PX = 224
# '1' will convert to single-channel greyscale - '3' will retain RGB colour
IMAGE_OUTPUT_CHANNELS = 3
# Images will be cropped from the border by this amount (from their original size) - may be useful for removing any
# overlays/labels
IMAGE_OUTER_CROP_PX = 0
IMAGE_TRIM_ENABLED = False

# ----------------------------------------------------------------------------------------------------------------------
# MODEL OPTIONS
# ----------------------------------------------------------------------------------------------------------------------
# 'xception', 'vgg16', or 'vgg19'
MODEL = "vgg16"
# None, 'avg' or 'max'
MODEL_POOLING = None

# ----------------------------------------------------------------------------------------------------------------------
# PCA OPTIONS
# ----------------------------------------------------------------------------------------------------------------------
# Set to false to skip running PCA
PCA_ENABLED = True
# PCA output will explain this percentage of variance - a lower value means higher compression, and more potential
# data loss
PCA_EXPLAINED_VARIANCE = 0.99

# ----------------------------------------------------------------------------------------------------------------------
# CLUSTERING OPTIONS
# ----------------------------------------------------------------------------------------------------------------------
# Determines the clustering method that will be used - 'dbscan' or 'kmeans'
CLUSTERING_METHOD = "kmeans"

K_MEANS_CLUSTER_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Range of epsilon values that will each be checked for the highest silhouette score
DBSCAN_EPS_RANGE = np.linspace(0.1, 10, 25).tolist()
DBSCAN_MIN_SAMPLES = 10
# Determines if outliers (label=-1) will impact silhouette score calculations
DBSCAN_SILHOUETTE_EXCLUDE_OUTLIERS = True

# ----------------------------------------------------------------------------------------------------------------------
# OUTPUT OPTIONS
# ----------------------------------------------------------------------------------------------------------------------
OUTPUT_CLUSTERS_PATH = "output"

# ----------------------------------------------------------------------------------------------------------------------
# MISC
# ----------------------------------------------------------------------------------------------------------------------
# Seed for randomness to maintain constant results
RANDOM_STATE = 45
