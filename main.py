import math
import config
import os
import shutil
import numpy as np
from PIL import Image, ImageChops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from keras import Model
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Concatenate


def main():
    print("Setting up feature extraction model...")
    feature_extraction_model = get_feature_extraction_model()

    print("Getting all images...")
    filepaths = get_image_filepaths()

    print("Extract all image features...")
    image_features = extract_image_features(filepaths, feature_extraction_model)

    if config.PCA_ENABLED:
        print("Applying PCA...")
        image_features = apply_pca(image_features)

    clustering_method = config.CLUSTERING_METHOD.lower()
    match clustering_method:
        case 'dbscan':
            print("Applying dbscan...")
            clusters = apply_dbscan(filepaths, image_features)
        case 'kmeans':
            print("Applying k-means...")
            clusters = apply_kmeans(filepaths, image_features)
        case _:
            print("Unknown clustering method", clustering_method, "use 'dbscan' or 'kmeans'")
            clusters = []

    output_clusters(clusters)


def determine_feature_model_vector_length():
    match config.MODEL:
        case 'xception':
            return 2048
        case 'vgg16':
            return 4096
        case 'vgg19':
            return 4096


def determine_feature_model_class():
    match config.MODEL:
        case 'xception':
            return Xception
        case 'vgg16':
            return VGG16
        case 'vgg19':
            return VGG19


def get_feature_extraction_model():
    input_tensor = None

    if config.IMAGE_OUTPUT_CHANNELS == 1:
        input_0 = Input(
            shape=(config.IMAGE_OUTPUT_WIDTH_PX, config.IMAGE_OUTPUT_HEIGHT_PX, config.IMAGE_OUTPUT_CHANNELS))
        input_tensor = Concatenate()([input_0, input_0, input_0])

    model_class = determine_feature_model_class()
    print("Using", model_class.__name__, "for feature extraction...")

    model = model_class(
        input_shape=(config.IMAGE_OUTPUT_WIDTH_PX, config.IMAGE_OUTPUT_HEIGHT_PX, 3), pooling=config.MODEL_POOLING,
        input_tensor=input_tensor)

    return Model(inputs=model.inputs, outputs=model.layers[-2].output)


def get_image_filepaths():
    path = os.path.join(os.path.dirname(__file__), config.IMAGE_FOLDER_PATH)
    filetype = config.IMAGE_INPUT_TYPE.lower()
    filepaths = []

    with os.scandir(path) as files:
        for file in files:
            if file.name.lower().endswith(filetype):
                filepaths.append(os.path.join(path, file.name))

    return filepaths


def preprocess_image(filepath):
    image = Image.open(filepath).convert(config.IMAGE_COLOUR_MODE)

    # Potentially apply outer crop
    if config.IMAGE_OUTER_CROP_PX:
        left = top = config.IMAGE_OUTER_CROP_PX
        right = image.width - config.IMAGE_OUTER_CROP_PX
        bottom = image.height - config.IMAGE_OUTER_CROP_PX
        image = image.crop((left, top, right, bottom))

    # Potentially remove empty content
    if config.IMAGE_TRIM_ENABLED:
        trim_mode = "RGB"
        trim = Image.new(trim_mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image.convert(trim_mode), trim)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            image = image.crop(bbox)

    # Potentially convert to single channel
    if config.IMAGE_OUTPUT_CHANNELS == 1:
        if config.IMAGE_COLOUR_MODE == 'RGB':
            image = image.convert('L')
        else:
            image = image.convert('LA')

    # Resize to specified dimensions
    image = image.resize((config.IMAGE_OUTPUT_WIDTH_PX, config.IMAGE_OUTPUT_HEIGHT_PX), resample=Image.BILINEAR)

    # Convert to array form for model input
    as_array = np.expand_dims(np.array(image), axis=0)

    return as_array


def extract_image_features(filepaths, model):
    image_features = []
    final_count = len(filepaths) - 1

    for count, filepath in enumerate(filepaths):
        print(count, "/", final_count)
        image = preprocess_image(filepath)
        features = model.predict(image, use_multiprocessing=True, verbose=0)
        image_features.append(features)

    return np.array(image_features).reshape(-1, determine_feature_model_vector_length())


def apply_pca(image_features):
    scaled = StandardScaler().fit_transform(image_features)
    pca = PCA(n_components=config.PCA_EXPLAINED_VARIANCE, random_state=config.RANDOM_STATE)
    pca.fit(image_features)
    reduced_features = pca.transform(scaled)

    return reduced_features


def apply_dbscan(filepaths, reduced_features):
    max_score = -math.inf
    max_score_cluster_epsilon = config.DBSCAN_EPS_RANGE[0]
    max_score_dbscan = None
    final_count = len(config.DBSCAN_EPS_RANGE) - 1

    for count, eps in enumerate(config.DBSCAN_EPS_RANGE):
        print(count, "/", final_count)
        dbscan = DBSCAN(eps=eps, n_jobs=-1, min_samples=config.DBSCAN_MIN_SAMPLES)
        cluster_labels = dbscan.fit_predict(reduced_features)

        x = reduced_features
        labels = cluster_labels
        if config.DBSCAN_SILHOUETTE_EXCLUDE_OUTLIERS:
            # Note: outliers will be labelled under '-1' - this mask will only accept real clusters
            non_noise_mask = [x >= 0 for x in cluster_labels]
            x = reduced_features[non_noise_mask]
            labels = cluster_labels[non_noise_mask]

        if len(np.unique(labels)) <= 1:
            continue

        score = silhouette_score(x, labels)

        if score > max_score:
            max_score = score
            max_score_cluster_epsilon = eps
            max_score_dbscan = dbscan

    if max_score < 0:
        print("DBSCAN failed to find clusters. Trying k-means instead...")
        return apply_kmeans(filepaths, reduced_features)

    print("DBSCAN: Optimal epsilon is", max_score_cluster_epsilon, "with a score of", max_score)

    clusters = {}
    for file, cluster_label in zip(filepaths, max_score_dbscan.labels_):
        if cluster_label not in clusters.keys():
            clusters[cluster_label] = []
            clusters[cluster_label].append(file)
        else:
            clusters[cluster_label].append(file)

    return clusters


def apply_kmeans(filepaths, reduced_features):
    max_score = -math.inf
    max_score_cluster_count = config.K_MEANS_CLUSTER_RANGE[0]
    max_score_kmeans = None
    final_count = len(config.K_MEANS_CLUSTER_RANGE) - 1

    for count, cluster_count in enumerate(config.K_MEANS_CLUSTER_RANGE):
        print(count, "/", final_count)
        kmeans = KMeans(n_clusters=cluster_count, init='k-means++', n_init='auto', random_state=config.RANDOM_STATE)
        cluster_labels = kmeans.fit_predict(reduced_features)

        score = silhouette_score(reduced_features, cluster_labels)

        if score > max_score:
            max_score = score
            max_score_cluster_count = cluster_count
            max_score_kmeans = kmeans

    print("K-means: Optimal cluster count is", max_score_cluster_count, "with a score of", max_score)

    clusters = {}
    for file, cluster_label in zip(filepaths, max_score_kmeans.labels_):
        if cluster_label not in clusters.keys():
            clusters[cluster_label] = []
            clusters[cluster_label].append(file)
        else:
            clusters[cluster_label].append(file)

    return clusters


def output_clusters(clusters):
    shutil.rmtree(config.OUTPUT_CLUSTERS_PATH)
    os.mkdir(config.OUTPUT_CLUSTERS_PATH)

    for cluster_label in clusters:
        output_path = os.path.join(os.path.dirname(__file__), config.OUTPUT_CLUSTERS_PATH, str(cluster_label))
        os.mkdir(output_path)

        filepaths = clusters[cluster_label]
        for filepath in filepaths:
            output_filepath = os.path.join(output_path, filepath[filepath.rindex(os.sep) + 1:])
            shutil.copyfile(filepath, output_filepath)


if __name__ == "__main__":
    main()
