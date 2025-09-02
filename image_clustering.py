from concurrent.futures import ThreadPoolExecutor
import os

import cv2
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
import yaml


SCALER_DICT = {'Standard': StandardScaler, 'Robust': RobustScaler, 'MinMax': MinMaxScaler, 'Normalizer': Normalizer}
DECOMPOSER_DICT = {'PCA': PCA, 'NMF': NMF, 'TSNE': TSNE}
CLUSTERER_DICT = {'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering, 'DBSCAN': DBSCAN}

def load_images_from_folder(folder):
    """
    Read image files as unint8 and BGR from the given folder
    and return them in a generator.
    """
    files = [file.path for file in os.scandir(folder)]
    with ThreadPoolExecutor() as executor:
        image_array = executor.map(cv2.imread, files)
    return image_array

def get_thumbnails(image_array, size):
    """
    Return a numpy array of resized images (float, RGB) from a given image iterable.
    """
    thumbs = np.array([cv2.resize(img, size) for img in image_array])
    thumbs = thumbs / 255
    thumbs = np.flip(thumbs, -1)    
    return thumbs

def get_sklearn_data(img_array):
    """
    Return an array of flattened images from an image array.
    """
    img_array = img_array.reshape((img_array.shape[0], -1))
    return img_array

def get_clusters(data, scaler, decomposer, clusterer):
    """
    Perform scaling, dimensional reduction and cluster analysis on th given data.
    Return the array of cluster labels and the dimensionally reduced data.
    """
    if scaler is not None:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.copy()
    data_decomposed = decomposer.fit_transform(data_scaled)
    clusters = clusterer.fit_predict(data_decomposed)
    return clusters, data_decomposed

def show_cluster_plot(clusters, data_decomposed, img_array):
    """
    Visualise the result of a cluster analysis based on cluster labels,
    the dimensionally reduced data and the original image array.
    """
    fig, ax = plt.subplots()
    ax.scatter(data_decomposed[:, 0], data_decomposed[:, 1], c=clusters)
    cmap = plt.get_cmap("gist_rainbow")
    colors = cmap(np.linspace(0, 1, max(clusters) + 1))
    for idx, img in enumerate(img_array):
        color = colors[clusters[idx]] if clusters[idx] != -1 else 'white'
        image_box = OffsetImage(img, zoom=0.4)
        image_box.image.axes = ax
        ab = AnnotationBbox(image_box,
                            (data_decomposed[idx, 0], data_decomposed[idx, 1]),
                            bboxprops={'facecolor': color})
        ax.add_artist(ab)                    
    plt.show()

def show_cluster_plot2(ax, clusters, data_decomposed, img_array):
    """
    Visualise the result of a cluster analysis based on cluster labels,
    the dimensionally reduced data and the original image array
    and using the given matplotlib Axes. 
    """
    ax.scatter(data_decomposed[:, 0], data_decomposed[:, 1], c=clusters)
    cmap = plt.get_cmap("gist_rainbow")
    colors = cmap(np.linspace(0, 1, max(clusters) + 1))
    for idx, img in enumerate(img_array):
        color = colors[clusters[idx]] if clusters[idx] != -1 else 'white'
        image_box = OffsetImage(img, zoom=0.4)
        image_box.image.axes = ax
        ab = AnnotationBbox(image_box,
                            (data_decomposed[idx, 0], data_decomposed[idx, 1]),
                            bboxprops={'facecolor': color})
        ax.add_artist(ab)

def get_workers_from_config():
    """
    Return initialised scaler, decomposer and clusterer as given in the config file.
    """
    with open('image_clustering_config.yml') as hdl:
        conf = yaml.load(hdl, Loader=yaml.Loader)
    scaler, decomposer, clusterer = conf['scaler'], conf['decomposer'], conf['clusterer']
    scaler = None if scaler == 'None' else SCALER_DICT[scaler]()
    if decomposer['type'] == 'TSNE':
        decomposer = DECOMPOSER_DICT['TSNE']()
    else:
        decomposer = DECOMPOSER_DICT[decomposer['type']](n_components=decomposer['components'])
    if clusterer['type'] == 'DBSCAN':
        clusterer = DBSCAN(min_samples=clusterer['dbscan_min'], eps=clusterer['dbscan_eps'])
    else:
        clusterer = CLUSTERER_DICT[clusterer['type']](n_clusters=clusterer['n_clusters'])
    return scaler, decomposer, clusterer

def main():
    image_array = load_images_from_folder(r"test_images")
    thumb_array = get_thumbnails(image_array, (150, 100))
    img_data = get_sklearn_data(thumb_array)

    scaler, decomposer, clusterer = get_workers_from_config()

    clusters, data_decomposed = get_clusters(img_data, scaler, decomposer, clusterer)
    
    show_cluster_plot(clusters, data_decomposed, thumb_array)

def main2(ax):
    image_array = load_images_from_folder(r"test_images")
    thumb_array = get_thumbnails(image_array, (150, 100))
    img_data = get_sklearn_data(thumb_array)

    scaler, decomposer, clusterer = get_workers_from_config()

    clusters, data_decomposed = get_clusters(img_data, scaler, decomposer, clusterer)
    
    show_cluster_plot2(ax, clusters, data_decomposed, thumb_array)

if __name__ == '__main__':
    main()
