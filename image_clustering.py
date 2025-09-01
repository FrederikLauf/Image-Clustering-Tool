import os
import numpy as np
from multiprocessing import Pool
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from concurrent.futures import ThreadPoolExecutor
import string
import yaml
import cv2

scaler_dict = {'Standard': StandardScaler, 'Robust': RobustScaler, 'MinMax': MinMaxScaler, 'Normalizer': Normalizer}
decomposer_dict = {'PCA': PCA, 'NMF': NMF, 'TSNE': TSNE}
clusterer_dict = {'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering, 'DBSCAN': DBSCAN}

FOLDER_PATH = r"test_images"

def load_images_from_folder(folder):
    files = [file.path for file in os.scandir(folder)]
    with ThreadPoolExecutor() as executor:
        image_array = executor.map(cv2.imread, files)
    return image_array
    
def get_thumbnails(image_array, size):
    thumbs = np.array([cv2.resize(img, size) for img in image_array])
    thumbs = thumbs / 255
    thumbs = np.flip(thumbs, -1)
    return thumbs

def get_sklearn_data(img_array):
    img_array = img_array.reshape((img_array.shape[0], -1))
    return img_array

def get_clusters(data, scaler, decomposer, clusterer):
    if scaler is not None:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.copy()
    data_decomposed = decomposer.fit_transform(data_scaled)
    clusters = clusterer.fit_predict(data_decomposed)
    return clusters, data_decomposed

def show_cluster_plot(clusters, data_decomposed, img_array):
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
    with open('image_clustering_config.yml') as hdl:
        conf = yaml.load(hdl, Loader=yaml.Loader)
    scaler, decomposer, clusterer = conf['scaler'], conf['decomposer'], conf['clusterer']
    if scaler == 'None':
        scaler = None
    else:
        scaler = scaler_dict[scaler]()
    if decomposer['type'] != 'TSNE':
        decomposer = decomposer_dict[decomposer['type']](n_components=decomposer['components'])
    else:
        decomposer = decomposer_dict['TSNE']()
    if clusterer['type'] != 'DBSCAN':
        clusterer = clusterer_dict[clusterer['type']](n_clusters=clusterer['n_clusters'])
    else:
        clusterer = DBSCAN(min_samples=clusterer['dbscan_min'], eps=clusterer['dbscan_eps'])

    return scaler, decomposer, clusterer

def main():
    image_array = load_images_from_folder(FOLDER_PATH)
    thumb_array = get_thumbnails(image_array, (150, 100))
    img_data = get_sklearn_data(thumb_array)

    scaler, decomposer, clusterer = get_workers_from_config()

    clusters, data_decomposed = get_clusters(img_data, scaler, decomposer, clusterer)
    
    show_cluster_plot(clusters, data_decomposed, thumb_array)
    
def main2(ax):
    image_array = load_images_from_folder(FOLDER_PATH)
    thumb_array = get_thumbnails(image_array, (150, 100))
    img_data = get_sklearn_data(thumb_array)

    scaler, decomposer, clusterer = get_workers_from_config()

    clusters, data_decomposed = get_clusters(img_data, scaler, decomposer, clusterer)
    
    show_cluster_plot2(ax, clusters, data_decomposed, thumb_array)
    
if __name__ == '__main__':
    main()

