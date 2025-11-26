from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import shutil

import cv2
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
import yaml


SCALER_DICT = {'Standard': StandardScaler, 'Robust': RobustScaler, 'MinMax': MinMaxScaler, 'Normalizer': Normalizer}
DECOMPOSER_DICT = {'PCA': PCA, 'NMF': NMF, 'TSNE': TSNE}
CLUSTERER_DICT = {'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering, 'DBSCAN': DBSCAN}
FILE_TYPES = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg', 'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png'] 


class ImageClusteringConfiguration:

    def __init__(self, scaler, decomposer, clusterer):
        self.scaler = scaler
        self.decomposer = decomposer
        self.clusterer = clusterer

    @classmethod
    def from_config_file(cls):
        with open('image_clustering_config.yml') as hdl:
            conf = yaml.load(hdl, Loader=yaml.Loader)
        clusterer = cls._init_clusterer(conf)
        scaler = cls._init_scaler(conf)
        decomposer = cls._init_decomposer(conf)
        return cls(scaler, decomposer, clusterer)

    @staticmethod
    def _init_scaler(conf):
        scaler = conf['scaler']
        return None if scaler == 'None' else SCALER_DICT[scaler]()

    @staticmethod
    def _init_decomposer(conf):
        decomposer = conf['decomposer']
        if decomposer['type'] == 'TSNE':
            return DECOMPOSER_DICT['TSNE']()
        else:
            return DECOMPOSER_DICT[decomposer['type']](n_components=decomposer['components'])

    @staticmethod
    def _init_clusterer(conf):
        clusterer = conf['clusterer']
        if clusterer['type'] == 'DBSCAN':
            return DBSCAN(min_samples=clusterer['dbscan_min'], eps=clusterer['dbscan_eps'])
        else:
            return CLUSTERER_DICT[clusterer['type']](n_clusters=clusterer['n_clusters'])

def _load_images_pil(files, size):
    images = [Image.open(file) for file in files]
    _ = [img.draft('RGB', size) for img in images]
    with ThreadPoolExecutor() as executor:
        image_generator = executor.map(np.asarray, images)
    return image_generator

def _get_thumbnails_pil(image_generator, size):
    with ThreadPoolExecutor() as executor:
        thumbnail_generator = executor.map(lambda x: cv2.resize(x, size), image_generator)
    thumbs =  np.array(list(thumbnail_generator))
    thumbs = thumbs / 255
    return thumbs

def _load_images_cv(files):
    with ThreadPoolExecutor() as executor:
        image_generator = executor.map(cv2.imread, files)
    return image_generator
    
def _get_thumbnails_cv(image_generator, size):
    with ThreadPoolExecutor() as executor:
        thumbnail_generator = executor.map(lambda x: cv2.resize(x, size), image_generator)
    thumbs =  np.array(list(thumbnail_generator))
    thumbs = thumbs / 255
    thumbs = np.flip(thumbs, -1)
    return thumbs

def load_images_from_folder_pil(folder, size=(150, 100)):
    files = [file.path for file in os.scandir(folder) if os.path.isfile(file.path)]
    files = [path for path in files if path.split('.')[-1].lower() in FILE_TYPES]
    image_generator = _load_images_pil(files, size)
    thumbnails = _get_thumbnails_pil(image_generator, size)
    return thumbnails
    
def load_images_from_folder_cv(folder, size=(150, 100)):
    files = [file.path for file in os.scandir(folder) if os.path.isfile(file.path)]
    files = [path for path in files if path.split('.')[-1].lower() in FILE_TYPES]
    image_generator = _load_images_cv(files)
    thumbnails = _get_thumbnails_cv(image_generator, size)
    return thumbnails


def get_sklearn_data(img_arr):
    """
    Return an array of flattened images from an image array.
    """
    if len(img_arr) == 0:
        return np.array([])
    return img_arr.reshape((img_arr.shape[0], -1))


def get_clusters(data, scaler, decomposer, clusterer):
    """
    Perform scaling, dimensional reduction and cluster analysis on th given data.
    Return the array of cluster labels and the dimensionally reduced data.
    """
    data_scaled = scaler.fit_transform(data) if scaler is not None else data.copy()
    data_decomposed = decomposer.fit_transform(data_scaled)
    clusters = clusterer.fit_predict(data_decomposed)
    return clusters, data_decomposed

class ImageClusterPlotter:
    
    def __init__(self, ax):
        self.cmap = plt.get_cmap("gist_rainbow")
        self.frame_alpha = 1.0
        self.ax = ax

    def show_cluster_plot(self, cluster_labels, data, x, y, img_array, display_scale):
        """
        Visualise the result of a cluster analysis based on cluster labels,
        the dimensionally reduced data and the original image array
        and using the given matplotlib Axes.
        """
        self.ax.scatter(data[:, x], data[:, y], c=cluster_labels)  # needed to set plot range
        cmap = self.cmap
        colors = cmap(np.linspace(0, 1, max(cluster_labels) + 1))
        for i, img in enumerate(img_array):
            color = colors[cluster_labels[i]] if cluster_labels[i] != -1 else 'white'
            image_box = OffsetImage(img, zoom=display_scale)
            image_box.image.axes = self.ax
            props = {'facecolor': color, 'edgecolor': color, 'alpha': self.frame_alpha}
            ab = AnnotationBbox(image_box, (data[i, x], data[i, y]), bboxprops=props)
            ab.set(gid=cluster_labels[i], picker=True)
            self.ax.add_artist(ab)

    def show_cluster_plot_for_cluster(self, clusters, data, x, y, img_array, cluster_number, display_scale):
        """
        Visualise the result of a cluster analysis for on selected cluster,
        the dimensionally reduced data and the original image array
        and using the given matplotlib Axes.
        """
        data = data[clusters == cluster_number]
        img_array = img_array[clusters == cluster_number]
        self.ax.scatter(data[:, x], data[:, y])  # needed to set plot range
        for i, img in enumerate(img_array):
            image_box = OffsetImage(img, zoom=display_scale)
            image_box.image.axes = self.ax
            props = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': self.frame_alpha}
            ab = AnnotationBbox(image_box, (data[i, x], data[i, y]), bboxprops=props)
            ab.set(gid=i, picker=True)
            self.ax.add_artist(ab)
            
    def set_frame_alpha_value(self, alpha):
        abs = [e for e in self.ax.get_children() if type(e) == AnnotationBbox]
        for ab in abs:
            ab.update({'alpha': alpha})
            self.ax.draw_artist(ab)
        self.frame_alpha = alpha


def copy_files_by_clusters(folder, clusters):
    files = [file for file in os.scandir(folder) if os.path.isfile(file.path)]
    image_cluster_folder = os.path.join(folder, 'image_clusters_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(image_cluster_folder)
    indices = set(clusters)
    for index in indices:
        os.mkdir(os.path.join(image_cluster_folder, str(index)))
    for cluster, file in zip(clusters, files):
        shutil.copy2(file.path, os.path.join(image_cluster_folder, str(cluster), file.name))


if __name__ == '__main__':
    pass