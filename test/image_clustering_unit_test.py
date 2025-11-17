import os
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import image_clustering.image_clustering as ic


class TestImageClustering:
    
    def test_load_images_from_folder_pil(self):
        thumbnail_array = ic.load_images_from_folder_pil('test_data', (200, 100))
        red_image = thumbnail_array[0]
        assert type(red_image) == np.ndarray
        assert red_image.shape == (100, 200, 3)
        assert np.all(red_image[0, 0] == np.array([1.0, 0.0, 0.0]))
        
    def test_load_images_from_folder_cv(self):
        thumbnail_array = ic.load_images_from_folder_cv('test_data', (200, 100))
        red_image = thumbnail_array[0]
        assert type(red_image) == np.ndarray
        assert red_image.shape == (100, 200, 3)
        assert np.all(red_image[0, 0] == np.array([1.0, 0.0, 0.0]))
        
    def test_get_sklearn_data(self):
        thumbs = ic.load_images_from_folder_pil('test_data', (200, 100))
        data = ic.get_sklearn_data(thumbs)
        assert data.shape == (1, 60000)
        
    def test_get_workers_from_config(self):
        icc = ic.ImageClusteringConfiguration.from_config_file()
        scaler, decomposer, clusterer = icc.scaler, icc.decomposer, icc.clusterer
        assert type(scaler) == Normalizer
        assert type(decomposer) == PCA
        assert decomposer.n_components == 15
        assert type(clusterer) == AgglomerativeClustering
        assert clusterer.n_clusters == 5
