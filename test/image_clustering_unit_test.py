import os
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import image_clustering as ic


class TestImageClustering:
    
    def test_load_images_from_folder(self):
        image_array = ic.load_images_from_folder('test_images')
        red_image = list(image_array)[0]
        assert type(red_image) == np.ndarray
        assert np.all(red_image[0, 0] == np.array([0, 0, 255]))
    
    def test_get_thumbnails(self):
        image_array = ic.load_images_from_folder('test_images')
        thumbs = ic.get_thumbnails(image_array, (50, 50))
        red_thumb = thumbs[0]
        assert type(thumbs) == np.ndarray
        assert np.all(red_thumb[0, 0] == np.array([1.0, 0.0, 0.0]))
        
    def test_get_sklearn_data(self):
        image_array = ic.load_images_from_folder('test_images')
        thumbs = ic.get_thumbnails(image_array, (50, 50))
        data = ic.get_sklearn_data(thumbs)
        m, n = data.shape
        assert (m, n) == (1, 7500)
        
    def test_get_workers_from_config(self):
        scaler, decomposer, clusterer = ic.get_workers_from_config()
        assert type(scaler) == Normalizer
        assert type(decomposer) == PCA
        assert decomposer.n_components == 15
        assert type(clusterer) == AgglomerativeClustering
        assert clusterer.n_clusters == 5
