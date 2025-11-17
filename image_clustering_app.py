import os
import sys

import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.backends.qt_compat import QtWidgets
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow
import yaml

import gui.gui_form
import image_clustering.image_clustering as imc


FILE_TYPES = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg', 'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png'] 


class ImageLoader(QObject):

    finished = pyqtSignal()
    thumb_array = pyqtSignal(np.ndarray)
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, selected_folder, image_scaling):
        super().__init__()
        self.selected_folder = selected_folder
        self.image_scaling = image_scaling

    def load_image_data(self):
        m = self.image_scaling
        n = int(2 * m / 3)
        thumb_array = imc.load_images_from_folder_pil(self.selected_folder, (m, n))
        image_data = imc.get_sklearn_data(thumb_array)
        self.thumb_array.emit(thumb_array)
        self.image_data.emit(image_data)
        self.finished.emit()


class ImageClusteringApp(QMainWindow, gui.gui_form.Ui_MainWindow):

    def __init__(self, parent=None):
        # init ui
        super(ImageClusteringApp, self).__init__(parent)
        self.setupUi(self)
        # init canvas
        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.static_canvas.mpl_connect('pick_event', self._on_cluster_representative_clicked)
        self.matplotlibBaseLayout.addWidget(NavigationToolbar(self.static_canvas, self))
        self.matplotlibBaseLayout.addWidget(self.static_canvas)
        self._static_ax = self.static_canvas.figure.subplots()
        # init data
        self.selected_folder = None
        self.current_cluster_labels = None
        self.data_decomposed = None
        self.img_data = None
        self.thumb_array = None
        self.worker = None
        self.thread = None
        # init state
        self.display_state = ['all', None]

    # ---------utility methods------------------------------------------------------------------
    
    def _deinit_data_and_view(self):
        self._static_ax.cla()
        self.static_canvas.draw()
        self.current_cluster_labels = None
        self.img_data = None
        self.thumb_array = None
        self.display_state = ['all', None]

    def _show_warning_message(self, text):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Warning")
        msg_box.setText(text)
        msg_box.exec()

    def _get_folder_path(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select folder")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        selected_folder = file_dialog.getExistingDirectory()
        if selected_folder == '':
            return None
        return selected_folder

    def _make_config_from_input(self):
        try:
            scaler = self.scalerComboBox.currentText()
            decomposer = self.decomposerComboBox.currentText()
            components = int(self.componentsLineEdit.text())
            clusterer = self.clustererComboBox.currentText()
            n_clusters = int(self.nClustersLineEdit.text())
            dbscan_min = int(self.dbscanMinLineEdit.text())
            dbscan_eps = float(self.dbscanEpsLineEdit.text())
            image_scaling = int(self.preScalingLabel.text())
        except ValueError as e:
            self._show_warning_message("Input values could not be interpreted. Please enter valid parameters. (Hint: {})".format(e))
            return None
        config = {'image_scaling': image_scaling,
                  'scaler': scaler,
                  'decomposer': {'type': decomposer, 'components': components},
                  'clusterer': {'type': clusterer, 'n_clusters': n_clusters, 'dbscan_min': dbscan_min, 'dbscan_eps': dbscan_eps}}
        with open('image_clustering_config.yml', 'w') as hdl:
            hdl.write(yaml.dump(config))
        return config
        
    # ----------plotting methods----------------------------------------------------------------

    def _plot_all_clusters(self):
        if all(i is not None for i in (self.current_cluster_labels, self.data_decomposed, self.thumb_array)):
            x = self.xDimensionScrollbar.value()
            y = self.yDimensionScrollbar.value()
            display_scale = float(self.displayScalingLabel.text())
            self._static_ax.cla()
            imc.show_cluster_plot2(self._static_ax,
                                   self.current_cluster_labels, self.data_decomposed,
                                   x, y, self.thumb_array, display_scale)
            self.static_canvas.draw()

    def _plot_single_cluster(self, cluster_number):
        if all(i is not None for i in (self.current_cluster_labels, self.data_decomposed, self.thumb_array)):
            x = self.xDimensionScrollbar.value()
            y = self.yDimensionScrollbar.value()
            display_scale = float(self.displayScalingLabel.text())
            self._static_ax.cla()
            imc.show_cluster_plot_for_cluster(self._static_ax,
                                              self.current_cluster_labels, self.data_decomposed,
                                              x, y, self.thumb_array, cluster_number, display_scale)
            self.static_canvas.draw()

    def _plot_cluster_view(self):
        if self.display_state[0] == 'all':
            self._plot_all_clusters()
        elif self.display_state[0] == 'single':
            self._plot_single_cluster(self.display_state[1])

    # #---------callback methods----------------------------------------------------------------
    
    # ---------matplotlib-----------------------------------------------------------------------
    
    def _on_cluster_representative_clicked(self, event):
        if self.display_state[0] == 'all':
            thumb = event.artist
            cluster = thumb.get_gid()
            self._plot_single_cluster(cluster)
            self.display_state = ['single', cluster]
        elif self.display_state[0] == 'single':
            cluster = self.display_state[1]
            thumb = event.artist
            i = thumb.get_gid()
            files = [file.path for file in os.scandir(self.selected_folder) if os.path.isfile(file.path)]
            files = [path for path in files if path.split('.')[-1].lower() in FILE_TYPES]
            filtered_files = [e[1] for e in enumerate(files) if self.current_cluster_labels[e[0]] == cluster]
            image = cv2.imread(filtered_files[i])
            cv2.namedWindow("Image_" + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow("Image_" + str(i), image)
            
    # -------Qt---------------------------------------------------------------------------------

    def on_select_folder_button_clicked(self):
        selected_folder = self._get_folder_path()  # TEST empty folder
        if selected_folder is not None:
            self._deinit_data_and_view()
            self.selected_folder = selected_folder
            # initiate loading of images
            self.selectFolderButton.setEnabled(False)
            self.selectFolderButton.setText("select folder (currently loading {}, please wait)".format(selected_folder))
            self.selectFolderButton.repaint()
            image_scaling = int(self.preScalingLabel.text())
            self.worker = ImageLoader(selected_folder, image_scaling)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.load_image_data)
            self.worker.thumb_array.connect(lambda data: setattr(self, 'thumb_array', data))
            self.worker.image_data.connect(lambda data: setattr(self, 'img_data', data))
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(lambda: self.selectFolderButton.setEnabled(True))
            self.thread.finished.connect(lambda: self.selectFolderButton.setText("select folder (currently {})".format(selected_folder)))
            self.thread.start()

    def on_export_button_clicked(self):
        if self.selected_folder != '' and self.current_cluster_labels is not None:
            imc.copy_files_by_clusters(self.selected_folder, self.current_cluster_labels)
        else:
            self._show_warning_message("No clusters to export. Please select an image folder and apply clustering first.")

    def on_apply_button_clicked(self):
        self.display_state = ['all', None]
        if self.thumb_array is None or self.img_data is None:
            self._show_warning_message("No data to analyse. Please select an image folder first.")
            return
        config = self._make_config_from_input()
        if config is None:
            return
        try:
            icc = imc.ImageClusteringConfiguration.from_config_file()
            clusters, data_decomposed = imc.get_clusters(self.img_data, icc.scaler, icc.decomposer, icc.clusterer)
            self.current_cluster_labels = clusters
            self.data_decomposed = data_decomposed
            components = len(self.data_decomposed[0])
            self.xDimensionScrollbar.setValue(0)
            self.yDimensionScrollbar.setValue(1)
            self.xDimensionScrollbar.setMaximum(components - 1)
            self.yDimensionScrollbar.setMaximum(components - 1)
            self._plot_all_clusters()
        except Exception as e:
            self._show_warning_message("Image clustering failed. Please adjust input parameters. (Hint: {})".format(e))

    def on_x_dimension_scrollbar_changed(self, new_value):
        self.xDimensionLabel.setText(str(new_value))
        self._plot_cluster_view()

    def on_y_dimension_scrollbar_changed(self, new_value):
        self.yDimensionLabel.setText(str(new_value))
        self._plot_cluster_view()

    def on_display_scaling_slider_changed(self, new_value):
        self.displayScalingLabel.setText(str(new_value / 100))
        self._plot_cluster_view()

    def on_prescaling_slider_changed(self, new_value):
        self.preScalingLabel.setText(str(new_value))


def main():
    app = QApplication(sys.argv)
    form = ImageClusteringApp()
    form.show()
    app.exec()


if __name__ == '__main__':
    main()
