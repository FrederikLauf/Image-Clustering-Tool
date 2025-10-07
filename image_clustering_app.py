import sys

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow
import yaml

import gui.gui_form
import image_clustering.image_clustering as imc


class ImageLoader(QObject):

    finished = pyqtSignal()
    thumb_array = pyqtSignal(np.ndarray)
    image_data = pyqtSignal(np.ndarray)
    
    def __init__(self, selected_folder):
        super().__init__()
        self.selected_folder = selected_folder

    def load_image_data(self):
        image_array = imc.load_images_from_folder(self.selected_folder)
        thumb_array = imc.get_thumbnails(image_array, (150, 100))
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

    # ---------utility methods-----------------------------------------------------------------

    def _handle_thumb_array(self, thumb_array):
        self.thumb_array = thumb_array

    def _handle_image_data(self, image_data):
        self.img_data = image_data
    
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
        except ValueError as e:
            self._show_warning_message("Input values could not be interpreted. Please enter valid parameters. (Hint: {})".format(e))
            return None
        config = {'scaler': scaler,
                  'decomposer': {'type': decomposer, 'components': components},
                  'clusterer': {'type': clusterer, 'n_clusters': n_clusters, 'dbscan_min': dbscan_min, 'dbscan_eps': dbscan_eps}}
        with open('image_clustering_config.yml', 'w') as hdl:
            hdl.write(yaml.dump(config))
        return config

    def _plot_clusters(self):
        if all(i is not None for i in (self.current_cluster_labels, self.data_decomposed, self.thumb_array)):
            x = self.xDimensionScrollbar.value()
            y = self.yDimensionScrollbar.value()
            self._static_ax.cla()
            imc.show_cluster_plot2(self._static_ax,
                                   self.current_cluster_labels, self.data_decomposed,
                                   x, y, self.thumb_array)
            self.static_canvas.draw()

    # ---------callback methods-----------------------------------------------------------------

    def on_select_folder_button_clicked(self):
        selected_folder = self._get_folder_path()  # TEST empty folder
        if selected_folder is not None:
            self.selected_folder = selected_folder
            # erase plot and data
            self._static_ax.cla()
            self.static_canvas.draw()
            self.current_cluster_labels = None
            self.img_data = None
            self.thumb_array = None
            # initiate loading of images
            self.selectFolderButton.setEnabled(False)
            self.selectFolderButton.setText("select folder (currently loading {}, please wait)".format(selected_folder))
            self.selectFolderButton.repaint()
            self.worker = ImageLoader(selected_folder)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.load_image_data)
            self.worker.thumb_array.connect(self._handle_thumb_array)
            self.worker.image_data.connect(self._handle_image_data)
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
        if self.thumb_array is None or self.img_data is None:
            self._show_warning_message("No data to analyse. Please select an image folder first.")
            return
        config = self._make_config_from_input()
        if config is None:
            return
        try:
            scaler, decomposer, clusterer = imc.get_workers_from_config()
            clusters, data_decomposed = imc.get_clusters(self.img_data, scaler, decomposer, clusterer)
            self.current_cluster_labels = clusters
            self.data_decomposed = data_decomposed
            components = len(self.data_decomposed[0])
            self.xDimensionScrollbar.setValue(0)
            self.yDimensionScrollbar.setValue(1)
            self.xDimensionScrollbar.setMaximum(components - 1)
            self.yDimensionScrollbar.setMaximum(components - 1)
            self._plot_clusters()
        except Exception as e:
            self._show_warning_message("Image clustering failed. Please adjust input parameters. (Hint: {})".format(e))

    def on_x_dimension_scrollbar_changed(self, new_value):
        self.xDimensionLabel.setText(str(new_value))
        self._plot_clusters()
        
    def on_y_dimension_scrollbar_changed(self, new_value):
        self.yDimensionLabel.setText(str(new_value))
        self._plot_clusters()


def main():
    app = QApplication(sys.argv)
    form = ImageClusteringApp()
    form.show()
    app.exec()


if __name__ == '__main__':
    main()
