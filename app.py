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
import image_clustering as imc


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
        super(ImageClusteringApp, self).__init__(parent)
        self.setupUi(self)

        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.matplotlibBaseLayout.addWidget(NavigationToolbar(self.static_canvas, self))
        self.matplotlibBaseLayout.addWidget(self.static_canvas)
        self._static_ax = self.static_canvas.figure.subplots()

        self.img_data = None
        self.thumb_array = None
        self.worker = None
        self.thread = None 

    def _handle_thumb_array(self, thumb_array):
        self.thumb_array = thumb_array

    def _handle_image_data(self, image_data):
        self.img_data = image_data
        
    def _get_folder_path(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select folder")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        selected_folder = file_dialog.getExistingDirectory()
        return selected_folder

    def on_select_folder_button_clicked(self):
        selected_folder = self._get_folder_path()
        if selected_folder != '':
            self.img_data = None
            self.thumb_array = None
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
        return selected_folder

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
            self._static_ax.cla()
            imc.show_cluster_plot2(self._static_ax, clusters, data_decomposed, self.thumb_array)
            self.static_canvas.draw()
        except Exception as e:
            self._show_warning_message("Image clustering failed. Please adjust input parameters. (Hint: {})".format(e))

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
            return
        config = {'scaler': scaler,
                  'decomposer': {'type': decomposer, 'components': components},
                  'clusterer': {'type': clusterer, 'n_clusters': n_clusters, 'dbscan_min': dbscan_min, 'dbscan_eps': dbscan_eps}}
        with open('image_clustering_config.yml', 'w') as hdl:
            hdl.write(yaml.dump(config))
        return config

    def _show_warning_message(self, text):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Warning")
        msg_box.setText(text)
        msg_box.exec()


def main():
    app = QApplication(sys.argv)
    form = ImageClusteringApp()
    form.show()
    app.exec()


if __name__ == '__main__':
    main()
