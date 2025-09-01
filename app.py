import sys

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
import yaml

import gui.gui_form
import image_clustering as imc


class ImageClusteringApp(QtWidgets.QMainWindow, gui.gui_form.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ImageClusteringApp, self).__init__(parent)
        self.setupUi(self)
        
        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.matplotlibBaseLayout.addWidget(NavigationToolbar(self.static_canvas, self))
        self.matplotlibBaseLayout.addWidget(self.static_canvas)
        
        self._static_ax = self.static_canvas.figure.subplots()
        
        self.img_data = None
        self.thumb_array = None

    def on_select_folder_button_clicked(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select folder")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        selected_folder = None
        selected_folder = file_dialog.getExistingDirectory()
        if selected_folder is not None:
            self.selectFolderButton.setText("select folder (currently loading {}, please wait)".format(selected_folder))
            self.selectFolderButton.repaint()
            image_array = imc.load_images_from_folder(selected_folder)
            self.thumb_array = imc.get_thumbnails(image_array, (150, 100))
            self.img_data = imc.get_sklearn_data(self.thumb_array)
            self.selectFolderButton.setText("select folder (currently {})".format(selected_folder))
        return selected_folder

    def on_apply_button_clicked(self):
        if self.thumb_array is not None and self.img_data is not None:
            scaler = self.scalerComboBox.currentText()
            decomposer = self.decomposerComboBox.currentText()
            components = int(self.componentsLineEdit.text())
            clusterer = self.clustererComboBox.currentText()
            n_clusters = int(self.nClustersLineEdit.text())
            dbscan_min = int(self.dbscanMinLineEdit.text())
            dbscan_eps = int(self.dbscanEpsLineEdit.text())
            config = {'scaler': scaler,
                      'decomposer': {'type': decomposer, 'components': components},
                      'clusterer': {'type': clusterer, 'n_clusters': n_clusters, 'dbscan_min': dbscan_min, 'dbscan_eps': dbscan_eps}}
            with open('image_clustering_config.yml', 'w') as hdl:
                hdl.write(yaml.dump(config))
                
            scaler, decomposer, clusterer = imc.get_workers_from_config()
            clusters, data_decomposed = imc.get_clusters(self.img_data, scaler, decomposer, clusterer)
            
            self._static_ax.cla()
            imc.show_cluster_plot2(self._static_ax, clusters, data_decomposed, self.thumb_array)
            self.static_canvas.draw()
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Information")
            msg_box.setText("There is no data to analyse. Please select a folder first.")
            msg_box.exec()


def main():
    app = QApplication(sys.argv)
    form = ImageClusteringApp()
    form.show()
    app.exec()


if __name__ == '__main__':
    main()
