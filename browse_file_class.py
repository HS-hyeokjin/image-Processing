from PyQt5.QtWidgets import *
from PyQt5 import uic

class browse_file_class(QMainWindow):
    def __init__(self):
        from PyQt5.uic import loadUi
        super(browse_file_class, self).__init__()
        loadUi('Browse_file_app.ui',self)
        self.btnBrowse.clicked.connect(lambda: self.open_file())

    def open_file(self):
        from PyQt5 import QtWidgets, QtCore
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,'Open File',QtCore.QDir.rootPath(),'*.*')
        self.lblFilename.setText(fileName)

def browse_file_app():
    app = QApplication([])
    window = browse_file_class()
    window.show()
    app.exec()

