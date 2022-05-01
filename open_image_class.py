from PyQt5.QtWidgets import QMainWindow

class open_image_class(QMainWindow):
    def __init__(self):
        from PyQt5.uic import loadUi
        super(open_image_class, self).__init__()
        loadUi('open_image_app.ui',self)
        self.btnBrowse.clicked.connect(lambda :self.open_image())

    def open_file(self):
        from PyQt5 import QtWidgets, QtCore
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,'Open File',QtCore.QDir.rootPath(),'*.*')
        self.lblFilename.setText(fileName)

    def show_image(self, fileName):
        import qimage2ndarray
        from skimage import io
        from PyQt5 import QtGui
        try:
            image = io.imread(fileName)
            image = qimage2ndarray.array2qimage(image)
            qpixmap = QtGui.QPixmap.fromImage(image)
            self.lblIamge.setPixmap(qpixmap)
        except Exception as e:
            print('Error: {}'.format(e))


