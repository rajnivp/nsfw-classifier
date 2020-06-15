# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'images.ui'

# Created by: PyQt5 UI code generator 5.14.2


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QFileDialog
from category.test import predict, load_model
import os

mapper = {0: "drawing-SFW", 1: "hentai-NSFW", 2: "neutral-SFW", 3: "porn-NSFW", 4: "sexy-SFW"}


class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self.image_number = 0
        self.directory = "category/test/test"
        self.imagelist = os.listdir(self.directory)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.image_view = QtWidgets.QLabel(self.centralwidget)
        self.image_view.setGeometry(QtCore.QRect(100, 20, 600, 400))
        self.image_view.setAlignment(QtCore.Qt.AlignCenter)
        self.image_view.setObjectName("image_view")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 425, 250, 60))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setGeometry(QtCore.QRect(600, 500, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.nextButton.setFont(font)
        self.nextButton.setObjectName("nextButton")

        self.prevButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevButton.setGeometry(QtCore.QRect(100, 500, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.prevButton.setFont(font)
        self.prevButton.setObjectName("prevButton")

        self.predictButton = QtWidgets.QPushButton(self.centralwidget)
        self.predictButton.setGeometry(QtCore.QRect(350, 500, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.predictButton.setFont(font)
        self.predictButton.setObjectName("predictButton")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)
        self.actionFile = QtWidgets.QAction(MainWindow)
        self.actionFile.setObjectName("actionFile")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.actionOpen.triggered.connect(self.setExistingDirectory)
        self.nextButton.clicked.connect(self.next_button)
        self.prevButton.clicked.connect(self.prev_button)
        self.predictButton.clicked.connect(self.predict)
        self.showimage()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Classifier"))

        self.image_view.setText(_translate("MainWindow", "Image"))
        self.label.setText(_translate("MainWindow", ""))
        self.nextButton.setText(_translate("MainWindow", "Next"))
        self.prevButton.setText(_translate("MainWindow", "Prev"))
        self.predictButton.setText(_translate("MainWindow", "Predict"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionFile.setText(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))

    def predict(self):
        model = load_model()
        pred, (probs, classes) = predict(self.image_path, model)
        pred, probs = mapper[pred], max(probs)
        self.label.setText("{}({}%)".format(pred, round(probs * 100, 0)))
        self.label.adjustSize()

    def next_button(self):
        self.image_number += 1
        self.label.setText("")
        if self.image_number >= len(self.imagelist):
            self.image_number = 0
        self.showimage()

    def prev_button(self):
        self.image_number -= 1
        self.label.setText("")
        if abs(self.image_number) > len(self.imagelist):
            self.image_number = -1
        self.showimage()

    def showimage(self):

        self.image_path = self.directory + '/' + self.imagelist[self.image_number]
        pixmap = QPixmap(self.image_path)

        self.image_view.setPixmap(pixmap)
        # self.image_view.resize(pixmap.width(), pixmap.height())
        # self.image_view.adjustSize()

    def setExistingDirectory(self):
        self.dialog = QDialog()
        options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
        self.file = QFileDialog.getOpenFileName(self.dialog, "Open Folder", options=options)
        self.directory = "/".join(self.file[0].split('/')[0:-1])
        self.file = self.file[0].split('/')[-1]
        self.imagelist = os.listdir(self.directory)
        self.image_number = self.imagelist.index(self.file)
        self.label.setText("")

        self.showimage()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
