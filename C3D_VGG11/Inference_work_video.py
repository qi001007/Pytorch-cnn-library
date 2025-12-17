import cv2
import sys
from PyQt6.QtWidgets import QApplication, QWidget
import numpy as np


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Inference_video')
        self.resize(640, 480)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = MainWindow()
    Window.show()

    sys.exit(app.exec())



