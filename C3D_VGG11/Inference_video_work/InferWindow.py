import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow

from C3D_VGG11.Inference_video_work.Inference_UI import Ui_MainWindow
from C3D_VGG11.Inference_video_work.ImageGet import ImageGet


class InferWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Inference_video')
        # 播放视频的定时器
        self.timer = QTimer()
        # 把视频逻辑封装成一个对象
        self.player = ImageGet(self.image, self.timer)  # 传 QLabel 给它
        # 连接定时器
        self.timer.timeout.connect(self.player.play_video_frame)   # type: ignore
        # 启动视频
        self.player.start_video(r"E:\pycharm_project\pytorch_001\Pytorch框架下的卷积神经网络库\C3D_VGG11\data_ucf_101\Archery\v_Archery_g01_c01.avi")  # 替换为你的视频文件路径

    def resizeEvent(self, event):
        """窗口大小改变时重新调整视频帧大小"""
        super().resizeEvent(event)
        # 用当前帧重新显示（不读取新帧！）
        # 通过player访问当前帧
        if hasattr(self.player, 'current_frame_rgb') and self.player.current_frame_rgb is not None:
            # 调用player的显示方法
            self.player.display_on_image(self.player.current_frame_rgb.copy())

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        if hasattr(self.player, 'video_cap') and self.player.video_cap is not None:
            self.player.video_cap.release()
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = InferWindow()
    Window.show()

    sys.exit(app.exec())



