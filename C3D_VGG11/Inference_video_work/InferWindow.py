import sys

from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

from Inference_UI import Ui_MainWindow
from DataRead import DataRead
from ImageGet import ImageGet
from ButtonConnect import ButtonConnect
from inference import Inference
from FileSelect import FileSelect

from globals import MW
from C3D_VGG11.model import C3D_VGG11


class InferWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, model, clip_num):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Inference_video')

        # 接收外界参数
        MW.model = model
        MW.clip_num = clip_num
        MW.FILE_FIELDS = ['file_dir', 'model_dir', 'data_label_dir', 'wts_dir']
        MW.DIR_FIELDS = ['data_dir']
        self.PARAMS_MAP = {'model_dir': Path(),
                           'file_dir': Path(),
                           'data_dir': Path(),
                           'data_label_dir': Path(),
                           'wts_dir': Path()}
        MW.PARAMS_MAP = self.PARAMS_MAP
        self.startEvent()

        # 接收Ui_MainWindow控件
        MW.image = self.image
        MW.res = self.res
        MW.pre = self.pre
        MW.data_path = self.data_path
        MW.data_label_path = self.data_label_path
        MW.wts_path = self.wts_path
        MW.model_path = self.model_path
        MW.file_path = self.file_path

        # 工具类
        # 播放视频的定时器
        self.timer = QTimer()
        MW.timer = self.timer
        # 文件选择工具
        self.file_selecter = FileSelect(parent=self)
        MW.file_selecter = self.file_selecter
        # 获取文件工具
        self.data_reader = DataRead()
        MW.data_reader = self.data_reader
        MW.video_path_list = self.data_reader.get_video_path_list
        # 推理工具
        self.inferer = Inference()
        MW.inferer = self.inferer
        # 把视频逻辑封装成一个对象,获取视频帧工具
        self.player = ImageGet()  # 传 QLabel 给它
        MW.player = self.player
        # 连接定时器
        self.timer.timeout.connect(self.player.play_video_frame)   # type: ignore
        # 按钮链接函数工具
        self.button = ButtonConnect(parent=self)
        MW.button = self.button
        MW.list_dir = self.button.list_dir

        # 按钮
        # 定义需要长按功能按键(注意：连发间隔 ＞ 防抖定时器时长)
        self.last.setAutoRepeat(True)  # 按住自动重复发 clicked
        self.last.setAutoRepeatDelay(300)  # 首次等 300 ms
        self.last.setAutoRepeatInterval(100)  # 之后每 150 ms 发一次
        self.next.setAutoRepeat(True)  # 按住自动重复发 clicked
        self.next.setAutoRepeatDelay(300)  # 首次等 300 ms
        self.next.setAutoRepeatInterval(100)  # 之后每 150 ms 发一次
        # 播放按钮
        self.start.clicked.connect(self.button.video_start)
        self.stop.clicked.connect(self.button.video_stop)
        self.restart.clicked.connect(self.button.video_restart)
        # 切换视频按钮(换视频就关掉上一个的推理)(两个函数顺序执行 → 包成 lambda 或自定义槽函数)(一次点击触发多个槽，就分别 connect )
        self.last.clicked.connect(lambda: (self.button.on_last(),
                                           self.inferer.infer_button_state(),
                                           self.inferer.show_infer(),
                                           self.file_selecter.show_dir()))
        self.next.clicked.connect(lambda: (self.button.on_next(),
                                           self.inferer.infer_button_state(),
                                           self.inferer.show_infer(),
                                           self.file_selecter.show_dir()))
        # 推理按钮
        self.infer.clicked.connect(self.inferer.infer_button)
        self.global_infer.clicked.connect(lambda: (self.inferer.global_infer_button(),
                                                   self.inferer.infer_button_state()))
        # 文件路径选择按钮
        self.model_dir.clicked.connect(lambda: self.file_selecter.dir_select('model_dir'))
        self.file_dir.clicked.connect(lambda: self.file_selecter.dir_select('file_dir'))
        self.data_dir.clicked.connect(lambda: self.file_selecter.dir_select('data_dir'))
        self.data_label_dir.clicked.connect(lambda: self.file_selecter.dir_select('data_label_dir'))
        self.wts_dir.clicked.connect(lambda: self.file_selecter.dir_select('wts_dir'))

        # 输入框
        # self.model_path.editingFinished.connect(lambda: self.file_selecter.get_dir('model_dir'))
        self.file_path.editingFinished.connect(lambda: self.file_selecter.get_dir('file_dir'))

    def startEvent(self):
        def pick_file(owner, title="选择文件", start_dir="", filters="所有文件 (*.*)"):
            start = str(start_dir or Path.cwd())
            path, _ = QFileDialog.getOpenFileName(
                owner, title, start, filters
            )
            return Path(path) if path else Path()

        def pick_dir(owner, title="选择目录", start_dir=""):
            start = str(start_dir or Path.cwd())
            path = QFileDialog.getExistingDirectory(
                owner, title, start
            )
            return Path(path) if path else Path()

        for key, path_obj in self.PARAMS_MAP.items():
            if not path_obj or path_obj == Path():  # 空路径
                # 根据字段类型决定是选择文件还是目录
                if key in MW.FILE_FIELDS:
                    new_path = pick_file(owner=self, title=f"选择 {key}")
                    print(new_path)
                elif key in MW.DIR_FIELDS:
                    new_path = pick_dir(owner=self, title=f"选择 {key}")
                    print(new_path)
                else:
                    # 默认当作文件处理
                    new_path = pick_file(owner=self, title=f"选择 {key}")
                    print(new_path)

                if new_path and new_path.exists():  # 用户确实选了
                    self.PARAMS_MAP[key] = new_path
                else:  # 取消
                    self.PARAMS_MAP[key] = Path()

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
    Model = C3D_VGG11(num_classes=101)
    Window = InferWindow(model=Model, clip_num=16)
    Window.show()

    sys.exit(app.exec())
