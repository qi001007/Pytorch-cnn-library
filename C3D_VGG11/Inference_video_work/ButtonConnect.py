import cv2
from PyQt6.QtCore import QObject, QTimer

from globals import MW


# noinspection PyUnresolvedReferences
class ButtonConnect(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 定义防抖时钟
        self._skip_timer = QTimer(self)
        self._skip_timer.setSingleShot(True)
        self._skip_timer.timeout.connect(lambda: None)  # 占位，后面重连

        self.list_dir = 0
        self.video_path_list = MW.video_path_list
        self.current_video_dir = self.video_path_list[0]

    @staticmethod
    def video_start():
        if not MW.timer.isActive():
            MW.timer.start(30)

    @staticmethod
    def video_stop():
        if MW.timer.isActive():
            MW.timer.stop()

    @staticmethod
    def video_restart():
        MW.timer.stop()
        MW.player.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        MW.timer.start(30)

    def switch_video(self, index: int):
        """公共换视频接口：停播 → 换文件 → 回到第一帧 → 继续播"""
        MW.timer.stop()
        self.list_dir = index
        self.current_video_dir = self.video_path_list[index]
        # 让 ImageGet 重新 open 新视频
        MW.player.start_video(self.current_video_dir)  # 内部会重新 cv2.VideoCapture
        MW.timer.start(30)

    def _switch_debounced(self, step: int):
        """真正换视频，0.2 s 内只执行一次"""
        self.switch_video(max(0, min(len(self.video_path_list) - 1,
                                     self.list_dir + step)))

    def on_last(self):
        self._skip_timer.stop()          # 连续点/按住时重置
        self._skip_timer.timeout.disconnect()  # 清掉旧槽
        self._skip_timer.start(80)      # 100 ms 内只切一次
        self._skip_timer.timeout.connect(lambda: self._switch_debounced(-1))

    def on_next(self):
        self._skip_timer.stop()
        self._skip_timer.timeout.disconnect()  # 清掉旧槽
        self._skip_timer.start(80)
        self._skip_timer.timeout.connect(lambda: self._switch_debounced(1))
