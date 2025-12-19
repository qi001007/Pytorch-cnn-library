import cv2
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap, QImage

from globals import MW


class ImageGet:
    def __init__(self):
        # 视频相关变量
        self.video_cap = None  # 视频捕获对象
        self.current_frame_rgb = None
        self.image = MW.image
        # 允许QLabel缩放内容(使视频随缩放可动)
        self.image.setScaledContents(True)
        # 启动视频
        self.start_video(MW.PARAMS_MAP['file_dir'])  # 替换为你的视频文件路径

    def aspect_ratio_preserving(self, frame):
        # 获取QLabel的当前大小
        lab_w = self.image.width()
        lab_h = self.image.height()
        # 获取frame的当前大小
        h, w = frame.shape[:2]
        # 计算等比例尺寸
        scale = min(lab_w / w, lab_h / h)
        new_size = QSize(int(w * scale), int(h * scale))
        return lab_w, lab_h, new_size

    def start_video(self, video_path):
        # 初始打印
        MW.inferer.show_infer(),
        MW.file_selecter.show_dir()
        """开始播放视频"""
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            print("无法打开视频文件")
            return
        # 开始播放（30毫秒一帧，约33fps）
        MW.timer.start(30)

    def play_video_frame(self):
        """播放视频的每一帧"""
        if self.video_cap is None:
            print('还未创建VideoCapture')
            return
        ret, frame = self.video_cap.read()
        total_frames_num = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_num = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 推理
        MW.inferer.infer(ret, frame, current_frame_num)
        if ret:
            # 将BGR转换为RGB
            self.current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 显示在image上
            self.display_on_image(self.current_frame_rgb)
        else:
            # 帧获取失败，可能是视频结束或其他错误
            # 检查视频是否真的结束了
            if current_frame_num >= total_frames_num - 1:
                # 视频正常结束，循环播放
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # 非正常结束，跳过错过的帧
                print(f"帧获取失败，当前位置: {current_frame_num}")
                # 跳过当前帧，继续下一帧
                pass

    def display_on_image(self, frame):
        """将帧显示在image QLabel上，支持缩放变形"""
        lab_w, lab_h, new_size = self.aspect_ratio_preserving(frame)
        # 避免视频没地方显示
        if lab_w > 0 and lab_h > 0:
            # 先创建副本，避免修改输入数组
            # 调整帧大小以匹配QLabel（这里会变形）
            resize_frame = cv2.resize(frame, (new_size.width()-3, new_size.height()-3),
                                      interpolation=cv2.INTER_AREA)
            # 将numpy数组转换为QImage
            height, width, channels = resize_frame.shape
            bytes_per_line = channels * width
            # 转化为QImage的数据类型
            q_image = QImage(
                resize_frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            self.image.setPixmap(pixmap)
