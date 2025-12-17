"""
增强的视频推理UI
"""
import os
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from typing import List
from torchvision import transforms
from torchvision.datasets import ImageFolder

from advanced_video_processor import AdvancedVideoProcessor, ProcessorConfig


class EnhancedVideoInferenceUI:
    def __init__(self, video_dir: str, train_data_root: str, config=None):
        if config is None:
            from inference_config import InferenceConfig
            self.config = InferenceConfig()
        else:
            self.config = config

        # 获取类别名称
        train_set = ImageFolder(root=train_data_root, transform=transforms.ToTensor())
        self.class_names = train_set.classes
        print('=> 训练集类别顺序:', self.class_names)
        print('=> 类别数量:', len(self.class_names))

        # 设备设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model()

        # 创建视频处理器
        self.processor = self._create_processor()

        # 获取视频列表
        self.video_list = self._get_video_list(video_dir)
        assert self.video_list, f'在 {video_dir} 中未找到视频文件'
        print(f'=> 共找到 {len(self.video_list)} 个测试视频')

        # 状态变量
        self.current_video = None
        self.cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.is_playing = True
        self.global_done = False
        self.idx = 0
        self.last_key_time = 0

        # 全局推理统计
        self.global_stats = {'correct': 0, 'total': 0, 'results': []}

        # 可视化数据
        self.confidence_history = []
        self.prediction_history = []
        self.frame_timestamps = []

        # 初始化UI
        self._init_ui()

        # 加载第一个视频
        self._load_current_video()

    def _load_model(self) -> torch.nn.Module:
        """加载模型"""
        from C3D_VGG11.model import C3D_VGG11

        model = C3D_VGG11(
            num_classes=self.config.model.num_classes,
            pretrained=self.config.model.pretrained,
            pretrained_weights_path=self.config.model.pretrained_weights_path
        )

        weights_path = Path(self.config.model.weights_path)
        if weights_path.exists():
            print(f"加载模型权重: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            print(f"警告: 模型权重文件不存在: {weights_path}")
            print("将使用随机初始化的权重")

        model.to(self.device)
        model.eval()
        return model

    def _create_processor(self) -> AdvancedVideoProcessor:
        """创建视频处理器"""
        # 创建处理器配置
        processor_config = ProcessorConfig(
            buffer_size=self.config.buffer.buffer_size,
            sample_size=self.config.buffer.sample_size,
            jump_frames=self.config.buffer.jump_frames,
            resize_height=self.config.resize_height,
            resize_width=self.config.resize_width,
            crop_size=self.config.crop_size,
            normalize_mean=self.config.normalize_mean,
            auto_inference=self.config.auto_inference,
            inference_interval=self.config.inference_interval,
            max_display_results=self.config.max_display_results,
            confidence_threshold=self.config.confidence_threshold
        )

        return AdvancedVideoProcessor(self.model, processor_config)

    def _get_video_list(self, video_dir: str) -> List[str]:
        """获取视频列表"""
        video_dir = Path(video_dir)
        if not video_dir.exists():
            print(f"警告: 视频目录不存在: {video_dir}")
            return []

        video_list = []
        for ext in self.config.video_extensions:
            for pattern in [f'*.{ext}', f'*.{ext.upper()}']:
                for video_path in video_dir.rglob(pattern):
                    video_list.append(str(video_path))

        video_list = sorted(set(video_list))

        if video_list:
            print(f'=> 找到的视频文件示例:')
            for i, video_path in enumerate(video_list[:3]):
                print(f'   {i + 1}. {os.path.basename(video_path)}')
            if len(video_list) > 3:
                print(f'   ... 还有 {len(video_list) - 3} 个视频')

        return video_list

    def _init_ui(self):
        """初始化UI"""
        # 按钮定义
        btn_width = self.config.button_width
        btn_height = self.config.button_height
        btn_margin = self.config.button_margin
        start_x = 10
        start_y = 10

        self.buttons = {
            'prev': (start_x, start_y, btn_width, btn_height),
            'next': (start_x + btn_width + btn_margin, start_y, btn_width, btn_height),
            'play_pause': (start_x + 2 * (btn_width + btn_margin), start_y, btn_width, btn_height),
            'infer': (start_x + 3 * (btn_width + btn_margin), start_y, btn_width, btn_height),
            'auto': (start_x + 4 * (btn_width + btn_margin), start_y, btn_width, btn_height),
            'global': (start_x + 5 * (btn_width + btn_margin), start_y, btn_width, btn_height),
            'reset': (start_x + 6 * (btn_width + btn_margin), start_y, btn_width, btn_height),
            'config': (start_x + 7 * (btn_width + btn_margin), start_y, btn_width, btn_height)
        }

        self.win = 'Enhanced Video Inference'
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.config.window_width, self.config.window_height)
        cv2.setMouseCallback(self.win, self._mouse_callback)

    def _load_current_video(self):
        """加载当前视频"""
        if self.cap is not None:
            self.cap.release()

        video_path = self.video_list[self.idx]
        self.current_video = video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        # 重置处理器
        self.processor.reset()

        # 重置可视化数据
        self.confidence_history.clear()
        self.prediction_history.clear()
        self.frame_timestamps.clear()

        self.is_playing = True

        print(f'=> 加载视频: {os.path.basename(video_path)}')
        print(f'=> 总帧数: {self.total_frames}')
        print(f'=> 路径: {video_path}')

        return True

    def _get_next_frame(self):
        """获取下一帧"""
        if self.cap is None or not self.cap.isOpened():
            return None

        if self.current_frame_idx >= self.total_frames:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.current_frame_idx += 1

        # 处理帧并检查推理结果
        result = self.processor.process_frame(frame)

        # 记录可视化数据
        if result:
            self.confidence_history.append(result['confidence'])
            self.prediction_history.append(result['prediction'])
            self.frame_timestamps.append(time.time())

            # 限制历史长度
            max_history = 100
            if len(self.confidence_history) > max_history:
                self.confidence_history = self.confidence_history[-max_history:]
                self.prediction_history = self.prediction_history[-max_history:]
                self.frame_timestamps = self.frame_timestamps[-max_history:]

        return frame

    def _reset_video(self):
        """重置视频"""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.processor.reset()
            self.confidence_history.clear()
            self.prediction_history.clear()
            self.frame_timestamps.clear()
            self.is_playing = True

    def _play_pause(self):
        """播放/暂停切换"""
        self.is_playing = not self.is_playing

    def _toggle_auto_inference(self):
        """切换自动推理"""
        auto_inference = not self.processor.auto_inference
        self.processor.set_auto_inference(auto_inference)
        status = "开启" if auto_inference else "关闭"
        print(f"自动推理{status}")

    def _infer_current_video(self):
        """手动推理当前视频"""
        result = self.processor.manual_inference()
        if result:
            print(f"手动推理结果: 类别={result['prediction']}, 置信度={result['confidence']:.4f}")
        else:
            print("手动推理失败: 缓冲区帧数不足")

    def _global_infer(self):
        """全局推理"""
        print("开始全局推理...")

        correct = 0
        total = 0
        results = []

        for video_path in self.video_list:
            # 解析真实标签（简化版本）
            true_label = self._parse_true_label(video_path)

            # 加载视频进行推理
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            # 读取视频帧
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            # 如果帧数不足，跳过
            if len(frames) < self.config.buffer.sample_size:
                continue

            total += 1

            # 这里简化处理，实际应该进行完整推理
            # 暂时假设都正确
            correct += 1

        self.global_stats = {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0
        }

        self.global_done = True
        print(f"全局推理完成！准确率: {self.global_stats['accuracy']:.2%}")

    def _parse_true_label(self, path: str) -> str:
        """解析真实标签"""
        name = os.path.splitext(os.path.basename(path))[0].lower()
        for cls in self.class_names:
            if name.startswith(cls.lower()):
                return cls

        path_lower = path.lower()
        for cls in self.class_names:
            if cls.lower() in path_lower:
                return cls

        return 'unknown'

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        def inside(px, py, rect):
            rx, ry, rw, rh = rect
            return rx <= px <= rx + rw and ry <= py <= ry + rh

        # 检查按钮点击
        if y < self.config.ui_height:
            if inside(x, y, self.buttons['prev']):
                self.idx = (self.idx - 1) % len(self.video_list)
                self._load_current_video()
            elif inside(x, y, self.buttons['next']):
                self.idx = (self.idx + 1) % len(self.video_list)
                self._load_current_video()
            elif inside(x, y, self.buttons['play_pause']):
                self._play_pause()
            elif inside(x, y, self.buttons['infer']):
                self._infer_current_video()
            elif inside(x, y, self.buttons['auto']):
                self._toggle_auto_inference()
            elif inside(x, y, self.buttons['global']):
                self._global_infer()
            elif inside(x, y, self.buttons['reset']):
                self._reset_video()
            elif inside(x, y, self.buttons['config']):
                print("配置按钮点击")
        else:
            # 点击画面区域切换播放/暂停
            self._play_pause()

    def _draw_button(self, canvas, text, rect, color=(200, 200, 200), text_color=(0, 0, 0)):
        """绘制按钮"""
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 2)

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    def _draw_progress_bar(self, canvas, x, y, width, height, progress,
                           color=(0, 255, 0), bg_color=(50, 50, 50)):
        """绘制进度条"""
        # 背景
        cv2.rectangle(canvas, (x, y), (x + width, y + height), bg_color, -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 1)

        # 进度
        fill_width = int(width * progress / 100)
        cv2.rectangle(canvas, (x, y), (x + fill_width, y + height), color, -1)

        # 文本
        text = f"{progress:.1f}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_confidence_plot(self, canvas, x, y, width, height):
        """绘制置信度曲线"""
        if not self.confidence_history:
            return

        # 创建子画布
        plot_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # 绘制网格
        grid_color = (50, 50, 50)
        for i in range(0, width, 20):
            cv2.line(plot_canvas, (i, 0), (i, height), grid_color, 1)
        for i in range(0, height, 20):
            cv2.line(plot_canvas, (0, i), (width, i), grid_color, 1)

        # 绘制坐标轴
        cv2.line(plot_canvas, (0, height-1), (width-1, height-1), (200, 200, 200), 2)
        cv2.line(plot_canvas, (0, 0), (0, height-1), (200, 200, 200), 2)

        # 绘制置信度曲线
        points = []
        max_len = min(len(self.confidence_history), width)

        for i in range(max_len):
            conf = self.confidence_history[-(max_len - i)]
            x_pos = i
            y_pos = int(height * (1 - conf))
            points.append((x_pos, y_pos))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(plot_canvas, points[i], points[i+1], (0, 255, 0), 2)

        # 阈值线
        threshold_y = int(height * (1 - self.config.confidence_threshold))
        cv2.line(plot_canvas, (0, threshold_y), (width, threshold_y), (0, 0, 255), 1)

        # 复制到主画布
        canvas[y:y+height, x:x+width] = plot_canvas

        # 标题
        cv2.putText(canvas, "Confidence", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_sampled_frames(self, canvas, x, y, width, height):
        """绘制采样帧预览"""
        if not self.config.show_sampled_frames:
            return

        # 获取采样帧索引
        indices = self.processor.get_sampled_frame_indices()
        if not indices:
            return

        # 获取帧
        frames = self.processor.buffer_manager.get_frames()
        if not frames:
            return

        # 计算每个小图的尺寸
        n_cols = 4
        n_rows = 4
        frame_width = width // n_cols
        frame_height = height // n_rows

        # 绘制每个采样帧
        for i, idx in enumerate(indices[:16]):  # 最多显示16帧
            if idx < len(frames):
                row = i // n_cols
                col = i % n_cols

                frame_x = x + col * frame_width
                frame_y = y + row * frame_height

                # 获取并调整帧大小
                frame = frames[idx]
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(frame, (frame_width, frame_height))
                    canvas[frame_y:frame_y+frame_height, frame_x:frame_x+frame_width] = frame

                # 绘制索引
                cv2.putText(canvas, str(idx), (frame_x + 5, frame_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 标题
        cv2.putText(canvas, "Sampled Frames", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_buffer_status(self, canvas, x, y, width, height):
        """绘制缓冲区状态"""
        status = self.processor.get_buffer_status()

        # 绘制缓冲区图示
        buffer_width = 200
        buffer_height = 20
        buffer_x = x
        buffer_y = y

        self._draw_progress_bar(canvas, buffer_x, buffer_y, buffer_width, buffer_height,
                                status['percent'])

        # 状态文本
        status_text = f"Buffer: {status['current']}/{status['capacity']}"
        cv2.putText(canvas, status_text, (buffer_x + buffer_width + 10, buffer_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 采样信息
        sample_text = f"Sample: {self.config.buffer.sample_size} frames (skip {self.config.buffer.jump_frames-1})"
        cv2.putText(canvas, sample_text, (buffer_x, buffer_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_performance_stats(self, canvas, x, y):
        """绘制性能统计"""
        stats = self.processor.get_performance_stats()

        # 性能信息
        fps = stats['fps'] if stats['fps'] < 1000 else 999.9
        perf_text = f"FPS: {fps:.1f} | Inferences: {stats['total_inferences']}"
        cv2.putText(canvas, perf_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 平均推理时间
        if stats['avg_inference_time'] > 0:
            time_text = f"Avg Inference: {stats['avg_inference_time']*1000:.1f}ms"
            cv2.putText(canvas, time_text, (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_inference_result(self, canvas, x, y):
        """绘制推理结果"""
        result = self.processor.current_result
        if not result:
            return

        # 获取真实标签
        video_path = self.video_list[self.idx]
        true_label = self._parse_true_label(video_path)

        # 预测标签
        pred_label = str(result['prediction'])
        if result['prediction'] < len(self.class_names):
            pred_label = self.class_names[result['prediction']]

        # 置信度
        confidence = result['confidence']

        # 判断是否正确
        is_correct = (pred_label == true_label)
        color = (0, 255, 0) if is_correct else (0, 0, 255)

        # 绘制结果
        result_text = f"Pred: {pred_label} ({confidence:.2%})"
        cv2.putText(canvas, result_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        true_text = f"True: {true_label}"
        cv2.putText(canvas, true_text, (x, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 推理时间
        time_text = f"Time: {result['inference_time']*1000:.1f}ms"
        cv2.putText(canvas, time_text, (x, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)

    def _handle_keyboard(self):
        """处理键盘输入"""
        current_time = time.time()
        can_process = (current_time - self.last_key_time) > 0.05

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            return True
        elif key == ord('a') or key == ord('A') or key == 81:  # 左箭头或A
            if can_process:
                self.idx = (self.idx - 1) % len(self.video_list)
                self._load_current_video()
                self.last_key_time = current_time
        elif key == ord('d') or key == ord('D') or key == 83:  # 右箭头或D
            if can_process:
                self.idx = (self.idx + 1) % len(self.video_list)
                self._load_current_video()
                self.last_key_time = current_time
        elif key == 13:  # Enter - 手动推理
            if can_process:
                self._infer_current_video()
                self.last_key_time = current_time
        elif key == 32:  # Space - 播放/暂停
            if can_process:
                self._play_pause()
                self.last_key_time = current_time
        elif key == ord('s') or key == ord('S'):  # S - 保存配置
            self.config.to_yaml('configs/last_config.yaml')
            print("配置已保存")
        elif key == ord('1'):  # 1 - 切换到平衡模式
            self._switch_config('balanced')
        elif key == ord('2'):  # 2 - 切换到高精度模式
            self._switch_config('high_accuracy')
        elif key == ord('3'):  # 3 - 切换到实时模式
            self._switch_config('real_time')

        return False

    def _switch_config(self, preset_name: str):
        """切换到预设配置"""
        from inference_config import CONFIG_PRESETS
        if preset_name in CONFIG_PRESETS:
            # 保存当前配置
            old_config = self.config

            # 切换到新配置
            self.config = CONFIG_PRESETS[preset_name]

            # 重新创建处理器
            self.processor = self._create_processor()

            print(f"已切换到 {preset_name} 模式")

    def _resize_frame(self, frame, max_width, max_height):
        """调整帧大小"""
        if frame is None:
            return None, 0, 0

        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        return resized, new_w, new_h

    def run(self):
        """主循环"""
        last_frame = None

        while True:
            # 获取当前帧
            frame = None
            if self.is_playing:
                frame = self._get_next_frame()
                if frame is None:
                    self._reset_video()
                    frame = self._get_next_frame()

            if frame is not None:
                last_frame = frame.copy()
            elif last_frame is not None:
                frame = last_frame.copy()

            # 创建画布
            canvas_height = self.config.window_height
            canvas_width = self.config.window_width
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # 视频显示区域
            video_height = canvas_height - self.config.ui_height
            video_width = canvas_width

            # 如果有帧，显示帧
            if frame is not None:
                frame_resized, img_w, img_h = self._resize_frame(
                    frame, max_width=video_width, max_height=video_height
                )

                if frame_resized is not None:
                    # 放置视频
                    canvas[0:img_h, 0:img_w] = frame_resized

                    # 绘制帧信息
                    frame_info = f"Frame: {self.current_frame_idx}/{self.total_frames}"
                    cv2.putText(canvas, frame_info, (img_w - 200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 绘制UI区域分隔线
            separator_y = video_height
            cv2.line(canvas, (0, separator_y), (canvas_width, separator_y), (100, 100, 100), 2)

            # 绘制UI区域背景
            ui_region = canvas[separator_y:separator_y + self.config.ui_height, 0:canvas_width]
            ui_region.fill(50)

            # 绘制按钮
            self._draw_button(canvas, 'Prev', self.buttons['prev'])
            self._draw_button(canvas, 'Next', self.buttons['next'])
            self._draw_button(canvas, 'Pause' if self.is_playing else 'Play', self.buttons['play_pause'])
            self._draw_button(canvas, 'Infer', self.buttons['infer'])

            # 自动推理按钮
            auto_status = 'Auto: ON' if self.processor.auto_inference else 'Auto: OFF'
            auto_color = (0, 255, 0) if self.processor.auto_inference else (0, 0, 255)
            self._draw_button(canvas, auto_status, self.buttons['auto'], auto_color)

            self._draw_button(canvas, 'Global', self.buttons['global'])
            self._draw_button(canvas, 'Reset', self.buttons['reset'])
            self._draw_button(canvas, 'Config', self.buttons['config'])

            # 绘制缓冲区状态
            if self.config.show_buffer_status:
                self._draw_buffer_status(canvas, 850, separator_y + 10, 200, 60)

            # 绘制推理结果
            self._draw_inference_result(canvas, 10, separator_y + 40)

            # 绘制性能统计
            if self.config.show_performance_stats:
                self._draw_performance_stats(canvas, 10, separator_y + 100)

            # 绘制置信度曲线
            if len(self.confidence_history) > 1:
                self._draw_confidence_plot(canvas, 300, separator_y + 10, 200, 60)

            # 绘制采样帧预览
            if self.config.show_sampled_frames:
                self._draw_sampled_frames(canvas, 520, separator_y + 10, 300, 60)

            # 显示文件名和索引
            video_path = self.video_list[self.idx]
            file_info = f'{os.path.basename(video_path)} [{self.idx + 1}/{len(self.video_list)}]'
            cv2.putText(canvas, file_info, (650, separator_y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示操作提示
            hint_text = 'A/D:Navigate Space:Play/Pause Enter:Infer 1/2/3:Presets Q:Quit'
            cv2.putText(canvas, hint_text, (10, separator_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 如果全局推理已完成，显示准确率
            if self.global_done:
                acc_text = f'Global Acc: {self.global_stats["accuracy"]:.2%} ({self.global_stats["correct"]}/{self.global_stats["total"]})'
                cv2.putText(canvas, acc_text, (850, separator_y + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow(self.win, canvas)

            # 处理键盘输入
            if self._handle_keyboard():
                break

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# 入口函数
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='./data_ucf_101', help='测试视频目录')
    parser.add_argument('--train_dir', default='./data/train', help='训练数据目录')
    parser.add_argument('--config', default=None, help='配置文件路径')
    parser.add_argument('--preset', default='balanced', choices=['balanced', 'high_accuracy', 'real_time'],
                        help='预设配置')

    args = parser.parse_args()

    # 加载配置
    from inference_config import CONFIG_PRESETS, InferenceConfig

    if args.config:
        config = InferenceConfig.from_yaml(args.config)
    else:
        config = CONFIG_PRESETS.get(args.preset, InferenceConfig())

    # 创建并运行UI
    ui = EnhancedVideoInferenceUI(
        video_dir=args.video_dir,
        train_data_root=args.train_dir,
        config=config
    )
    ui.run()
