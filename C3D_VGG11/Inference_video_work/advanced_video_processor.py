"""
高级视频处理器 - 实现滑动窗口、跳帧采样和实时推理
"""
import cv2
import numpy as np
import torch
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
# from PIL import Image
from torchvision import transforms


@dataclass
class ProcessorConfig:
    """视频处理器配置类"""
    # 缓冲区配置
    buffer_size: int = 64
    sample_size: int = 16
    jump_frames: int = 4

    # 视频预处理配置
    resize_height: int = 128
    resize_width: int = 171
    crop_size: int = 112
    normalize_mean: List[float] = None

    # 推理配置
    inference_interval: float = 0.1  # 推理间隔(秒)
    auto_inference: bool = True

    # 显示配置
    max_display_results: int = 5
    confidence_threshold: float = 0.3

    def __post_init__(self):
        if self.normalize_mean is None:
            self.normalize_mean = [90.0, 98.0, 102.0]


class FrameSampler:
    """帧采样器 - 负责从缓冲区中采样帧"""

    @staticmethod
    def sample_frames(frames: List[np.ndarray], buffer_size: int,
                      sample_size: int, jump_frames: int) -> List[np.ndarray]:
        """
        智能帧采样策略

        参数:
            frames: 帧列表
            buffer_size: 缓冲区大小 (64)
            sample_size: 采样数量 (16)
            jump_frames: 跳帧间隔 (4)

        返回:
            采样后的帧列表
        """
        if not frames:
            return []

        num_frames = len(frames)

        # 如果帧数刚好等于sample_size，直接返回
        if num_frames == sample_size:
            return frames

        # 如果帧数少于sample_size，补帧
        if num_frames < sample_size:
            sampled_frames = frames.copy()
            if sampled_frames:
                last_frame = sampled_frames[-1]
                while len(sampled_frames) < sample_size:
                    sampled_frames.append(last_frame)
            return sampled_frames

        # 如果帧数大于等于buffer_size，从最后buffer_size帧中跳帧采样
        if num_frames >= buffer_size:
            # 取最后buffer_size帧
            start_idx = num_frames - buffer_size
            actual_frames = frames[start_idx:]
            step = jump_frames
            indices = [i * step for i in range(sample_size)]
            indices = [min(idx, buffer_size - 1) for idx in indices]
            return [actual_frames[idx] for idx in indices]

        # 如果帧数在sample_size和buffer_size之间，等间隔采样
        step = max(1, num_frames // sample_size)
        indices = [min(i * step, num_frames - 1) for i in range(sample_size)]
        return [frames[idx] for idx in indices]


class BufferManager:
    """缓冲区管理器 - 负责管理帧缓冲区"""

    def __init__(self, buffer_size: int = 64):
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """添加帧到缓冲区"""
        if timestamp is None:
            timestamp = time.time()

        self.frame_buffer.append(frame)
        self.timestamps.append(timestamp)

        # 返回当前缓冲区状态
        return len(self.frame_buffer), self.buffer_size

    def get_frames(self) -> List[np.ndarray]:
        """获取所有帧"""
        return list(self.frame_buffer)

    def get_recent_frames(self, n: int) -> List[np.ndarray]:
        """获取最近n帧"""
        if n <= 0:
            return []
        return list(self.frame_buffer)[-n:]

    def clear(self):
        """清空缓冲区"""
        self.frame_buffer.clear()
        self.timestamps.clear()

    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return len(self.frame_buffer) >= self.buffer_size

    def get_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        return {
            'current': len(self.frame_buffer),
            'capacity': self.buffer_size,
            'percent': (len(self.frame_buffer) / self.buffer_size) * 100,
            'is_full': self.is_full()
        }


class VideoPreprocessor:
    """视频预处理器 - 负责帧的预处理"""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        # 注意：训练时没有使用normalize，所以我们也应该保持一致
        # 训练时使用的是：读取->resize->crop->to_tensor
        # 没有减去均值
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.resize_height, config.resize_width)),
            transforms.CenterCrop(config.crop_size),
            transforms.ToTensor(),  # 这将转换到[0,1]范围
        ])

        print(f"预处理器配置: resize=({config.resize_height}, {config.resize_width}), crop={config.crop_size}")
        print(f"是否归一化: {'否' if self.config.normalize_mean is None else '是'}")

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """预处理单帧 - 输入应该是RGB格式的numpy数组"""
        # 确保输入是RGB格式
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # 已经是RGB，直接处理
            pass
        else:
            # 假设是BGR，转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 应用变换，ToTensor会自动将值从[0,255]缩放到[0,1]
        tensor = self.transform(frame)

        return tensor

    def process_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """处理多帧为模型输入"""
        processed_frames = []
        for frame in frames:
            tensor = self.preprocess_frame(frame)
            processed_frames.append(tensor)

        # 堆叠帧，得到 [sample_size, channels, height, width]
        stacked = torch.stack(processed_frames, dim=0)  # [16, 3, 112, 112]

        # 添加batch维度并调整维度顺序为 [batch, channels, depth, height, width]
        # 需要从 [16, 3, 112, 112] 转换为 [1, 3, 16, 112, 112]
        stacked = stacked.permute(1, 0, 2, 3)  # [3, 16, 112, 112]
        stacked = stacked.unsqueeze(0)  # [1, 3, 16, 112, 112]

        # 重要：根据训练时的实际处理，决定是否归一化
        # 检查config中是否启用了归一化
        if self.config.normalize_mean:
            # 注意：训练时normalize被注释掉了，所以我们应该保持一致
            # 但为了测试，我们可以尝试两种方式
            mean_tensor = torch.tensor(self.config.normalize_mean,
                                       dtype=stacked.dtype,
                                       device=stacked.device)
            mean_tensor = mean_tensor.view(1, 3, 1, 1, 1) / 255.0  # 缩放并调整维度
            stacked = stacked - mean_tensor

        return stacked

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """归一化张量 - 减去训练时使用的均值"""
        if self.config.normalize_mean:
            # 创建与tensor相同设备的均值张量
            # 注意：我们的tensor是 [1, 3, 16, 112, 112]
            # 均值是 [90.0, 98.0, 102.0]，但我们需要考虑tensor的值范围
            # 因为ToTensor已经将值从[0,255]缩放到[0,1]
            # 所以均值也需要相应缩放
            mean_tensor = torch.tensor(self.config.normalize_mean,
                                       dtype=tensor.dtype,
                                       device=tensor.device)
            mean_tensor = mean_tensor.view(1, 3, 1, 1, 1) / 255.0  # 缩放并调整维度

            return tensor - mean_tensor
        return tensor


class AdvancedVideoProcessor:
    """高级视频处理器 - 主类"""

    def __init__(self, model: torch.nn.Module, config: ProcessorConfig):
        self.model = model
        self.config = config

        # 初始化组件
        self.buffer_manager = BufferManager(config.buffer_size)
        self.frame_sampler = FrameSampler()
        self.preprocessor = VideoPreprocessor(config)

        # 推理状态
        self.auto_inference = config.auto_inference
        self.inference_interval = config.inference_interval
        self.last_inference_time = 0

        # 结果存储
        self.current_result = None
        self.result_history = deque(maxlen=config.max_display_results)
        self.inference_count = 0

        # 性能统计
        self.frame_count = 0
        self.total_inference_time = 0

        # 设置模型为评估模式
        self.model.eval()

        # 设备设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"视频处理器初始化完成:")
        print(f"  缓冲区大小: {config.buffer_size}")
        print(f"  采样帧数: {config.sample_size}")
        print(f"  跳帧间隔: {config.jump_frames}")
        print(f"  设备: {self.device}")

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        处理单帧图像

        参数:
            frame: 输入帧 (BGR格式)

        返回:
            推理结果字典，如果未进行推理则返回None
        """
        # 转换并添加帧到缓冲区
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.buffer_manager.add_frame(rgb_frame)
        self.frame_count += 1

        # 检查是否应该进行推理
        should_infer = False

        if self.auto_inference:
            # 检查缓冲区是否满
            if self.buffer_manager.is_full():
                current_time = time.time()
                # 检查推理间隔
                if current_time - self.last_inference_time >= self.inference_interval:
                    should_infer = True

        # 执行推理
        if should_infer:
            result = self._perform_inference()
            if result:
                self.last_inference_time = time.time()
                return result

        return None

    def manual_inference(self) -> Optional[Dict[str, Any]]:
        """手动触发推理"""
        current_frames = len(self.buffer_manager.frame_buffer)
        if current_frames >= self.config.sample_size:
            return self._perform_inference()
        return None

    def _perform_inference(self) -> Optional[Dict[str, Any]]:
        """执行推理"""
        try:
            # 获取所有帧
            all_frames = self.buffer_manager.get_frames()

            # 采样帧
            sampled_frames = self.frame_sampler.sample_frames(
                all_frames,
                self.config.buffer_size,
                self.config.sample_size,
                self.config.jump_frames
            )

            if len(sampled_frames) < self.config.sample_size:
                print(f"警告: 采样帧数不足: {len(sampled_frames)} < {self.config.sample_size}")
                return None

            # 处理帧
            start_time = time.time()
            processed_tensor = self.preprocessor.process_frames(sampled_frames)

            # 归一化
            processed_tensor = self.preprocessor.normalize_tensor(processed_tensor)

            # 打印调试信息
            print(f"推理输入张量形状: {processed_tensor.shape}")
            print(f"输入范围: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]")

            # 移动到设备
            processed_tensor = processed_tensor.to(self.device)

            # 推理
            with torch.no_grad():
                output = self.model(processed_tensor)

            # 计算softmax
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

            # 计算推理时间
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.inference_count += 1

            # 准备结果
            result = {
                'prediction': pred_idx.item(),
                'confidence': confidence.item(),
                'timestamp': time.time(),
                'frame_count': self.frame_count,
                'inference_time': inference_time,
                'buffer_status': self.buffer_manager.get_status(),
                'sampled_frames': len(sampled_frames)
            }

            # 更新当前结果和历史
            self.current_result = result
            self.result_history.append(result)

            print(f"推理完成: 类别={pred_idx.item()}, 置信度={confidence.item():.4f}, 时间={inference_time*1000:.1f}ms")

            return result

        except Exception as e:
            print(f"推理错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_buffer_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        return self.buffer_manager.get_status()

    def get_recent_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的n个推理结果"""
        return list(self.result_history)[-n:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_inference_time = 0
        if self.inference_count > 0:
            avg_inference_time = self.total_inference_time / self.inference_count

        fps = 0
        if self.frame_count > 0 and self.last_inference_time > 0:
            fps = self.frame_count / (time.time() - self.last_inference_time + 1e-6)

        return {
            'total_frames': self.frame_count,
            'total_inferences': self.inference_count,
            'avg_inference_time': avg_inference_time,
            'fps': fps
        }

    def reset(self):
        """重置处理器状态"""
        self.buffer_manager.clear()
        self.frame_count = 0
        self.current_result = None
        self.result_history.clear()
        self.inference_count = 0
        self.total_inference_time = 0
        self.last_inference_time = time.time()

    def set_auto_inference(self, enabled: bool):
        """设置自动推理开关"""
        self.auto_inference = enabled

    def set_inference_interval(self, interval: float):
        """设置推理间隔"""
        self.inference_interval = max(0.01, interval)  # 最小10ms

    def get_sampled_frame_indices(self) -> List[int]:
        """获取最近一次推理的采样帧索引"""
        if not self.current_result:
            return []

        buffer_len = len(self.buffer_manager.frame_buffer)
        if buffer_len >= self.config.buffer_size:
            step = self.config.jump_frames
            start_idx = buffer_len - self.config.buffer_size
            indices = [start_idx + i * step for i in range(self.config.sample_size)]
            indices = [min(idx, buffer_len - 1) for idx in indices]
            return indices
        else:
            step = max(1, buffer_len // self.config.sample_size)
            indices = [min(i * step, buffer_len - 1) for i in range(self.config.sample_size)]
            return indices
