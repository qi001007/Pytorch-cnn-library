"""
推理配置管理
"""
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    model_class: str = "C3D_VGG11"
    num_classes: int = 101
    pretrained: bool = False  # 改为False，因为推理时不需要预训练权重
    weights_path: str = r"E:\pycharm_project\pytorch_001\Pytorch框架下的卷积神经网络库\C3D_VGG11\model_wts\2025-12-10-09-24model\best_model_2025-12-10-14-00.pth"
    pretrained_weights_path: str = "ucf101-caffe.pth"  # 添加预训练权重路径


@dataclass
class BufferConfig:
    """缓冲区配置"""
    buffer_size: int = 64
    sample_size: int = 16
    jump_frames: int = 4


@dataclass
class InferenceConfig:
    """推理配置"""
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)

    # 缓冲区配置
    buffer: BufferConfig = field(default_factory=BufferConfig)

    # 预处理配置
    resize_height: int = 128
    resize_width: int = 171
    crop_size: int = 112
    normalize_mean: List[float] = None
    # normalize_mean: List[float] = field(default_factory=lambda: [90.0, 98.0, 102.0])

    # 推理行为配置
    auto_inference: bool = True
    inference_interval: float = 0.1  # 秒
    confidence_threshold: float = 0.3

    # UI配置
    window_width: int = 1200
    window_height: int = 800
    ui_height: int = 140
    button_width: int = 100
    button_height: int = 40
    button_margin: int = 10

    # 显示配置
    max_display_results: int = 10
    show_sampled_frames: bool = True
    show_buffer_status: bool = True
    show_performance_stats: bool = True

    # 视频配置
    video_extensions: List[str] = field(default_factory=lambda: ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv'])
    default_fps: int = 30

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InferenceConfig':
        """从YAML文件加载配置"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            print(f"配置文件不存在: {yaml_path}")
            return cls()

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)

    def update_from_args(self, args: Dict[str, Any]):
        """从参数字典更新配置"""
        for key, value in args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


# 默认配置
DEFAULT_CONFIG = InferenceConfig()

# 预定义配置
CONFIG_PRESETS = {
    'high_accuracy': InferenceConfig(
        buffer=BufferConfig(buffer_size=96, sample_size=16, jump_frames=6),
        inference_interval=0.2,
        confidence_threshold=0.5
    ),
    'real_time': InferenceConfig(
        buffer=BufferConfig(buffer_size=48, sample_size=16, jump_frames=3),
        inference_interval=0.05,
        confidence_threshold=0.2
    ),
    'balanced': InferenceConfig(
        buffer=BufferConfig(buffer_size=64, sample_size=16, jump_frames=4),
        inference_interval=0.1,
        confidence_threshold=0.3
    )
}
