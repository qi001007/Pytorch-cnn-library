#!/usr/bin/env python3
"""
主启动脚本
"""
import argparse
import sys
import os

from enhanced_inference_ui import EnhancedVideoInferenceUI
from inference_config import CONFIG_PRESETS, InferenceConfig

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='增强视频推理系统')

    # 基本参数
    parser.add_argument('--video_dir', default=r'E:\pycharm_project\pytorch_001\Pytorch框架下的卷积神经网络库\C3D_VGG11\data_ucf_101',
                        help='测试视频目录 (默认: ./data_ucf_101)')
    parser.add_argument('--train_dir', default=r'E:\pycharm_project\pytorch_001\Pytorch框架下的卷积神经网络库\C3D_VGG11\data\train',
                        help='训练数据目录 (默认: ./data/train)')

    # 配置参数
    parser.add_argument('--config', default=None,
                        help='配置文件路径 (默认: 使用预设配置)')
    parser.add_argument('--preset', default='balanced',
                        choices=['balanced', 'high_accuracy', 'real_time'],
                        help='预设配置模式 (默认: balanced)')

    # 缓冲区参数
    parser.add_argument('--buffer_size', type=int, default=64,
                        help='缓冲区大小 (默认: 64)')
    parser.add_argument('--sample_size', type=int, default=16,
                        help='采样帧数 (默认: 16)')
    parser.add_argument('--jump_frames', type=int, default=4,
                        help='跳帧间隔 (默认: 4)')

    # 推理参数
    parser.add_argument('--auto_inference', action='store_true', default=True,
                        help='启用自动推理 (默认: True)')
    parser.add_argument('--inference_interval', type=float, default=0.1,
                        help='推理间隔(秒) (默认: 0.1)')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                        help='置信度阈值 (默认: 0.3)')

    # UI参数
    parser.add_argument('--window_width', type=int, default=1200,
                        help='窗口宽度 (默认: 1200)')
    parser.add_argument('--window_height', type=int, default=800,
                        help='窗口高度 (默认: 800)')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = InferenceConfig.from_yaml(args.config)
    else:
        config = CONFIG_PRESETS.get(args.preset, InferenceConfig())

    # 更新命令行参数
    config_args = {
        'buffer_size': args.buffer_size,
        'sample_size': args.sample_size,
        'jump_frames': args.jump_frames,
        'auto_inference': args.auto_inference,
        'inference_interval': args.inference_interval,
        'confidence_threshold': args.confidence_threshold,
        'window_width': args.window_width,
        'window_height': args.window_height
    }

    # 确保目录存在
    if not os.path.exists(args.video_dir):
        print(f"警告: 视频目录不存在: {args.video_dir}")
        print("请确保视频目录存在或使用 --video_dir 参数指定正确路径")
        return

    if not os.path.exists(args.train_dir):
        print(f"警告: 训练目录不存在: {args.train_dir}")
        print("请确保训练目录存在或使用 --train_dir 参数指定正确路径")
        return

    # 创建并运行UI
    try:
        ui = EnhancedVideoInferenceUI(
            video_dir=args.video_dir,
            train_data_root=args.train_dir,
            config=config
        )
        ui.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
