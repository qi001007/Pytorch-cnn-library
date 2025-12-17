创建配置目录和示例配置
# 创建配置目录
mkdir -p configs

# 创建示例配置文件
python3 -c 
from inference_config import InferenceConfig, CONFIG_PRESETS

# 保存默认配置
config = InferenceConfig()
config.to_yaml('configs/default_config.yaml')

# 保存所有预设配置
for name, preset in CONFIG_PRESETS.items():
    preset.to_yaml(f'configs/{name}_config.yaml')

print('配置文件已创建:')
print('  configs/default_config.yaml')
for name in CONFIG_PRESETS.keys():
    print(f'  configs/{name}_config.yaml')


运行方式
# 使用默认配置运行
python run_inference.py

# 使用预设配置运行
python run_inference.py --preset real_time

# 自定义参数运行
python run_inference.py --buffer_size 96 --sample_size 16 --jump_frames 6

# 使用配置文件运行
python run_inference.py --config configs/high_accuracy_config.yaml

# 指定视频目录
python run_inference.py --video_dir /path/to/your/videos