import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, dataset_path, images_path, clip_len):
        self.dataset_path = dataset_path  # 数据集的地址
        self.split = images_path  # 训练集，测试集，验证集的名字
        self.clip_len = clip_len  # 生成的数据的的深度的值

        # 后续数据预处理的值
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # 读取对应训练集/验证集/测试集下的各种类别的行为动作
        # 每个行为动作下的视频已经被处理成单个图片数据
        # 将对应动作的数据的文件名作为标签保存到classes列表中，对应的动作数据集的路径保存到self.fnames列表中，标签和数据是一一对应的状态
        folder = os.path.join(self.dataset_path, images_path)
        self.fnames = []
        classes = []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                classes.append(label)
        print('Number of {} videos: {:d}'.format(images_path, len(self.fnames)))

        # 保存类别名列表（用于后续查表）
        self.classes = sorted(set(classes))
        # 获取对应视频的标签，并将标签转化为int的数字类型，同时转化为array格式
        self.label2index = {label: index for index, label in enumerate(self.classes)}
        self.label_array = np.array([self.label2index[label] for label in classes], dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])  # 加载对应类别的动作数据集，并转化为（帧数， 高， 宽， 3通道）
        buffer = VideoDataset.crop(buffer, self.clip_len, self.crop_size)  # 在数据的深度，高度，宽度方向进行随机裁剪，将数据数据转化为（clip_len， 112, 112， 3）
        # buffer = VideoDataset.normalize(buffer)  # 对模型进行归一化处理
        buffer = VideoDataset.to_tensor(buffer)  # 对维度进行转化

        # 获取对应视频的标签数据
        label = np.array(self.label_array[index])

        # 返回torch格式的特征和标签
        return torch.from_numpy(buffer), torch.from_numpy(label)

    def load_frames(self, file_dir):
        # 将文件夹下的数据集进行排序
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        # 获取该动作数据的长度
        frame_count = len(frames)
        # 生成一个空的（frame_count， resize_height， resize_width， 3）维度的数据
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        # 遍历循环获取动作的路径
        for i, frame_name in enumerate(frames):
            # 利用cv去读取图片数据，并转化为np.array格式
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            # 不断遍历循环赋值给buffer
            buffer[i] = frame
        return buffer

    @staticmethod
    def crop(buffer, clip_len, crop_size):
        time_index = np.random.randint(buffer.shape[0] - clip_len)  # 生成一个深度方向上的随机长度
        height_index = np.random.randint(buffer.shape[1] - crop_size)  # 生成一个高度方向上的随机长度
        width_index = np.random.randint(buffer.shape[2] - crop_size)  # 生成一个宽度方向上的随机长度

        # 利用切片在视频上进行提取，获得一个（clip_len， 112, 112, 3）数据
        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :]

        return buffer

    @staticmethod
    def normalize(buffer):
        # 进行归一化
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    @staticmethod
    def to_tensor(buffer):
        # 进行维度的转化，将最后的一个维调转到第一维
        return buffer.transpose((3, 0, 1, 2))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset_path='data', images_path='train', clip_len=16)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    val_data = VideoDataset(dataset_path='data', images_path='val', clip_len=16)
    val_ader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)

    test_data = VideoDataset(dataset_path='data', images_path='test', clip_len=16)
    test_ader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)
