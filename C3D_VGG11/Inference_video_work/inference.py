import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque

from globals import MW


class Inference:
    def __init__(self):
        self.device = None
        # 引入双端队列
        self.clip = deque(maxlen=MW.clip_num)
        self.step = 1
        self.button_state = False
        self.global_button_state = False
        self.class_names = []
        self.probs = []
        self.label = 0
        self.model_init()

    def model_init(self):
        # 定义模型训练的设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 加载数据集标签
        label_path = MW.PARAMS_MAP['data_label_dir']
        if not label_path or label_path == '.' or Path(label_path).is_dir():
            raise ValueError('请先选择标签文件（.txt），而不是目录')
        with open(label_path, 'r') as f:
            self.class_names = f.readlines()
            # print(class_names)
            f.close()

        # 加载模型，并将模型参数加载到模型中
        state_dict = torch.load(MW.PARAMS_MAP['wts_dir'], map_location=self.device)
        MW.model.load_state_dict(state_dict)

        # 将模型放入到设备中，并设置验证模式
        MW.model.to(self.device)
        MW.model.eval()

    @staticmethod
    def center_crop(frame):
        frame = frame[8:120, 30:142, :]
        return np.array(frame).astype(np.uint8)

    def global_infer_button(self):
        self.global_button_state = not self.global_button_state

    def infer_button(self):
        self.button_state = not self.button_state

    def infer_button_state(self):
        if self.global_button_state:
            self.button_state = True
        else:
            self.button_state = False

    def infer(self, ret, frame, frame_num):
        # 无论按钮开关，只要帧有效就入队 → 真正滑动
        if ret and frame_num % self.step == 0:
            tmp = cv2.resize(frame, (112, 112))
            # 归一化
            # tmp = tmp - np.array([[[90.0, 98.0, 102.0]]])
            self.clip.append(tmp)  # deque 自动保持 maxlen=16
        # 按钮关 或 缓存不够，直接返回
        if not self.button_state or len(self.clip) < 16:
            return
        inputs = np.array(self.clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        inputs = torch.from_numpy(inputs)
        inputs = inputs.to(device=self.device,  # ← 搬到模型同设备
                           non_blocking=True)
        try:
            with torch.no_grad():
                outputs = MW.model(inputs)
        except RuntimeError as e:
            print(e)
            raise
        self.probs = torch.nn.Softmax(dim=1)(outputs)
        self.label = torch.max(self.probs, 1)[1].detach().cpu().numpy()[0]

    def show_infer(self):
        if not self.button_state or len(self.clip) < 16:
            MW.res.setText('res:')
            MW.pre.setText('pre:')
        else:
            MW.res.setText('res: ' + self.class_names[self.label])
            MW.pre.setText('pre: {:.2f}%'.format(self.probs[0][self.label]*100))
