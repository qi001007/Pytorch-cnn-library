import os
import torch
from torch import nn
from torchsummary import summary


class C3D_VGG11(nn.Module):
    def __init__(self, num_classes, pretrained=False, pretrained_weights_path='ucf101-caffe.pth'):
        super(C3D_VGG11, self).__init__()
        self.block1 = nn.Sequential(nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)))
        self.block2 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.block3 = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.block4 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)))
        self.block5 = nn.Sequential(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                              stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))
        self.block6 = nn.Sequential(nn.Flatten(),
                                    # 使用 Flatten 配合 nn.LazyLinear 可以避免硬编码
                                    nn.LazyLinear(2048), nn.ReLU(), nn.Dropout(),
                                    nn.Linear(in_features=2048, out_features=2048), nn.ReLU(), nn.Dropout(),
                                    nn.Linear(in_features=2048, out_features=num_classes))

        # 在 __init__ 里显式写 self.__init__weight() 的目的就是：在模型实例化时立即执行权重初始化
        self.__init__weight()

        if pretrained:
            # 检查预训练权重文件是否存在
            if os.path.exists(pretrained_weights_path):
                self.__load__pretrained_weights(start_level=1, end_level=16, weights_path=pretrained_weights_path)
            else:
                print(f"警告: 预训练权重文件不存在: {pretrained_weights_path}")
                print("将使用随机初始化的权重")

    # 前向传播函数
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        y = x
        return y

    # 权重初始化函数
    def __init__weight(self):
        # 用假数据触发LazyLinear初始化
        dummy = torch.randn(1, 3, 16, 112, 112)  # N,C,D,H,W 按你实际来
        # 现在所有 LazyModule 都变成正常 Parameter
        _ = self.forward(dummy)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 凯明初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

            if isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    # 加载预训练权重

    def __load__pretrained_weights(self, start_level=1, end_level=16, weights_path='ucf101-caffe.pth'):
        # 检查文件是否存在
        if not os.path.exists(weights_path):
            print(f"错误: 预训练权重文件不存在: {weights_path}")
            return

        # torch.load把pth中的保存的state_dict读成py中的字典
        pre_weights = torch.load(weights_path)
        print(f"加载预训练权重: {weights_path}")
        print(pre_weights.keys())
        # .state_dict()中是py中的字典形式
        self_weights = self.state_dict()

        pre_name = list(pre_weights.keys())
        self_name = list(self_weights.keys())

        if len(pre_name) == len(self_name):
            try:
                # 按顺序 1-1 映射
                name_map = dict(zip(pre_name, self_name))
                # print(name_map)  # 可删掉，仅调试用
            except Exception as e:
                print('构建映射时出错：', e)
                return  # 退出函数
        else:
            print('两模型权重层数不一致')
            return

        for i, (name, weight) in enumerate(pre_weights.items(), 1):
            if start_level <= i <= end_level:
                if self_weights[name_map[name]].shape == weight.shape:
                    self_weights[name_map[name]] = weight
                    print(f'loaded {name} → {name_map[name]}')
                else:
                    print(f'第{i}参数层形状对应可能出错：原参数权重{name_map[name]}层---预训练权重{name}层')
                    print(f'原参数权重{name_map[name]}层shape: {self_weights[name_map[name]].shape} \n'
                          f'预训练权重{name}层shape: {weight.shape}')
                    raise ValueError
        # 写回模型
        self.load_state_dict(self_weights)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # inputs = torch.rand(1, 3, 16, 112, 112)
    C3D_VGG11 = C3D_VGG11(num_classes=101, pretrained=True).to(device)
    # print(summary(C3D_VGG11, inputs))
    print(summary(C3D_VGG11, input_size=(3, 16, 112, 112)))
