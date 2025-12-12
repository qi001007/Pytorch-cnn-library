from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               # 是否打乱
                               shuffle=True,
                               # 使用进程几工作
                               num_workers=0)

# 获得一个Batch的数据
# enumerate 把每个 batch 编号（step）和内容（b_x, b_y）一起返回
b_x, b_y = None, None
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
print(class_label)
# print(batch_x)
# print(batch_y)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224

# 可视化一个Batch的图像
# 1. 新建一张画布，宽 12 英寸、高 5 英寸（1 英寸=72 pt）
plt.figure(figsize=(12, 5))
# 2. 遍历当前批次中的每一张图片
for ii, label in enumerate(batch_y):
    # 3. 在 4×16 的网格里定位第 ii+1 个子图
    plt.subplot(4, 16, ii + 1)                # 参数顺序：行、列、序号（从 1 开始）
    # 4. 把第 ii 张灰度图画出来
    plt.imshow(batch_x[ii, :, :],             # 要显示的数据
               cmap='gray')              # 指定灰度色表，否则默认是伪彩色
    # 5. 在子图上方写标题：把数字转成汉字/字符串
    plt.title(class_label[batch_y[ii]],       # 根据标签取对应汉字
              size=10)                        # 字号 10，防止拥挤
    # 6. 不画坐标轴，保持图片区域干净
    plt.axis("off")
    # 7. 调整子图之间的横向间隙（width space）
    #    0.05 表示子图左右只留 5% 空隙，让 64 张图更紧凑
    plt.subplots_adjust(wspace=0.05)
# 8. 把所有子图一次性弹窗显示出来
plt.show()
