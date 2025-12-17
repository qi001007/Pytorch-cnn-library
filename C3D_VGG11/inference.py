import numpy as np
import torch
import cv2
from model import C3D_VGG11
from torch.autograd import Variable
from torch import nn


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def inference():
    # 定义模型训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据集标签
    with open("./data/labels.txt", 'r') as f:
        class_names = f.readlines()
        # print(class_names)
        f.close()

    # 加载模型，并将模型参数加载到模型中
    model = C3D_VGG11(num_classes=101)
    state_dict = torch.load('model_wts/2025-12-10-09-24model/best_model_2025-12-10-14-00.pth',
                            map_location=device)
    model.load_state_dict(state_dict)

    # 将模型放入到设备中，并设置验证模式
    model.to(device)
    model.eval()

    video = "data_ucf_101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()  # 读取视频帧
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)

            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()
