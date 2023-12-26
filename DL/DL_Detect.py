import torch
from torchvision import transforms
from PIL import Image
from torch import nn


# 加载模型
def load_model(ckp_path):
    net = nn.Sequential(
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(1024, 256), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 32), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(32, 2))

    net.load_state_dict(torch.load(ckp_path, map_location=torch.device('cpu')))
    return net

def predict(img_path, net):
    image = Image.open(img_path)
    # 定义图像预处理的转换
    transform = transforms.Compose([
        #transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    X = transform(image)    # 转换为二维向量
    X = X.reshape(1,1,128,128)
    y = net(X).argmax(axis=1)
    return y


if __name__=="__main__":
    # dev_iter = get_data_itr(test_image_path, label_path, batch_size)
    net = load_model("./Checkpoints/ckp1.pth")
    image = "./DateSet/Dev/4bf7ec7357b4bd1511ffc2cb6a33c149.exe.png"
    print(predict(image, net))


