import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm


# 构建自己的数据集
class CustomDataset(Dataset):
    def __init__(self, image_names, labels, data_path='path/to/your/images', transform=None):
        self.image_names = image_names
        self.labels = labels
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = f"{self.data_path}/{self.image_names[idx]}"
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# 加载训练图片数据并预处理为二维向量
def get_data_itr(image_path, label_path, batch_size=32):
    image_names = os.listdir(image_path)
    with open(label_path, "r") as f:
        all_labels = json.load(f)
    labels = []
    for image in image_names:
        labels.append(all_labels[image])
    # 定义图像预处理的转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    # 创建自定义数据集
    custom_dataset = CustomDataset(image_names, labels, data_path=image_path, transform=transform)
    # 创建 DataLoader
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True) # 包含batch_data, batch_labels
    return data_loader

# 通过迭代 DataLoader 获取批量数据
#for batch_data, batch_labels in data_loader:
    # 在这里执行你的训练代码
    # batch_data 是批量的图像数据，batch_labels 是对应的标签

# 模型定义
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

# 模型训练
def train(net, train_iter, test_iter, num_epochs, lr, device):
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    #animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter)
    for epoch in tqdm(range(num_epochs)):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward() # 求导
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch:{epoch}\t', f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    torch.save(net.state_dict(), "./Checkpoints/ckp2.pth")

# 模型评估
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    "使用GPU计算模型在数据集上的精度"
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]






if __name__=="__main__":
    train_image_path = "./DateSet/Train/"
    test_image_path = "./DateSet/Test/"
    label_path = "DateSet/labels.json"
    batch_size = 32
    train_iter = get_data_itr(train_image_path, label_path, batch_size)
    test_iter = get_data_itr(test_image_path, label_path, batch_size)

    lr, num_epochs = 0.01, 10
    train(net, train_iter, test_iter, num_epochs, lr, torch.device('cuda:0'))


    """
    # 模型测试
    X = torch.randn(1, 1, 128, 128)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    """