import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import random  # 新增：用于随机选择样本
#import numpy as np  # 新增：用于数组操作


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=20, shuffle=True)


def evaluate(test_data, net, device):
    n_correct = 0
    n_total = 0
    net.eval()
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    net.train()
    return n_correct / n_total


def visualize_random_samples(net, device, num_samples=3):
    """
    从训练集中随机选择样本，展示原图和预测结果
    """
    # 重新加载训练数据集（不使用DataLoader，直接访问Dataset）
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST("", train=True, transform=transform, download=True)

    # 随机选择样本索引
    random_indices = random.sample(range(len(train_dataset)), num_samples)

    # 创建图形
    plt.figure(figsize=(12, 4))

    for i, idx in enumerate(random_indices):
        # 获取单个样本
        image, true_label = train_dataset[idx]

        # 准备输入数据
        image_tensor = image.to(device)
        # 展平图像并添加batch维度
        input_tensor = image_tensor.view(1, -1)

        # 模型预测
        net.eval()
        with torch.no_grad():
            output = net(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        # 将图像转换为numpy数组用于显示（移除通道维度）
        image_np = image_tensor.cpu().numpy().squeeze()

        # 创建子图
        ax = plt.subplot(1, num_samples, i + 1)

        # 显示图像
        ax.imshow(image_np, cmap='gray')

        # 设置标题：预测结果和真实标签
        title = f'Predicted: {predicted_label}\nTrue: {true_label}'
        color = 'green' if predicted_label == true_label else 'red'
        ax.set_title(title, color=color, fontsize=12)

        # 移除坐标轴
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("random_samples_prediction.png", dpi=150, bbox_inches='tight')
    print("Random samples prediction visualization saved at ~/random_samples_prediction.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Now we are using device: {device}")

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)
    epochs = []
    accuracies = []
    initial_acc = evaluate(test_data, net, device)
    print("initial accuracy:", initial_acc)

    epochs.append(-1)
    accuracies.append(initial_acc)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        for (x, y) in train_data:
            x = x.to(device)
            y = y.to(device)
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(test_data, net, device)
        print("epoch", epoch, "accuracy:", acc)

        epochs.append(epoch)
        accuracies.append(acc)

    # 绘制训练准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xticks(epochs)
    plt.savefig("accuracy_curve.png", dpi=150, bbox_inches='tight')
    print("The accuracy function is saved at ~/accuracy_curve.png")
    plt.close()  # 关闭图形，避免与后面的图形冲突

    # === 新增功能：随机选择三张训练集图片进行预测和可视化 ===
    print("\n" + "=" * 50)
    print("Visualizing random samples from training set...")
    print("=" * 50)
    visualize_random_samples(net, device, num_samples=3)


if __name__ == "__main__":
    main()