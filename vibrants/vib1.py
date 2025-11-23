import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=512, shuffle=True)


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


def visualize_predictions(net, test_loader, device, epoch, num_samples=10):
    """可视化模型预测结果并保存为图片"""
    net.eval()
    with torch.no_grad():
        # 获取一个批次的测试数据
        dataiter = iter(test_loader)
        images, labels = next(dataiter)

        # 选择前num_samples个样本
        images = images[:num_samples].to(device)
        labels = labels[:num_samples].to(device)

        # 获取预测结果
        outputs = net(images.view(-1, 28 * 28))
        _, predicted = torch.max(outputs, 1)

        # 创建可视化图形
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(num_samples):
            ax = axes[i]
            # 将图像转换为numpy格式并移除通道维度
            img = images[i].cpu().numpy().squeeze()
            # 显示图像
            ax.imshow(img, cmap='gray')
            # 设置标题（真实标签和预测标签）
            color = 'green' if predicted[i] == labels[i] else 'red'
            ax.set_title(f"True: {labels[i].item()}\nPred: {predicted[i].item()}",
                         color=color, fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"epoch_{epoch}_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction visualization for epoch {epoch} at ~/epoch_{epoch}_predictions.png")

    net.train()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Now we are using device: {device}")

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    # 创建一个专门用于可视化的测试数据加载器（不打乱顺序）
    to_tensor = transforms.Compose([transforms.ToTensor()])
    visual_set = MNIST("", train=False, transform=to_tensor, download=True)
    visual_loader = DataLoader(visual_set, batch_size=512, shuffle=False)

    net = Net().to(device)
    epochs = []
    accuracies = []
    initial_acc = evaluate(test_data, net, device)
    print("initial accuracy:", initial_acc)

    epochs.append(-1)
    accuracies.append(initial_acc)

    # 保存初始状态的预测结果
    visualize_predictions(net, visual_loader, device, -1)

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

        # 保存当前epoch的预测可视化
        visualize_predictions(net, visual_loader, device, epoch)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xticks(epochs)
    plt.savefig("accuracy_curve.png", dpi=150, bbox_inches='tight')
    print("The accuracy curve is saved at ~/accuracy_curve.png")


if __name__ == "__main__":
    main()