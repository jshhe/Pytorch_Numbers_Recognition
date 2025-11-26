
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import random


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
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


def visualize_random_predictions(net, test_dataset, device, epoch, num_samples=10):
    net.eval()
    with torch.no_grad():
        all_indices = list(range(len(test_dataset)))
        random_indices = random.sample(all_indices, num_samples)

        images = []
        true_labels = []
        for idx in random_indices:
            img, label = test_dataset[idx]
            images.append(img)
            true_labels.append(label)

        images = torch.stack(images).to(device)
        true_labels = torch.tensor(true_labels).to(device)

        outputs = net(images.view(-1, 28 * 28))
        _, predicted = torch.max(outputs, 1)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(num_samples):
            ax = axes[i]
            img = images[i].cpu().numpy().squeeze()
            # 显示图像
            ax.imshow(img, cmap='gray')
            color = 'green' if predicted[i] == true_labels[i] else 'red'
            ax.set_title(f"True: {true_labels[i].item()}\nPred: {predicted[i].item()}",
                         color=color, fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"epoch_{epoch}_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved random prediction visualization for epoch {epoch} at ~/epoch_{epoch}_predictions.png")

    net.train()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Now we are using device: {device}")

    # 获取数据加载器
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    to_tensor = transforms.Compose([transforms.ToTensor()])
    test_dataset = MNIST("", train=False, transform=to_tensor, download=True)

    net = Net().to(device)
    epochs = []
    accuracies = []
    initial_acc = evaluate(test_data, net, device)
    print("initial accuracy:", initial_acc)

    epochs.append(-1)
    accuracies.append(initial_acc)

    visualize_random_predictions(net, test_dataset, device, -1)

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

        visualize_random_predictions(net, test_dataset, device, epoch)

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