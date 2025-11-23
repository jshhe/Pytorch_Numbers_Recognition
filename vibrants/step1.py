import os
import torch
import random
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
    return DataLoader(data_set, batch_size=256, shuffle=True)


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


def show_random_prediction(net, test_dataset, device):
    idx = random.randint(0, len(test_dataset) - 1)
    image, true_label = test_dataset[idx]

    image_tensor = image.to(device)
    image_flat = image_tensor.view(1, -1)  # 展平为1x784

    net.eval()
    with torch.no_grad():
        output = net(image_flat)
        pred_label = torch.argmax(output, dim=1).item()
    net.train()

    img_np = image_tensor.cpu().squeeze().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np, cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {pred_label}", fontsize=14)
    plt.axis('off')

    pred_img_path = "prediction_example.png"
    plt.savefig(pred_img_path, dpi=150, bbox_inches='tight')
    plt.close()

    return os.path.abspath(pred_img_path), true_label, pred_label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Now we are using device: {device}")

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    to_tensor = transforms.Compose([transforms.ToTensor()])
    full_test_dataset = MNIST("", train=False, transform=to_tensor, download=True)

    net = Net().to(device)

    initial_acc = evaluate(test_data, net, device)
    print("Initial accuracy:", initial_acc)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 8
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
        print(f"Epoch {epoch}, Accuracy: {acc:.3f}")

    pred_img_path, true_label, pred_label = show_random_prediction(net, full_test_dataset, device)
    print(f"- Image saved at: {pred_img_path}")


if __name__ == "__main__":
    main()