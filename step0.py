import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

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
            outputs = net(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    net.train()
    return n_correct / n_total

def main():
    torch.manual_seed(114514)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(114514)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Now we are using : {device}.")

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)
    epochs = []
    accuracies = []
    initial_acc = evaluate(test_data, net, device)
    print(f"Initial accuracy: {initial_acc*100}%")

    epochs.append(-1)
    accuracies.append(initial_acc)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 8
    for epoch in range(num_epochs):
        for (x, y) in train_data:
            x = x.to(device)
            y = y.to(device)
            net.zero_grad()
            output = net(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(test_data, net, device)
        print(f"epoch {epoch}, accuracy : {acc*100}%")

        epochs.append(epoch)
        accuracies.append(acc)

    model_path = "mnist_net.pth"
    torch.save(net.state_dict(), model_path)
    print(f"Model weights saved to: {os.path.abspath(model_path)}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xticks(epochs)
    plt.savefig("accuracy_curve.png", dpi=150, bbox_inches='tight')
    pic_path = "accuracy_curve.png"
    print(f"The accuracy function is saved at {os.path.abspath(pic_path)}")

if __name__ == "__main__":
    main()