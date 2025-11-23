import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = get_data_loader(is_train=True)
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 8

    # 训练模型
    net.train()
    for epoch in range(num_epochs):
        for (x, y) in train_data:
            x = x.to(device)
            y = y.to(device)
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

    image_path = input("Enter image path: ")

    img = Image.open(image_path).convert('L').resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    net.eval()
    with torch.no_grad():
        output = net(input_tensor.view(-1, 28 * 28))
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"Predicted digit: {predicted_class}")


if __name__ == "__main__":
    main()