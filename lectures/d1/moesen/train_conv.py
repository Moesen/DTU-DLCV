import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print(
        "The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator"
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
trainset = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.RandomRotation(10)]
    ),
)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
testset = datasets.MNIST(
    "./data", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(14 * 14 * 16, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convolutional(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


if __name__ == "__main__":
    model = Network()
    model.to(device)
    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Get the first minibatch
    data = next(iter(train_loader))[0].to(device)
    # Try running the model on a minibatch
    num_epochs = 5

    for epoch in tqdm(range(num_epochs), unit="epoch"):
        # For each epoch
        train_correct = 0
        for minibatch_no, (data, target) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = F.nll_loss(torch.log(output), target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
        # Comput the test accuracy
        test_correct = 0
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            predicted = output.argmax(1).cpu()
            test_correct += (target == predicted).sum().item()
        train_acc = train_correct / len(trainset)
        test_acc = test_correct / len(testset)
        print(
            "Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(
                test=100 * test_acc, train=100 * train_acc
            )
        )
